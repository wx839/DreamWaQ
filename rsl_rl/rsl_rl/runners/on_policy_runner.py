# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import warnings
import numpy as np
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    ActorCriticRecurrent,
    ActorCritic_DWAQ,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        
        # AdaBoot: Adaptive Bootstrapping from estimator network
        # Extract adaboot_enabled before passing alg_cfg to PPO (PPO doesn't need this parameter)
        self.adaboot_enabled = self.alg_cfg.get("adaboot_enabled", False)
        if self.adaboot_enabled:
            self.episode_rewards = []  # Store recent episode rewards for CV calculation
            self.cv_reward = float('inf')  # Coefficient of Variation of episode rewards
            self.p_boot = 0.7  # Bootstrapping probability (start high for exploration)
            # Track cumulative rewards per environment for current episodes
            self.current_episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
            print("[AdaBoot] Adaptive Bootstrapping enabled")
            print("[AdaBoot] Logic: High CV (unstable) → High p_boot (more exploration)")
            print("[AdaBoot]        Low CV (stable) → Low p_boot (more exploitation)")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            
            # AdaBoot: Compute bootstrapping probability at the start of each iteration
            boot_mask = None
            if self.adaboot_enabled:
                # Compute CV(R) and p_boot if enough episode rewards collected
                if len(self.episode_rewards) >= self.env.num_envs:
                    self.cv_reward = self._compute_reward_cv()
                    # AdaBoot logic: High CV (unstable) → High p_boot (more bootstrap/exploration)
                    #                Low CV (stable) → Low p_boot (more ground truth/exploitation)
                    # Use tanh to map CV to [0,1], with smooth transition around CV=1
                    self.p_boot = torch.tanh(torch.tensor(self.cv_reward)).item()
                    # Clamp p_boot to [min_p_boot, max_p_boot] for stability and train-inference consistency
                    min_p_boot = 0.1  # At least 10% envs use CENet (train-inference consistency)
                    max_p_boot = 0.9  # At most 90% envs use CENet (keep some ground truth for learning)
                    self.p_boot = max(min_p_boot, min(max_p_boot, self.p_boot))
                else:
                    # Not enough data yet, start with high exploration (more CENet)
                    self.p_boot = 0.7  # High initial p_boot for exploration
                
                # DEBUG: Print p_boot every 10 iterations
                if it % 10 == 0:
                    num_episodes = len(self.episode_rewards)
                    mean_r = np.mean(self.episode_rewards) if num_episodes > 0 else 0.0
                    print(f"[DEBUG AdaBoot] Iter {it}: p_boot={self.p_boot:.4f}, CV={self.cv_reward:.4f}, "
                          f"#episodes={num_episodes}, mean_reward={mean_r:.2f}")
                
                # Generate boot_mask: True=bootstrap (use CENet), False=use ground truth
                boot_mask = torch.rand(self.env.num_envs, device=self.device) < self.p_boot
            
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions (with AdaBoot mask if enabled)
                    actions = self.alg.act(obs, boot_mask=boot_mask)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    
                    # AdaBoot: Collect episode rewards for CV calculation
                    if self.adaboot_enabled:
                        # Accumulate rewards for current episodes
                        self.current_episode_rewards += rewards.squeeze(-1)
                        
                        # When an episode finishes (done=True), record its total reward
                        done_mask = dones.squeeze(-1) > 0
                        if done_mask.any():
                            # Get episode rewards for finished episodes
                            finished_episode_rewards = self.current_episode_rewards[done_mask]
                            num_finished = finished_episode_rewards.shape[0]
                            
                            # Add to episode rewards buffer
                            self.episode_rewards.extend(finished_episode_rewards.cpu().tolist())
                            
                            # Keep only recent rewards (sliding window)
                            max_window = self.env.num_envs * 10  # Keep 10x num_envs for better statistics
                            if len(self.episode_rewards) > max_window:
                                self.episode_rewards = self.episode_rewards[-max_window:]
                            
                            # Reset rewards for finished episodes
                            self.current_episode_rewards[done_mask] = 0
                            
                            # Debug print periodically
                            if it % 50 == 0 and _ == 0:
                                print(f"[DEBUG AdaBoot] Iter {it}: Collected {num_finished} episodes, "
                                      f"total_buffer={len(self.episode_rewards)}, mean_r={finished_episode_rewards.mean().item():.2f}")

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # Add AdaBoot metrics to loss_dict for logging
            if self.adaboot_enabled:
                loss_dict["adaboot/cv_reward"] = self.cv_reward
                loss_dict["adaboot/p_boot"] = self.p_boot
                loss_dict["adaboot/num_bootstrap"] = int(boot_mask.sum().item()) if boot_mask is not None else 0
                loss_dict["adaboot/num_groundtruth"] = int((~boot_mask).sum().item()) if boot_mask is not None else self.env.num_envs

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )

            # Save model
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _get_default_obs_sets(self) -> list[str]:
        """Get the the default observation sets required for the algorithm.

        .. note::
            See :func:`resolve_obs_groups` for more details on the handling of observation sets.
        """
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        return default_sets

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

    def _compute_reward_cv(self) -> float:
        """Compute coefficient of variation (CV) of episode rewards.
        
        CV = std(R) / mean(R)
        
        Returns:
            Coefficient of variation of recent episode rewards
        """
        if len(self.episode_rewards) == 0:
            return float('inf')
        
        import numpy as np
        rewards_array = np.array(self.episode_rewards)
        mean_r = np.mean(rewards_array)
        std_r = np.std(rewards_array)
        
        # Avoid division by zero
        if abs(mean_r) < 1e-8:
            return float('inf')
        
        cv = std_r / abs(mean_r)
        return cv

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))             # 获取类
        
        # Remove adaboot_enabled from alg_cfg before passing to PPO (handled at runner level)
        alg_cfg_for_ppo = {k: v for k, v in self.alg_cfg.items() if k != "adaboot_enabled"}
        
        alg: PPO = alg_class(                                        # 初始化PPO对象
            actor_critic, storage, device=self.device, **alg_cfg_for_ppo, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg

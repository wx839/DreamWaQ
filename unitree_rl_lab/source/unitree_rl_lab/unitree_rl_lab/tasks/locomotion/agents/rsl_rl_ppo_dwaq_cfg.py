# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DWAQActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for ActorCritic_DWAQ policy."""
    
    class_name = "ActorCritic_DWAQ"
    
    # DWAQ specific parameters
    history_length: int = 5
    """Number of observation steps to keep in history for encoder."""
    
    cenet_out_dim: int = 35
    """Output dimension of encoder (3 for velocity + 16 for mean + 16 for logvar)."""
    
    # Standard actor-critic parameters
    init_noise_std: float = 1.0
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    
    # Observation normalization
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False


@configclass
class DWAQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for PPO algorithm with DWAQ encoder-decoder losses."""
    
    # Standard PPO parameters
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    
    # DWAQ encoder-decoder loss coefficients
    dwaq_cfg: dict = {
        "vel_est_loss_coef": 1.0,    # Velocity estimation loss weight
        "recon_loss_coef": 1.0,      # Reconstruction loss weight
        "kl_latent_loss_coef": 0.01, # KL divergence for latent
    }
    
    # AdaBoot: AFalsedaptive Bootstrapping from estimator network (DreamWaQ paper)
    adaboot_enabled: bool = False
    """Enable adaptive bootstrapping: dynamically switch between CENet estimate and ground truth velocity.
    - Training early (high CV): Use more ground truth velocity to help learning
    - Training late (low CV): Use more CENet estimate to improve sim-to-real robustness
    Formula: p_boot = 1 - tanh(CV(R)), where CV = std(episode_rewards) / mean(episode_rewards)
    """


@configclass
class DWAQPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for training with DWAQ policy."""
    
    num_steps_per_env = 24
    max_iterations = 10000  # 修改为10000次迭代（可根据需要调整）
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    
    # Logging configuration
    logger: str = "tensorboard"  # 可选: "tensorboard", "wandb", "neptune"
    # wandb_project: str = "unitree_dwaq"  # 如果使用wandb，取消注释
    # neptune_project: str = "username/unitree_dwaq"  # 如果使用neptune，取消注释
    
    policy = DWAQActorCriticCfg()
    algorithm = DWAQAlgorithmCfg()

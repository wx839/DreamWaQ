from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict
from typing import Any

class ActorCritic_DWAQ(nn.Module):
    is_recurrent = False
    
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # DWAQ specific parameters
        history_length: int = 5,
        cenet_out_dim: int = 35,  # 3 (vel) + 32 (latent mu + sigma) by default
        # Standard parameters
        activation: str = "elu",
        init_noise_std: float = 1.0,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        actor_obs_normalization: bool | str = False,
        critic_obs_normalization: bool | str = False,
        **kwargs: Any,
    ):
        super().__init__()

        self.activation = get_activation(activation)
        self.history_length = history_length
        self.cenet_out_dim = cenet_out_dim
        self.obs_groups = obs_groups  # Save obs_groups for later use
        
        # Extract observation dimensions
        policy_obs = torch.cat([obs[group] for group in obs_groups["policy"]], dim=-1)    # O_t 45
        critic_obs = torch.cat([obs[group] for group in obs_groups["critic"]], dim=-1)    # critic input dim && decoder output dim
        
        num_policy_obs = policy_obs.shape[-1]    # 45
        num_critic_obs = critic_obs.shape[-1]
        
        # Store dimensions
        self.num_policy_obs = num_policy_obs
        self.num_critic_obs = num_critic_obs                   # decoder output dim
        self.num_actions = num_actions
        
        # Observation history for encoder (only policy observations)
        cenet_in_dim = num_policy_obs * (history_length + 1)          # 45 * 6 = 270
        
        # Actor input: o_t + v̂_t + z_t (paper table)
        actor_input_dim = num_policy_obs + 3 + 16                     # 45 + 3 + 16 = 64
        critic_input_dim = num_critic_obs     # S_t

        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim,128),
            self.activation,
            nn.Linear(128,64),
            self.activation,
        )
        # Encoder outputs: v̂_t(3) + μ_z(16) + log σ_z(16) = 35 dims
        self.encode_vel = nn.Linear(64, 3)                         # v̂_t: 3-dim, direct output
        self.encode_mean_latent = nn.Linear(64, 16)                # μ_z: 16-dim
        self.encode_logvar_latent = nn.Linear(64, 16)              # log σ_z: 16-dim

        # Decoder predicts the NEXT timestep's policy observation o_{t+1}
        # Input: v̂_t + z_t = 3 + 16 = 19 dims (paper table)
        # Output: next policy observation (45 dims for Go2)
        self.decoder = nn.Sequential(
            nn.Linear(19, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, num_policy_obs)  # Predict next policy observation ô_{t+1}
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # Observation history buffer [num_envs, history_length, obs_dim]
        self.obs_history = None
        self.num_envs = None
        
        # Store next policy observation for decoder reconstruction loss
        self.next_policy_obs = None
        # Store ground truth velocity for velocity estimation loss L_est
        self.ground_truth_vel = None
        # Store current policy observation for reconstructing encoder input
        self.current_policy_obs = None

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """Reset observation history for done environments."""
        if dones is not None and self.obs_history is not None:
            self.obs_history[dones] = 0.0
            if self.next_policy_obs is not None:
                self.next_policy_obs[dones] = 0.0
            if self.ground_truth_vel is not None:
                self.ground_truth_vel[dones] = 0.0
            if self.current_policy_obs is not None:
                self.current_policy_obs[dones] = 0.0

    def forward(self):
        raise NotImplementedError
    
    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def cenet_forward(self, obs_history):
        # Encoder forward
        distribution = self.encoder(obs_history)
        
        # Direct velocity output (no sampling needed)
        vel = self.encode_vel(distribution)  # v̂_t: 3-dim
        
        # Latent distribution parameters
        mean_latent = self.encode_mean_latent(distribution)      # μ_z: 16-dim
        logvar_latent = self.encode_logvar_latent(distribution)  # log σ_z: 16-dim
        
        # Sample latent z_t using reparameterization trick
        latent = self.reparameterise(mean_latent, logvar_latent)  # z_t: 16-dim
        
        # Concatenate for decoder and policy: v̂_t + z_t = 19-dim
        code = torch.cat((vel, latent), dim=-1)
        
        # Decoder predicts next observation
        decode = self.decoder(code)
        
        # Return: code(19), vel(3), decode(45), mean_latent(16), logvar_latent(16)
        return code, vel, decode, mean_latent, logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization (placeholder for compatibility)."""
        pass

    def _update_obs_history(self, obs: torch.Tensor) -> None:
        """Update observation history buffer."""
        if self.num_envs is None:
            self.num_envs = obs.shape[0]
            self.obs_history = torch.zeros(
                self.num_envs, self.history_length, self.num_policy_obs,
                device=obs.device, dtype=obs.dtype
            )
        
        # Shift history and add new observation (avoid inplace operation)
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)

    def act(self, obs: TensorDict, boot_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Sample actions from the policy.
        
        Args:
            obs: Observation dictionary
            boot_mask: [num_envs] bool tensor. True=use CENet estimate (bootstrap), False=use ground truth
        """
        # Extract policy observations
        policy_obs = torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)
        
        # Extract critic observations to get ground truth velocity
        critic_obs = torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)
        
        batch_size = policy_obs.shape[0]
        
        # Check if this is rollout (batch_size == num_envs) or update (mini_batch)
        if self.num_envs is None or batch_size == self.num_envs:
            # Rollout phase: update observation history and store targets
            # Store current policy obs as "next" target for the previous timestep's decoder
            self.next_policy_obs = policy_obs.clone()
            # Extract and store ground truth velocity from critic obs (first 3 dims: base_lin_vel)
            # Note: Assuming base_lin_vel is the first observation term in critic obs
            self.ground_truth_vel = critic_obs[:, :3].clone()  # First 3 dims are base_lin_vel
            # Store current observation for reconstructing encoder input during training
            self.current_policy_obs = policy_obs.clone()
            self._update_obs_history(policy_obs)
            # Concatenate history with current obs: [history_length + 1, obs_dim]
            obs_with_history = torch.cat([self.obs_history, policy_obs.unsqueeze(1)], dim=1)
            obs_history_flat = obs_with_history.reshape(obs_with_history.shape[0], -1)
        else:
            # Update phase (mini_batch): create temporary history by repeating current obs
            # This is a simplified approach - ideally we'd store the actual history
            obs_with_history = policy_obs.unsqueeze(1).repeat(1, self.history_length + 1, 1)
            obs_history_flat = obs_with_history.reshape(batch_size, -1)
        
        # Get encoded representation: code = v̂_t + z_t (19-dim), vel_pred is estimated velocity (3-dim)
        code, vel_pred, _, _, _ = self.cenet_forward(obs_history_flat)
        
        # AdaBoot: Dynamically choose velocity source based on boot_mask
        if boot_mask is not None and self.num_envs is not None and batch_size == self.num_envs:
            # boot_mask: True=use CENet estimate (bootstrap), False=use ground truth
            # Separate velocity and latent from code
            estimated_vel = vel_pred  # CENet estimated velocity [batch_size, 3]
            latent = code[:, 3:]      # Latent state z_t [batch_size, 16]
            
            # Choose velocity based on mask: bootstrap → CENet, not bootstrap → ground truth
            vel_for_policy = torch.where(
                boot_mask.unsqueeze(1),        # [batch_size, 1]
                estimated_vel,                 # CENet estimate (bootstrap)
                self.ground_truth_vel          # Ground truth (not bootstrap)
            )
            
            # Reconstruct actor input: o_t + v_t + z_t (45 + 3 + 16 = 64)
            actor_input = torch.cat([policy_obs, vel_for_policy, latent], dim=-1)
        else:
            # Default behavior: use full code (v̂_t + z_t)
            # Concatenate code with current observation: o_t + v̂_t + z_t = 45 + 19 = 64
            actor_input = torch.cat((policy_obs, code), dim=-1)
        
        # Update distribution and sample
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Deterministic action for inference/deployment.
        
        Note: This function ALWAYS uses CENet estimated velocity (no ground truth available).
        To ensure train-inference consistency, AdaBoot should use min_p_boot > 0 during training
        so that the Actor learns to handle CENet estimates, not just perfect ground truth.
        """
        # Extract policy observations
        policy_obs = torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)
        
        batch_size = policy_obs.shape[0]
        
        # Check if this is rollout or inference
        if self.num_envs is None or batch_size == self.num_envs:
            # Update observation history
            self._update_obs_history(policy_obs)
            # Concatenate history with current obs: [history_length + 1, obs_dim]
            obs_with_history = torch.cat([self.obs_history, policy_obs.unsqueeze(1)], dim=1)
            obs_history_flat = obs_with_history.reshape(obs_with_history.shape[0], -1)
        else:
            # Create temporary history by repeating current obs
            obs_with_history = policy_obs.unsqueeze(1).repeat(1, self.history_length + 1, 1)
            obs_history_flat = obs_with_history.reshape(batch_size, -1)
        
        # Get encoded representation: code = v̂_t + z_t (19-dim)
        code, _, _, _, _ = self.cenet_forward(obs_history_flat)
        
        # Concatenate code with current observation: o_t + v̂_t + z_t = 45 + 19 = 64
        actor_input = torch.cat((policy_obs, code), dim=-1)
        
        # Return mean action
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Evaluate value function using privileged observations."""
        # Extract critic observations (privileged)
        critic_obs = torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)
        value = self.critic(critic_obs)
        return value
    
    def get_encoder_outputs(self) -> tuple | None:
        """Get encoder outputs for computing CENet losses.
        
        Returns:
            Tuple of (code, vel_pred, decode, mean_latent, logvar_latent, target_obs, vel_gt)
            - code: v̂_t + z_t (19-dim)
            - vel_pred: predicted velocity ṽ_t (3-dim)
            - decode: predicted next observation ô_{t+1} (45-dim)
            - mean_latent: μ_z (16-dim)
            - logvar_latent: log σ_z (16-dim)
            - target_obs: actual next policy observation o_{t+1} (45-dim)
            - vel_gt: ground truth velocity v_t (3-dim)
        """
        if self.obs_history is None or self.current_policy_obs is None or self.next_policy_obs is None or self.ground_truth_vel is None:
            return None
        
        # Reconstruct encoder input: concatenate history with current obs
        # This creates a new tensor that supports autograd (not an inference tensor)
        obs_with_history = torch.cat([self.obs_history, self.current_policy_obs.unsqueeze(1)], dim=1)
        obs_history_flat = obs_with_history.reshape(obs_with_history.shape[0], -1)
        code, vel_pred, decode, mean_latent, logvar_latent = self.cenet_forward(obs_history_flat)
        
        # Target is the NEXT timestep's policy observation for reconstruction
        target_obs = self.next_policy_obs.clone()
        # Ground truth velocity for L_est
        vel_gt = self.ground_truth_vel.clone()
        
        return code, vel_pred, decode, mean_latent, logvar_latent, target_obs, vel_gt



        











def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
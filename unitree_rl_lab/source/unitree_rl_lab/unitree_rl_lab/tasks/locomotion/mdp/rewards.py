from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    ground_height: float = 0.04
) -> torch.Tensor:
    """Foot clearance reward following DreamWaQ paper (Table I) - velocity-weighted height penalty.
    
    Formula: r_clearance = -Σ_k (p_des - p_f,z,k)^2 * v_f,xy,k
    
    Key insight: Single target height (e.g., 0.08m) with velocity weighting:
        - Swing phase (high v_xy): Large penalty if not at target → forces lifting
        - Stance phase (low v_xy): Small penalty → allows ground contact
    
    This is the original DreamWaQ design - velocity naturally distinguishes phases.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取接触状态 (更严格的阈值 5.0N 防止误判)
    is_contact = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 5.0
    
    # 使用单一期望高度（DreamWaQ原始设计）
    # 速度加权会自动区分摆动/支撑相
    desired_height = torch.full_like(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2], 
        target_height
    )
    
    # 1. 计算高度误差平方: (p_des - p_actual)^2
    actual_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_z_error_sq = torch.square(desired_height - actual_height)
    
    # 2. 计算足部XY平面速度: v_xy
    foot_vel_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    
    # 3. 按论文公式: (p_des - p)^2 * v_xy (速度越大惩罚越重)
    raw_cost = foot_z_error_sq * foot_vel_xy
    
    # 4. 对所有腿求和
    reward = torch.sum(raw_cost, dim=1)
    
    # 调试打印
    print_interval = 10 if env.num_envs <= 100 else 50
    if hasattr(env, 'episode_length_buf') and env.episode_length_buf[0] % print_interval == 0:
        print(f"\n{'='*80}")
        print(f"步数: {env.episode_length_buf[0].item():4d} | 时间: {env.episode_length_buf[0].item() * env.step_dt:.2f}s")
        print(f"{'='*80}")
        print(f"{'足部':<12} | {'高度(m)':<9} | {'期望(m)':<9} | {'速度xy':<9} | {'接触':<6} | {'cost':<10}")
        print(f"{'-'*80}")
        
        for i, body_id in enumerate(asset_cfg.body_ids):
            body_name = asset.body_names[body_id]
            h_actual = actual_height[0, i].item()
            h_des = desired_height[0, i].item()
            v_xy = foot_vel_xy[0, i].item()
            contact = "接触" if is_contact[0, i].item() else "摆动"
            cost_val = raw_cost[0, i].item()
            
            print(f"{body_name:<12} | {h_actual:>7.4f}m | {h_des:>7.4f}m | {v_xy:>7.4f} | {contact:<6} | {cost_val:>8.6f}")
        
        print(f"{'='*80}\n")
        # print(f"######## Learning iteration {env.common_step_counter // env.max_episode_length}/{env.max_iterations} ########\n")
    
    return reward


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""

def monitor_foot_height(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"), 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    print_interval: int = 50
) -> torch.Tensor:
    """Monitor and print foot height over time with contact information.
    
    This is a monitoring reward that doesn't affect training but provides detailed logging.
    Returns zero reward.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 每隔一定步数打印详细信息
    if hasattr(env, 'episode_length_buf') and env.episode_length_buf[0] % print_interval == 0:
        foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # Z坐标（离地高度）
        foot_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # Z方向速度
        is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0  # 接触状态
        
        current_step = env.episode_length_buf[0].item()
        current_time = current_step * env.step_dt
        
        print(f"\n{'='*70}")
        print(f"步数: {current_step:4d} | 时间: {current_time:.3f}s")
        print(f"{'='*70}")
        print(f"{'足部名称':<15} | {'高度(m)':<10} | {'速度(m/s)':<12} | {'接触状态'}")
        print(f"{'-'*70}")
        
        # 打印第一个环境的所有足部信息
        for i, body_id in enumerate(asset_cfg.body_ids):
            body_name = asset.body_names[body_id]
            height = foot_heights[0, i].item()
            velocity = foot_velocities[0, i].item()
            contact = "接触地面" if is_contact[0, i].item() else "空中"
            
            print(f"{body_name:<15} | {height:>8.4f}m | {velocity:>+10.4f} | {contact}")
        
        print(f"{'='*70}\n")
    
    # 返回零奖励（仅用于监控）
    return torch.zeros(env.num_envs, device=env.device)


def swing_foot_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.08,
    velocity_threshold: float = 0.3
) -> torch.Tensor:
    """Penalize swing phase feet for deviating from target height.
    
    Key change: Instead of rewarding "higher is better", penalize deviation from target.
    This prevents over-lifting (0.2m+) and encourages staying near 0.06m.
    
    Only applies to feet that are:
    1. Not in contact with ground (force < 5N)
    2. Moving horizontally (v_xy > threshold)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 检测摆动相：无接触 + 高速度
    is_contact = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 5.0
    foot_vel_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    is_swinging = (~is_contact) & (foot_vel_xy > velocity_threshold)
    
    # 计算高度偏差平方（惩罚过高或过低）
    actual_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_error_sq = torch.square(actual_height - target_height)
    
    # 只对摆动相的腿计算惩罚
    penalty_per_foot = torch.where(is_swinging, height_error_sq, torch.zeros_like(height_error_sq))
    
    return torch.sum(penalty_per_foot, dim=1)


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
    
#自定义添加
def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second-order finite difference (jerk) of actions using L2 squared kernel.
    
    This computes the squared norm of the discrete second derivative:
    (a_t - 2*a_{t-1} + a_{t-2})^2
    
    This encourages smooth action trajectories by penalizing rapid changes in acceleration.
    """
    # Get current action and two previous actions from action manager
    action_t = env.action_manager.action
    action_t1 = env.action_manager.prev_action  
    
    # Check if we have access to t-2 action (stored in action manager buffer)
    # If not available (e.g., first few timesteps), use prev_action as fallback
    if hasattr(env.action_manager, "prev_prev_action"):
        action_t2 = env.action_manager.prev_prev_action
    else:
        # Fallback: assume prev_prev_action = prev_action (zero second derivative initially)
        action_t2 = action_t1
    
    # Compute second-order finite difference (discrete second derivative / jerk)
    second_order_diff = action_t - 2.0 * action_t1 + action_t2
    
    # Return L2 squared norm
    return torch.sum(torch.square(second_order_diff), dim=1)


def power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the variance in power distribution across joints.
    
    This computes the squared variance of mechanical power (torque × velocity) across all joints:
    var(τ · θ̇)²
    
    This encourages even power distribution, preventing some joints from being overloaded
    while others remain idle.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get joint velocities and applied torques
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    
    # Compute mechanical power for each joint: P = τ · ω
    power = joint_torque * joint_vel
    
    # Compute variance of power across joints (dim=1 is the joint dimension)
    power_variance = torch.var(power, dim=1)
    
    # Return squared variance
    return torch.square(power_variance)
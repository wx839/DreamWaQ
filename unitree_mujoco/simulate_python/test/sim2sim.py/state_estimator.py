import numpy as np
from .config import DeployConfig

class StateEstimator:
    def __init__(self):
        # 记录上一次的动作，初始化为0
        self.last_action = np.zeros(12, dtype=np.float32)
        
        # 历史观测缓存队列：存储最近 HISTORY_STEPS 帧的 45维观测
        self.obs_buffer = [np.zeros(DeployConfig.OBS_DIM, dtype=np.float32) 
                           for _ in range(DeployConfig.HISTORY_STEPS)]

    def get_projected_gravity(self, quaternion):
        """
        将四元数转换为重力向量在机器人坐标系下的投影
        quaternion: [w, x, y, z]
        """
        w, x, y, z = quaternion
        # 计算旋转矩阵的第三列（即世界坐标系Z轴在身体坐标系下的表示）
        # 这是典型的重力投影计算公式
        res = np.zeros(3, dtype=np.float32)
        res[0] = 2.0 * (x * z - w * y)
        res[1] = 2.0 * (y * z + w * x)
        res[2] = 1.0 - 2.0 * (x * x + y * y)
        return res

    def update_and_get_obs(self, low_state, high_state):
        """
        low_state: 从 rt/lowstate 获取的电机和IMU原始数据
        high_state: 从 rt/sportmodestate 获取的机器人本体位姿信息
        """
        # 1. 提取各组成部分
        # 基座角速度 (IMU 陀螺仪数据)
        base_ang_vel = np.array(low_state.imu_state.gyroscope, dtype=np.float32)
        
        # 投影重力 (使用 IMU 四元数)
        projected_gravity = self.get_projected_gravity(low_state.imu_state.quaternion)
        
        # 速度指令 (来自 config)
        velocity_commands = DeployConfig.CMD_VEL
        
        # 关节相对位置 (当前角度 - 默认角度)
        current_q = np.array([motor.q for motor in low_state.motor_state[:12]], dtype=np.float32)
        joint_pos_rel = current_q - DeployConfig.DEFAULT_JOINT_ANGLES
        
        # 关节速度
        joint_vel = np.array([motor.dq for motor in low_state.motor_state[:12]], dtype=np.float32)
        
        # 2. 拼接当前帧观测 (45维)
        current_obs = np.concatenate([
            base_ang_vel,       # 3
            projected_gravity,  # 3
            velocity_commands,  # 3
            joint_pos_rel,      # 12
            joint_vel,          # 12
            self.last_action    # 12
        ]).astype(np.float32)

        # 3. 更新缓存队列 (先进先出)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(current_obs)

        # 4. 准备返回结果
        # current_obs_batch: 用于 Policy (1, 45)
        # history_obs_batch: 用于 Encoder (1, 135) -> 假设 HISTORY_STEPS=3
        current_obs_batch = current_obs.reshape(1, -1)
        history_obs_batch = np.concatenate(self.obs_buffer).reshape(1, -1)

        return current_obs_batch, history_obs_batch

    def set_last_action(self, action):
        """更新记录上一次动作，供下一帧使用"""
        self.last_action = action

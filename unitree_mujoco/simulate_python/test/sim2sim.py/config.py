import numpy as np

class DeployConfig:
    # 模型路径 (建议使用绝对路径或相对于 main_deploy.py 的路径)
    # ENCODER_PATH = "./encoder.onnx"
    # POLICY_PATH = "./policy.onnx"
    
    ENCODER_PATH = "/home/wx/DreamWaQ/logs/rsl_rl/unitree_go2_velocity_dwaq/2026-01-12_18-49-46/exported/encoder.onnx"
    POLICY_PATH = "/home/wx/DreamWaQ/logs/rsl_rl/unitree_go2_velocity_dwaq/2026-01-12_18-49-46/exported/policy.onnx"
    
    # 动作参数
    ACTION_SCALE = 0.25
    
    # Go2 默认姿态 (根据图二填写，注意顺序通常为: FR, FL, RR, RL)
    # 每个腿的顺序是: Hip, Thigh, Calf
    DEFAULT_JOINT_ANGLES = np.array([
        -0.1, 0.8, -1.5,  # FR
        0.1, 0.8, -1.5,  # FL
        -0.1, 1.0, -1.5,  # RR
        0.1, 1.0, -1.5   # RL
    ], dtype=np.float32)
    
    # PD 控制增益 (需要与训练时一致)
    KP = 20.0
    KD = 0.5
    
    # 观测参数 (根据图三 DreamWaQ 典型设置)
    # 假设输入包含: 指令(3), 关节pos(12), 关节vel(12), 上一次动作(12), 重力向量(3)
    OBS_DIM = 45 
    HISTORY_STEPS = 5 # 历史帧长度
    
    # 指令速度 (vx, vy, yaw_rate)
    CMD_VEL = np.array([0.5, 0.0, 0.0], dtype=np.float32)
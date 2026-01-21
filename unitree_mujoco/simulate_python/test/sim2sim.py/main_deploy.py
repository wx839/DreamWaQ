import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_default

from .config import DeployConfig
from .policy_runner import PolicyRunner
from .state_estimator import StateEstimator

class DreamWaQDeployer:
    def __init__(self):
        # 初始化通信
        ChannelFactoryInitialize(1, "lo")
        
        # 初始化模块
        self.policy = PolicyRunner()
        self.estimator = StateEstimator()
        
        # 定义 DDS 状态缓存
        self.low_state = None
        self.high_state = None
        
        # 设置订阅者
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 10)
        
        self.high_state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.high_state_sub.Init(self.high_state_handler, 10)
        
        # 设置发布者
        self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_pub.Init()

    def low_state_handler(self, msg):
        self.low_state = msg

    def high_state_handler(self, msg):
        self.high_state = msg

    def run(self):
        print("DreamWaQ Sim2Sim 部署脚本已启动...")
        
        while True:
            # 确保已经收到数据
            if self.low_state is None:
                time.sleep(0.01)
                continue

            # 1. 状态估计与预处理
            curr_obs, hist_obs = self.estimator.update_and_get_obs(self.low_state, self.high_state)
            
            # 2. 模型推理
            action = self.policy.get_action(curr_obs, hist_obs)
            
            # 3. 更新 Last Action 供下一帧使用
            self.estimator.set_last_action(action)
            
            # 4. 构造指令并映射到关节空间
            low_cmd = unitree_go_msg_dds__LowCmd_default()
            target_q = DeployConfig.DEFAULT_JOINT_ANGLES + action * DeployConfig.ACTION_SCALE
            
            for i in range(12):
                low_cmd.motor_cmd[i].mode = 0x01
                low_cmd.motor_cmd[i].q = target_q[i]
                low_cmd.motor_cmd[i].kp = DeployConfig.KP
                low_cmd.motor_cmd[i].kd = DeployConfig.KD
                low_cmd.motor_cmd[i].dq = 0.0
                low_cmd.motor_cmd[i].tau = 0.0
            
            # 5. 发送指令
            self.low_cmd_pub.Write(low_cmd)
            
            # 控制频率建议在 50Hz - 100Hz，取决于你训练时的 dt
            time.sleep(0.02)

if __name__ == "__main__":
    deployer = DreamWaQDeployer()
    deployer.run()
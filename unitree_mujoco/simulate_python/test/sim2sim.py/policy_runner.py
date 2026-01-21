import onnxruntime as ort
import numpy as np
from .config import DeployConfig

class PolicyRunner:
    def __init__(self):
        # 初始化推理会话
        self.encoder_session = ort.InferenceSession(DeployConfig.ENCODER_PATH)
        self.policy_session = ort.InferenceSession(DeployConfig.POLICY_PATH)
        
        # 获取输入名称
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.policy_input_name = self.policy_session.get_inputs()[0].name
        self.policy_latent_name = self.policy_session.get_inputs()[1].name

    def get_action(self, current_obs, history_obs):
        """
        current_obs: 当前帧观测 (1, OBS_DIM)
        history_obs: 拼接的历史观测 (1, HISTORY_STEPS * OBS_DIM)
        """
        # 1. 运行 Encoder 获得 Latent (隐含状态)
        latent = self.encoder_session.run(None, {self.encoder_input_name: history_obs})[0]
        
        # 2. 运行 Policy 获得 Action
        # 注意：DreamWaQ 的 Policy 通常同时输入当前观测和 Latent
        action = self.policy_session.run(None, {
            self.policy_input_name: current_obs,
            self.policy_latent_name: latent
        })[0]
        
        return action.flatten()
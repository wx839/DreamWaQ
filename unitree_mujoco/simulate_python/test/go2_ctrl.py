import os
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_   #高层
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ 
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
import pygame
import sys



# print("PID:", os.getpid(), "__name__ =", __name__)
# DDS 全局变量
latest_low_state = None
latest_high_state = None

#从机体unitree_sdk拿观测,仿真和实机通用
class PolicyObservationFromUnitreeSDK:
    def __init__(
        self,
        num_joints: int,
        action_dim: int,
        default_joint_pos: np.ndarray,
        cmd_scale=(1.0, 1.0, 1.0),
    ):
        """
        Args:
            num_joints: number of actuated joints
            action_dim: dimension of policy action
            default_joint_pos: nominal joint pose used in training (N,)
            cmd_scale: (vx_scale, vy_scale, wz_scale)
        """
        self.num_joints = num_joints
        self.action_dim = action_dim
        self.default_joint_pos = default_joint_pos.copy()

        self.cmd_scale = np.array(cmd_scale)

        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self.cmd_vel = np.zeros(3, dtype=np.float32)

    #关节映射，eg:仿真中0号关节对应实机3号关节，1号关节对应实机0号关节
    joint_ids_map = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])

    def set_velocity_command(self, vx, vy, wz):
        self.cmd_vel[0] = vx * self.cmd_scale[0]
        self.cmd_vel[1] = vy * self.cmd_scale[1]
        self.cmd_vel[2] = wz * self.cmd_scale[2]
    def _base_ang_vel(self, low_state):
        # imu gyroscope is already body-frame angular velocity
        return np.array(low_state.imu_state.gyroscope, dtype=np.float32)

    #重力投影-------------------这个有点怪（issaclab play的时候打印出的这三个值 模长大于一）后续要再验证一下，肯嗯是计算方法有问题
    def _projected_gravity(self, low_state):
        # quaternion from Unitree: [w, x, y, z]
        q = low_state.imu_state.quaternion
        quat = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

        rot = R.from_quat(quat)
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # project gravity into body frame
        g_body = rot.inv().apply(g_world)
        return g_body.astype(np.float32)

    #速度指令 
    def _velocity_commands(self):
        return self.cmd_vel.astype(np.float32)

    #关节相对位置 
    def _joint_pos_rel(self, low_state):
        # 1. SDK / 电机顺序
        q_sdk = np.zeros(self.num_joints, dtype=np.float32)
        for i in range(self.num_joints):
            q_sdk[i] = low_state.motor_state[i].q
        # 2. 先算相对位置（仍是 SDK 顺序）
        q_rel_sdk = q_sdk - self.default_joint_pos  # default_joint_pos 也是 SDK 顺序
        # 3. 再重排成训练顺序
        q_rel_train = q_rel_sdk[self.joint_ids_map]

        return q_rel_train

    #关节速度  重排逻辑和上面一样
    def _joint_vel_rel(self, low_state):
        dq_sdk = np.zeros(self.num_joints, dtype=np.float32)
        for i in range(self.num_joints):
            dq_sdk[i] = low_state.motor_state[i].dq
        dq_train = dq_sdk[self.joint_ids_map]
        return dq_train

    def _last_action(self):
        return self.last_action.astype(np.float32)
    
    def get_observation(self, low_state, high_state=None):
        """
        Build policy observation vector.
        high_state is kept for extensibility but not used here.
        """

        obs = np.concatenate(
            [
                self._base_ang_vel(low_state),
                self._projected_gravity(low_state),
                self._velocity_commands(),
                self._joint_pos_rel(low_state),
                self._joint_vel_rel(low_state),
                self._last_action(),
            ],
            axis=0,
        )

        return obs

    def update_last_action(self, action):
        self.last_action[:] = action


def HighStateHandler(msg: SportModeState_):
    global latest_high_state
    latest_high_state = msg


def LowStateHandler(msg: LowState_):
    global latest_low_state
    latest_low_state = msg

 
# 状态机定义
class RobotState:
    DISABLED = 0      # 关节失能
    STANDING = 1      # 站立模式
    RL_CONTROL = 2    # RL 控制模式


class GamepadController:
    """手柄控制器，用于读取手柄输入并控制状态机"""
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[INFO] Gamepad connected: {self.joystick.get_name()}")
        else:
            print("[WARNING] No gamepad found!")
            self.joystick = None
        
        # 按钮映射 (根据实际手柄可能需要调整)
        self.BUTTON_A = 0
        self.BUTTON_B = 1
        self.BUTTON_START = 7
        self.BUTTON_L1 = 4  # L1 肩键按钮索引
        
        # 轴映射 (用于速度控制)
        self.AXIS_LX = 0  # 左摇杆 X 轴
        self.AXIS_LY = 1  # 左摇杆 Y 轴
        self.AXIS_RX = 3  # 右摇杆 X 轴 (改为轴3，轴2可能是扳机)
        
    def get_state(self):
        """获取手柄当前状态"""
        if self.joystick is None:
            return None, None, None
        
        pygame.event.pump()  # 更新事件
        
        # 读取按钮状态
        l1_pressed = self.joystick.get_button(self.BUTTON_L1) if self.joystick.get_numbuttons() > self.BUTTON_L1 else False
        a_pressed = self.joystick.get_button(self.BUTTON_A) if self.joystick.get_numbuttons() > self.BUTTON_A else False
        b_pressed = self.joystick.get_button(self.BUTTON_B) if self.joystick.get_numbuttons() > self.BUTTON_B else False
        start_pressed = self.joystick.get_button(self.BUTTON_START) if self.joystick.get_numbuttons() > self.BUTTON_START else False
        
        # 读取摇杆值 (用于速度控制)
        lx = self.joystick.get_axis(self.AXIS_LX) if self.joystick.get_numaxes() > self.AXIS_LX else 0.0
        ly = self.joystick.get_axis(self.AXIS_LY) if self.joystick.get_numaxes() > self.AXIS_LY else 0.0
        rx = self.joystick.get_axis(self.AXIS_RX) if self.joystick.get_numaxes() > self.AXIS_RX else 0.0
        
        buttons = {
            'l1': l1_pressed,
            'a': a_pressed,
            'b': b_pressed,
            'start': start_pressed
        }
        
        axes = {
            'lx': lx,
            'ly': -ly,  # Y轴反转（通常向前是负值）
            'rx': rx
        }
        
        return buttons, axes, None
    
    def cleanup(self):
        """清理资源"""
        if self.joystick:
            self.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    
    #scale参数,这里是拼接后，对最后的45维向量统一相乘
    base_ang_vel_sacle=[0.25, 0.25, 0.25]
    projected_gravity_sacle=[1.0, 1.0, 1.0]
    velocity_commands_sacle=[1.0, 1.0, 1.0]
    joint_pos_rel_sacle=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    joint_vel_rel_sacle=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    last_action_sacle=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    obs_scale = np.array(
        base_ang_vel_sacle
        + projected_gravity_sacle
        + velocity_commands_sacle
        + joint_pos_rel_sacle
        + joint_vel_rel_sacle
        + last_action_sacle,
        dtype=np.float32,
    )
    
    
    # DDS 初始化，这里的(1, "lo")要和unitree_mujoco.py中设置的一样
    ChannelFactoryInitialize(1, "lo")
    hight_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)

    hight_state_suber.Init(HighStateHandler, 10)
    low_state_suber.Init(LowStateHandler, 10)

    low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_puber.Init()
    crc = CRC()

    #ONNX模型加载，一个encoder一个policy
    onnx_path_01 = "/home/wx/DreamWaQ/logs/rsl_rl/unitree_go2_velocity_dwaq/2026-01-26_18-40-45/exported/policy.onnx"   # ← 改成你的 policy 路径
    onnx_path_02 = "/home/wx/DreamWaQ/logs/rsl_rl/unitree_go2_velocity_dwaq/2026-01-26_18-40-45/exported/encoder.onnx"
    sess_policy = ort.InferenceSession(
        onnx_path_01,
        providers=["CPUExecutionProvider"],
    )
    sess_encoder = ort.InferenceSession(
        onnx_path_02,
        providers=["CPUExecutionProvider"],
    )
    obs_history_name = sess_encoder.get_inputs()[0].name
    latent_name = sess_encoder.get_outputs()[0].name

    policy_input_name = sess_policy.get_inputs()[0].name
    act_name = sess_policy.get_outputs()[0].name

    NUM_JOINTS = 12
    ACTION_DIM = 12
    #发送顺序，也就是机体的顺序
    default_q = np.array([
        # FR_hip, FR_thigh, FR_calf,
        # FL_hip, FL_thigh, FL_calf,
        # RR_hip, RR_thigh, RR_calf,
        # RL_hip, RL_thigh, RL_calf,
        -0.1, 0.8, -1.5,
        0.1, 0.8, -1.5,
        -0.1, 1.0, -1.5,
        0.1,1.0, -1.5,
    ], dtype=np.float32)
    #  [-0.2343531   0.19057316 -0.62015957  
    #   0.7084377  -1.4762688  -0.23799106
    #   0.0443995  -1.099353    2.4444668   
    #   2.2744527   1.7502182   1.6025759 ]
    
    joint_ids_map=[3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]

    obs_builder = PolicyObservationFromUnitreeSDK(
        num_joints=NUM_JOINTS,
        action_dim=ACTION_DIM,
        default_joint_pos=default_q,
        cmd_scale=(1.0, 1.0, 1.0),   # vx, vy, wz
    )
   
    HISTORY_LEN = 6
    OBS_DIM = 45  

    obs_history = np.zeros(
        (HISTORY_LEN, OBS_DIM),
        dtype=np.float32
    )
    
    # LowCmd init 
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0

    for i in range(20):  #因为宇树的LowCmd里有20个电机命令槽位，虽然只用了12个但是初始化的时候全初始化一下
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q= 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0
    
    # 初始化手柄控制器
    gamepad = GamepadController()
    
    # 状态机初始化
    current_state = RobotState.DISABLED
    previous_buttons = {'l1': False, 'a': False, 'b': False, 'start': False}
    
    print("[INFO] State machine initialized. Current state: DISABLED")
    print("[INFO] Controls:")
    print("  L1 + B -> Disable motors")
    print("  L1 + A -> Standing mode")
    print("  START  -> RL control mode")
    print("  Left stick -> control velocity (in RL mode)")
    print("  Right stick X -> control yaw rate (in RL mode)")
    
    print("[INFO] RL control loop started.")

    #---------------------------------进入站立状态------------------------------------------------
    #因为训练的时候，每个循环开始的时候，狗是有默认站立姿态的，所以这里也让它先站起来，不然趴着的话，它会直接弹飞，后续训练的时候，可以训练一下跌到回正或者站立
    # STAND_STEPS = 500      # 500 * 0.002 = 1 秒
    # CONTROL_DT = 0.002

    # print("[INFO] Standing up to default pose...")

    # for step in range(STAND_STEPS):
    #     for i in range(NUM_JOINTS):
    #         cmd.motor_cmd[i].mode = 0x01
    #         cmd.motor_cmd[i].q = default_q[i]     # 默认站立
    #         cmd.motor_cmd[i].kp = 25.0           
    #         cmd.motor_cmd[i].kd = 1.5
    #         cmd.motor_cmd[i].tau = 0.0

    #     cmd.crc = crc.Crc(cmd)
    #     low_cmd_puber.Write(cmd)
    #     time.sleep(CONTROL_DT)
   
    #进入主控制循环（带状态机）-------------------------------------------------------------------------------------------
    # CONTROL_DT = 0.002
    CONTROL_DT = 0.005
    STAND_KP = 25.0
    STAND_KD = 0.5
    RL_KP = 25.0
    RL_KD = 0.5
    
    try:
        while True:
            # 读取手柄输入
            buttons, axes, _ = gamepad.get_state()
            
            # 调试：打印所有按键和摇杆状态
            if buttons:
                # 检测任意按键按下时打印状态
                any_button_pressed = any([buttons['l1'], buttons['a'], buttons['b'], buttons['start']])
                if any_button_pressed:
                    print(f"[DEBUG] 按键状态: L1={buttons['l1']}, A={buttons['a']}, B={buttons['b']}, START={buttons['start']}")
                
                # 检测按键状态变化（边缘触发）
                # L1 + B -> 失能模式
                if buttons['l1'] and buttons['b'] and not previous_buttons['b']:
                    current_state = RobotState.DISABLED
                    print("[STATE] Switched to DISABLED mode")
                
                # L1 + A -> 站立模式
                elif buttons['l1'] and buttons['a'] and not previous_buttons['a']:
                    current_state = RobotState.STANDING
                    print("[STATE] Switched to STANDING mode")
                
                # START -> RL控制模式
                elif buttons['start'] and not previous_buttons['start']:
                    current_state = RobotState.RL_CONTROL
                    print("[STATE] Switched to RL_CONTROL mode")
                
                previous_buttons = buttons.copy()
            
            # 根据当前状态执行相应控制
            if current_state == RobotState.DISABLED:
                # 失能模式：关节零力矩
                for i in range(NUM_JOINTS):
                    cmd.motor_cmd[i].mode = 0x01
                    cmd.motor_cmd[i].q = 0.0
                    cmd.motor_cmd[i].kp = 0.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = 0.0
                    cmd.motor_cmd[i].tau = 0.0
                
                cmd.crc = crc.Crc(cmd)
                low_cmd_puber.Write(cmd)
                time.sleep(CONTROL_DT)
                continue
            
            elif current_state == RobotState.STANDING:
                # 设置速度指令
                vx_cmd = 0.0
                vy_cmd = 0.0
                wz_cmd = 0.0
                obs_builder.set_velocity_command(
                    vx=vx_cmd,
                    vy=vy_cmd,
                    wz=wz_cmd,
                )
                
                
                # 调试：打印速度指令
                # print(f"\r[DEBUG] 速度指令: vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, wz={wz_cmd:.3f}", end="")
                
                # 构建 policy observation
                test_obs = obs_builder.get_observation(
                    latest_low_state,
                    latest_high_state,
                )
                print(test_obs)



                # 站立模式：保持默认姿态
                for i in range(NUM_JOINTS):
                    cmd.motor_cmd[i].mode = 0x01
                    cmd.motor_cmd[i].q = default_q[i]
                    cmd.motor_cmd[i].kp = STAND_KP
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = STAND_KD
                    cmd.motor_cmd[i].tau = 0.0
                
                cmd.crc = crc.Crc(cmd)
                low_cmd_puber.Write(cmd)
                time.sleep(CONTROL_DT)
                continue
            
            elif current_state == RobotState.RL_CONTROL:
                # RL控制模式：使用策略网络
                if latest_low_state is None:
                    print("[DEBUG] waiting for low_state ...", end="\r")
                    time.sleep(0.001)
                    continue

                # 从手柄摇杆读取速度指令（默认为0）
                vx_cmd = 0.0
                vy_cmd = 0.0
                wz_cmd = 0.0
                
                if axes:
                    # 调试：打印原始摇杆值和应用死区后的值
                    print(f"\r[RAW] ly={axes['ly']:.3f}, lx={axes['lx']:.3f}, rx={axes['rx']:.3f}", end=" | ")
                    
                    # 应用死区（降低死区以便更容易触发）
                    deadzone = 0.1  # 降低死区
                    vx_raw = axes['ly']
                    vy_raw = -axes['lx']  # 反转左摇杆X轴方向
                    wz_raw = -axes['rx']  # 反转右摇杆X轴方向
                    
                    vx_cmd = vx_raw if abs(vx_raw) > deadzone else 0.0
                    vy_cmd = vy_raw if abs(vy_raw) > deadzone else 0.0
                    wz_cmd = wz_raw if abs(wz_raw) > deadzone else 0.0
                    
                    print(f"[AFTER] vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, wz={wz_cmd:.3f}", end=" | ")
                    
                    # 速度缩放
                    vx_cmd *= 1.0 # 最大前进速度 0.5 m/s
                    vy_cmd *= 0.4  # 最大侧向速度 0.3 m/s
                    wz_cmd *= 1.0  # 最大转向速度 1.0 rad/s

                # 设置速度指令
                obs_builder.set_velocity_command(
                    vx=vx_cmd,
                    vy=vy_cmd,
                    wz=wz_cmd,
                )
                
                # 调试：打印速度指令
                print(f"\r[DEBUG] 速度指令: vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, wz={wz_cmd:.3f}", end="")
                
                # 构建 policy observation
                obs = obs_builder.get_observation(
                    latest_low_state,
                    latest_high_state,
                )
                
                # 调试：打印观测中的速度指令部分（索引6-9是速度指令）
                vel_cmd_in_obs = obs[6:9]
                print(f" | obs中速度指令: [{vel_cmd_in_obs[0]:.3f}, {vel_cmd_in_obs[1]:.3f}, {vel_cmd_in_obs[2]:.3f}]    ", end="")

                obs = obs.astype(np.float32)[None, :]  # [1, obs_dim] ONNX / PyTorch 默认 float32 numpy 默认 float64 增加batch维度为[1,45]
                obs = obs * obs_scale
                obs = obs.astype(np.float32)
                obs = np.clip(obs, -100.0, 100.0)

                # 更新obs_history
                obs_now = obs[0]
                obs_history[:-1] = obs_history[1:]
                obs_history[-1] = obs_now

                #手动输入网络输入调试

                # ONNX 推理
                obs_history_flat = obs_history.reshape(1, -1).astype(np.float32)

                latens = sess_encoder.run(
                    [latent_name],
                    {obs_history_name: obs_history_flat},
                )[0] 
                
                actor_input = np.concatenate(
                    [obs, latens],
                    axis=-1
                )  # 拼接当前 obs 和 encoder 输出的 latent
                # [1, latent_dim]   
                # 拿到指令
                action = sess_policy.run(
                    [act_name],
                    {policy_input_name: actor_input},
                )[0][0]  # [action_dim]

                # 记录 last_action
                obs_builder.update_last_action(action)

                # 计算实际目标位置 ，要和训练的时候一样
                action_offset = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
                action_scale =  [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
                #获取电机对应的policy输出
                q_target_policy = action_offset + action_scale * action
                 
                # 与训练时一致的 action clip
                q_target_policy = np.clip(q_target_policy, -100.0, 100.0)

                # print("Action:", q_target_policy)
        
                # 发送，这里关节顺序一定要对应上 
                # 这里控制是位置控制，给定目标位置，剩下的由底层PD控制器来完成
                for i in range(NUM_JOINTS):
                    sdk_i = joint_ids_map[i]

                    cmd.motor_cmd[sdk_i].mode = 0x01
                    cmd.motor_cmd[sdk_i].q = float(q_target_policy[i])
                    cmd.motor_cmd[sdk_i].kp = RL_KP
                    cmd.motor_cmd[sdk_i].dq = 0.0
                    cmd.motor_cmd[sdk_i].kd = RL_KD
                    cmd.motor_cmd[sdk_i].tau = 0.0

                cmd.crc = crc.Crc(cmd)
                low_cmd_puber.Write(cmd)

            time.sleep(CONTROL_DT)
    
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        gamepad.cleanup()
        sys.exit(0)
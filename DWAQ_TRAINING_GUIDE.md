# DWAQ (DreamWaQ) 训练指南

## 概述

已成功集成 ActorCritic_DWAQ 模型到 unitree_rl_lab 框架中，用于 Go2 机器人的运动控制训练。

## 架构说明

### ActorCritic_DWAQ 工作原理

你对架构的理解基本正确，具体如下：

1. **Critic网络**: 使用特权观测（privileged observations）评估状态价值
   - 输入：包含base_lin_vel, joint_effort等特权信息的观测
   - 输出：状态价值估计

2. **Encoder网络**: 从观测历史中学习
   - 输入：5步观测历史（45维观测 × 5 = 225维）
   - 输出：32维编码（3维速度 + 29维环境隐式表达）
   - 使用VAE结构，包含均值和方差分支

3. **Actor网络**: 生成动作
   - 输入：32维编码 + 45维当前观测 = 77维
   - 输出：12维动作（Go2的12个关节）

4. **Decoder网络**: 重建当前观测
   - 输入：32维编码
   - 输出：45维观测（重建）
   - 用于训练encoder，通过重建损失更新权重

### 损失函数

训练过程中包含以下损失：

1. **PPO标准损失**:
   - Surrogate loss (策略损失)
   - Value loss (价值函数损失)
   - Entropy loss (熵正则化)

2. **DWAQ特有损失**:
   - `recon_loss`: 重建损失（MSE between decoded and target observation）
   - `kl_vel_loss`: 速度编码的KL散度损失
   - `kl_latent_loss`: 环境隐式表达的KL散度损失

## 文件修改说明

### 1. rsl_rl/rsl_rl/modules/actor_critic_DWAQ.py
- ✅ 修改构造函数以兼容框架
- ✅ 添加观测历史管理 (obs_history buffer)
- ✅ 实现 `act()`, `evaluate()`, `act_inference()` 方法
- ✅ 添加 `get_encoder_outputs()` 用于损失计算
- ✅ 修正decoder输出维度为45（匹配policy观测）

### 2. rsl_rl/rsl_rl/modules/__init__.py
- ✅ 导出 ActorCritic_DWAQ 类

### 3. rsl_rl/rsl_rl/algorithms/ppo.py
- ✅ 添加 dwaq_cfg 参数支持
- ✅ 在 `update()` 方法中添加encoder-decoder损失计算
- ✅ 记录和返回DWAQ相关损失指标

### 4. unitree_rl_lab配置文件
- ✅ 创建 `rsl_rl_ppo_dwaq_cfg.py` 配置文件
- ✅ 在 go2/__init__.py 注册新环境 "Unitree-Go2-Velocity-DWAQ"

## 使用方法

### 训练命令

```bash
cd /home/langyi/Projects/DreamWaQ/unitree_rl_lab

# 使用DWAQ模型训练Go2
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity-DWAQ --headless

# 或者使用完整命令
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity-DWAQ --headless
```

### 配置参数调整

如需调整DWAQ参数，编辑 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_dwaq_cfg.py`:

```python
@configclass
class DWAQActorCriticCfg(RslRlPpoActorCriticCfg):
    history_length: int = 5          # 观测历史长度
    cenet_out_dim: int = 32          # 编码器输出维度 (3+29)
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]

@configclass
class DWAQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    dwaq_cfg: dict = {
        "recon_loss_coef": 1.0,      # 重建损失权重
        "kl_vel_loss_coef": 0.01,    # 速度KL损失权重
        "kl_latent_loss_coef": 0.01, # 隐变量KL损失权重
    }
```

## 观测维度说明

### Go2 Policy观测 (45维)
- base_ang_vel: 3维
- projected_gravity: 3维
- velocity_commands: 3维
- joint_pos_rel: 12维
- joint_vel_rel: 12维
- last_action: 12维

### Go2 Critic观测 (60维 - 特权观测)
- base_lin_vel: 3维 (特权)
- base_ang_vel: 3维
- projected_gravity: 3维
- velocity_commands: 3维
- joint_pos_rel: 12维
- joint_vel_rel: 12维
- joint_effort: 12维 (特权)
- last_action: 12维

### Encoder处理
- 输入: 45维 × 5步历史 = 225维
- 输出: 32维 (3维速度 + 29维隐变量)

## 预期训练流程

1. **初始化**: 创建ActorCritic_DWAQ模型，初始化观测历史缓冲区
2. **采样**: 每步执行`act()`，更新观测历史，通过encoder生成编码
3. **评估**: 使用`evaluate()`基于特权观测计算价值
4. **更新**: 
   - 计算PPO标准损失
   - 调用`get_encoder_outputs()`获取encoder输出
   - 计算重建损失和KL损失
   - 反向传播更新所有网络

## 监控指标

训练过程中会记录以下DWAQ相关指标：
- `recon`: 重建损失
- `kl_vel`: 速度编码KL散度
- `kl_latent`: 环境隐变量KL散度

可在TensorBoard或wandb中查看这些指标。

## 调试建议

1. **如果重建损失很高**: 增加`recon_loss_coef`或检查decoder网络容量
2. **如果KL损失爆炸**: 降低`kl_vel_loss_coef`和`kl_latent_loss_coef`
3. **如果训练不稳定**: 尝试降低学习率或增加`num_mini_batches`

## 注意事项

1. 确保环境正确配置，所有依赖已安装
2. DWAQ模型需要更多内存（存储观测历史）
3. 训练速度可能比标准PPO稍慢（额外的encoder-decoder计算）
4. 建议先用较小的`num_envs`测试（如1024），确认能正常运行后再增加

## 下一步

现在可以运行训练命令测试整个流程。如果遇到错误，请检查：
1. 模块导入是否正确
2. 观测维度是否匹配
3. 损失计算是否有数值问题

祝训练顺利！🚀

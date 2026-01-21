# AdaBoot 实现总结

## 修改的文件

### 1. `rsl_rl/rsl_rl/modules/actor_critic_DWAQ.py`

**修改内容**：
- `act()` 方法新增 `boot_mask` 参数
- 根据 `boot_mask` 动态选择速度来源：
  - `boot_mask[i]=True` → 使用 CENet 估算速度（bootstrap）
  - `boot_mask[i]=False` → 使用 Ground Truth 真实速度

**关键代码**：
```python
def act(self, obs: TensorDict, boot_mask: torch.Tensor | None = None, **kwargs):
    # ... 获取 CENet 估算速度和真实速度 ...
    
    if boot_mask is not None:
        # 根据 mask 选择速度
        vel_for_policy = torch.where(
            boot_mask.unsqueeze(1),
            estimated_vel,      # CENet 估算 (bootstrap)
            self.ground_truth_vel  # 真实速度 (不 bootstrap)
        )
        # Actor 输入: o_t + v_t + z_t
        actor_input = torch.cat([policy_obs, vel_for_policy, latent], dim=-1)
```

---

### 2. `rsl_rl/rsl_rl/algorithms/ppo.py`

**修改内容**：
- `act()` 方法新增 `boot_mask` 参数
- 传递 `boot_mask` 给 `policy.act()`

**关键代码**：
```python
def act(self, obs: TensorDict, boot_mask: torch.Tensor | None = None):
    self.transition.actions = self.policy.act(obs, boot_mask=boot_mask).detach()
```

---

### 3. `rsl_rl/rsl_rl/runners/on_policy_runner.py`

**修改内容**：
- `__init__()` 中添加 AdaBoot 初始化逻辑
- `learn()` 方法中实现 CV 计算、p_boot 计算和 boot_mask 生成
- 添加 `_compute_reward_cv()` 辅助方法
- 收集 episode rewards 用于 CV 计算
- 在日志中记录 AdaBoot 指标

**关键代码**：
```python
# 初始化
self.adaboot_enabled = self.alg_cfg.get("adaboot_enabled", False)
self.episode_rewards = []
self.cv_reward = float('inf')
self.p_boot = 0.0

# 训练循环
if self.adaboot_enabled:
    # 计算 CV 和 p_boot
    self.cv_reward = self._compute_reward_cv()
    self.p_boot = 1.0 - torch.tanh(torch.tensor(self.cv_reward)).item()
    
    # 生成 boot_mask
    boot_mask = torch.rand(self.env.num_envs, device=self.device) < self.p_boot

# 执行 rollout
actions = self.alg.act(obs, boot_mask=boot_mask)
```

---

### 4. `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

**修改内容**：
- 在 `RslRlPpoAlgorithmCfg` 中添加 `adaboot_enabled` 配置选项

**关键代码**：
```python
algorithm = RslRlPpoAlgorithmCfg(
    # ... 其他参数 ...
    adaboot_enabled=False,  # 设置为 True 启用 AdaBoot
)
```

---

## AdaBoot 工作原理

### 论文公式
```
CV(R) = std(R) / mean(R)
p_boot = 1 - tanh(CV(R))
```

### 训练流程

```
每个 Iteration:
  1. 收集 episode rewards from 4096 envs
     ↓
  2. 计算 CV(R) = std(episode_rewards) / mean(episode_rewards)
     ↓
  3. 计算 p_boot = 1 - tanh(CV(R))
     - 训练初期: CV 大 → p_boot 小 → 多用真实速度
     - 训练后期: CV 小 → p_boot 大 → 多用估算速度
     ↓
  4. 生成 boot_mask: [4096] bool tensor
     mask[i] = (random() < p_boot)
     ↓
  5. Rollout (24 steps):
     For each step:
       - Policy 根据 mask 选择速度:
           vel[i] = CENet估算 if mask[i] else 真实速度
       - Actor 输入: o_t (45) + vel (3) + z_t (16) = 64
       - 执行动作，收集数据
     ↓
  6. PPO Update (使用 98,304 个样本)
```

### 训练进度示例

| Iteration | CV    | p_boot | Bootstrap % | Ground Truth % | 说明 |
|-----------|-------|--------|-------------|----------------|------|
| 0         | 1.50  | 0.09   | 9.5%        | 90.5%          | 初期：多用真实速度帮助学习 |
| 500       | 0.80  | 0.34   | 33.6%       | 66.4%          | 中期：逐渐过渡 |
| 1000      | 0.50  | 0.54   | 53.8%       | 46.2%          | 中期：平衡状态 |
| 2000      | 0.30  | 0.71   | 70.9%       | 29.1%          | 后期：更多估算 |
| 5000      | 0.15  | 0.85   | 85.1%       | 14.9%          | 后期：接近纯估算 |
| 10000     | 0.05  | 0.95   | 95.0%       | 5.0%           | 稳定：高鲁棒性 |

---

## 如何使用

### 方式 1：修改配置文件（推荐）

在 Go2 的 agent 配置中启用 AdaBoot：

**文件**: `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/agents/rsl_rl_ppo_cfg_dwaq.py`

```python
@configclass
class UnitreeGo2RoughPPORunnerCfg(BasePPORunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        # ... 其他参数 ...
        adaboot_enabled=True,  # ← 启用 AdaBoot
    )
```

### 方式 2：命令行参数（如果支持）

```bash
python scripts/rsl_rl/train.py \
    --task=Isaac-Velocity-Go2-v0 \
    --algorithm.adaboot_enabled=True
```

### 验证 AdaBoot 是否生效

训练日志中会显示：
```
[AdaBoot] Adaptive Bootstrapping enabled
```

TensorBoard 中会记录以下指标：
- `adaboot/cv_reward`: 当前 episode reward 的 CV
- `adaboot/p_boot`: 当前 bootstrapping 概率
- `adaboot/num_bootstrap`: 使用 CENet 估算的环境数量
- `adaboot/num_groundtruth`: 使用真实速度的环境数量

---

## 关键优势

1. **训练初期加速学习**：使用真实速度辅助 policy 快速学习
2. **训练后期提升鲁棒性**：逐渐过渡到估算速度，适应 sim-to-real 误差
3. **自动化调整**：根据训练稳定性（CV）自动调整 bootstrapping 比例
4. **与论文一致**：完全符合 DreamWaQ 论文的 AdaBoot 设计

---

## 测试验证

运行测试脚本验证实现：
```bash
python test_adaboot.py
```

预期输出：所有测试通过 ✓

---

## 注意事项

1. **仅适用于 DWAQ Policy**：AdaBoot 需要 `ActorCritic_DWAQ` 才能工作
2. **需要 Critic Observations**：必须在 critic obs 中包含 `base_lin_vel`（真实速度）
3. **Episode Rewards 收集**：依赖环境在 `extras` 中返回 episode 信息
4. **默认关闭**：需要手动设置 `adaboot_enabled=True` 启用

---

## 实现验证清单

- [x] `actor_critic_DWAQ.py` 支持 boot_mask 参数
- [x] `ppo.py` 传递 boot_mask
- [x] `on_policy_runner.py` 实现 CV 计算和 mask 生成
- [x] 配置文件添加 `adaboot_enabled` 选项
- [x] 代码无语法错误
- [x] 测试脚本验证逻辑正确
- [x] 与论文公式一致
- [x] 日志记录 AdaBoot 指标

---

## 修改总结

**总共修改 4 个文件，新增约 150 行代码**

核心逻辑：
- Policy 输入构建时，根据 boot_mask 动态选择速度来源
- 训练初期多用真实速度（Ground Truth），后期多用 CENet 估算（Bootstrap）
- 完全符合 DreamWaQ 论文 AdaBoot 机制

**实现完成，可以开始训练！** ✓

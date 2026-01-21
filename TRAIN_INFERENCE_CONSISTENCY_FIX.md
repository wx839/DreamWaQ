# 训练-推理一致性问题修复 + AdaBoot公式修正

## 问题1：训练-推理不一致 (Train-Inference Mismatch)

**症状**：即使 `p_boot=0`（使用ground truth速度），机器人在play/推理时仍然站不稳。

### 根本原因
- `boot_mask` 全为 False
- Actor输入 = `[policy_obs, ground_truth_vel, latent]`
- **Actor学会依赖完美的速度信息**
- CENet虽然被训练（通过L_est loss），但Actor从未使用CENet的输出

### 推理/Play阶段
- `act_inference()` 函数没有ground truth可用
- Actor输入 = `[policy_obs, CENet_estimated_vel, latent]`  
- **强制使用CENet估计的速度**
- Actor从未在训练时见过CENet的估计值

### 为什么会失败
1. **分布偏移**：训练时见到完美速度，推理时得到有误差的估计
2. **策略脆弱性**：Actor学习的策略假设速度是准确的
3. **泛化失败**：面对CENet的估计误差时，策略崩溃

## 解决方案

### 核心思路：确保训练时Actor也见到CENet估计

通过设置 **最小bootstrap概率 `min_p_boot = 0.1`**：

```python
# 修改前：p_boot可以为0
self.p_boot = 1.0 - torch.tanh(torch.tensor(self.cv_reward)).item()
self.p_boot = max(0.0, min(1.0, self.p_boot))  # [0, 1]

# 修改后：p_boot最小为0.1
self.p_boot = 1.0 - torch.tanh(torch.tensor(self.cv_reward)).item()
min_p_boot = 0.1  # 至少10%环境使用CENet
self.p_boot = max(min_p_boot, min(1.0, self.p_boot))  # [0.1, 1]
```

### 效果

1. **训练时**：
   - 至少10%的环境使用CENet估计速度
   - Actor必须学会处理CENet的估计误差
   - 策略更加鲁棒

2. **推理时**：
   - 使用CENet估计速度（与训练时一致）
   - Actor已经适应了CENet的误差特性
   - 性能不会崩溃

## 原理说明

### AdaBoot的设计意图
- **早期训练**（CV高，不稳定）：更多使用CENet → 鼓励探索
- **后期训练**（CV低，稳定）：更多使用ground truth → 精细优化

### 为什么需要min_p_boot
- **训练稳定性**：10% CENet + 90% GT = 既探索又稳定
- **推理一致性**：Actor见过CENet的输出，不会在部署时惊讶
- **渐进式学习**：从不完美估计开始，逐步提升

## 代码修改位置

### 1. runner初始化
**文件**: `rsl_rl/runners/on_policy_runner.py`
**位置**: 第67行
```python
self.p_boot = 0.1  # 初始值从0.0改为0.1
```

### 2. p_boot计算逻辑  
**文件**: `rsl_rl/runners/on_policy_runner.py`
**位置**: 第101-108行
```python
min_p_boot = 0.1  # 设置最小值
self.p_boot = max(min_p_boot, min(1.0, self.p_boot))
```

### 3. 文档说明
**文件**: `rsl_rl/modules/actor_critic_DWAQ.py`
**位置**: `act_inference()` docstring
添加了说明为何需要min_p_boot

## 预期效果

修复后，即使在训练早期（CV高，p_boot应该接近1）：
- 实际p_boot会被限制在[0.1, 1]范围
- Actor始终能见到CENet的估计
- 推理时不会因为训练-推理不一致而失败

## 进一步优化建议

如果仍有问题，可以考虑：

1. **调整min_p_boot**：
   - 增大到0.2-0.3：更强的一致性保证
   - 减小到0.05：更多使用ground truth

2. **动态调整策略**：
   ```python
   # 早期训练多用CENet，后期可以降低
   if it < 1000:
       min_p_boot = 0.3
   else:
       min_p_boot = 0.1
   ```

3. **监控CENet质量**：
   - 观察 `vel_est_loss` 指标
   - 如果loss很大，说明CENet估计不准，需要更多训练

## 理论依据

这个修复基于机器学习的一个基本原则：
> **训练分布应该覆盖测试分布**

如果训练时只见perfect data，测试时见noisy data → 必然失败。

---

## 问题2：AdaBoot公式逻辑错误 ⚠️

### 问题发现

训练日志显示：
```
CV=5.1170 (极高，训练非常不稳定)
p_boot=0.1000 (被限制在最小值)
mean_reward=-38.70 (性能很差)
```

### 原始公式的问题

**错误公式**：`p_boot = 1.0 - tanh(CV)`

当CV=5.1时：
- `tanh(5.1) ≈ 1.0`
- `p_boot = 1.0 - 1.0 ≈ 0.0`
- **含义**：训练不稳定时，少用CENet，多用ground truth

**这与AdaBoot设计理念相反！**

### 正确的逻辑

AdaBoot的核心思想：
- **训练不稳定（CV高）→ 鼓励探索 → 多用CENet（p_boot高）**
- **训练稳定（CV低）→ 精细优化 → 多用ground truth（p_boot低）**

### 修正后的公式

**正确公式**：`p_boot = tanh(CV)`

| CV值 | tanh(CV) | p_boot | 含义 |
|------|----------|--------|------|
| 0.5  | 0.46     | 0.46   | 稳定，少探索 |
| 1.0  | 0.76     | 0.76   | 中等，平衡 |
| 2.0  | 0.96     | 0.90   | 不稳定，多探索（限制≤0.9）|
| 5.0  | 0.99     | 0.90   | 极不稳定，最大探索 |

加上限制：`p_boot ∈ [0.1, 0.9]`
- **下限0.1**：确保训练-推理一致性
- **上限0.9**：保留10%的ground truth用于学习

### 代码修改

```python
# 修改前（错误）
self.p_boot = 1.0 - torch.tanh(torch.tensor(self.cv_reward)).item()
min_p_boot = 0.1
self.p_boot = max(min_p_boot, min(1.0, self.p_boot))

# 修改后（正确）
self.p_boot = torch.tanh(torch.tensor(self.cv_reward)).item()
min_p_boot = 0.1  # 训练-推理一致性
max_p_boot = 0.9  # 保留ground truth用于学习
self.p_boot = max(min_p_boot, min(max_p_boot, self.p_boot))
```

### 初始值调整

```python
# 修改前
self.p_boot = 0.1  # 初始值过低

# 修改后
self.p_boot = 0.7  # 初始值较高，鼓励早期探索
```

### 预期效果

修正后，当CV=5.1时：
- `p_boot = tanh(5.1) ≈ 0.99` → 限制为0.9
- **90%环境使用CENet探索**（不是之前的10%）
- 训练应该更加稳定，reward逐渐提升
- 随着训练进行，CV降低，p_boot自动减小，更多使用ground truth优化

---

## 完整修改总结

### 修改位置

1. **初始化** (`on_policy_runner.py:66`)
   - 初始p_boot: 0.1 → 0.7

2. **p_boot计算** (`on_policy_runner.py:101-113`)
   - 公式：`1.0 - tanh(CV)` → `tanh(CV)`
   - 范围：[0.1, 1.0] → [0.1, 0.9]
   - 初始值：0.1 → 0.7

### 预期训练曲线

```
迭代初期（CV高≈5）：
  p_boot ≈ 0.9 → 90% CENet探索 → 快速学习
  
迭代中期（CV降至≈1）：
  p_boot ≈ 0.76 → 平衡探索与利用
  
迭代后期（CV低≈0.3）：
  p_boot ≈ 0.3 → 70% ground truth精细优化
```

### 理论支持

这符合强化学习的经典范式：
- **探索-利用权衡（Exploration-Exploitation Tradeoff）**
- 早期探索，后期利用
- CV自然地衡量了训练的稳定性

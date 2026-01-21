# 四足机器人防拖地与抬腿高度控制奖励函数设置指南

## 概述
本文档详细描述了一套用于解决四足机器人腿部拖地和抬腿高度偏低问题的奖励函数设置方案。该方案已在 GO2 机器人项目中验证有效。

---

## 一、奖励函数配置参数

### 1.1 在配置文件中添加以下参数

```python
class rewards:
    # 目标参数
    soft_dof_pos_limit = 0.9          # 关节位置软限制（占URDF限制的比例）
    base_height_target = 0.30         # 目标机身高度（米，相对地面）
    clearance_height_target = -0.18   # 目标抬腿高度（米，相对机身中心，负值表示在机身下方）
    max_contact_force = 40.0          # 最大接触力阈值（牛顿）
    
    class scales:
        # 核心奖励权重 - 防拖地与抬腿控制
        foot_clearance = -0.01        # 抬腿高度控制（惩罚）
        collision = -0.2              # 腿部碰撞惩罚（惩罚大腿/小腿拖地）
        base_height = -1.0            # 机身高度保持（惩罚）
        
        # 辅助奖励权重
        tracking_lin_vel = 1.0        # 线速度跟踪（奖励）
        tracking_ang_vel = 0.5        # 角速度跟踪（奖励）
        orientation = -0.2            # 姿态保持（惩罚）
        ang_vel_xy = -0.05            # XY轴角速度（惩罚）
        lin_vel_z = -2.0              # Z轴线速度（惩罚）
        dof_acc = -2.5e-7             # 关节加速度（惩罚）
        joint_power = -2.e-5          # 关节功率（惩罚）
        action_rate = -0.01           # 动作变化率（惩罚）
        smoothness = -0.01            # 平滑性（惩罚）
```

### 1.2 资产配置参数

```python
class asset:
    foot_name = "foot"                              # 脚部刚体名称模式
    penalize_contacts_on = ["thigh", "calf"]        # 需要惩罚接触的部位（大腿、小腿）
    terminate_after_contacts_on = ["base"]          # 触发终止的接触部位（机身）
```

---

## 二、奖励函数实现代码

### 2.1 foot_clearance - 抬腿高度控制

**目的**: 在脚摆动时（横向移动），强制脚保持在目标高度，防止抬得太低或拖地。

**实现代码**:
```python
def _reward_foot_clearance(self):
    """
    抬腿高度控制奖励函数
    - 将脚的位置和速度转换到机身本体坐标系
    - 当脚有横向速度时（摆动相），惩罚其高度偏离目标值
    - 高度误差 × 横向速度 = 惩罚值（摆动越快，惩罚越大）
    """
    # 步骤1: 计算脚相对机身的位置（世界坐标系）
    cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    
    # 步骤2: 计算脚相对机身的速度（世界坐标系）
    cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    
    # 步骤3: 转换到机身本体坐标系
    for i in range(len(self.feet_indices)):
        footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
    
    # 步骤4: 计算脚的高度误差（Z轴）
    height_error = torch.square(
        footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target
    ).view(self.num_envs, -1)
    
    # 步骤5: 计算脚的横向速度（XY平面）
    foot_leteral_vel = torch.sqrt(
        torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
    ).view(self.num_envs, -1)
    
    # 步骤6: 返回惩罚值（只在摆动时惩罚高度偏差）
    return torch.sum(height_error * foot_leteral_vel, dim=1)
```

**关键参数说明**:
- `clearance_height_target = -0.18`: 
  - 负值表示脚在机身中心下方
  - -0.18米 ≈ 机身下方18厘米
  - 如果机身高度0.42米，则脚离地约0.24米
  
**调整建议**:
- 若拖地严重：增大权重至 `-0.05` 或更大，或调整目标至 `-0.15`（抬得更高）
- 若抬得太高影响速度：减小权重至 `-0.005`，或调整目标至 `-0.20`

---

### 2.2 collision - 腿部碰撞惩罚

**目的**: 直接惩罚大腿或小腿与地面/障碍物的接触，防止拖地。

**实现代码**:
```python
def _reward_collision(self):
    """
    腿部碰撞惩罚函数
    - 检测大腿(thigh)和小腿(calf)是否与地面接触
    - 任何接触力大于0.1N都会被惩罚
    - 返回所有碰撞点的数量
    """
    # 计算惩罚接触部位的接触力范数
    contact_force_norm = torch.norm(
        self.contact_forces[:, self.penalised_contact_indices, :], 
        dim=-1
    )
    
    # 统计接触力大于阈值(0.1N)的接触点数量
    collision_count = torch.sum(
        1. * (contact_force_norm > 0.1), 
        dim=1
    )
    
    return collision_count
```

**前置要求**:
在环境初始化时，需要设置 `penalised_contact_indices`：

```python
# 在 _create_envs() 或类似初始化函数中
penalised_contact_names = []
for name in self.cfg.asset.penalize_contacts_on:  # ["thigh", "calf"]
    penalised_contact_names.extend([s for s in self.body_names if name in s])

self.penalised_contact_indices = torch.zeros(
    len(penalised_contact_names), 
    dtype=torch.long, 
    device=self.device, 
    requires_grad=False
)

for i, name in enumerate(penalised_contact_names):
    self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
        self.envs[0], 
        self.actor_handles[0], 
        name
    )
```

**调整建议**:
- 若拖地频繁但权重已很大：检查机身高度设置和初始姿态
- 若误惩罚正常步态：降低接触力阈值（如改为 `> 0.5`）或减小权重至 `-0.1`

---

### 2.3 base_height - 机身高度保持

**目的**: 保持机身在目标高度，防止机身下沉导致腿部拖地。

**实现代码**:
```python
def _reward_base_height(self):
    """
    机身高度保持奖励函数
    - 计算机身相对地面的平均高度
    - 惩罚与目标高度的偏差（平方误差）
    - 防止机身下沉或过高
    """
    # 计算机身相对地形的平均高度
    # root_states[:, 2] 是机身Z坐标
    # measured_heights 是机身周围地形高度采样点
    base_height = torch.mean(
        self.root_states[:, 2].unsqueeze(1) - self.measured_heights, 
        dim=1
    )
    
    # 计算与目标高度的平方误差
    height_error = torch.square(base_height - self.cfg.rewards.base_height_target)
    
    return height_error
```

**前置要求**:
需要实现地形高度采样 `measured_heights`：

```python
# 在 post_physics_step() 或观测更新中
self.measured_heights = self._get_heights()

def _get_heights(self, env_ids=None):
    """
    采样机身周围地形高度
    返回形状: [num_envs, num_height_points]
    """
    # 实现地形高度查询逻辑
    # 对于平地可以简单返回0
    if self.cfg.terrain.mesh_type == 'plane':
        return torch.zeros(
            self.num_envs, 
            self.num_height_points, 
            device=self.device, 
            requires_grad=False
        )
    # 对于复杂地形需要采样地形网格
    # ... 具体实现见项目地形模块
```

**关键参数说明**:
- `base_height_target = 0.30`: 
  - 目标机身高度为0.30米（相对地面）
  - 根据机器人尺寸调整（GO2约0.30-0.42米）
  
**调整建议**:
- 若机身晃动大：增大权重至 `-2.0` 或 `-3.0`
- 若需要更高机身：调整目标至 `0.35` 或 `0.40`
- 若在复杂地形：考虑使用更大权重以保持稳定

---

## 三、完整集成步骤

### 3.1 配置文件设置
在你的机器人配置类中添加：

```python
class YourRobotCfg:
    class asset:
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
    
    class rewards:
        # 目标参数
        base_height_target = 0.30
        clearance_height_target = -0.18
        max_contact_force = 40.0
        
        class scales:
            # 三大核心奖励
            foot_clearance = -0.01
            collision = -0.2
            base_height = -1.0
            
            # 其他奖励...
```

### 3.2 环境类实现
在你的环境类中添加奖励函数：

```python
class YourRobotEnv(BaseEnv):
    def __init__(self, ...):
        # ... 初始化代码 ...
        self._prepare_reward_function()
    
    def _prepare_reward_function(self):
        """
        自动注册所有权重非零的奖励函数
        查找所有 _reward_<name> 方法并添加到奖励函数列表
        """
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt  # 缩放到时间步
        
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            func_name = '_reward_' + name
            self.reward_functions.append(getattr(self, func_name))
    
    def compute_reward(self):
        """
        计算总奖励
        调用所有注册的奖励函数并求和
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
    
    # 添加三个核心奖励函数
    def _reward_foot_clearance(self):
        # [使用上面2.1的完整代码]
        pass
    
    def _reward_collision(self):
        # [使用上面2.2的完整代码]
        pass
    
    def _reward_base_height(self):
        # [使用上面2.3的完整代码]
        pass
```

### 3.3 必要的状态变量
确保环境中有以下变量：

```python
# 机器人状态
self.root_states          # [num_envs, 13] 机身位置、姿态、速度
self.base_quat            # [num_envs, 4] 机身四元数
self.feet_pos             # [num_envs, num_feet, 3] 脚的世界坐标
self.feet_vel             # [num_envs, num_feet, 3] 脚的速度
self.contact_forces       # [num_envs, num_bodies, 3] 接触力
self.measured_heights     # [num_envs, num_points] 地形高度采样

# 索引
self.feet_indices                # 脚的刚体索引
self.penalised_contact_indices   # 需要惩罚的接触部位索引
```

---

## 四、验证与调试

### 4.1 验证清单
- [ ] 配置文件中三个权重已设置且非零
- [ ] 三个奖励函数已实现并注册
- [ ] `penalised_contact_indices` 正确初始化
- [ ] `measured_heights` 能正确返回地形高度
- [ ] 日志中能看到三个奖励项的数值

### 4.2 调试命令
在训练日志中查看各项奖励：

```python
# 在 reset_idx() 中记录奖励统计
for key in self.episode_sums.keys():
    self.extras["episode"]['rew_' + key] = torch.mean(
        self.episode_sums[key][env_ids]
    ) / self.max_episode_length_s
```

期望看到：
- `rew_foot_clearance`: 负值，约 -0.01 到 -0.1
- `rew_collision`: 负值，理想情况接近0（无碰撞）
- `rew_base_height`: 负值，约 -0.5 到 -2.0

### 4.3 常见问题

**Q: foot_clearance 奖励始终为0**
- 检查 `feet_vel` 是否正确更新
- 确认 `quat_rotate_inverse` 函数可用
- 验证 `feet_indices` 索引正确

**Q: collision 惩罚过大，影响正常步态**
- 检查 `penalize_contacts_on` 是否只包含大腿/小腿
- 确认没有误将脚部包含在内
- 尝试提高接触力阈值（0.1 → 0.5）

**Q: base_height 导致机身抖动**
- 检查 `measured_heights` 是否平滑
- 降低权重（-1.0 → -0.5）
- 确认 `base_height_target` 设置合理

---

## 五、使用此文档的 AI Prompt

```
请根据以下奖励函数设置指南，在我的四足机器人项目中实现防拖地和抬腿高度控制功能：

1. 在配置文件中添加以下参数：
   - base_height_target = 0.30
   - clearance_height_target = -0.18
   - foot_clearance = -0.01
   - collision = -0.2
   - base_height = -1.0
   - penalize_contacts_on = ["thigh", "calf"]

2. 实现三个奖励函数：
   - _reward_foot_clearance(): 控制摆动腿的高度，防止抬得太低
   - _reward_collision(): 惩罚大腿/小腿与地面接触
   - _reward_base_height(): 保持机身高度稳定

3. 具体实现细节请参考《四足机器人防拖地与抬腿高度控制奖励函数设置指南》中的完整代码。

4. 确保以下前置条件：
   - feet_pos, feet_vel 状态变量正确更新
   - penalised_contact_indices 正确初始化
   - measured_heights 地形高度采样可用
   - _prepare_reward_function() 自动注册机制工作正常

请按照文档第三部分"完整集成步骤"逐步实现。
```

---

## 六、参数调优参考表

| 场景 | foot_clearance | collision | base_height | clearance_target |
|------|----------------|-----------|-------------|------------------|
| 标准设置 | -0.01 | -0.2 | -1.0 | -0.18 |
| 严重拖地 | -0.05 | -0.5 | -1.5 | -0.15 |
| 复杂地形 | -0.02 | -0.3 | -2.0 | -0.15 |
| 高速奔跑 | -0.005 | -0.1 | -0.5 | -0.20 |
| 精确控制 | -0.03 | -0.4 | -1.5 | -0.16 |

---

## 七、技术原理说明

### 7.1 为什么 clearance_height_target 是负值
- 机器人本体坐标系原点在**机身中心**
- Z轴向上为正，向下为负
- 脚通常在机身下方，因此Z坐标为负
- `-0.18` 表示脚在机身中心下方18厘米

### 7.2 为什么 foot_clearance 使用横向速度加权
- **摆动相**（脚在空中）：横向速度大，需要保持目标高度
- **支撑相**（脚在地面）：横向速度小，允许高度变化
- 这样避免惩罚支撑腿，只约束摆动腿

### 7.3 为什么需要三个奖励配合
- `foot_clearance`: 主动控制抬腿高度（预防性）
- `collision`: 检测实际拖地（反应性）
- `base_height`: 维持全局高度（稳定性）
- 三者互补，形成完整的防拖地体系

---

## 版本信息
- 文档版本: 1.0
- 测试平台: Unitree GO2
- 仿真环境: Isaac Gym
- 验证日期: 2026-01-20

---

## 联系与支持
如在实施过程中遇到问题，请检查：
1. 机器人URDF模型中刚体命名是否包含 "thigh", "calf", "foot"
2. 机器人初始姿态是否合理（腿不应初始就拖地）
3. 仿真时间步是否设置正确（通常0.005-0.01秒）
4. 奖励权重是否已乘以时间步（在 _prepare_reward_function 中）

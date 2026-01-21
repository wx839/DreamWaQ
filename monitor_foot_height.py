#!/usr/bin/env python3
"""
监控和可视化四足机器人腿离地高度的脚本

使用方法：
1. 在训练或测试时运行，会自动记录足部高度数据
2. 可以实时打印或保存为CSV文件
3. 可以生成可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import os


class FootHeightMonitor:
    """足部高度监控器"""
    
    def __init__(self, save_dir="logs/foot_height", enable_plot=True):
        self.save_dir = save_dir
        self.enable_plot = enable_plot
        self.data = {
            'time': [],
            'FL_foot': [],  # Front Left
            'FR_foot': [],  # Front Right
            'RL_foot': [],  # Rear Left
            'RR_foot': [],  # Rear Right
            'FL_contact': [],
            'FR_contact': [],
            'RL_contact': [],
            'RR_contact': [],
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        self.csv_filename = os.path.join(
            save_dir, 
            f"foot_height_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
    def record(self, time, foot_heights, contact_states, foot_names=None):
        """记录一帧数据
        
        Args:
            time: 当前时间（秒）
            foot_heights: dict或list，各足部高度 {foot_name: height}
            contact_states: dict或list，各足部接触状态 {foot_name: is_contact}
            foot_names: 足部名称列表（如果foot_heights是list）
        """
        self.data['time'].append(time)
        
        if isinstance(foot_heights, dict):
            for key in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
                self.data[key].append(foot_heights.get(key, 0.0))
                self.data[f'{key.split("_")[0]}_contact'].append(contact_states.get(key, False))
        else:
            # 假设顺序是 FL, FR, RL, RR
            foot_keys = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
            for i, key in enumerate(foot_keys):
                if i < len(foot_heights):
                    self.data[key].append(foot_heights[i])
                    self.data[f'{key.split("_")[0]}_contact'].append(contact_states[i] if i < len(contact_states) else False)
    
    def save_csv(self):
        """保存数据为CSV文件"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['time', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot', 
                         'FL_contact', 'FR_contact', 'RL_contact', 'RR_contact']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(self.data['time'])):
                row = {key: self.data[key][i] for key in fieldnames}
                writer.writerow(row)
        
        print(f"数据已保存到: {self.csv_filename}")
    
    def plot(self, save_fig=True):
        """绘制足部高度变化图"""
        if not self.enable_plot or len(self.data['time']) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        time = np.array(self.data['time'])
        
        # 绘制足部高度
        colors = {'FL': 'red', 'FR': 'blue', 'RL': 'green', 'RR': 'orange'}
        for foot in ['FL', 'FR', 'RL', 'RR']:
            heights = np.array(self.data[f'{foot}_foot'])
            contacts = np.array(self.data[f'{foot}_contact'])
            
            # 绘制高度曲线
            ax1.plot(time, heights, label=f'{foot} foot', color=colors[foot], linewidth=2)
            
            # 标记接触地面的区域
            contact_mask = contacts > 0
            if np.any(contact_mask):
                ax1.fill_between(time, 0, heights, where=contact_mask, 
                               alpha=0.2, color=colors[foot], label=f'{foot} contact')
        
        ax1.set_xlabel('时间 (s)', fontsize=12)
        ax1.set_ylabel('离地高度 (m)', fontsize=12)
        ax1.set_title('四足机器人足部离地高度变化', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 绘制接触状态时序图
        for i, foot in enumerate(['FL', 'FR', 'RL', 'RR']):
            contacts = np.array(self.data[f'{foot}_contact'])
            ax2.plot(time, contacts * 0.8 + i, label=f'{foot}', color=colors[foot], linewidth=2)
            ax2.fill_between(time, i, i + contacts * 0.8, alpha=0.5, color=colors[foot])
        
        ax2.set_xlabel('时间 (s)', fontsize=12)
        ax2.set_ylabel('足部', fontsize=12)
        ax2.set_title('足部接触状态时序图', fontsize=14, fontweight='bold')
        ax2.set_yticks([0.4, 1.4, 2.4, 3.4])
        ax2.set_yticklabels(['FL', 'FR', 'RL', 'RR'])
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim(-0.2, 4)
        
        plt.tight_layout()
        
        if save_fig:
            fig_filename = self.csv_filename.replace('.csv', '.png')
            plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {fig_filename}")
        
        plt.show()
    
    def print_summary(self):
        """打印统计摘要"""
        if len(self.data['time']) == 0:
            print("无数据")
            return
        
        print("\n" + "="*70)
        print("足部高度统计摘要")
        print("="*70)
        
        for foot in ['FL', 'FR', 'RL', 'RR']:
            heights = np.array(self.data[f'{foot}_foot'])
            contacts = np.array(self.data[f'{foot}_contact'])
            
            print(f"\n{foot} 足:")
            print(f"  平均高度: {np.mean(heights):.4f} m")
            print(f"  最大高度: {np.max(heights):.4f} m")
            print(f"  最小高度: {np.min(heights):.4f} m")
            print(f"  标准差:   {np.std(heights):.4f} m")
            print(f"  接触时间占比: {np.mean(contacts)*100:.1f}%")
        
        print("="*70 + "\n")


# 使用示例
if __name__ == "__main__":
    # 创建监控器
    monitor = FootHeightMonitor(save_dir="logs/foot_height", enable_plot=True)
    
    # 模拟数据（实际使用时从环境中获取）
    print("模拟数据示例...")
    dt = 0.02  # 20ms步长
    for step in range(500):
        time = step * dt
        
        # 模拟简单的步态
        phase = (time * 2) % 1.0  # 2Hz步频
        
        # 模拟足部高度（简单正弦波）
        foot_heights = {
            'FL_foot': max(0, 0.05 * np.sin(2 * np.pi * phase)),
            'FR_foot': max(0, 0.05 * np.sin(2 * np.pi * (phase + 0.5))),
            'RL_foot': max(0, 0.05 * np.sin(2 * np.pi * (phase + 0.5))),
            'RR_foot': max(0, 0.05 * np.sin(2 * np.pi * phase)),
        }
        
        # 模拟接触状态（高度接近0时接触）
        contact_states = {
            key: height < 0.01 for key, height in foot_heights.items()
        }
        
        monitor.record(time, foot_heights, contact_states)
        
        # 每50步打印一次
        if step % 50 == 0:
            print(f"\n步数: {step}, 时间: {time:.2f}s")
            for key, height in foot_heights.items():
                contact = "接触" if contact_states[key] else "空中"
                print(f"  {key}: {height:.4f}m ({contact})")
    
    # 保存和可视化
    monitor.save_csv()
    monitor.print_summary()
    monitor.plot(save_fig=True)

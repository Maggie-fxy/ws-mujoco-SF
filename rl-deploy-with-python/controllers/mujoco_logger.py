"""
MuJoCo 仿真数据记录和可视化模块
简化版本，适配 SolefootController
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process
import os


class MuJoCoLogger:
    """简化版的 MuJoCo 数据记录器"""
    
    def __init__(self, dt):
        """
        初始化日志记录器
        
        Args:
            dt (float): 仿真时间步长
        """
        self.state_log = defaultdict(list)
        self.dt = dt
        self.plot_process = None
        self.step_count = 0
        
    def log_state(self, key, value):
        """记录单个状态值"""
        self.state_log[key].append(value)
        
    def log_states(self, data_dict):
        """
        记录多个状态值
        
        Args:
            data_dict (dict): 包含状态数据的字典
        """
        for key, value in data_dict.items():
            self.log_state(key, value)
        
        self.step_count += 1
        
    def should_plot(self, plot_interval=200):
        """
        检查是否应该触发绘图
        
        Args:
            plot_interval (int): 绘图间隔步数
            
        Returns:
            bool: 是否应该绘图
        """
        return self.step_count > 0 and self.step_count % plot_interval == 0
        
    def plot_states(self):
        """启动绘图进程"""
        if self.plot_process is not None and self.plot_process.is_alive():
            self.plot_process.terminate()
            
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()
        
    def _plot(self):
        """内部绘图方法"""
        try:
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            # 尝试设置中文字体，如果失败则使用英文
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
            except:
                pass  # 如果中文字体不可用，使用默认字体
            
            nb_rows = 3
            nb_cols = 3
            fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(15, 12))
            fig.suptitle(f'MuJoCo Simulation Data Visualization (Steps: {self.step_count})', fontsize=16)
            
            # 计算时间轴
            if self.state_log:
                max_len = max(len(values) for values in self.state_log.values())
                time = np.linspace(0, max_len * self.dt, max_len)
            else:
                time = np.array([0])
                
            log = self.state_log
            
            # 1. 关节位置 (1,0)
            ax = axs[1, 0]
            if 'dof_pos' in log and log['dof_pos']:
                dof_pos_data = np.array(log['dof_pos'])
                if dof_pos_data.ndim > 1:
                    # 显示所有关节数据（14个关节）
                    joint_names = ['abad_L', 'hip_L', 'knee_L', 'ankle_L', 'abad_R', 'hip_R', 'knee_R', 'ankle_R', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
                    for i in range(min(14, dof_pos_data.shape[1])):
                        joint_name = joint_names[i] if i < len(joint_names) else f'Joint_{i+1}'
                        ax.plot(time[:len(log['dof_pos'])], dof_pos_data[:, i], 
                               label=joint_name, alpha=0.7)
                else:
                    ax.plot(time[:len(log['dof_pos'])], dof_pos_data, label='Joint Position')
            if 'dof_pos_target' in log and log['dof_pos_target']:
                target_data = np.array(log['dof_pos_target'])
                if target_data.ndim > 1:
                    # 显示前4个关节的目标位置（避免图例过于拥挤）
                    for i in range(min(4, target_data.shape[1])):
                        ax.plot(time[:len(log['dof_pos_target'])], target_data[:, i], 
                               '--', alpha=0.5, label=f'Target_{i+1}')
                else:
                    ax.plot(time[:len(log['dof_pos_target'])], target_data, '--', label='Target')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Position [rad]')
            ax.set_title('Joint Position')
            ax.legend()
            ax.grid(True)
            
            # 2. 关节速度 (1,1)
            ax = axs[1, 1]
            if 'dof_vel' in log and log['dof_vel']:
                dof_vel_data = np.array(log['dof_vel'])
                if dof_vel_data.ndim > 1:
                    # 显示所有关节速度数据（14个关节）
                    joint_names = ['abad_L', 'hip_L', 'knee_L', 'ankle_L', 'abad_R', 'hip_R', 'knee_R', 'ankle_R', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
                    for i in range(min(14, dof_vel_data.shape[1])):
                        joint_name = joint_names[i] if i < len(joint_names) else f'Joint_{i+1}'
                        ax.plot(time[:len(log['dof_vel'])], dof_vel_data[:, i], 
                               label=joint_name, alpha=0.7)
                else:
                    ax.plot(time[:len(log['dof_vel'])], dof_vel_data, label='Joint Velocity')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Velocity [rad/s]')
            ax.set_title('Joint Velocity')
            ax.legend()
            ax.grid(True)
            
            # 3. 基础速度 X (0,0)
            ax = axs[0, 0]
            if 'base_vel_x' in log and log['base_vel_x']:
                ax.plot(time[:len(log['base_vel_x'])], log['base_vel_x'], label='实际')
            if 'command_x' in log and log['command_x']:
                ax.plot(time[:len(log['command_x'])], log['command_x'], label='命令')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Linear Velocity [m/s]')
            ax.set_title('Base Velocity X')
            ax.legend()
            ax.grid(True)
            
            # 4. 基础速度 Y (0,1)
            ax = axs[0, 1]
            if 'base_vel_y' in log and log['base_vel_y']:
                ax.plot(time[:len(log['base_vel_y'])], log['base_vel_y'], label='Actual')
            if 'command_y' in log and log['command_y']:
                ax.plot(time[:len(log['command_y'])], log['command_y'], label='Command')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Linear Velocity [m/s]')
            ax.set_title('Base Velocity Y')
            ax.legend()
            ax.grid(True)
            
            # 5. Base Angular Velocity Yaw (0,2)
            ax = axs[0, 2]
            if 'base_vel_yaw' in log and log['base_vel_yaw']:
                ax.plot(time[:len(log['base_vel_yaw'])], log['base_vel_yaw'], label='Actual')
            if 'command_yaw' in log and log['command_yaw']:
                ax.plot(time[:len(log['command_yaw'])], log['command_yaw'], label='Command')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Angular Velocity [rad/s]')
            ax.set_title('Base Angular Velocity Yaw')
            ax.legend()
            ax.grid(True)
            
            # 6. Base Velocity Z (1,2)
            ax = axs[1, 2]
            if 'base_vel_z' in log and log['base_vel_z']:
                ax.plot(time[:len(log['base_vel_z'])], log['base_vel_z'], label='Actual')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Linear Velocity [m/s]')
            ax.set_title('Base Velocity Z')
            ax.legend()
            ax.grid(True)
            
            # 7. Contact Forces (2,0)
            ax = axs[2, 0]
            if 'contact_forces_z' in log and log['contact_forces_z']:
                forces_data = np.array(log['contact_forces_z'])
                if forces_data.ndim > 1:
                    for i in range(min(4, forces_data.shape[1])):
                        ax.plot(time[:len(log['contact_forces_z'])], forces_data[:, i], 
                               label=f'Foot {i+1}')
                else:
                    ax.plot(time[:len(log['contact_forces_z'])], forces_data, label='Contact Force')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Force [N]')
            ax.set_title('Vertical Contact Forces')
            ax.legend()
            ax.grid(True)
            
            # 8. Power (2,1)
            ax = axs[2, 1]
            if 'power' in log and log['power']:
                ax.plot(time[:len(log['power'])], log['power'])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Power [W]')
            ax.set_title('Total Power')
            ax.grid(True)
            
            # 9. Joint Torques (2,2)
            ax = axs[2, 2]
            if 'dof_torque' in log and log['dof_torque']:
                torque_data = np.array(log['dof_torque'])
                if torque_data.ndim > 1:
                    # 显示所有关节力矩数据（14个关节）
                    joint_names = ['abad_L', 'hip_L', 'knee_L', 'ankle_L', 'abad_R', 'hip_R', 'knee_R', 'ankle_R', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
                    for i in range(min(14, torque_data.shape[1])):
                        joint_name = joint_names[i] if i < len(joint_names) else f'Joint_{i+1}'
                        ax.plot(time[:len(log['dof_torque'])], torque_data[:, i], 
                               label=joint_name, alpha=0.7)
                else:
                    ax.plot(time[:len(log['dof_torque'])], torque_data, label='Joint Torque')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Torque [Nm]')
            ax.set_title('Joint Torques')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"绘图过程中出现错误: {e}")
            
    def reset(self):
        """重置日志数据"""
        self.state_log.clear()
        self.step_count = 0
        
    def save_data(self, filename=None):
        """
        保存数据到文件
        
        Args:
            filename (str): 保存文件名，默认自动生成
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mujoco_log_{timestamp}.npz"
            
        # 转换数据格式
        save_dict = {}
        for key, values in self.state_log.items():
            save_dict[key] = np.array(values)
            
        np.savez(filename, **save_dict)
        print(f"数据已保存到: {filename}")
        
    def __del__(self):
        """析构函数，清理进程"""
        if self.plot_process is not None and self.plot_process.is_alive():
            self.plot_process.terminate()

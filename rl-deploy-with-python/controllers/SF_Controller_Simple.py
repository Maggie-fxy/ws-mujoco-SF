#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import yaml
import numpy as np
import mujoco

class SF_Controller_Simple:
    """简化版SoleFoot控制器，只保留可视化功能"""
    
    def __init__(self):
        """初始化控制器"""
        # 模型文件路径
        self.XML_MODEL_PATH = os.path.expanduser("~/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A/xml/robot_with_arm.xml")
        print(f"模型路径: {self.XML_MODEL_PATH}")
        
        # 关节名称
        self.JOINT_NAMES = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint",
            "J1", "J2", "J3", "J4", "J5", "J6"
        ]
        self.joint_num = len(self.JOINT_NAMES)
        
        # 控制参数
        self.DECI = 10  # 控制频率降采样因子
        self.step_count = 0
        self.viewer = None
        self.start_controller = False
        self.mode = "STAND"  # 控制模式: STAND, WALK
        self.simulation_paused = False  # 仿真暂停状态
        
        # 初始化MuJoCo
        self.initialize_mujoco()
    
    def initialize_mujoco(self):
        """初始化MuJoCo模型和可视化界面"""
        try:
            # 加载MuJoCo模型
            print(f"正在加载MuJoCo模型...")
            self.model = mujoco.MjModel.from_xml_path(self.XML_MODEL_PATH)
            self.data = mujoco.MjData(self.model)
            
            # 计算仿真和控制频率
            self.sim_freq = int(0.2/ self.model.opt.timestep)
            self.control_freq = int(self.sim_freq / self.DECI)
            print(f"仿真频率: {self.sim_freq}Hz, 控制频率: {self.control_freq}Hz")
            
            # 初始化默认位置
            mujoco.mj_resetData(self.model, self.data)
            self.default_qpos = self.data.qpos.copy()
            
            # 初始化可视化界面
            try:
                import mujoco.viewer as viewer
                self.viewer = viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
                
                # 设置摄像机参数
                self.viewer.cam.distance = 2.0  # 设置摄像机距离
                self.viewer.cam.elevation = -20  # 设置摄像机仰角
                self.viewer.cam.azimuth = 90  # 设置摄像机方位角
                print("MuJoCo可视化界面初始化成功")
            except Exception as e:
                print(f"MuJoCo可视化界面初始化失败: {e}")
                self.viewer = None
            
            return True
        except Exception as e:
            print(f"MuJoCo初始化失败: {e}")
            return False
    
    def key_callback(self, key, scancode=None, action=None, mods=None):
        """键盘回调函数"""
        if action is not None and action != 1:
            return
        
        print(f"键盘输入: key={key}, scancode={scancode}, action={action}, mods={mods}")
        
        if key == 32:  # 空格键 - 切换控制器状态
            self.start_controller = not self.start_controller
            print("切换控制器状态: " + ("启动" if self.start_controller else "停止"))
        elif key == 80 or key == 112:  # P/p键 - 暂停/恢复仿真
            self.simulation_paused = not self.simulation_paused
            print("仿真状态: " + ("暂停" if self.simulation_paused else "运行中"))
        elif key == 82 or key == 114:  # R/r键 - 重置仿真
            self.reset_simulation()
            print("重置仿真")
        elif key == 83 or key == 115:  # S/s键 - 站立模式
            self.mode = "STAND"
            print("切换到站立模式")
        elif key == 87 or key == 119:  # W/w键 - 行走模式
            self.mode = "WALK"
            print("切换到行走模式")
        elif key == 72 or key == 104:  # H/h键 - 显示帮助
            self.show_help()
        elif key == 81 or key == 113 or key == 256:  # Q/q键或ESC - 退出
            print("退出程序")
            if self.viewer is not None:
                self.viewer.close()
    
    def reset_simulation(self):
        """重置仿真"""
        try:
            mujoco.mj_resetData(self.model, self.data)
            self.step_count = 0
            self.simulation_paused = False  # 重置时恢复仿真
            print("仿真已重置")
        except Exception as e:
            print(f"重置仿真失败: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n=== 控制帮助 ===")
        print("键盘控制:")
        print("  空格键 - 切换控制器状态(启动/停止)")
        print("  P/p键  - 暂停/恢复仿真")
        print("  R/r键  - 重置仿真")
        print("  S/s键  - 切换到站立模式")
        print("  W/w键  - 切换到行走模式")
        print("  H/h键  - 显示帮助信息")
        print("  Q/q键  - 退出程序")
        print("  ESC键  - 退出程序")
        print("\n当前状态:")
        controller_status = "启动" if self.start_controller else "停止"
        sim_status = "暂停" if self.simulation_paused else "运行中"
        print(f"  控制器: {controller_status}")
        print(f"  仿真: {sim_status}")
        print(f"  模式: {self.mode}")
        print(f"  仿真步数: {self.step_count}")
        print("=" * 30)
    
    def run(self):
        """主运行循环"""
        print("开始运行控制器...")
        print("按 'H' 键查看帮助信息")
        
        # 设置控制频率
        control_dt = 1.0 / self.control_freq
        
        # 主循环
        while True:
            try:
                # 记录当前时间
                start_time = time.time()
                
                # 只有在仿真未暂停时才执行仿真步进
                if not self.simulation_paused:
                    mujoco.mj_step(self.model, self.data)
                    self.step_count += 1
                
                # 更新可视化界面
                if self.viewer is not None:
                    try:
                        self.viewer.sync()
                        # 检查是否需要退出
                        if not self.viewer.is_running():
                            print("可视化界面已关闭，退出程序")
                            break
                    except Exception as e:
                        print(f"更新可视化界面失败: {e}")
                        break
                
                # 控制频率延时
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 每秒显示一次状态
                if not self.simulation_paused and self.step_count % self.sim_freq == 0:
                    print(f"仿真时间: {self.step_count / self.sim_freq:.2f}秒, 模式: {self.mode}")
                elif self.simulation_paused and self.step_count % (self.sim_freq * 2) == 0:
                    print(f"仿真已暂停 - 按 'P' 键恢复, 当前步数: {self.step_count}")
            
            except Exception as e:
                print(f"主循环中出错: {e}")
                break

if __name__ == "__main__":
    controller = SF_Controller_Simple()
    controller.run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import mujoco
import numpy as np

def main():
    """测试MuJoCo可视化界面"""
    # 模型路径 - 使用SF_Controller.py中的正确路径
    xml_path = os.path.expanduser("~/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A/xml/robot_with_arm.xml")
    
    print(f"加载模型: {xml_path}")
    
    # 检查文件是否存在
    if not os.path.exists(xml_path):
        print(f"错误: 模型文件不存在: {xml_path}")
        return
    
    try:
        # 加载MuJoCo模型
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # 计算仿真和控制频率
        sim_freq = int(1.0 / model.opt.timestep)
        control_freq = int(sim_freq / 10)  # 控制频率为仿真频率的1/10
        
        # 初始化默认位置
        mujoco.mj_resetData(model, data)
        
        # 定义键盘回调函数
        def key_callback(key, scancode=None, action=None, mods=None):
            print(f"键盘输入: key={key}, scancode={scancode}, action={action}, mods={mods}")
            if key == 81 or key == 113 or key == 256:  # Q/q键或ESC
                print("退出程序")
                if viewer is not None:
                    viewer.close()
        
        # 初始化可视化界面
        import mujoco.viewer as viewer
        viewer_instance = viewer.launch_passive(model, data, key_callback=key_callback)
        
        # 设置摄像机参数
        viewer_instance.cam.distance = 2.0  # 设置摄像机距离
        viewer_instance.cam.elevation = -20  # 设置摄像机仰角
        viewer_instance.cam.azimuth = 90  # 设置摄像机方位角
        
        print("MuJoCo可视化界面初始化成功")
        
        # 设置控制频率
        control_dt = 1.0 / control_freq
        step_count = 0
        
        # 主循环
        while True:
            # 记录当前时间
            start_time = time.time()
            
            # 执行MuJoCo仿真步进
            mujoco.mj_step(model, data)
            step_count += 1
            
            # 更新可视化界面
            if viewer_instance is not None:
                try:
                    viewer_instance.sync()
                    # 检查是否需要退出
                    if not viewer_instance.is_running():
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
            if step_count % sim_freq == 0:
                print(f"仿真时间: {step_count / sim_freq:.2f}秒")
    
    except Exception as e:
        print(f"初始化失败: {e}")

if __name__ == "__main__":
    main()

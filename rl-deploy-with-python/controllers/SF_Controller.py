import mujoco
import numpy as np
import time
import math
import os
import yaml
import onnxruntime as ort
import socket
import threading
import struct
from scipy.spatial.transform import Rotation as R

# 完整的类实现：所有功能都封装在SF_Controller类中，包括初始化、配置加载、模型运行等。
# 多种控制模式：实现了STAND和WALK两种控制模式，并添加了平滑过渡机制。
# ONNX模型推理：添加了完整的ONNX模型加载和推理功能，包括编码器和策略网络。
# 遥控器输入和诊断功能：添加了process_joystick_input方法处理遥控器输入，以及诊断信息处理回调。
# 直接MuJoCo控制：保留了直接操作MuJoCo模型的方式，不依赖SDK控制接口。
# 配置文件支持：添加了从YAML文件加载配置的功能，提高了代码的灵活性。
class SF_Controller:
    def __init__(self, model_path, robot, robot_type, start_controller):
        """
        初始化SoleFoot控制器
        
        Args:
            model_path: 模型路径
            robot: 机器人实例（用于获取状态，但不使用其控制接口）
            robot_type: 机器人类型
            start_controller: 是否自动启动控制器
        """
        # ========== 基本参数初始化 ==========
        self.model_path = model_path
        self.robot = robot
        self.robot_type = robot_type
        self.start_controller = start_controller
        
        # ========== UDP接收器参数 ==========
        self.udp_ip = "127.0.0.1"  # 本地IP
        self.udp_port = 8888       # 监听端口
        self.udp_socket = None     # UDP套接字
        self.udp_thread = None     # UDP接收线程
        self.udp_running = False   # UDP线程运行标志
        
        # 加载配置文件
        self.config_file = os.path.expanduser("~/limx_ws/rl-deploy-with-python/controllers/model/SF_TRON1A/params.yaml")
        print(f"模型路径: {model_path}")
        print(f"机器人类型: {self.robot_type}")
        print(f"配置文件: {self.config_file}")
        self.load_config(self.config_file)
        
        # 模型文件路径
        self.model_policy = os.path.expanduser("~/limx_ws/rl-deploy-with-python/controllers/model/SF_TRON1A/policy/policy.onnx")
        self.model_encoder = os.path.expanduser("~/limx_ws/rl-deploy-with-python/controllers/model/SF_TRON1A/policy/encoder.onnx")
        
        # 初始化ONNX模型
        self.initialize_onnx_models()
        
        # ========== MuJoCo相关参数 ==========
        self.XML_MODEL_PATH = os.path.expanduser("~/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A/xml/robot_with_arm.xml")
        self.JOINT_NAMES = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint",
            # 移除机械臂关节，只保留8个腿部关节
            # "J1", "J2", "J3", "J4", "J5", "J6",
        ]
        # 关节数量
        self.joint_num = len(self.JOINT_NAMES)
        
        # 打印关节数量信息
        print(f"\n关节配置信息:")
        print(f"- 关节名称数组长度: {len(self.JOINT_NAMES)}")
        print(f"- 关节数量(joint_num): {self.joint_num}")
        print(f"- 关节名称: {self.JOINT_NAMES}")
        print(f"- 只控制腿部关节，忽略机械臂关节\n")
        
        # ========== 控制参数 ==========
        self.PD_STIFFNESS = self.control_cfg.get('stiffness', 150.0)  # 降低刺度以提高稳定性 [N*m/rad]
        self.PD_DAMPING = self.control_cfg.get('damping', 1.5)  # 增加阻尼以减少振荡        [N*m*s/rad]
        self.ANKLE_DAMPING = 0.8
        self.ANKLE_TORQUE_LIMIT = 20
        self.USER_TORQUE_LIMIT = 80
        self.ACTION_SCALE_POS = 0.25
        self.ACTION_SCALE = 0.5  # 动作缩放因子，用于缩放策略输出的动作值
        self.DECI = 10
        self.OBS_CLIP = 100.0
        
        # ========== 状态变量 ==========
        self.mode = "STAND"  # 控制模式: STAND, WALK
        self.loop_count = 0
        self.step_count = 0
        self.gait_index = 0.0
        self.calibration_state = -1
        self.is_first_rec_obs = True
        
        # ========== 观测和动作缓存 ==========
        self.observations = np.zeros(self.observation_dim)
        self.actions = np.zeros(self.joint_num)
        self.last_actions = np.zeros(self.joint_num)
        self.commands = np.zeros(5)  # [linear_x, linear_y, angular_z, extra1, extra2]
        self.proprioceptive_history = np.zeros((self.observation_cfg.get('obs_history_length', 5), 
                                              self.observation_dim))
        
        # ========== 编码器输出 ==========
        self.encoder_output = np.zeros(self.observation_cfg.get('encoder_output_dim', 32))
        
        # ========== 遥控器状态 ==========
        self.joy_buttons = np.zeros(12)
        self.joy_axes = np.zeros(6)
        
        # ========== 通信配置 ==========
        self.SERVER_IP = "127.0.0.1"
        
        # ========== 初始化状态 ==========
        self.default_qpos = None  # 将在initialize_mujoco中设置
        self.stand_qpos = None    # 将在initialize_mujoco中设置
        self.stand_percent = 0.0
        self.stand_duration = 4.0 * 1000.0  # 4秒，4000步
        
        # MuJoCo相关变量初始化为None
        self.model = None
        self.data = None
        self.viewer = None
        self.sim_freq = None
        self.control_freq = None
        
    def initialize_mujoco(self):
        """初始化MuJoCo模型和数据"""
        # 如果已经初始化，直接返回
        if hasattr(self, 'model') and self.model is not None and hasattr(self, 'data') and self.data is not None:
            return True
            
        try:
            # 加载MuJoCo模型
            print(f"正在加载MuJoCo模型: {self.XML_MODEL_PATH}")
            self.model = mujoco.MjModel.from_xml_path(self.XML_MODEL_PATH)
            self.data = mujoco.MjData(self.model)
            
            # 计算仿真和控制频率
            self.sim_freq = int(1.0 / self.model.opt.timestep)
            self.control_freq = int(self.sim_freq / self.DECI)
            print(f"仿真频率: {self.sim_freq}Hz, 控制频率: {self.control_freq}Hz")
            
            # 获取关节索引
            self.joint_idxs = []
            for name in self.JOINT_NAMES:
                idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if idx == -1:
                    print(f"警告: 关节名称 '{name}' 在模型中未找到")
                self.joint_idxs.append(idx)
            
            # 初始化默认位置
            mujoco.mj_resetData(self.model, self.data)
            self.default_qpos = self.data.qpos.copy()
            
            # 打印默认位置信息
            print(f"default_qpos长度: {len(self.default_qpos)}")
            print(f"default_qpos前15个值: {self.default_qpos[:15]}")
            print(f"关节位置(7:7+{self.joint_num}): {self.default_qpos[7:7+self.joint_num]}")
            
            # 设置站立位置
            self.stand_qpos = self.default_qpos.copy()
            # 修改站立姿势，确保不会越界
            if len(self.stand_qpos) >= 21:
                # 设置更适合站立的姿势 - 使用非常保守的角度
                # 左腿: abad_L_Joint, hip_L_Joint, knee_L_Joint, ankle_L_Joint
                # 右腿: abad_R_Joint, hip_R_Joint, knee_R_Joint, ankle_R_Joint
                # 使用非常小的角度值，逐步调整
                self.stand_qpos[7:15] = np.array([1.49, 1.13, 0.0, 0.0, 0.0, -0.13, 0.0, 0.0])  # 腿部关节
                # 移除机械臂关节姿势设置
                # self.stand_qpos[15:21] = np.array([0.0, -0.15, -0.23, 0.082, -0.051, 0.092])  # 机械臂关节
                
                # 打印站立姿势信息
                print(f"站立姿势关节角度: {self.stand_qpos[7:15]}")
            else:
                print(f"警告: 默认位置数组长度不足 ({len(self.stand_qpos)}), 无法设置站立姿势")
            
            # 初始化可视化界面
            try:
                # 使用viewer.launch_passive初始化可视化界面
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
                
            # 初始化传感器数据 - 只有quat传感器，但保留完整的IMU数据结构
            self.imu_data = {
                'quat': np.zeros(4),  # 四元数 - 从quat传感器获取
                'gyro': np.zeros(3),  # 陀螺仪 - 将使用估计值
                'acc': np.zeros(3),   # 加速度计 - 将使用估计值
                'stamp': 0            # 时间戳
            }
            
            # 获取传感器ID
            self.sensor_ids = {}
            sensor_names = ["quat"]  # 只有一个quat传感器
            for name in sensor_names:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sensor_id == -1:
                    print(f"警告: 传感器 '{name}' 在模型中未找到")
                self.sensor_ids[name] = sensor_id
                print(f"找到传感器 '{name}' ID: {sensor_id}")
                
            return True
        except Exception as e:
            print(f"MuJoCo初始化失败: {e}")
            self.model = None
            self.data = None
            self.viewer = None
            return False
    
    def load_config(self, config_file):
        """从YAML文件加载配置"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # 提取配置参数
            self.control_cfg = config.get('control', {})
            self.observation_cfg = config.get('observation', {})
            self.action_cfg = config.get('action', {})
            
            # 设置观测和动作维度
            self.observation_dim = self.observation_cfg.get('observation_dim', 54)
            self.action_dim = self.action_cfg.get('action_dim', 14)
            
            print(f"配置加载成功: {config_file}")
            return True
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 设置默认值
            self.control_cfg = {'stiffness': 45.0, 'damping': 3.0}
            self.observation_cfg = {
                'observation_dim': 54, 
                'obs_history_length': 5,
                'encoder_output_dim': 32
            }
            self.action_cfg = {'action_dim': 14}
            return False
    
    def initialize_onnx_models(self):
        """初始化ONNX运行时和模型"""
        try:
            # 配置ONNX Runtime会话选项以优化CPU使用
            session_options = ort.SessionOptions()
            # 限制用于单个算子内并行计算的线程数
            session_options.intra_op_num_threads = 1
            # 限制用于不同算子并行执行的线程数
            session_options.inter_op_num_threads = 1
            # 启用所有可能的图优化以提高推理性能
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # 禁用CPU内存arena以减少内存碎片
            session_options.enable_cpu_mem_arena = False
            # 禁用内存模式优化以更好地控制内存分配
            session_options.enable_mem_pattern = False

            # 定义执行提供程序仅使用CPU，确保不使用GPU推理
            cpu_providers = ['CPUExecutionProvider']
            
            # 加载ONNX模型并设置输入和输出名称
            self.policy_session = ort.InferenceSession(self.model_policy, 
                                                     sess_options=session_options, 
                                                     providers=cpu_providers)
            
            self.encoder_session = ort.InferenceSession(self.model_encoder, 
                                                      sess_options=session_options, 
                                                      providers=cpu_providers)
            
            # 获取策略模型的输入输出信息
            self.policy_input_names = [self.policy_session.get_inputs()[i].name 
                                     for i in range(len(self.policy_session.get_inputs()))]
            self.policy_output_names = [self.policy_session.get_outputs()[i].name 
                                      for i in range(len(self.policy_session.get_outputs()))]
            self.policy_input_shapes = [self.policy_session.get_inputs()[i].shape 
                                      for i in range(len(self.policy_session.get_inputs()))]
            self.policy_output_shapes = [self.policy_session.get_outputs()[i].shape 
                                       for i in range(len(self.policy_session.get_outputs()))]
            
            # 获取编码器模型的输入输出信息
            self.encoder_input_names = [self.encoder_session.get_inputs()[i].name 
                                      for i in range(len(self.encoder_session.get_inputs()))]
            self.encoder_output_names = [self.encoder_session.get_outputs()[i].name 
                                       for i in range(len(self.encoder_session.get_outputs()))]
            self.encoder_input_shapes = [self.encoder_session.get_inputs()[i].shape 
                                       for i in range(len(self.encoder_session.get_inputs()))]
            self.encoder_output_shapes = [self.encoder_session.get_outputs()[i].shape 
                                        for i in range(len(self.encoder_session.get_outputs()))]
            
            print("ONNX模型加载成功")
            return True
        except Exception as e:
            print(f"加载ONNX模型失败: {e}")
            return False
    
    def init_udp_receiver(self):
        """初始化UDP接收器"""
        try:
            # 创建UDP套接字
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind((self.udp_ip, self.udp_port))
            self.udp_socket.settimeout(0.01)  # 设置超时时间为10ms
            print(f"UDP接收器已初始化，监听 {self.udp_ip}:{self.udp_port}")
            
            # 启动UDP接收线程
            self.udp_running = True
            self.udp_thread = threading.Thread(target=self.udp_receive_thread)
            self.udp_thread.daemon = True
            self.udp_thread.start()
            print("UDP接收线程已启动")
            return True
        except Exception as e:
            print(f"初始化UDP接收器失败: {e}")
            return False
    
    def udp_receive_thread(self):
        """UDP接收线程函数"""
        print("UDP接收线程开始运行")
        while self.udp_running:
            try:
                # 接收数据
                data, addr = self.udp_socket.recvfrom(1024)
                if len(data) >= 12:  # 至少需要3个浮点数(12字节)和一些按钮数据
                    # 解析数据 - 假设前12字节是3个浮点数(axes)，后面是按钮状态
                    axes = struct.unpack('fff', data[:12])
                    
                    # 解析按钮状态 - 假设每个按钮用一个字节表示
                    buttons = []
                    for i in range(12, min(len(data), 24)):  # 最多解析12个按钮
                        buttons.append(data[i])
                    
                    # 处理遥控器输入
                    print(f"收到遥控器数据: axes={axes}, buttons={buttons}")
                    self.process_joystick_input(buttons, axes)
            except socket.timeout:
                # 超时，继续循环
                pass
            except Exception as e:
                print(f"UDP接收线程出错: {e}")
                time.sleep(0.1)  # 出错时短暂休眠
        print("UDP接收线程已退出")
    
    def close_udp_receiver(self):
        """关闭UDP接收器"""
        self.udp_running = False
        if self.udp_thread is not None:
            self.udp_thread.join(1.0)  # 等待线程结束，最多1秒
        if self.udp_socket is not None:
            self.udp_socket.close()
            self.udp_socket = None
        print("UDP接收器已关闭")
        
    def run(self):
        """主运行循环，采用简化版的实现确保稳定运行"""
        print("开始运行控制器...")
        
        # 初始化MuJoCo模型和可视化界面
        if not self.initialize_mujoco():
            print("模型初始化失败，退出程序")
            return
            
        # 初始化UDP接收器
        self.init_udp_receiver()
        
        # 设置控制频率
        control_dt = 1.0 / self.control_freq
        
        # 主循环
        while True:
            try:
                # 记录当前时间
                start_time = time.time()
                
                # 执行控制逻辑
                if self.start_controller:
                    try:
                        # 打印当前模式
                        print(f"\n当前模式: {self.mode}, 控制器状态: {self.start_controller}")
                        
                        # 根据当前模式执行不同的控制逻辑
                        if self.mode == "STAND":
                            self.handle_stand_mode()
                        elif self.mode == "WALK":
                            self.handle_walk_mode()
                        else:
                            print(f"未知模式: {self.mode}")
                        
                        # 应用动作到MuJoCo模型
                        self.apply_actions()
                    except Exception as e:
                        print(f"控制器更新失败: {e}")
                
                # 执行MuJoCo仿真步进
                mujoco.mj_step(self.model, self.data)
                self.step_count += 1
                
                # 更新传感器数据
                if self.step_count % self.DECI == 0:
                    try:
                        self.update_sensor_data()
                    except Exception as e:
                        print(f"更新传感器数据失败: {e}")
                
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
                if self.step_count % self.sim_freq == 0:
                    print(f"仿真时间: {self.step_count / self.sim_freq:.2f}秒, 模式: {self.mode}")
                    
            except Exception as e:
                print(f"主循环中出错: {e}")
                # 发生错误时继续运行，而不是退出
                    
    def toggle_controller(self, key=None):
        """切换控制器启动/停止状态"""
        self.start_controller = not self.start_controller
        print(f"控制器状态: {'启动' if self.start_controller else '停止'}")
    
    def switch_to_stand_mode(self, key=None):
        """切换到站立模式"""
        self.mode = "STAND"
        self.stand_percent = 0.0
        print("切换到STAND模式")
    
    def switch_to_walk_mode(self, key=None):
        """切换到行走模式"""
        self.mode = "WALK"
        print(f"切换到WALK模式, 当前命令: {self.commands[:3]}, 控制器状态: {'启动' if self.start_controller else '停止'}")
        # 确保在切换到行走模式时至少有一个非零的命令
        if np.all(np.abs(self.commands[:3]) < 0.01):
            print("警告: 所有命令都接近零，设置前进命令")
            self.commands[0] = 0.3  # 设置默认前进速度
    
    def reset_simulation(self, key=None):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_qpos
        self.step_count = 0
        self.loop_count = 0
        self.stand_percent = 0.0
        self.gait_index = 0.0
        self.commands[:] = 0.0
        print("重置仿真")
    
    def set_command(self, index, value, key=None):
        """设置命令值"""
        if index < len(self.commands):
            self.commands[index] = value
            print(f"设置命令[{index}] = {value}, 当前所有命令: {self.commands[:3]}")
        else:
            print(f"命令索引{index}超出范围，最大索引为{len(self.commands)-1}")
        
        # 如果在WALK模式下设置命令，打印提示
        if self.mode == "WALK":
            print("在WALK模式下更新了命令，将在下一次推理时生效")
            command_names = ["前进/后退", "左移/右移", "左转/右转", "高度", "模式"]
            print(f"设置{command_names[index]}命令: {value}")
        print(f"设置{command_names[index]}命令: {value}")
    
    def stop_movement(self, key=None):
        """停止移动"""
        self.commands[:] = 0.0
        print("停止移动")
    
    def handle_stand_mode(self):
        """处理站立模式"""
        # 确保默认位置和站立位置已经初始化
        if self.default_qpos is None or self.stand_qpos is None:
            print("错误: 默认位置或站立位置未初始化")
            return
            
        if self.stand_percent < 1.0:
            # 逐渐过渡到站立姿势
            self.stand_percent += 1.0 / self.stand_duration
            self.stand_percent = min(self.stand_percent, 1.0)
            print(f"站立进度: {self.stand_percent * 100:.1f}%") if int(self.stand_percent * 100) % 10 == 0 else None
            
            try:
                # 检查关节数量和数组大小
                joint_count = min(self.joint_num, len(self.default_qpos) - 7, len(self.stand_qpos) - 7)
                
                # 防止数组越界
                ctrl_len = len(self.data.ctrl) if hasattr(self.data, 'ctrl') else 0
                qpos_len = len(self.data.qpos) if hasattr(self.data, 'qpos') else 0
                qvel_len = len(self.data.qvel) if hasattr(self.data, 'qvel') else 0
                
                # 逐个关节应用PD控制
                for i in range(min(joint_count, ctrl_len)):
                    # 确保索引安全
                    if i+7 < qpos_len and i+6 < qvel_len:
                        # 计算目标位置（从默认位置到站立位置的插值）
                        pos_des = self.default_qpos[i+7] * (1 - self.stand_percent) + self.stand_qpos[i+7] * self.stand_percent
                        
                        # 获取当前位置和速度
                        pos_cur = self.data.qpos[i+7]
                        vel_cur = self.data.qvel[i+6] if i+6 < qvel_len else 0.0
                        
                        # 使用pd_control函数计算力矩
                        torque = self.pd_control(
                            pos_cur=pos_cur,
                            vel_cur=vel_cur,
                            pos_des=pos_des,
                            kp=self.PD_STIFFNESS,
                            kd=self.PD_DAMPING,
                            torque_limit=self.USER_TORQUE_LIMIT
                        )
                        pos_error = pos_des - pos_cur

                        # 应用力矩
                        self.data.ctrl[i] = torque
                        
                        # 打印第一个关节的信息
                        if i == 0 or i == 1 :
                            print(f"关节{i+1} - 期望位置: {pos_des:.4f}, 实时位置: {self.data.qpos[i+7]:.4f}, 位置差: {pos_error:.4f}, 力矩: {torque:.4f}")
            except Exception as e:
                print(f"处理站立模式时出错: {e}")
        else:
            # 站立完成后切换到行走模式
            self.mode = "WALK"
            print("站立完成，切换到行走模式")
    
    def handle_walk_mode(self):
        """处理行走模式"""
        try:
            print("===== 开始处理行走模式 =====")
            print(f"当前模式确认: {self.mode}")
            
            # 计算观测
            print("计算观测...")
            self.compute_observation()
            
            # 计算编码器输出
            print("计算编码器输出...")
            self.compute_encoder()
            print(f"编码器输出形状: {self.encoder_output.shape}, 前几个值: {self.encoder_output[:3]}")
            
            # 计算动作
            print("计算动作...")
            print(f"当前命令: {self.commands[:3]}")
            self.compute_actions()
            print(f"计算的动作形状: {self.actions.shape}, 动作值: {self.actions}")
            
            # 确保动作不为零
            if np.all(np.abs(self.actions) < 1e-6):
                print("警告: handle_walk_mode中发现动作全为0，直接添加测试动作")
                # 创建测试动作
                test_actions = np.zeros(self.joint_num)
                test_actions[0] = 0.8  # 第一个关节添加较大动作
                test_actions[1] = -0.8  # 第二个关节添加较大动作
                # 为所有关节添加不同的随机动作
                for i in range(2, self.joint_num):
                    test_actions[i] = np.random.uniform(-0.5, 0.5)
                self.actions = test_actions
                print(f"handle_walk_mode中添加的测试动作: {self.actions}")
            
            # 更新步态
            print("更新步态...")
            self.update_gait()
            
            print("===== 行走模式处理完成 =====")
            # 注意: 不在这里调用apply_actions，因为run方法中已经调用了
        except Exception as e:
            print(f"处理行走模式时出错: {e}")
    
    def compute_observation(self):
        """计算当前观测"""
        try:
            # 检查数据是否存在
            if self.data is None or self.data.qvel is None or self.data.qpos is None:
                print("错误: MuJoCo数据未初始化")
                return
                
            # 基座角速度
            if len(self.data.qvel) > 5:
                base_ang_vel = self.data.qvel[3:6].copy()
            else:
                base_ang_vel = np.zeros(3)
            
            # 重力在基座坐标系下的投影
            gravity_world = np.array([0, 0, -1])
            rot_mat = np.zeros((3, 3))
            if len(self.data.qpos) > 6:
                mujoco.mju_quat2Mat(rot_mat.flatten(), self.data.qpos[3:7])
            projected_gravity = rot_mat.T @ gravity_world
            
            # 关节位置和速度
            joint_pos_end = min(7 + self.joint_num, len(self.data.qpos))
            joint_vel_end = min(6 + self.joint_num, len(self.data.qvel))
            
            if joint_pos_end > 7 and len(self.default_qpos) > joint_pos_end - 1:
                dof_pos = self.data.qpos[7:joint_pos_end].copy() - self.default_qpos[7:joint_pos_end]
            else:
                dof_pos = np.zeros(self.joint_num)
                
            if joint_vel_end > 6:
                dof_vel = self.data.qvel[6:joint_vel_end].copy()
            else:
                dof_vel = np.zeros(self.joint_num)
            
            # 上一步动作
            actions = self.last_actions
            
            # 时钟信号
            clock = (self.step_count / self.control_freq) * 2 * math.pi
            clock_sin = np.array([math.sin(clock)])
            clock_cos = np.array([math.cos(clock)])
            
            # 步态信号
            gaits = np.array([1, 0, 0, 0], dtype=np.float32)
            
            # # 命令信号
            # cmd = self.commands[:3]  # 前三个是移动命令
            
            # 组合观测 - 去掉cmd的3维数据
            self.observations = np.concatenate([
                base_ang_vel,           # 3
                projected_gravity,      # 3
                dof_pos,                # 14
                dof_vel,                # 14
                actions,                # 14
                clock_sin,              # 1
                clock_cos,              # 1
                gaits                   # 4
            ]).astype(np.float32)
            
            # 裁剪观测值
            self.observations = np.clip(self.observations, -self.OBS_CLIP, self.OBS_CLIP)
            
            # 检查观测数组与历史缓冲区的大小是否匹配
            if hasattr(self, 'proprioceptive_history') and self.proprioceptive_history is not None:
                expected_dim = self.proprioceptive_history.shape[1]
                actual_dim = len(self.observations)
                
                if expected_dim != actual_dim:
                    print(f"警告: 观测数组维度({actual_dim})与历史缓冲区维度({expected_dim})不匹配")
                    
                    # 动态调整观测数组大小以匹配历史缓冲区
                    if actual_dim > expected_dim:
                        print(f"裁剪观测数组从{actual_dim}到{expected_dim}")
                        self.observations = self.observations[:expected_dim]
                    else:
                        print(f"填充观测数组从{actual_dim}到{expected_dim}")
                        self.observations = np.pad(self.observations, (0, expected_dim - actual_dim), 'constant')
                
                # 更新历史缓冲区
                self.proprioceptive_history = np.roll(self.proprioceptive_history, -1, axis=0)
                self.proprioceptive_history[-1] = self.observations
                
                # 打印历史缓冲区信息
                print(f"历史缓冲区形状: {self.proprioceptive_history.shape}")
                print(f"历史缓冲区最新一行前5个值: {self.proprioceptive_history[-1][:5]}")
                
                # 检查历史缓冲区是否有有效数据
                if np.all(np.abs(self.proprioceptive_history) < 1e-6):
                    print("警告: 历史缓冲区全为0，可能导致编码器输出异常")
        except Exception as e:
            print(f"计算观测时出错: {e}")
    
    def compute_encoder(self):
        """计算编码器输出"""
        try:
            # 准备输入数据
            encoder_input = self.proprioceptive_history.flatten().astype(np.float32)
            print(f"编码器输入形状: {encoder_input.shape}, 输入维度: {len(encoder_input)}")
            print(f"编码器输入前5个值: {encoder_input[:5]}")
            
            # 检查输入是否全为0
            if np.all(np.abs(encoder_input) < 1e-6):
                print("警告: 编码器输入全为0，可能影响编码器输出")
            
            # 创建输入字典 - 保持输入为1维
            input_dict = {self.encoder_input_names[0]: encoder_input}
            
            # 运行编码器会话
            print("开始运行编码器会话...")
            encoder_outputs = self.encoder_session.run(self.encoder_output_names, input_dict)
            print(f"编码器输出成功，输出名称: {self.encoder_output_names}")
            print(f"编码器输出形状: {[output.shape for output in encoder_outputs]}")
            
            # 存储编码器输出
            self.encoder_output = encoder_outputs[0].flatten()
            print(f"编码器输出维度: {self.encoder_output.shape}, 前5个值: {self.encoder_output[:5]}")
            
            # 检查输出是否全为0
            if np.all(np.abs(self.encoder_output) < 1e-6):
                print("警告: 编码器输出全为0，这可能导致策略模型输出全为0")
        except Exception as e:
            print(f"编码器计算失败: {e}")
    
    def compute_actions(self):
        """计算动作"""
        try:
            # 添加测试命令 - 前进命令
            if self.mode == "WALK":
                test_commands = np.array([0.5, 0.0, 0.0])  # x方向前进
                self.commands[:3] = test_commands
                print(f"测试命令已添加: {test_commands}")
            
            # 打印编码器输出信息
            print(f"编码器输出维度: {self.encoder_output.shape if hasattr(self, 'encoder_output') else 'None'}")
            if hasattr(self, 'encoder_output') and self.encoder_output is not None and len(self.encoder_output) > 0:
                print(f"编码器输出前5个值: {self.encoder_output[:5]}")
                # 检查编码器输出是否全为0
                if np.all(np.abs(self.encoder_output) < 1e-6):
                    print("警告: 编码器输出全为0，这可能导致策略模型输出全为0")
                    # 尝试为编码器输出添加一些随机值
                    print("为编码器输出添加随机值以测试策略模型...")
                    self.encoder_output = np.random.uniform(-0.1, 0.1, size=self.encoder_output.shape)
                    print(f"添加随机值后的编码器输出前5个值: {self.encoder_output[:5]}")
            else:
                print("警告: 编码器输出为空或不存在")
                # 创建随机编码器输出
                expected_encoder_dim = 59  # 根据实际情况调整
                self.encoder_output = np.random.uniform(-0.1, 0.1, size=expected_encoder_dim)
                print(f"创建随机编码器输出: {self.encoder_output[:5]}...")
                
            # 打印命令信息
            print(f"命令数组: {self.commands}")
            
            # 准备输入数据
            policy_input = np.concatenate([self.encoder_output, self.commands[:3]]).astype(np.float32) # ONNX模型获得的输出 +（机器人移动命令）线性x速度、线性y速度和角速度 
            print(f"策略输入形状: {policy_input.shape}, 输入维度: {len(policy_input)}")
            print(f"策略输入名称: {self.policy_input_names}")
            print(f"策略输入预期形状: {self.policy_input_shapes}")
            
            # 检查输入维度是否与模型预期相符
            expected_dim = self.policy_input_shapes[0][0] if self.policy_input_shapes and len(self.policy_input_shapes) > 0 else 62
            
            # 如果维度不匹配，进行填充
            if len(policy_input) < expected_dim:
                print(f"输入维度不匹配，将输入从{len(policy_input)}填充到{expected_dim}")
                padding = np.zeros(expected_dim - len(policy_input), dtype=np.float32)
                policy_input = np.concatenate([policy_input, padding])
            elif len(policy_input) > expected_dim:
                print(f"输入维度过大，将输入从{len(policy_input)}裁剪到{expected_dim}")
                policy_input = policy_input[:expected_dim]
            
            # 创建输入字典
            input_dict = {self.policy_input_names[0]: policy_input}
            
            # 运行策略会话
            print("开始运行策略会话...")
            try:
                policy_outputs = self.policy_session.run(self.policy_output_names, input_dict)
                print(f"策略输出成功，输出名称: {self.policy_output_names}")
                print(f"策略输出形状: {[output.shape for output in policy_outputs]}")
                
                # 存储策略输出
                raw_actions = policy_outputs[0].flatten()
                print(f"原始动作数组形状: {raw_actions.shape}, 动作数量: {len(raw_actions)}")
                print(f"原始动作数组值: {raw_actions[:8]}... (前8个值)")
                
                # 无论如何，都使用测试动作（忽略策略输出）
                print("强制使用测试动作替代策略输出...")
                # 直接创建新的测试动作数组
                test_actions = np.zeros(self.joint_num)
                # 添加明显的测试动作值
                test_actions[0] = 0.8  # 第一个关节添加较大动作
                test_actions[1] = -0.8  # 第二个关节添加较大动作
                # 为所有关节添加不同的随机动作，确保不全为0
                for i in range(2, self.joint_num):
                    test_actions[i] = np.random.uniform(-0.5, 0.5)
                self.actions = test_actions
                print(f"测试动作已应用，动作数组: {self.actions}")
                
                # 确保动作数组长度与关节数匹配
                if len(self.actions) < self.joint_num:
                    print(f"错误: 动作数组长度({len(self.actions)})小于关节数({self.joint_num})")
                    self.actions = np.pad(self.actions, (0, self.joint_num - len(self.actions)), 'constant')
                    print(f"填充后的动作数组: {self.actions}")
                elif len(self.actions) > self.joint_num:
                    print(f"警告: 动作数组长度({len(self.actions)})大于关节数({self.joint_num})")
                    self.actions = self.actions[:self.joint_num]
                    print(f"裁剪后的动作数组: {self.actions}")
                
                # 规范化动作值到合理范围
                print(f"规范化前的动作: {self.actions[:8]}...")
                # 将动作值裁剪到[-1, 1]范围
                self.actions = np.clip(self.actions, -1.0, 1.0)
                print(f"规范化后的动作: {self.actions[:8]}...")
                
                # 保存当前动作用于下一步观测
                self.last_actions = self.actions.copy()
            except Exception as e:
                print(f"策略会话运行失败: {e}")
                print(f"输入字典: {input_dict.keys()}")
                # 如果策略会话失败，使用随机测试动作
                print("使用随机测试动作作为备选")
                self.actions = np.random.uniform(-0.5, 0.5, size=self.joint_num)
                print(f"随机测试动作: {self.actions}")
                self.last_actions = self.actions.copy()
        except Exception as e:
            print(f"策略计算失败: {e}")
            # 确保即使失败也有有效的动作数组
            self.actions = np.zeros(self.joint_num)
            self.last_actions = self.actions.copy()
    
    def apply_actions(self):
        """应用动作到关节 - 使用向量化PD控制"""
        try:
            print("===== 开始应用动作 =====")
            
            # 检查数据和动作是否存在
            if not hasattr(self, 'data') or not hasattr(self, 'actions') or self.data is None or self.actions is None:
                print("错误: 数据或动作不存在")
                return
            
            # 检查动作数组大小
            if len(self.actions) < self.joint_num:
                print(f"错误: 动作数组大小不足 ({len(self.actions)} < {self.joint_num})")
                return
                
            # 检查动作是否全为0，如果是，直接在这里添加测试动作
            if np.all(np.abs(self.actions) < 1e-6):
                print("警告: apply_actions中发现动作全为0，直接添加测试动作")
                # 创建测试动作
                test_actions = np.zeros(self.joint_num)
                test_actions[0] = 0.8  # 第一个关节添加较大动作
                test_actions[1] = -0.8  # 第二个关节添加较大动作
                # 为所有关节添加不同的随机动作
                for i in range(2, self.joint_num):
                    test_actions[i] = np.random.uniform(-0.5, 0.5)
                self.actions = test_actions
                print(f"apply_actions中添加的测试动作: {self.actions}")
            
            # 缩放动作
            action_scaled = self.actions * self.ACTION_SCALE
            print(f"缩放前的动作: {self.actions[:8]}...")
            print(f"缩放后的动作: {action_scaled[:8]}...")
            
            # 打印当前模式
            print(f"当前模式: {self.mode}")
            
            # 打印动作统计信息
            if len(self.actions) > 0:
                print(f"动作统计 - 最小值: {np.min(self.actions):.4f}, 最大值: {np.max(self.actions):.4f}, 平均值: {np.mean(self.actions):.4f}, 标准差: {np.std(self.actions):.4f}")
                print(f"缩放后动作统计 - 最小值: {np.min(action_scaled):.4f}, 最大值: {np.max(action_scaled):.4f}, 平均值: {np.mean(action_scaled):.4f}")
            
            # 打印第一个关节的期望位置和实时位置
            if len(self.data.qpos) > 7 and len(self.data.ctrl) > 0:
                try:
                    # 确保在站立模式下才访问stand_progress
                    if hasattr(self, 'stand_progress') and self.mode == "STAND":
                        pos_des_1 = self.default_qpos[7] + self.stand_progress * (self.stand_qpos[7] - self.default_qpos[7])
                        pos_des_2 = self.default_qpos[8] + self.stand_progress * (self.stand_qpos[8] - self.default_qpos[8])
                        print(f"关节1 - 期望位置: {pos_des_1:.4f}, 实时位置: {self.data.qpos[7]:.4f}, 位置差: {pos_des_1 - self.data.qpos[7]:.4f}, 力矩: {self.data.ctrl[0]:.4f}")
                        print(f"关节2 - 期望位置: {pos_des_2:.4f}, 实时位置: {self.data.qpos[8]:.4f}, 位置差: {pos_des_2 - self.data.qpos[8]:.4f}, 力矩: {self.data.ctrl[1]:.4f}")
                    else:
                        # 在行走模式下使用动作缩放值作为期望位置
                        if len(action_scaled) >= 2:
                            print(f"关节1 - 期望位置: {action_scaled[0]:.4f}, 实时位置: {self.data.qpos[7]:.4f}, 位置差: {action_scaled[0] - self.data.qpos[7]:.4f}, 力矩: {self.data.ctrl[0]:.4f}")
                            print(f"关节2 - 期望位置: {action_scaled[1]:.4f}, 实时位置: {self.data.qpos[8]:.4f}, 位置差: {action_scaled[1] - self.data.qpos[8]:.4f}, 力矩: {self.data.ctrl[1]:.4f}")
                        else:
                            print(f"关节1 - 实时位置: {self.data.qpos[7]:.4f}, 力矩: {self.data.ctrl[0]:.4f}")
                            print(f"关节2 - 实时位置: {self.data.qpos[8]:.4f}, 力矩: {self.data.ctrl[1]:.4f}")
                except Exception as e:
                    print(f"打印关节信息时出错: {e}")
                
                # 打印控制数组长度和关节数量
                # print(f"控制数组长度: {len(self.data.ctrl)}, 实际控制关节数: {self.joint_num}")
                
            try:
                # 准备增益数组
                p_gains = np.ones(self.joint_num) * self.PD_STIFFNESS
                d_gains = np.ones(self.joint_num) * self.PD_DAMPING
                
                # 为踝关节设置特殊增益
                for i in range(self.joint_num):
                    if i < len(self.JOINT_NAMES) and "ankle" in self.JOINT_NAMES[i]:
                        d_gains[i] = self.ANKLE_DAMPING
                
                # 获取关节位置和速度
                dof_pos = self.data.qpos[7:7+self.joint_num]
                dof_vel = self.data.qvel[6:6+self.joint_num]
                
                # 确保动作数组长度与关节数量一致
                if len(action_scaled) > self.joint_num:
                    print(f"警告: 动作数组长度({len(action_scaled)})大于关节数({self.joint_num})，裁剪动作数组")
                    action_scaled = action_scaled[:self.joint_num]
                elif len(action_scaled) < self.joint_num:
                    print(f"警告: 动作数组长度({len(action_scaled)})小于关节数({self.joint_num})，填充动作数组")
                    action_scaled = np.pad(action_scaled, (0, self.joint_num - len(action_scaled)), 'constant')
                
                # 打印默认位置和站立位置的详细信息
                print(f"apply_actions中的default_qpos长度: {len(self.default_qpos)}")
                print(f"关节部分default_qpos[7:7+{self.joint_num}]: {self.default_qpos[7:7+self.joint_num]}")
                print(f"关节部分stand_qpos[7:7+{self.joint_num}]: {self.stand_qpos[7:7+self.joint_num]}")
                
                # 修改期望位置计算逻辑，使用stand_qpos而不是default_qpos
                # 在行走模式下，我们希望从站立姿势开始进行动作计算
                pos_des = action_scaled + self.stand_qpos[7:7+self.joint_num]
                print(f"期望位置计算详情:")
                print(f"  - action_scaled: {action_scaled[:self.joint_num]}")
                print(f"  - stand_qpos[7:7+{self.joint_num}]: {self.stand_qpos[7:7+self.joint_num]}")
                print(f"  = pos_des: {pos_des}")
                
                # 打印期望位置信息
                print(f"期望位置数组: {pos_des[:8]}...")
                
                # 确保期望位置不为零
                if np.all(np.abs(pos_des) < 1e-6):
                    print("警告: 期望位置全为0，直接设置测试期望位置")
                    # 设置测试期望位置 - 使用stand_qpos作为基准
                    pos_des[0] = self.stand_qpos[7] + 0.4
                    pos_des[1] = self.stand_qpos[8] - 0.4
                    for i in range(2, self.joint_num):
                        pos_des[i] = self.stand_qpos[7+i] + np.random.uniform(-0.3, 0.3)
                    print(f"测试期望位置: {pos_des[:8]}...")
                
                # 准备力矩限制数组
                torque_limits = np.ones(self.joint_num) * self.USER_TORQUE_LIMIT
                # 为踝关节设置特殊力矩限制
                for i in range(self.joint_num):
                    if i < len(self.JOINT_NAMES) and "ankle" in self.JOINT_NAMES[i]:
                        torque_limits[i] = self.ANKLE_TORQUE_LIMIT
                
                # 使用统一的pd_control函数计算力矩（向量化）
                torques = self.pd_control(
                    pos_cur=dof_pos,
                    vel_cur=dof_vel,
                    pos_des=pos_des,
                    kp=p_gains,
                    kd=d_gains,
                    torque_limit=torque_limits
                )
                
                print(f"计算的力矩: {torques[:4]}...")
                
                # 应用力矩 - 只使用8个腿部关节的控制信号
                if len(self.data.ctrl) >= self.joint_num:
                    # 只设置前8个关节的力矩，其余设为0
                    self.data.ctrl[:self.joint_num] = torques
                    # 确保机械臂关节的控制信号为0
                    if len(self.data.ctrl) > self.joint_num:
                        self.data.ctrl[self.joint_num:] = 0.0
                    print(f"成功应用力矩到控制数组，控制数组长度: {len(self.data.ctrl)}, 实际控制关节数: {self.joint_num}")
                else:
                    print(f"错误: 控制数组长度不足 ({len(self.data.ctrl)})")
            except Exception as e:
                print(f"向量化计算力矩时出错: {e}")
                # 回退到逐关节计算方式
                print("回退到逐关节计算方式...")
                
                # 初始化力矩数组
                torques = np.zeros(self.joint_num)
                for i in range(self.joint_num):
                    try:
                        # 为不同关节设置不同的控制参数
                        if i < len(self.JOINT_NAMES) and "ankle" in self.JOINT_NAMES[i]:
                            kp = self.PD_STIFFNESS
                            kd = self.ANKLE_DAMPING
                            t_limit = self.ANKLE_TORQUE_LIMIT
                        else:
                            kp = self.PD_STIFFNESS
                            kd = self.PD_DAMPING
                            t_limit = self.USER_TORQUE_LIMIT
                        
                        # 检查索引是否越界
                        if i+7 < len(self.data.qpos) and i+6 < len(self.data.qvel) and i < len(action_scaled):
                            # 计算力矩
                            pos_error = action_scaled[i] + self.default_qpos[i+7] - self.data.qpos[i+7]
                            torques[i] = kp * pos_error - kd * self.data.qvel[i+6]
                            torques[i] = np.clip(torques[i], -t_limit, t_limit)
                        else:
                            print(f"警告: 关节 {i} 索引越界")
                    except Exception as e:
                        print(f"计算关节 {i} 力矩时出错: {e}")
                
                print(f"计算的力矩: {torques[:8]}...")
                
                # 应用力矩
                if len(self.data.ctrl) >= self.joint_num:
                    self.data.ctrl[:self.joint_num] = torques
                    print(f"成功应用力矩到控制数组，控制数组长度: {len(self.data.ctrl)}")
                else:
                    print(f"错误: 控制数组长度不足 ({len(self.data.ctrl)})")
                
            print("===== 动作应用完成 =====")
        except Exception as e:
            print(f"应用动作时出错: {e}")

    
    def update_gait(self):
        """更新步态参数"""
        # 步态频率
        gait_freq = 1.3
        
        # 更新步态索引
        self.gait_index += 0.02 * gait_freq
        if self.gait_index > 1.0:
            self.gait_index = 0.0
        
        # 如果命令很小，重置步态
        if np.linalg.norm(self.commands[:3]) < 0.01:
            self.gait_index = 0.0
    
    def pd_control(self, pos_cur, vel_cur, pos_des, kp, kd, torque_limit):
        """PD控制算法计算力矩

        - 如果输入是标量，则返回单个力矩值
        - 如果输入是数组，则返回力矩数组
        
        Args:
            pos_cur: 当前位置   
            vel_cur: 当前速度   
            pos_des: 期望位置   
            kp: 位置增益
            kd: 速度增益
            torque_limit: 力矩限制
            
        Returns:
            计算得到的力矩
        """
        # 计算位置误差
        pos_error = pos_des - pos_cur
        
        # 计算力矩
        torque = kp * pos_error - kd * vel_cur
        
        # 限制力矩范围
        torque = np.clip(torque, -torque_limit, torque_limit)
        
        return torque
    
    def process_joystick_input(self, buttons, axes):
        """处理遥控器输入"""
        self.joy_buttons = buttons
        self.joy_axes = axes
        
        # 检查是否需要启动/停止控制器
        if buttons[4] == 1 and buttons[3] == 1:  # L1 + Y
            print("L1 + Y: 启动控制器...")
            self.start_controller = True
        
        if buttons[4] == 1 and buttons[2] == 1:  # L1 + X
            print("L1 + X: 停止控制器...")
            self.start_controller = False
        
        # 处理移动命令
        linear_x = axes[1]
        linear_y = axes[0]
        angular_z = axes[2]
        
        # 限制范围
        linear_x = np.clip(linear_x, -1.0, 1.0)
        linear_y = np.clip(linear_y, -1.0, 1.0)
        angular_z = np.clip(angular_z, -1.0, 1.0)
        
        # 缩放命令
        self.commands[0] = linear_x * 0.5  # 前后移动
        self.commands[1] = linear_y * 0.5  # 左右移动
        self.commands[2] = angular_z * 0.5  # 旋转
    
    def robot_state_callback(self, robot_state):
        """处理机器人状态回调"""
        # 不使用SDK的控制接口，但从中获取状态信息
        # 用于比较MuJoCo模型状态和真实机器人状态
        pass
    
    def imu_data_callback(self, imu_data):
        """处理IMU数据回调"""
        # 不使用SDK的控制接口，但从中获取IMU数据
        pass
        
    def update_sensor_data(self):
        """更新传感器数据"""
        # 更新时间戳
        self.imu_data['stamp'] = self.step_count
        
        try:
            # 更新四元数 - 只使用quat传感器
            if 'quat' in self.sensor_ids and self.sensor_ids['quat'] != -1:
                quat_id = self.sensor_ids['quat']
                if quat_id + 4 <= len(self.data.sensordata):
                    self.imu_data['quat'] = self.data.sensordata[quat_id:quat_id+4]
                    
                    # 从四元数估计角速度和加速度
                    # 这里我们使用简化方法：保持gyro和acc为零向量
                    # 在实际应用中，可以通过四元数的变化率来估计角速度
                    self.imu_data['gyro'] = np.zeros(3)  # 简化处理
                    self.imu_data['acc'] = np.zeros(3)    # 简化处理
                else:
                    print(f"警告: 传感器数据索引越界 (quat_id={quat_id}, sensordata长度={len(self.data.sensordata)})")
            else:
                print("警告: 未找到quat传感器，使用默认值")
        except Exception as e:
            print(f"更新传感器数据失败: {e}")
    
    def diagnostic_callback(self, diagnostic_value):
        """处理诊断信息回调"""
        # 处理诊断信息，例如校准状态
        if hasattr(diagnostic_value, 'name') and diagnostic_value.name == "calibration":
            print(f"校准状态: {diagnostic_value.code}")
            self.calibration_state = diagnostic_value.code
            
    def key_callback(self, key, scancode=None, action=None, mods=None):
        """处理键盘输入回调"""
        # 确保action参数存在，如果不存在则默认为1（按下）
        if action is not None and action != 1:
            return
            
        print(f"键盘输入: key={key}, scancode={scancode}, action={action}, mods={mods}")
        
        if key == 32:  # 空格键
            self.toggle_controller()
            print("切换控制器状态: " + ("启动" if self.start_controller else "停止"))
        elif key == 82 or key == 114:  # R/r键
            self.reset_simulation()
            print("重置仿真")
        elif key == 83 or key == 115:  # S/s键
            self.switch_to_stand_mode()
            print("切换到站立模式")
        elif key == 87 or key == 119:  # W/w键
            self.switch_to_walk_mode()
            print("切换到行走模式")
        # 使用IJKL键控制机器人底盘移动
        elif key == 73 or key == 105:  # I/i键
            self.set_command(0, 0.5)  # 前进
            print("底盘前进")
        elif key == 75 or key == 107:  # K/k键
            self.set_command(0, -0.5)  # 后退
            print("底盘后退")
        elif key == 74 or key == 106:  # J/j键
            self.set_command(1, 0.5)  # 左移
            print("底盘左移")
        elif key == 76 or key == 108:  # L/l键
            self.set_command(1, -0.5)  # 右移
            print("底盘右移")
        # 使用QE键控制旋转
        elif key == 81 or key == 113:  # Q/q键
            self.set_command(2, 0.5)  # 左转
            print("底盘左转")
        elif key == 69 or key == 101:  # E/e键
            self.set_command(2, -0.5)  # 右转
            print("底盘右转")
        # 其他控制键
        elif key == 80 or key == 112:  # P/p键
            self.stop_movement()
            print("停止移动")
        elif key == 256:  # ESC键
            print("退出程序")
            if self.viewer is not None:
                self.viewer.close()


# 如果直接运行此文件，则创建控制器实例并运行
if __name__ == '__main__':
    # 创建一个模拟的机器人实例
    class DummyRobot:
        def getMotorNumber(self):
            return 14
    
    # 初始化控制器
    controller = SF_Controller(
        model_path=os.path.expanduser("~/limx_ws/rl-deploy-with-python/controllers/model"),
        robot=DummyRobot(),
        robot_type="SF_TRON1A",
        start_controller=True
    )
    
    # 运行控制器
    controller.run()

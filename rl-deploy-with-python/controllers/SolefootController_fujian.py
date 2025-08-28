import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
import yaml
import os
import json
from datetime import datetime
import threading
import logging
from mujoco_logger import MuJoCoLogger

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class MuJoCoDirectController:
    def __init__(self, model_path, model_dir, robot_type):   
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mj_model.opt.timestep = 0.0025  # 400Hz: 1/400 = 0.0025s (物理仿真频率)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.default_qpos = self.mj_data.qpos.copy()#[0.   0.    0.840815   1.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
        # Launch the MuJoCo viewer in passive mode with custom settings 
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self.key_callback)
        try:
            # self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
            self.viewer.cam.distance = 5  # Set camera distance
            self.viewer.cam.elevation = -20  # Set camera elevation
            logging.info("MuJoCo viewer 已启动")
        except Exception as e:
            logging.error(f"启动viewer失败: {e}", exc_info=True)
            self.viewer = None

        self.dt = self.mj_model.opt.timestep  # Get simulation timestep
        print("Simulation timestep: ", self.dt)
        self.fps = 1 / self.dt  # Calculate frames per second (FPS)

        # 加载配置和模型
        self.load_config(f'{model_dir}/{robot_type}/params.yaml')
        
        # 初始化ONNX模型
        self.initialize_onnx_models(model_dir, robot_type)
        
        # 打印关节信息（在配置加载之后）
        # self.print_joint_info(14)
        
        # 获取关节和传感器索引
        self.joint_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint",
            "J1", "J2", "J3", "J4", "J5", "J6"
        ]
        self.joint_num = len(self.joint_names)#14
        
        # 获取传感器ID
        self.imu_quat_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")#42
        self.imu_gyro_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")#43
        self.imu_acc_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")#44

        # 初始化状态变量
        self.reset_state()
        
        # 添加模式控制变量
        self.mode = "WALK"  # 默认行走模式
        self.loop_count = 0
        
        # 初始化仿真控制变量
        self.simulation_paused = False  # 仿真暂停状态
        
        # 可调节的PD控制参数
        self.adjustable_kp = 45.0  # 可调节的刚度参数
        self.adjustable_kd = 1.5   # 可调节的阻尼参数
        self.ankle_kd = 0.8        # 踝关节阻尼参数
        
        # 初始化数据记录功能
        self.init_data_logging()
        
        # 初始化MuJoCo日志记录器
        self.mujoco_logger = MuJoCoLogger(self.dt)
        logging.info(f"MuJoCo Logger 初始化完成，时间步长: {self.dt}s")
        
        # 初始化线程锁
        self._init_lock = threading.Lock()
        self._init_flag = True
        
        # # 分析观测维度
        # print("\n正在分析观测向量维度...")
        # self.calculate_observation_size()
    
    def _is_safe_path(self, path):
        """验证路径安全性，防止路径遍历攻击"""
        try:
            # 规范化路径
            normalized_path = os.path.normpath(os.path.abspath(path))
            
            # 检查是否包含危险的路径组件
            dangerous_patterns = ['..', '~', '$']
            for pattern in dangerous_patterns:
                if pattern in path:
                    return False
            
            # 检查是否在允许的目录范围内
            home_dir = os.path.expanduser("~")
            if not normalized_path.startswith(home_dir):
                return False
                
            return True
        except Exception:
            return False

    def init_data_logging(self):
        """初始化数据记录功能"""
        # 创建保存目录 - 使用相对路径更安全
        base_dir = os.path.expanduser("~")
        self.data_save_dir = os.path.join(base_dir, "limx_ws", "pointfoot-mujoco-sim", "obs_data")
        
        # 验证路径安全性
        if not self._is_safe_path(self.data_save_dir):
            raise ValueError(f"不安全的路径: {self.data_save_dir}")
            
        os.makedirs(self.data_save_dir, exist_ok=True)
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.obs_json_file = os.path.join(self.data_save_dir, f"obs_data_{timestamp}.json")
        
        # 初始化数据存储列表
        self.obs_data_log = []
        self.obs_print_counter = 0
        self.obs_save_interval = 10  # 每10次观测保存一次
        self.obs_print_interval = 50  # 每50次观测打印一次详细信息
        
        print(f"\n=== 数据记录初始化完成 ===")
        print(f"JSON文件: {self.obs_json_file}")
        print(f"保存间隔: 每{self.obs_save_interval}次观测")
        print(f"打印间隔: 每{self.obs_print_interval}次观测")

    def load_config(self, config_path):
        # 验证配置文件路径安全性
        if not self._is_safe_path(config_path) or not config_path.endswith('.yaml'):
            raise ValueError(f"不安全的配置文件路径: {config_path}")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            logging.error(f"配置文件加载失败: {e}")
            raise
        self.config = config
        # Assign configuration parameters to controller variables
        self.joint_names = config['PointfootCfg']['joint_names']
        self.init_state = config['PointfootCfg']['init_state']['default_joint_angle']
        self.stand_duration = config['PointfootCfg']['stand_mode']['stand_duration']
        self.control_cfg = config['PointfootCfg']['control']#{'stiffness': 45.0, 'damping': 3, 'ankle_joint_damping': 1.5, 'ankle_joint_torque_limit': 20, 'action_scale_pos': 0.25, 'decimation': 10, 'user_torque_limit': 80}
        self.rl_cfg = config['PointfootCfg']['normalization']#{'clip_scales': {'clip_observations': 100.0, 'clip_actions': 100.0}, 'obs_scales': {'lin_vel': 2.0, 'ang_vel': 0.25, 'dof_pos': 1.0, 'dof_vel': 0.05}}
        self.obs_scales = config['PointfootCfg']['normalization']['obs_scales']#{'lin_vel': 2.0, 'ang_vel': 0.25, 'dof_pos': 1.0, 'dof_vel': 0.05}
        self.actions_size = config['PointfootCfg']['size']['actions_size']
        self.commands_size = config['PointfootCfg']['size']['commands_size']#5
        self.observations_size = config['PointfootCfg']['size']['observations_size']#54
        self.obs_history_length = config['PointfootCfg']['size']['obs_history_length']
        self.encoder_output_size = config['PointfootCfg']['size']['encoder_output_size']
        self.imu_orientation_offset = np.array(list(config['PointfootCfg']['imu_orientation_offset'].values()))
        self.user_cmd_cfg = config['PointfootCfg']['user_cmd_scales']#{'lin_vel_x': 1.0, 'lin_vel_y': 0.5, 'ang_vel_yaw': 1.0}
        self.loop_frequency = config['PointfootCfg']['loop_frequency']#500  # 收发指令频率(与仿真不同步)
        self.encoder_input_size = self.obs_history_length * self.observations_size#270
        # Initialize variables for actions, observations, and commands
        self.proprio_history_vector = np.zeros(self.obs_history_length * self.observations_size)
        self.encoder_out = np.zeros(self.encoder_output_size)
        self.actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(self.commands_size)  # command to the robot (e.g., velocity, rotation)
        self.scaled_commands = np.zeros(self.commands_size)
        self.base_lin_vel = np.zeros(3)  # base linear velocity
        self.base_position = np.zeros(3)  # robot base position
        self.loop_count = 0  # loop iteration count
        self.stand_percent = 0  # percentage of time the robot has spent in stand mode
        self.policy_session = None  # ONNX model session for policy inference
        self.joint_num = len(self.joint_names)  # number of joints

        self.ankle_joint_damping = config['PointfootCfg']['control']['ankle_joint_damping']#1.5
        self.ankle_joint_torque_limit = config['PointfootCfg']['control']['ankle_joint_torque_limit']#20

        self.gait_frequencies = config['PointfootCfg']['gait']['frequencies']
        self.gait_swing_height = config['PointfootCfg']['gait']['swing_height']

        # Initialize joint angles based on the initial configuration
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]
        
        # Set initial mode to "STAND"
        self.mode = "WALK"
        
        # Initialize gait index for gait timing
        self.gait_index = 0.0
        
        # Initialize action scale for converting actions to joint commands
        self.action_scale = 0.25  # Default action scale
    
    def key_callback(self, keycode):
        """键盘回调函数"""
        if keycode == ord('s') or keycode == ord('S'):
            self.mode = "STAND"
            self.stand_percent = 0.0  # 重置站立进度
            print(f"切换到站立模式: {self.mode}, 重置站立进度")
        elif keycode == ord('w') or keycode == ord('W'):
            self.mode = "WALK"
            print(f"切换到行走模式: {self.mode}")
        elif keycode == ord('p') or keycode == ord('P'):
            self.simulation_paused = not self.simulation_paused
            status = "暂停" if self.simulation_paused else "运行中"
            print(f"仿真状态: {status}")

        elif keycode == ord('r') or keycode == ord('R'):
            self.reset_simulation()
            print("仿真已重置")
        elif keycode == ord('h') or keycode == ord('H'):
            self.show_help()
        elif keycode == ord('q') or keycode == ord('Q'):
            print("退出程序")
            exit(0)
        elif keycode == ord('u') or keycode == ord('U'):  # 增加Kp
            self.adjustable_kp = min(200.0, self.adjustable_kp + 5.0)  # 添加上限
            print(f"Kp增加: {self.adjustable_kp:.1f}")
        elif keycode == ord('j') or keycode == ord('J'):  # 减少Kp
            self.adjustable_kp = max(0.0, self.adjustable_kp - 5.0)
            print(f"Kp减少: {self.adjustable_kp:.1f}")
        elif keycode == ord('i') or keycode == ord('I'):  # 增加Kd
            self.adjustable_kd = min(20.0, self.adjustable_kd + 0.5)  # 添加上限
            print(f"Kd增加: {self.adjustable_kd:.1f}")
        elif keycode == ord('k') or keycode == ord('K'):  # 减少Kd
            self.adjustable_kd = max(0.0, self.adjustable_kd - 0.5)
            print(f"Kd减少: {self.adjustable_kd:.1f}")
        # 踝关节Kd参数调节功能
        elif keycode == ord('o') or keycode == ord('O'):  # 增加踝关节Kd
            self.ankle_kd = min(10.0, self.ankle_kd + 0.5)  # 添加上限
            print(f"踝关节Kd增加: {self.ankle_kd:.1f}")
        elif keycode == ord('l') or keycode == ord('L'):  # 减少踝关节Kd
            self.ankle_kd = max(0.0, self.ankle_kd - 0.5)
            print(f"踝关节Kd减少: {self.ankle_kd:.1f}")
        
        # 数据可视化功能
        elif keycode == ord('v') or keycode == ord('V'):  # 手动触发数据可视化
            print(f"\n📈 手动触发数据可视化（当前步数: {self.mujoco_logger.step_count}）")
            self.mujoco_logger.plot_states()
        
        elif keycode == ord('c') or keycode == ord('C'):  # 清除日志数据
            print("\n🗑️ 重置日志数据")
            self.mujoco_logger.reset()
        
        elif keycode == ord('d') or keycode == ord('D'):  # 保存日志数据
            print("\n💾 保存日志数据")
            self.mujoco_logger.save_data()

    def initialize_onnx_models(self, model_dir, robot_type):
        # Configure ONNX Runtime session options to optimize CPU usage
        session_options = ort.SessionOptions()
        # Limit the number of threads used for parallel computation within individual operators
        session_options.intra_op_num_threads = 1
        # Limit the number of threads used for parallel execution of different operators
        session_options.inter_op_num_threads = 1
        # Enable all possible graph optimizations to improve inference performance
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable CPU memory arena to reduce memory fragmentation
        session_options.enable_cpu_mem_arena = False
        # Disable memory pattern optimization to have more control over memory allocation
        session_options.enable_mem_pattern = False

        # Define execution providers to use CPU only, ensuring no GPU inference
        cpu_providers = ['CPUExecutionProvider']

        # Load the ONNX model for policy
        self.policy_session = ort.InferenceSession(f'{model_dir}/{robot_type}/policy/policy.onnx', 
                                                   providers=cpu_providers, 
                                                   sess_options=session_options)
        
        # Load the ONNX model for encoder
        self.encoder_session = ort.InferenceSession(f'{model_dir}/{robot_type}/policy/encoder.onnx', 
                                                    providers=cpu_providers, 
                                                    sess_options=session_options)



    def print_joint_info(self, num_joints=8):   #1-8 leg, 9-14 arm
        """打印关节ID和名字"""
        print(f"\n=== 前{num_joints}个关节信息 ===")
        for i in range(min(num_joints, len(self.joint_names))):
            joint_name = self.joint_names[i]
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            print(f"关节{i}: ID={joint_id:2d}, 名字={joint_name}")
        print("=" * 40)
    
    def reset_state(self):
        """重置机器人状态到初始位置"""
        # 设置初始关节位置（使用配置文件中的值）
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                # 使用配置文件中的初始角度
                init_angle = self.init_state.get(joint_name, 0.0)
                self.init_joint_angles[i] = init_angle
        # 前向运动学计算
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def reset_simulation(self):
        """重置仿真状态"""
        try:
            # 重置 MuJoCo 数据
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            
            # 重置控制器状态
            self.loop_count = 0
            self.stand_percent = 0.0
            self.simulation_paused = False
            self.last_actions = np.zeros(self.joint_num)
            # 重置到初始状态
            self.reset_state()
            
            print("仿真已完全重置")
        except Exception as e:
            print(f"重置仿真失败: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n=== MuJoCo 机器人控制器帮助 ===")
        print("基本控制:")
        print("  S/s键  - 切换到站立模式")
        print("  W/w键  - 切换到行走模式")
        print("  P/p键  - 暂停/恢复仿真")
        print("  R/r键  - 重置仿真")
        print("  Q/q键  - 退出程序")
        print("  H/h键  - 显示帮助信息")
        sim_status = "暂停" if self.simulation_paused else "运行中"
        print(f"  仿真状态: {sim_status}")
        print(f"  控制模式: {self.mode}")
        print(f"  当前Kp: {self.adjustable_kp:.1f}")
        print(f"  当前Kd: {self.adjustable_kd:.1f}")
        print(f"  踝关节Kd: {self.ankle_kd:.1f}")
        print("\n键盘控制:")
        print("  S/s: 切换到站立模式")
        print("  W/w: 切换到行走模式")
        print("  P/p: 暂停/恢复仿真")
        print("  T/t: 切换步进/连续模式")
        if self.step_mode:
            print("  N/n/空格: 执行单步仿真")
            print("  1-9: 执行多步仿真")
        print("\nPD参数调节:")
        print("  U/u: 增加Kp (+5.0)")
        print("  J/j: 减少Kp (-5.0)")
        print("  I/i: 增加Kd (+0.5)")
        print("  K/k: 减少Kd (-0.5)")
        print("  O/o: 增加踝关节Kd (+0.5)")
        print("  L/l: 减少踝关节Kd (-0.5)")
        print("\n数据可视化:")
        print("  V/v: 手动触发数据可视化")
        print("  C/c: 清除/重置日志数据")
        print("  D/d: 保存日志数据到文件")
        print(f"  当前记录步数: {self.mujoco_logger.step_count}")
        print("  自动绘图: 仅在第200步触发")
        print("\n其他:")
        print("  H/h: 显示帮助信息")
        print("  Q/q: 退出程序")
        print("=" * 50)
    
    def get_kd_array(self):
        """生成带有踝关节专用kd参数的kd数组"""
        kd = np.full(self.joint_num, self.adjustable_kd)
        # 踝关节索引: ankle_L_Joint (3), ankle_R_Joint (7)
        ankle_indices = [3, 7]  # 左右踝关节索引
        for idx in ankle_indices:
            if idx < self.joint_num:
                kd[idx] = self.ankle_kd
        return kd
    
    def should_execute_step(self):
        """判断是否应该执行仿真步骤"""
        # 只检查是否暂停，没有暂停则正常执行
        return not self.simulation_paused
        
    def get_sensor_data(self):
        """获取传感器数据"""
        # 腿部关节
        leg_pos = self.mj_data.sensordata[:8]
        leg_vel = self.mj_data.sensordata[8:16]
        leg_torque = self.mj_data.sensordata[16:24]
        
        # 机械臂关节
        arm_pos = self.mj_data.sensordata[24:30]
        arm_vel = self.mj_data.sensordata[30:36]
        arm_torque = self.mj_data.sensordata[36:42]
        
        # IMU数据
        imu_quat = self.mj_data.sensordata[42:46]
        imu_gyro = self.mj_data.sensordata[46:49]
        imu_acc = self.mj_data.sensordata[49:52]
        
        return {
            'leg': {'pos': leg_pos, 'vel': leg_vel, 'torque': leg_torque},
            'arm': {'pos': arm_pos, 'vel': arm_vel, 'torque': arm_torque},
            'imu': {'quat': imu_quat, 'gyro': imu_gyro, 'acc': imu_acc}
        }
    
    def get_imu_data(self):
        """直接从MuJoCo传感器获取IMU数据"""
        # 获取四元数
        if self.imu_quat_id >= 0:
            quat_start = self.mj_model.sensor_adr[self.imu_quat_id]
            quat = self.mj_data.sensordata[quat_start:quat_start+4]
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            
        # 获取角速度
        if self.imu_gyro_id >= 0:
            gyro_start = self.mj_model.sensor_adr[self.imu_gyro_id]
            gyro = self.mj_data.sensordata[gyro_start:gyro_start+3]
        else:
            gyro = np.array([0.0, 0.0, 0.0])
            
        # 获取加速度
        if self.imu_acc_id >= 0:
            acc_start = self.mj_model.sensor_adr[self.imu_acc_id]
            acc = self.mj_data.sensordata[acc_start:acc_start+3]
        else:
            acc = np.array([0.0, 0.0, -9.81])
            
        return quat, gyro, acc
    
    def set_joint_commands(self, positions, velocities, torques, kp, kd):
        """带有力矩限制"""
        try:
            sensor_data = self.get_sensor_data()
            
            # 获取关节位置和速度
            joint_leg_pos = sensor_data['leg']['pos']
            joint_arm_pos = sensor_data['arm']['pos']
            joint_pos = np.concatenate([joint_leg_pos, joint_arm_pos])
            
            joint_leg_vel = sensor_data['leg']['vel']
            joint_arm_vel = sensor_data['arm']['vel']
            joint_vel = np.concatenate([joint_leg_vel, joint_arm_vel])
            
            for i, joint_name in enumerate(self.joint_names):
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0 and i < len(positions) and i < len(joint_pos):
                    pos_error = positions[i] - joint_pos[i]
                    vel_error = velocities[i] - joint_vel[i]
                    
                    # 计算控制力矩
                    control_torque = kp[i] * pos_error + kd[i] * vel_error + torques[i]
                    
                    if i < len(self.mj_data.ctrl):
                        self.mj_data.ctrl[i] = control_torque
                        
                        # 记录力矩数据用于日志
                        if not hasattr(self, 'last_torques'):
                            self.last_torques = np.zeros(self.joint_num)
                        if i < len(self.last_torques):
                            self.last_torques[i] = control_torque
                        
        except Exception as e:
            print(f"设置关节控制命令时出错: {e}")
    
    def should_execute_step(self):
        """判断是否应该执行仿真步骤"""
        # 只检查是否暂停，没有暂停则正常执行
        return not self.simulation_paused
            
    def get_sensor_data(self):
        """获取传感器数据"""
        # 腿部关节
        leg_pos = self.mj_data.sensordata[:8]
        leg_vel = self.mj_data.sensordata[8:16]
        leg_torque = self.mj_data.sensordata[16:24]
        
        # 机械臂关节
        arm_pos = self.mj_data.sensordata[24:30]
        arm_vel = self.mj_data.sensordata[30:36]
        arm_torque = self.mj_data.sensordata[36:42]
        
        # IMU数据
        imu_quat = self.mj_data.sensordata[42:46]
        imu_gyro = self.mj_data.sensordata[46:49]
        imu_acc = self.mj_data.sensordata[49:52]
        
        return {
            'leg': {'pos': leg_pos, 'vel': leg_vel, 'torque': leg_torque},
            'arm': {'pos': arm_pos, 'vel': arm_vel, 'torque': arm_torque},
            'imu': {'quat': imu_quat, 'gyro': imu_gyro, 'acc': imu_acc}
        }
    
    def get_imu_data(self):
        """直接从MuJoCo传感器获取IMU数据"""
        # 获取四元数
        if self.imu_quat_id >= 0:
            # print("self.imu_quat_id =", self.imu_quat_id)
            quat_start = self.mj_model.sensor_adr[self.imu_quat_id]
            quat = self.mj_data.sensordata[quat_start:quat_start+4]
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
                
        # 获取角速度
        if self.imu_gyro_id >= 0:
            gyro_start = self.mj_model.sensor_adr[self.imu_gyro_id]
            gyro = self.mj_data.sensordata[gyro_start:gyro_start+3]
        else:
            gyro = np.array([0.0, 0.0, 0.0])
                
        # 获取加速度
        if self.imu_acc_id >= 0:
            acc_start = self.mj_model.sensor_adr[self.imu_acc_id]
            acc = self.mj_data.sensordata[acc_start:acc_start+3]
        else:
            acc = np.array([0.0, 0.0, -9.81])
                
        return quat, gyro, acc
    
    def set_joint_commands(self, positions, velocities, torques, kp, kd):
        """带有力矩限制"""
        try:
            sensor_data = self.get_sensor_data()
            
            # 获取关节位置和速度
            joint_leg_pos = sensor_data['leg']['pos']
            joint_arm_pos = sensor_data['arm']['pos']
            joint_pos = np.concatenate([joint_leg_pos, joint_arm_pos])
            
            joint_leg_vel = sensor_data['leg']['vel']
            joint_arm_vel = sensor_data['arm']['vel']
            joint_vel = np.concatenate([joint_leg_vel, joint_arm_vel])
            
            for i, joint_name in enumerate(self.joint_names):
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0 and i < len(positions) and i < len(joint_pos):
                    pos_error = positions[i] - joint_pos[i]
                    vel_error = velocities[i] - joint_vel[i]
                    
                    # 计算控制力矩
                    control_torque = kp[i] * pos_error + kd[i] * vel_error + torques[i]
                    
                    if i < len(self.mj_data.ctrl):
                        self.mj_data.ctrl[i] = control_torque
                        
                        # 记录力矩数据用于日志
                        if not hasattr(self, 'last_torques'):
                            self.last_torques = np.zeros(self.joint_num)
                        if i < len(self.last_torques):
                            self.last_torques[i] = control_torque
                            
        except Exception as e:
            print(f"设置关节控制命令时出错: {e}")
            # 出错时将所有力矩设为0
            for i in range(len(self.mj_data.ctrl)):
                self.mj_data.ctrl[i] = 0.0
    
    def compute_observation(self):
        """计算观测向量（包含全局速度计算用于控制逻辑）"""
        # 获取IMU数据
        quat, gyro, acc = self.get_imu_data()
        # gyro = 
        # 获取关节状态
        sensor_data = self.get_sensor_data()
        joint_leg_pos = sensor_data['leg']['pos'] #(8,)
        joint_arm_pos = sensor_data['arm']['pos'] #(6,)
        joint_pos = np.concatenate([joint_leg_pos, joint_arm_pos]) #(14,)
        
        joint_leg_vel = sensor_data['leg']['vel'] #(8,)
        joint_arm_vel = sensor_data['arm']['vel'] #(6,)
        joint_vel = np.concatenate([joint_leg_vel, joint_arm_vel]) #(14,)
        
        # ========== 全局速度计算（用于控制逻辑） ==========
        # 从 MuJoCo直接获取线性速度（已经是全局的）
        self.base_lin_vel_global = self.mj_data.qvel[:3].copy()  # 全局线性速度 [vx, vy, vz]
        
        # 将局部角速度转换为全局角速度
        # 方法1：使用机器人在世界坐标系中的姿态四元数
        robot_quat = self.mj_data.qpos[3:7]  # 机器人在世界坐标系中的四元数 [qw, qx, qy, qz]
        local_ang_vel = self.mj_data.qvel[3:6].copy()  # 局部角速度 [wx, wy, wz]
        
        # 计算旋转矩阵（从局部坐标系到全局坐标系）
        rotation_matrix = R.from_quat(robot_quat).as_matrix()
        
        # 转换为全局角速度
        self.base_ang_vel_global = np.dot(rotation_matrix, local_ang_vel)
        gyro = self.base_ang_vel_global
        # 保存局部角速度用于对比和调试
        self.base_ang_vel_local = local_ang_vel
        
        # ========== 观测向量计算（保持ONNX兼容性） ==========
        # 计算重力投影
        imu_orientation = quat
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()
        gravity_vector = np.array([0, 0, -1])
        projected_gravity = np.dot(inverse_rot, gravity_vector)
        
        # 应用IMU偏移校正（保持原有的局部角速度用于观测向量）
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()
        base_ang_vel = np.dot(rot, gyro)  # 局部角速度（用于观测向量，保持ONNX兼容性）
        projected_gravity = np.dot(rot, projected_gravity)
        
        # 步态信息
        gait = np.array([self.gait_frequencies, 0.5, 0.5, self.gait_swing_height])
        self.gait_index += 0.02 * gait[0]
        if self.gait_index > 1.0:
            self.gait_index = 0.0
        gait_clock = np.array([np.sin(self.gait_index * 2 * np.pi), 
                              np.cos(self.gait_index * 2 * np.pi)])
        
        # 构建观测向量（保持原有维度和结构）
        joint_pos_input = (joint_pos - self.init_joint_angles) * self.obs_scales['dof_pos']
        #观测关节角度-初始关节角度
        obs = np.concatenate([
            gyro * self.obs_scales['ang_vel'],  # 局部角速度
            projected_gravity,
            joint_pos_input,
            joint_vel * self.obs_scales['dof_vel'],
            self.last_actions,
            gait_clock,
            gait
        ])
        
        # 记录和打印obs数据
        self.log_observation_data(obs, base_ang_vel, projected_gravity, joint_pos_input, joint_vel, gait_clock, gait)
        
        # 打印全局速度信息（用于调试）
        if self.loop_count % 100 == 0:  # 每100步打印一次
            print(f"\n=== 速度转换信息 (Step {self.loop_count}) ===")
            print(f"全局线性速度: [{self.base_lin_vel_global[0]:.3f}, {self.base_lin_vel_global[1]:.3f}, {self.base_lin_vel_global[2]:.3f}] m/s")
            print(f"局部角速度: [{self.base_ang_vel_local[0]:.3f}, {self.base_ang_vel_local[1]:.3f}, {self.base_ang_vel_local[2]:.3f}] rad/s")
            print(f"全局角速度: [{self.base_ang_vel_global[0]:.3f}, {self.base_ang_vel_global[1]:.3f}, {self.base_ang_vel_global[2]:.3f}] rad/s")
            print(f"机器人姿态四元数: [{robot_quat[0]:.3f}, {robot_quat[1]:.3f}, {robot_quat[2]:.3f}, {robot_quat[3]:.3f}]")
            print(f"局部角速度(观测用): [{base_ang_vel[0]:.3f}, {base_ang_vel[1]:.3f}, {base_ang_vel[2]:.3f}] rad/s")
        
        return obs
    
    def log_observation_data(self, obs, base_ang_vel, projected_gravity, joint_pos_input, joint_vel, gait_clock, gait):
        """记录和打印obs数据"""
        self.obs_print_counter += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 构建统一的obs数据结构
        obs_data_entry = {
            'timestamp': current_time,
            'simulation_time': float(self.mj_data.time),
            'loop_count': int(self.loop_count),
            'mode': self.mode,
            'obs_dimension': len(obs),
            'components': {
                'base_ang_vel': base_ang_vel.tolist(),
                'projected_gravity': projected_gravity.tolist(),
                'joint_pos_scaled': joint_pos_input.tolist(),
                'joint_vel_scaled': joint_vel.tolist(),
                'last_actions': self.last_actions.tolist(),
                'gait_clock': gait_clock.tolist(),
                'gait_params': gait.tolist()
            },
            'complete_obs': obs.tolist()
        }
        
        # 每隔一定次数打印详细信息（JSON格式）
        # if self.obs_print_counter % self.obs_print_interval == 0:
            # print(f"\n=== OBS Data JSON Format [{current_time}] ===")
            # print(json.dumps(obs_data_entry, indent=2, ensure_ascii=False))
            # print("=" * 80)
        
        # 每隔一定次数保存数据
        if self.obs_print_counter % self.obs_save_interval == 0:
            # 保存到JSON数据结构
            self.obs_data_log.append(obs_data_entry)
            
            # 定期保存JSON文件（每100次保存一次）
            if len(self.obs_data_log) % 100 == 0:
                with open(self.obs_json_file, 'w') as jsonfile:
                    json.dump(self.obs_data_log, jsonfile, indent=2, ensure_ascii=False)
                print(f"\n💾 已保存 {len(self.obs_data_log)} 条obs数据到: {self.obs_json_file}")
        
        # 简单的进度显示
        if self.obs_print_counter % 100 == 0:
            print(f"\n📊 OBS数据记录进度: {self.obs_print_counter} 次观测 | 模式: {self.mode} | 时间: {self.mj_data.time:.2f}s")
    
    def finalize_data_logging(self):
        """完成数据记录，保存所有剩余数据"""
        if hasattr(self, 'obs_data_log') and self.obs_data_log:
            # 保存最终的JSON数据（使用统一格式）
            with open(self.obs_json_file, 'w') as jsonfile:
                json.dump(self.obs_data_log, jsonfile, indent=2, ensure_ascii=False)
            
            print(f"\n=== 数据记录完成 ===")
            print(f"总计记录 {self.obs_print_counter} 次观测")
            print(f"保存 {len(self.obs_data_log)} 条详细数据")
            print(f"JSON文件: {self.obs_json_file}")
            print(f"数据保存目录: {self.data_save_dir}")
            
            if self.obs_data_log:
                print(json.dumps(self.obs_data_log[-1], indent=2, ensure_ascii=False))
            print("=" * 50)
    
    def calculate_observation_size(self):
        """计算观测向量的实际维度"""
        # 模拟一次观测计算来获取实际维度
        try:
            obs = self.compute_observation()
            actual_size = len(obs)
            config_size = self.observations_size
            
            print(f"\n=== 观测维度分析 ===")
            print(f"实际观测维度: {actual_size}")
            print(f"配置文件中的维度: {config_size}")
            
            # 详细分解观测向量的构成
            print(f"\n观测向量构成分析:")
            print(f"- base_ang_vel: 3维 (角速度)")
            print(f"- projected_gravity: 3维 (重力投影)")
            print(f"- joint_pos_input: {self.joint_num}维 (关节位置)")
            print(f"- joint_vel: {self.joint_num}维 (关节速度)")
            print(f"- last_actions: {len(self.last_actions)}维 (上次动作)")
            print(f"- gait_clock: 2维 (步态时钟)")
            print(f"- gait: 4维 (步态参数)")
            expected = 3 + 3 + self.joint_num + self.joint_num + len(self.last_actions) + 2 + 4
            print(f"- 预期总维度: {expected}")
            print("=" * 40)
            
            return actual_size
        except Exception as e:
            print(f"计算观测维度时出错: {e}")
            return None
    
    def update_history_buffer(self, obs):
        """更新历史观测缓冲区"""
        if not hasattr(self, 'proprio_history_buffer'):
            # 初始化历史缓冲区
            input_size = self.obs_history_length * self.observations_size
            self.proprio_history_buffer = np.zeros(input_size)
            # 用当前观测填充整个历史长度
            for i in range(self.obs_history_length):
                self.proprio_history_buffer[i * self.observations_size:(i + 1) * self.observations_size] = obs
        else:
            # 左移现有历史缓冲区
            self.proprio_history_buffer[:-self.observations_size] = self.proprio_history_buffer[self.observations_size:]
            # 在末尾添加当前观测
            self.proprio_history_buffer[-self.observations_size:] = obs
    
    def compute_encoder(self):
        """计算编码器输出"""
        # 将历史缓冲区转换为输入张量
        input_tensor = np.concatenate([self.proprio_history_buffer], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        
        # 获取编码器输入输出名称和形状
        encoder_input_names = [input.name for input in self.encoder_session.get_inputs()]
        encoder_output_names = [output.name for output in self.encoder_session.get_outputs()]
        encoder_input_shapes = [input.shape for input in self.encoder_session.get_inputs()]
        encoder_output_shapes = [output.shape for output in self.encoder_session.get_outputs()]
        
        # 创建输入字典
        inputs = {encoder_input_names[0]: input_tensor}
        
        # 运行编码器
        output = self.encoder_session.run(encoder_output_names, inputs)
        
        # 展平输出并存储
        self.encoder_out = np.array(output).flatten()
        
        # 打印编码器详细信息（每50步打印一次）
        if hasattr(self, 'loop_count') and self.loop_count % 50 == 0:
            print(f"\n=== 编码器输出信息 (Step {self.loop_count}) ===")
            print(f"编码器原始输出形状: {np.array(output).shape}")
            print(f"编码器展平后形状: {self.encoder_out.shape}")
            print(f"编码器输出维度: {len(self.encoder_out)}")
            # print(f"编码器输出前5个值: {self.encoder_out[:5]}")
            # print(f"编码器输入形状: {input_tensor.shape}")
            # print(f"编码器输入维度: {len(input_tensor)}")
    
    def compute_actions(self, obs):
        """计算动作"""
        # 打印各组件的维度信息
        # print(f"=== Policy Input Debug Info ===")
        # print(f"Encoder output shape: {self.encoder_out.shape}")
        # print(f"Observation shape: {obs.shape}")
        # print(f"Scaled commands shape: {self.scaled_commands.shape}")
        
        # 拼接编码器输出、观测和命令
        input_tensor = np.concatenate([self.encoder_out, obs, self.scaled_commands], axis=0)#3+54+5=62
        input_tensor = input_tensor.astype(np.float32)
        
        # 获取策略输入输出名称
        policy_input_names = [input.name for input in self.policy_session.get_inputs()]# ['nn_input']
        policy_output_names = [output.name for output in self.policy_session.get_outputs()]# ['nn_output']
        policy_input_shapes = [self.policy_session.get_inputs()[i].shape for i in range(self.policy_session.get_inputs().__len__())]
        policy_output_shapes = [self.policy_session.get_outputs()[i].shape for i in range(self.policy_session.get_outputs().__len__())]
        
        # print(f"Policy input expected shapes: {policy_input_shapes}")
        # print(f"Policy output shapes: {policy_output_shapes}")
        # print(f"Actual concatenated input shape: {input_tensor.shape}")
        
        # 检查维度匹配
        expected_size = policy_input_shapes[0][0] if len(policy_input_shapes[0]) > 0 else policy_input_shapes[0]
        actual_size = input_tensor.shape[0]
        
        # 创建输入字典
        inputs = {policy_input_names[0]: input_tensor}
        
        # 运行策略
        output = self.policy_session.run(policy_output_names, inputs)
        
        # 展平输出并返回
        actions = np.array(output).flatten() 
        return actions


    
    def handle_stand_mode(self):
        """处理站立模式 - 使用obs数据进行闭环控制"""
        
        # 计算观测数据
        obs = self.compute_observation()
        
        # 更新历史缓冲区
        self.update_history_buffer(obs)
        
        # 设置站立模式的命令（较小的速度命令，保持稳定）
        stand_commands = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [vx=0, vy=0, wz=0, Kp=0, Kd=0]
        self.scaled_commands = stand_commands  # 直接使用命令，不需要缩放
        
        # 使用ONNX推理
        self.compute_encoder()
        actions = self.compute_actions(obs)
        
        # 限制动作范围（站立模式使用更小的动作幅度）
        actions = np.clip(actions, -1.0, 1.0)  # 比行走模式更保守的动作范围
        self.last_actions = actions.copy()
        
        # 转换为关节命令
        joint_positions = self.init_joint_angles + actions * self.action_scale
        joint_velocities = np.zeros(self.joint_num)
        joint_torques = np.zeros(self.joint_num)
        
        # 初始化站立模式的PD参数（如果需要）
        with self._init_lock:
            if self._init_flag:
                self.adjustable_kp = 50.0  # 站立模式使用较高刚度
                self.adjustable_kd = 2.0   # 站立模式使用较高阻尼
                self._init_flag = False
        
        # 使用可调节的PD参数
        kp = np.full(self.joint_num, self.adjustable_kp)  # 刚度
        kd = self.get_kd_array()  # 阻尼（包含踝关节专用参数）
        
        # 设置关节命令
        self.set_joint_commands(joint_positions, joint_velocities, 
                                   joint_torques, kp, kd)
    
    def handle_walk_mode(self):
        """处理行走模式"""
        walk_flag = 1
        
        # 设置行走模式的速度命令
        walk_commands = np.array([1.5, 0.0, 0.0, 0.0, 0.0])  # [vx=1.5m/s, vy=0, wz=0, Kp=0, Kd=0]
        self.scaled_commands = walk_commands  # 设置前进速度
        
        # 计算观测
        obs = self.compute_observation()
        
        # 更新历史缓冲区
        self.update_history_buffer(obs)
        
        # 使用ONNX推理
        self.compute_encoder()
        actions = self.compute_actions(obs)
        
        # 限制动作范围
        actions = np.clip(actions, -5.0, 5.0)
        self.last_actions = actions.copy()
        
        # 转换为关节命令
        joint_positions = self.init_joint_angles + actions * self.action_scale
        joint_velocities = np.zeros(self.joint_num)
        joint_torques = np.zeros(self.joint_num)
        # 注意：walk_flag变量未定义，这里可能是代码错误
        # 初始化行走模式的PD参数（线程安全）
        with self._init_lock:
            if self._init_flag:
                self.adjustable_kp = 45.0  # 可调节的刚度参数
                self.adjustable_kd = 1.5   # 可调节的阻尼参数
                self._init_flag = False

        # 使用可调节的PD参数
        kp = np.full(self.joint_num, self.adjustable_kp)
        kd = self.get_kd_array()  # 阻尼（包含踝关节专用参数）
        
        # 设置控制命令
        self.set_joint_commands(joint_positions, joint_velocities, 
                              joint_torques, kp, kd)
    
    def update(self):
        """更新机器人状态基于当前模式"""
        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()
        
        # 记录数据到MuJoCo Logger
        self._log_mujoco_data()
        
        # 检查是否需要绘图（仅在第50步）
        if self.mujoco_logger.step_count == 50:
            print(f"\n📈 触发数据可视化（步数: {self.mujoco_logger.step_count}）")
            self.mujoco_logger.plot_states()
        
        # 每100次循环打印一次模式状态
        if self.loop_count % 100 == 0:
            print(f"当前模式: {self.mode}, 循环计数: {self.loop_count}")
            print("按 'S' 切换到站立模式, 按 'W' 切换到行走模式, 按 'Q' 退出")
        
        # 增加循环计数
        self.loop_count += 1
    
    def _log_mujoco_data(self):
        """
        收集并记录MuJoCo仿真数据
        记录与原版Logger相同的数据类型
        """
        try:
            # 获取当前传感器数据
            sensor_data = self.get_sensor_data()
            quat, gyro, acc = self.get_imu_data()
            
            # 获取关节状态
            joint_leg_pos = sensor_data['leg']['pos']  # 腿部关节位置
            joint_arm_pos = sensor_data['arm']['pos']  # 手臂关节位置
            joint_pos = np.concatenate([joint_leg_pos, joint_arm_pos])  # 所有关节位置
            
            joint_leg_vel = sensor_data['leg']['vel']  # 腿部关节速度
            joint_arm_vel = sensor_data['arm']['vel']  # 手臂关节速度
            joint_vel = np.concatenate([joint_leg_vel, joint_arm_vel])  # 所有关节速度
            
            # 计算基础速度（从IMU数据）
            # 注意：这里使用角速度，线速度在当前实现中未使用
            base_ang_vel = gyro
            
            # 获取命令数据
            commands = getattr(self, 'scaled_commands', np.zeros(5))
            
            # 获取最后的动作
            last_actions = getattr(self, 'last_actions', np.zeros(self.joint_num))
            
            # 计算功率（简化版本，基于关节力矩和速度）
            # 这里使用一个简化的功率计算
            if hasattr(self, 'last_torques'):
                power = np.sum(np.abs(self.last_torques * joint_vel))
            else:
                power = 0.0
            
            # 准备记录数据
            log_data = {
                # 关节数据
                'dof_pos': joint_pos,
                'dof_vel': joint_vel,
                'dof_pos_target': self.init_joint_angles + last_actions * self.action_scale,
                
                # 基础速度数据（使用真实的全局速度数据）
                'base_vel_x': getattr(self, 'base_lin_vel_global', np.zeros(3))[0],  # 全局线速度X
                'base_vel_y': getattr(self, 'base_lin_vel_global', np.zeros(3))[1],  # 全局线速度Y
                'base_vel_z': getattr(self, 'base_lin_vel_global', np.zeros(3))[2],  # 全局线速度Z
                'base_vel_yaw': getattr(self, 'base_ang_vel_global', np.zeros(3))[2],  # 全局Yaw角速度
                
                # 命令数据
                'command_x': commands[0] if len(commands) > 0 else 0.0,
                'command_y': commands[1] if len(commands) > 1 else 0.0,
                'command_yaw': commands[2] if len(commands) > 2 else 0.0,
                
                # 动作和力矩
                'actions': last_actions,
                'dof_torque': getattr(self, 'last_torques', np.zeros(self.joint_num)),
                
                # 功率
                'power': power,
                
                # 接触力（简化版本，这里使用零值，实际需要从MuJoCo获取）
                'contact_forces_z': np.zeros(4),  # 假设4个足端
            }
            
            # 记录数据
            self.mujoco_logger.log_states(log_data)
            
        except Exception as e:
            logging.warning(f"数据记录过程中出现错误: {e}")
    
    def check_stand(self):
        quat, _, _ = self.get_imu_data()
        rot = R.from_quat(quat).as_matrix()
        return rot[2,2] > 0.8

    def run_control_loop(self, duration=120.0, render=True):
        """运行控制循环
        
        频率架构：
        - 物理仿真频率: 50Hz (timestep=0.02s)
        - 收发指令频率: 500Hz (与仿真不同步，独立高频控制)
        - 神经网络推理频率: 50Hz (decimation=10)
        - 可视化渲染频率: 60Hz
        """

        render_fps = 60
        last_sync_time = time.time()
        decimation = 10  # 每10次指令发送执行一次神经网络推理 (500Hz指令 -> 50Hz网络)
        last_update_time = time.time()
        
        # 初始化命令
        self.commands = np.zeros(5)  # [vx, vy, wz, Kp, Kd]
        self.last_actions = np.zeros(self.joint_num)
        
        while self.mj_data.time < duration:
            # 检查是否需要执行仿真步骤
            should_step = self.should_execute_step()
            
            if should_step:
                self.update()
                # if self.check_stand():
                #     pass
                # else:
                #     break

                # 每个控制周期执行一次模式更新
                for _ in range(decimation):
                    mujoco.mj_step(self.mj_model, self.mj_data)
                
                # 同步viewer显示
                if render and hasattr(self, 'viewer') and self.viewer is not None:
                    if (time.time() - last_sync_time) > (1.0 / render_fps - 1e-3):
                        self.viewer.sync()
                        last_sync_time = time.time()
            
            delay_time = (self.mj_model.opt.timestep * decimation - (time.time() - last_update_time))
            if delay_time > 0:
                time.sleep(delay_time)
            last_update_time = time.time()
        mujoco.mj_sleep(0.5)

        
        # 在控制循环结束时保存所有数据
        self.finalize_data_logging()

# 类定义结束

# 使用示例
if __name__ == "__main__":
    model_path = "/home/xinyun/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A/xml/robot_with_arm.xml"
    model_dir = "/home/xinyun/limx_ws/rl-deploy-with-python/controllers/model"
    robot_type = "SF_TRON1A"
    
    controller = MuJoCoDirectController(model_path, model_dir, robot_type)
    controller.run_control_loop(duration=120.0, render=True)    #运行120秒
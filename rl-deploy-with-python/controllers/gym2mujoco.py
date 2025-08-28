import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import XBotLCfg
import torch
import onnxruntime as ort
import yaml
 
# 指令
class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0
 
# 四元数转欧拉角
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])
 
def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    从mujoco 中获取观测值
    '''
    # 关节位置
    q = data.qpos.astype(np.double)
    # 关节速度（应该是广义速度，前三个是Body的线速度）
    dq = data.qvel.astype(np.double)
    # 调整四元素顺序，四元数的顺序是 [w, x, y, z]。而这段代码将其顺序调整为 [x, y, z, w]
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    # 四元数转旋转矩阵
    r = R.from_quat(quat)
    # 提取广义速度的的前三个分量，然后从世界坐标系转到本体坐标系
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    # 获取角速度的信息XYZ
    omega = data.sensor('angular-velocity').data.astype(np.double)
    #  将重力方向（假设在世界坐标系下为 0, 0, -1）转换到机器人基坐标系中
    # 就是表示重力的影响方向。为Z轴的负方向。
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)
 
def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    # PD控制器，无前馈
    return (target_q - q) * kp + (target_dq - dq) * kd
 
def initialize_onnx_models(model_policy, model_encoder):
    """Initialize ONNX models for policy and encoder
    
    Args:
        model_policy (str): Path to policy ONNX model
        model_encoder (str): Path to encoder ONNX model
        
    Returns:
        tuple: (policy_session, encoder_session, policy_info, encoder_info)
    """
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
    
    # Load the ONNX model and set up input and output names
    policy_session = ort.InferenceSession(model_policy, sess_options=session_options, providers=cpu_providers)

    policy_input_names = [policy_session.get_inputs()[i].name for i in range(policy_session.get_inputs().__len__())]
    policy_output_names = [policy_session.get_outputs()[i].name for i in range(policy_session.get_outputs().__len__())]
    policy_input_shapes = [policy_session.get_inputs()[i].shape for i in range(policy_session.get_inputs().__len__())]
    policy_output_shapes = [policy_session.get_outputs()[i].shape for i in range(policy_session.get_outputs().__len__())]

    encoder_session = ort.InferenceSession(model_encoder, sess_options=session_options, providers=cpu_providers)


    encoder_input_names = [encoder_session.get_inputs()[i].name for i in range(encoder_session.get_inputs().__len__())]
    encoder_output_names = [encoder_session.get_outputs()[i].name for i in range(encoder_session.get_outputs().__len__())]
    encoder_input_shapes = [encoder_session.get_inputs()[i].shape for i in range(encoder_session.get_inputs().__len__())]
    encoder_output_shapes = [encoder_session.get_outputs()[i].shape for i in range(encoder_session.get_outputs().__len__())]
    
    # Return sessions and metadata
    policy_info = {
        'input_names': policy_input_names,
        'output_names': policy_output_names,
        'input_shapes': policy_input_shapes,
        'output_shapes': policy_output_shapes
    }
    
    encoder_info = {
        'input_names': encoder_input_names,
        'output_names': encoder_output_names,
        'input_shapes': encoder_input_shapes,
        'output_shapes': encoder_output_shapes
    }
    
    return policy_session, encoder_session, policy_info, encoder_info
 
def load_config(config_file):
    """Load configuration from YAML file
    
    Args:
        config_file (str): Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary with all loaded parameters
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    pointfoot_cfg = config['PointfootCfg']
    
    # Create a structured configuration object
    cfg_dict = {
        'joint_names': pointfoot_cfg['joint_names'],
        'init_state': pointfoot_cfg['init_state']['default_joint_angle'],
        'stand_duration': pointfoot_cfg['stand_mode']['stand_duration'],
        'control_cfg': pointfoot_cfg['control'],
        'rl_cfg': pointfoot_cfg['normalization'],
        'obs_scales': pointfoot_cfg['normalization']['obs_scales'],
        'actions_size': pointfoot_cfg['size']['actions_size'],
        'commands_size': pointfoot_cfg['size']['commands_size'],
        'observations_size': pointfoot_cfg['size']['observations_size'],
        'obs_history_length': pointfoot_cfg['size']['obs_history_length'],
        'encoder_output_size': pointfoot_cfg['size']['encoder_output_size'],
        'imu_orientation_offset': np.array(list(pointfoot_cfg['imu_orientation_offset'].values())),
        'user_cmd_cfg': pointfoot_cfg['user_cmd_scales'],
        'loop_frequency': pointfoot_cfg['loop_frequency'],
        'ankle_joint_damping': pointfoot_cfg['control']['ankle_joint_damping'],
        'ankle_joint_torque_limit': pointfoot_cfg['control']['ankle_joint_torque_limit'],
        'gait_frequencies': pointfoot_cfg['gait']['frequencies'],
        'gait_swing_height': pointfoot_cfg['gait']['swing_height'],
    }
    
    # Calculate derived values
    cfg_dict['encoder_input_size'] = cfg_dict['obs_history_length'] * cfg_dict['observations_size']
    cfg_dict['joint_num'] = len(cfg_dict['joint_names'])
    
    # Initialize joint angles based on the initial configuration
    init_joint_angles = np.zeros(len(cfg_dict['joint_names']))
    for i, joint_name in enumerate(cfg_dict['joint_names']):
        init_joint_angles[i] = cfg_dict['init_state'][joint_name]
    cfg_dict['init_joint_angles'] = init_joint_angles
    
    return cfg_dict
 
def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.
    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
    Returns:
        None
    """
    # 加载模型
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    # 时间步长
    model.opt.timestep = cfg.sim_config.dt
    # 创建数据结构
    data = mujoco.MjData(model)
    # 进行仿真步骤
    mujoco.mj_step(model, data)
    # 可视化
    viewer = mujoco_viewer.MujocoViewer(model, data)
 
    # 初始化目标动作，为位置
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
 
    # 历史观测放入一个队列
    hist_obs = deque()
    #  frame_stack= 15，num_single_obs观测维度= 47（无特权的正常观测）
    # 15*47个观测
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
 
    count_lowlevel = 0
 
    # 迭代的次数是通过计算仿真持续时间 sim_duration 除以时间步长 dt 得到的
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
 
        # Obtain an observation
        # 获取观察值，可以看到，mujoco里获取观测是十分容易的。
        q, dq, quat, v, omega, gvec = get_obs(data)
        # 从q中取后面num_actions个，因为前面的是广义的位置姿态什么的，不是关节角
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]
 
        # 1000hz -> 100hz
        # 通过计数降低频率
        # 以下是10个dt 执行一次
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 定义观测 1*47
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            # 欧拉角
            eu_ang = quaternion_to_euler_array(quat)
            # 欧拉角的标准范围通常是 [-π, π]。如果某个欧拉角的值大于 π，则减去 2π
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            
            #开始给观测赋值
            # 一个0.64秒的周期性波形，模拟相位发生器
            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            # obs_scales 配置为2
            # 对观测的值进行量纲的缩放
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel # 2
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel # 0.25
            obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos   # 1
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel  # 0.05
            # 上次的动作，经过剪裁后的action.
            obs[0, 29:41] = action
            # 角速度
            obs[0, 41:44] = omega
            # 姿态欧拉角
            obs[0, 44:47] = eu_ang
            
            # 18，观测阈值剪裁，这样做是为了防止观测数据中出现异常值或过大的数值，这对于训练稳定的控制算法非常重要
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
 
            # 加入队列
            hist_obs.append(obs)
            # 保持队列长度一致
            hist_obs.popleft()
 
            # 策略（RL控制器的）输入
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            # 在15个历史观测下
            # 将历史观测（hist_obs[i]）中的数据拼接到 policy_input 数组中，移位47。
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            
            # policy_input输入47 * 15，个数据，然后输出动作（12维的）
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            # 对输出的动作进行剪裁 ，正负18
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            # 关节力矩 = 关节位置增量（就是相对零点的），动作输出 * 0.25
            target_q = action * cfg.control.action_scale
 
        # 期望速度为0。
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        # PD控制器输出力矩
        # q是当前位置，未使用默认位置，那么就是都是基于同一零点的绝对位置
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        # 限制力矩范围 正负200
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        # 输出的控制就是力矩
        # 这里10ms中输出了10个力矩，并没有做插值，但是按1ms的关节反馈，输出力矩应该是按反馈变化的。
        # 但这10ms内的目标关节位置是不变的
        data.ctrl = tau
 
        # 步进
        mujoco.mj_step(model, data)
        # 仿真渲染
        viewer.render()
        count_lowlevel += 1
 
    viewer.close()
 
    # 分析：
    # 在sim2sim中，从策略获取到动作（基于零点的一个绝对位置，弧度）后
    # 1对动作进行剪裁（不是软限位，正负18）
    # 2对动作进行缩放 （0.25）
    # 3作为目标的关节位置。
    # 4 PD控制器算力矩（action 位置+ 默认位置 - 当前关节位置）
    # 5 限制力矩范围
    # 6下发到电机去
 
    # 对比一下play:
    # 获取到action
    # 1随机延迟模拟
    # 2高斯噪声模拟
    # 3对动作进行剪裁（正负18）
    # 4对动作进行缩放（0.25）
    # 5 PD控制器 （action - 当前关节位置）
    # 5限制力矩范围
    # 6下发到电机去
 
    # 对于噪声延迟等，仿真可以暂时去掉
    # 默认位置的定义：default_joint_angles =  # = target angles [rad] when action = 0.0
    # 若定义的默认位置不是0，那么sim2sim 中的PD控制器，就需要加上默认定义位置这一项
    # 因为训练出来的action输出是基于默认关节位置的。
    # 仿真中并没有定义关节软限位
    # 也没有限制关节的转动速度
    # 关节的正方向应该是从urdf,mujoco 的xml ，isaac的USD是约定统一的。
    # 在训练和sim时包括real 时，对于剪裁及缩放，一定要使用相同的参数。
 
if __name__ == '__main__':
    model_path = "/home/xinyun/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A/xml/robot_with_arm.xml"
    model_dir = "/home/xinyun/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A"
    robot_type = "SF_TRON1A"

    # 硬编码的策略模型路径（需要修改为实际路径）
    policy_model_path = "/path/to/your/policy.pt"  # 请更改为实际的策略模型路径
 
    # 配置
    class Sim2simCfg(XBotLCfg):
        # 新建的数据结构    
        class sim_config:
            # 使用硬编码的SF_TRON1A模型路径
            mujoco_model_path = model_path
            # 仿真时间120s
            sim_duration = 120.0
            # 1000HZ 
            dt = 0.001
            # 控制频率 100HZ
            decimation = 10
 
        class robot_config:
            # KP KD 和关节力矩限制
            kps = np.array([200, 200, 200, 200, 200, 200, 200, 200, 45, 45, 45, 45, 45, 45], dtype=np.double)
            kds = np.array([1.5, 1.5, 1.5, 0.8, 1.5, 1.5, 1.5, 0.8, 1.5, 1.5, 1.5, 0.8, 1.5, 1.5], dtype=np.double)
            tau_limit = 200. * np.ones(14, dtype=np.double)
 
    # 控制策略 （加载的模型）
    policy = torch.jit.load(policy_model_path)
    run_mujoco(policy, Sim2simCfg())
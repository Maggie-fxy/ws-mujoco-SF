import mujoco
import numpy as np
import time
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

class MuJoCoDirectController:
    def __init__(self, model_path, model_dir, robot_type):
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化渲染器（可选）
        self.renderer = mujoco.Renderer(self.model)
        
        # 加载配置和ONNX模型
        self.load_config(f'{model_dir}/{robot_type}/params.yaml')
        self.initialize_onnx_models(model_dir, robot_type)
        
        # 获取关节和传感器索引
        self.joint_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint",
            "J1", "J2", "J3", "J4", "J5", "J6"
        ]
        self.joint_num = len(self.joint_names)
        
        # 获取传感器ID
        self.imu_quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
        self.imu_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.imu_acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")
        
        # 初始化状态变量
        self.reset_state()
        
    def reset_state(self):
        """重置机器人状态到初始位置"""
        # 设置初始关节位置
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.data.qpos[joint_id] = self.init_joint_angles[i]
        
        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)
        
    def get_robot_state(self):
        """直接从MuJoCo获取机器人状态"""
        joint_positions = []
        joint_velocities = []
        joint_torques = []
        
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                joint_positions.append(self.data.qpos[joint_id])
                joint_velocities.append(self.data.qvel[joint_id])
                joint_torques.append(self.data.ctrl[joint_id])
        
        return np.array(joint_positions), np.array(joint_velocities), np.array(joint_torques)
    
    def get_imu_data(self):
        """直接从MuJoCo传感器获取IMU数据"""
        # 获取四元数
        if self.imu_quat_id >= 0:
            quat_start = self.model.sensor_adr[self.imu_quat_id]
            quat = self.data.sensordata[quat_start:quat_start+4]
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            
        # 获取角速度
        if self.imu_gyro_id >= 0:
            gyro_start = self.model.sensor_adr[self.imu_gyro_id]
            gyro = self.data.sensordata[gyro_start:gyro_start+3]
        else:
            gyro = np.array([0.0, 0.0, 0.0])
            
        # 获取加速度
        if self.imu_acc_id >= 0:
            acc_start = self.model.sensor_adr[self.imu_acc_id]
            acc = self.data.sensordata[acc_start:acc_start+3]
        else:
            acc = np.array([0.0, 0.0, -9.81])
            
        return quat, gyro, acc
    
    def set_joint_commands(self, positions, velocities, torques, kp, kd):
        """设置关节控制命令"""
        joint_pos, joint_vel, _ = self.get_robot_state()
        
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0 and i < len(positions):
                # PD控制
                pos_error = positions[i] - joint_pos[i]
                vel_error = velocities[i] - joint_vel[i]
                control_torque = kp[i] * pos_error + kd[i] * vel_error + torques[i]
                
                # 设置控制力矩
                self.data.ctrl[joint_id] = control_torque
    
    def compute_observation(self):
        """计算观测向量"""
        # 获取IMU数据
        quat, gyro, acc = self.get_imu_data()
        
        # 获取关节状态
        joint_pos, joint_vel, _ = self.get_robot_state()
        
        # 计算重力投影
        imu_orientation = quat
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()
        gravity_vector = np.array([0, 0, -1])
        projected_gravity = np.dot(inverse_rot, gravity_vector)
        
        # 应用IMU偏移校正
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()
        base_ang_vel = np.dot(rot, gyro)
        projected_gravity = np.dot(rot, projected_gravity)
        
        # 步态信息
        gait = np.array([self.gait_frequencies, 0.5, 0.5, self.gait_swing_height])
        self.gait_index += 0.02 * gait[0]
        if self.gait_index > 1.0:
            self.gait_index = 0.0
        gait_clock = np.array([np.sin(self.gait_index * 2 * np.pi), 
                              np.cos(self.gait_index * 2 * np.pi)])
        
        # 构建观测向量
        joint_pos_input = (joint_pos - self.init_joint_angles) * self.obs_scales['dof_pos']
        
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],
            projected_gravity,
            joint_pos_input,
            joint_vel * self.obs_scales['dof_vel'],
            self.last_actions,
            gait_clock,
            gait
        ])
        
        return obs
    
    def run_control_loop(self, duration=60.0, render=True):
        """运行控制循环"""
        dt = self.model.opt.timestep
        steps_per_control = int(0.02 / dt)  # 50Hz控制频率
        
        start_time = time.time()
        step_count = 0
        
        # 初始化命令
        self.commands = np.zeros(5)  # [vx, vy, wz, 0, 0]
        self.last_actions = np.zeros(self.joint_num)
        
        while time.time() - start_time < duration:
            # 每个控制周期执行一次策略计算
            if step_count % steps_per_control == 0:
                # 计算观测
                obs = self.compute_observation()
                
                # 更新历史缓冲区
                self.update_history_buffer(obs)
                
                # 计算编码器输出
                self.compute_encoder()
                
                # 计算动作
                actions = self.compute_actions(obs)
                
                # 限制动作范围
                actions = np.clip(actions, -1.0, 1.0)
                self.last_actions = actions
                
                # 转换为关节命令
                joint_positions = self.init_joint_angles + actions * self.action_scale
                joint_velocities = np.zeros(self.joint_num)
                joint_torques = np.zeros(self.joint_num)
                kp = np.full(self.joint_num, self.control_cfg['stiffness'])
                kd = np.full(self.joint_num, self.control_cfg['damping'])
                
                # 设置控制命令
                self.set_joint_commands(joint_positions, joint_velocities, 
                                      joint_torques, kp, kd)
            
            # 执行物理仿真步
            mujoco.mj_step(self.model, self.data)
            
            # 渲染（可选）
            if render and step_count % 10 == 0:  # 降低渲染频率
                self.renderer.update_scene(self.data)
                self.renderer.render()
            
            step_count += 1
            
            # 控制仿真速度
            time.sleep(max(0, dt - (time.time() % dt)))

# use
if __name__ == "__main__":
    model_path = "/home/xinyun/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A/xml/robot_with_arm.xml"
    model_dir = "/home/xinyun/limx_ws/pointfoot-mujoco-sim/robot-description/pointfoot/SF_TRON1A"
    robot_type = "SF_TRON1A"
    
    controller = MuJoCoDirectController(model_path, model_dir, robot_type)
    controller.run_control_loop(duration=30.0, render=True)
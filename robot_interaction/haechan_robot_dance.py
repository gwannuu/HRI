import time
import numpy as np
import mujoco
import mujoco.viewer
import pickle
import subprocess

from simulated_robot import SimulatedRobot
from robot import Robot

# Function to load .pkl file
def load_motion(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    motion = data.get('full_pose')
    return motion

# 로봇 명령을 보내는 함수
def robot_control(motion_data):

    # while not stop_event.is_set():
    while True:
        # if pause_event.is_set():
            # 일시 중지 상태에서는 대기
            # time.sleep(0.1)
            # continue
        
        # Read the current position from the real robot
        pwm = real_robot.read_position()
        pwm = np.array(pwm)

        # 로봇 동작 완료를 위한 카운터 또는 조건 설정
        step_count = 0
        max_steps = 900  # 로봇이 동작할 총 스텝 수

        # subprocess.run(["open", music_path])
        for k in range(max_steps):

            target_pwm = sim_robot._pos2pwm(motion_data[k])
            target_pwm = np.array(target_pwm)
            print (target_pwm)

            # Smoothly interpolate between the current and target positions
            num_iter =2
            sleep_time = 0.013
            smooth_mover = np.linspace(pwm, target_pwm, num_iter)
            for i in range(num_iter):
                # 30fps에 맞추기 위해 대기 시간 설정
                time.sleep(sleep_time)
                intermediate_pwm = smooth_mover[i]
                real_robot.set_goal_pos([int(pos) for pos in intermediate_pwm])

                # Update the simulation with the intermediate positions
                mujoco_data.qpos[:6] = sim_robot._pwm2pos(intermediate_pwm)

            step_count += 1
            print(f"Step: {step_count}")

            pwm = target_pwm

        # 로봇 동작 완료 확인
        # 로봇의 제어가 끝나면 stop_event가 설정되어 루프를 빠져나와야함
        if step_count >= max_steps:
            print("Robot motion completed.")
            exit()
            # robot_done_event.set()  # 로봇 동작 완료 이벤트 설정

        break  # 루프 종료


if __name__=="__main__":
    
    mujoco_model = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
    mujoco_data = mujoco.MjData(mujoco_model)

    # Initialize simulated and real robots
    sim_robot = SimulatedRobot(mujoco_model, mujoco_data)
    real_robot = Robot(device_name='COM3')

    # Set the robot to position control mode
    real_robot._set_position_control()
    real_robot._enable_torque()

    # Read initial PWM positions from the real robot
    pwm = real_robot.read_position()
    pwm = np.array(pwm)
    mujoco_data.qpos[:6] = sim_robot._pwm2pos(pwm)
    mujoco_data.qpos[1] = -sim_robot._pwm2pos(pwm[1])
    
    pkl_file = 'motions/APT/6_joints_robot/test_7.pkl'
    # music_path = './test_7.mp4'
    music_path = "music/Papa Nugs - Hyperdrive.mp3"
    motion_data = load_motion(pkl_file)
    
    # Set up events for controlling the robot
    robot_control(motion_data)
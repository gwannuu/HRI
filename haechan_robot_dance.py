import time
import numpy as np
import mujoco
import mujoco.viewer
import pickle

from interface import SimulatedRobot
from robot import Robot

# 로봇 명령을 보내는 함수
def robot_control(stop_event, pause_event, robot_done_event):

    m = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
    d = mujoco.MjData(m)

    # Initialize simulated and real robots
    r = SimulatedRobot(m, d)
    robot = Robot(device_name='/dev/tty.usbmodem58760436701')

    # Set the robot to position control mode
    robot._set_position_control()
    robot._enable_torque()

    # Read initial PWM positions from the real robot
    pwm = robot.read_position()
    pwm = np.array(pwm)
    d.qpos[:6] = r._pwm2pos(pwm)
    d.qpos[1] = -r._pwm2pos(pwm[1])

    # Function to load .pkl file
    def load_motion(pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        motion = data.get('full_pose')
        return motion
    
    pkl_file = './test_7.pkl'
    # video_path = './test_7.mp4'
    motion_data = load_motion(pkl_file)

    while not stop_event.is_set():
        if pause_event.is_set():
            # 일시 중지 상태에서는 대기
            time.sleep(0.1)
            continue

    # Launch the Mujoco viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():

            # Read the current position from the real robot
            pwm = robot.read_position()
            pwm = np.array(pwm)

            # Update the simulation with the real robot's current position
            d.qpos[:6] = r._pwm2pos(pwm)
            mujoco.mj_forward(m, d)

            # 로봇 동작 완료를 위한 카운터 또는 조건 설정
            step_count = 0
            max_steps = 900  # 로봇이 동작할 총 스텝 수

            # subprocess.run(["open", video_path])
            for k in range(max_steps):

                target_pwm = r._pos2pwm(motion_data[k])
                target_pwm = np.array(target_pwm)
                print (target_pwm)

                # Smoothly interpolate between the current and target positions
                smooth_mover = np.linspace(pwm, target_pwm, 2)
                for i in range(2):
                    # 30fps에 맞추기 위해 대기 시간 설정
                    time.sleep(0.013)
                    intermediate_pwm = smooth_mover[i]
                    robot.set_goal_pos([int(pos) for pos in intermediate_pwm])

                    # Update the simulation with the intermediate positions
                    d.qpos[:6] = r._pwm2pos(intermediate_pwm)
                    mujoco.mj_forward(m, d)
                    viewer.sync()

                step_count += 1
                print(f"Step: {step_count}")

                pwm = target_pwm

            # 로봇 동작 완료 확인
            # 로봇의 제어가 끝나면 stop_event가 설정되어 루프를 빠져나와야함
            if step_count >= max_steps:
                print("Robot motion completed.")
                robot_done_event.set()  # 로봇 동작 완료 이벤트 설정
                break  # 루프 종료

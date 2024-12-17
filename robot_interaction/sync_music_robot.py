import cv2
import numpy as np
import threading
import time
from queue import Queue
import pygame
import mujoco

# pygame 초기화
pygame.mixer.init()

# 음악 재생 클래스
class MusicPlayer:
    def __init__(self, filename):
        self.filename = filename
        self.is_paused = False

    def play(self):
        if not pygame.mixer.music.get_busy() and not self.is_paused:
            pygame.mixer.music.load(self.filename)
            pygame.mixer.music.play()
        elif self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False

    def pause(self):
        if pygame.mixer.music.get_busy() and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True

    def stop(self):
        pygame.mixer.music.stop()
        self.is_paused = False

# 로봇 명령을 보내는 함수
def robot_control(stop_event, pause_event, robot_done_event):
    # Mujoco 시뮬레이션 초기화, 혹은 외부에서 모델 선언 후 받아오기
    # model = mujoco.MjModel.from_xml_path('./low_cost_robot/scene.xml')
    # data = mujoco.MjData(model)

    # 로봇 동작 완료를 위한 카운터 또는 조건 설정
    step_count = 0
    max_steps = 900  # 로봇이 동작할 총 스텝 수

    while not stop_event.is_set():
        if pause_event.is_set():
            # 일시 중지 상태에서는 대기
            time.sleep(0.1)
            continue


        # 시뮬레이션 스텝 진행
        # mujoco.mj_step(model, data)
        # 또는 mujoco_py의 경우:
        # sim.step()

        # 스텝 카운트 증가
        step_count += 1
        print(f"Step: {step_count}")

        # 로봇 동작 완료 확인
        # 로봇의 제어가 끝나면 stop_event가 설정되어 루프를 빠져나와야함
        if step_count >= max_steps:
            print("Robot motion completed.")
            robot_done_event.set()  # 로봇 동작 완료 이벤트 설정
            break  # 루프 종료

        # 30fps에 맞추기 위해 대기 시간 설정 노가다 필요
        time.sleep(1/30)


# 카메라로부터 프레임을 읽어오는 함수 (스레드)
def camera_capture(frame_queue, stop_event):
    cap = cv2.VideoCapture(1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
        else:
            break
    cap.release()

# 임계값을 확인하는 함수
def check_threshold(frame_buffer: List[np.ndarray],
                   diff_buffer: List[np.ndarray],
                   threshold: float,
                   max_frames: int) -> bool:
    """Check if frame differences exceed threshold.
    
    Args:
        frame_buffer: Buffer of recent frames
        diff_buffer: Buffer of frame differences
        threshold: Threshold for movement detection
        max_frames: Maximum number of frames to consider
    
    Returns:
        bool: True if threshold is exceeded
    """
    movement_threshold = 10
    if len(frame_buffer) > 1:
        keypoints = frame_buffer[-1]
        prev_keypoints = frame_buffer[-2]
        moved = False
        if len(prev_keypoints) == len(keypoints):
            distances = np.sqrt(np.sum((keypoints - prev_keypoints) ** 2, axis=1))
            if np.any(distances > movement_threshold):
                moved = True


        # diff = cv2.absdiff(frame_buffer[-1], frame_buffer[-2])
        # mean_diff = np.mean(diff)
        
        if len(diff_buffer) >= max_frames - 1:
            diff_buffer.pop(0)
        diff_buffer.append(moved)

    if len(diff_buffer) == max_frames - 1:
        count = diff_buffer.count(True)
        return count > threshold
    else:
        return False

# 메인 실행 함수
def main():
    # 임계값 설정
    threshold = 2.0  # 임의의 쓰레숄드 값
    frame_buffer = []
    diff_buffer = []
    max_frames = 15
    loop_running = False

    # 이벤트 및 쓰레드 설정
    stop_event = threading.Event()
    pause_event = threading.Event()
    pause_event.set()  # 초기에는 일시 중지 상태로 설정
    camera_stop_event = threading.Event()
    frame_queue = Queue()
    pose_queue = Queue()
    # 로봇 동작 완료 이벤트 추가
    robot_done_event = threading.Event()

    # 음악 재생기 초기화
    music_player = MusicPlayer('music/Papa Nugs - Hyperdrive.mp3')  # 실제 음악 파일 경로로 변경하세요

    # 카메라 캡처 스레드 시작
    camera_thread = threading.Thread(target=camera_capture, args=(frame_queue, camera_stop_event))
    camera_thread.start()

    # 로봇 제어 스레드 시작
    robot_thread = threading.Thread(target=robot_control, args=(stop_event, pause_event, robot_done_event))
    robot_thread.start()

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                # 프레임을 그레이스케일로 변환
                                # 프레임을 표시
                cv2.imshow('Camera Feed', frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 버퍼에 프레임 추가
                if len(frame_buffer) >= max_frames:
                    frame_buffer.pop(0)
                frame_buffer.append(gray)

                # 임계값을 넘는지 확인
                threshold_exceeded = check_threshold(frame_buffer, diff_buffer, threshold, max_frames)

                if threshold_exceeded and not loop_running:
                    # 루프 시작: 음악 재생 및 로봇 제어 재개
                    loop_running = True
                    pause_event.clear()  # 로봇 제어 재개
                    music_player.play()   # 음악 재생 또는 재개
                    print("Loop started: Music and robot control running.")

                elif not threshold_exceeded and loop_running:
                    # 루프 중지: 음악 및 로봇 제어 일시 중지
                    loop_running = False
                    pause_event.set()    # 로봇 제어 일시 중지
                    music_player.pause() # 음악 일시 중지
                    print("Loop stopped: Music and robot control paused.")

            # 로봇 동작 완료 여부 확인
            if robot_done_event.is_set():
                print("Robot motion has completed.")
                loop_running = False
                music_player.stop()
                break  # 메인 루프 종료

            # 음악 재생 완료 여부 확인
            if not pygame.mixer.music.get_busy() and loop_running:
                print("Music playback has finished.")
                loop_running = False
                pause_event.set()  # 로봇 제어 일시 중지
                break  # 메인 루프 종료

            # ESC 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # 종료 시 자원 해제
        camera_stop_event.set()
        camera_thread.join()
        music_player.stop()
        stop_event.set()
        robot_thread.join()
        cv2.destroyAllWindows()

    print("Program terminated.")

if __name__ == "__main__":
    main()
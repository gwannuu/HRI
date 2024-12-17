from robot_interaction.log import logger
from robot_interaction.util import ComponentStateManager, DanceSystemConfig, performance_monitor


import cv2


import threading
from queue import Full, Queue

from robot_interaction.music_player import generate_buzzer_sound
from ultralytics import YOLO

@performance_monitor
def camera_capture(frame_queue: Queue,
                  pose_queue: Queue,
                  stop_event: threading.Event,
                  status_monitor: ComponentStateManager) -> None:
    """Capture frames from camera and add to queue.

    Args:
        frame_queue: Queue for storing captured frames
        stop_event: Event to signal thread termination
        status_monitor: System status monitor
    """
    try:
        model = YOLO('yolov8n-pose.pt')

        cap = cv2.VideoCapture(DanceSystemConfig.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return

        status_monitor.update_status('camera', 'running')
        generate_buzzer_sound()
        

        while not stop_event.is_set():
            ret, frame = cap.read()
            results = model(frame, verbose=False)
            keypoints = []

            for result in results:
                # keypoints 정보 추출
                kpts = result.keypoints.xy.cpu().numpy() if result.keypoints else None
                if kpts is not None and len(kpts) > 0:
                    keypoints = kpts[0]  # 첫 번째 사람의 keypoints만 사용

            if not ret:
                logger.warning("Failed to capture frame")
                continue

            try:
                frame_queue.put(frame, timeout=DanceSystemConfig.QUEUE_TIMEOUT)
                pose_queue.put(keypoints, timeout=DanceSystemConfig.QUEUE_TIMEOUT)
            except Full:
                frame_queue.get()  # Remove oldest frame
                frame_queue.put(frame)
                pose_queue.get()  # Remove oldest frame
                pose_queue.put(keypoints)


    except Exception as e:
        logger.error(f"Camera capture error: {e}")
    finally:
        cap.release()
        status_monitor.update_status('camera', 'stopped')
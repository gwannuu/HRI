import cv2
import numpy as np
import threading
import time
from queue import Queue, Full
import pygame
import mutagen
import mujoco
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
from contextlib import contextmanager
from time import perf_counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """System configuration parameters."""
    THRESHOLD: float = 10.0
    MAX_FRAMES: int = 15
    FPS: int = 30
    MAX_ROBOT_STEPS: int = 900
    CAMERA_INDEX: int = 0
    MUSIC_PATH: str = 'music/Papa Nugs - Hyperdrive.mp3'
    FRAME_QUEUE_SIZE: int = 100
    QUEUE_TIMEOUT: float = 1.0
    THREAD_JOIN_TIMEOUT: float = 5.0
    
# System Status and Performance Monitoring:
class SystemStatus:
    """Monitors and maintains system component status."""
    
    def __init__(self):
        self._status: Dict[str, str] = {
            'robot': 'idle',
            'music': 'stopped',
            'camera': 'inactive'
        }
        self._lock = threading.Lock()

    def update_status(self, component: str, status: str) -> None:
        """Update the status of a system component."""
        with self._lock:
            if component in self._status:
                self._status[component] = status
                logger.info(f"{component} status updated to: {status}")
            else:
                logger.warning(f"Unknown component: {component}")

    def get_status(self, component: str) -> str:
        """Get the current status of a system component."""
        with self._lock:
            return self._status.get(component, 'unknown')

def performance_monitor(func):
    """Decorator for monitoring function performance."""
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - start
        logger.debug(f"{func.__name__} took {duration:.3f} seconds")
        return result
    return wrapper

# Music Player:
class MusicPlayer:
    """Handles music playback using pygame mixer."""

    def __init__(self, filename: str):
        """Initialize the music player.
        
        Args:
            filename: Path to the music file
        """
        self.filename = filename
        self.is_paused = False
        pygame.mixer.init()
        logger.info("Music player initialized")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        pygame.mixer.quit()

    def play(self) -> None:
        """Start or resume music playback."""
        try:
            if not pygame.mixer.music.get_busy() and not self.is_paused:
                pygame.mixer.music.load(self.filename)
                pygame.mixer.music.play()
                logger.info("Music started")
            elif self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                logger.info("Music resumed")
        except Exception as e:
            logger.error(f"Error in music playback: {e}")

    def pause(self) -> None:
        """Pause music playback."""
        try:
            if pygame.mixer.music.get_busy() and not self.is_paused:
                pygame.mixer.music.pause()
                self.is_paused = True
                logger.info("Music paused")
        except Exception as e:
            logger.error(f"Error pausing music: {e}")

    def stop(self) -> None:
        """Stop music playback."""
        try:
            pygame.mixer.music.stop()
            self.is_paused = False
            logger.info("Music stopped")
        except Exception as e:
            logger.error(f"Error stopping music: {e}")
            

@performance_monitor
def robot_control(stop_event: threading.Event, 
                 pause_event: threading.Event,
                 robot_done_event: threading.Event,
                 status_monitor: SystemStatus) -> None:
    """Control robot movements and simulation.
    
    Args:
        stop_event: Event to signal thread termination
        pause_event: Event to signal pause/resume
        robot_done_event: Event to signal robot task completion
        status_monitor: System status monitor
    """
    try:
        step_count = 0
        status_monitor.update_status('robot', 'running')

        while not stop_event.is_set():
            if pause_event.is_set():
                status_monitor.update_status('robot', 'paused')
                time.sleep(1/Config.FPS)
                continue

            # Simulation step would go here
            step_count += 1
            logger.debug(f"Robot step: {step_count}")

            if step_count >= Config.MAX_ROBOT_STEPS:
                logger.info("Robot motion completed")
                robot_done_event.set()
                break

            time.sleep(1/Config.FPS)

    except Exception as e:
        logger.error(f"Error in robot control: {e}")
    finally:
        status_monitor.update_status('robot', 'stopped')

@performance_monitor
def camera_capture(frame_queue: Queue, 
                  stop_event: threading.Event,
                  status_monitor: SystemStatus) -> None:
    """Capture frames from camera and add to queue.
    
    Args:
        frame_queue: Queue for storing captured frames
        stop_event: Event to signal thread termination
        status_monitor: System status monitor
    """
    try:
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return

        status_monitor.update_status('camera', 'running')

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue

            try:
                frame_queue.put(frame, timeout=Config.QUEUE_TIMEOUT)
            except Full:
                frame_queue.get()  # Remove oldest frame
                frame_queue.put(frame)

    except Exception as e:
        logger.error(f"Camera capture error: {e}")
    finally:
        cap.release()
        status_monitor.update_status('camera', 'stopped')
        
        
@performance_monitor
def check_threshold(frame_buffer: List[np.ndarray],
                   diff_buffer: List[float],
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
    if len(frame_buffer) > 1:
        diff = cv2.absdiff(frame_buffer[-1], frame_buffer[-2])
        mean_diff = np.mean(diff)
        
        if len(diff_buffer) >= max_frames - 1:
            diff_buffer.pop(0)
        diff_buffer.append(mean_diff)

    if len(diff_buffer) == max_frames - 1:
        avg_change = np.mean(diff_buffer)
        return avg_change > threshold
    return False

class SystemController:
    """Main system controller class."""

    def __init__(self):
        self.status_monitor = SystemStatus()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.camera_stop_event = threading.Event()
        self.robot_done_event = threading.Event()
        self.frame_queue = Queue(maxsize=Config.FRAME_QUEUE_SIZE)
        
        self.frame_buffer = []
        self.diff_buffer = []
        self.loop_running = False

    def initialize_threads(self):
        """Initialize and start system threads."""
        self.camera_thread = threading.Thread(
            target=camera_capture,
            args=(self.frame_queue, self.camera_stop_event, self.status_monitor)
        )
        
        self.robot_thread = threading.Thread(
            target=robot_control,
            args=(self.stop_event, self.pause_event, self.robot_done_event, 
                  self.status_monitor)
        )

        self.pause_event.set()  # Start paused
        self.camera_thread.start()
        self.robot_thread.start()

    def shutdown(self):
        """Gracefully shut down all system components."""
        logger.info("Initiating system shutdown")
        self.camera_stop_event.set()
        self.stop_event.set()
        
        self.camera_thread.join(timeout=Config.THREAD_JOIN_TIMEOUT)
        self.robot_thread.join(timeout=Config.THREAD_JOIN_TIMEOUT)
        
        cv2.destroyAllWindows()
        logger.info("System shutdown complete")

    def dancewithme(self,music_name:str):
        """Main system execution loop."""
        try:
            with MusicPlayer(Config.MUSIC_PATH) as music_player:
                self.initialize_threads()

                while True:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        if len(self.frame_buffer) >= Config.MAX_FRAMES:
                            self.frame_buffer.pop(0)
                        self.frame_buffer.append(gray)

                        threshold_exceeded = check_threshold(
                            self.frame_buffer,
                            self.diff_buffer,
                            Config.THRESHOLD,
                            Config.MAX_FRAMES
                        )

                        self.handle_threshold_state(threshold_exceeded, music_player)

                    if self.should_stop():
                        break

                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.shutdown()

    def handle_threshold_state(self, threshold_exceeded: bool, 
                             music_player: MusicPlayer) -> None:
        """Handle system state based on threshold detection."""
        if threshold_exceeded and not self.loop_running:
            self.loop_running = True
            self.pause_event.clear()
            music_player.play()
            logger.info("Loop started: Music and robot control running")

        elif not threshold_exceeded and self.loop_running:
            self.loop_running = False
            self.pause_event.set()
            music_player.pause()
            logger.info("Loop stopped: Music and robot control paused")

    def should_stop(self) -> bool:
        """Check if the system should stop."""
        if self.robot_done_event.is_set():
            logger.info("Robot motion has completed")
            return True

        if not pygame.mixer.music.get_busy() and self.loop_running:
            logger.info("Music playback has finished")
            self.pause_event.set()
            return True

        return False

def main():
    """Main entry point of the application."""
    logger.info("Starting system")
    controller = SystemController()
    controller.run()
    logger.info("System terminated")

if __name__ == "__main__":
    main()
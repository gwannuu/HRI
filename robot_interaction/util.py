from dataclasses import dataclass
from time import perf_counter

from robot_interaction.log import logger
from typing import Dict
import pickle

import threading

@dataclass
class DanceSystemConfig:
    """System configuration parameters."""
    THRESHOLD: float = 10.0
    MAX_FRAMES: int = 15
    FPS: int = 30
    MAX_ROBOT_STEPS: int = 900
    CAMERA_INDEX: int = 0
    CAMERA_ON: bool = False
    FRAME_QUEUE_SIZE: int = 100
    QUEUE_TIMEOUT: float = 1.0
    THREAD_JOIN_TIMEOUT: float = 5.0
    PORT_NAME = "COM3"
    # PORT_NAME: str = "/dev/tty.usbmodem58A60700081"

# System Status and Performance Monitoring:
class ComponentStateManager:
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


# Function to load .pkl file
def load_motion(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    motion = data.get('full_pose')
    return motion
import cv2
import numpy as np
import threading
import time
from queue import Queue
import pygame
import mujoco
import logging
from typing import List
import pickle
from robot_interaction.camera_capture import camera_capture
from robot_interaction.log import logger
from robot_interaction.music_player import MusicPlayer
from robot_interaction.util import ComponentStateManager, DanceSystemConfig, load_motion, performance_monitor
from simulated_robot import SimulatedRobot
from robot import Robot

@performance_monitor
def robot_control(stop_event: threading.Event, 
                 pause_event: threading.Event,
                 robot_done_event: threading.Event,
                 status_monitor: ComponentStateManager,
                 motion_data: pickle,
                 real_robot: Robot,
                 sim_robot: SimulatedRobot,
                 mujoco_data:mujoco.MjData) -> None:
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
        max_steps = 900
        pwm = real_robot.read_position()
        pwm = np.array(pwm)
        for k in range(max_steps):
            if stop_event.is_set():
                exit()
            
        # while not stop_event.is_set():
            if pause_event.is_set():
                status_monitor.update_status('robot', 'paused')
                time.sleep(1/DanceSystemConfig.FPS)
                continue
            

    
        # subprocess.run(["open", music_path])


            target_pwm = sim_robot._pos2pwm(motion_data[k])
            target_pwm = np.array(target_pwm)
            print (target_pwm)

            # Smoothly interpolate between the current and target positions
            smooth_mover = np.linspace(pwm, target_pwm, 2)
            for i in range(2):
                # 30fps에 맞추기 위해 대기 시간 설정
                time.sleep(0.013)
                intermediate_pwm = smooth_mover[i]
                real_robot.set_goal_pos([int(pos) for pos in intermediate_pwm])

                # Update the simulation with the intermediate positions
                mujoco_data.qpos[:6] = sim_robot._pwm2pos(intermediate_pwm)

            step_count += 1
            print(f"Step: {step_count}")
            logger.debug(f"Robot step: {step_count}")


            pwm = target_pwm
            
            # if step_count >= DanceSystemConfig.MAX_ROBOT_STEPS:
            #     logger.info("Robot motion completed")
            #     robot_done_event.set()
            #     break

    except Exception as e:
        logger.error(f"Error in robot control: {e}")
    finally:
        status_monitor.update_status('robot', 'stopped')

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

class DanceInteractionSystem:
    """Main system controller class."""

    def __init__(self, music_path: str = None, dance_path: str = None):
        self.status_monitor = ComponentStateManager()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.camera_stop_event = threading.Event()
        self.robot_done_event = threading.Event()
        self.frame_queue = Queue(maxsize=DanceSystemConfig.FRAME_QUEUE_SIZE)
        
        self.frame_buffer = []
        self.diff_buffer = []
        self.loop_running = False
        
        self.music_path = music_path
        self.motion_data = load_motion(dance_path)
        
        #robot_setting
        self.mujoco_model = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        # Initialize simulated and real robots
        self.sim_robot = SimulatedRobot(self.mujoco_model,self.mujoco_data)
        self.real_robot = Robot(device_name=DanceSystemConfig.PORT_NAME)
        self._initialize_robot()


    def _initialize_robot(self):
        # Set the robot to position control mode
        self.real_robot._set_position_control()
        self.real_robot._enable_torque()

        # Read initial PWM positions from the real robot
        pwm = self.real_robot.read_position()
        pwm = np.array(pwm)
        self.mujoco_data.qpos[:6] = self.sim_robot._pwm2pos(pwm)
        self.mujoco_data.qpos[1] = -self.sim_robot._pwm2pos(pwm[1])

    def initialize_threads(self):
        """Initialize and start system threads."""
        self.camera_thread = threading.Thread(
            target=camera_capture,
            args=(self.frame_queue, self.camera_stop_event, self.status_monitor)
        )
        
        self.robot_thread = threading.Thread(
            target=robot_control,
            args=(self.stop_event, self.pause_event, self.robot_done_event, 
                  self.status_monitor, self.motion_data,self.real_robot, self.sim_robot,self.mujoco_data)
        )

        self.pause_event.set()  # Start paused
        self.camera_thread.start()
        self.robot_thread.start()

    def shutdown(self):
        """Gracefully shut down all system components."""
        logger.info("Initiating system shutdown")
        self.camera_stop_event.set()
        self.stop_event.set()
        
        self.camera_thread.join(timeout=DanceSystemConfig.THREAD_JOIN_TIMEOUT)
        self.robot_thread.join(timeout=DanceSystemConfig.THREAD_JOIN_TIMEOUT)
        
        
        cv2.destroyAllWindows()
        logger.info("System shutdown complete")

    def dancewithme(self):
        """Main system execution loop."""
        try:
            with MusicPlayer(self.music_path) as music_player:
                self.initialize_threads()

                while True:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        if len(self.frame_buffer) >= DanceSystemConfig.MAX_FRAMES:
                            self.frame_buffer.pop(0)
                        self.frame_buffer.append(gray)

                        threshold_exceeded = check_threshold(
                            self.frame_buffer,
                            self.diff_buffer,
                            DanceSystemConfig.THRESHOLD,
                            DanceSystemConfig.MAX_FRAMES
                        )

                        self.handle_threshold_state(threshold_exceeded, music_player)

                    if self.should_stop():
                        self.shutdown()
                        break

                    if cv2.waitKey(1) & 0xFF == 27:
                        self.shutdown()
                        break

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.shutdown()
    
    def danceforme(self):
        """Main system execution loop."""
        try:
            with MusicPlayer(self.music_path) as music_player:
                self.initialize_threads()

                while True:
                    # if not self.frame_queue.empty():
                    #     frame = self.frame_queue.get()
                    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    #     if len(self.frame_buffer) >= DanceSystemConfig.MAX_FRAMES:
                    #         self.frame_buffer.pop(0)
                    #     self.frame_buffer.append(gray)

                    #     threshold_exceeded = check_threshold(
                    #         self.frame_buffer,
                    #         self.diff_buffer,
                    #         DanceSystemConfig.THRESHOLD,
                    #         DanceSystemConfig.MAX_FRAMES
                    #     )

                    self.handle_threshold_state(True, music_player)

                    if self.should_stop():
                        self.shutdown()
                        music_player.stop()
                        break

                    if cv2.waitKey(1) & 0xFF == 27:
                        self.shutdown()
                        music_player.stop()
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

def run_robot(music_path: str, dance_path: str, withme: bool):
    """Main entry point of the application."""
    logger.info("Starting system")
    controller = DanceInteractionSystem(
        music_path=music_path,
        dance_path=dance_path,
    )
    if withme:
        controller.dancewithme()
    else:
        controller.danceforme()
    logger.info("System terminated")

if __name__ == "__main__":
    run_robot(
        music_path = "music/Zedd & Alessia Cara_Stay.mp3",
        dance_path = "motions/APT/test_1.pkl",
        withme=True,
    )

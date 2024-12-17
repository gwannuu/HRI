from robot_interaction.log import logger


import pygame
            
import numpy as np
import sounddevice as sd
import time


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
            


def generate_buzzer_sound(frequency=1000, duration=1.0, volume=0.5, sample_rate=44100):
    """
    부저음을 생성하고 재생하는 함수
    
    Parameters:
    - frequency: 소리의 주파수 (기본값: 1000 Hz)
    - duration: 소리 재생 시간 (초, 기본값: 0.5초)
    - volume: 소리 볼륨 (0.0 ~ 1.0, 기본값: 0.5)
    - sample_rate: 샘플링 레이트 (기본값: 44100 Hz)
    """
    # 시간 배열 생성
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 사인파 생성 (기본 부저음)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    
    # # 추가적인 효과: 삼각파 혼합 (부저음 질감 추가)
    # triangle_wave = volume * 0.5 * signal.sawtooth(2 * np.pi * frequency * t)
    # mixed_tone = tone + triangle_wave
    
    # 사운드 재생
    sd.play(tone, sample_rate)
    sd.wait()
from robot_interaction.log import logger


import pygame


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
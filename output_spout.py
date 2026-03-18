import numpy as np

try:
    import pygame
    from pygame.locals import DOUBLEBUF, NOFRAME, OPENGL
    import SpoutGL

    SPOUT_AVAILABLE = True
except ImportError:
    SPOUT_AVAILABLE = False


class SpoutOutput:
    def __init__(self, name: str, width: int, height: int):
        self.enabled = False
        self.width = width
        self.height = height

        if not SPOUT_AVAILABLE:
            print("[Spout] SpoutGL or pygame not installed. Spout output disabled.")
            return

        # Create a small offscreen pygame window for the OpenGL context
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = '-10000,-10000'
        pygame.init()
        pygame.display.set_mode((1, 1), DOUBLEBUF | OPENGL | NOFRAME)

        self.sender = SpoutGL.SpoutSender()
        self.sender.setSenderName(name)
        self.enabled = True
        print(f"[Spout] Sender '{name}' created ({width}x{height}).")

    def send(self, frame_bgra: np.ndarray):
        if not self.enabled:
            return
        self.sender.sendImage(
            frame_bgra,
            self.width,
            self.height,
            SpoutGL.enums.GL_BGRA_EXT,
            False,
            0,
        )

    def release(self):
        if self.enabled:
            self.sender.releaseSender()
            pygame.quit()
            self.enabled = False

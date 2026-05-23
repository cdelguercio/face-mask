import numpy as np

try:
    import pygame
    from pygame.locals import DOUBLEBUF, NOFRAME, OPENGL
    import SpoutGL

    SPOUT_AVAILABLE = True
except ImportError:
    SPOUT_AVAILABLE = False


def _print_gl_diagnostics():
    """Print the GL renderer / vendor pygame ended up with.

    Useful when Spout silently fails — confirms which GPU adapter the
    pygame OpenGL context is bound to. Spout requires sender and receiver
    to be on the same adapter, so if this prints "Intel" but Resolume is
    on the NVIDIA, that's the bug.
    """
    try:
        from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
        vendor = glGetString(GL_VENDOR)
        renderer = glGetString(GL_RENDERER)
        version = glGetString(GL_VERSION)

        def _decode(b):
            if b is None:
                return "<null>"
            return b.decode("utf-8", errors="replace") if isinstance(b, bytes) else str(b)

        print(f"[Spout]   GL_VENDOR  : {_decode(vendor)}")
        print(f"[Spout]   GL_RENDERER: {_decode(renderer)}")
        print(f"[Spout]   GL_VERSION : {_decode(version)}")
    except Exception as e:
        print(f"[Spout]   GL diagnostics unavailable: {e}")


class SpoutOutput:
    def __init__(self, name: str, width: int, height: int):
        self.enabled = False
        self.width = width
        self.height = height
        self.name = name
        self._frames_sent = 0
        self._frames_failed = 0
        self._last_report_at = 0

        if not SPOUT_AVAILABLE:
            print("[Spout] SpoutGL or pygame not installed. Spout output disabled.")
            return

        # Create a small offscreen pygame window for the OpenGL context.
        # The context becomes the current GL context for this thread,
        # which is what SpoutGL.sendImage needs to upload textures.
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = '-10000,-10000'
        pygame.init()
        pygame.display.set_mode((1, 1), DOUBLEBUF | OPENGL | NOFRAME)

        print(f"[Spout] pygame {pygame.version.ver}, "
              f"SpoutGL {getattr(SpoutGL, '__version__', '?')}")
        _print_gl_diagnostics()

        self.sender = SpoutGL.SpoutSender()
        self.sender.setSenderName(name)
        self.enabled = True
        print(f"[Spout] Sender '{name}' registered at {width}x{height}.")
        print("[Spout] If Resolume shows the sender but texture stays 0x0:")
        print("[Spout]   - confirm GL_RENDERER above matches the GPU Resolume runs on")
        print("[Spout]   - in NVIDIA Control Panel, force python.exe to the discrete GPU")

    def send(self, frame_bgra: np.ndarray):
        if not self.enabled:
            return

        # Pump the pygame event queue so the GL context stays valid on Windows.
        # We intentionally do NOT use pygame.event.pump() — that was stealing
        # keyboard input. peek() pumps internally without dequeuing events.
        try:
            pygame.event.peek(pygame.NOEVENT)
        except Exception:
            pass

        ok = self.sender.sendImage(
            frame_bgra,
            self.width,
            self.height,
            SpoutGL.enums.GL_BGRA_EXT,
            False,
            0,
        )

        if ok:
            self._frames_sent += 1
        else:
            self._frames_failed += 1

        # Periodically report send/fail counts so silent failures become visible.
        self._last_report_at += 1
        if self._last_report_at >= 300:  # ~every 10s at 30fps
            total = self._frames_sent + self._frames_failed
            if self._frames_failed > 0:
                print(f"[Spout] {self._frames_sent}/{total} frames sent OK, "
                      f"{self._frames_failed} failed. "
                      f"Sender '{self.name}' may be invisible to receivers.")
            self._last_report_at = 0

    def release(self):
        if self.enabled:
            self.sender.releaseSender()
            pygame.quit()
            self.enabled = False

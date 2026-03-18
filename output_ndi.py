import numpy as np

try:
    import NDIlib as ndi

    NDI_AVAILABLE = True
except (ImportError, OSError):
    NDI_AVAILABLE = False


class NDIOutput:
    def __init__(self, name: str):
        self.enabled = False

        if not NDI_AVAILABLE:
            print("[NDI] ndi-python not installed. NDI output disabled.")
            return

        if not ndi.initialize():
            print("[NDI] NDI runtime not found. Install NDI Tools from ndi.video")
            return

        send_settings = ndi.SendCreate()
        send_settings.ndi_name = name
        self.ndi_send = ndi.send_create(send_settings)

        if self.ndi_send is None:
            print("[NDI] Failed to create NDI sender.")
            return

        self.video_frame = ndi.VideoFrameV2()
        self.video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
        self.enabled = True
        print(f"[NDI] Sender '{name}' created.")

    def send(self, frame_bgra: np.ndarray):
        if not self.enabled:
            return
        self.video_frame.data = frame_bgra
        self.video_frame.xres = frame_bgra.shape[1]
        self.video_frame.yres = frame_bgra.shape[0]
        ndi.send_send_video_v2(self.ndi_send, self.video_frame)

    def release(self):
        if self.enabled:
            ndi.send_destroy(self.ndi_send)
            ndi.destroy()
            self.enabled = False

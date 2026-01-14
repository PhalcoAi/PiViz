# piviz/core/exporter.py
"""
Export Manager
==============
Handles screenshots and video recording.
"""

import moderngl
import numpy as np
from PIL import Image
import time
import os
from typing import Optional

try:
    import imageio

    HAS_VIDEO_SUPPORT = True
except ImportError:
    HAS_VIDEO_SUPPORT = False
    print("Warning: 'imageio' not found. Video export disabled.")
    print("Install with: pip install imageio[ffmpeg]")


class Exporter:
    def __init__(self, ctx: moderngl.Context, window_size):
        self.ctx = ctx
        self.width, self.height = window_size
        self._recording = False
        self._video_writer = None
        self._frame_count = 0
        self._output_dir = "exports"

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def resize(self, width, height):
        self.width = width
        self.height = height

    def take_screenshot(self, filename: Optional[str] = None, include_ui: bool = False):
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self._output_dir, f"screenshot_{timestamp}.png")

        fbo = self.ctx.detect_framebuffer()
        pixels = fbo.read(components=3, alignment=1)
        img = Image.frombytes('RGB', fbo.size, pixels)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filename)
        print(f"Snapshot saved: {filename}")
        return filename

    def start_recording(self, filename: Optional[str] = None, fps: int = 60):
        if not HAS_VIDEO_SUPPORT:
            print("Video support unavailable.")
            return

        if self._recording:
            return

        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self._output_dir, f"video_{timestamp}.mp4")

        print(f"Recording started: {filename}")

        # FIX: macro_block_size=1 prevents warning about dimensions not divisible by 16
        self._video_writer = imageio.get_writer(
            filename,
            fps=fps,
            quality=9,
            macro_block_size=1
        )
        self._recording = True
        self._frame_count = 0

    def stop_recording(self):
        if not self._recording:
            return

        if self._video_writer:
            self._video_writer.close()

        print(f"Recording stopped. Captured {self._frame_count} frames.")
        self._recording = False
        self._video_writer = None

    def capture_frame(self):
        if not self._recording or not self._video_writer:
            return

        fbo = self.ctx.detect_framebuffer()
        pixels = fbo.read(components=3, alignment=1)

        # Convert to numpy array
        image = np.frombuffer(pixels, dtype='uint8').reshape((fbo.size[1], fbo.size[0], 3))
        image = np.flipud(image)

        self._video_writer.append_data(image)
        self._frame_count += 1

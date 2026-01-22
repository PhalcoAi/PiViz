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

        # Use screen framebuffer and current viewport
        # This ensures we capture exactly what is visible, handling resize/maximize correctly
        fbo = self.ctx.screen
        x, y, width, height = self.ctx.viewport

        pixels = fbo.read(viewport=(x, y, width, height), components=3, alignment=1)
        img = Image.frombytes('RGB', (width, height), pixels)
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

        # Use screen framebuffer and current viewport
        fbo = self.ctx.screen
        x, y, width, height = self.ctx.viewport

        pixels = fbo.read(viewport=(x, y, width, height), components=3, alignment=1)

        # Convert to numpy array
        image = np.frombuffer(pixels, dtype='uint8').reshape((height, width, 3))
        image = np.flipud(image)

        try:
            self._video_writer.append_data(image)
            self._frame_count += 1
        except ValueError:
            # Handle case where window size changed during recording
            # imageio expects constant frame size
            print("Warning: Window resized during recording. Stopping to prevent corruption.")
            self.stop_recording()

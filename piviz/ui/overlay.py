# piviz/ui/overlay.py
"""
Performance Overlay for PiViz
=============================

Modern HUD-style overlay displaying:
- FPS and frame time
- CPU/GPU statistics
- Memory usage
- Custom scene stats
"""

import imgui
import psutil
import numpy as np
import time
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from ..core.studio import PiVizStudio
    from ..core.theme import Theme

# Try to import GPU utilities (optional)
try:
    import GPUtil

    HAS_GPU_UTIL = True
except ImportError:
    HAS_GPU_UTIL = False


class PiVizOverlay:
    """
    Performance monitoring overlay.
    """

    def __init__(self, studio: 'PiVizStudio'):
        self.studio = studio
        self._theme: Optional['Theme'] = None
        self.scale = 1.0  # <--- NEW: Track UI scale

        # Performance history buffers
        self.fps_history = np.zeros(120, dtype=np.float32)
        self.frame_time_history = np.zeros(120, dtype=np.float32)
        self.cpu_history = np.zeros(60, dtype=np.float32)
        self.gpu_history = np.zeros(60, dtype=np.float32)

        # Stats cache
        self._frame_count = 0
        self._last_cpu_update = 0.0
        self._last_gpu_update = 0.0

        # Cached values
        self._cpu_percent = 0.0
        self._ram_used_gb = 0.0
        self._gpu_percent = 0.0
        self._gpu_temp = 0.0
        self._vram_used_mb = 0.0
        self._vram_percent = 0.0
        self._gpu_name = "N/A"

        # Scene stats (set by user scene)
        self.scene_stats: Dict[str, Any] = {}

        # Timing
        self._start_time = time.time()

        self._detect_gpu()

    # --- NEW METHOD: This was missing ---
    def set_scale(self, scale: float):
        """Update UI scale factor."""
        self.scale = scale

    # ------------------------------------

    def _detect_gpu(self):
        """Detect GPU on startup."""
        if HAS_GPU_UTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self._gpu_name = gpus[0].name
                    # Removed truncation so it fits dynamically
            except Exception:
                self._gpu_name = "Unknown"

    def set_theme(self, theme: 'Theme'):
        """Update colors from theme."""
        self._theme = theme

    def set_scene_stat(self, key: str, value: Any):
        """Set a custom scene statistic to display."""
        self.scene_stats[key] = value

    def clear_scene_stats(self):
        """Clear all scene statistics."""
        self.scene_stats.clear()

    def render(self):
        """Render the overlay."""
        self._update_stats()

        if self._theme is None:
            return

        io = imgui.get_io()
        accent = self._theme.accent
        text_dim = self._theme.text_secondary

        # === TOP-LEFT: Performance ===
        self._draw_performance_panel(io, accent, text_dim)

        # === TOP-RIGHT: System ===
        self._draw_system_panel(io, accent, text_dim)

        # === BOTTOM-LEFT: Scene Stats ===
        if self.scene_stats:
            self._draw_scene_panel(io, accent, text_dim)

    def _draw_performance_panel(self, io, accent, text_dim):
        """Draw FPS and timing panel."""
        # Scale dimensions
        padding = 15 * self.scale
        width = 260 * self.scale

        imgui.set_next_window_position(padding, padding)
        imgui.set_next_window_size(width, 0)

        flags = (imgui.WINDOW_NO_DECORATION |
                 imgui.WINDOW_NO_BACKGROUND |
                 imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        imgui.begin("##perf", flags=flags)

        # Title
        imgui.text_colored("PERFORMANCE", accent[0], accent[1], accent[2], 1.0)
        imgui.same_line(spacing=10 * self.scale)
        elapsed = time.time() - self._start_time
        imgui.text_colored(f"| {elapsed:.0f}s", text_dim[0], text_dim[1], text_dim[2], 1.0)
        imgui.spacing()

        # FPS
        fps = self.fps_history[-1]
        fps_color = self._get_fps_color(fps)
        imgui.text_colored(f"{fps:.0f}", *fps_color)
        imgui.same_line()
        imgui.text_colored("FPS", text_dim[0], text_dim[1], text_dim[2], 1.0)
        imgui.same_line(spacing=20 * self.scale)

        # Frame time
        frame_ms = self.frame_time_history[-1] * 1000
        imgui.text_colored(f"{frame_ms:.2f}", 0.9, 0.9, 0.9, 1.0)
        imgui.same_line()
        imgui.text_colored("ms", text_dim[0], text_dim[1], text_dim[2], 1.0)

        # FPS graph
        imgui.plot_lines("##fps_graph", self.fps_history,
                         scale_min=0, scale_max=max(144, float(np.max(self.fps_history)) * 1.1),
                         graph_size=(240 * self.scale, 35 * self.scale))

        # Min/Avg/Max
        valid = self.fps_history[self.fps_history > 0]
        if len(valid) > 0:
            fps_min, fps_avg, fps_max = np.min(valid), np.mean(valid), np.max(valid)
            imgui.text_colored(f"min {fps_min:.0f}", text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=12 * self.scale)
            imgui.text_colored(f"avg {fps_avg:.0f}", text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=12 * self.scale)
            imgui.text_colored(f"max {fps_max:.0f}", text_dim[0], text_dim[1], text_dim[2], 1.0)

        imgui.end()

    def _draw_system_panel(self, io, accent, text_dim):
        """Draw system resources panel."""
        # Calculate width dynamically based on GPU name length
        base_width = 205
        name_width = imgui.calc_text_size(self._gpu_name).x + 50
        panel_width = max(base_width, name_width) * self.scale
        padding = 15 * self.scale

        imgui.set_next_window_position(io.display_size.x - panel_width - padding, padding)
        imgui.set_next_window_size(panel_width, 0)

        flags = (imgui.WINDOW_NO_DECORATION |
                 imgui.WINDOW_NO_BACKGROUND |
                 imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        imgui.begin("##system", flags=flags)

        imgui.text_colored("SYSTEM", accent[0], accent[1], accent[2], 1.0)
        imgui.spacing()

        # CPU
        cpu_color = self._get_usage_color(self._cpu_percent)
        imgui.text_colored("CPU", text_dim[0], text_dim[1], text_dim[2], 1.0)
        imgui.same_line(spacing=8 * self.scale)
        imgui.text_colored(f"{self._cpu_percent:.0f}%", *cpu_color)
        imgui.same_line(spacing=15 * self.scale)

        # RAM
        imgui.text_colored("RAM", text_dim[0], text_dim[1], text_dim[2], 1.0)
        imgui.same_line(spacing=8 * self.scale)
        imgui.text_colored(f"{self._ram_used_gb:.1f}GB", 0.9, 0.9, 0.9, 1.0)

        # CPU graph
        imgui.plot_lines("##cpu_graph", self.cpu_history,
                         scale_min=0, scale_max=100,
                         graph_size=(panel_width - 15, 25 * self.scale))

        # GPU section
        if HAS_GPU_UTIL and self._gpu_name != "N/A":
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text_colored("GPU", accent[0], accent[1], accent[2], 1.0)
            imgui.same_line(spacing=8 * self.scale)
            imgui.text_colored(self._gpu_name, text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.spacing()

            # Load & Temp
            gpu_color = self._get_usage_color(self._gpu_percent)
            imgui.text_colored("Load", text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=5 * self.scale)
            imgui.text_colored(f"{self._gpu_percent:.0f}%", *gpu_color)
            imgui.same_line(spacing=12 * self.scale)

            temp_color = self._get_temp_color(self._gpu_temp)
            imgui.text_colored("Temp", text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=5 * self.scale)
            imgui.text_colored(f"{self._gpu_temp:.0f}°C", *temp_color)

            # VRAM
            imgui.text_colored("VRAM", text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=5 * self.scale)
            imgui.text_colored(f"{self._vram_used_mb:.0f}MB", 0.9, 0.9, 0.9, 1.0)

            imgui.plot_lines("##gpu_graph", self.gpu_history,
                             scale_min=0, scale_max=100,
                             graph_size=(panel_width - 15, 25 * self.scale))

        imgui.end()

    def _draw_scene_panel(self, io, accent, text_dim):
        """Draw custom scene statistics."""
        padding = 15 * self.scale
        # Adjust Y position based on scale
        imgui.set_next_window_position(padding, io.display_size.y - (100 * self.scale))
        imgui.set_next_window_size(220 * self.scale, 0)

        flags = (imgui.WINDOW_NO_DECORATION |
                 imgui.WINDOW_NO_BACKGROUND |
                 imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        imgui.begin("##scene_stats", flags=flags)

        imgui.text_colored("SCENE", accent[0], accent[1], accent[2], 1.0)
        imgui.spacing()

        for key, value in self.scene_stats.items():
            imgui.text_colored(str(key), text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=10 * self.scale)
            if isinstance(value, float):
                imgui.text_colored(f"{value:.2f}", 0.9, 0.9, 0.9, 1.0)
            elif isinstance(value, int):
                imgui.text_colored(f"{value:,}", 0.9, 0.9, 0.9, 1.0)
            else:
                imgui.text_colored(str(value), 0.9, 0.9, 0.9, 1.0)

        imgui.end()

    def _update_stats(self):
        """Update performance statistics."""
        self._frame_count += 1
        io = imgui.get_io()
        current_time = time.time()

        # FPS (every frame)
        self.fps_history = np.roll(self.fps_history, -1)
        self.fps_history[-1] = io.framerate

        self.frame_time_history = np.roll(self.frame_time_history, -1)
        self.frame_time_history[-1] = 1.0 / max(io.framerate, 0.001)

        # CPU/RAM (every 0.5s)
        if current_time - self._last_cpu_update > 0.5:
            self._last_cpu_update = current_time
            self._cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            self._ram_used_gb = mem.used / (1024 ** 3)

            self.cpu_history = np.roll(self.cpu_history, -1)
            self.cpu_history[-1] = self._cpu_percent

        # GPU (every 1s)
        if HAS_GPU_UTIL and current_time - self._last_gpu_update > 1.0:
            self._last_gpu_update = current_time
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self._gpu_percent = gpu.load * 100
                    self._gpu_temp = gpu.temperature or 0
                    self._vram_used_mb = gpu.memoryUsed
                    self._vram_percent = (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0

                    self.gpu_history = np.roll(self.gpu_history, -1)
                    self.gpu_history[-1] = self._gpu_percent
            except Exception:
                pass

    def _get_fps_color(self, fps):
        if fps >= 60:
            return (0.3, 1.0, 0.4, 1.0)
        elif fps >= 30:
            return (1.0, 0.8, 0.2, 1.0)
        else:
            return (1.0, 0.3, 0.3, 1.0)

    def _get_usage_color(self, percent):
        if percent < 50:
            return (0.3, 1.0, 0.4, 1.0)
        elif percent < 80:
            return (1.0, 0.8, 0.2, 1.0)
        else:
            return (1.0, 0.3, 0.3, 1.0)

    def _get_temp_color(self, temp):
        if temp < 60:
            return (0.3, 1.0, 0.4, 1.0)
        elif temp < 80:
            return (1.0, 0.8, 0.2, 1.0)
        else:
            return (1.0, 0.3, 0.3, 1.0)

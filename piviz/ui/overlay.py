"""
Performance Overlay for PiViz
==============================

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
from collections import deque
import threading

if TYPE_CHECKING:
    from ..core.studio import PiVizStudio
    from ..core.theme import Theme

# Try to import GPU utilities (optional)
try:
    import GPUtil
    HAS_GPU_UTIL = True
except ImportError:
    HAS_GPU_UTIL = False

# Shared status colors used by FPS, usage, and temperature indicators
_COLOR_OK   = (0.3, 1.0, 0.4, 1.0)
_COLOR_WARN = (1.0, 0.8, 0.2, 1.0)
_COLOR_CRIT = (1.0, 0.3, 0.3, 1.0)


class PiVizOverlay:
    """
    Performance monitoring overlay with accurate FPS measurement.
    """

    def __init__(self, studio: 'PiVizStudio'):
        self.studio = studio
        self._theme: Optional['Theme'] = None
        self.scale = 1.0

        # Use deque for O(1) append/popleft instead of np.roll
        self._history_size = 120
        self._frame_times = deque(maxlen=self._history_size)

        # For graph display (updated periodically, not every frame)
        self.fps_history = np.zeros(self._history_size, dtype=np.float32)
        self.frame_time_history = np.zeros(self._history_size, dtype=np.float32)
        self.cpu_history = np.zeros(60, dtype=np.float32)
        self.gpu_history = np.zeros(60, dtype=np.float32)

        # Circular buffer indices (avoid np.roll which is O(n))
        self._fps_idx = 0
        self._cpu_idx = 0
        self._gpu_idx = 0

        self._last_frame_time = time.perf_counter()
        self._frame_count = 0

        # Smoothed FPS with exponential moving average
        self._smoothed_fps = 60.0
        self._ema_alpha = 0.1  # Lower = smoother, higher = more responsive

        # Stats cache update intervals
        self._last_cpu_update = 0.0
        self._last_gpu_update = 0.0
        self._last_graph_update = 0.0

        # CPU update interval (reduced from 0.5s to 1s to save CPU cycles)
        self._cpu_update_interval = 1.0
        self._gpu_update_interval = 1.0
        self._graph_update_interval = 0.1  # 10 Hz

        # Cache for panel dimensions
        self._system_panel_width = None  # Will be calculated on first render
        self._system_panel_needs_remeasure = True
        self._last_gpu_name = ""

        self._cpu_percent = 0.0
        self._ram_used_gb = 0.0
        self._gpu_percent = 0.0
        self._gpu_temp = 0.0
        self._vram_used_mb = 0.0
        self._vram_percent = 0.0
        self._gpu_name = "N/A"

        self._display_fps = 60.0
        self._display_frame_ms = 16.67

        self.scene_stats: Dict[str, Any] = {}
        self._start_time = time.time()

        self._gpu_lock = threading.Lock()
        self._gpu_thread_running = False

        self._detect_gpu()

    def set_scale(self, scale: float):
        """Update UI scale factor."""
        self.scale = scale
        self._system_panel_needs_remeasure = True  # Remeasure on scale change

    def _detect_gpu(self):
        """Detect GPU on startup."""
        if HAS_GPU_UTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self._gpu_name = gpus[0].name
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

    def _push_hud_style(self):
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *self._theme.panel)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12 * self.scale, 10 * self.scale))

    def _pop_hud_style(self):
        imgui.pop_style_var(2)
        imgui.pop_style_color()

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
        padding = 15 * self.scale
        width = 260 * self.scale

        imgui.set_next_window_position(padding, padding)
        imgui.set_next_window_size(width, 0)

        flags = (imgui.WINDOW_NO_DECORATION |
                 imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        self._push_hud_style()
        imgui.begin("##perf", flags=flags)

        # Title
        imgui.text_colored("PERFORMANCE", accent[0], accent[1], accent[2], 1.0)
        imgui.same_line(spacing=10 * self.scale)
        elapsed = time.time() - self._start_time
        imgui.text_colored(f"| {elapsed:.0f}s", text_dim[0], text_dim[1], text_dim[2], 1.0)
        imgui.spacing()

        # FPS - use smoothed display value
        fps = self._display_fps
        fps_color = self._get_fps_color(fps)
        imgui.text_colored(f"{fps:.0f}", *fps_color)
        imgui.same_line()
        imgui.text_colored("FPS", text_dim[0], text_dim[1], text_dim[2], 1.0)
        imgui.same_line(spacing=20 * self.scale)

        # Frame time
        text_primary = self._theme.text_primary
        imgui.text_colored(f"{self._display_frame_ms:.2f}", *text_primary)
        imgui.same_line()
        imgui.text_colored("ms", text_dim[0], text_dim[1], text_dim[2], 1.0)

        # FPS graph
        imgui.plot_lines("##fps_graph", self.fps_history,
                         scale_min=0, scale_max=max(144, float(np.max(self.fps_history)) * 1.1),
                         graph_size=(236 * self.scale, 35 * self.scale))

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
        self._pop_hud_style()

    def _draw_system_panel(self, io, accent, text_dim):
        """Draw system resources panel with proper right alignment and dynamic width."""
        padding = 50 * self.scale
        text_primary = self._theme.text_primary

        # Force measurement on first render or when needed
        needs_measure = (
                self._system_panel_width is None or  # First render
                self._system_panel_needs_remeasure or
                self._last_gpu_name != self._gpu_name
        )

        if needs_measure:
            self._system_panel_needs_remeasure = False
            self._last_gpu_name = self._gpu_name

            # === MEASUREMENT PASS ===
            imgui.set_next_window_position(-1000, -1000)
            imgui.set_next_window_bg_alpha(0.0)

            measure_flags = (
                    imgui.WINDOW_NO_DECORATION |
                    imgui.WINDOW_NO_INPUTS |
                    imgui.WINDOW_ALWAYS_AUTO_RESIZE
            )

            imgui.begin("##system_measure", flags=measure_flags)

            # Render content to measure (with maximum expected values)
            imgui.text_colored("SYSTEM", accent[0], accent[1], accent[2], 1.0)
            imgui.spacing()

            imgui.text_colored("CPU", text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=8 * self.scale)
            imgui.text_colored("100%", *text_primary)
            imgui.same_line(spacing=15 * self.scale)
            imgui.text_colored("RAM", text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=8 * self.scale)
            imgui.text_colored("99.9GB", *text_primary)

            if HAS_GPU_UTIL and self._gpu_name != "N/A":
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                imgui.text_colored("GPU", accent[0], accent[1], accent[2], 1.0)

                # Measure GPU name width
                imgui.push_text_wrap_pos(0)  # No wrapping for measurement
                imgui.text(self._gpu_name)
                imgui.pop_text_wrap_pos()

                imgui.spacing()

                # Measure status line
                imgui.text_colored("Load", text_dim[0], text_dim[1], text_dim[2], 1.0)
                imgui.same_line(spacing=5 * self.scale)
                imgui.text_colored("100%", *text_primary)
                imgui.same_line(spacing=12 * self.scale)
                imgui.text_colored("Temp", text_dim[0], text_dim[1], text_dim[2], 1.0)
                imgui.same_line(spacing=5 * self.scale)
                imgui.text_colored("100°C", *text_primary)

            # Get measured width
            measured = imgui.get_window_width()

            # Set width with minimum constraint
            min_width = 350 * self.scale
            # Use measured width directly if it's reasonable, otherwise clamp
            # The issue was likely that 'measured' was too large initially or accumulating
            self._system_panel_width = max(min_width, min(measured, 350 * self.scale))

            imgui.end()

        # If width is still None (shouldn't happen), use default
        if self._system_panel_width is None:
            self._system_panel_width = 350 * self.scale

        # === VISIBLE RENDER PASS ===
        graph_width = self._system_panel_width - 24 * self.scale

        x_pos = io.display_size.x - self._system_panel_width - 15 * self.scale
        y_pos = padding

        imgui.set_next_window_position(x_pos, y_pos)
        imgui.set_next_window_size(self._system_panel_width, 0)  # Force width

        visible_flags = (
                imgui.WINDOW_NO_DECORATION |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )

        self._push_hud_style()
        imgui.begin("##system", flags=visible_flags)

        # Render actual content
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
        imgui.text_colored(f"{self._ram_used_gb:.1f}GB", *text_primary)

        # CPU graph
        imgui.plot_lines("##cpu_graph", self.cpu_history,
                         scale_min=0, scale_max=100,
                         graph_size=(graph_width, 25 * self.scale))

        # GPU section
        if HAS_GPU_UTIL and self._gpu_name != "N/A":
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text_colored("GPU", accent[0], accent[1], accent[2], 1.0)

            imgui.push_text_wrap_pos(self._system_panel_width - 24 * self.scale)
            imgui.text_colored(self._gpu_name, text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.pop_text_wrap_pos()
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
            imgui.text_colored(f"{self._vram_used_mb:.0f}MB", *text_primary)

            imgui.plot_lines("##gpu_graph", self.gpu_history,
                             scale_min=0, scale_max=100,
                             graph_size=(graph_width, 25 * self.scale))

        imgui.end()
        self._pop_hud_style()

    def _draw_scene_panel(self, io, accent, text_dim):
        """Draw custom scene statistics."""
        padding = 15 * self.scale
        imgui.set_next_window_position(padding, io.display_size.y - (120 * self.scale))
        imgui.set_next_window_size(220 * self.scale, 0)

        flags = (imgui.WINDOW_NO_DECORATION |
                 imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        self._push_hud_style()
        imgui.begin("##scene_stats", flags=flags)

        text_primary = self._theme.text_primary

        imgui.text_colored("SCENE", accent[0], accent[1], accent[2], 1.0)
        imgui.spacing()

        for key, value in self.scene_stats.items():
            imgui.text_colored(str(key), text_dim[0], text_dim[1], text_dim[2], 1.0)
            imgui.same_line(spacing=10 * self.scale)
            if isinstance(value, float):
                imgui.text_colored(f"{value:.2f}", *text_primary)
            elif isinstance(value, int):
                imgui.text_colored(f"{value:,}", *text_primary)
            else:
                imgui.text_colored(str(value), *text_primary)

        imgui.end()
        self._pop_hud_style()

    def _update_stats(self):
        """Update performance statistics with ACCURATE timing."""
        current_time = time.perf_counter()
        wall_time = time.time()

        actual_frame_time = current_time - self._last_frame_time
        self._last_frame_time = current_time

        # Clamp to reasonable range (avoid division issues on first frame)
        actual_frame_time = max(actual_frame_time, 0.0001)  # Min ~10000 FPS
        actual_frame_time = min(actual_frame_time, 1.0)  # Max 1 FPS

        actual_fps = 1.0 / actual_frame_time

        # EMA smoothing prevents jitter while remaining responsive
        self._smoothed_fps = (self._ema_alpha * actual_fps +
                              (1 - self._ema_alpha) * self._smoothed_fps)

        self._display_fps = self._smoothed_fps
        self._display_frame_ms = actual_frame_time * 1000
        self._frame_times.append(actual_frame_time)
        self._frame_count += 1

        # === Update graphs at reduced frequency (10 Hz) ===
        if wall_time - self._last_graph_update > self._graph_update_interval:
            self._last_graph_update = wall_time

            # Use circular buffer instead of np.roll
            self.fps_history[self._fps_idx] = self._smoothed_fps
            self.frame_time_history[self._fps_idx] = actual_frame_time
            self._fps_idx = (self._fps_idx + 1) % self._history_size

        # === CPU/RAM (every 1s to reduce overhead) ===
        if wall_time - self._last_cpu_update > self._cpu_update_interval:
            self._last_cpu_update = wall_time
            # psutil.cpu_percent() is blocking - consider using interval=None
            self._cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self._ram_used_gb = mem.used / (1024 ** 3)

            self.cpu_history[self._cpu_idx] = self._cpu_percent
            self._cpu_idx = (self._cpu_idx + 1) % 60

        # === GPU (every 1s) ===
        if HAS_GPU_UTIL and wall_time - self._last_gpu_update > self._gpu_update_interval:
            self._last_gpu_update = wall_time
            self._update_gpu_stats()

    def _update_gpu_stats(self):
        """Update GPU statistics (can be called from thread)."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                with self._gpu_lock:
                    self._gpu_percent = gpu.load * 100
                    self._gpu_temp = gpu.temperature or 0
                    self._vram_used_mb = gpu.memoryUsed
                    self._vram_percent = (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0

                self.gpu_history[self._gpu_idx] = self._gpu_percent
                self._gpu_idx = (self._gpu_idx + 1) % 60
        except Exception:
            pass

    def _get_fps_color(self, fps):
        return _COLOR_OK if fps >= 60 else (_COLOR_WARN if fps >= 30 else _COLOR_CRIT)

    def _get_usage_color(self, percent):
        return _COLOR_OK if percent < 50 else (_COLOR_WARN if percent < 80 else _COLOR_CRIT)

    def _get_temp_color(self, temp):
        return _COLOR_OK if temp < 60 else (_COLOR_WARN if temp < 80 else _COLOR_CRIT)

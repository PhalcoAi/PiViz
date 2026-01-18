# piviz/core/studio.py
"""
PiViz Studio - Main Application Engine
======================================
"""

import logging
import sys
import numpy as np
import time as time_module
import re

try:
    from .gpu_selector import auto_select_gpu

    _gpu_result = auto_select_gpu(verbose=True)
except ImportError:
    _gpu_result = None


# --- LOG SILENCING ---
class MGLWSilencer(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


LOGGERS_TO_SILENCE = [
    'moderngl_window',
    'moderngl_window.context.base.window',
    'moderngl_window.context.pyglet.window'
]
for name in LOGGERS_TO_SILENCE:
    logger = logging.getLogger(name)
    logger.addFilter(MGLWSilencer())
    logger.propagate = False

import moderngl_window as mglw
import moderngl
import imgui
import math
import os
import platform
import traceback
from typing import Optional, Union, Set

from moderngl_window.integrations.imgui import ModernglWindowRenderer

from .camera import Camera
from .scene import PiVizFX
from .theme import Theme, DARK_THEME, LIGHT_THEME, get_theme
from .exporter import Exporter
from ..ui.overlay import PiVizOverlay
from ..ui.manager import UIManager
from ..ui.viewcube import ViewCube
from ..graphics.environment import GridRenderer, AxesRenderer
from ..graphics import primitives as pgfx


class PiVizStudio(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "πViz Studio"

    _local_res = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
    if not os.path.exists(_local_res):
        os.makedirs(_local_res)
    resource_dir = _local_res

    window_size = (1600, 900)
    aspect_ratio = None
    resizable = True
    samples = 4
    vsync = True
    _startup_scene = None
    _banner_printed = False

    def __init__(self, scene_fx: Optional[PiVizFX] = None, **kwargs):
        self._print_welcome_banner()

        # Handle scene class attribute
        if hasattr(self.__class__, 'scene_class') and self.__class__.scene_class:
            scene_fx = self.__class__.scene_class()

        # Store startup scene for later initialization
        if scene_fx is not None:
            PiVizStudio._startup_scene = scene_fx

        # Pre-run configuration phase check
        # moderngl-window calls __init__ twice: once for config, once with context
        if 'ctx' not in kwargs:
            return

        super().__init__(**kwargs)

        # --- CORE STATE ---
        self._theme = DARK_THEME
        self._theme_name = "dark"
        self._keys_pressed: Set[int] = set()

        # --- RESIZE STATE ---
        self._resize_timer = 0.0
        self._pending_width = self.wnd.size[0]
        self._pending_height = self.wnd.size[1]
        self._is_resizing = False
        self._last_window_size = self.wnd.size

        # --- UI SCALE ---
        self.ui_scale = 1.0

        # --- CAMERA ---
        self.camera = Camera()
        self.camera.resize(*self.wnd.size)

        # --- IMGUI ---
        imgui.create_context()
        self.imgui_renderer = ModernglWindowRenderer(self.wnd)

        # --- UI COMPONENTS ---
        self.overlay = PiVizOverlay(self)
        self.ui_manager = UIManager(self)
        self.viewcube = ViewCube(size=120)

        # --- EXPORTER ---
        self.exporter = Exporter(self.ctx, self.wnd.size)

        # --- ENVIRONMENT RENDERERS ---
        self.grid_renderer = GridRenderer(self.ctx, self._theme)
        self.axes_renderer = AxesRenderer(self.ctx, self._theme)

        # --- APPLY THEME ---
        self.overlay.set_theme(self._theme)
        self.viewcube.set_theme(self._theme)

        # --- DISPLAY FLAGS ---
        self.show_grid = True
        self.show_axes = True
        self.use_orthographic = False
        self.show_overlay = True

        # --- INITIALIZE UI SCALE ---
        self._update_ui_scale(*self.wnd.size)

        # --- SCENE INITIALIZATION ---
        self.scene: Optional[PiVizFX] = None
        if PiVizStudio._startup_scene:
            self._init_scene(PiVizStudio._startup_scene)
            PiVizStudio._startup_scene = None

    def _print_welcome_banner(self, detailed: bool = False):
        """
        Print a compact, aligned startup banner with controls.
        """
        if getattr(PiVizStudio, '_banner_printed', False):
            return
        PiVizStudio._banner_printed = True

        import re
        import platform

        # --- Configuration ---
        WIDTH = 78
        BORDER_COLOR = "\033[94m"  # Blue
        TEXT_COLOR = "\033[97m"  # White
        LABEL_COLOR = "\033[90m"  # Grey
        ACCENT_COLOR = "\033[92m"  # Green
        TITLE_COLOR = "\033[96m"  # Cyan
        RESET = "\033[0m"

        # --- Helpers ---
        def visible_len(s):
            """Returns length of string ignoring ANSI color codes."""
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return len(ansi_escape.sub('', s))

        def print_border(top=False, bottom=False, middle=False):
            if top:
                c, l, r = '╔', '═', '╗'
            elif bottom:
                c, l, r = '╚', '═', '╝'
            elif middle:
                c, l, r = '╠', '═', '╣'
            else:
                return
            print(f"{BORDER_COLOR}{c}{l * (WIDTH - 2)}{r}{RESET}")

        def print_line(content, center=False):
            v_len = visible_len(content)
            # Truncate if too long to prevent exploding the box
            if v_len > WIDTH - 4:
                content = content[:WIDTH - 7] + "..."
                v_len = visible_len(content)

            padding = WIDTH - 4 - v_len
            print(f"{BORDER_COLOR}║ {RESET}{content}{' ' * padding} {BORDER_COLOR}║{RESET}")

        # --- Data Gathering ---
        gpu_info = "Unknown GPU"
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus: gpu_info = gpus[0].name
        except:
            pass

        # --- Print Compact Banner ---
        print("")  # Top spacing
        print_border(top=True)

        # Header
        header = f"{ACCENT_COLOR}πViz Studio {LABEL_COLOR}v1.0.1{RESET}   {TITLE_COLOR}Interactive 3D Engine{RESET}"
        print_line(header)

        print_border(middle=True)

        # System Info Row
        sys_info = f"{LABEL_COLOR}System:{RESET} {platform.system()} │ Python {platform.python_version()} │ {gpu_info}"
        print_line(sys_info)

        print_border(middle=True)

        # Controls Section (Compact Columns)
        # Mouse Row
        mouse = f"{LABEL_COLOR}Mouse:{RESET}  {TEXT_COLOR}L-Drag:{RESET} Orbit  │  {TEXT_COLOR}R-Drag:{RESET} Pan  │  {TEXT_COLOR}Scroll:{RESET} Zoom"
        print_line(mouse)

        # Keys Row
        keys = f"{LABEL_COLOR}Keys:{RESET}   {TEXT_COLOR}H:{RESET} Home  │  {TEXT_COLOR}G/A:{RESET} Grid/Axes  │  {TEXT_COLOR}T:{RESET} Theme  │  {TEXT_COLOR}0-3:{RESET} Views"
        print_line(keys)

        print_border(bottom=True)
        print(f" {ACCENT_COLOR}Ready.{RESET} Launching window...\n")

    def _init_scene(self, scene: PiVizFX):
        """Initialize a scene with the current context."""
        self.scene = scene
        scene._internal_init(self.ctx, self.wnd, self)

    def _update_ui_scale(self, width, height):
        """Update UI scaling based on window size."""
        self.ui_scale = max(1.0, width / 1920.0)
        imgui.get_io().font_global_scale = self.ui_scale
        self.overlay.set_scale(self.ui_scale)

    def run(self):
        """Run the application."""
        try:
            mglw.run_window_config(self.__class__)
        except Exception as e:
            self._print_crash_report(e)
            sys.exit(1)

    def _print_crash_report(self, e: Exception):
        """Print formatted crash report with wrapped text."""
        import traceback
        import re
        import textwrap

        # --- Colors ---
        c_red = "\033[91m"
        c_grey = "\033[90m"
        c_reset = "\033[0m"
        c_white = "\033[97m"

        # --- Configuration ---
        WIDTH = 60

        # --- Helpers ---
        def visible_len(s):
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return len(ansi_escape.sub('', s))

        def print_line(content, color=c_red, text_color=c_white):
            v_len = visible_len(content)
            padding = WIDTH - 4 - v_len
            print(f"{color}║ {text_color}{content}{' ' * padding} {color}║{c_reset}")

        # --- Print Box ---
        print(f"\n{c_red}╔{'═' * (WIDTH - 2)}╗{c_reset}")

        # Title
        print_line("CRITICAL ERROR", text_color=c_white)
        print(f"{c_red}╠{'═' * (WIDTH - 2)}╣{c_reset}")
        error_msg = str(e)
        wrapped_lines = textwrap.wrap(error_msg, width=WIDTH - 4)

        for line in wrapped_lines:
            print_line(line)

        print(f"{c_red}╚{'═' * (WIDTH - 2)}╝{c_reset}")

        # --- Stack Trace ---
        print(f"\n{c_grey}--- Stack Trace ---{c_reset}")
        traceback.print_exc()

    # =========================================================================
    # THEME MANAGEMENT
    # =========================================================================

    @property
    def theme(self) -> Theme:
        return self._theme

    def set_theme(self, theme: Union[str, Theme]):
        """Set the application theme."""
        if isinstance(theme, str):
            self._theme = get_theme(theme)
            self._theme_name = theme
        else:
            self._theme = theme
            self._theme_name = theme.name

        self.grid_renderer.set_theme(self._theme)
        self.axes_renderer.set_theme(self._theme)
        self.overlay.set_theme(self._theme)
        self.viewcube.set_theme(self._theme)

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        self.set_theme("light" if self._theme_name == "dark" else "dark")

    # =========================================================================
    # VIEW MANAGEMENT
    # =========================================================================

    def _fit_view_to_scene(self):
        """Fit the camera view to show the entire scene."""
        if self.scene and hasattr(self.scene, 'get_bounds'):
            min_bound, max_bound = self.scene.get_bounds()
            self.camera.fit_to_bounds(np.array(min_bound), np.array(max_bound))
        else:
            self.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.camera.distance = 15.0
            self.camera.azimuth = 45.0
            self.camera.elevation = 30.0

    # =========================================================================
    # RESIZE HANDLING
    # =========================================================================

    def _check_for_resize(self):
        """Poll-based resize detection - catches ALL resize types."""
        current_size = (self.wnd.width, self.wnd.height)

        if current_size != self._last_window_size:
            width, height = current_size
            self._last_window_size = current_size

            # Immediate lightweight updates
            self.imgui_renderer.resize(width, height)
            self.camera.resize(width, height)

            # Schedule deferred heavy updates
            self._pending_width = width
            self._pending_height = height
            self._resize_timer = 0.15
            self._is_resizing = True

    def _delayed_resize(self, width, height):
        """Execute heavy resize operations after dragging stops."""
        self._is_resizing = False

        # Update UI scale
        self._update_ui_scale(width, height)

        # Resize exporter (reallocates GPU buffers)
        self.exporter.resize(width, height)

        # Resize scene if it has buffers
        if self.scene:
            self.scene.resize(width, height)

    def on_resize(self, width: int, height: int):
        """Event-based resize handler (fallback, main detection is poll-based)."""
        # Handled by _check_for_resize() in render loop
        pass

    # =========================================================================
    # RENDER LOOP
    # =========================================================================

    def render(self, time: float, frame_time: float):
        """Compatibility wrapper for moderngl-window."""
        self.on_render(time, frame_time)

    def on_render(self, time: float, frame_time: float):
        """Main render loop."""
        try:
            if not hasattr(self, 'ctx'):
                return

            # Poll for resize (catches horizontal/vertical/diagonal)
            self._check_for_resize()

            # Handle resize timer
            if self._resize_timer > 0:
                self._resize_timer -= frame_time
                if self._resize_timer <= 0:
                    self._delayed_resize(self._pending_width, self._pending_height)
                    self._resize_timer = 0

            # Lightweight render during active resize
            if self._is_resizing:
                self._render_resize_preview()
                return

            # Full render
            self._render_full(time, frame_time)

        except Exception as e:
            self._print_crash_report(e)
            self.wnd.close()

    def _render_resize_preview(self):
        """Minimal render during resize for smooth dragging."""
        bg = self._theme.background
        self.ctx.clear(*bg[:3])

        # Minimal ImGui frame (required to prevent crashes)
        imgui.new_frame()
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def _render_full(self, time: float, frame_time: float):
        """Full scene render for normal operation."""
        self._process_input(frame_time)

        # Start ImGui frame
        imgui.new_frame()

        # Update viewcube
        self.viewcube.update(frame_time, self.camera)

        # Clear and set render state
        bg = self._theme.background
        self.ctx.clear(*bg[:3])
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Get view/projection matrices
        view = self.camera.get_view_matrix()
        proj = self.camera.get_orthographic_matrix() if self.use_orthographic else self.camera.get_projection_matrix()

        # Initialize primitive graphics context
        pgfx._init_context(self.ctx, view, proj)

        # Render environment
        if self.show_grid:
            self.grid_renderer.render(view, proj, self.camera)
        if self.show_axes:
            self.axes_renderer.render(view, proj)

        # Render scene
        if self.scene:
            self.scene.render(time, frame_time)
            self.scene.loop(frame_time)

            # Flush batched primitives
            pgfx.flush_all()

            # Render scene UI
            if hasattr(self.scene, 'render_ui'):
                self.scene.render_ui()

        # Capture frame for export (before UI overlay)
        if self.exporter._recording:
            self.exporter.capture_frame()

        # Render UI
        self._render_ui()

        # Finalize ImGui
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def _process_input(self, dt: float):
        """Process held keys for camera movement."""
        if self.wnd.keys.LEFT in self._keys_pressed:
            self.camera.on_key_hold('left', dt)
        if self.wnd.keys.RIGHT in self._keys_pressed:
            self.camera.on_key_hold('right', dt)
        if self.wnd.keys.UP in self._keys_pressed:
            self.camera.on_key_hold('up', dt)
        if self.wnd.keys.DOWN in self._keys_pressed:
            self.camera.on_key_hold('down', dt)

    # =========================================================================
    # UI RENDERING
    # =========================================================================

    def _render_ui(self):
        """Render all UI components."""
        if self.show_overlay:
            self.overlay.render()

        self.viewcube.render(self.camera)
        self._draw_view_toggles()
        self._draw_top_controls()
        self.ui_manager.render()

    def _draw_view_toggles(self):
        """Draw minimal Grid/Axes/Ortho toggles near the ViewCube."""
        io = imgui.get_io()
        x = 15
        y = io.display_size.y - 270

        imgui.set_next_window_position(x, y)
        imgui.set_next_window_size(0, 0)

        flags = (
                imgui.WINDOW_NO_DECORATION |
                imgui.WINDOW_NO_BACKGROUND |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )

        imgui.begin("##view_toggles", flags=flags)

        accent = self._theme.accent
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *accent)

        _, self.show_grid = imgui.checkbox("Grid", self.show_grid)
        _, self.show_axes = imgui.checkbox("Axes", self.show_axes)
        _, self.use_orthographic = imgui.checkbox("Ortho", self.use_orthographic)

        imgui.pop_style_color()
        imgui.end()

    def _draw_top_controls(self):
        """Draw top-right control buttons: Fit View, Record, Screenshot, Theme."""
        io = imgui.get_io()

        # Button configuration
        button_size = 32 * self.ui_scale
        margin = 15 * self.ui_scale
        spacing = 8 * self.ui_scale
        num_buttons = 4
        total_w = (button_size * num_buttons) + (spacing * (num_buttons - 1))

        # Position: top-right
        start_x = io.display_size.x - total_w - margin - 20
        y = margin - 10

        imgui.set_next_window_position(start_x, y)
        imgui.set_next_window_size(total_w + 16, button_size + 16)

        flags = (
                imgui.WINDOW_NO_DECORATION |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_BACKGROUND |
                imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )

        imgui.begin("##top_controls", flags=flags)

        draw_list = imgui.get_window_draw_list()

        # --- FIT VIEW BUTTON ---
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 0,
            "Fit View (Home)", self._fit_view_to_scene
        )
        self._draw_fit_icon(draw_list, cx, cy, button_size, col)

        # --- RECORD BUTTON ---
        is_rec = self.exporter._recording
        is_flash = is_rec and (int(time_module.time() * 2) % 2 == 0)

        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 1,
            "Stop Recording" if is_rec else "Record Video (MP4)",
            lambda: self.exporter.stop_recording() if is_rec else self.exporter.start_recording(),
            is_active=is_rec, is_flash=is_flash
        )
        rec_col = imgui.get_color_u32_rgba(1, 0.2, 0.2, 1) if is_rec else col
        draw_list.add_circle_filled(cx, cy, button_size * 0.25, rec_col, 16)

        # --- SCREENSHOT BUTTON ---
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 2,
            "Take Screenshot (Clean)", lambda: self.exporter.take_screenshot()
        )
        self._draw_camera_icon(draw_list, cx, cy, button_size, col)

        # --- THEME BUTTON ---
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 3,
            "Toggle Theme (T)", self.toggle_theme
        )

        if self._theme_name == 'dark':
            self._draw_moon_icon(draw_list, cx, cy, button_size * 0.32)
        else:
            self._draw_sun_icon(draw_list, cx, cy, button_size * 0.28)

        imgui.end()

    def _draw_circle_button(self, draw_list, start_x, y, button_size, spacing,
                            offset_idx, tooltip, callback, is_active=False, is_flash=False):
        """Draw a circular button and handle interaction."""
        io = imgui.get_io()

        cx = start_x + (button_size / 2) + (offset_idx * (button_size + spacing)) + 8
        cy = y + button_size / 2 + 8

        mx, my = io.mouse_pos
        is_hovered = ((mx - cx) ** 2 + (my - cy) ** 2) < (button_size / 2 + 2) ** 2

        # Determine background color
        if is_flash:
            bg = (0.8, 0.1, 0.1, 0.8)
        elif is_active:
            bg = (*self._theme.accent[:3], 0.6)
        elif is_hovered:
            bg = (*self._theme.accent[:3], 0.3)
        else:
            bg = (*self._theme.panel[:3], 0.6)

        draw_list.add_circle_filled(cx, cy, button_size / 2 + 2, imgui.get_color_u32_rgba(*bg), 24)

        if is_hovered and imgui.is_mouse_clicked(0):
            callback()

        if is_hovered:
            imgui.set_tooltip(tooltip)

        col = imgui.get_color_u32_rgba(*self._theme.text_primary)
        return cx, cy, col

    def _draw_fit_icon(self, draw_list, cx, cy, button_size, col):
        """Draw fit/home icon."""
        r = button_size * 0.22

        # Outer square
        draw_list.add_rect(cx - r, cy - r, cx + r, cy + r, col, rounding=2.0, thickness=1.5)

        # Center dot
        draw_list.add_circle_filled(cx, cy, r * 0.25, col, 8)

        # Corner arrows
        arrow_len = r * 0.4
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for dx, dy in corners:
            ox = cx + dx * r * 0.7
            oy = cy + dy * r * 0.7
            draw_list.add_line(ox, oy, ox - dx * arrow_len, oy - dy * arrow_len, col, 1.5)

    def _draw_camera_icon(self, draw_list, cx, cy, button_size, col):
        """Draw camera/screenshot icon."""
        r = button_size * 0.2

        draw_list.add_rect(
            cx - r * 1.2, cy - r * 0.8,
            cx + r * 1.2, cy + r * 0.8,
            col, rounding=2.0, thickness=1.5
        )
        draw_list.add_circle(cx, cy, r * 0.5, col, num_segments=12, thickness=1.5)
        draw_list.add_rect_filled(cx + r * 0.6, cy - r * 1.1, cx + r * 1.0, cy - r * 0.8, col)

    def _draw_sun_icon(self, draw_list, cx, cy, radius):
        """Draw sun icon for light theme."""
        col = imgui.get_color_u32_rgba(*self._theme.text_primary)

        draw_list.add_circle_filled(cx, cy, radius * 0.45, col, 16)

        num_rays = 8
        for i in range(num_rays):
            angle = (i / num_rays) * 2 * math.pi - math.pi / 8
            inner_r = radius * 0.6
            outer_r = radius * 1.0
            x1 = cx + math.cos(angle) * inner_r
            y1 = cy + math.sin(angle) * inner_r
            x2 = cx + math.cos(angle) * outer_r
            y2 = cy + math.sin(angle) * outer_r
            draw_list.add_line(x1, y1, x2, y2, col, 2.0 * self.ui_scale)

    def _draw_moon_icon(self, draw_list, cx, cy, radius):
        """Draw moon icon for dark theme."""
        col = imgui.get_color_u32_rgba(*self._theme.text_primary)
        bg_col = imgui.get_color_u32_rgba(*self._theme.background)

        draw_list.add_circle_filled(cx, cy, radius, col, 24)

        cut_offset = radius * 0.35
        draw_list.add_circle_filled(cx + cut_offset, cy - cut_offset, radius * 0.7, bg_col, 24)

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def on_key_event(self, key, action, modifiers):
        """Handle keyboard events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        self.imgui_renderer.key_event(key, action, modifiers)

        # Track pressed keys
        if action == self.wnd.keys.ACTION_PRESS:
            self._keys_pressed.add(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self._keys_pressed.discard(key)

        # Handle key presses
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.G:
                self.show_grid = not self.show_grid
            elif key == self.wnd.keys.A:
                self.show_axes = not self.show_axes
            elif key == self.wnd.keys.T:
                self.toggle_theme()
            elif key == self.wnd.keys.NUMBER_0:
                self.camera.set_view('iso')
            elif key == self.wnd.keys.NUMBER_1:
                self.camera.set_view('front')
            elif key == self.wnd.keys.NUMBER_3:
                self.camera.set_view('top')
            elif key == self.wnd.keys.H:
                self._fit_view_to_scene()

        # Forward to scene
        if self.scene:
            self.scene.key_event(key, action, modifiers)

    def on_mouse_position_event(self, x, y, dx, dy):
        """Handle mouse position events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        self.imgui_renderer.mouse_position_event(x, y, dx, dy)

        if not self.imgui_renderer.io.want_capture_mouse and self.scene:
            self.scene.mouse_position_event(x, y, dx, dy)

    def on_mouse_drag_event(self, x, y, dx, dy):
        """Handle mouse drag events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        self.imgui_renderer.mouse_drag_event(x, y, dx, dy)

        if not self.imgui_renderer.io.want_capture_mouse:
            self.camera.on_mouse_drag(x, y, dx, dy)
            if self.scene:
                self.scene.mouse_drag_event(x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        """Handle mouse scroll events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        io = imgui.get_io()
        io.mouse_wheel = y_offset
        if hasattr(io, 'mouse_wheel_horizontal'):
            io.mouse_wheel_horizontal = x_offset

        if not io.want_capture_mouse:
            self.camera.on_mouse_scroll(x_offset, y_offset)
            if self.scene:
                self.scene.mouse_scroll_event(x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        """Handle mouse press events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        self.imgui_renderer.mouse_press_event(x, y, button)

        if not self.imgui_renderer.io.want_capture_mouse:
            mods = getattr(self.wnd, 'modifiers', 0)
            self.camera.on_mouse_press(x, y, button, mods)
            if self.scene:
                self.scene.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        """Handle mouse release events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        self.imgui_renderer.mouse_release_event(x, y, button)
        self.camera.on_mouse_release(x, y, button)

        if self.scene:
            self.scene.mouse_release_event(x, y, button)

# piviz/core/studio.py
"""
PiViz Studio - Main Application Engine
======================================
"""

import logging
import sys


# --- LOG SILENCING ---
class MGLWSilencer(logging.Filter):
    def filter(self, record): return record.levelno >= logging.WARNING


LOGGERS_TO_SILENCE = ['moderngl_window', 'moderngl_window.context.base.window', 'moderngl_window.context.pyglet.window']
for name in LOGGERS_TO_SILENCE:
    logger = logging.getLogger(name)
    logger.addFilter(MGLWSilencer())
    logger.propagate = False
# ---------------------

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
from .theme import Theme, DARK_THEME, get_theme
from .exporter import Exporter  # <--- NEW IMPORT
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

    def __init__(self, scene_fx: Optional[PiVizFX] = None, **kwargs):
        self._print_welcome_banner()

        if hasattr(self.__class__, 'scene_class') and self.__class__.scene_class:
            scene_fx = self.__class__.scene_class()

        if scene_fx is not None:
            PiVizStudio._startup_scene = scene_fx
            return
        self._pending_scene = scene_fx

        super().__init__(**kwargs)

        # Theme
        self._theme = DARK_THEME
        self._theme_name = 'dark'

        # Camera
        self.camera = Camera()
        self.camera.resize(*self.wnd.size)

        # Input State
        self._keys_pressed: Set[int] = set()

        # ImGui
        imgui.create_context()
        self.imgui_renderer = ModernglWindowRenderer(self.wnd)

        # UI / Overlay
        self.overlay = PiVizOverlay(self)
        self.ui_manager = UIManager(self)
        self.viewcube = ViewCube(size=120)

        # Exporter (Recording/Screenshots)
        self.exporter = Exporter(self.ctx, self.window_size)  # <--- NEW

        # Environment
        self.grid_renderer = GridRenderer(self.ctx, self._theme)
        self.axes_renderer = AxesRenderer(self.ctx, self._theme)
        self.overlay.set_theme(self._theme)
        self.viewcube.set_theme(self._theme)

        # Display flags
        self.show_grid = True
        self.show_axes = True
        self.show_overlay = True

        self.ui_scale = 1.0
        self._update_ui_scale(*self.window_size)

        self.scene: Optional[PiVizFX] = None
        if PiVizStudio._startup_scene:
            self._init_scene(PiVizStudio._startup_scene)

    def _print_welcome_banner(self):
        """Print a clean, professional startup banner."""
        if getattr(PiVizStudio, '_banner_printed', False): return
        PiVizStudio._banner_printed = True
        c_blue, c_green, c_grey, c_reset = "\033[94m", "\033[92m", "\033[90m", "\033[0m"
        gpu_info = "Unknown GPU"
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus: gpu_info = gpus[0].name
        except:
            pass
        print(f"\n{c_blue}╔══════════════════════════════════════════════════════╗")
        print(f"║ {c_green}πViz Studio{c_blue} v0.1.1             {c_grey}Interactive 3D Engine{c_blue} ║")
        print(f"╚══════════════════════════════════════════════════════╝{c_reset}")
        print(f" {c_grey}► System:{c_reset}   {platform.system()} {platform.release()}")
        print(f" {c_grey}► Python:{c_reset}   {platform.python_version()}")
        print(f" {c_grey}► Device:{c_reset}   {gpu_info}")
        print(f" {c_grey}► Context:{c_reset}  OpenGL 3.3+ Core Profile")
        print(f"\n {c_green}Ready.{c_reset} Launching window...\n")

    def _init_scene(self, scene: PiVizFX):
        self.scene = scene
        scene._internal_init(self.ctx, self.wnd, self)

    def _update_ui_scale(self, width, height):
        self.ui_scale = max(1.0, width / 1920.0)
        imgui.get_io().font_global_scale = self.ui_scale
        self.overlay.set_scale(self.ui_scale)
        self.exporter.resize(width, height)

    def run(self):
        try:
            mglw.run_window_config(self.__class__)
        except Exception as e:
            self._print_crash_report(e)
            sys.exit(1)

    def _print_crash_report(self, e: Exception):
        c_red, c_grey, c_reset = "\033[91m", "\033[90m", "\033[0m"
        print(f"\n{c_red}╔══════════════════════════════════════════════════════╗")
        print(f"║ CRITICAL ERROR                                       ║")
        print(f"╚══════════════════════════════════════════════════════╝{c_reset}\n{e}")
        print(f"\n{c_grey}--- Stack Trace ---{c_reset}")
        traceback.print_exc()

    # =====================
    # Theme Management
    # =====================

    @property
    def theme(self) -> Theme:
        return self._theme

    def set_theme(self, theme: Union[str, Theme]):
        if isinstance(theme, str):
            self._theme_name = theme.lower()
            self._theme = get_theme(theme)
        else:
            self._theme = theme
            self._theme_name = theme.name
        self.grid_renderer.set_theme(self._theme)
        self.axes_renderer.set_theme(self._theme)
        self.overlay.set_theme(self._theme)
        self.viewcube.set_theme(self._theme)

    def toggle_theme(self):
        self.set_theme('light' if self._theme_name == 'dark' else 'dark')

    # =====================
    # Render Loop
    # =====================

    def on_render(self, time: float, frame_time: float):
        try:
            self._process_input(frame_time)

            # Start UI frame early
            imgui.new_frame()

            self.viewcube.update(frame_time, self.camera)

            bg = self._theme.background
            self.ctx.clear(*bg[:3])
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            view = self.camera.get_view_matrix()
            proj = self.camera.get_projection_matrix()
            pgfx._init_context(self.ctx, view, proj)

            if self.show_grid: self.grid_renderer.render(view, proj, self.camera)
            if self.show_axes: self.axes_renderer.render(view, proj)

            if self.scene:
                self.scene.render(time, frame_time)
                self.scene.loop(frame_time)
                if hasattr(self.scene, 'render_ui'): self.scene.render_ui()

            # --- EXPORT CAPTURE (Clean 3D view) ---
            if self.exporter._recording:
                self.exporter.capture_frame()

            self._render_ui()

            imgui.render()
            self.imgui_renderer.render(imgui.get_draw_data())

        except Exception as e:
            self._print_crash_report(e)
            self.wnd.close()

    def _process_input(self, dt: float):
        if self.wnd.keys.LEFT in self._keys_pressed: self.camera.on_key_hold('left', dt)
        if self.wnd.keys.RIGHT in self._keys_pressed: self.camera.on_key_hold('right', dt)
        if self.wnd.keys.UP in self._keys_pressed: self.camera.on_key_hold('up', dt)
        if self.wnd.keys.DOWN in self._keys_pressed: self.camera.on_key_hold('down', dt)

    # =====================
    # UI
    # =====================

    def _render_ui(self):
        if self.show_overlay: self.overlay.render()
        self.viewcube.render(self.camera)

        self._draw_view_toggles()
        self._draw_top_controls()

        self.ui_manager.render()

    def _draw_view_toggles(self):
        """Draw minimal Grid/Axes toggles near the ViewCube."""
        io = imgui.get_io()
        x, y = 15, io.display_size.y - 250
        imgui.set_next_window_position(x, y)
        imgui.set_next_window_size(0, 0)
        flags = (
                imgui.WINDOW_NO_DECORATION | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        imgui.begin("##view_toggles", flags=flags)

        accent = self._theme.accent
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *accent)
        _, self.show_grid = imgui.checkbox("Grid", self.show_grid)
        _, self.show_axes = imgui.checkbox("Axes", self.show_axes)
        imgui.pop_style_color()
        imgui.end()

    def _draw_top_controls(self):
        """Draw Top-Right Controls: Theme Toggle + Export Buttons."""
        io = imgui.get_io()

        # Config
        button_size = 32 * self.ui_scale
        icon_size_scale = 0.5
        margin = 15 * self.ui_scale
        spacing = 8 * self.ui_scale

        # Calculate width for 3 buttons (REC, SNAP, THEME) + spacing
        num_buttons = 3
        total_w = (button_size * num_buttons) + (spacing * (num_buttons - 1))

        # Position: Top-Right
        start_x = io.display_size.x - total_w - margin - 20
        y = margin - 10

        imgui.set_next_window_position(start_x, y)
        imgui.set_next_window_size(total_w + 16, button_size + 16)  # Add buffer

        flags = (imgui.WINDOW_NO_DECORATION |
                 imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_NO_BACKGROUND |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        imgui.begin("##top_controls", flags=flags)

        draw_list = imgui.get_window_draw_list()

        # --- Helper for circular buttons ---
        def draw_circle_btn(offset_idx, tooltip, callback, is_active=False, is_flash=False):
            # Calculate center for this button
            cx = start_x + (button_size / 2) + (offset_idx * (button_size + spacing)) + 8
            cy = y + button_size / 2 + 8

            # Hover check
            mx, my = io.mouse_pos
            is_hovered = ((mx - cx) ** 2 + (my - cy) ** 2) < (button_size / 2 + 2) ** 2

            # Background Color
            if is_flash:
                bg = (0.8, 0.1, 0.1, 0.8)  # Red flash
            elif is_active:
                bg = (*self._theme.accent[:3], 0.6)  # Active state
            elif is_hovered:
                bg = (*self._theme.accent[:3], 0.3)  # Hover state
            else:
                bg = (*self._theme.panel[:3], 0.6)  # Default

            draw_list.add_circle_filled(cx, cy, button_size / 2 + 2,
                                        imgui.get_color_u32_rgba(*bg), 24)

            # Handle Click
            if is_hovered and imgui.is_mouse_clicked(0):
                callback()

            if is_hovered:
                imgui.set_tooltip(tooltip)

            return cx, cy, imgui.get_color_u32_rgba(*self._theme.text_primary)

        # ---------------------------
        # 1. RECORD BUTTON
        # ---------------------------
        import time
        is_rec = self.exporter._recording
        is_flash = is_rec and (int(time.time() * 2) % 2 == 0)

        cx, cy, col = draw_circle_btn(0, "Record Video (MP4)" if not is_rec else "Stop Recording",
                                      lambda: self.exporter.stop_recording() if is_rec else self.exporter.start_recording(),
                                      is_active=is_rec, is_flash=is_flash)

        # Draw Rec Icon (Circle)
        rec_col = imgui.get_color_u32_rgba(1, 0.2, 0.2, 1) if is_rec else col
        draw_list.add_circle_filled(cx, cy, button_size * 0.25, rec_col, 16)

        # ---------------------------
        # 2. SCREENSHOT BUTTON
        # ---------------------------
        cx, cy, col = draw_circle_btn(1, "Take Screenshot (Clean)", lambda: self.exporter.take_screenshot())

        # Draw Camera Icon (Simple Box + Lens)
        r = button_size * 0.2
        # Body
        draw_list.add_rect(cx - r * 1.2, cy - r * 0.8, cx + r * 1.2, cy + r * 0.8, col, rounding=2.0, thickness=1.5)
        # Lens
        draw_list.add_circle(cx, cy, r * 0.5, col, num_segments=12, thickness=1.5)
        # Flash bump
        draw_list.add_rect_filled(cx + r * 0.6, cy - r * 1.1, cx + r * 1.0, cy - r * 0.8, col)

        # ---------------------------
        # 3. THEME BUTTON
        # ---------------------------
        cx, cy, col = draw_circle_btn(2, "Toggle Theme (T)", self.toggle_theme)

        if self._theme_name == 'dark':
            self._draw_moon_icon(draw_list, cx, cy, button_size * 0.32, self._theme.text_primary)
        else:
            self._draw_sun_icon(draw_list, cx, cy, button_size * 0.28, self._theme.text_primary)

        imgui.end()

    def _draw_sun_icon(self, draw_list, cx, cy, radius, color):
        col = imgui.get_color_u32_rgba(*color)
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

    def _draw_moon_icon(self, draw_list, cx, cy, radius, color):
        col = imgui.get_color_u32_rgba(*color)
        bg_col = imgui.get_color_u32_rgba(*self._theme.background)
        draw_list.add_circle_filled(cx, cy, radius, col, 24)
        cut_offset = radius * 0.35
        draw_list.add_circle_filled(cx + cut_offset, cy - cut_offset,
                                    radius * 0.7, bg_col, 24)

    # =====================
    # Events
    # =====================

    def on_resize(self, width: int, height: int):
        self.imgui_renderer.resize(width, height)
        self.camera.resize(width, height)
        self._update_ui_scale(width, height)
        if self.scene:
            self.scene.resize(width, height)

    def on_key_event(self, key, action, modifiers):
        self.imgui_renderer.key_event(key, action, modifiers)

        if action == self.wnd.keys.ACTION_PRESS:
            self._keys_pressed.add(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self._keys_pressed.discard(key)

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

        if self.scene:
            self.scene.key_event(key, action, modifiers)

    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui_renderer.mouse_position_event(x, y, dx, dy)
        if not self.imgui_renderer.io.want_capture_mouse:
            if self.scene: self.scene.mouse_position_event(x, y, dx, dy)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.imgui_renderer.mouse_drag_event(x, y, dx, dy)
        if not self.imgui_renderer.io.want_capture_mouse:
            self.camera.on_mouse_drag(x, y, dx, dy)
            if self.scene: self.scene.mouse_drag_event(x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        io = imgui.get_io()
        io.mouse_wheel = y_offset
        if hasattr(io, 'mouse_wheel_horizontal'):
            io.mouse_wheel_horizontal = x_offset

        if not io.want_capture_mouse:
            self.camera.on_mouse_scroll(x_offset, y_offset)
            if self.scene: self.scene.mouse_scroll_event(x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        self.imgui_renderer.mouse_press_event(x, y, button)
        if not self.imgui_renderer.io.want_capture_mouse:
            mods = getattr(self.wnd, 'modifiers', 0)
            self.camera.on_mouse_press(x, y, button, mods)
            if self.scene: self.scene.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        self.imgui_renderer.mouse_release_event(x, y, button)
        self.camera.on_mouse_release(x, y, button)
        if self.scene: self.scene.mouse_release_event(x, y, button)

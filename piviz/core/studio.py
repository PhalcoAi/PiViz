import logging
import sys
import numpy as np
import platform
import re
import textwrap

try:
    from .gpu_selector import auto_select_gpu

    _gpu_result = auto_select_gpu(verbose=True)
except ImportError:
    _gpu_result = None


# --- LOG SILENCING ---
class MGLWSilencer(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


for name in [
    'moderngl_window',
    'moderngl_window.context.base.window',
    'moderngl_window.context.pyglet.window'
]:
    logger = logging.getLogger(name)
    logger.addFilter(MGLWSilencer())
    logger.propagate = False

import moderngl_window as mglw
import moderngl
import imgui
import os
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
from ..ui.toolbar import Toolbar
from ..ui.toggles import ViewToggles
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

    # Font Awesome 6 Free Solid codepoints
    ICON_HOME = "\uf015"  # fa-house
    ICON_EXPAND = "\uf424"  # fa-expand
    ICON_CAMERA = "\uf030"  # fa-camera
    ICON_VIDEO = "\uf03d"  # fa-video
    ICON_STOP = "\uf04d"  # fa-stop  (recording active)
    ICON_SUN = "\uf185"  # fa-sun
    ICON_MOON = "\uf186"  # fa-moon
    ICON_GRID = "\uf00a"  # fa-table-cells
    ICON_AXES = "\uf1b2"  # fa-cube
    ICON_CUBE = "\uf1b2"  # fa-cube
    ICON_EYE = "\uf06e"  # fa-eye
    ICON_COG = "\uf013"  # fa-gear

    def __init__(self, scene_fx: Optional[PiVizFX] = None, **kwargs):
        self._print_welcome_banner()

        if hasattr(self.__class__, 'scene_class') and self.__class__.scene_class:
            scene_fx = self.__class__.scene_class()

        if scene_fx is not None:
            PiVizStudio._startup_scene = scene_fx

        # Pre-run configuration phase check
        # moderngl-window calls __init__ twice: once for config, once with context
        if 'ctx' not in kwargs:
            return

        super().__init__(**kwargs)

        self._theme = DARK_THEME
        self._theme_name = "dark"
        self._keys_pressed: Set[int] = set()

        self._resize_timer = 0.0
        self._pending_width = self.wnd.size[0]
        self._pending_height = self.wnd.size[1]
        self._is_resizing = False
        self._last_window_size = self.wnd.size

        self.ui_scale = 1.0

        self.camera = Camera()
        self.camera.resize(*self.wnd.size)

        imgui.create_context()
        self.imgui_renderer = ModernglWindowRenderer(self.wnd)

        self._icon_font = None
        self._load_fonts()

        self.overlay = PiVizOverlay(self)
        self.ui_manager = UIManager(self)
        self.viewcube = ViewCube(size=120)
        self.toolbar = Toolbar(self)
        self.view_toggles = ViewToggles(self)

        self.exporter = Exporter(self.ctx, self.wnd.size)

        self.grid_renderer = GridRenderer(self.ctx, self._theme)
        self.axes_renderer = AxesRenderer(self.ctx, self._theme)

        self.overlay.set_theme(self._theme)
        self.viewcube.set_theme(self._theme)

        self.show_grid = True
        self.show_axes = True
        self.use_orthographic = False
        self.show_overlay = True

        self._update_ui_scale(*self.wnd.size)
        env_scale = os.environ.get('PIVIZ_UI_SCALE')
        if env_scale:
            try:
                self.ui_scale = float(env_scale)
                imgui.get_io().font_global_scale = self.ui_scale
                self.overlay.set_scale(self.ui_scale)
                print(f"[UI Scale] Manual override: {self.ui_scale}")
            except ValueError:
                pass

        self.scene: Optional[PiVizFX] = None
        if PiVizStudio._startup_scene:
            self._init_scene(PiVizStudio._startup_scene)
            PiVizStudio._startup_scene = None

    def _find_icon_font(self) -> str | None:
        """Locate fa-solid-900.ttf: resources dir → system paths → auto-download."""
        cached = os.path.join(self.resource_dir, 'fonts', 'fa-solid-900.ttf')
        if os.path.exists(cached):
            return cached

        system_paths = [
            '/usr/share/fonts/opentype/font-awesome/FontAwesome.otf',
            '/usr/share/fonts/truetype/font-awesome/fa-solid-900.ttf',
            '/usr/local/share/fonts/fa-solid-900.ttf',
        ]
        for p in system_paths:
            if os.path.exists(p):
                return p

        # Auto-download from jsDelivr (OFL licensed, ~80 KB)
        url = (
            'https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.5.2'
            '/webfonts/fa-solid-900.ttf'
        )
        try:
            import urllib.request
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            urllib.request.urlretrieve(url, cached)
            return cached
        except Exception as e:
            print(f"[piviz] icon font download failed: {e}")
            return None

    def _load_fonts(self):
        """Load UI fonts and Font Awesome 6 icon font."""
        io = imgui.get_io()
        io.fonts.clear()

        system_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/TTF/DejaVuSans.ttf',
            'C:\\Windows\\Fonts\\arial.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
        ]
        text_loaded = False
        for fp in system_fonts:
            if os.path.exists(fp):
                try:
                    io.fonts.add_font_from_file_ttf(fp, 20.0)
                    text_loaded = True
                    break
                except Exception:
                    pass
        if not text_loaded:
            io.fonts.add_font_default()

        icon_path = self._find_icon_font()
        if icon_path:
            try:
                self._icon_font = io.fonts.add_font_from_file_ttf(
                    icon_path, 22.0,
                    glyph_ranges=imgui.GlyphRanges([0xF000, 0xFFFF, 0])
                )
            except Exception as e:
                print(f"[piviz] icon font load failed: {e}")
                self._icon_font = None
        else:
            self._icon_font = None

        self.imgui_renderer.refresh_font_texture()

    def _print_welcome_banner(self, detailed: bool = False):
        if getattr(PiVizStudio, '_banner_printed', False):
            return
        PiVizStudio._banner_printed = True

        W = 74
        DIM = "\033[90m"
        WHITE = "\033[97m"
        ORANGE = "\033[38;5;208m"
        RESET = "\033[0m"

        ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        def vlen(s):
            return len(ansi_re.sub('', s))

        def row(content):
            pad = W - 2 - vlen(content)
            print(f"{DIM}│{RESET} {content}{' ' * max(0, pad - 1)}{DIM}│{RESET}")

        def div():
            print(f"{DIM}├{'─' * (W - 2)}┤{RESET}")

        gpu_info = "—"
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus: gpu_info = gpus[0].name
        except Exception:
            pass

        print()
        print(f"{DIM}╭{'─' * (W - 2)}╮{RESET}")
        row(f"{ORANGE}πViz{RESET}  {WHITE}v2.1.0{RESET}  {DIM}·  Interactive 3D Engine{RESET}")
        div()
        row(f"{DIM}{platform.system()}  ·  Python {platform.python_version()}  ·  {gpu_info}{RESET}")
        div()
        row(f"{DIM}Orbit{RESET}  {WHITE}L-drag{RESET}    {DIM}Pan{RESET}  {WHITE}Shift+drag{RESET}    {DIM}Zoom{RESET}  {WHITE}Scroll{RESET}")
        row(f"{DIM}H{RESET}  Home    {DIM}G{RESET}  Grid    {DIM}A{RESET}  Axes    {DIM}T{RESET}  Theme    {DIM}0–3{RESET}  Views")
        print(f"{DIM}╰{'─' * (W - 2)}╯{RESET}")
        print(f"  {DIM}ready.{RESET}\n")

    def _init_scene(self, scene: PiVizFX):
        """Initialize a scene with the current context."""
        self.scene = scene
        scene._internal_init(self.ctx, self.wnd, self)

    def _update_ui_scale(self, width, height):
        """Update UI scaling based on actual monitor DPI and physical resolution."""
        # Get monitor info for adaptive scaling
        monitor_info = self._get_monitor_info()

        if monitor_info and monitor_info['dpi']:
            # DPI-based scaling (best method)
            # 96 DPI = 1.0 scale (standard)
            # 144 DPI = 1.5 scale (high-DPI laptop)
            # 192 DPI = 2.0 scale (4K monitor)
            base_dpi = 192
            self.ui_scale = monitor_info['dpi'] / base_dpi

            # Clamp to reasonable range
            self.ui_scale = max(0.8, min(self.ui_scale, 2.5))

            print(f"[UI Scale] Monitor DPI: {monitor_info['dpi']:.0f}, Scale: {self.ui_scale:.2f}")
        else:
            # Fallback: resolution-based scaling
            # Use physical monitor resolution, not window size
            if monitor_info and monitor_info['native_width']:
                ref_w = monitor_info['native_width']
                ref_h = monitor_info['native_height']
            else:
                # Ultimate fallback
                ref_w, ref_h = 1920.0, 1080.0

            scale_x = width / ref_w
            scale_y = height / ref_h
            self.ui_scale = max(0.8, min(scale_x, scale_y))

            print(f"[UI Scale] Resolution-based: {self.ui_scale:.2f}")

        imgui.get_io().font_global_scale = self.ui_scale
        self.overlay.set_scale(self.ui_scale)

    def _get_monitor_info(self):
        """Get current monitor information (DPI, resolution, physical size)."""
        try:
            import screeninfo
            from screeninfo import get_monitors

            monitors = get_monitors()

            if not monitors:
                return None

            # Try to find the monitor containing the window
            window_x, window_y = self.wnd.position if hasattr(self.wnd, 'position') else (0, 0)

            current_monitor = None
            for monitor in monitors:
                # Check if window center is on this monitor
                window_center_x = window_x + self.wnd.width // 2
                window_center_y = window_y + self.wnd.height // 2

                if (monitor.x <= window_center_x < monitor.x + monitor.width and
                        monitor.y <= window_center_y < monitor.y + monitor.height):
                    current_monitor = monitor
                    break

            # Fallback to primary monitor
            if current_monitor is None:
                current_monitor = monitors[0]

            # Calculate DPI
            dpi = None
            if current_monitor.width_mm and current_monitor.width_mm > 0:
                # DPI = pixels / inches
                # mm to inches: divide by 25.4
                dpi_x = current_monitor.width / (current_monitor.width_mm / 25.4)
                dpi_y = current_monitor.height / (current_monitor.height_mm / 25.4)
                dpi = (dpi_x + dpi_y) / 2

            monitor_info = {
                'name': current_monitor.name,
                'dpi': dpi,
                'native_width': current_monitor.width,
                'native_height': current_monitor.height,
                'physical_width_mm': current_monitor.width_mm,
                'physical_height_mm': current_monitor.height_mm,
                'is_primary': current_monitor.is_primary if hasattr(current_monitor, 'is_primary') else False
            }

            print(f"[Monitor] {monitor_info['name']}: {monitor_info['native_width']}x{monitor_info['native_height']}")
            if dpi:
                print(
                    f"[Monitor] DPI: {dpi:.1f}, Physical: {monitor_info['physical_width_mm']}x{monitor_info['physical_height_mm']}mm")

            return monitor_info

        except ImportError:
            print("[Monitor] screeninfo not available, install with: pip install screeninfo")
            return None
        except Exception as e:
            print(f"[Monitor] Detection failed: {e}")
            return None

    def run(self):
        """Run the application."""
        try:
            mglw.run_window_config(self.__class__)
        except Exception as e:
            self._print_crash_report(e)
            sys.exit(1)

    def _print_crash_report(self, e: Exception):
        W = 62
        RED = "\033[91m"
        DIM = "\033[90m"
        WHITE = "\033[97m"
        RESET = "\033[0m"

        def row(content, color=WHITE):
            pad = W - 4 - len(content)
            print(f"{DIM}│{RESET} {color}{content}{' ' * max(0, pad)} {DIM}│{RESET}")

        print(f"\n{DIM}╭{'─' * (W - 2)}╮{RESET}")
        row("Error", RED)
        print(f"{DIM}├{'─' * (W - 2)}┤{RESET}")

        for line in textwrap.wrap(str(e), width=W - 4):
            row(line)

        print(f"{DIM}╰{'─' * (W - 2)}╯{RESET}")
        print(f"\n{DIM}stack trace ↓{RESET}")
        traceback.print_exc()

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

    def _fit_view_to_scene(self):
        """Fit the camera view to show the entire scene."""
        if self.scene and hasattr(self.scene, 'get_bounds'):
            min_bound, max_bound = self.scene.get_bounds()
            self.camera.fit_to_bounds(np.array(min_bound), np.array(max_bound))
            self.camera.set_view('iso')
        else:
            self.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.camera.distance = 15.0
            self.camera.azimuth = 45.0
            self.camera.elevation = 30.0
            self.camera.set_view('iso')

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

        # Resize exporter (reallocates GPU buffers)
        self.exporter.resize(width, height)

        # Resize scene if it has buffers
        if self.scene:
            self.scene.resize(width, height)

    def on_resize(self, width: int, height: int):
        """Event-based resize handler (fallback, main detection is poll-based)."""
        # Handled by _check_for_resize() in render loop
        pass

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

        imgui.new_frame()

        self.viewcube.update(frame_time, self.camera)

        bg = self._theme.background
        self.ctx.clear(*bg[:3])
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        view = self.camera.get_view_matrix()
        proj = self.camera.get_orthographic_matrix() if self.use_orthographic else self.camera.get_projection_matrix()

        pgfx._init_context(self.ctx, view, proj)

        if self.show_grid:
            self.grid_renderer.render(view, proj, self.camera)
        if self.show_axes:
            self.axes_renderer.render(view, proj)

        if self.scene:
            self.scene.render(time, frame_time)
            self.scene.loop(frame_time)
            pgfx.flush_all()

            if hasattr(self.scene, 'render_ui'):
                self.scene.render_ui()

        # Capture before UI overlay so the HUD isn't in the recording
        if self.exporter._recording:
            self.exporter.capture_frame()

        self._render_ui()

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

    def _push_ui_style(self) -> int:
        """Push theme-derived ImGui colors for all native widgets. Returns push count."""
        t = self._theme
        p = t.panel
        a = t.accent
        ah = t.accent_hover

        # Frame bg is slightly offset from panel for depth perception
        d = 0.07 if t.name != 'light' else -0.07
        fr = max(0.0, min(1.0, p[0] + d))
        fg = max(0.0, min(1.0, p[1] + d))
        fb = max(0.0, min(1.0, p[2] + d))

        colors = [
            (imgui.COLOR_TEXT, (*t.text_primary,)),
            (imgui.COLOR_CHECK_MARK, (*a[:3], 1.0)),
            (imgui.COLOR_SLIDER_GRAB, (*a[:3], 0.90)),
            (imgui.COLOR_SLIDER_GRAB_ACTIVE, (*ah[:3], 1.0)),
            (imgui.COLOR_FRAME_BACKGROUND, (fr, fg, fb, 0.82)),
            (imgui.COLOR_FRAME_BACKGROUND_HOVERED, (*a[:3], 0.22)),
            (imgui.COLOR_FRAME_BACKGROUND_ACTIVE, (*a[:3], 0.38)),
            (imgui.COLOR_BUTTON, (*p[:3], 0.75)),
            (imgui.COLOR_BUTTON_HOVERED, (*a[:3], 0.32)),
            (imgui.COLOR_BUTTON_ACTIVE, (*a[:3], 0.52)),
            (imgui.COLOR_HEADER, (*a[:3], 0.20)),
            (imgui.COLOR_HEADER_HOVERED, (*a[:3], 0.32)),
            (imgui.COLOR_HEADER_ACTIVE, (*a[:3], 0.50)),
            (imgui.COLOR_TITLE_BACKGROUND, (*p[:3], 1.0)),
            (imgui.COLOR_TITLE_BACKGROUND_ACTIVE, (*a[:3], 0.88)),
            (imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, (*p[:3], 0.80)),
            (imgui.COLOR_POPUP_BACKGROUND, (*p[:3], 0.97)),
            (imgui.COLOR_SEPARATOR, (*t.text_secondary[:3], 0.40)),
            (imgui.COLOR_SCROLLBAR_BACKGROUND, (*p[:3], 0.30)),
            (imgui.COLOR_SCROLLBAR_GRAB, (*p[:3], 0.75)),
            (imgui.COLOR_SCROLLBAR_GRAB_HOVERED, (*a[:3], 0.55)),
        ]
        for color_id, rgba in colors:
            imgui.push_style_color(color_id, *rgba)
        return len(colors)

    def _pop_ui_style(self, count: int):
        imgui.pop_style_color(count)

    def _render_ui(self):
        """Render all UI components."""
        n = self._push_ui_style()

        if self.show_overlay:
            self.overlay.render()

        self.viewcube.render(self.camera)
        self.view_toggles.render()
        self.toolbar.render()
        self.ui_manager.render()

        self._pop_ui_style(n)

    def on_key_event(self, key, action, modifiers):
        """Handle keyboard events."""
        if not hasattr(self, 'imgui_renderer'):
            return

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
            elif key == self.wnd.keys.H:
                self._fit_view_to_scene()

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

    def _map_mouse_button(self, button: int) -> int:
        """Map window-specific button ID to canonical 1=left, 2=right, 3=middle."""
        if hasattr(self.wnd, 'mouse') and hasattr(self.wnd.mouse, 'LEFT'):
            if button == self.wnd.mouse.LEFT:
                return 1
            if button == self.wnd.mouse.RIGHT:
                return 2
            if button == self.wnd.mouse.MIDDLE:
                return 3
        else:
            if button == 1:
                return 1
            if button == 4:
                return 2
            if button == 2:
                return 3
        return button

    def on_mouse_press_event(self, x, y, button):
        """Handle mouse press events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        self.imgui_renderer.mouse_press_event(x, y, button)

        if not self.imgui_renderer.io.want_capture_mouse:
            mods = getattr(self.wnd, 'modifiers', 0)
            self.camera.on_mouse_press(x, y, self._map_mouse_button(button), mods)
            if self.scene:
                self.scene.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        """Handle mouse release events."""
        if not hasattr(self, 'imgui_renderer'):
            return

        self.imgui_renderer.mouse_release_event(x, y, button)
        self.camera.on_mouse_release(x, y, self._map_mouse_button(button))

        if self.scene:
            self.scene.mouse_release_event(x, y, button)

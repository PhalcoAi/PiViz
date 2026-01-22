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
import os
import platform

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
import traceback
from typing import Optional, Union, Set
from pathlib import Path

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

    # Material Symbols Unicode codes (Google Material Icons)
    ICON_HOME = "\ue88a"  # home
    ICON_EXPAND = "\ue3c9"  # fit_screen
    ICON_CAMERA = "\ue3af"  # photo_camera
    ICON_VIDEO = "\ue04b"  # videocam
    ICON_SUN = "\ue518"  # wb_sunny
    ICON_MOON = "\ue51c"  # nights_stay
    ICON_GRID = "\ue3ec"  # grid_on
    ICON_AXES = "\ue558"  # 3d_rotation
    ICON_CUBE = "\ue164"  # view_in_ar
    ICON_EYE = "\ue8f4"  # visibility
    ICON_COG = "\ue8b8"  # settings

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

        # Load Fonts (with Material Symbols)
        self._icon_font = None
        self._load_fonts()

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

    # =========================================================================
    # FONT LOADING (Cross-platform with Material Symbols)
    # =========================================================================

    def _load_fonts(self):
        """Load fonts including Material Symbols with cross-platform support."""
        io = imgui.get_io()
        io.fonts.clear()

        # 1. Load default text font
        text_font_loaded = self._load_system_font(io)

        if not text_font_loaded:
            io.fonts.add_font_default()
            print("[Font] Using ImGui default font")

        # 2. Load Material Symbols icon font
        self._load_material_icons(io)

        # Build and refresh texture
        # io.fonts.build()
        self.imgui_renderer.refresh_font_texture()

    def _load_system_font(self, io):
        """Load system font with cross-platform detection."""
        system = platform.system()

        # Platform-specific font paths
        font_candidates = []

        if system == "Linux":
            font_candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
                os.path.expanduser("~/.local/share/fonts/DejaVuSans.ttf")
            ]
        elif system == "Windows":
            font_candidates = [
                "C:\\Windows\\Fonts\\segoeui.ttf",  # Segoe UI (modern)
                "C:\\Windows\\Fonts\\arial.ttf",  # Arial (classic)
                "C:\\Windows\\Fonts\\calibri.ttf"  # Calibri
            ]
        elif system == "Darwin":  # macOS
            font_candidates = [
                "/System/Library/Fonts/SFNS.ttf",  # San Francisco
                "/System/Library/Fonts/Helvetica.ttc",  # Helvetica
                "/System/Library/Fonts/HelveticaNeue.ttc",  # Helvetica Neue
                "/Library/Fonts/Arial.ttf"  # Arial
            ]

        # Try each candidate
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    io.fonts.add_font_from_file_ttf(
                        font_path,
                        16.0 * self.ui_scale,
                        None,
                        io.fonts.get_glyph_ranges_default()
                    )
                    print(f"[Font] Loaded: {os.path.basename(font_path)}")
                    return True
                except Exception as e:
                    print(f"[Font] Failed to load {font_path}: {e}")
                    continue

        return False

    def _load_material_icons(self, io):
        """Load Google Material Symbols with automatic download if missing."""
        # Try multiple locations for the font file
        possible_paths = [
            # 1. Local resources directory
            os.path.join(self.resource_dir, 'MaterialSymbolsRounded.ttf'),
            os.path.join(self.resource_dir, 'fonts', 'MaterialSymbolsRounded.ttf'),

            # 2. Package resources
            os.path.join(os.path.dirname(__file__), '..', 'resources', 'MaterialSymbolsRounded.ttf'),
            os.path.join(os.path.dirname(__file__), '..', 'resources', 'fonts', 'MaterialSymbolsRounded.ttf'),

            # 3. User home directory
            os.path.expanduser('~/.piviz/fonts/MaterialSymbolsRounded.ttf'),
        ]

        icon_font_path = None
        for path in possible_paths:
            if os.path.exists(path):
                icon_font_path = path
                break

        # If not found, try to download it
        if icon_font_path is None:
            icon_font_path = self._download_material_symbols()

        if icon_font_path and os.path.exists(icon_font_path):
            try:
                # Material Symbols uses Private Use Area (PUA)
                # Range: U+E000 to U+F8FF
                glyph_ranges = imgui.GlyphRanges([
                    0xe000, 0xf8ff,  # PUA
                    0
                ])

                self._icon_font = io.fonts.add_font_from_file_ttf(
                    icon_font_path,
                    24.0 * self.ui_scale,
                    None,
                    glyph_ranges
                )
                print(f"[Font] Loaded Material Symbols: {os.path.basename(icon_font_path)}")
                return True

            except Exception as e:
                print(f"[Font] Failed to load Material Symbols: {e}")
                self._icon_font = None
                return False
        else:
            print("[Font] Material Symbols not found, using vector fallback")
            self._icon_font = None
            return False

    def _download_material_symbols(self):
        """Download Material Symbols font if not present."""
        try:
            import urllib.request

            # Create fonts directory
            fonts_dir = os.path.join(self.resource_dir, 'fonts')
            os.makedirs(fonts_dir, exist_ok=True)

            target_path = os.path.join(fonts_dir, 'MaterialSymbolsRounded.ttf')

            # Check if already exists
            if os.path.exists(target_path):
                return target_path

            print("[Font] Downloading Material Symbols...")

            # Direct download URL for Material Symbols Rounded (Variable font)
            url = "https://github.com/google/material-design-icons/raw/master/variablefont/MaterialSymbolsRounded%5BFILL%2CGRAD%2Copsz%2Cwght%5D.ttf"

            # Download with timeout
            with urllib.request.urlopen(url, timeout=15) as response:
                font_data = response.read()

            # Write to file
            with open(target_path, 'wb') as f:
                f.write(font_data)

            print(f"[Font] Downloaded Material Symbols to {target_path}")
            return target_path

        except Exception as e:
            print(f"[Font] Download failed: {e}")
            print("[Font] Please manually download MaterialSymbolsRounded.ttf to:")
            print(f"       {os.path.join(self.resource_dir, 'fonts', 'MaterialSymbolsRounded.ttf')}")
            print("[Font] Download from: https://fonts.google.com/icons")
            return None

    # =========================================================================
    # WELCOME BANNER
    # =========================================================================

    def _print_welcome_banner(self, detailed: bool = False):
        """Print a compact, aligned startup banner with controls."""
        if getattr(PiVizStudio, '_banner_printed', False):
            return
        PiVizStudio._banner_printed = True

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

        # --- Print Banner ---
        print("")
        print_border(top=True)
        header = f"{ACCENT_COLOR}πViz Studio {LABEL_COLOR}v1.0.2{RESET}   {TITLE_COLOR}Interactive 3D Engine{RESET}"
        print_line(header)
        print_border(middle=True)
        sys_info = f"{LABEL_COLOR}System:{RESET} {platform.system()} │ Python {platform.python_version()} │ {gpu_info}"
        print_line(sys_info)
        print_border(middle=True)
        mouse = f"{LABEL_COLOR}Mouse:{RESET}  {TEXT_COLOR}L-Drag:{RESET} Orbit  │  {TEXT_COLOR}R-Drag:{RESET} Pan  │  {TEXT_COLOR}Scroll:{RESET} Zoom"
        print_line(mouse)
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
        ref_w, ref_h = 1920.0, 1080.0
        scale_x = width / ref_w
        scale_y = height / ref_h
        self.ui_scale = max(1.0, min(scale_x, scale_y))
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
        """Print formatted crash report."""
        import textwrap
        c_red = "\033[91m"
        c_grey = "\033[90m"
        c_reset = "\033[0m"
        c_white = "\033[97m"
        WIDTH = 60

        def visible_len(s):
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return len(ansi_escape.sub('', s))

        def print_line(content, color=c_red, text_color=c_white):
            v_len = visible_len(content)
            padding = WIDTH - 4 - v_len
            print(f"{color}║ {text_color}{content}{' ' * padding} {color}║{c_reset}")

        print(f"\n{c_red}╔{'═' * (WIDTH - 2)}╗{c_reset}")
        print_line("CRITICAL ERROR", text_color=c_white)
        print(f"{c_red}╠{'═' * (WIDTH - 2)}╣{c_reset}")

        error_msg = str(e)
        wrapped_lines = textwrap.wrap(error_msg, width=WIDTH - 4)
        for line in wrapped_lines:
            print_line(line)

        print(f"{c_red}╚{'═' * (WIDTH - 2)}╝{c_reset}")
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
            self.camera.set_view('iso')
        else:
            self.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.camera.distance = 15.0
            self.camera.azimuth = 45.0
            self.camera.elevation = 30.0
            self.camera.set_view('iso')

    # =========================================================================
    # RESIZE HANDLING
    # =========================================================================

    def _check_for_resize(self):
        """Poll-based resize detection."""
        current_size = (self.wnd.width, self.wnd.height)

        if current_size != self._last_window_size:
            width, height = current_size
            self._last_window_size = current_size

            self.imgui_renderer.resize(width, height)
            self.camera.resize(width, height)

            self._pending_width = width
            self._pending_height = height
            self._resize_timer = 0.15
            self._is_resizing = True

    def _delayed_resize(self, width, height):
        """Execute heavy resize operations after dragging stops."""
        self._is_resizing = False
        self._update_ui_scale(width, height)
        self.exporter.resize(width, height)
        if self.scene:
            self.scene.resize(width, height)

    def on_resize(self, width: int, height: int):
        """Event-based resize handler (fallback)."""
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

            self._check_for_resize()

            if self._resize_timer > 0:
                self._resize_timer -= frame_time
                if self._resize_timer <= 0:
                    self._delayed_resize(self._pending_width, self._pending_height)
                    self._resize_timer = 0

            if self._is_resizing:
                self._render_resize_preview()
                return

            self._render_full(time, frame_time)

        except Exception as e:
            self._print_crash_report(e)
            self.wnd.close()

    def _render_resize_preview(self):
        """Minimal render during resize."""
        bg = self._theme.background
        self.ctx.clear(*bg[:3])
        imgui.new_frame()
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def _render_full(self, time: float, frame_time: float):
        """Full scene render."""
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
        """Draw Grid/Axes/Ortho toggles."""
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
        """Draw top-right control buttons."""
        io = imgui.get_io()

        button_size = 32 * self.ui_scale
        margin = 15 * self.ui_scale
        spacing = 8 * self.ui_scale
        num_buttons = 4
        total_w = (button_size * num_buttons) + (spacing * (num_buttons - 1))

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

        # Home button
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 0,
            "Fit View (Home)", self._fit_view_to_scene
        )
        self._draw_home_icon(draw_list, cx, cy, button_size, col)

        # Record button
        is_rec = self.exporter._recording
        is_flash = is_rec and (int(time_module.time() * 2) % 2 == 0)
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 1,
            "Stop Recording" if is_rec else "Record Video",
            lambda: self.exporter.stop_recording() if is_rec else self.exporter.start_recording(),
            is_active=is_rec, is_flash=is_flash
        )
        rec_col = imgui.get_color_u32_rgba(1, 0.2, 0.2, 1) if is_rec else col
        draw_list.add_circle_filled(cx, cy, button_size * 0.25, rec_col, 16)
        if not is_rec:
            self._draw_video_icon(draw_list, cx, cy, button_size, col)

        # Screenshot button
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 2,
            "Screenshot", lambda: self.exporter.take_screenshot()
        )
        self._draw_camera_icon(draw_list, cx, cy, button_size, col)

        # Theme button
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
        """Draw circular button."""
        io = imgui.get_io()
        cx = start_x + (button_size / 2) + (offset_idx * (button_size + spacing)) + 8
        cy = y + button_size / 2 + 8
        mx, my = io.mouse_pos
        is_hovered = ((mx - cx) ** 2 + (my - cy) ** 2) < (button_size / 2 + 2) ** 2

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

    # =========================================================================
    # ICON DRAWING (with Material Symbols support + vector fallback)
    # =========================================================================

    def _draw_icon_text(self, draw_list, cx, cy, button_size, icon_unicode, col):
        """Draw icon using Material Symbols font with fallback."""
        if self._icon_font:
            imgui.push_font(self._icon_font)
            text_size = imgui.calc_text_size(icon_unicode)
            text_x = cx - text_size.x / 2
            text_y = cy - text_size.y / 2
            draw_list.add_text(text_x, text_y, col, icon_unicode)
            imgui.pop_font()
            return True
        return False

    def _draw_home_icon(self, draw_list, cx, cy, button_size, col):
        """Draw home icon."""
        if not self._draw_icon_text(draw_list, cx, cy, button_size, self.ICON_HOME, col):
            s = button_size * 0.18
            draw_list.add_line(cx - s * 1.3, cy - s * 0.1, cx, cy - s * 1.5, col, 2.0)
            draw_list.add_line(cx, cy - s * 1.5, cx + s * 1.3, cy - s * 0.1, col, 2.0)
            draw_list.add_rect(cx - s * 1.0, cy - s * 0.1, cx + s * 1.0, cy + s * 1.2, col, rounding=0.0, thickness=2.0)

    def _draw_camera_icon(self, draw_list, cx, cy, button_size, col):
        """Draw camera icon."""
        if not self._draw_icon_text(draw_list, cx, cy, button_size, self.ICON_CAMERA, col):
            r = button_size * 0.2
            draw_list.add_rect(cx - r * 1.2, cy - r * 0.8, cx + r * 1.2, cy + r * 0.8, col, rounding=2.0, thickness=1.5)
            draw_list.add_circle(cx, cy, r * 0.5, col, num_segments=12, thickness=1.5)
            draw_list.add_rect_filled(cx + r * 0.6, cy - r * 1.1, cx + r * 1.0, cy - r * 0.8, col)

    def _draw_video_icon(self, draw_list, cx, cy, button_size, col):
        """Draw video icon."""
        r = button_size * 0.2
        draw_list.add_rect(cx - r * 1.2, cy - r * 0.8, cx + r * 1.2, cy + r * 0.8, col, rounding=2.0, thickness=1.5)
        draw_list.add_triangle_filled(cx + r * 0.4, cy - r * 0.4, cx + r * 0.4, cy + r * 0.4, cx + r * 1.0, cy, col)

    def _draw_sun_icon(self, draw_list, cx, cy, radius):
        """Draw sun icon."""
        col = imgui.get_color_u32_rgba(*self._theme.text_primary)
        if not self._draw_icon_text(draw_list, cx, cy, radius * 2, self.ICON_SUN, col):
            draw_list.add_circle_filled(cx, cy, radius * 0.45, col, 16)
            num_rays = 8
            for i in range(num_rays):
                angle = (i / num_rays) * 2 * math.pi - math.pi / 8
                inner_r, outer_r = radius * 0.6, radius * 1.0
                x1 = cx + math.cos(angle) * inner_r
                y1 = cy + math.sin(angle) * inner_r
                x2 = cx + math.cos(angle) * outer_r
                y2 = cy + math.sin(angle) * outer_r
                draw_list.add_line(x1, y1, x2, y2, col, 2.0 * self.ui_scale)

    def _draw_moon_icon(self, draw_list, cx, cy, radius):
        """Draw moon icon."""
        col = imgui.get_color_u32_rgba(*self._theme.text_primary)
        if not self._draw_icon_text(draw_list, cx, cy, radius * 2, self.ICON_MOON, col):
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

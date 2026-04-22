"""
Toolbar - Top action buttons for PiViz
=======================================

Circular buttons: Fit View, Record, Screenshot, Theme toggle.
Icon drawing helpers (vector fallback if no icon font).
"""

import imgui
import math
import time as time_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.studio import PiVizStudio


class Toolbar:
    """Top-right circular action buttons."""

    def __init__(self, studio: 'PiVizStudio'):
        self.studio = studio

    def render(self):
        studio = self.studio
        io = imgui.get_io()

        button_size = 32 * studio.ui_scale
        margin = 15 * studio.ui_scale
        spacing = 8 * studio.ui_scale
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

        # Fit View
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 0,
            "Fit View (Home)", studio._fit_view_to_scene
        )
        self._draw_home_icon(draw_list, cx, cy, button_size, col)

        # Record
        is_rec = studio.exporter._recording
        is_flash = is_rec and (int(time_module.time() * 2) % 2 == 0)
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 1,
            "Stop Recording" if is_rec else "Record Video (MP4)",
            lambda: studio.exporter.stop_recording() if is_rec else studio.exporter.start_recording(),
            is_active=is_rec, is_flash=is_flash
        )
        if is_rec:
            stop_col = imgui.get_color_u32_rgba(1.0, 0.25, 0.25, 1.0)
            self._draw_icon_text(draw_list, cx, cy, button_size, studio.ICON_STOP, stop_col)
        else:
            self._draw_video_icon(draw_list, cx, cy, button_size, col)

        # Screenshot
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 2,
            "Take Screenshot (Clean)", lambda: studio.exporter.take_screenshot()
        )
        self._draw_camera_icon(draw_list, cx, cy, button_size, col)

        # Theme toggle
        cx, cy, col = self._draw_circle_button(
            draw_list, start_x, y, button_size, spacing, 3,
            "Toggle Theme (T)", studio.toggle_theme
        )
        if studio._theme_name == 'dark':
            self._draw_moon_icon(draw_list, cx, cy, button_size * 0.32)
        else:
            self._draw_sun_icon(draw_list, cx, cy, button_size * 0.28)

        imgui.end()

    def _draw_circle_button(self, draw_list, start_x, y, button_size, spacing,
                            offset_idx, tooltip, callback, is_active=False, is_flash=False):
        studio = self.studio
        io = imgui.get_io()

        cx = start_x + (button_size / 2) + (offset_idx * (button_size + spacing)) + 8
        cy = y + button_size / 2 + 8

        mx, my = io.mouse_pos
        is_hovered = ((mx - cx) ** 2 + (my - cy) ** 2) < (button_size / 2 + 2) ** 2

        if is_flash:
            bg = (0.8, 0.1, 0.1, 0.8)
        elif is_active:
            bg = (*studio.theme.accent[:3], 0.6)
        elif is_hovered:
            bg = (*studio.theme.accent[:3], 0.3)
        else:
            bg = (*studio.theme.panel[:3], 0.6)

        draw_list.add_circle_filled(cx, cy, button_size / 2 + 2, imgui.get_color_u32_rgba(*bg), 24)

        if is_hovered and imgui.is_mouse_clicked(0):
            callback()

        if is_hovered:
            imgui.set_tooltip(tooltip)

        return cx, cy, imgui.get_color_u32_rgba(*studio.theme.text_primary)

    def _draw_icon_text(self, draw_list, cx, cy, size, icon_unicode, col):
        icon_font = self.studio._icon_font
        if icon_font:
            imgui.push_font(icon_font)
        text_size = imgui.calc_text_size(icon_unicode)
        draw_list.add_text(cx - text_size.x / 2, cy - text_size.y / 2, col, icon_unicode)
        if icon_font:
            imgui.pop_font()

    def _draw_home_icon(self, draw_list, cx, cy, button_size, col):
        if self.studio._icon_font:
            self._draw_icon_text(draw_list, cx, cy, button_size, self.studio.ICON_HOME, col)
            return
        s = button_size * 0.18
        draw_list.add_line(cx - s * 1.3, cy - s * 0.1, cx, cy - s * 1.5, col, 2.0)
        draw_list.add_line(cx, cy - s * 1.5, cx + s * 1.3, cy - s * 0.1, col, 2.0)
        draw_list.add_rect(cx - s, cy - s * 0.1, cx + s, cy + s * 1.2, col, rounding=0.0, thickness=2.0)
        draw_list.add_rect(cx + s * 0.3, cy - s, cx + s * 0.7, cy - s * 0.5, col, rounding=0.0, thickness=1.5)

    def _draw_camera_icon(self, draw_list, cx, cy, button_size, col):
        if self.studio._icon_font:
            self._draw_icon_text(draw_list, cx, cy, button_size, self.studio.ICON_CAMERA, col)
            return
        r = button_size * 0.2
        draw_list.add_rect(cx - r * 1.2, cy - r * 0.8, cx + r * 1.2, cy + r * 0.8, col, rounding=2.0, thickness=1.5)
        draw_list.add_circle(cx, cy, r * 0.5, col, num_segments=12, thickness=1.5)
        draw_list.add_rect_filled(cx + r * 0.6, cy - r * 1.1, cx + r * 1.0, cy - r * 0.8, col)

    def _draw_video_icon(self, draw_list, cx, cy, button_size, col):
        if self.studio._icon_font:
            self._draw_icon_text(draw_list, cx, cy, button_size, self.studio.ICON_VIDEO, col)
            return
        r = button_size * 0.2
        draw_list.add_rect(cx - r * 1.2, cy - r * 0.8, cx + r * 1.2, cy + r * 0.8, col, rounding=2.0, thickness=1.5)
        draw_list.add_triangle_filled(cx + r * 1.2, cy - r * 0.4, cx + r * 1.2, cy + r * 0.4, cx + r * 1.8, cy, col)

    def _draw_sun_icon(self, draw_list, cx, cy, radius):
        col = imgui.get_color_u32_rgba(*self.studio.theme.text_primary)
        if self.studio._icon_font:
            self._draw_icon_text(draw_list, cx, cy, radius * 2, self.studio.ICON_SUN, col)
            return
        draw_list.add_circle_filled(cx, cy, radius * 0.45, col, 16)
        for i in range(8):
            angle = (i / 8) * 2 * math.pi - math.pi / 8
            x1 = cx + math.cos(angle) * radius * 0.6
            y1 = cy + math.sin(angle) * radius * 0.6
            x2 = cx + math.cos(angle) * radius
            y2 = cy + math.sin(angle) * radius
            draw_list.add_line(x1, y1, x2, y2, col, 2.0 * self.studio.ui_scale)

    def _draw_moon_icon(self, draw_list, cx, cy, radius):
        col = imgui.get_color_u32_rgba(*self.studio.theme.text_primary)
        if self.studio._icon_font:
            self._draw_icon_text(draw_list, cx, cy, radius * 2, self.studio.ICON_MOON, col)
            return
        bg_col = imgui.get_color_u32_rgba(*self.studio.theme.background)
        draw_list.add_circle_filled(cx, cy, radius, col, 24)
        draw_list.add_circle_filled(cx + radius * 0.35, cy - radius * 0.35, radius * 0.7, bg_col, 24)
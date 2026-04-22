"""
ViewToggles - Grid / Axes / Ortho checkboxes for PiViz
=======================================================

Rendered in the bottom-right corner.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.studio import PiVizStudio


class ViewToggles:
    """Grid / Axes / Ortho toggle checkboxes."""

    def __init__(self, studio: 'PiVizStudio'):
        self.studio = studio

    def render(self):
        studio = self.studio
        io = imgui.get_io()
        panel = studio.theme.panel

        margin = 15
        # pivot=(1, 1) anchors by the bottom-right corner of the window
        imgui.set_next_window_position(
            io.display_size.x - margin,
            io.display_size.y - margin,
            imgui.ALWAYS,
            pivot_x=1.0,
            pivot_y=1.0
        )
        imgui.set_next_window_size(0, 0)
        # Slight transparency for a floating-chip look
        imgui.set_next_window_bg_alpha(panel[3] * 0.80)

        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *panel)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 6.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (9.0, 7.0))

        flags = (
            imgui.WINDOW_NO_DECORATION |
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )

        imgui.begin("##view_toggles", flags=flags)

        _, studio.show_grid = imgui.checkbox("Grid", studio.show_grid)
        _, studio.show_axes = imgui.checkbox("Axes", studio.show_axes)
        _, studio.use_orthographic = imgui.checkbox("Ortho", studio.use_orthographic)

        imgui.end()
        imgui.pop_style_var(2)
        imgui.pop_style_color()
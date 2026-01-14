# piviz/ui/manager.py
"""
UI Manager for PiViz
====================

Manages UI widgets with automatic layout to avoid overlapping
with ViewCube, toolbar, and performance overlay.
"""

import imgui
from typing import Dict, Optional, TYPE_CHECKING

from .widgets import WidgetBase

if TYPE_CHECKING:
    from ..core.studio import PiVizStudio


class UIManager:
    """
    Manages UI widgets for scenes with automatic positioning.

    Layout zones (automatically avoided):
    - Top-left: Performance overlay
    - Top-right: System stats
    - Bottom-left: ViewCube
    - Bottom-center: Toolbar
    - Right side: User widgets panel

    Usage:
        def setup(self, ui_manager):
            ui_manager.add_widget("my_button", Button(...))
            ui_manager.add_widget("my_slider", Slider(...))
    """

    # Layout constants
    PANEL_WIDTH = 280
    PANEL_PADDING = 15
    SAFE_MARGIN_TOP = 180  # Below performance overlay
    SAFE_MARGIN_BOTTOM = 80  # Above toolbar
    SAFE_MARGIN_LEFT = 150  # Right of ViewCube

    def __init__(self, studio: 'PiVizStudio'):
        self.studio = studio
        self.widgets: Dict[str, WidgetBase] = {}
        self._panel_visible = True
        self._panel_title = "Controls"
        self._panel_position = 'right'  # 'right', 'left', 'top-right'
        self._panel_collapsed = False

    def add_widget(self, name: str, widget: WidgetBase) -> WidgetBase:
        """
        Add a widget to the UI.

        Args:
            name: Unique identifier for the widget
            widget: Widget instance

        Returns:
            The widget instance (for chaining)
        """
        self.widgets[name] = widget
        return widget

    def remove_widget(self, name: str) -> bool:
        """Remove a widget by name."""
        if name in self.widgets:
            del self.widgets[name]
            return True
        return False

    def get_widget(self, name: str) -> Optional[WidgetBase]:
        """Get a widget by name."""
        return self.widgets.get(name)

    def clear(self):
        """Remove all widgets."""
        self.widgets.clear()

    def set_panel_title(self, title: str):
        """Set the panel title."""
        self._panel_title = title

    def set_panel_position(self, position: str):
        """
        Set panel position.

        Args:
            position: 'right', 'left', or 'top-right'
        """
        self._panel_position = position

    def show_panel(self):
        """Show the widget panel."""
        self._panel_visible = True

    def hide_panel(self):
        """Hide the widget panel."""
        self._panel_visible = False

    def toggle_panel(self):
        """Toggle panel visibility."""
        self._panel_visible = not self._panel_visible

    def _calculate_panel_position(self):
        """Calculate safe panel position avoiding UI elements."""
        io = imgui.get_io()
        screen_w = io.display_size.x
        screen_h = io.display_size.y

        if self._panel_position == 'right':
            # Right side, vertically centered in safe zone
            x = screen_w - self.PANEL_WIDTH - self.PANEL_PADDING
            y = self.SAFE_MARGIN_TOP
            max_height = screen_h - self.SAFE_MARGIN_TOP - self.SAFE_MARGIN_BOTTOM

        elif self._panel_position == 'left':
            # Left side, below ViewCube
            x = self.PANEL_PADDING
            y = screen_h - 200 - self.SAFE_MARGIN_BOTTOM  # Above toolbar, below viewcube area
            max_height = 180  # Limited height on left

        elif self._panel_position == 'top-right':
            # Top right, below system stats
            x = screen_w - self.PANEL_WIDTH - self.PANEL_PADDING
            y = self.SAFE_MARGIN_TOP + 50
            max_height = screen_h - self.SAFE_MARGIN_TOP - self.SAFE_MARGIN_BOTTOM - 50

        else:
            # Default to right
            x = screen_w - self.PANEL_WIDTH - self.PANEL_PADDING
            y = self.SAFE_MARGIN_TOP
            max_height = screen_h - self.SAFE_MARGIN_TOP - self.SAFE_MARGIN_BOTTOM

        return x, y, max_height

    def render(self):
        """Render all widgets in an auto-positioned panel."""
        if not self.widgets or not self._panel_visible:
            return

        x, y, max_height = self._calculate_panel_position()

        imgui.set_next_window_position(x, y, imgui.ONCE)
        imgui.set_next_window_size(self.PANEL_WIDTH, 0)
        imgui.set_next_window_size_constraints(
            (self.PANEL_WIDTH, 0),
            (self.PANEL_WIDTH, max_height)
        )

        # Style the panel
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 12))

        flags = imgui.WINDOW_NO_RESIZE

        expanded, self._panel_visible = imgui.begin(
            self._panel_title,
            closable=True,
            flags=flags
        )

        if expanded:
            for name, widget in self.widgets.items():
                if hasattr(widget, 'visible') and not widget.visible:
                    continue

                # Render widget
                widget.render()

                # Add spacing between widgets
                imgui.spacing()

        imgui.end()
        imgui.pop_style_var(2)

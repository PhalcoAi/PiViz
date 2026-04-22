"""
UI Widgets for PiViz
====================

ImGui-based widgets with a simple API.
"""

import imgui
from typing import Callable, Optional, List, Tuple


class WidgetBase:
    """Base class for all widgets."""

    def __init__(self, visible: bool = True):
        self.visible = visible

    def render(self):
        pass


class Label(WidgetBase):
    """Text label widget."""

    def __init__(self,
                 text: str = "",
                 align: str = 'left',
                 color: Optional[Tuple[float, float, float, float]] = None,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.text = text
        self.align = align
        self.color = color or (0.9, 0.9, 0.9, 1.0)

    def render(self):
        if not self.visible:
            return
        imgui.text_colored(self.text, *self.color)


class Button(WidgetBase):
    """Clickable button widget."""

    def __init__(self,
                 label: str = "Button",
                 callback: Optional[Callable[[], None]] = None,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.label = label
        self.callback = callback

    def render(self):
        if not self.visible:
            return
        if imgui.button(self.label):
            if self.callback:
                self.callback()


class Slider(WidgetBase):
    """Value slider widget."""

    def __init__(self,
                 label: str = "Value",
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 initial_val: float = 0.5,
                 callback: Optional[Callable[[float], None]] = None,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.callback = callback

    def render(self):
        if not self.visible:
            return
        changed, new_value = imgui.slider_float(self.label, self.value, self.min_val, self.max_val)
        if changed:
            self.value = new_value
            if self.callback:
                self.callback(new_value)

    def set_value(self, value: float):
        self.value = value


class Checkbox(WidgetBase):
    """Checkbox widget."""

    def __init__(self,
                 label: str = "Option",
                 is_checked: bool = False,
                 callback: Optional[Callable[[bool], None]] = None,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.label = label
        self.is_checked = is_checked
        self.callback = callback

    def render(self):
        if not self.visible:
            return
        changed, new_value = imgui.checkbox(self.label, self.is_checked)
        if changed:
            self.is_checked = new_value
            if self.callback:
                self.callback(new_value)


class ToggleSwitch(WidgetBase):
    """Toggle switch widget (styled checkbox)."""

    def __init__(self,
                 label: str = "",
                 is_on: bool = False,
                 callback: Optional[Callable[[bool], None]] = None,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.label = label
        self.is_on = is_on
        self.callback = callback

    def render(self):
        if not self.visible:
            return
        display_label = self.label if self.label else "##toggle"
        changed, new_value = imgui.checkbox(display_label, self.is_on)
        if changed:
            self.is_on = new_value
            if self.callback:
                self.callback(new_value)


class TextInput(WidgetBase):
    """Text input field widget."""

    def __init__(self,
                 label: str = "##input",
                 initial_text: str = "",
                 callback: Optional[Callable[[str], None]] = None,
                 max_length: int = 256,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.label = label
        self.text = initial_text
        self.callback = callback
        self.max_length = max_length

    def render(self):
        if not self.visible:
            return
        changed, new_text = imgui.input_text(self.label, self.text, self.max_length)
        if changed:
            self.text = new_text
            if self.callback:
                self.callback(new_text)


class Dropdown(WidgetBase):
    """Dropdown selection widget."""

    def __init__(self,
                 label: str = "##dropdown",
                 options: Optional[List[str]] = None,
                 selected_index: int = 0,
                 callback: Optional[Callable[[str], None]] = None,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.label = label
        self.options = options or []
        self.selected_index = selected_index
        self.callback = callback

    def render(self):
        if not self.visible:
            return
        current = self.options[self.selected_index] if self.options else ""
        if imgui.begin_combo(self.label, current):
            for i, option in enumerate(self.options):
                is_selected = (i == self.selected_index)
                if imgui.selectable(option, is_selected)[0]:
                    self.selected_index = i
                    if self.callback:
                        self.callback(option)
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

    @property
    def selected_option(self) -> str:
        return self.options[self.selected_index] if self.options else ""


class ProgressBar(WidgetBase):
    """Progress bar widget."""

    def __init__(self,
                 label: str = "",
                 min_val: float = 0.0,
                 max_val: float = 100.0,
                 value: float = 0.0,
                 visible: bool = True):
        super().__init__(visible=visible)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value

    def render(self):
        if not self.visible:
            return
        fraction = (self.value - self.min_val) / (self.max_val - self.min_val)
        fraction = max(0.0, min(1.0, fraction))
        overlay = self.label if self.label else f"{self.value:.0f}%"
        imgui.progress_bar(fraction, (0, 0), overlay)

    def set_value(self, value: float):
        self.value = value
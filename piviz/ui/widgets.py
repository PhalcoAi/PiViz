# piviz/ui/widgets.py
"""
UI Widgets for PiViz
====================

ImGui-based widgets with a simple API.
Maintains backward compatibility with PhalcoPulse widget signatures.
"""

import imgui
from typing import Callable, Optional, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class WidgetBase:
    """Base class for all widgets."""
    rect: Tuple[int, int, int, int]  # (x, y, width, height)
    visible: bool = True
    
    def render(self):
        """Render the widget. Override in subclasses."""
        pass


class Label(WidgetBase):
    """Text label widget."""
    
    def __init__(self, 
                 rect: Tuple[int, int, int, int],
                 text: str = "",
                 align: str = 'left',
                 color: Optional[Tuple[float, float, float, float]] = None):
        self.rect = rect
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
                 rect: Tuple[int, int, int, int],
                 text: str = "Button",
                 callback: Optional[Callable[[], None]] = None):
        self.rect = rect
        self.text = text
        self.callback = callback
    
    def render(self):
        if not self.visible:
            return
        if imgui.button(self.text, width=self.rect[2]):
            if self.callback:
                self.callback()


class Slider(WidgetBase):
    """Value slider widget."""
    
    def __init__(self,
                 rect: Tuple[int, int, int, int],
                 label: str = "Value",
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 initial_val: float = 0.5,
                 callback: Optional[Callable[[float], None]] = None):
        self.rect = rect
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.callback = callback
    
    def render(self):
        if not self.visible:
            return
        changed, new_value = imgui.slider_float(
            self.label, self.value, self.min_val, self.max_val
        )
        if changed:
            self.value = new_value
            if self.callback:
                self.callback(new_value)
    
    def set_value(self, value: float):
        self.value = value


class Checkbox(WidgetBase):
    """Checkbox widget."""
    
    def __init__(self,
                 rect: Tuple[int, int, int, int],
                 label: str = "Option",
                 is_checked: bool = False,
                 callback: Optional[Callable[[bool], None]] = None):
        self.rect = rect
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
                 rect: Tuple[int, int, int, int],
                 is_on: bool = False,
                 callback: Optional[Callable[[bool], None]] = None,
                 label: str = ""):
        self.rect = rect
        self.is_on = is_on
        self.callback = callback
        self.label = label
    
    def render(self):
        if not self.visible:
            return
        # Use checkbox with custom styling
        label = self.label if self.label else "##toggle"
        changed, new_value = imgui.checkbox(label, self.is_on)
        if changed:
            self.is_on = new_value
            if self.callback:
                self.callback(new_value)


class TextInput(WidgetBase):
    """Text input field widget."""
    
    def __init__(self,
                 rect: Tuple[int, int, int, int],
                 initial_text: str = "",
                 callback: Optional[Callable[[str], None]] = None,
                 label: str = "##input",
                 max_length: int = 256):
        self.rect = rect
        self.text = initial_text
        self.callback = callback
        self.label = label
        self.max_length = max_length
    
    def render(self):
        if not self.visible:
            return
        changed, new_text = imgui.input_text(
            self.label, self.text, self.max_length
        )
        if changed:
            self.text = new_text
            if self.callback:
                self.callback(new_text)


class Dropdown(WidgetBase):
    """Dropdown selection widget."""
    
    def __init__(self,
                 rect: Tuple[int, int, int, int],
                 options: List[str],
                 selected_index: int = 0,
                 callback: Optional[Callable[[str], None]] = None,
                 label: str = "##dropdown"):
        self.rect = rect
        self.options = options
        self.selected_index = selected_index
        self.callback = callback
        self.label = label
    
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
                 rect: Tuple[int, int, int, int],
                 min_val: float = 0.0,
                 max_val: float = 100.0,
                 value: float = 0.0,
                 label: str = ""):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
    
    def render(self):
        if not self.visible:
            return
        fraction = (self.value - self.min_val) / (self.max_val - self.min_val)
        fraction = max(0.0, min(1.0, fraction))
        overlay = self.label if self.label else f"{self.value:.0f}%"
        imgui.progress_bar(fraction, (self.rect[2], self.rect[3]), overlay)
    
    def set_value(self, value: float):
        self.value = value

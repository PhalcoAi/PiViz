# piviz/core/wrapper.py
"""
PiViz Script Wrapper
====================

Provides a VPython-like scripting interface for PiViz.
Allows users to write simple scripts without defining classes.
"""

import sys
import time
from typing import Callable, Optional
from .studio import PiVizStudio
from .scene import PiVizFX
from ..graphics import primitives as pgfx
from ..ui.widgets import Button, Slider, Checkbox, Label, ToggleSwitch

# Global state for the script wrapper
_setup_func: Optional[Callable] = None
_update_func: Optional[Callable] = None
_studio_instance: Optional[PiVizStudio] = None
_scene_instance: Optional['ScriptScene'] = None


class ScriptScene(PiVizFX):
    """Internal scene class that delegates to user functions."""
    
    def setup(self):
        if _setup_func:
            _setup_func()
            
    def render(self, time_val, dt):
        if _update_func:
            _update_func(dt)


def run(setup: Optional[Callable] = None, update: Optional[Callable] = None):
    """
    Run a PiViz application with the given setup and update functions.
    
    Args:
        setup: Function called once at startup.
        update: Function called every frame with delta_time.
    """
    global _setup_func, _update_func, _studio_instance, _scene_instance
    
    _setup_func = setup
    _update_func = update
    
    _scene_instance = ScriptScene()
    _studio_instance = PiVizStudio(scene_fx=_scene_instance)
    _studio_instance.run()


# ============================================================
# UI Shortcuts
# ============================================================

def add_button(text, callback):
    """Add a button to the UI panel."""
    if _scene_instance and _scene_instance.ui_manager:
        _scene_instance.ui_manager.add_widget(text, Button(text, callback))

def add_slider(label, min_val, max_val, initial, callback):
    """Add a slider to the UI panel."""
    if _scene_instance and _scene_instance.ui_manager:
        _scene_instance.ui_manager.add_widget(label, Slider(label, min_val, max_val, initial, callback))

def add_checkbox(label, initial, callback):
    """Add a checkbox to the UI panel."""
    if _scene_instance and _scene_instance.ui_manager:
        _scene_instance.ui_manager.add_widget(label, Checkbox(label, initial, callback))

def add_label(text):
    """Add a text label to the UI panel."""
    if _scene_instance and _scene_instance.ui_manager:
        _scene_instance.ui_manager.add_widget(text, Label(text))


# ============================================================
# VPython-like API Shortcuts
# ============================================================

def sphere(pos=(0,0,0), radius=1.0, color=(1,1,1)):
    """Draw a sphere immediately (for use in update loop)."""
    pgfx.draw_sphere(center=pos, radius=radius, color=color)

def box(pos=(0,0,0), size=(1,1,1), color=(1,1,1), axis=(0,0,0)):
    """Draw a box/cube immediately."""
    # Convert axis vector to rotation angles if needed, 
    # but for now we just pass rotation as Euler angles in 'axis' for simplicity
    # or we could implement vector-to-euler conversion.
    # Assuming 'axis' here means rotation in Euler angles for direct mapping to draw_cube
    pgfx.draw_cube(center=pos, size=size, color=color, rotation=axis)

def cylinder(pos=(0,0,0), axis=(1,0,0), radius=1.0, color=(1,1,1)):
    """Draw a cylinder from pos to pos+axis."""
    start = pos
    end = (pos[0]+axis[0], pos[1]+axis[1], pos[2]+axis[2])
    pgfx.draw_cylinder(start=start, end=end, radius=radius, color=color)

def cone(pos=(0,0,0), axis=(1,0,0), radius=1.0, color=(1,1,1)):
    """Draw a cone from pos to pos+axis."""
    base = pos
    tip = (pos[0]+axis[0], pos[1]+axis[1], pos[2]+axis[2])
    pgfx.draw_cone(base=base, tip=tip, radius=radius, color=color)

def arrow(pos=(0,0,0), axis=(1,0,0), color=(1,1,1), shaftwidth=0.1):
    """Draw an arrow from pos to pos+axis."""
    start = pos
    end = (pos[0]+axis[0], pos[1]+axis[1], pos[2]+axis[2])
    pgfx.draw_arrow(start=start, end=end, color=color, width_radius=shaftwidth)

def ring(pos=(0,0,0), axis=(0,0,1), radius=1.0, thickness=0.1, color=(1,1,1)):
    """Draw a ring (torus approximation)."""
    # Not implemented in primitives yet, could approximate with cylinders
    pass

def curve(points, color=(1,1,1), radius=0.1):
    """Draw a curve/line through points."""
    pgfx.draw_path(points, color=color, width=radius*10) # width is in pixels for lines

def rate(fps):
    """Limit frame rate (sleep)."""
    # In a real game loop, this is handled by the windowing system (vsync).
    # But we can add a small sleep if needed.
    target_dt = 1.0 / fps
    # This is a naive implementation, real rate control happens in the engine loop
    pass

# Expose vector class if we had one, or just use tuples/numpy
vec = lambda x, y, z: (x, y, z)

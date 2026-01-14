# piviz/__init__.py
"""
PiViz - Modern 3D Visualization Framework
==========================================

A high-performance, GPU-accelerated visualization framework built on ModernGL.
Designed for scientific visualization, simulations, and interactive 3D graphics.

Backward compatible with PhalcoPulse API.

Basic Usage:
    from piviz import PiVizStudio, PiVizFX, pgfx

    class MyScene(PiVizFX):
        def setup(self):
            pass
        
        def render(self, time, dt):
            pgfx.draw_cube(center=(0, 0, 0), size=1.0, color=(1, 0.5, 0))
    
    if __name__ == '__main__':
        studio = PiVizStudio(scene_fx=MyScene())
        studio.run()
"""

__version__ = "0.1.0"
__author__ = "Yogesh Phalak"

from .core.studio import PiVizStudio
from .core.scene import PiVizFX
from .core.camera import Camera
from .core.usd_player import USDPlayer

from .ui.manager import UIManager
from .ui.widgets import (
    Label, Button, Slider, Checkbox, ToggleSwitch,
    TextInput, Dropdown, ProgressBar
)
from .graphics.colors import Colors, Palette, Colormap
from .graphics import primitives as pgfx


def play_usd(file_path: str):
    """
    Launch the native PiViz USD Player for the given file.

    Args:
        file_path (str): Path to .usdc or .usd file.
    """
    # Create the scene
    player_scene = USDPlayer(file_path)

    class AutoPlayer(USDPlayer):
        def setup(self, ui_manager):
            # Call parent setup to load file
            super().setup()
            # Inject controls
            self.render_ui_controls(ui_manager)

    # Run studio
    PiVizStudio(scene_fx=AutoPlayer(file_path)).run()


# Backward compatibility aliases (PhalcoPulse)
PhalcoPulseStudio = PiVizStudio
PhalcoPulseFX = PiVizFX

# UI widgets
from .ui.widgets import (
    Label,
    Button,
    Slider,
    Checkbox,
    ToggleSwitch,
    TextInput,
    Dropdown,
    ProgressBar,
)

__all__ = [
    # Main classes
    'PiVizStudio',
    'PiVizFX',
    'pgfx',

    # Backward compatibility
    'PhalcoPulseStudio',
    'PhalcoPulseFX',

    # UI Widgets
    'Label',
    'Button',
    'Slider',
    'Checkbox',
    'ToggleSwitch',
    'TextInput',
    'Dropdown',
    'ProgressBar',
]

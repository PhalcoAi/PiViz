"""
Base Scene Class for PiViz
==========================

Users inherit from PiVizFX to create custom visualizations.
Provides backward compatibility with PhalcoPulse API.
"""

from typing import Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import moderngl
    from .studio import PiVizStudio
from ..graphics import primitives as pgfx


class PiVizFX:
    """
    Base class for all PiViz visualizations.
    
    Override setup() for initialization and render() for drawing.
    The graphics context and window are available as self.ctx and self.wnd.
    
    Example:
        class MyScene(PiVizFX):
            def setup(self):
                self.angle = 0
            
            def render(self, time, dt):
                self.angle += dt * 45
                pgfx.draw_cube(rotation=(0, 0, self.angle))
    """

    def __init__(self):
        """Initialize scene. Called before OpenGL context is available."""
        self.ctx: Optional['moderngl.Context'] = None
        self.wnd = None
        self.studio: Optional['PiVizStudio'] = None
        self._initialized = False
        self.scale = 1.0

    def _internal_init(self, ctx, wnd, studio):
        """Internal initialization with OpenGL context. Called by Studio."""
        self.ctx = ctx
        self.wnd = wnd
        self.studio = studio

        if wnd:
            self.resize(*wnd.size)

        try:
            self.setup()
        except TypeError:
            # Fall back to PhalcoPulse signature with ui_manager
            self.setup(studio.ui_manager)

        self._initialized = True

    def setup(self, ui_manager=None):
        """
        Called once after OpenGL context is ready.
        Override this to initialize your scene.
        
        Args:
            ui_manager: (Optional) UI manager for adding widgets.
                       Included for backward compatibility with PhalcoPulse.
        """
        pass

    def render(self, time: float, frame_time: float):
        """
        Called every frame to render the scene.
        Override this to draw your visualization.
        
        Args:
            time: Total elapsed time in seconds
            frame_time: Time since last frame (delta time)
        """
        pass

    # Backward compatibility alias
    def loop(self, delta_time: float):
        """
        PhalcoPulse compatibility alias for render().
        Override render() instead for new code.
        """
        pass

    def resize(self, width: int, height: int):
        """Called when window is resized."""
        self.scale = self.calculate_scale(width, height)

    def calculate_scale(self, width: int, height: int) -> float:
        """
        Calculate scale factor based on window size.
        Uses the smaller dimension ratio to ensure consistent scaling
        across different aspect ratios (e.g. ultrawide monitors).
        """
        ref_w, ref_h = 1920.0, 1080.0
        scale_x = width / ref_w
        scale_y = height / ref_h
        return max(1.0, min(scale_x, scale_y))

    def get_bounds(self) -> Tuple[tuple, tuple]:
        """
        Get the bounding box of the scene.
        Uses automatically tracked bounds from primitives.
        
        Returns:
            (min_bound, max_bound) where each is (x, y, z)
        """
        return pgfx.get_scene_bounds()

    def key_event(self, key, action, modifiers):
        """Called on keyboard events."""
        pass

    def mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        """Called when mouse moves (no buttons pressed)."""
        pass

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        """Called when mouse is dragged (button held)."""
        pass

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        """Called on mouse scroll."""
        pass

    def mouse_press_event(self, x: int, y: int, button: int):
        """Called when mouse button is pressed."""
        pass

    def mouse_release_event(self, x: int, y: int, button: int):
        """Called when mouse button is released."""
        pass

    @property
    def ui_manager(self):
        """Access to UI manager for adding widgets."""
        if self.studio:
            return self.studio.ui_manager
        return None

    @property
    def camera(self):
        """Access to camera for view control."""
        if self.studio:
            return self.studio.camera
        return None

    @property
    def overlay(self):
        """Access to performance overlay."""
        if self.studio:
            return self.studio.overlay
        return None


# Backward compatibility alias
PhalcoPulseFX = PiVizFX

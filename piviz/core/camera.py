# piviz/core/camera.py
"""
3D Camera Controller
=====================================

Navigation Controls (SolidWorks Standard):
- Middle Mouse Drag: Orbit (rotate view around target)
  - Drag right → Scene appears to rotate left (camera orbits right)
  - Drag up → Scene appears to tilt down (camera orbits up)
- Middle Mouse + Shift (or Right Mouse): Pan
  - Drag right → Scene moves right
  - Drag up → Scene moves up
- Scroll Wheel: Zoom
  - Scroll up → Zoom in
  - Scroll down → Zoom out
- Arrow Keys: Rotate view

Coordinate System: Z-up (engineering convention)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import math


@dataclass
class CameraState:
    """Immutable camera state for save/restore."""
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    distance: float = 15.0
    azimuth: float = 45.0
    elevation: float = 30.0
    fov: float = 45.0
    near: float = 0.01
    far: float = 1000.0


class Camera:
    """
    Orbital camera with Z-up coordinate system.

    The camera orbits around a target point. Azimuth is the horizontal
    angle (0° = looking from +Y toward origin), elevation is the vertical
    angle (0° = horizontal, 90° = looking from above).
    """

    VIEWS = {
        'front': (0, 0),
        'back': (180, 0),
        'left': (-90, 0),
        'right': (90, 0),
        'top': (0, 89),
        'bottom': (0, -89),
        'iso': (45, 35),
    }

    def __init__(self, target=(0, 0, 0), distance=15.0, azimuth=45.0, elevation=30.0):
        self.target = np.array(target, dtype=np.float32)
        self.distance = distance
        self.azimuth = azimuth  # Horizontal angle in degrees
        self.elevation = elevation  # Vertical angle in degrees

        self.fov = 45.0
        self.near = 0.01
        self.far = 1000.0
        self.aspect = 16 / 9

        self.min_distance = 0.1
        self.max_distance = 500.0
        self.min_elevation = -89.0
        self.max_elevation = 89.0

        # Sensitivity settings
        self.orbit_sensitivity = 0.4
        self.pan_sensitivity = 0.001
        self.zoom_sensitivity = 0.1
        self.key_rotate_speed = 90.0  # degrees per second

        self._initial_state = self.get_state()

        # Mouse state
        self._is_orbiting = False
        self._is_panning = False

    def get_state(self) -> CameraState:
        return CameraState(
            target=self.target.copy(),
            distance=self.distance,
            azimuth=self.azimuth,
            elevation=self.elevation,
            fov=self.fov,
            near=self.near,
            far=self.far
        )

    def set_state(self, state: CameraState):
        self.target = state.target.copy()
        self.distance = state.distance
        self.azimuth = state.azimuth
        self.elevation = state.elevation
        self.fov = state.fov
        self.near = state.near
        self.far = state.far

    def reset(self):
        self.set_state(self._initial_state)

    def set_view(self, view_name: str):
        if view_name.lower() in self.VIEWS:
            self.azimuth, self.elevation = self.VIEWS[view_name.lower()]

    def get_position(self) -> np.ndarray:
        """Calculate camera world position from spherical coordinates."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)

        # Spherical to Cartesian (Z-up)
        x = self.distance * math.cos(el) * math.sin(az)
        y = self.distance * math.cos(el) * math.cos(az)
        z = self.distance * math.sin(el)

        return self.target + np.array([x, y, z], dtype=np.float32)

    def get_right_vector(self) -> np.ndarray:
        """Get camera's right vector in world space."""
        az = math.radians(self.azimuth)
        # Right is perpendicular to view direction in XY plane
        return np.array([math.cos(az), -math.sin(az), 0], dtype=np.float32)

    def get_up_vector(self) -> np.ndarray:
        """Get camera's up vector in world space."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)

        # Up vector depends on elevation
        x = -math.sin(el) * math.sin(az)
        y = -math.sin(el) * math.cos(az)
        z = math.cos(el)

        return np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        """Calculate view matrix."""
        position = self.get_position()

        forward = self.target - position
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 0, 1], dtype=np.float32)

        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 0.001:
            world_up = np.array([0, 1, 0], dtype=np.float32)
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, position)
        view[1, 3] = -np.dot(up, position)
        view[2, 3] = np.dot(forward, position)

        return view

    def get_projection_matrix(self) -> np.ndarray:
        """Calculate perspective projection matrix."""
        f = 1.0 / math.tan(math.radians(self.fov) / 2)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0

        return proj

    def resize(self, width: int, height: int):
        self.aspect = width / max(height, 1)

    # === Camera Movement (SolidWorks Style) ===

    def orbit(self, dx: float, dy: float):
        """
        Orbit camera around target.

        SolidWorks behavior:
        - Drag mouse RIGHT → Camera orbits RIGHT → Scene appears to rotate LEFT
        - Drag mouse UP → Camera orbits UP → Scene appears to tilt DOWN
        """
        # Positive dx (drag right) should increase azimuth (orbit right)
        self.azimuth += dx * self.orbit_sensitivity

        # Positive dy (drag up) should decrease elevation (look more downward)
        self.elevation += dy * self.orbit_sensitivity

        # Clamp and wrap
        self.elevation = np.clip(self.elevation, self.min_elevation, self.max_elevation)
        self.azimuth = self.azimuth % 360

    def pan(self, dx: float, dy: float):
        """
        Pan camera (move target point).

        SolidWorks behavior:
        - Drag mouse RIGHT → Scene moves RIGHT (target moves right)
        - Drag mouse UP → Scene moves UP (target moves up)
        """
        # Get camera coordinate system
        right = self.get_right_vector()
        up = self.get_up_vector()

        # Scale pan speed with distance for consistent feel
        pan_scale = self.distance * self.pan_sensitivity

        # Move target in screen space direction
        # Drag right → move target right → scene moves right
        self.target += right * (dx * pan_scale)
        # Drag up → move target up → scene moves up
        self.target += up * (dy * pan_scale)

    def zoom(self, delta: float):
        """
        Zoom camera (change distance to target).

        SolidWorks behavior:
        - Scroll UP (positive delta) → Zoom IN (decrease distance)
        - Scroll DOWN (negative delta) → Zoom OUT (increase distance)
        """
        # Positive scroll = zoom in = smaller distance
        factor = 1.0 - delta * self.zoom_sensitivity
        self.distance *= factor
        self.distance = np.clip(self.distance, self.min_distance, self.max_distance)

    def fit_to_bounds(self, min_bound: np.ndarray, max_bound: np.ndarray, padding: float = 1.5):
        """Fit view to show given bounding box."""
        center = (min_bound + max_bound) / 2
        size = np.linalg.norm(max_bound - min_bound)

        self.target = center.astype(np.float32)
        self.distance = size * padding / (2 * math.tan(math.radians(self.fov) / 2))

    # === Input Event Handlers ===

    def on_mouse_press(self, x, y, button, modifiers=None):
        """Handle mouse button press."""
        # Check for shift modifier
        shift_pressed = False
        if modifiers is not None:
            if isinstance(modifiers, int):
                shift_pressed = bool(modifiers & 1)
            else:
                shift_pressed = getattr(modifiers, 'shift', False)

        # Button mapping:
        # 1 = Left, 2 = Right, 3 = Middle (may vary by backend)

        if button == 1:  # Left click
            if shift_pressed:
                self._is_panning = True
            else:
                self._is_orbiting = True

        elif button == 2:  # Right click = Pan
            self._is_panning = True

        elif button == 3:  # Middle click
            if shift_pressed:
                self._is_panning = True
            else:
                self._is_orbiting = True

    def on_mouse_release(self, x, y, button):
        """Handle mouse button release."""
        self._is_orbiting = False
        self._is_panning = False

    def on_mouse_drag(self, x, y, dx, dy):
        """Handle mouse drag."""
        if self._is_orbiting:
            self.orbit(dx, dy)
        elif self._is_panning:
            self.pan(dx, dy)

    def on_mouse_scroll(self, x_offset, y_offset):
        """Handle mouse scroll."""
        self.zoom(y_offset)

    def on_key_hold(self, key_name: str, dt: float):
        """Handle continuous key press for smooth rotation."""
        step = self.key_rotate_speed * dt

        if key_name == 'left':
            self.azimuth -= step
        elif key_name == 'right':
            self.azimuth += step
        elif key_name == 'up':
            self.elevation += step
        elif key_name == 'down':
            self.elevation -= step

        self.elevation = np.clip(self.elevation, self.min_elevation, self.max_elevation)
        self.azimuth = self.azimuth % 360

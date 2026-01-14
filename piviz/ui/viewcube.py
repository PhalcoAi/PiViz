# piviz/ui/viewcube.py
"""
ViewCube - Interactive 3D Orientation Widget
============================================

A 3D cube in the corner that:
- Shows current view orientation (rotates with scene)
- Click faces to snap to orthographic views (Top, Front, Right, etc.)
- Click corners for isometric views
- Click edges for 45° views
- Highlights on hover

Inspired by AutoCAD/Fusion 360/SolidWorks ViewCube.
"""

import imgui
import numpy as np
import math
from typing import TYPE_CHECKING, Optional, Tuple, List

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.theme import Theme


class ViewCube:
    """
    Interactive 3D ViewCube widget.

    Renders a cube that rotates with the camera and allows
    click-to-snap view changes.
    """

    # Face definitions: (name, normal, view_azimuth, view_elevation)
    FACES = {
        'TOP': ((0, 0, 1), 0, 89),
        'BOTTOM': ((0, 0, -1), 0, -89),
        'FRONT': ((0, -1, 0), 0, 0),
        'BACK': ((0, 1, 0), 180, 0),
        'RIGHT': ((1, 0, 0), 90, 0),
        'LEFT': ((-1, 0, 0), -90, 0),
    }

    # Corner definitions: (name, position, azimuth, elevation)
    CORNERS = {
        'iso_ftr': ((1, -1, 1), 45, 35),  # Front-Top-Right
        'iso_ftl': ((-1, -1, 1), -45, 35),  # Front-Top-Left
        'iso_btr': ((1, 1, 1), 135, 35),  # Back-Top-Right
        'iso_btl': ((-1, 1, 1), -135, 35),  # Back-Top-Left
        'iso_fbr': ((1, -1, -1), 45, -35),  # Front-Bottom-Right
        'iso_fbl': ((-1, -1, -1), -45, -35),  # Front-Bottom-Left
        'iso_bbr': ((1, 1, -1), 135, -35),  # Back-Bottom-Right
        'iso_bbl': ((-1, 1, -1), -135, -35),  # Back-Bottom-Left
    }

    def __init__(self, size: int = 120):
        """
        Initialize ViewCube.

        Args:
            size: Widget size in pixels
        """
        self.size = size
        self.padding = 15
        self.cube_scale = 0.35  # Cube size relative to widget

        # Interaction state
        self._hovered_face: Optional[str] = None
        self._hovered_corner: Optional[str] = None
        self._is_hovered = False

        # Animation
        self._animating = False
        self._anim_start_az = 0.0
        self._anim_start_el = 0.0
        self._anim_target_az = 0.0
        self._anim_target_el = 0.0
        self._anim_progress = 0.0
        self._anim_duration = 0.3  # seconds

        # Colors (will be set by theme)
        self._face_color = (0.25, 0.25, 0.28, 0.9)
        self._face_hover_color = (1.0, 0.45, 0.1, 0.95)
        self._edge_color = (0.4, 0.4, 0.45, 1.0)
        self._text_color = (0.9, 0.9, 0.9, 1.0)
        self._bg_color = (0.1, 0.1, 0.12, 0.7)

    def set_theme(self, theme: 'Theme'):
        """Update colors from theme."""
        self._face_color = (*theme.panel[:3], 0.9)
        self._face_hover_color = (*theme.accent[:3], 0.95)
        self._edge_color = (*theme.grid_major[:3], 1.0)
        self._text_color = theme.text_primary
        self._bg_color = (*theme.panel[:3], 0.7)

    def update(self, dt: float, camera: 'Camera') -> bool:
        """
        Update animation state.

        Returns:
            True if camera should be updated from animation
        """
        if not self._animating:
            return False

        self._anim_progress += dt / self._anim_duration

        if self._anim_progress >= 1.0:
            self._anim_progress = 1.0
            self._animating = False
            camera.azimuth = self._anim_target_az
            camera.elevation = self._anim_target_el
            return True

        # Smooth easing (ease-out cubic)
        t = 1.0 - (1.0 - self._anim_progress) ** 3

        # Interpolate angles (handle wrap-around for azimuth)
        az_diff = self._anim_target_az - self._anim_start_az
        if az_diff > 180:
            az_diff -= 360
        elif az_diff < -180:
            az_diff += 360

        camera.azimuth = self._anim_start_az + az_diff * t
        camera.elevation = self._anim_start_el + (self._anim_target_el - self._anim_start_el) * t

        return True

    def _start_animation(self, camera: 'Camera', target_az: float, target_el: float):
        """Start smooth transition to target view."""
        self._animating = True
        self._anim_start_az = camera.azimuth
        self._anim_start_el = camera.elevation
        self._anim_target_az = target_az
        self._anim_target_el = target_el
        self._anim_progress = 0.0

    def render(self, camera: 'Camera') -> Optional[Tuple[float, float]]:
        """
        Render the ViewCube widget.

        Args:
            camera: Current camera for orientation

        Returns:
            (azimuth, elevation) if a view was clicked, None otherwise
        """
        io = imgui.get_io()

        # Position in bottom-left corner
        pos_x = self.padding
        pos_y = io.display_size.y - self.size - self.padding - 50  # Above toolbar

        # Create invisible window for the widget
        imgui.set_next_window_position(pos_x, pos_y)
        imgui.set_next_window_size(self.size, self.size)

        flags = (imgui.WINDOW_NO_DECORATION |
                 imgui.WINDOW_NO_BACKGROUND |
                 imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_NO_SAVED_SETTINGS |
                 imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                 imgui.WINDOW_NO_NAV)

        imgui.begin("##viewcube", flags=flags)

        draw_list = imgui.get_window_draw_list()

        # Widget center
        cx = pos_x + self.size / 2
        cy = pos_y + self.size / 2
        radius = self.size * self.cube_scale

        # Draw background circle
        draw_list.add_circle_filled(cx, cy, radius * 1.4,
                                    imgui.get_color_u32_rgba(*self._bg_color), 32)

        # Get rotation matrix from camera
        az = math.radians(camera.azimuth)
        el = math.radians(camera.elevation)

        # Check mouse position
        mouse_x, mouse_y = io.mouse_pos
        mouse_in_widget = (abs(mouse_x - cx) < radius * 1.3 and
                           abs(mouse_y - cy) < radius * 1.3)

        clicked_view = None
        self._hovered_face = None
        self._hovered_corner = None

        # Project and draw faces (painter's algorithm - draw back to front)
        faces_to_draw = []

        for name, (normal, view_az, view_el) in self.FACES.items():
            # Rotate normal by camera orientation
            rotated = self._rotate_point(normal, az, el)

            # Face center in screen space
            screen_x = cx + rotated[0] * radius
            screen_y = cy - rotated[1] * radius  # Flip Y for screen coords

            # Depth for sorting (Z after rotation)
            depth = rotated[2]

            # Only draw front-facing faces (normal pointing toward camera)
            if depth > -0.1:
                faces_to_draw.append((depth, 'face', name, screen_x, screen_y,
                                      normal, view_az, view_el))

        # Add corners
        for name, (pos, view_az, view_el) in self.CORNERS.items():
            rotated = self._rotate_point(pos, az, el)
            screen_x = cx + rotated[0] * radius * 0.85
            screen_y = cy - rotated[1] * radius * 0.85
            depth = rotated[2]

            if depth > 0:
                faces_to_draw.append((depth, 'corner', name, screen_x, screen_y,
                                      pos, view_az, view_el))

        # Sort by depth (back to front)
        faces_to_draw.sort(key=lambda x: x[0])

        # Draw faces and corners
        for item in faces_to_draw:
            depth, item_type, name, sx, sy, _, view_az, view_el = item

            if item_type == 'face':
                # Check hover
                face_size = radius * 0.6 * (0.5 + depth * 0.5)  # Size based on depth
                is_hovered = (mouse_in_widget and
                              abs(mouse_x - sx) < face_size and
                              abs(mouse_y - sy) < face_size)

                if is_hovered:
                    self._hovered_face = name
                    color = self._face_hover_color
                    if imgui.is_mouse_clicked(0):
                        clicked_view = (view_az, view_el)
                else:
                    # Fade based on depth
                    alpha = 0.5 + depth * 0.4
                    color = (*self._face_color[:3], alpha)

                # Draw face as rounded rectangle
                half = face_size / 2
                draw_list.add_rect_filled(
                    sx - half, sy - half,
                    sx + half, sy + half,
                    imgui.get_color_u32_rgba(*color),
                    rounding=4.0
                )

                # Draw label
                if depth > 0.3:  # Only show label for visible faces
                    label = name[0]  # First letter: T, B, F, etc.
                    text_size = imgui.calc_text_size(label)
                    text_alpha = min(1.0, (depth - 0.3) * 2)
                    draw_list.add_text(
                        sx - text_size.x / 2,
                        sy - text_size.y / 2,
                        imgui.get_color_u32_rgba(*self._text_color[:3], text_alpha),
                        label
                    )

            elif item_type == 'corner':
                # Draw corner as small circle
                corner_radius = radius * 0.12
                is_hovered = (mouse_in_widget and
                              (mouse_x - sx) ** 2 + (mouse_y - sy) ** 2 < corner_radius ** 2 * 4)

                if is_hovered:
                    self._hovered_corner = name
                    color = self._face_hover_color
                    corner_radius *= 1.3
                    if imgui.is_mouse_clicked(0):
                        clicked_view = (view_az, view_el)
                else:
                    alpha = 0.4 + depth * 0.4
                    color = (*self._edge_color[:3], alpha)

                draw_list.add_circle_filled(sx, sy, corner_radius,
                                            imgui.get_color_u32_rgba(*color), 12)

        # Draw axis indicators
        self._draw_axis_indicators(draw_list, cx, cy, radius, az, el)

        imgui.end()

        # Handle click - start animation
        if clicked_view and not self._animating:
            self._start_animation(camera, clicked_view[0], clicked_view[1])

        return clicked_view

    def _rotate_point(self, point: Tuple[float, float, float],
                      azimuth: float, elevation: float) -> np.ndarray:
        """Rotate a point by camera angles."""
        x, y, z = point

        # Rotate around Z (azimuth)
        cos_az, sin_az = math.cos(azimuth), math.sin(azimuth)
        x1 = x * cos_az - y * sin_az
        y1 = x * sin_az + y * cos_az
        z1 = z

        # Rotate around X (elevation)
        cos_el, sin_el = math.cos(elevation), math.sin(elevation)
        x2 = x1
        y2 = y1 * cos_el - z1 * sin_el
        z2 = y1 * sin_el + z1 * cos_el

        return np.array([x2, z2, y2])  # Swap Y/Z for screen space

    def _draw_axis_indicators(self, draw_list, cx: float, cy: float,
                              radius: float, az: float, el: float):
        """Draw RGB axis lines at the center of the cube."""
        axis_length = radius * 0.7  # Longer axes

        # Origin at cube center
        ax_cx = cx
        ax_cy = cy

        # X axis (red)
        x_end = self._rotate_point((-1, 0, 0), az, el)
        draw_list.add_line(
            ax_cx, ax_cy,
            ax_cx + x_end[0] * axis_length,
            ax_cy - x_end[1] * axis_length,
            imgui.get_color_u32_rgba(0.95, 0.3, 0.3, 1.0), 2.5
        )

        # Y axis (green)
        y_end = self._rotate_point((0, -1, 0), az, el)
        draw_list.add_line(
            ax_cx, ax_cy,
            ax_cx + y_end[0] * axis_length,
            ax_cy - y_end[1] * axis_length,
            imgui.get_color_u32_rgba(0.3, 0.95, 0.4, 1.0), 2.5
        )

        # Z axis (blue)
        z_end = self._rotate_point((0, 0, 1), az, el)
        draw_list.add_line(
            ax_cx, ax_cy,
            ax_cx + z_end[0] * axis_length,
            ax_cy - z_end[1] * axis_length,
            imgui.get_color_u32_rgba(0.3, 0.5, 0.95, 1.0), 2.5
        )
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

Coordinate System (matches Camera):
- Z-up
- Azimuth 0° = looking from +Y toward origin (Front view)
- Elevation 0° = horizontal, +90° = top view
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

    # Face definitions: (normal_in_world_space, view_azimuth, view_elevation)
    # Normal points OUTWARD from the face
    # View angles are what the camera should be set to when clicking that face
    #
    # Camera convention (from camera.py):
    #   azimuth=0, elevation=0 → camera at +Y looking toward origin (FRONT)
    #   azimuth=90, elevation=0 → camera at +X looking toward origin (RIGHT)
    #   azimuth=0, elevation=89 → camera above looking down (TOP)
    FACES = {
        'FRONT': ((0, 1, 0), 0, 0),  # Face normal +Y, camera looks from +Y
        'BACK': ((0, -1, 0), 180, 0),  # Face normal -Y, camera looks from -Y
        'RIGHT': ((1, 0, 0), 90, 0),  # Face normal +X, camera looks from +X
        'LEFT': ((-1, 0, 0), -90, 0),  # Face normal -X, camera looks from -X
        'TOP': ((0, 0, 1), 0, 89),  # Face normal +Z, camera looks from above
        'BOTTOM': ((0, 0, -1), 0, -89),  # Face normal -Z, camera looks from below
    }

    # Corner definitions: (position, azimuth, elevation)
    # Positions are cube corner coordinates
    CORNERS = {
        'iso_ftr': ((1, 1, 1), 45, 35),  # Front-Top-Right
        'iso_ftl': ((-1, 1, 1), -45, 35),  # Front-Top-Left
        'iso_btr': ((1, -1, 1), 135, 35),  # Back-Top-Right
        'iso_btl': ((-1, -1, 1), -135, 35),  # Back-Top-Left
        'iso_fbr': ((1, 1, -1), 45, -35),  # Front-Bottom-Right
        'iso_fbl': ((-1, 1, -1), -45, -35),  # Front-Bottom-Left
        'iso_bbr': ((1, -1, -1), 135, -35),  # Back-Bottom-Right
        'iso_bbl': ((-1, -1, -1), -135, -35),  # Back-Bottom-Left
    }

    # Edge definitions: (position, azimuth, elevation)
    EDGES = {
        # Top edges
        'edge_tf': ((0, 1, 1), 0, 45),  # Top-Front
        'edge_tb': ((0, -1, 1), 180, 45),  # Top-Back
        'edge_tr': ((1, 0, 1), 90, 45),  # Top-Right
        'edge_tl': ((-1, 0, 1), -90, 45),  # Top-Left
        # Bottom edges
        'edge_bf': ((0, 1, -1), 0, -45),  # Bottom-Front
        'edge_bb': ((0, -1, -1), 180, -45),  # Bottom-Back
        'edge_br': ((1, 0, -1), 90, -45),  # Bottom-Right
        'edge_bl': ((-1, 0, -1), -90, -45),  # Bottom-Left
        # Middle edges (horizontal)
        'edge_fr': ((1, 1, 0), 45, 0),  # Front-Right
        'edge_fl': ((-1, 1, 0), -45, 0),  # Front-Left
        'edge_br_m': ((1, -1, 0), 135, 0),  # Back-Right
        'edge_bl_m': ((-1, -1, 0), -135, 0),  # Back-Left
    }

    def __init__(self, size: int = 130):
        """
        Initialize ViewCube.

        Args:
            size: Widget size in pixels
        """
        self.size = size
        self.padding = 15
        self.cube_scale = 0.38

        # Interaction state
        self._hovered_face: Optional[str] = None
        self._hovered_corner: Optional[str] = None
        self._hovered_edge: Optional[str] = None
        self._is_hovered = False

        # Animation
        self._animating = False
        self._anim_start_az = 0.0
        self._anim_start_el = 0.0
        self._anim_target_az = 0.0
        self._anim_target_el = 0.0
        self._anim_progress = 0.0
        self._anim_duration = 0.25

        # Hover animation
        self._hover_pulse = 0.0
        self._hover_pulse_speed = 4.0

        # Modern color scheme
        self._face_color = (0.18, 0.19, 0.22, 0.85)
        self._face_color_light = (0.24, 0.25, 0.28, 0.9)
        self._face_hover_color = (0.35, 0.65, 0.95, 0.95)
        self._face_active_color = (0.45, 0.75, 1.0, 1.0)
        self._edge_color = (0.35, 0.36, 0.4, 0.8)
        self._edge_hover_color = (0.5, 0.78, 1.0, 0.9)
        self._corner_color = (0.28, 0.29, 0.33, 0.75)
        self._corner_hover_color = (0.45, 0.75, 1.0, 0.95)
        self._text_color = (0.85, 0.87, 0.9, 1.0)
        self._text_shadow_color = (0.0, 0.0, 0.0, 0.5)
        self._bg_color = (0.08, 0.08, 0.1, 0.6)
        self._bg_ring_color = (0.2, 0.21, 0.24, 0.4)
        self._glow_color = (0.35, 0.65, 0.95, 0.3)

        # Axis colors
        self._axis_x_color = (0.9, 0.35, 0.35, 1.0)
        self._axis_y_color = (0.4, 0.85, 0.45, 1.0)
        self._axis_z_color = (0.35, 0.55, 0.95, 1.0)

    def set_theme(self, theme: 'Theme'):
        """Update colors from theme."""
        accent = theme.accent[:3]
        self._face_hover_color = (*accent, 0.95)
        self._face_active_color = (min(1, accent[0] + 0.1),
                                   min(1, accent[1] + 0.1),
                                   min(1, accent[2] + 0.1), 1.0)
        self._edge_hover_color = (*accent, 0.9)
        self._corner_hover_color = (*accent, 0.95)
        self._glow_color = (*accent, 0.3)
        self._text_color = theme.text_primary

    def update(self, dt: float, camera: 'Camera') -> bool:
        """
        Update animation state.

        Returns:
            True if camera should be updated from animation
        """
        # Update hover pulse
        self._hover_pulse += dt * self._hover_pulse_speed
        if self._hover_pulse > math.pi * 2:
            self._hover_pulse -= math.pi * 2

        if not self._animating:
            return False

        self._anim_progress += dt / self._anim_duration

        if self._anim_progress >= 1.0:
            self._anim_progress = 1.0
            self._animating = False
            camera.azimuth = self._anim_target_az
            camera.elevation = self._anim_target_el
            return True

        # Smooth easing (ease-out expo)
        t = 1.0 - math.pow(2, -10 * self._anim_progress)

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

    def _lerp_color(self, c1: Tuple, c2: Tuple, t: float) -> Tuple:
        """Linear interpolate between two colors."""
        return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(len(c1)))

    def _rotate_point_by_camera(self, point: Tuple[float, float, float],
                                azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        """
        Transform a world-space point to screen-space based on camera orientation.

        This matches the camera's coordinate system:
        - Azimuth: rotation around Z axis (0° = +Y direction)
        - Elevation: rotation up/down from horizontal

        Returns (screen_x_offset, screen_y_offset, depth) where:
        - screen_x_offset: positive = right on screen
        - screen_y_offset: positive = up on screen
        - depth: positive = toward camera (visible)
        """
        x, y, z = point

        # Convert to radians
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)

        # Camera is at spherical coordinates (azimuth, elevation) looking at origin
        # We need to transform the point into camera's view space

        # Step 1: Rotate around Z by -azimuth (align camera direction with +Y)
        cos_az = math.cos(-az)
        sin_az = math.sin(-az)
        x1 = x * cos_az - y * sin_az
        y1 = x * sin_az + y * cos_az
        z1 = z

        # Step 2: Rotate around X by -elevation (align camera direction with horizontal)
        cos_el = math.cos(-el)
        sin_el = math.sin(-el)
        x2 = x1
        y2 = y1 * cos_el - z1 * sin_el
        z2 = y1 * sin_el + z1 * cos_el

        # Now: +Y is toward camera, +X is right, +Z is up
        # Map to screen: x->screen_x, z->screen_y, y->depth
        screen_x = x2  # Right on screen
        screen_y = z2  # Up on screen
        depth = y2  # Toward camera (positive = visible)

        return np.array([screen_x, screen_y, depth])

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
        pos_y = io.display_size.y - self.size - self.padding - 50

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

        # Get camera angles
        az = camera.azimuth
        el = camera.elevation

        # Check mouse position
        mouse_x, mouse_y = io.mouse_pos
        dist_from_center = math.sqrt((mouse_x - cx) ** 2 + (mouse_y - cy) ** 2)
        mouse_in_widget = dist_from_center < radius * 1.5

        clicked_view = None
        self._hovered_face = None
        self._hovered_corner = None
        self._hovered_edge = None

        # Draw background
        self._draw_background(draw_list, cx, cy, radius, mouse_in_widget)

        # Collect all drawable items for depth sorting
        items_to_draw = []

        # Add faces
        for name, (normal, view_az, view_el) in self.FACES.items():
            transformed = self._rotate_point_by_camera(normal, az, el)
            screen_x = cx + transformed[0] * radius
            screen_y = cy - transformed[1] * radius  # Flip Y for screen coords
            depth = transformed[2]

            if depth > -0.2:
                items_to_draw.append({
                    'type': 'face',
                    'name': name,
                    'screen_pos': (screen_x, screen_y),
                    'depth': depth,
                    'view': (view_az, view_el),
                    'normal': normal
                })

        # Add edges
        for name, (pos, view_az, view_el) in self.EDGES.items():
            transformed = self._rotate_point_by_camera(pos, az, el)
            norm = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            screen_x = cx + (transformed[0] / norm) * radius * 0.92
            screen_y = cy - (transformed[1] / norm) * radius * 0.92
            depth = transformed[2] / norm

            if depth > 0.1:
                items_to_draw.append({
                    'type': 'edge',
                    'name': name,
                    'screen_pos': (screen_x, screen_y),
                    'depth': depth,
                    'view': (view_az, view_el),
                    'pos': pos
                })

        # Add corners
        for name, (pos, view_az, view_el) in self.CORNERS.items():
            transformed = self._rotate_point_by_camera(pos, az, el)
            norm = math.sqrt(3)
            screen_x = cx + (transformed[0] / norm) * radius * 0.95
            screen_y = cy - (transformed[1] / norm) * radius * 0.95
            depth = transformed[2] / norm

            if depth > 0.15:
                items_to_draw.append({
                    'type': 'corner',
                    'name': name,
                    'screen_pos': (screen_x, screen_y),
                    'depth': depth,
                    'view': (view_az, view_el),
                    'pos': pos
                })

        # Sort by depth (back to front)
        items_to_draw.sort(key=lambda x: x['depth'])

        # Draw all items
        for item in items_to_draw:
            sx, sy = item['screen_pos']
            depth = item['depth']
            view_az, view_el = item['view']

            if item['type'] == 'face':
                clicked = self._draw_face(draw_list, item['name'], sx, sy, depth,
                                          radius, mouse_x, mouse_y, mouse_in_widget, io)
                if clicked:
                    clicked_view = (view_az, view_el)

            elif item['type'] == 'edge':
                clicked = self._draw_edge(draw_list, item['name'], sx, sy, depth,
                                          radius, mouse_x, mouse_y, mouse_in_widget, io)
                if clicked:
                    clicked_view = (view_az, view_el)

            elif item['type'] == 'corner':
                clicked = self._draw_corner(draw_list, item['name'], sx, sy, depth,
                                            radius, mouse_x, mouse_y, mouse_in_widget, io)
                if clicked:
                    clicked_view = (view_az, view_el)

        # Draw axis indicators on top
        self._draw_axis_indicators(draw_list, cx, cy, radius, az, el)

        imgui.end()

        # Handle click - start animation
        if clicked_view and not self._animating:
            self._start_animation(camera, clicked_view[0], clicked_view[1])

        return clicked_view

    def _draw_background(self, draw_list, cx: float, cy: float,
                         radius: float, is_hovered: bool):
        """Draw the background with subtle visual effects."""
        # Outer glow when hovered
        if is_hovered:
            glow_alpha = 0.15 + 0.05 * math.sin(self._hover_pulse)
            for i in range(3):
                r = radius * (1.5 + i * 0.08)
                alpha = glow_alpha * (1 - i * 0.3)
                draw_list.add_circle(cx, cy, r,
                                     imgui.get_color_u32_rgba(*self._glow_color[:3], alpha),
                                     48, 2.0)

        # Background circle with gradient effect
        steps = 8
        for i in range(steps):
            t = i / steps
            r = radius * (1.35 - t * 0.25)
            alpha = self._bg_color[3] * (0.3 + t * 0.7)
            draw_list.add_circle_filled(cx, cy, r,
                                        imgui.get_color_u32_rgba(*self._bg_color[:3], alpha), 48)

        # Subtle ring
        draw_list.add_circle(cx, cy, radius * 1.25,
                             imgui.get_color_u32_rgba(*self._bg_ring_color), 48, 1.5)

    def _draw_face(self, draw_list, name: str, sx: float, sy: float,
                   depth: float, radius: float, mouse_x: float, mouse_y: float,
                   mouse_in_widget: bool, io) -> bool:
        """Draw a face of the cube. Returns True if clicked."""
        base_size = radius * 0.55
        face_size = base_size * (0.6 + depth * 0.4)

        hit_size = face_size * 1.1
        is_hovered = (mouse_in_widget and
                      abs(mouse_x - sx) < hit_size and
                      abs(mouse_y - sy) < hit_size and
                      self._hovered_corner is None)

        clicked = False
        if is_hovered:
            self._hovered_face = name
            if imgui.is_mouse_clicked(0):
                clicked = True

        if is_hovered:
            pulse = 0.5 + 0.5 * math.sin(self._hover_pulse * 2)
            color = self._lerp_color(self._face_hover_color, self._face_active_color, pulse * 0.3)
            glow_size = face_size * 1.3
            draw_list.add_rect_filled(
                sx - glow_size / 2, sy - glow_size / 2,
                sx + glow_size / 2, sy + glow_size / 2,
                imgui.get_color_u32_rgba(*self._glow_color[:3], 0.2),
                rounding=8.0
            )
        else:
            base_alpha = 0.4 + depth * 0.5
            color = (*self._lerp_color(self._face_color[:3],
                                       self._face_color_light[:3],
                                       depth * 0.5 + 0.25),
                     base_alpha)

        half = face_size / 2
        rounding = 5.0 + depth * 3.0

        draw_list.add_rect_filled(
            sx - half, sy - half,
            sx + half, sy + half,
            imgui.get_color_u32_rgba(*color),
            rounding=rounding
        )

        border_alpha = 0.3 + depth * 0.3
        if is_hovered:
            border_color = (*self._face_hover_color[:3], 0.8)
        else:
            border_color = (*self._edge_color[:3], border_alpha)

        draw_list.add_rect(
            sx - half, sy - half,
            sx + half, sy + half,
            imgui.get_color_u32_rgba(*border_color),
            rounding=rounding,
            thickness=1.5 if is_hovered else 1.0
        )

        if depth > 0.2:
            label = name[0]
            text_size = imgui.calc_text_size(label)
            text_alpha = min(1.0, (depth - 0.2) * 1.5)

            tx = sx - text_size.x / 2
            ty = sy - text_size.y / 2

            shadow_offset = 1.0
            draw_list.add_text(
                tx + shadow_offset, ty + shadow_offset,
                imgui.get_color_u32_rgba(*self._text_shadow_color[:3], text_alpha * 0.5),
                label
            )

            if is_hovered:
                text_color = (1.0, 1.0, 1.0, 1.0)
            else:
                text_color = (*self._text_color[:3], text_alpha)

            draw_list.add_text(tx, ty, imgui.get_color_u32_rgba(*text_color), label)

        return clicked

    def _draw_edge(self, draw_list, name: str, sx: float, sy: float,
                   depth: float, radius: float, mouse_x: float, mouse_y: float,
                   mouse_in_widget: bool, io) -> bool:
        """Draw an edge of the cube. Returns True if clicked."""
        edge_length = radius * 0.18 * (0.6 + depth * 0.4)
        edge_width = radius * 0.06 * (0.6 + depth * 0.4)

        hit_radius = edge_length * 1.2
        dist_sq = (mouse_x - sx) ** 2 + (mouse_y - sy) ** 2
        is_hovered = (mouse_in_widget and
                      dist_sq < hit_radius ** 2 and
                      self._hovered_face is None and
                      self._hovered_corner is None)

        clicked = False
        if is_hovered:
            self._hovered_edge = name
            if imgui.is_mouse_clicked(0):
                clicked = True

        if is_hovered:
            color = self._edge_hover_color
            draw_list.add_circle_filled(sx, sy, edge_length * 1.5,
                                        imgui.get_color_u32_rgba(*self._glow_color[:3], 0.15), 16)
        else:
            alpha = 0.3 + depth * 0.4
            color = (*self._edge_color[:3], alpha)

        draw_list.add_circle_filled(sx, sy, edge_width * (1.5 if is_hovered else 1.0),
                                    imgui.get_color_u32_rgba(*color), 12)

        return clicked

    def _draw_corner(self, draw_list, name: str, sx: float, sy: float,
                     depth: float, radius: float, mouse_x: float, mouse_y: float,
                     mouse_in_widget: bool, io) -> bool:
        """Draw a corner of the cube. Returns True if clicked."""
        corner_radius = radius * 0.09 * (0.7 + depth * 0.3)

        hit_radius = corner_radius * 2.5
        dist_sq = (mouse_x - sx) ** 2 + (mouse_y - sy) ** 2
        is_hovered = (mouse_in_widget and dist_sq < hit_radius ** 2)

        clicked = False
        if is_hovered:
            self._hovered_corner = name
            if imgui.is_mouse_clicked(0):
                clicked = True

        if is_hovered:
            color = self._corner_hover_color
            draw_radius = corner_radius * 1.4

            draw_list.add_circle_filled(sx, sy, corner_radius * 2.5,
                                        imgui.get_color_u32_rgba(*self._glow_color[:3], 0.2), 16)

            draw_list.add_circle(sx, sy, draw_radius * 1.3,
                                 imgui.get_color_u32_rgba(*self._corner_hover_color[:3], 0.5),
                                 16, 1.5)
        else:
            alpha = 0.35 + depth * 0.4
            color = (*self._corner_color[:3], alpha)
            draw_radius = corner_radius

        draw_list.add_circle_filled(sx, sy, draw_radius,
                                    imgui.get_color_u32_rgba(*color), 16)

        return clicked

    def set_theme(self, theme: 'Theme'):
        """Update colors from theme."""
        # Derive all colors from theme
        is_dark = theme.name == 'dark'

        if is_dark:
            self._face_color = (0.18, 0.19, 0.22, 0.85)
            self._face_color_light = (0.24, 0.25, 0.28, 0.9)
            self._edge_color = (0.35, 0.36, 0.4, 0.8)
            self._corner_color = (0.28, 0.29, 0.33, 0.75)
            self._text_color = (0.85, 0.87, 0.9, 1.0)
            self._text_shadow_color = (0.0, 0.0, 0.0, 0.5)
            self._bg_color = (0.08, 0.08, 0.1, 0.6)
            self._bg_ring_color = (0.2, 0.21, 0.24, 0.4)
        else:
            # Light theme - darker cube for visibility
            self._face_color = (0.25, 0.26, 0.3, 0.9)
            self._face_color_light = (0.32, 0.33, 0.38, 0.92)
            self._edge_color = (0.4, 0.4, 0.45, 0.85)
            self._corner_color = (0.35, 0.36, 0.4, 0.8)
            self._text_color = (0.95, 0.95, 0.97, 1.0)
            self._text_shadow_color = (0.0, 0.0, 0.0, 0.3)
            self._bg_color = (0.15, 0.15, 0.18, 0.7)
            self._bg_ring_color = (0.25, 0.25, 0.28, 0.5)

        # Accent-based colors from theme
        accent = theme.accent[:3]
        self._face_hover_color = (*accent, 0.95)
        self._face_active_color = (min(1, accent[0] + 0.1),
                                   min(1, accent[1] + 0.1),
                                   min(1, accent[2] + 0.1), 1.0)
        self._edge_hover_color = (*accent, 0.9)
        self._corner_hover_color = (*accent, 0.95)
        self._glow_color = (*accent, 0.3)

    def _draw_axis_indicators(self, draw_list, cx: float, cy: float,
                              radius: float, az: float, el: float):
        """Draw RGB axis lines with improved visuals."""
        axis_length = radius * 0.6
        axis_width = 2.0

        # Axis directions in world space
        axes = [
            ((1, 0, 0), self._axis_x_color, 'X'),  # +X axis (red)
            ((0, 1, 0), self._axis_y_color, 'Y'),  # +Y axis (green)
            ((0, 0, 1), self._axis_z_color, 'Z'),  # +Z axis (blue)
        ]

        # Sort by depth (back to front)
        axis_depths = []
        for direction, color, label in axes:
            transformed = self._rotate_point_by_camera(direction, az, el)
            axis_depths.append((transformed[2], direction, color, label, transformed))

        axis_depths.sort(key=lambda x: x[0])

        for depth, direction, color, label, transformed in axis_depths:
            end_x = cx + transformed[0] * axis_length
            end_y = cy - transformed[1] * axis_length  # Flip Y

            # Fade based on depth
            alpha = 0.5 + depth * 0.5
            line_color = (*color[:3], max(0.2, alpha))

            # Draw line
            draw_list.add_line(cx, cy, end_x, end_y,
                               imgui.get_color_u32_rgba(*line_color), axis_width)

            # Axis tip indicator
            if depth > -0.3:
                tip_radius = 4.0 * (0.7 + max(0, depth) * 0.3)
                draw_list.add_circle_filled(end_x, end_y, tip_radius,
                                            imgui.get_color_u32_rgba(*line_color), 12)

                # Label near tip
                if depth > 0:
                    label_offset = 10
                    label_x = end_x + (transformed[0] * label_offset if abs(transformed[0]) > 0.3 else 0)
                    label_y = end_y - (transformed[1] * label_offset if abs(transformed[1]) > 0.3 else -label_offset)

                    text_size = imgui.calc_text_size(label)
                    draw_list.add_text(
                        label_x - text_size.x / 2,
                        label_y - text_size.y / 2,
                        imgui.get_color_u32_rgba(*line_color[:3], alpha * 0.9),
                        label
                    )

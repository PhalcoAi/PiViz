# examples/shapes_showcase.py
"""
PiViz Shapes Showcase
=====================

A comprehensive demonstration of all available primitive shapes and drawing
functions in PiViz. This serves as both a visual reference and a code guide
for using the pgfx API.

Primitives Demonstrated:
------------------------
1.  draw_sphere()      - 3D sphere with configurable detail
2.  draw_cube()        - 3D cube with rotation support
3.  draw_cylinder()    - Cylinder between two points
4.  draw_cone()        - Cone from base to tip
5.  draw_plane()       - Flat plane with configurable normal
6.  draw_triangle()    - Single triangle with flat color
7.  draw_face()        - Triangle with per-vertex colors
8.  draw_line()        - Line segment with width
9.  draw_path()        - Connected line segments
10. draw_arrow()       - Arrow with head (cylinder + cone)
11. draw_point()       - Single point/dot
12. draw_particles()   - Batch-optimized point cloud

Material Functions:
-------------------
- set_material_shiny() - Enable specular highlights
- set_material_matte() - Disable specular (clay-like)

Controls:
---------
- 1-4: Switch between showcase sections
- M: Toggle matte/shiny materials
- R: Reset rotation

Author: Yogesh Phalak
"""

import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors, Palette
from piviz.ui import Label, Button, Checkbox
from piviz.graphics.primitives import set_material_shiny, set_material_matte


class ShapesShowcase(PiVizFX):
    """
    Comprehensive showcase of all PiViz primitive shapes.

    Organized into sections:
    - Section 1: Basic Solids (sphere, cube, cylinder, cone)
    - Section 2: Flat Primitives (plane, triangle, face)
    - Section 3: Lines and Paths (line, path, arrow, point)
    - Section 4: Batch Rendering (particles, many shapes)
    """

    def setup(self):
        # Animation state
        self.rotation = 0.0
        self.time_val = 0.0

        # Display options
        self.current_section = 0  # 0 = all, 1-4 = individual sections
        self.use_shiny = True
        self.show_labels = True
        self.animate = True

        # Generate particle data for Section 4
        self._generate_particles()

        # Setup UI
        self._setup_ui()

        # Camera setup
        if self.camera:
            self.camera.set_view('iso')
            self.camera.distance = 25.0

        # Initial material
        set_material_shiny(shiny=True, shininess=48.0, specular=0.5)

    def _generate_particles(self):
        """Generate particle cloud for batch rendering demo."""
        n = 5000
        # Spherical distribution
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.random.uniform(0, np.pi, n)
        r = np.random.uniform(0.5, 2.0, n)

        self.particle_positions = np.zeros((n, 3), dtype='f4')
        self.particle_positions[:, 0] = r * np.sin(phi) * np.cos(theta)
        self.particle_positions[:, 1] = r * np.sin(phi) * np.sin(theta)
        self.particle_positions[:, 2] = r * np.cos(phi) + 1.5

        # Color by height
        heights = self.particle_positions[:, 2]
        h_norm = (heights - heights.min()) / (heights.max() - heights.min())
        self.particle_colors = np.zeros((n, 3), dtype='f4')
        self.particle_colors[:, 0] = h_norm
        self.particle_colors[:, 1] = 0.3
        self.particle_colors[:, 2] = 1 - h_norm

        self.particle_sizes = np.random.uniform(2, 5, n).astype('f4')

    def _setup_ui(self):
        """Setup UI controls."""
        if not self.ui_manager:
            return

        self.ui_manager.set_panel_title("Shapes Showcase")

        self.lbl_section = Label((0, 0, 200, 20), "Section: All Shapes", color=(1, 1, 1))
        self.ui_manager.add_widget("section", self.lbl_section)

        # Section buttons
        self.ui_manager.add_widget("btn_all", Button((0, 0, 60, 25), "All", lambda: self._set_section(0)))
        self.ui_manager.add_widget("btn_s1", Button((0, 0, 60, 25), "Solids", lambda: self._set_section(1)))
        self.ui_manager.add_widget("btn_s2", Button((0, 0, 60, 25), "Flat", lambda: self._set_section(2)))
        self.ui_manager.add_widget("btn_s3", Button((0, 0, 60, 25), "Lines", lambda: self._set_section(3)))
        self.ui_manager.add_widget("btn_s4", Button((0, 0, 60, 25), "Batch", lambda: self._set_section(4)))

        # Options
        self.ui_manager.add_widget(
            "chk_shiny",
            Checkbox((0, 0, 150, 20), "Shiny Materials", self.use_shiny, self._toggle_shiny)
        )
        self.ui_manager.add_widget(
            "chk_animate",
            Checkbox((0, 0, 150, 20), "Animate", self.animate, lambda v: setattr(self, 'animate', v))
        )

    def _set_section(self, section):
        """Switch to a specific section."""
        self.current_section = section
        names = ["All Shapes", "Basic Solids", "Flat Primitives", "Lines & Paths", "Batch Rendering"]
        self.lbl_section.text = f"Section: {names[section]}"

        # Adjust camera for different sections
        if self.camera:
            if section == 4:
                self.camera.distance = 15.0
            else:
                self.camera.distance = 25.0

    def _toggle_shiny(self, value):
        """Toggle shiny/matte materials."""
        self.use_shiny = value
        if value:
            set_material_shiny(shiny=True, shininess=48.0, specular=0.5)
        else:
            set_material_matte()

    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action != 1:  # Press only
            return

        # Number keys for sections
        if key == 49:  # 1
            self._set_section(1)
        elif key == 50:  # 2
            self._set_section(2)
        elif key == 51:  # 3
            self._set_section(3)
        elif key == 52:  # 4
            self._set_section(4)
        elif key == 48:  # 0
            self._set_section(0)
        elif key == 77:  # M
            self._toggle_shiny(not self.use_shiny)
        elif key == 82:  # R
            self.rotation = 0.0

    def render(self, time, dt):
        """Main render loop."""
        # Update animation
        if self.animate:
            self.rotation += dt * 30
            self.time_val = time

        # Draw ground plane (always visible)
        pgfx.draw_plane(size=(20, 20), color=(0.3, 0.3, 0.35))

        # Draw requested section(s)
        if self.current_section == 0 or self.current_section == 1:
            self._draw_section_1_solids()

        if self.current_section == 0 or self.current_section == 2:
            self._draw_section_2_flat()

        if self.current_section == 0 or self.current_section == 3:
            self._draw_section_3_lines()

        if self.current_section == 0 or self.current_section == 4:
            self._draw_section_4_batch()

    # =========================================================================
    # SECTION 1: Basic Solid Shapes
    # =========================================================================

    def _draw_section_1_solids(self):
        """
        Section 1: Basic 3D Solid Shapes

        Demonstrates:
        - draw_sphere(): Spherical shape with adjustable tessellation
        - draw_cube(): Box shape with rotation support
        - draw_cylinder(): Tube between two points
        - draw_cone(): Pointed cone from base to tip
        """
        x_offset = -6 if self.current_section == 0 else 0
        y_offset = 4 if self.current_section == 0 else 0

        # ---------------------------------------------------------------------
        # SPHERE
        # pgfx.draw_sphere(center, radius, color, detail=12)
        #
        # Parameters:
        #   center: (x, y, z) tuple - center position
        #   radius: float - sphere radius
        #   color: (r, g, b) or (r, g, b, a) - color values 0-1
        #   detail: int - tessellation level (higher = smoother, default 12)
        # ---------------------------------------------------------------------
        pgfx.draw_sphere(
            center=(x_offset - 3, y_offset, 1.0),
            radius=1.0,
            color=Palette.Standard10[3],  # Red
            detail=16
        )

        # Animated smaller sphere
        bounce = np.sin(self.time_val * 3) * 0.3
        pgfx.draw_sphere(
            center=(x_offset - 3, y_offset, 2.5 + bounce),
            radius=0.3,
            color=Palette.Standard10[1],  # Orange
            detail=12
        )

        # ---------------------------------------------------------------------
        # CUBE
        # pgfx.draw_cube(center, size, color, rotation=(0,0,0))
        #
        # Parameters:
        #   center: (x, y, z) tuple - center position
        #   size: float or (w, h, d) - uniform or per-axis size
        #   color: (r, g, b) or (r, g, b, a) - color values 0-1
        #   rotation: (rx, ry, rz) - rotation in degrees (currently Z only)
        # ---------------------------------------------------------------------
        pgfx.draw_cube(
            center=(x_offset, y_offset, 1.0),
            size=1.5,
            color=Palette.Standard10[2],  # Green
            rotation=(0, 0, self.rotation)
        )

        # Non-uniform cube (rectangular box)
        pgfx.draw_cube(
            center=(x_offset, y_offset, 2.8),
            size=(0.5, 0.5, 1.0),  # Tall thin box
            color=Palette.Standard10[4],  # Purple
            rotation=(0, 0, -self.rotation * 0.5)
        )

        # ---------------------------------------------------------------------
        # CYLINDER
        # pgfx.draw_cylinder(start, end, radius, color, detail=16)
        #
        # Parameters:
        #   start: (x, y, z) tuple - starting point
        #   end: (x, y, z) tuple - ending point
        #   radius: float - cylinder radius
        #   color: (r, g, b) or (r, g, b, a) - color values 0-1
        #   detail: int - number of sides (higher = smoother, default 16)
        # ---------------------------------------------------------------------
        pgfx.draw_cylinder(
            start=(x_offset + 3, y_offset, 0),
            end=(x_offset + 3, y_offset, 2.5),
            radius=0.6,
            color=Palette.Standard10[0],  # Blue
            detail=24
        )

        # Angled cylinder
        angle = np.radians(self.rotation)
        end_x = x_offset + 3 + np.cos(angle) * 1.5
        end_y = y_offset + np.sin(angle) * 1.5
        pgfx.draw_cylinder(
            start=(x_offset + 3, y_offset, 2.5),
            end=(end_x, end_y, 3.5),
            radius=0.2,
            color=Palette.Standard10[5],  # Brown
            detail=12
        )

        # ---------------------------------------------------------------------
        # CONE
        # pgfx.draw_cone(base, tip, radius, color, detail=16)
        #
        # Parameters:
        #   base: (x, y, z) tuple - center of the base circle
        #   tip: (x, y, z) tuple - point of the cone
        #   radius: float - base radius
        #   color: (r, g, b) or (r, g, b, a) - color values 0-1
        #   detail: int - number of sides (higher = smoother, default 16)
        # ---------------------------------------------------------------------
        pgfx.draw_cone(
            base=(x_offset + 6, y_offset, 0),
            tip=(x_offset + 6, y_offset, 2.5),
            radius=0.8,
            color=Palette.Standard10[6],  # Pink
            detail=24
        )

        # Inverted cone (tip at bottom)
        pgfx.draw_cone(
            base=(x_offset + 6, y_offset, 4.0),
            tip=(x_offset + 6, y_offset, 2.8),
            radius=0.5,
            color=Palette.Standard10[7],  # Grey
            detail=16
        )

    # =========================================================================
    # SECTION 2: Flat Primitives
    # =========================================================================

    def _draw_section_2_flat(self):
        """
        Section 2: Flat/2D Primitives

        Demonstrates:
        - draw_plane(): Flat rectangular surface
        - draw_triangle(): Single triangle with uniform color
        - draw_face(): Triangle with per-vertex colors (gradient)
        """
        x_offset = -6 if self.current_section == 0 else 0
        y_offset = -4 if self.current_section == 0 else 0

        # ---------------------------------------------------------------------
        # PLANE
        # pgfx.draw_plane(size, color, center=(0,0,0), normal=(0,0,1))
        #
        # Parameters:
        #   size: (width, height) tuple - dimensions of the plane
        #   color: (r, g, b) or (r, g, b, a) - color values 0-1
        #   center: (x, y, z) tuple - center position (default origin)
        #   normal: (nx, ny, nz) - plane normal direction (default up)
        # ---------------------------------------------------------------------

        # Horizontal plane (default normal)
        pgfx.draw_plane(
            size=(2.5, 2.5),
            color=Palette.Standard10[0],  # Blue
            center=(x_offset - 3, y_offset, 0.01)
        )

        # Vertical plane (wall)
        pgfx.draw_plane(
            size=(2.0, 1.5),
            color=Palette.Standard10[2],  # Green
            center=(x_offset - 3, y_offset - 1.5, 1.0),
            normal=(0, 1, 0)  # Facing Y direction
        )

        # Angled plane
        angle = np.radians(30)
        pgfx.draw_plane(
            size=(2.0, 2.0),
            color=Palette.Standard10[1],  # Orange
            center=(x_offset - 3, y_offset, 1.8),
            normal=(0, -np.sin(angle), np.cos(angle))  # Tilted
        )

        # ---------------------------------------------------------------------
        # TRIANGLE
        # pgfx.draw_triangle(v1, v2, v3, color)
        #
        # Parameters:
        #   v1, v2, v3: (x, y, z) tuples - the three vertices
        #   color: (r, g, b) or (r, g, b, a) - uniform color for entire triangle
        #
        # Note: Vertices should be in counter-clockwise order when viewed
        #       from the front (determines which side is "front")
        # ---------------------------------------------------------------------

        # Basic triangle
        pgfx.draw_triangle(
            v1=(x_offset + 0, y_offset - 1, 0.01),
            v2=(x_offset + 2, y_offset - 1, 0.01),
            v3=(x_offset + 1, y_offset + 1, 0.01),
            color=Palette.Standard10[3]  # Red
        )

        # 3D triangle (not on ground)
        wave = np.sin(self.time_val * 2) * 0.5
        pgfx.draw_triangle(
            v1=(x_offset + 0, y_offset - 1, 1.5),
            v2=(x_offset + 2, y_offset - 1, 1.5 + wave),
            v3=(x_offset + 1, y_offset + 1, 2.5),
            color=Palette.Standard10[4]  # Purple
        )

        # Multiple triangles forming a shape
        center_x, center_y = x_offset + 1, y_offset
        for i in range(6):
            angle1 = np.radians(i * 60 + self.rotation * 0.5)
            angle2 = np.radians((i + 1) * 60 + self.rotation * 0.5)
            pgfx.draw_triangle(
                v1=(center_x, center_y, 3.5),
                v2=(center_x + np.cos(angle1) * 1.0, center_y + np.sin(angle1) * 1.0, 3.5),
                v3=(center_x + np.cos(angle2) * 1.0, center_y + np.sin(angle2) * 1.0, 3.5),
                color=Palette.Standard10[i % 10]
            )

        # ---------------------------------------------------------------------
        # FACE (Per-Vertex Color Triangle)
        # pgfx.draw_face(v1, v2, v3, c1, c2, c3)
        #
        # Parameters:
        #   v1, v2, v3: (x, y, z) tuples - the three vertices
        #   c1, c2, c3: (r, g, b) tuples - color at each vertex
        #
        # The colors are interpolated across the triangle surface,
        # creating smooth gradients. Useful for:
        # - Vertex-colored meshes
        # - Smooth color transitions
        # - Heat maps and visualizations
        # ---------------------------------------------------------------------

        # RGB gradient triangle
        pgfx.draw_face(
            v1=(x_offset + 4, y_offset - 1, 0.01),
            v2=(x_offset + 6, y_offset - 1, 0.01),
            v3=(x_offset + 5, y_offset + 1, 0.01),
            c1=Colors.RED,
            c2=Colors.GREEN,
            c3=Colors.BLUE
        )

        # Gradient face with animation
        t = self.time_val
        pgfx.draw_face(
            v1=(x_offset + 4, y_offset - 1, 1.5),
            v2=(x_offset + 6, y_offset - 1, 1.5),
            v3=(x_offset + 5, y_offset + 1, 2.5),
            c1=(np.sin(t) * 0.5 + 0.5, 0.2, 0.2),
            c2=(0.2, np.sin(t + 2) * 0.5 + 0.5, 0.2),
            c3=(0.2, 0.2, np.sin(t + 4) * 0.5 + 0.5)
        )

        # Heat map style gradient
        pgfx.draw_face(
            v1=(x_offset + 4, y_offset + 1.5, 0.01),
            v2=(x_offset + 6, y_offset + 1.5, 0.01),
            v3=(x_offset + 5, y_offset + 3, 0.01),
            c1=(0.0, 0.0, 1.0),  # Cold (blue)
            c2=(1.0, 1.0, 0.0),  # Warm (yellow)
            c3=(1.0, 0.0, 0.0)  # Hot (red)
        )

    # =========================================================================
    # SECTION 3: Lines, Paths, and Points
    # =========================================================================

    def _draw_section_3_lines(self):
        """
        Section 3: Lines, Paths, Arrows, and Points

        Demonstrates:
        - draw_line(): Single line segment
        - draw_path(): Connected line segments
        - draw_arrow(): Line with arrowhead
        - draw_point(): Single point/dot
        """
        x_offset = 6 if self.current_section == 0 else 0
        y_offset = 4 if self.current_section == 0 else 0

        # ---------------------------------------------------------------------
        # LINE
        # pgfx.draw_line(start, end, color, width=1.0)
        #
        # Parameters:
        #   start: (x, y, z) tuple - starting point
        #   end: (x, y, z) tuple - ending point
        #   color: (r, g, b) or (r, g, b, a) - line color
        #   width: float - line thickness in pixels (default 1.0)
        #
        # Note: Line width support depends on GPU/driver. Some systems
        #       may clamp width to 1.0.
        # ---------------------------------------------------------------------

        # Basic lines with different widths
        for i, width in enumerate([1.0, 2.0, 3.0, 5.0]):
            pgfx.draw_line(
                start=(x_offset - 2, y_offset - 2 + i * 0.8, 0.1),
                end=(x_offset + 2, y_offset - 2 + i * 0.8, 0.1),
                color=Palette.Standard10[i],
                width=width
            )

        # Vertical line
        pgfx.draw_line(
            start=(x_offset - 3, y_offset, 0),
            end=(x_offset - 3, y_offset, 3),
            color=Colors.YELLOW,
            width=3.0
        )

        # Diagonal animated line
        angle = np.radians(self.rotation)
        pgfx.draw_line(
            start=(x_offset, y_offset, 2),
            end=(x_offset + np.cos(angle) * 2, y_offset + np.sin(angle) * 2, 2),
            color=Colors.CYAN,
            width=2.0
        )

        # ---------------------------------------------------------------------
        # PATH
        # pgfx.draw_path(points, color, width=1.0)
        #
        # Parameters:
        #   points: list of (x, y, z) tuples - vertices of the path
        #   color: (r, g, b) or (r, g, b, a) - path color
        #   width: float - line thickness in pixels (default 1.0)
        #
        # Draws connected line segments through all points.
        # More efficient than multiple draw_line() calls.
        # ---------------------------------------------------------------------

        # Sine wave path
        wave_points = []
        for i in range(50):
            t = i / 49.0 * 4 * np.pi
            x = x_offset - 2 + (i / 49.0) * 4
            y = y_offset + 2
            z = np.sin(t + self.time_val * 2) * 0.5 + 1.0
            wave_points.append((x, y, z))

        pgfx.draw_path(
            points=wave_points,
            color=Palette.Standard10[5],  # Brown
            width=2.0
        )

        # Spiral path
        spiral_points = []
        for i in range(100):
            t = i / 99.0 * 4 * np.pi
            r = 0.3 + i / 99.0 * 1.2
            x = x_offset + 3 + np.cos(t + self.time_val) * r
            y = y_offset + np.sin(t + self.time_val) * r
            z = i / 99.0 * 2.5 + 0.1
            spiral_points.append((x, y, z))

        pgfx.draw_path(
            points=spiral_points,
            color=Palette.Standard10[6],  # Pink
            width=1.5
        )

        # ---------------------------------------------------------------------
        # ARROW
        # pgfx.draw_arrow(start, end, color, head_size=0.1, head_radius=None, width_radius=0.03)
        #
        # Parameters:
        #   start: (x, y, z) tuple - arrow tail position
        #   end: (x, y, z) tuple - arrow head position
        #   color: (r, g, b) or (r, g, b, a) - arrow color
        #   head_size: float - length of arrowhead (default 0.1)
        #   head_radius: float - radius of arrowhead cone (default 2.5x width)
        #   width_radius: float - radius of arrow shaft (default 0.03)
        #
        # Composed of a cylinder (shaft) and cone (head).
        # Useful for vectors, directions, forces.
        # ---------------------------------------------------------------------

        # Basic arrow
        pgfx.draw_arrow(
            start=(x_offset - 2, y_offset - 3, 0.1),
            end=(x_offset - 2, y_offset - 3, 2.0),
            color=Colors.RED,
            head_size=0.3,
            width_radius=0.05
        )

        # Coordinate axes arrows
        origin = (x_offset, y_offset - 3, 0.1)
        pgfx.draw_arrow(start=origin, end=(x_offset + 1.5, y_offset - 3, 0.1),
                        color=Colors.RED, head_size=0.2, width_radius=0.04)
        pgfx.draw_arrow(start=origin, end=(x_offset, y_offset - 1.5, 0.1),
                        color=Colors.GREEN, head_size=0.2, width_radius=0.04)
        pgfx.draw_arrow(start=origin, end=(x_offset, y_offset - 3, 1.6),
                        color=Colors.BLUE, head_size=0.2, width_radius=0.04)

        # Animated rotating arrow
        angle = np.radians(self.rotation * 2)
        pgfx.draw_arrow(
            start=(x_offset + 3, y_offset - 3, 1.0),
            end=(x_offset + 3 + np.cos(angle) * 1.5, y_offset - 3 + np.sin(angle) * 1.5, 1.0),
            color=Colors.MAGENTA,
            head_size=0.25,
            width_radius=0.06
        )

        # ---------------------------------------------------------------------
        # POINT
        # pgfx.draw_point(position, color, size=5.0)
        #
        # Parameters:
        #   position: (x, y, z) tuple - point location
        #   color: (r, g, b) or (r, g, b, a) - point color
        #   size: float - point diameter in pixels (default 5.0)
        #
        # Note: For many points, use draw_particles() instead (much faster).
        # ---------------------------------------------------------------------

        # Row of points with varying sizes
        for i in range(8):
            pgfx.draw_point(
                position=(x_offset - 2 + i * 0.6, y_offset + 3, 0.5),
                color=Palette.Standard10[i % 10],
                size=4.0 + i * 2
            )

        # Animated point
        pulse = np.sin(self.time_val * 4) * 0.5 + 0.5
        pgfx.draw_point(
            position=(x_offset, y_offset + 3.5, 1.0 + pulse),
            color=(1.0, pulse, 0.0),
            size=15.0
        )

    # =========================================================================
    # SECTION 4: Batch Rendering
    # =========================================================================

    def _draw_section_4_batch(self):
        """
        Section 4: Batch Rendering for Performance

        Demonstrates:
        - draw_particles(): Optimized point cloud rendering
        - Many spheres/cylinders (automatically batched in v2.0)
        """
        x_offset = 6 if self.current_section == 0 else 0
        y_offset = -4 if self.current_section == 0 else 0

        # ---------------------------------------------------------------------
        # PARTICLES
        # pgfx.draw_particles(positions, colors, sizes=1.0)
        #
        # Parameters:
        #   positions: numpy array (N, 3) - particle positions
        #   colors: numpy array (N, 3) - RGB colors per particle
        #   sizes: float or numpy array (N,) - size per particle
        #
        # Highly optimized for rendering thousands to millions of points.
        # All particles rendered in a single draw call.
        #
        # Performance tips:
        # - Use numpy arrays (not lists) for positions/colors
        # - Use float32 dtype for best performance
        # - Pre-allocate arrays if updating every frame
        # ---------------------------------------------------------------------

        # Rotate particle cloud
        angle = np.radians(self.rotation * 0.5)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        rotated = self.particle_positions.copy()
        rotated[:, 0] = self.particle_positions[:, 0] * cos_a - self.particle_positions[:, 1] * sin_a
        rotated[:, 1] = self.particle_positions[:, 0] * sin_a + self.particle_positions[:, 1] * cos_a

        # Offset to section position
        rotated[:, 0] += x_offset
        rotated[:, 1] += y_offset

        pgfx.draw_particles(
            positions=rotated,
            colors=self.particle_colors,
            sizes=self.particle_sizes
        )

        # ---------------------------------------------------------------------
        # AUTOMATIC BATCHING (v2.0)
        #
        # In PiViz v2.0, all draw_sphere(), draw_cylinder(), draw_triangle()
        # calls are automatically batched. You don't need to do anything
        # special - just call the functions normally and they will be
        # rendered efficiently.
        #
        # Example: Drawing 100 spheres in a loop results in only 1 draw call.
        # ---------------------------------------------------------------------

        # Grid of spheres (automatically batched)
        grid_size = 5
        for i in range(grid_size):
            for j in range(grid_size):
                # Animated wave height
                wave = np.sin(self.time_val * 3 + i * 0.5 + j * 0.5) * 0.3

                pgfx.draw_sphere(
                    center=(x_offset - 5 + i * 1.2, y_offset + j * 1.2, 4.0 + wave),
                    radius=0.3,
                    color=((i / grid_size), 0.5, (j / grid_size)),
                    detail=8  # Lower detail for many spheres
                )

        # Connected spheres with cylinders (automatically batched)
        prev_pos = None
        for i in range(10):
            angle = np.radians(i * 36 + self.rotation)
            x = x_offset + 3 + np.cos(angle) * 2
            y = y_offset + 2 + np.sin(angle) * 2
            z = 4.0 + np.sin(self.time_val * 2 + i) * 0.3

            pos = (x, y, z)

            pgfx.draw_sphere(
                center=pos,
                radius=0.2,
                color=Palette.Standard10[i % 10],
                detail=8
            )

            if prev_pos is not None:
                pgfx.draw_cylinder(
                    start=prev_pos,
                    end=pos,
                    radius=0.05,
                    color=(0.5, 0.5, 0.5),
                    detail=8
                )

            prev_pos = pos


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("""
PiViz Shapes Showcase
=====================

This example demonstrates all available primitive shapes in PiViz.

Controls:
  0     - Show all sections
  1     - Basic Solids (sphere, cube, cylinder, cone)
  2     - Flat Primitives (plane, triangle, face)
  3     - Lines and Paths (line, path, arrow, point)
  4     - Batch Rendering (particles, many shapes)
  M     - Toggle matte/shiny materials
  R     - Reset rotation

Mouse:
  Left drag   - Rotate camera
  Right drag  - Pan camera
  Scroll      - Zoom
""")

    studio = PiVizStudio(scene_fx=ShapesShowcase())
    studio.run()

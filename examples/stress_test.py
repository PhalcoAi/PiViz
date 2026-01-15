# examples/stress_test.py
"""
PiViz Performance Stress Test
=============================

A comprehensive benchmark to test rendering performance with:
- Spheres (nodes)
- Cylinders (connections/springs)
- Lines (fast alternative)
- Particles (point clouds)

Use this to verify performance improvements after upgrading to v2.0.

Controls:
- 1-5: Switch test scenarios
- +/-: Increase/decrease object count
- L: Toggle between cylinders and lines for connections
- P: Toggle pause
- R: Reset to default

Author: Yogesh Phalak
"""

import numpy as np
import time
from piviz import PiVizStudio, PiVizFX, pgfx
from piviz.ui import Label, Slider, Button, Checkbox, ToggleSwitch


class StressTest(PiVizFX):
    """
    Comprehensive rendering stress test.
    """

    def setup(self):
        print("\n" + "=" * 60)
        print(" PiViz STRESS TEST")
        print("=" * 60)

        # Test parameters
        self.test_mode = 1  # 1-5 different scenarios
        self.paused = False
        self.use_lines = False  # Use lines instead of cylinders
        self.show_connections = True
        self.time_offset = 0.0

        # Object counts (adjustable)
        self.sphere_count = 500
        self.cylinder_count = 1000
        self.particle_count = 50000

        # Performance tracking
        self._frame_times = []
        self._last_report = time.time()

        # Generate initial data
        self._generate_test_data()

        # Setup UI
        self._setup_ui()

        # Camera setup
        if self.camera:
            self.camera.set_view('iso')
            self.camera.distance = 30.0

    def _initialize_empty_triangles(self):
        """Initialize empty triangle list."""
        if not hasattr(self, 'triangles'):
            self.triangles = []

    def _generate_test_data(self):
        """Generate test geometry based on current mode."""
        np.random.seed(42)  # Reproducible results

        if self.test_mode == 1:
            # Mode 1: Grid of spheres (stress test instancing)
            self._generate_sphere_grid()
        elif self.test_mode == 2:
            # Mode 2: Spring-mass network (spheres + cylinders)
            self._generate_spring_network()
        elif self.test_mode == 3:
            # Mode 3: Random connections (maximum cylinder stress)
            self._generate_random_connections()
        elif self.test_mode == 4:
            # Mode 4: Particle galaxy (point cloud)
            self._generate_particle_cloud()
        elif self.test_mode == 5:
            # Mode 5: Mixed everything
            self._generate_mixed_scene()
        elif self.test_mode == 6:
            # Mode 6: Triangle mesh (batched triangles)
            self._generate_triangle_mesh()

    def _generate_sphere_grid(self):
        """Generate a 3D grid of spheres."""
        n = int(np.cbrt(self.sphere_count))
        self.sphere_count = n ** 3  # Adjust to perfect cube

        spacing = 2.0
        offset = (n - 1) * spacing / 2

        self.sphere_positions = []
        self.sphere_colors = []
        self.sphere_radii = []

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    pos = np.array([
                        x * spacing - offset,
                        y * spacing - offset,
                        z * spacing - offset
                    ])
                    self.sphere_positions.append(pos)

                    # Color based on position
                    color = (
                        (x / n),
                        (y / n),
                        (z / n)
                    )
                    self.sphere_colors.append(color)
                    self.sphere_radii.append(0.3 + np.random.random() * 0.3)

        self.sphere_positions = np.array(self.sphere_positions, dtype='f4')
        self.sphere_colors = np.array(self.sphere_colors, dtype='f4')
        self.sphere_radii = np.array(self.sphere_radii, dtype='f4')

        # No connections in this mode
        self.connections = []
        self.particles = None
        self.triangles = []  # Initialize empty

        print(f"[Mode 1] Grid: {self.sphere_count} spheres")

    def _generate_spring_network(self):
        """Generate a spring-mass network."""
        n = int(self.sphere_count)

        # Random positions in a cube
        self.sphere_positions = (np.random.random((n, 3)) - 0.5) * 20
        self.sphere_positions = self.sphere_positions.astype('f4')

        # Colors based on height
        heights = self.sphere_positions[:, 2]
        h_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
        self.sphere_colors = np.zeros((n, 3), dtype='f4')
        self.sphere_colors[:, 0] = h_norm  # Red channel
        self.sphere_colors[:, 2] = 1 - h_norm  # Blue channel
        self.sphere_colors[:, 1] = 0.3  # Some green

        self.sphere_radii = np.ones(n, dtype='f4') * 0.3

        # Generate connections (k-nearest neighbors)
        self.connections = []
        k = min(6, n - 1)  # Connect to 6 nearest neighbors

        for i in range(n):
            distances = np.linalg.norm(self.sphere_positions - self.sphere_positions[i], axis=1)
            nearest = np.argsort(distances)[1:k + 1]  # Skip self
            for j in nearest:
                if i < j:  # Avoid duplicates
                    self.connections.append((i, j))

        # Limit connections
        if len(self.connections) > int(self.cylinder_count):
            indices = np.random.choice(len(self.connections), int(self.cylinder_count), replace=False)
            self.connections = [self.connections[i] for i in indices]

        self.particles = None
        self.triangles = []  # Initialize empty
        print(f"[Mode 2] Network: {n} spheres, {len(self.connections)} connections")

    def _generate_random_connections(self):
        """Generate random spheres with many random connections."""
        n = int(min(self.sphere_count, 200))  # Limit spheres for this test

        self.sphere_positions = (np.random.random((n, 3)) - 0.5) * 15
        self.sphere_positions = self.sphere_positions.astype('f4')

        self.sphere_colors = np.random.random((n, 3)).astype('f4')
        self.sphere_radii = np.ones(n, dtype='f4') * 0.4

        # Generate MANY random connections
        self.connections = []
        num_connections = int(self.cylinder_count)

        for _ in range(num_connections):
            i, j = np.random.choice(n, 2, replace=False)
            self.connections.append((int(i), int(j)))

        self.particles = None
        self.triangles = []  # Initialize empty
        print(f"[Mode 3] Random: {n} spheres, {len(self.connections)} connections")

    def _generate_particle_cloud(self):
        """Generate a particle cloud (galaxy-like)."""
        n = int(self.particle_count)

        # Spiral galaxy distribution
        t = np.random.power(0.5, n) * 4.0
        theta = t * 2 + np.random.normal(0, 0.3, n)

        # Add spiral arms
        arm = np.random.randint(0, 2, n)
        theta += arm * np.pi

        r = t + np.random.exponential(0.3, n)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.normal(0, 0.1 / (r * 0.3 + 1), n)

        self.particles = np.column_stack([x, y, z]).astype('f4')

        # Colors
        self.particle_colors = np.zeros((n, 3), dtype='f4')

        # Core: yellow/white
        core_mask = r < 1.0
        self.particle_colors[core_mask] = [1.0, 0.9, 0.7]

        # Arms: blue
        arm_mask = ~core_mask
        self.particle_colors[arm_mask, 0] = 0.3
        self.particle_colors[arm_mask, 1] = 0.5
        self.particle_colors[arm_mask, 2] = 1.0

        self.particle_sizes = np.random.uniform(1, 4, n).astype('f4')

        # No spheres/connections in this mode
        self.sphere_positions = np.zeros((0, 3), dtype='f4')
        self.sphere_colors = np.zeros((0, 3), dtype='f4')
        self.sphere_radii = np.zeros(0, dtype='f4')
        self.connections = []
        self.triangles = []  # Initialize empty

        print(f"[Mode 4] Particles: {n} points")

    def _generate_mixed_scene(self):
        """Generate a scene with everything."""
        # Spheres
        n_spheres = int(min(self.sphere_count, 300))
        self.sphere_positions = (np.random.random((n_spheres, 3)) - 0.5) * 20
        self.sphere_positions = self.sphere_positions.astype('f4')
        self.sphere_colors = np.random.random((n_spheres, 3)).astype('f4')
        self.sphere_radii = np.random.uniform(0.2, 0.5, n_spheres).astype('f4')

        # Connections
        self.connections = []
        for _ in range(int(min(self.cylinder_count, 500))):
            i, j = np.random.choice(n_spheres, 2, replace=False)
            self.connections.append((int(i), int(j)))

        # Particles
        n_particles = int(self.particle_count // 2)
        self.particles = (np.random.random((n_particles, 3)) - 0.5) * 25
        self.particles = self.particles.astype('f4')
        self.particle_colors = np.random.random((n_particles, 3)).astype('f4')
        self.particle_sizes = np.random.uniform(1, 3, n_particles).astype('f4')
        self.triangles = []  # Initialize empty

        print(f"[Mode 5] Mixed: {n_spheres} spheres, {len(self.connections)} connections, {n_particles} particles")

    def _generate_triangle_mesh(self):
        """Generate a mesh of triangles (terrain-like surface)."""
        # Create a grid of triangles forming a wavy surface
        grid_size = int(np.sqrt(self.sphere_count / 2))  # Reuse sphere_count as triangle density
        grid_size = max(10, min(grid_size, 100))

        spacing = 0.5
        offset = (grid_size - 1) * spacing / 2

        # Generate height map (wavy terrain)
        self.triangle_vertices = []
        self.triangle_colors = []

        # Create vertex grid
        vertices = np.zeros((grid_size, grid_size, 3), dtype='f4')
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * spacing - offset
                y = j * spacing - offset
                # Wavy height
                z = np.sin(x * 0.5) * np.cos(y * 0.5) * 2.0
                z += np.sin(x * 1.5 + y) * 0.5
                vertices[i, j] = [x, y, z]

        # Create triangles from grid
        self.triangles = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                v00 = vertices[i, j]
                v10 = vertices[i + 1, j]
                v01 = vertices[i, j + 1]
                v11 = vertices[i + 1, j + 1]

                # Height-based color (terrain coloring)
                avg_height = (v00[2] + v10[2] + v01[2]) / 3
                if avg_height < -1:
                    color = (0.2, 0.3, 0.8, 1.0)  # Deep blue (water)
                elif avg_height < 0:
                    color = (0.3, 0.5, 0.9, 1.0)  # Light blue (shallow)
                elif avg_height < 0.5:
                    color = (0.3, 0.7, 0.3, 1.0)  # Green (grass)
                elif avg_height < 1.5:
                    color = (0.6, 0.5, 0.3, 1.0)  # Brown (dirt)
                else:
                    color = (0.9, 0.9, 0.95, 1.0)  # White (snow)

                # Two triangles per grid cell
                self.triangles.append((tuple(v00), tuple(v10), tuple(v11), color))
                self.triangles.append((tuple(v00), tuple(v11), tuple(v01), color))

        # No spheres/connections/particles in this mode
        self.sphere_positions = np.zeros((0, 3), dtype='f4')
        self.sphere_colors = np.zeros((0, 3), dtype='f4')
        self.sphere_radii = np.zeros(0, dtype='f4')
        self.connections = []
        self.particles = None

        print(f"[Mode 6] Triangle Mesh: {len(self.triangles)} triangles ({grid_size}x{grid_size} grid)")

    def _setup_ui(self):
        """Setup UI controls."""
        if not self.ui_manager:
            return

        self.ui_manager.set_panel_title("Stress Test Controls")

        y = 0
        self.lbl_fps = Label((0, 0, 200, 20), "FPS: --", color=(0.3, 1.0, 0.4))
        self.ui_manager.add_widget("fps", self.lbl_fps)

        self.lbl_mode = Label((0, 0, 200, 20), f"Mode: {self.test_mode}", color=(1, 1, 1))
        self.ui_manager.add_widget("mode", self.lbl_mode)

        self.lbl_objects = Label((0, 0, 200, 20), "Objects: --", color=(0.7, 0.7, 0.7))
        self.ui_manager.add_widget("objects", self.lbl_objects)

        self.lbl_drawcalls = Label((0, 0, 200, 20), "Draw calls: --", color=(0.7, 0.7, 0.7))
        self.ui_manager.add_widget("drawcalls", self.lbl_drawcalls)

        # Mode buttons
        for i in range(1, 7):
            self.ui_manager.add_widget(
                f"btn_mode{i}",
                Button((0, 0, 40, 25), f"M{i}", lambda m=i: self._set_mode(m))
            )

        # Toggles
        self.ui_manager.add_widget(
            "chk_lines",
            Checkbox((0, 0, 150, 20), "Use Lines (fast)", self.use_lines,
                     lambda v: setattr(self, 'use_lines', v))
        )

        self.ui_manager.add_widget(
            "chk_conn",
            Checkbox((0, 0, 150, 20), "Show Connections", self.show_connections,
                     lambda v: setattr(self, 'show_connections', v))
        )

        # Count sliders
        def set_spheres(v):
            self.sphere_count = int(v)
            self._generate_test_data()

        def set_cylinders(v):
            self.cylinder_count = int(v)
            self._generate_test_data()

        self.ui_manager.add_widget(
            "sld_spheres",
            Slider((0, 0, 150, 20), "Spheres", 10, 2000, self.sphere_count, set_spheres)
        )

        self.ui_manager.add_widget(
            "sld_cylinders",
            Slider((0, 0, 150, 20), "Connections", 10, 5000, self.cylinder_count, set_cylinders)
        )

    def _set_mode(self, mode):
        """Switch test mode."""
        self.test_mode = mode
        self._generate_test_data()
        self.lbl_mode.text = f"Mode: {mode}"

    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action != 1:  # Press only
            return

        # Number keys for modes
        if 49 <= key <= 54:  # 1-6
            self._set_mode(key - 48)
        elif key == 76:  # L
            self.use_lines = not self.use_lines
        elif key == 80:  # P
            self.paused = not self.paused
        elif key == 82:  # R
            self.sphere_count = 500
            self.cylinder_count = 1000
            self._generate_test_data()

    def render(self, time_val, dt):
        """Main render loop."""
        if self.paused:
            time_val = self.time_offset
        else:
            self.time_offset = time_val

        # Track frame time
        frame_start = time.perf_counter()

        # Animation factor
        anim = np.sin(time_val * 0.5) * 0.5 + 0.5

        # Render spheres
        n_spheres = len(self.sphere_positions)
        for i in range(n_spheres):
            pos = self.sphere_positions[i].copy()

            # Subtle animation
            if self.test_mode != 4:  # Not particle mode
                pos[2] += np.sin(time_val * 2 + i * 0.1) * 0.2

            pgfx.draw_sphere(
                center=tuple(pos),
                radius=float(self.sphere_radii[i]) if i < len(self.sphere_radii) else 0.3,
                color=tuple(self.sphere_colors[i]),
                detail=8 if n_spheres > 500 else 12
            )

        # Render connections
        if self.show_connections and self.connections:
            for i, j in self.connections:
                if i >= n_spheres or j >= n_spheres:
                    continue

                start = self.sphere_positions[i]
                end = self.sphere_positions[j]

                # Animate slightly
                start = start.copy()
                end = end.copy()
                start[2] += np.sin(time_val * 2 + i * 0.1) * 0.2
                end[2] += np.sin(time_val * 2 + j * 0.1) * 0.2

                # Color based on strain (mock)
                dist = np.linalg.norm(end - start)
                strain = (dist - 2.0) / 2.0
                if strain < 0:
                    color = (0.2, 0.5, 1.0)  # Blue: compressed
                else:
                    color = (1.0, 0.3, 0.2)  # Red: stretched

                if self.use_lines:
                    pgfx.draw_line(tuple(start), tuple(end), color=color, width=2.0)
                else:
                    pgfx.draw_cylinder(
                        start=tuple(start),
                        end=tuple(end),
                        radius=0.05,
                        color=color,
                        detail=8
                    )

        # Render particles
        if self.particles is not None and len(self.particles) > 0:
            # Rotate particles
            angle = time_val * 0.3
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            rotated = self.particles.copy()
            rotated[:, 0] = self.particles[:, 0] * cos_a - self.particles[:, 1] * sin_a
            rotated[:, 1] = self.particles[:, 0] * sin_a + self.particles[:, 1] * cos_a

            pgfx.draw_particles(rotated, self.particle_colors, self.particle_sizes)

        # Render triangles (Mode 6)
        if hasattr(self, 'triangles') and self.triangles:
            for v1, v2, v3, color in self.triangles:
                # Animate height slightly
                v1 = (v1[0], v1[1], v1[2] + np.sin(time_val + v1[0] * 0.5) * 0.1)
                v2 = (v2[0], v2[1], v2[2] + np.sin(time_val + v2[0] * 0.5) * 0.1)
                v3 = (v3[0], v3[1], v3[2] + np.sin(time_val + v3[0] * 0.5) * 0.1)
                pgfx.draw_triangle(v1, v2, v3, color)

        # Track performance
        frame_time = time.perf_counter() - frame_start
        self._frame_times.append(frame_time)

        # Update UI periodically
        if time.time() - self._last_report > 0.5:
            self._update_performance_ui()
            self._last_report = time.time()

    def _update_performance_ui(self):
        """Update performance labels."""
        if not self._frame_times:
            return

        avg_time = sum(self._frame_times) / len(self._frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        # Color based on FPS
        if fps >= 55:
            color = (0.3, 1.0, 0.4)  # Green
        elif fps >= 30:
            color = (1.0, 0.8, 0.2)  # Yellow
        else:
            color = (1.0, 0.3, 0.3)  # Red

        self.lbl_fps.text = f"FPS: {fps:.1f} ({avg_time * 1000:.1f}ms)"
        self.lbl_fps.color = color

        # Object counts
        n_spheres = len(self.sphere_positions)
        n_conn = len(self.connections) if self.show_connections else 0
        n_particles = len(self.particles) if self.particles is not None else 0

        self.lbl_objects.text = f"Spheres: {n_spheres}, Conn: {n_conn}, Particles: {n_particles}"

        # Estimated draw calls
        if hasattr(pgfx, 'flush_all'):
            # v2.0: batched
            draw_calls = 3 + (1 if n_particles > 0 else 0)  # spheres + cylinders + lines + particles
            self.lbl_drawcalls.text = f"Draw calls: ~{draw_calls} (batched)"
        else:
            # v1.x: individual
            draw_calls = n_spheres + n_conn + (1 if n_particles > 0 else 0)
            self.lbl_drawcalls.text = f"Draw calls: ~{draw_calls} (individual)"

        self._frame_times.clear()


# ============================================================
# QUICK BENCHMARK (No UI)
# ============================================================

def run_benchmark():
    """
    Run a quick benchmark without UI.
    Useful for automated testing.
    """
    print("\n" + "=" * 60)
    print(" PiViz QUICK BENCHMARK")
    print("=" * 60 + "\n")

    import moderngl

    # Check OpenGL
    try:
        ctx = moderngl.create_standalone_context()
        print(f"OpenGL Renderer: {ctx.info['GL_RENDERER']}")
        print(f"OpenGL Version: {ctx.info['GL_VERSION']}")
        ctx.release()
    except Exception as e:
        print(f"OpenGL Error: {e}")
        return

    print("\nRunning stress test... (Press Ctrl+C to stop)\n")
    print("Controls:")
    print("  1-5: Switch test modes")
    print("  L: Toggle lines/cylinders")
    print("  +/-: Adjust object count")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    run_benchmark()
    PiVizStudio(scene_fx=StressTest()).run()

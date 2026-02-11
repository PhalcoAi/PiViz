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
- 1-6: Switch test scenarios
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
        self.test_mode = 1  # 1-6 different scenarios
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

    def _generate_test_data(self):
        """Generate test geometry based on current mode."""
        np.random.seed(42)  # Reproducible results

        if self.test_mode == 1:
            self._generate_sphere_grid()
        elif self.test_mode == 2:
            self._generate_spring_network()
        elif self.test_mode == 3:
            self._generate_random_connections()
        elif self.test_mode == 4:
            self._generate_particle_cloud()
        elif self.test_mode == 5:
            self._generate_mixed_scene()
        elif self.test_mode == 6:
            self._generate_triangle_mesh()

    def _generate_sphere_grid(self):
        """Generate a 3D grid of spheres."""
        n = int(np.cbrt(self.sphere_count))
        self.sphere_count = n ** 3

        spacing = 2.0
        offset = (n - 1) * spacing / 2

        positions = []
        colors = []
        radii = []

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    positions.append([
                        x * spacing - offset,
                        y * spacing - offset,
                        z * spacing - offset
                    ])
                    colors.append((x / n, y / n, z / n))
                    radii.append(0.3 + np.random.random() * 0.3)

        self.sphere_positions = np.array(positions, dtype='f4')
        self.sphere_colors = np.array(colors, dtype='f4')
        self.sphere_radii = np.array(radii, dtype='f4')
        self.connections = np.zeros((0, 2), dtype='i4')
        self.particles = None
        self.triangles_v = None

        print(f"[Mode 1] Grid: {self.sphere_count} spheres")

    def _generate_spring_network(self):
        """Generate a spring-mass network."""
        n = int(self.sphere_count)

        self.sphere_positions = ((np.random.random((n, 3)) - 0.5) * 20).astype('f4')

        heights = self.sphere_positions[:, 2]
        h_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
        self.sphere_colors = np.zeros((n, 3), dtype='f4')
        self.sphere_colors[:, 0] = h_norm
        self.sphere_colors[:, 2] = 1 - h_norm
        self.sphere_colors[:, 1] = 0.3
        self.sphere_radii = np.ones(n, dtype='f4') * 0.3

        # k-nearest neighbor connections
        connections = []
        k = min(6, n - 1)
        for i in range(n):
            distances = np.linalg.norm(self.sphere_positions - self.sphere_positions[i], axis=1)
            nearest = np.argsort(distances)[1:k + 1]
            for j in nearest:
                if i < j:
                    connections.append((i, j))

        if len(connections) > int(self.cylinder_count):
            indices = np.random.choice(len(connections), int(self.cylinder_count), replace=False)
            connections = [connections[i] for i in indices]

        self.connections = np.array(connections, dtype='i4') if connections else np.zeros((0, 2), dtype='i4')
        self.particles = None
        self.triangles_v = None
        print(f"[Mode 2] Network: {n} spheres, {len(self.connections)} connections")

    def _generate_random_connections(self):
        """Generate random spheres with many random connections."""
        n = int(min(self.sphere_count, 200))

        self.sphere_positions = ((np.random.random((n, 3)) - 0.5) * 15).astype('f4')
        self.sphere_colors = np.random.random((n, 3)).astype('f4')
        self.sphere_radii = np.ones(n, dtype='f4') * 0.4

        num_connections = int(self.cylinder_count)
        conn_i = np.random.randint(0, n, num_connections)
        conn_j = np.random.randint(0, n - 1, num_connections)
        conn_j[conn_j >= conn_i] += 1  # avoid self-connections
        self.connections = np.column_stack([conn_i, conn_j]).astype('i4')

        self.particles = None
        self.triangles_v = None
        print(f"[Mode 3] Random: {n} spheres, {len(self.connections)} connections")

    def _generate_particle_cloud(self):
        """Generate a particle cloud (galaxy-like)."""
        n = int(self.particle_count)

        t = np.random.power(0.5, n) * 4.0
        theta = t * 2 + np.random.normal(0, 0.3, n)
        arm = np.random.randint(0, 2, n)
        theta += arm * np.pi
        r = t + np.random.exponential(0.3, n)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.normal(0, 0.1 / (r * 0.3 + 1), n)

        self.particles = np.column_stack([x, y, z]).astype('f4')
        self.particle_colors = np.zeros((n, 3), dtype='f4')

        core_mask = r < 1.0
        self.particle_colors[core_mask] = [1.0, 0.9, 0.7]
        arm_mask = ~core_mask
        self.particle_colors[arm_mask, 0] = 0.3
        self.particle_colors[arm_mask, 1] = 0.5
        self.particle_colors[arm_mask, 2] = 1.0
        self.particle_sizes = np.random.uniform(1, 4, n).astype('f4')

        self.sphere_positions = np.zeros((0, 3), dtype='f4')
        self.sphere_colors = np.zeros((0, 3), dtype='f4')
        self.sphere_radii = np.zeros(0, dtype='f4')
        self.connections = np.zeros((0, 2), dtype='i4')
        self.triangles_v = None
        print(f"[Mode 4] Particles: {n} points")

    def _generate_mixed_scene(self):
        """Generate a scene with everything."""
        n_spheres = int(min(self.sphere_count, 300))
        self.sphere_positions = ((np.random.random((n_spheres, 3)) - 0.5) * 20).astype('f4')
        self.sphere_colors = np.random.random((n_spheres, 3)).astype('f4')
        self.sphere_radii = np.random.uniform(0.2, 0.5, n_spheres).astype('f4')

        num_conn = int(min(self.cylinder_count, 500))
        conn_i = np.random.randint(0, n_spheres, num_conn)
        conn_j = np.random.randint(0, n_spheres - 1, num_conn)
        conn_j[conn_j >= conn_i] += 1
        self.connections = np.column_stack([conn_i, conn_j]).astype('i4')

        n_particles = int(self.particle_count // 2)
        self.particles = ((np.random.random((n_particles, 3)) - 0.5) * 25).astype('f4')
        self.particle_colors = np.random.random((n_particles, 3)).astype('f4')
        self.particle_sizes = np.random.uniform(1, 3, n_particles).astype('f4')
        self.triangles_v = None

        print(f"[Mode 5] Mixed: {n_spheres} spheres, {len(self.connections)} connections, {n_particles} particles")

    def _generate_triangle_mesh(self):
        """Generate a mesh of triangles (terrain-like surface)."""
        grid_size = int(np.sqrt(self.sphere_count / 2))
        grid_size = max(10, min(grid_size, 100))

        spacing = 0.5
        offset = (grid_size - 1) * spacing / 2

        # Vectorized vertex grid
        ix = np.arange(grid_size, dtype='f4')
        iy = np.arange(grid_size, dtype='f4')
        gx, gy = np.meshgrid(ix * spacing - offset, iy * spacing - offset, indexing='ij')
        gz = np.sin(gx * 0.5) * np.cos(gy * 0.5) * 2.0 + np.sin(gx * 1.5 + gy) * 0.5

        # Build triangle arrays
        n_tris = (grid_size - 1) * (grid_size - 1) * 2
        self.triangles_v1 = np.empty((n_tris, 3), dtype='f4')
        self.triangles_v2 = np.empty((n_tris, 3), dtype='f4')
        self.triangles_v3 = np.empty((n_tris, 3), dtype='f4')
        self.triangles_colors = np.empty((n_tris, 4), dtype='f4')
        # Store base z for animation
        self.triangles_base_x = np.empty(n_tris, dtype='f4')

        idx = 0
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                v00 = np.array([gx[i, j], gy[i, j], gz[i, j]], dtype='f4')
                v10 = np.array([gx[i+1, j], gy[i+1, j], gz[i+1, j]], dtype='f4')
                v01 = np.array([gx[i, j+1], gy[i, j+1], gz[i, j+1]], dtype='f4')
                v11 = np.array([gx[i+1, j+1], gy[i+1, j+1], gz[i+1, j+1]], dtype='f4')

                avg_h = (v00[2] + v10[2] + v11[2]) / 3
                if avg_h < -1:
                    color = (0.2, 0.3, 0.8, 1.0)
                elif avg_h < 0:
                    color = (0.3, 0.5, 0.9, 1.0)
                elif avg_h < 0.5:
                    color = (0.3, 0.7, 0.3, 1.0)
                elif avg_h < 1.5:
                    color = (0.6, 0.5, 0.3, 1.0)
                else:
                    color = (0.9, 0.9, 0.95, 1.0)

                self.triangles_v1[idx] = v00
                self.triangles_v2[idx] = v10
                self.triangles_v3[idx] = v11
                self.triangles_colors[idx] = color
                self.triangles_base_x[idx] = v00[0]
                idx += 1

                self.triangles_v1[idx] = v00
                self.triangles_v2[idx] = v11
                self.triangles_v3[idx] = v01
                self.triangles_colors[idx] = color
                self.triangles_base_x[idx] = v00[0]
                idx += 1

        self.triangles_v = True  # flag that we have triangle data

        self.sphere_positions = np.zeros((0, 3), dtype='f4')
        self.sphere_colors = np.zeros((0, 3), dtype='f4')
        self.sphere_radii = np.zeros(0, dtype='f4')
        self.connections = np.zeros((0, 2), dtype='i4')
        self.particles = None

        print(f"[Mode 6] Triangle Mesh: {n_tris} triangles ({grid_size}x{grid_size} grid)")

    def _setup_ui(self):
        """Setup UI controls."""
        if not self.ui_manager:
            return

        self.ui_manager.set_panel_title("Stress Test Controls")

        self.lbl_fps = Label("FPS: --", color=(0.3, 1.0, 0.4, 1.0))
        self.ui_manager.add_widget("fps", self.lbl_fps)

        self.lbl_mode = Label(f"Mode: {self.test_mode}", color=(1, 1, 1, 1.0))
        self.ui_manager.add_widget("mode", self.lbl_mode)

        self.lbl_objects = Label("Objects: --", color=(0.7, 0.7, 0.7, 1.0))
        self.ui_manager.add_widget("objects", self.lbl_objects)

        self.lbl_drawcalls = Label("Draw calls: --", color=(0.7, 0.7, 0.7, 1.0))
        self.ui_manager.add_widget("drawcalls", self.lbl_drawcalls)

        for i in range(1, 7):
            self.ui_manager.add_widget(
                f"btn_mode{i}",
                Button(f"M{i}", lambda m=i: self._set_mode(m))
            )

        self.ui_manager.add_widget(
            "chk_lines",
            Checkbox("Use Lines (fast)", self.use_lines,
                     lambda v: setattr(self, 'use_lines', v))
        )
        self.ui_manager.add_widget(
            "chk_conn",
            Checkbox("Show Connections", self.show_connections,
                     lambda v: setattr(self, 'show_connections', v))
        )

        def set_spheres(v):
            self.sphere_count = int(v)
            self._generate_test_data()

        def set_cylinders(v):
            self.cylinder_count = int(v)
            self._generate_test_data()

        self.ui_manager.add_widget(
            "sld_spheres",
            Slider("Spheres", 10, 2000, self.sphere_count, set_spheres)
        )
        self.ui_manager.add_widget(
            "sld_cylinders",
            Slider("Connections", 10, 5000, self.cylinder_count, set_cylinders)
        )

    def _set_mode(self, mode):
        """Switch test mode."""
        self.test_mode = mode
        self._generate_test_data()
        self.lbl_mode.text = f"Mode: {mode}"

    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action != 1:
            return
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
        """Main render loop — fully vectorized, no per-object Python loops."""
        if self.paused:
            time_val = self.time_offset
        else:
            self.time_offset = time_val

        frame_start = time.perf_counter()
        n_spheres = len(self.sphere_positions)

        # =============================================================
        # RENDER SPHERES — vectorized animation + bulk batch API
        # =============================================================
        if n_spheres > 0 and self.test_mode != 4:
            # Vectorized z-animation for ALL spheres at once
            anim_positions = self.sphere_positions.copy()  # (n, 3)
            # Compute sin offsets for all spheres in one numpy call
            indices = np.arange(n_spheres, dtype='f4')
            anim_positions[:, 2] += np.sin(time_val * 2 + indices * 0.1) * 0.2

            detail = 8 if n_spheres > 500 else 12
            pgfx.draw_spheres_batch(
                centers=anim_positions,
                radii=self.sphere_radii,
                colors=self.sphere_colors,
                detail=detail
            )

        # =============================================================
        # RENDER CONNECTIONS — vectorized color + bulk batch API
        # =============================================================
        n_conn = len(self.connections)
        if self.show_connections and n_conn > 0 and n_spheres > 0:
            # Gather start/end positions using fancy indexing (vectorized)
            conn_i = self.connections[:, 0]
            conn_j = self.connections[:, 1]

            # Filter valid connections
            valid = (conn_i < n_spheres) & (conn_j < n_spheres)
            conn_i = conn_i[valid]
            conn_j = conn_j[valid]
            n_valid = len(conn_i)

            if n_valid > 0:
                starts = self.sphere_positions[conn_i].copy()  # (n_valid, 3)
                ends = self.sphere_positions[conn_j].copy()    # (n_valid, 3)

                # Vectorized animation
                starts[:, 2] += np.sin(time_val * 2 + conn_i.astype('f4') * 0.1) * 0.2
                ends[:, 2] += np.sin(time_val * 2 + conn_j.astype('f4') * 0.1) * 0.2

                # Vectorized color by strain
                diffs = ends - starts
                dists = np.linalg.norm(diffs, axis=1)
                strain = (dists - 2.0) / 2.0

                colors = np.empty((n_valid, 3), dtype='f4')
                compressed = strain < 0
                colors[compressed] = [0.2, 0.5, 1.0]   # Blue: compressed
                colors[~compressed] = [1.0, 0.3, 0.2]   # Red: stretched

                if self.use_lines:
                    pgfx.draw_lines_batch(starts, ends, colors, width=2.0)
                else:
                    pgfx.draw_cylinders_batch(
                        starts=starts,
                        ends=ends,
                        radii=0.05,
                        colors=colors,
                        detail=8
                    )

        # =============================================================
        # RENDER PARTICLES — already vectorized
        # =============================================================
        if self.particles is not None and len(self.particles) > 0:
            angle = time_val * 0.3
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotated = self.particles.copy()
            rotated[:, 0] = self.particles[:, 0] * cos_a - self.particles[:, 1] * sin_a
            rotated[:, 1] = self.particles[:, 0] * sin_a + self.particles[:, 1] * cos_a
            pgfx.draw_particles(rotated, self.particle_colors, self.particle_sizes)

        # =============================================================
        # RENDER TRIANGLES (Mode 6) — vectorized animation
        # =============================================================
        if self.triangles_v is not None:
            n_tri = len(self.triangles_v1)
            # Vectorized height animation
            z_offset = np.sin(time_val + self.triangles_base_x * 0.5) * 0.1

            # Temporarily modify z then write to triangle batch
            v1 = self.triangles_v1.copy()
            v2 = self.triangles_v2.copy()
            v3 = self.triangles_v3.copy()
            v1[:, 2] += z_offset
            v2[:, 2] += z_offset
            v3[:, 2] += z_offset

            # Use internal batch arrays directly for maximum speed
            import piviz.graphics.primitives as _pgfx
            needed = _pgfx._tri_count + n_tri
            _pgfx._tri_v1 = _pgfx._ensure_capacity(_pgfx._tri_v1, needed)
            _pgfx._tri_v2 = _pgfx._ensure_capacity(_pgfx._tri_v2, needed)
            _pgfx._tri_v3 = _pgfx._ensure_capacity(_pgfx._tri_v3, needed)
            _pgfx._tri_c1 = _pgfx._ensure_capacity(_pgfx._tri_c1, needed)
            _pgfx._tri_c2 = _pgfx._ensure_capacity(_pgfx._tri_c2, needed)
            _pgfx._tri_c3 = _pgfx._ensure_capacity(_pgfx._tri_c3, needed)

            s = _pgfx._tri_count
            _pgfx._tri_v1[s:s+n_tri] = v1
            _pgfx._tri_v2[s:s+n_tri] = v2
            _pgfx._tri_v3[s:s+n_tri] = v3
            _pgfx._tri_c1[s:s+n_tri] = self.triangles_colors
            _pgfx._tri_c2[s:s+n_tri] = self.triangles_colors
            _pgfx._tri_c3[s:s+n_tri] = self.triangles_colors
            _pgfx._tri_count = s + n_tri

            _pgfx._update_bounds(
                np.min(v1, axis=0),
                np.max(v1, axis=0)
            )

        # Track performance
        frame_time = time.perf_counter() - frame_start
        self._frame_times.append(frame_time)

        if time.time() - self._last_report > 0.5:
            self._update_performance_ui()
            self._last_report = time.time()

    def _update_performance_ui(self):
        """Update performance labels."""
        if not self._frame_times:
            return

        avg_time = sum(self._frame_times) / len(self._frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        if fps >= 55:
            color = (0.3, 1.0, 0.4, 1.0)
        elif fps >= 30:
            color = (1.0, 0.8, 0.2, 1.0)
        else:
            color = (1.0, 0.3, 0.3, 1.0)

        self.lbl_fps.text = f"FPS: {fps:.1f} ({avg_time * 1000:.1f}ms)"
        self.lbl_fps.color = color

        n_spheres = len(self.sphere_positions)
        n_conn = len(self.connections) if self.show_connections else 0
        n_particles = len(self.particles) if self.particles is not None else 0

        self.lbl_objects.text = f"Spheres: {n_spheres}, Conn: {n_conn}, Particles: {n_particles}"

        draw_calls = 3 + (1 if n_particles > 0 else 0)
        self.lbl_drawcalls.text = f"Draw calls: ~{draw_calls} (batched)"

        self._frame_times.clear()


# ============================================================
# QUICK BENCHMARK (No UI)
# ============================================================

def run_benchmark():
    """Run a quick benchmark without UI."""
    print("\n" + "=" * 60)
    print(" PiViz QUICK BENCHMARK")
    print("=" * 60 + "\n")

    import moderngl
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
    print("  1-6: Switch test modes")
    print("  L: Toggle lines/cylinders")
    print("  +/-: Adjust object count")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    run_benchmark()
    PiVizStudio(scene_fx=StressTest()).run()
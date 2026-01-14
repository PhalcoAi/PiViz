# examples/galaxy_stress_test.py

"""
Galaxy Simulation Stress Test
=============================
A stress test simulating and rendering a spiral galaxy with stars and dust particles.
Demonstrates handling of large particle counts with smooth animation.
"""

import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Label, Colors, Palette


class GalaxySimulation(PiVizFX):

    def setup(self):
        print("Generating galaxy geometry...")
        self.num_stars = 100000
        self.num_dust = 20000

        # 1. Generate Data
        s_r, s_theta, s_z, s_color, s_size = self._generate_stars_polar()
        d_r, d_theta, d_z, d_color, d_size = self._generate_dust_polar()

        # 2. Store CONSTANT initial state
        self.radii = np.concatenate([s_r, d_r]).astype('f4')
        self.initial_thetas = np.concatenate([s_theta, d_theta]).astype('f4')
        self.z_coords = np.concatenate([s_z, d_z]).astype('f4')

        self.colors = np.vstack([s_color, d_color]).astype('f4')
        self.sizes = np.concatenate([s_size, d_size]).astype('f4')

        # FIX: Ensure radii are strictly positive to avoid sqrt(-1) errors
        self.radii = np.maximum(self.radii, 0.01)

        # Calculate angular velocity
        self.speeds = (0.5 / (np.sqrt(self.radii) + 0.1)).astype('f4')

        # 3. Setup UI
        if self.ui_manager:
            self.ui_manager.set_panel_title("Galaxy Stats")
            self.ui_manager.add_widget("l_stars", Label((0, 0, 0, 0), f"Stars: {self.num_stars:,}", color=Colors.WHITE))
            self.ui_manager.add_widget("l_dust", Label((0, 0, 0, 0), f"Dust: {self.num_dust:,}", color=Colors.GREY))
            self.lbl_status = Label((0, 0, 0, 0), "Status: Init", color=Colors.GREEN)
            self.ui_manager.add_widget("l_status", self.lbl_status)

        if self.camera:
            self.camera.set_view('iso')
            self.camera.distance = 45.0

    def render(self, time, dt):
        # 4. Animate
        current_thetas = self.initial_thetas + (self.speeds * time * 0.5)

        # 5. Convert to Cartesian
        x = self.radii * np.cos(current_thetas)
        y = self.radii * np.sin(current_thetas)
        positions = np.column_stack([x, y, self.z_coords]).astype('f4')

        # 6. Update UI
        if self.lbl_status:
            self.lbl_status.text = f"Sim Time: {time:.2f}s"

        # 7. Draw
        pgfx.draw_particles(positions, self.colors, self.sizes)

    def _generate_stars_polar(self):
        n = self.num_stars
        num_arms = 2
        arm_length = 4.0

        t = np.random.power(0.5, n) * arm_length
        arm_index = np.random.randint(0, num_arms, n)
        arm_offset = arm_index * (2 * np.pi / num_arms)

        spread = np.random.normal(0, 0.5, n) * (t * 0.3 + 0.2)
        theta = t * 1.5 + arm_offset + spread
        r = t + np.random.exponential(0.3, n)

        z_scale = 0.1 / (r * 0.2 + 1)
        z = np.random.normal(0, z_scale, n)

        # Colors
        star_type = np.random.random(n)
        colors = np.zeros((n, 3))

        # Using scientific palette look
        blue_mask = star_type < 0.3
        colors[blue_mask] = Palette.Standard10[0][:3]  # Blue

        white_mask = (star_type >= 0.3) & (star_type < 0.5)
        colors[white_mask] = (0.9, 0.9, 1.0)  # White-ish

        yellow_mask = (star_type >= 0.5) & (star_type < 0.75)
        colors[yellow_mask] = Palette.Standard10[1][:3]  # Orange/Yellow

        red_mask = star_type >= 0.75
        colors[red_mask] = Palette.Standard10[3][:3]  # Red

        # Bulge
        bulge_mask = np.random.random(n) < 0.15
        r[bulge_mask] = np.random.exponential(0.3, np.sum(bulge_mask))
        z[bulge_mask] = np.random.normal(0, 0.2, np.sum(bulge_mask))
        colors[bulge_mask] = (1.0, 0.8, 0.6)

        # Brightness/Size
        colors *= np.random.uniform(0.5, 1.0, n)[:, np.newaxis]
        sizes = np.random.exponential(0.8, n)
        sizes = np.clip(sizes, 0.3, 4.0)
        sizes[bulge_mask] *= 1.5

        return r, theta, z, colors, sizes

    def _generate_dust_polar(self):
        n = self.num_dust
        t = np.random.uniform(0.5, 3.5, n)
        arm_index = np.random.randint(0, 2, n)
        arm_offset = arm_index * np.pi + 0.3

        theta = t * 1.5 + arm_offset + np.random.normal(0, 0.2, n)
        r = t + np.random.normal(0, 0.2, n)
        z = np.random.normal(0, 0.02, n)

        colors = np.zeros((n, 3))
        # Dark reddish-brown dust
        colors[:, 0] = np.random.uniform(0.4, 0.6, n)
        colors[:, 1] = np.random.uniform(0.2, 0.35, n)
        colors[:, 2] = np.random.uniform(0.15, 0.25, n)

        sizes = np.random.uniform(3.0, 8.0, n)

        return r, theta, z, colors, sizes


if __name__ == '__main__':
    PiVizStudio(scene_fx=GalaxySimulation()).run()

# examples/colors_showcase.py
"""
Color Palette & Colormap Showcase
=================================
Demonstrates the scientific colormaps and categorical palettes.
"""

from piviz import PiVizStudio, PiVizFX, pgfx, Colors, Palette, Colormap
import math


class ColorShowcase(PiVizFX):
    def setup(self):
        # Set camera for a good overview
        if self.camera:
            self.camera.distance = 18.0
            self.camera.elevation = 45.0
            self.camera.azimuth = 0.0

    def render(self, time, dt):
        # 1. Categorical Palette (Standard10) - Bottom Row
        # Shows distinct colors good for bar charts/categories
        for i, color in enumerate(Palette.Standard10):
            x = (i - 4.5) * 1.5
            pgfx.draw_cube(
                size=1.0,
                color=color,
                center=(x, -4, 0.5)
            )

        # 2. Colormap: Viridis - Middle Row
        # Shows continuous gradient mapped to height
        num_steps = 20
        for i in range(num_steps):
            t = i / (num_steps - 1)
            x = (i - num_steps / 2) * 0.6

            # Map height to value
            height = 1.0 + math.sin(t * math.pi) * 2.0

            # Get color from colormap
            color = Colormap.viridis(t)

            pgfx.draw_cylinder(
                start=(x, 0, 0),
                end=(x, 0, height),
                radius=0.25,
                color=color
            )

        # 3. Colormap: Plasma vs Coolwarm - Top Row
        for i in range(num_steps):
            t = i / (num_steps - 1)
            x = (i - num_steps / 2) * 0.6

            # Plasma Sphere
            c1 = Colormap.plasma(t)
            pgfx.draw_sphere(
                radius=0.25,
                color=c1,
                center=(x, 4, 0.25)
            )

            # Coolwarm Sphere (offset z)
            c2 = Colormap.coolwarm(t)
            pgfx.draw_sphere(
                radius=0.25,
                color=c2,
                center=(x, 4, 1.5)
            )

        # Draw a translucent plane to show alpha support
        pgfx.draw_plane(
            size=(20, 15),
            color=(0.2, 0.2, 0.25, 0.5),  # Transparent blue-grey
            center=(0, 0, -0.1)
        )


if __name__ == '__main__':
    studio = PiVizStudio(scene_fx=ColorShowcase())
    studio.run()

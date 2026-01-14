# examples/shapes_showcase.py
"""
Shapes Showcase
===============
Demonstrates all available primitive shapes in PiViz.
"""

from piviz import PiVizStudio, PiVizFX, pgfx, Colors, Palette


class ShapesShowcase(PiVizFX):
    """Showcase of all primitive shapes."""

    def setup(self):
        self.rotation = 0.0

    def render(self, time, dt):
        self.rotation += dt * 20

        # Ground plane
        pgfx.draw_plane(
            size=(15, 15),
            color=Colors.GREY
        )

        # Sphere (Palette Red)
        pgfx.draw_sphere(
            radius=0.8,
            color=Palette.Standard10[3],  # Red
            center=(-4, 0, 0.8),
            detail=24
        )

        # Cube (Palette Green)
        pgfx.draw_cube(
            size=1.4,
            color=Palette.Standard10[2],  # Green
            center=(0, 0, 0.7),
            rotation=(0, 0, self.rotation)
        )

        # Cylinder (Palette Blue)
        pgfx.draw_cylinder(
            start=(4, 0, 0),
            end=(4, 0, 2.5),
            radius=0.5,
            color=Palette.Standard10[0],  # Blue
            detail=24
        )

        # Triangle (Palette Olive)
        pgfx.draw_triangle(
            v1=(-2, 3, 0.01),
            v2=(0, 3, 0.01),
            v3=(-1, 5, 0.01),
            color=Palette.Standard10[8]  # Olive
        )

        # Gradient face using named Colors
        pgfx.draw_face(
            v1=(2, 3, 0.5),
            v2=(4, 3, 0.5),
            v3=(3, 5, 1.5),
            c1=Colors.RED,
            c2=Colors.GREEN,
            c3=Colors.BLUE
        )

        # Lines
        pgfx.draw_line(
            start=(-4, -3, 0),
            end=(-4, -3, 2),
            color=Colors.YELLOW,
            width=3.0
        )

        # Arrow
        pgfx.draw_arrow(
            start=(0, -3, 0),
            end=(0, -3, 2.5),
            color=Colors.CYAN,
            head_size=0.2,
            width_radius=0.03
        )

        # Points
        for i in range(5):
            pgfx.draw_point(
                position=(4, -3 + i * 0.5, 0.5),
                color=Palette.Standard10[1],  # Orange
                size=8.0
            )


if __name__ == '__main__':
    studio = PiVizStudio(scene_fx=ShapesShowcase())
    studio.run()

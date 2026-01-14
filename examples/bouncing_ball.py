# examples/bouncing_ball.py

from piviz import PiVizStudio, PiVizFX, pgfx, Colors, Palette


class BouncingBall(PiVizFX):
    """A simple bouncing ball physics demo."""

    def setup(self):
        self.position = 3.0  # Initial height
        self.velocity = 0.0  # Initial velocity
        self.gravity = -9.8  # Gravity acceleration
        self.restitution = 0.85  # Bounce coefficient
        self.ball_radius = 0.5

        # Use academic palette colors
        self.ball_color = Palette.Standard10[0]  # Academic Blue
        self.ground_color = Colors.GREY

    def render(self, time, dt):
        # Update physics
        self.velocity += self.gravity * dt
        self.position += self.velocity * dt

        # Bounce off ground
        if self.position < self.ball_radius:
            self.position = self.ball_radius
            self.velocity = -self.velocity * self.restitution

        # Draw ground plane
        pgfx.draw_plane(
            size=(8, 8),
            color=self.ground_color,
            center=(0, 0, 0)
        )

        # Draw ball
        pgfx.draw_sphere(
            radius=self.ball_radius,
            color=self.ball_color,
            center=(0, 0, self.position),
            detail=24
        )


if __name__ == '__main__':
    studio = PiVizStudio(scene_fx=BouncingBall())
    studio.run()

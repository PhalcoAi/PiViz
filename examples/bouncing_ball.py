"""
Bouncing Ball Simulation (VPython-style)
========================================

A simple physics simulation using the script-like wrapper.
"""

import piviz as pz

# Simulation state
ball_pos = [0, 0, 0]
ball_vel = [0, 0, 0]
gravity = -9.8
radius = 0.5
floor_z = -2.0
is_running = True


def toggle_simulation():
    global is_running
    is_running = not is_running


def reset_ball():
    global ball_pos, ball_vel
    ball_pos = [0, 0, 0]
    ball_vel = [0, 0, 0]


def set_gravity(value):
    global gravity
    gravity = -value


def setup():
    """Called once at startup."""
    print("Starting simulation...")
    pz.add_button("Start/Stop", toggle_simulation)
    pz.add_button("Reset", reset_ball)
    pz.add_slider("Gravity", 1.0, 20.0, 9.8, set_gravity)


def update(dt):
    """Called every frame."""
    global ball_pos, ball_vel

    if is_running:
        # Physics update
        ball_vel[2] += gravity * dt
        ball_pos[0] += ball_vel[0] * dt
        ball_pos[1] += ball_vel[1] * dt
        ball_pos[2] += ball_vel[2] * dt

        # Bounce
        if ball_pos[2] - radius < floor_z:
            ball_pos[2] = floor_z + radius
            ball_vel[2] = -ball_vel[2] * 0.8  # Damping

    # Draw scene
    # Floor
    pz.box(pos=(0, 0, floor_z - 0.1), size=(10, 10, 0.2), color=(0.3, 0.3, 0.3))

    # Ball
    pz.sphere(pos=tuple(ball_pos), radius=radius, color=(1, 0.2, 0.2))


if __name__ == '__main__':
    pz.run(setup, update)

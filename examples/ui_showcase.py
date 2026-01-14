# examples/ui_showcase.py
"""
UI Widgets Showcase
===================
Demonstrates standard widgets + Real-time mouse tracking.
"""

from piviz import (
    PiVizStudio, PiVizFX, pgfx,
    Label, Button, Slider, ToggleSwitch,
    Dropdown, ProgressBar
)


class UIShowcase(PiVizFX):
    """
    Showcase of all UI widgets with interactive elements.
    """

    def setup(self, ui_manager):
        # --- State Variables ---
        self.is_rotating = True
        self.rotation_speed = 45.0
        self.cube_color = (0.2, 0.7, 0.3)
        self.cube_scale = 1.0
        self.rotation = 0.0
        self.mouse_x = 0
        self.mouse_y = 0

        # --- Sidebar Controls ---
        ui_manager.set_panel_title("Interactive Demo")

        # 1. Telemetry Section
        ui_manager.add_widget("lbl_telemetry", Label(
            rect=(0, 0, 200, 20),
            text="TELEMETRY",
            color=(1.0, 0.6, 0.2, 1.0)  # Orange header
        ))

        # Mouse Tracker (This label will update in real-time)
        self.lbl_mouse = ui_manager.add_widget("mouse_pos", Label(
            rect=(0, 0, 200, 20),
            text="Mouse: (0, 0)",
            color=(0.7, 0.7, 0.7, 1.0)
        ))

        # Rotation Tracker
        self.lbl_rot = ui_manager.add_widget("rot_val", Label(
            rect=(0, 0, 200, 20),
            text="Rotation: 0.0°",
            color=(0.7, 0.7, 0.7, 1.0)
        ))

        # 2. Controls Section
        ui_manager.add_widget("spacer1", Label((0, 0, 0, 10), ""))  # Spacer

        ui_manager.add_widget("lbl_controls", Label(
            rect=(0, 0, 200, 20),
            text="CONTROLS",
            color=(0.2, 0.8, 1.0, 1.0)  # Blue header
        ))

        ui_manager.add_widget("rotate_toggle", ToggleSwitch(
            rect=(0, 0, 50, 25),
            is_on=self.is_rotating,
            callback=lambda v: setattr(self, 'is_rotating', v),
            label="Auto-Rotate"
        ))

        ui_manager.add_widget("speed_slider", Slider(
            rect=(0, 0, 200, 20),
            label="Speed",
            min_val=0, max_val=180,
            initial_val=self.rotation_speed,
            callback=lambda v: setattr(self, 'rotation_speed', v)
        ))

        # Scale slider
        ui_manager.add_widget("scale_slider", Slider(
            rect=(0, 0, 200, 20),
            label="Scale",
            min_val=0.5, max_val=2.5,
            initial_val=self.cube_scale,
            callback=lambda v: setattr(self, 'cube_scale', v)
        ))

        # Color dropdown
        self.color_options = {
            "Emerald": (0.2, 0.7, 0.3),
            "Amber": (1.0, 0.5, 0.1),
            "Azure": (0.2, 0.4, 0.9),
            "Ruby": (0.9, 0.2, 0.2),
            "Amethyst": (0.6, 0.2, 0.8),
        }
        ui_manager.add_widget("color_dropdown", Dropdown(
            rect=(0, 0, 150, 25),
            options=list(self.color_options.keys()),
            selected_index=0,
            callback=self._on_color_change,
            label="Material"
        ))

        # Reset button
        ui_manager.add_widget("reset_button", Button(
            rect=(0, 0, 100, 30),
            text="Reset View",
            callback=self._reset
        ))

    def _on_color_change(self, color_name):
        self.cube_color = self.color_options.get(color_name, (0.7, 0.7, 0.7))

    def _reset(self):
        self.rotation = 0.0
        self.cube_scale = 1.0
        self.rotation_speed = 45.0
        # Reset camera via property
        if self.camera:
            self.camera.reset()

    def mouse_position_event(self, x, y, dx, dy):
        """Track mouse movement."""
        self.mouse_x = x
        self.mouse_y = y

    def render(self, time, dt):
        # Update Logic
        if self.is_rotating:
            self.rotation += self.rotation_speed * dt

        # Update UI Labels (Real-time feedback)
        # We perform raycasting logic here (simplified for demo)
        world_x = (self.mouse_x - 800) / 100.0
        world_y = -(self.mouse_y - 450) / 100.0

        self.lbl_mouse.text = f"Mouse: {self.mouse_x}, {self.mouse_y} ({world_x:.1f}, {world_y:.1f})"
        self.lbl_rot.text = f"Rotation: {self.rotation % 360:.1f}°"

        # Render Scene
        # Ground plane
        pgfx.draw_plane(
            size=(10, 10),
            color=(0.25, 0.25, 0.28)
        )

        # Main Cube
        pgfx.draw_cube(
            size=self.cube_scale,
            color=self.cube_color,
            center=(0, 0, self.cube_scale / 2),
            rotation=(0, 0, self.rotation)
        )

        # Interactive marker following mouse (projected to ground)
        # This shows how we can use mouse input to draw 3D elements
        pgfx.draw_point(
            position=(world_x, world_y, 0.05),
            color=(1.0, 1.0, 0.0),
            size=10.0
        )


if __name__ == '__main__':
    PiVizStudio(scene_fx=UIShowcase()).run()

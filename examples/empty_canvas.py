# examples/empty_canvas.py
"""
Empty Canvas Example
===================
Demonstrates a minimal PiViz application with no additional UI or graphics.
"""

from piviz import PiVizStudio, PiVizFX, pgfx


class EmptyCanvas(PiVizFX):
    """
    An empty canvas.
    """

    def setup(self):
        # Put any initialization code here if needed
        pass

    def render(self, time, dt):
        # Put any per-frame rendering code here if needed
        pass


if __name__ == '__main__':
    PiVizStudio(scene_fx=EmptyCanvas()).run()

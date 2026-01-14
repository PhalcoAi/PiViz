# piviz/core/usd_player.py
"""
PiViz Native USD Player
=======================
Plays back .usdc / .usd simulation files.
"""

import numpy as np
import time
from typing import Optional, List, Dict

from .scene import PiVizFX
from ..graphics import primitives as pgfx
from ..ui.widgets import Label, Button, Slider, ToggleSwitch

try:
    from pxr import Usd, UsdGeom, Gf

    HAS_USD = True
except ImportError:
    HAS_USD = False


class USDPlayer(PiVizFX):
    def __init__(self, usd_path: str):
        super().__init__()
        self.usd_path = usd_path
        self.stage = None
        self.is_playing = False
        self.current_frame = 0.0
        self.speed = 1.0
        self.loop_playback = True

        self.point_instancers = []
        self.curves = []

    def setup(self):
        if not HAS_USD:
            print("CRITICAL ERROR: 'usd-core' library not found.")
            return

        print(f"Loading USD Stage: {self.usd_path}...")
        self.stage = Usd.Stage.Open(self.usd_path)
        if not self.stage:
            print(f"Failed to open stage: {self.usd_path}")
            return

        self.start_frame = self.stage.GetStartTimeCode()
        self.end_frame = self.stage.GetEndTimeCode()
        self.time_codes_per_sec = self.stage.GetTimeCodesPerSecond()
        self.current_frame = self.start_frame

        self._scan_stage()

        if self.camera:
            self.camera.set_view('iso')
            self.camera.distance = 20.0

    def _scan_stage(self):
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.PointInstancer):
                self.point_instancers.append(UsdGeom.PointInstancer(prim))
            elif prim.IsA(UsdGeom.BasisCurves):
                self.curves.append(UsdGeom.BasisCurves(prim))
            elif prim.IsA(UsdGeom.Points):
                self.point_instancers.append(UsdGeom.Points(prim))

    def render_ui_controls(self, ui_manager):
        ui_manager.set_panel_title("USD Player")
        self.lbl_frame = ui_manager.add_widget("frame", Label((0, 0, 0, 0), "Frame: 0"))

        def toggle_play(): self.is_playing = not self.is_playing

        ui_manager.add_widget("btn_play", Button((0, 0, 100, 30), "Play/Pause", toggle_play))

        def scrub(val):
            self.current_frame = val
            self.is_playing = False

        ui_manager.add_widget("scrubber",
                              Slider((0, 0, 200, 20), "Timeline", int(self.start_frame), int(self.end_frame),
                                     int(self.current_frame), scrub))

        ui_manager.add_widget("speed", Slider((0, 0, 200, 20), "Speed", 10, 500, int(self.speed * 100),
                                              lambda v: setattr(self, 'speed', v / 100.0)))

    def render(self, time_sec, dt):
        if not self.stage: return

        if self.is_playing:
            self.current_frame += dt * self.time_codes_per_sec * self.speed
            if self.current_frame > self.end_frame:
                if self.loop_playback:
                    self.current_frame = self.start_frame
                else:
                    self.current_frame = self.end_frame; self.is_playing = False

        for instancer in self.point_instancers:
            self._draw_instancer(instancer, self.current_frame)

        for curve in self.curves:
            self._draw_curve(curve, self.current_frame)

        if hasattr(self, 'lbl_frame'):
            self.lbl_frame.text = f"Frame: {self.current_frame:.1f} / {self.end_frame:.0f}"

    def _draw_instancer(self, instancer, time_code):
        if isinstance(instancer, UsdGeom.PointInstancer):
            positions_attr = instancer.GetPositionsAttr()
        else:
            positions_attr = instancer.GetPointsAttr()

        positions = positions_attr.Get(time_code)
        if positions is None or len(positions) == 0: return
        pos_np = np.array(positions, dtype='f4')

        prim = instancer.GetPrim()
        color_attr = prim.GetAttribute("primvars:displayColor")
        if not color_attr.IsValid():
            color_attr = prim.GetAttribute("displayColor")

        colors = color_attr.Get(time_code) if color_attr.IsValid() else None

        if colors is None or len(colors) == 0:
            col_np = np.array([0.7, 0.7, 0.7], dtype='f4')
            col_np = np.tile(col_np, (len(pos_np), 1))
        else:
            col_np = np.array(colors, dtype='f4')
            if len(col_np) == 1 and len(pos_np) > 1:
                col_np = np.tile(col_np, (len(pos_np), 1))

        if isinstance(instancer, UsdGeom.PointInstancer):
            scales_attr = instancer.GetScalesAttr()
            scales = scales_attr.Get(time_code)
            if scales is not None:
                scales_np = np.array(scales, dtype='f4')
                sizes_np = np.max(scales_np, axis=1) if len(scales_np) > 0 else 1.0
            else:
                sizes_np = 1.0
        else:
            widths_attr = instancer.GetWidthsAttr()
            widths = widths_attr.Get(time_code)
            sizes_np = np.array(widths, dtype='f4') if widths is not None else 1.0

        pgfx.draw_particles(pos_np, col_np, sizes=sizes_np)

    def _draw_curve(self, curve, time_code):
        # 1. Get Points
        points = curve.GetPointsAttr().Get(time_code)
        if points is None: return
        pts_np = np.array(points, dtype='f4')
        if len(pts_np) < 2: return

        # 2. Get Color
        prim = curve.GetPrim()
        color_attr = prim.GetAttribute("primvars:displayColor")
        if not color_attr.IsValid(): color_attr = prim.GetAttribute("displayColor")

        colors = color_attr.Get(time_code) if color_attr.IsValid() else None
        color = tuple(colors[0]) if colors and len(colors) > 0 else (1, 1, 1)

        # 3. Get Vertex Counts (to handle multiple curves in one primitive)
        counts_attr = curve.GetCurveVertexCountsAttr()
        counts = counts_attr.Get(time_code)

        if counts is None:
            # Single curve
            pgfx.draw_path(pts_np, color=color, width=1.5)
        else:
            # Multiple curves packed in one array
            start_idx = 0
            for count in counts:
                segment = pts_np[start_idx: start_idx + count]
                pgfx.draw_path(segment, color=color, width=1.5)
                start_idx += count

# piviz/graphics/environment.py
"""
Environment Renderers for PiViz
===============================

Modern infinite grid and coordinate axes with:
- Adaptive grid spacing based on camera distance
- Smooth fade at horizon
- Unit length markers
- Axis labels
"""

import moderngl
import numpy as np
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.theme import Theme
    from ..core.camera import Camera


class GridRenderer:
    """
    Features:
      - Variable line width (Minor/Major/Axis)
      - Infinite horizon fade (Euclidean distance)
      - Smooth LOD transitions (zoom fading)
      - Infinite Panning (Grid follows camera)
    """

    VERTEX_SHADER = '''
        #version 330 core

        uniform mat4 mvp;
        uniform vec2 u_offset; // <--- FIXED: Use // for GLSL comments

        in vec3 in_position;
        in vec4 in_color;

        out vec4 v_color;
        out vec3 v_pos; 

        void main() {
            // Apply the offset to the vertex position for World Space placement
            vec3 world_pos = in_position + vec3(u_offset, 0.0);

            gl_Position = mvp * vec4(world_pos, 1.0);

            v_color = in_color;

            // Critical: Pass LOCAL position (without offset) to fragment shader
            // This ensures the radial fade stays centered on the camera/screen
            v_pos = in_position; 
        }
    '''

    # Your preferred logic + u_opacity for LOD
    FRAGMENT_SHADER = '''
        #version 330 core

        uniform float u_limit;    // The max size of the grid geometry
        uniform float u_opacity;  // Controls layer transparency (for LOD fading)

        in vec4 v_color;
        in vec3 v_pos;
        out vec4 frag_color;

        void main() {
            float dist = length(v_pos.xy); // Chebyshev to Euclidean distance

            // 2. Horizon Fade
            // Starts fading at 66% of limit, fully transparent at 99%
            // This ensures the hard edge of the geometry is never seen.
            float horizon_alpha = 1.0 - smoothstep(u_limit * 0.66, u_limit * 0.99, dist);

            // 3. Combine: Vertex Color * Horizon Fade * LOD Opacity
            float final_alpha = v_color.a * horizon_alpha * u_opacity;

            frag_color = vec4(v_color.rgb, final_alpha);

            // Optimization
            if (frag_color.a <= 0.0) discard;
        }
    '''

    def __init__(self, ctx: moderngl.Context, theme: 'Theme'):
        self.ctx = ctx
        self.theme = theme
        self.spacing = 1.0
        self.extent = 1.0

        self.prog = ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )

        # Store separate VAOs/VBOs for each layer to allow different line widths
        self.layers = {
            'minor': {'vao': None, 'vbo': None, 'count': 0, 'width': 1.0},
            'major': {'vao': None, 'vbo': None, 'count': 0, 'width': 2.0},
            'axis': {'vao': None, 'vbo': None, 'count': 0, 'width': 3.0},
        }

        self._last_spacing = None

    @staticmethod
    def smoothstep(edge0, edge1, x):
        """Standard GLSL smoothstep implementation in Python"""
        x = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return x * x * (3 - 2 * x)

    def _build_grid(self, spacing: float, extent: float):
        verts_minor = []
        verts_major = []
        verts_axis = []

        major = spacing
        minor = spacing / 10.0

        def add_line(target_list, x1, y1, x2, y2, color):
            target_list.extend([x1, y1, 0.0, *color, x2, y2, 0.0, *color])

        # 1. Minor grid
        steps_minor = int(extent / minor)
        for i in range(-steps_minor, steps_minor + 1):
            p = i * minor
            if abs(i % 10) == 0: continue  # Skip overlapping major lines
            add_line(verts_minor, -extent, p, extent, p, self.theme.grid_minor)
            add_line(verts_minor, p, -extent, p, extent, self.theme.grid_minor)

        # 2. Major grid
        steps_major = int(extent / major)
        for i in range(-steps_major, steps_major + 1):
            p = i * major
            if i == 0: continue  # Skip axis
            add_line(verts_major, -extent, p, extent, p, self.theme.grid_major)
            add_line(verts_major, p, -extent, p, extent, self.theme.grid_major)

        # 3. Axis (World Origin)
        add_line(verts_axis, -extent, 0, extent, 0, self.theme.axis_x)
        add_line(verts_axis, 0, -extent, 0, extent, self.theme.axis_y)

        # Upload to GPU
        def update_layer(name, vertices):
            layer = self.layers[name]
            if layer['vbo']: layer['vbo'].release()
            if layer['vao']: layer['vao'].release()

            data = np.array(vertices, dtype='f4')
            layer['count'] = len(vertices) // 7
            if layer['count'] > 0:
                layer['vbo'] = self.ctx.buffer(data)
                layer['vao'] = self.ctx.vertex_array(
                    self.prog, [(layer['vbo'], '3f 4f', 'in_position', 'in_color')],
                )
            else:
                layer['vbo'] = None
                layer['vao'] = None

        update_layer('minor', verts_minor)
        update_layer('major', verts_major)
        update_layer('axis', verts_axis)

        self.spacing = spacing
        self.extent = extent

    def render(self, view: np.ndarray, proj: np.ndarray, camera: 'Camera'):
        distance = max(camera.distance, 0.001)

        # --- LOD Calculation ---
        log_dist = math.log10(distance * 1.618)
        base = math.floor(log_dist)
        fract = log_dist - base
        spacing = 10.0 ** base
        extent = spacing * 30.0

        if spacing != self._last_spacing:
            self._build_grid(spacing, extent)
            self._last_spacing = spacing

        # --- Uniform Updates ---
        mvp = proj @ view
        self.prog['mvp'].write(mvp.T.tobytes())

        if 'u_limit' in self.prog:
            self.prog['u_limit'].value = distance * 2.0

        # --- Panning / Snapping Logic ---

        cam_x, cam_y = camera.target[0], camera.target[1]
        snap_step = spacing
        offset_x = round(cam_x / snap_step) * snap_step
        offset_y = round(cam_y / snap_step) * snap_step

        # --- GL States ---
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # --- Render Layers ---
        minor_alpha = 1.0 - self.smoothstep(0.5, 1.0, fract)

        for name, layer in self.layers.items():
            if not layer['vao']: continue

            self.ctx.line_width = layer['width']

            opacity = 1.0
            if name == 'minor':
                opacity = minor_alpha

            # Skip invisible draws
            if opacity <= 0.001: continue

            if 'u_opacity' in self.prog:
                self.prog['u_opacity'].value = opacity

            # --- Apply Offset ---
            # Major/Minor grids move with the camera (snapped)
            # The Axis lines must stay at the true World Origin (0,0)
            if name == 'axis':
                self.prog['u_offset'].value = (0.0, 0.0)
            else:
                self.prog['u_offset'].value = (offset_x, offset_y)

            layer['vao'].render(moderngl.LINES)

    def set_theme(self, theme: 'Theme'):
        self.theme = theme
        self._build_grid(self.spacing, self.extent)


class AxesRenderer:
    """
    Coordinate axes renderer with unit markers.
    
    Features:
    - RGB color coding (X=red, Y=green, Z=blue)
    - Arrowheads
    - Unit tick marks
    - Optional labels
    """

    VERTEX_SHADER = '''
        #version 330
        
        uniform mat4 view;
        uniform mat4 projection;
        
        in vec3 in_position;
        in vec4 in_color;
        
        out vec4 v_color;
        
        void main() {
            gl_Position = projection * view * vec4(in_position, 1.0);
            v_color = in_color;
        }
    '''

    FRAGMENT_SHADER = '''
        #version 330
        
        in vec4 v_color;
        out vec4 frag_color;
        
        void main() {
            frag_color = v_color;
        }
    '''

    def __init__(self, ctx: moderngl.Context, theme: 'Theme'):
        self.ctx = ctx
        self.theme = theme
        self.axis_length = 5.0

        # Compile shaders
        self.prog = ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )

        self._build_geometry()

    def set_theme(self, theme: 'Theme'):
        """Update colors from theme."""
        self.theme = theme
        self._build_geometry()

    def _build_geometry(self):
        """Build axis lines, arrowheads, and tick marks."""
        vertices = []

        length = self.axis_length
        arrow_size = 0.15
        tick_size = 0.08

        # Colors from theme
        cx = self.theme.axis_x
        cy = self.theme.axis_y
        cz = self.theme.axis_z

        # X axis (red)
        vertices.extend([0, 0, 0, *cx, length, 0, 0, *cx])
        # X arrowhead
        vertices.extend([length, 0, 0, *cx, length - arrow_size, arrow_size * 0.5, 0, *cx])
        vertices.extend([length, 0, 0, *cx, length - arrow_size, -arrow_size * 0.5, 0, *cx])
        vertices.extend([length, 0, 0, *cx, length - arrow_size, 0, arrow_size * 0.5, *cx])
        vertices.extend([length, 0, 0, *cx, length - arrow_size, 0, -arrow_size * 0.5, *cx])

        # Y axis (green)
        vertices.extend([0, 0, 0, *cy, 0, length, 0, *cy])
        # Y arrowhead
        vertices.extend([0, length, 0, *cy, arrow_size * 0.5, length - arrow_size, 0, *cy])
        vertices.extend([0, length, 0, *cy, -arrow_size * 0.5, length - arrow_size, 0, *cy])
        vertices.extend([0, length, 0, *cy, 0, length - arrow_size, arrow_size * 0.5, *cy])
        vertices.extend([0, length, 0, *cy, 0, length - arrow_size, -arrow_size * 0.5, *cy])

        # Z axis (blue)
        vertices.extend([0, 0, 0, *cz, 0, 0, length, *cz])
        # Z arrowhead
        vertices.extend([0, 0, length, *cz, arrow_size * 0.5, 0, length - arrow_size, *cz])
        vertices.extend([0, 0, length, *cz, -arrow_size * 0.5, 0, length - arrow_size, *cz])
        vertices.extend([0, 0, length, *cz, 0, arrow_size * 0.5, length - arrow_size, *cz])
        vertices.extend([0, 0, length, *cz, 0, -arrow_size * 0.5, length - arrow_size, *cz])

        # Tick marks at unit intervals
        for i in range(1, int(length)):
            # X ticks
            vertices.extend([i, 0, -tick_size, *cx, i, 0, tick_size, *cx])
            vertices.extend([i, -tick_size, 0, *cx, i, tick_size, 0, *cx])
            # Y ticks
            vertices.extend([0, i, -tick_size, *cy, 0, i, tick_size, *cy])
            vertices.extend([-tick_size, i, 0, *cy, tick_size, i, 0, *cy])
            # Z ticks
            vertices.extend([-tick_size, 0, i, *cz, tick_size, 0, i, *cz])
            vertices.extend([0, -tick_size, i, *cz, 0, tick_size, i, *cz])

        vertices = np.array(vertices, dtype='f4')

        # Release old buffer if exists
        if hasattr(self, 'vbo'):
            self.vbo.release()
            self.vao.release()

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 4f', 'in_position', 'in_color')])
        self.vertex_count = len(vertices) // 7

    def render(self, view: np.ndarray, proj: np.ndarray):
        """Render the coordinate axes."""
        self.prog['view'].write(view.T.tobytes())
        self.prog['projection'].write(proj.T.tobytes())

        self.ctx.line_width = 2.0
        self.vao.render(moderngl.LINES)

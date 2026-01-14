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
    Modern infinite grid renderer.
    
    Features:
    - Adaptive spacing (adjusts to zoom level)
    - Smooth horizon fade
    - Major/minor grid lines
    - Unit markers
    """
    
    VERTEX_SHADER = '''
        #version 330
        
        uniform mat4 view;
        uniform mat4 projection;
        
        in vec3 in_position;
        
        out vec3 world_pos;
        out vec3 cam_pos;
        
        void main() {
            world_pos = in_position;
            gl_Position = projection * view * vec4(in_position, 1.0);
            
            // Extract camera position from view matrix
            mat3 rot = transpose(mat3(view));
            cam_pos = -rot * view[3].xyz;
        }
    '''
    
    FRAGMENT_SHADER = '''
        #version 330
        
        uniform vec4 color_major;
        uniform vec4 color_minor;
        uniform float grid_spacing;
        uniform float line_width;
        uniform float fade_distance;
        
        in vec3 world_pos;
        in vec3 cam_pos;
        
        out vec4 frag_color;
        
        float grid_line(float coord, float spacing, float width) {
            float line = abs(fract(coord / spacing - 0.5) - 0.5) * spacing;
            return 1.0 - smoothstep(0.0, width, line);
        }
        
        void main() {
            // Distance from camera for fade
            float dist = length(world_pos.xy - cam_pos.xy);
            float fade = 1.0 - smoothstep(fade_distance * 0.3, fade_distance, dist);
            
            // Major grid (every grid_spacing units)
            float major_x = grid_line(world_pos.x, grid_spacing, line_width);
            float major_y = grid_line(world_pos.y, grid_spacing, line_width);
            float major = max(major_x, major_y);
            
            // Minor grid (subdivisions)
            float minor_spacing = grid_spacing / 5.0;
            float minor_x = grid_line(world_pos.x, minor_spacing, line_width * 0.5);
            float minor_y = grid_line(world_pos.y, minor_spacing, line_width * 0.5);
            float minor = max(minor_x, minor_y) * (1.0 - major);
            
            // Combine with colors
            vec4 color = color_major * major + color_minor * minor;
            color.a *= fade;
            
            // Origin lines (thicker)
            float origin_width = line_width * 2.0;
            float origin_x = grid_line(world_pos.x, 1000.0, origin_width);
            float origin_y = grid_line(world_pos.y, 1000.0, origin_width);
            float origin = max(origin_x, origin_y);
            
            if (origin > 0.1) {
                color = mix(color, vec4(0.5, 0.5, 0.55, fade), origin * 0.5);
            }
            
            if (color.a < 0.01) discard;
            
            frag_color = color;
        }
    '''
    
    def __init__(self, ctx: moderngl.Context, theme: 'Theme'):
        self.ctx = ctx
        self.theme = theme
        
        # Compile shaders
        self.prog = ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        
        # Create a large ground plane
        size = 500.0
        vertices = np.array([
            [-size, -size, 0],
            [size, -size, 0],
            [size, size, 0],
            [-size, size, 0],
        ], dtype='f4')
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        
        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '3f', 'in_position')], self.ibo)
        
        self._update_uniforms()

    def set_theme(self, theme: 'Theme'):
        """Update colors from theme."""
        self.theme = theme
        self._update_uniforms()

    def _update_uniforms(self):
        """Update shader uniforms from theme."""
        self.prog['color_major'].value = self.theme.grid_major
        self.prog['color_minor'].value = self.theme.grid_minor

    def render(self, view: np.ndarray, proj: np.ndarray, camera: 'Camera'):
        """Render the grid."""
        # Adaptive grid spacing based on camera distance
        base_spacing = 1.0
        distance = camera.distance
        
        # Calculate appropriate grid spacing (powers of 10)
        if distance < 2:
            spacing = 0.1
        elif distance < 20:
            spacing = 1.0
        elif distance < 200:
            spacing = 10.0
        else:
            spacing = 100.0
        
        # Update uniforms
        self.prog['view'].write(view.T.tobytes())
        self.prog['projection'].write(proj.T.tobytes())
        self.prog['grid_spacing'].value = spacing
        self.prog['line_width'].value = spacing * 0.015
        self.prog['fade_distance'].value = distance * 3.0
        
        # Render with blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.CULL_FACE)


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

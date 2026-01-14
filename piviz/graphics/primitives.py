# piviz/graphics/primitives.py
"""
Primitive Drawing Functions (pgfx)
==================================
Immediate-mode style drawing functions.
"""

import moderngl
import numpy as np
import math
from typing import Tuple, Optional, Union, TYPE_CHECKING, List

# Global context reference
_ctx: Optional[moderngl.Context] = None
_programs: dict = {}
_current_view: Optional[np.ndarray] = None
_current_proj: Optional[np.ndarray] = None


def _get_program(name: str) -> moderngl.Program:
    global _ctx, _programs
    if name not in _programs:
        if name == 'solid':
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 model;
                    uniform mat4 view;
                    uniform mat4 projection;
                    in vec3 in_position;
                    in vec3 in_normal;
                    out vec3 v_normal;
                    out vec3 v_position;
                    void main() {
                        v_position = vec3(model * vec4(in_position, 1.0));
                        v_normal = mat3(transpose(inverse(model))) * in_normal;
                        gl_Position = projection * view * vec4(v_position, 1.0);
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform vec4 color;
                    uniform vec3 light_dir;
                    in vec3 v_normal;
                    in vec3 v_position;
                    out vec4 frag_color;
                    void main() {
                        vec3 norm = normalize(v_normal);
                        float diff = max(dot(norm, light_dir), 0.0);
                        float ambient = 0.3;
                        vec3 result = color.rgb * (ambient + diff * 0.7);
                        frag_color = vec4(result, color.a);
                    }
                '''
            )
        elif name == 'vertex_color':
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 model;
                    uniform mat4 view;
                    uniform mat4 projection;
                    in vec3 in_position;
                    in vec3 in_color;
                    out vec3 v_color;
                    void main() {
                        gl_Position = projection * view * model * vec4(in_position, 1.0);
                        v_color = in_color;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    in vec3 v_color;
                    out vec4 frag_color;
                    void main() {
                        frag_color = vec4(v_color, 1.0);
                    }
                '''
            )
        elif name == 'line':
            _programs[name] = _ctx.program(
                vertex_shader='''
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
                ''',
                fragment_shader='''
                    #version 330
                    in vec4 v_color;
                    out vec4 frag_color;
                    void main() {
                        frag_color = v_color;
                    }
                '''
            )
        elif name == 'particles':
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform float scale_factor;
                    in vec3 in_position;
                    in vec3 in_color;
                    in float in_size;
                    out vec3 v_color;
                    void main() {
                        gl_Position = projection * view * vec4(in_position, 1.0);
                        float dist = gl_Position.w;
                        gl_PointSize = (in_size * scale_factor) / (dist * 0.5 + 0.1);
                        v_color = in_color;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    in vec3 v_color;
                    out vec4 frag_color;
                    void main() {
                        vec2 coord = gl_PointCoord * 2.0 - 1.0;
                        float dist_sq = dot(coord, coord);
                        if (dist_sq > 1.0) discard;
                        float alpha = 1.0 - smoothstep(0.8, 1.0, sqrt(dist_sq));
                        frag_color = vec4(v_color, alpha);
                    }
                '''
            )
    return _programs[name]


def _create_model_matrix(center, rotation=(0, 0, 0), scale=(1, 1, 1)):
    T = np.eye(4, dtype='f4')
    T[0, 3], T[1, 3], T[2, 3] = center

    rx, ry, rz = [math.radians(a) for a in rotation]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype='f4')
    Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
    Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')

    S = np.diag([scale[0], scale[1], scale[2], 1.0]).astype('f4')
    return T @ Rz @ Ry @ Rx @ S


def _init_context(ctx, view, proj):
    global _ctx, _current_view, _current_proj
    _ctx = ctx
    _current_view = view
    _current_proj = proj


def _ensure_rgba(color):
    if len(color) == 3:
        return (*color, 1.0)
    return color


# === Drawing Functions ===

def draw_particles(positions, colors, sizes=1.0):
    global _ctx
    if _ctx is None: return
    prog = _get_program('particles')

    if not isinstance(positions, np.ndarray): positions = np.array(positions, dtype='f4')
    if not isinstance(colors, np.ndarray): colors = np.array(colors, dtype='f4')

    num = len(positions)
    if isinstance(sizes, (int, float)):
        sizes = np.full(num, sizes, dtype='f4')
    elif not isinstance(sizes, np.ndarray):
        sizes = np.array(sizes, dtype='f4')

    vbo_pos = _ctx.buffer(positions.astype('f4').tobytes())
    vbo_col = _ctx.buffer(colors.astype('f4').tobytes())
    vbo_size = _ctx.buffer(sizes.astype('f4').tobytes())

    vao = _ctx.vertex_array(prog, [
        (vbo_pos, '3f', 'in_position'),
        (vbo_col, '3f', 'in_color'),
        (vbo_size, '1f', 'in_size')
    ])

    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['scale_factor'].value = _ctx.viewport[3] * 0.002

    _ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    _ctx.depth_mask = False
    vao.render(moderngl.POINTS)
    _ctx.depth_mask = True
    vbo_pos.release();
    vbo_col.release();
    vbo_size.release();
    vao.release()


def draw_path(points, color=(1, 1, 1), width=1.0):
    """
    Draw a continuous line strip (path) from an array of points.

    Args:
        points: Numpy array of shape (N, 3)
        color: RGB or RGBA tuple
        width: Line width
    """
    global _ctx
    if _ctx is None: return
    prog = _get_program('line')
    color = _ensure_rgba(color)

    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype='f4')

    if len(points) < 2:
        return

    # Create color array matching the number of vertices
    # (Since 'line' shader expects per-vertex color)
    num_points = len(points)
    colors = np.tile(color, (num_points, 1)).astype('f4')

    vbo_pos = _ctx.buffer(points.tobytes())
    vbo_col = _ctx.buffer(colors.tobytes())

    vao = _ctx.vertex_array(prog, [
        (vbo_pos, '3f', 'in_position'),
        (vbo_col, '4f', 'in_color')
    ])

    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    _ctx.line_width = width

    vao.render(moderngl.LINE_STRIP)
    vbo_pos.release()
    vbo_col.release()
    vao.release()


def draw_plane(size=(5, 5), color=(0.5, 0.5, 0.5), center=(0, 0, 0), normal=(0, 0, 1)):
    global _ctx
    if _ctx is None: return
    prog = _get_program('solid')
    color = _ensure_rgba(color)

    w, h = size[0] / 2, size[1] / 2
    n = np.array(normal, dtype='f4');
    n /= np.linalg.norm(n)

    if abs(n[2]) < 0.9:
        right = np.cross(n, [0, 0, 1])
    else:
        right = np.cross(n, [1, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, n)

    c = np.array(center, dtype='f4')
    v1, v2 = c - right * w - up * h, c + right * w - up * h
    v3, v4 = c + right * w + up * h, c - right * w + up * h

    vertices = np.array([*v1, *n, *v2, *n, *v3, *n, *v1, *n, *v3, *n, *v4, *n], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_normal')])

    model = np.eye(4, dtype='f4')
    prog['model'].write(model.T.tobytes())
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['color'].value = color
    prog['light_dir'].value = (0.5, 0.3, 0.8)

    _ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    _ctx.enable(moderngl.CULL_FACE)
    vbo.release();
    vao.release()


def draw_cube(size=1.0, color=(0.7, 0.7, 0.7), center=(0, 0, 0), rotation=(0, 0, 0)):
    global _ctx
    if _ctx is None: return
    prog = _get_program('solid')
    color = _ensure_rgba(color)

    s = size / 2
    vertices = []
    faces = [
        ((0, 0, 1), (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)),
        ((0, 0, -1), (s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)),
        ((0, 1, 0), (-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)),
        ((0, -1, 0), (-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)),
        ((1, 0, 0), (s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)),
        ((-1, 0, 0), (-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)),
    ]
    for norm, v1, v2, v3, v4 in faces:
        for v in [v1, v2, v3, v1, v3, v4]:
            vertices.extend([*v, *norm])

    vbo = _ctx.buffer(np.array(vertices, dtype='f4').tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_normal')])

    model = _create_model_matrix(center, rotation)
    prog['model'].write(model.T.tobytes())
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['color'].value = color
    prog['light_dir'].value = (0.5, 0.3, 0.8)

    vao.render(moderngl.TRIANGLES)
    vbo.release();
    vao.release()


def draw_sphere(radius=0.5, color=(0.7, 0.7, 0.7), center=(0, 0, 0), detail=16):
    global _ctx
    if _ctx is None: return
    prog = _get_program('solid')
    color = _ensure_rgba(color)

    vertices = []
    for i in range(detail):
        lat0 = math.pi * (-0.5 + float(i) / detail)
        lat1 = math.pi * (-0.5 + float(i + 1) / detail)
        for j in range(detail):
            lon0 = 2 * math.pi * float(j) / detail
            lon1 = 2 * math.pi * float(j + 1) / detail

            def p(lat, lon):
                x = radius * math.cos(lat) * math.cos(lon)
                y = radius * math.cos(lat) * math.sin(lon)
                z = radius * math.sin(lat)
                return (x, y, z, x / radius, y / radius, z / radius)

            vertices.extend(
                [*p(lat0, lon0), *p(lat0, lon1), *p(lat1, lon1), *p(lat0, lon0), *p(lat1, lon1), *p(lat1, lon0)])

    vbo = _ctx.buffer(np.array(vertices, dtype='f4').tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_normal')])

    model = _create_model_matrix(center)
    prog['model'].write(model.T.tobytes())
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['color'].value = color
    prog['light_dir'].value = (0.5, 0.3, 0.8)

    vao.render(moderngl.TRIANGLES)
    vbo.release();
    vao.release()


def draw_cylinder(start=(0, 0, 0), end=(0, 0, 1), radius=0.2, color=(0.7, 0.7, 0.7), detail=24):
    global _ctx
    if _ctx is None: return
    prog = _get_program('solid')
    color = _ensure_rgba(color)

    start = np.array(start, dtype='f4');
    end = np.array(end, dtype='f4')
    axis = end - start;
    length = np.linalg.norm(axis)
    if length < 0.001: return
    axis /= length

    perp1 = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < 0.9 else np.cross(axis, [1, 0, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    vertices = []
    for i in range(detail):
        a0 = 2 * math.pi * i / detail;
        a1 = 2 * math.pi * (i + 1) / detail

        def p(ang, base):
            off = radius * (perp1 * math.cos(ang) + perp2 * math.sin(ang))
            return (*(base + off), *(off / radius))

        vertices.extend([*p(a0, start), *p(a1, start), *p(a1, end), *p(a0, start), *p(a1, end), *p(a0, end)])

    def draw_cap(center, sign):
        n = axis * sign
        for i in range(detail):
            a0, a1 = 2 * math.pi * i / detail, 2 * math.pi * (i + 1) / detail
            p1 = center + radius * (perp1 * math.cos(a0) + perp2 * math.sin(a0))
            p2 = center + radius * (perp1 * math.cos(a1) + perp2 * math.sin(a1))
            if sign > 0:
                vertices.extend([*center, *n, *p1, *n, *p2, *n])
            else:
                vertices.extend([*center, *n, *p2, *n, *p1, *n])

    draw_cap(end, 1.0);
    draw_cap(start, -1.0)

    vbo = _ctx.buffer(np.array(vertices, dtype='f4').tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_normal')])

    model = np.eye(4, dtype='f4')
    prog['model'].write(model.T.tobytes())
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['color'].value = color
    prog['light_dir'].value = (0.5, 0.3, 0.8)

    vao.render(moderngl.TRIANGLES)
    vbo.release();
    vao.release()


def draw_cone(base=(0, 0, 0), tip=(0, 0, 1), radius=0.3, color=(0.7, 0.7, 0.7), detail=24):
    global _ctx
    if _ctx is None: return
    prog = _get_program('solid')
    color = _ensure_rgba(color)

    base = np.array(base, dtype='f4');
    tip = np.array(tip, dtype='f4')
    axis = tip - base;
    height = np.linalg.norm(axis)
    if height < 0.001: return
    axis /= height

    perp1 = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < 0.9 else np.cross(axis, [1, 0, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    vertices = []
    slope = radius / height

    for i in range(detail):
        a0, a1 = 2 * math.pi * i / detail, 2 * math.pi * (i + 1) / detail
        p0 = base + radius * (perp1 * math.cos(a0) + perp2 * math.sin(a0))
        p1 = base + radius * (perp1 * math.cos(a1) + perp2 * math.sin(a1))
        n0 = p0 - base;
        n0 = (n0 / np.linalg.norm(n0) + axis * slope);
        n0 /= np.linalg.norm(n0)
        n1 = p1 - base;
        n1 = (n1 / np.linalg.norm(n1) + axis * slope);
        n1 /= np.linalg.norm(n1)
        vertices.extend([*p0, *n0, *p1, *n1, *tip, *axis])

    n_cap = -axis
    for i in range(detail):
        a0, a1 = 2 * math.pi * i / detail, 2 * math.pi * (i + 1) / detail
        p0 = base + radius * (perp1 * math.cos(a0) + perp2 * math.sin(a0))
        p1 = base + radius * (perp1 * math.cos(a1) + perp2 * math.sin(a1))
        vertices.extend([*base, *n_cap, *p1, *n_cap, *p0, *n_cap])

    vbo = _ctx.buffer(np.array(vertices, dtype='f4').tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_normal')])

    model = np.eye(4, dtype='f4')
    prog['model'].write(model.T.tobytes())
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['color'].value = color
    prog['light_dir'].value = (0.5, 0.3, 0.8)

    vao.render(moderngl.TRIANGLES)
    vbo.release();
    vao.release()


def draw_arrow(start, end, color=(1, 1, 1), head_size=0.1, head_radius=None, width_radius=0.03):
    start = np.array(start, dtype='f4');
    end = np.array(end, dtype='f4')
    d = end - start;
    l = np.linalg.norm(d)
    if l < 0.001: return
    d /= l
    hl = min(head_size if head_size > 0.1 else l * 0.2, l)
    hr = head_radius if head_radius else width_radius * 2.5
    split = end - d * hl
    draw_cylinder(tuple(start), tuple(split), radius=width_radius, color=color)
    draw_cone(tuple(split), tuple(end), radius=hr, color=color)


def draw_line(start, end, color=(1, 1, 1), width=1.0):
    global _ctx
    if _ctx is None: return
    prog = _get_program('line')
    color = _ensure_rgba(color)
    vertices = np.array([*start, *color, *end, *color], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    _ctx.line_width = width
    vao.render(moderngl.LINES)
    vbo.release();
    vao.release()


def draw_triangle(v1, v2, v3, color=(0.7, 0.7, 0.7)):
    global _ctx
    if _ctx is None: return
    prog = _get_program('solid')
    color = _ensure_rgba(color)
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
    norm = np.cross(v2 - v1, v3 - v1);
    norm /= np.linalg.norm(norm)
    vertices = np.array([*v1, *norm, *v2, *norm, *v3, *norm], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_normal')])
    model = np.eye(4, dtype='f4')
    prog['model'].write(model.T.tobytes())
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['color'].value = color
    prog['light_dir'].value = (0.5, 0.3, 0.8)
    _ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    _ctx.enable(moderngl.CULL_FACE)
    vbo.release();
    vao.release()


def draw_face(v1, v2, v3, c1=(1, 0, 0), c2=(0, 1, 0), c3=(0, 0, 1)):
    global _ctx
    if _ctx is None: return
    prog = _get_program('vertex_color')
    vertices = np.array([*v1, *c1, *v2, *c2, *v3, *c3], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
    model = np.eye(4, dtype='f4')
    prog['model'].write(model.T.tobytes())
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    _ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    _ctx.enable(moderngl.CULL_FACE)
    vbo.release();
    vao.release()


def draw_point(position, color=(1, 1, 1), size=5.0):
    global _ctx
    if _ctx is None: return
    prog = _get_program('line')
    color = _ensure_rgba(color)
    vertices = np.array([*position, *color], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    _ctx.point_size = size
    _ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    vao.render(moderngl.POINTS)
    vbo.release();
    vao.release()

# piviz/graphics/primitives.py
"""
PiViz High-Performance Primitives (v2.0)
========================================

ARCHITECTURE:
- All draw_*() calls are DEFERRED - they add to internal batches
- Actual GPU rendering happens in flush_all() called by the engine
- Geometry is cached and reused across frames
- Buffers are persistent and grow as needed (no per-frame allocations)

This gives 10-100x performance improvement with ZERO user code changes.

Author: Yogesh Phalak
"""

import moderngl
import numpy as np
import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field

# ============================================================
# GLOBAL STATE
# ============================================================

_ctx: Optional[moderngl.Context] = None
_programs: Dict[str, moderngl.Program] = {}
_current_view: Optional[np.ndarray] = None
_current_proj: Optional[np.ndarray] = None

# Batched rendering queues
_sphere_queue: List[Tuple] = []
_cube_queue: List[Tuple] = []
_cylinder_queue: List[Tuple] = []
_cone_queue: List[Tuple] = []
_line_queue: List[Tuple] = []
_triangle_queue: List[Tuple] = []

# Cached geometry (created once, reused forever)
_cached_sphere_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}
_cached_cube_vbo: Optional[Tuple[moderngl.Buffer, int]] = None
_cached_unit_cylinder_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}

# Persistent instance buffers (grow as needed, never shrink)
_instance_buffer: Optional[moderngl.Buffer] = None
_instance_buffer_size: int = 0
_line_buffer: Optional[moderngl.Buffer] = None
_line_buffer_size: int = 0
_triangle_buffer: Optional[moderngl.Buffer] = None
_triangle_buffer_size: int = 0


# ============================================================
# MATERIAL SETTINGS
# ============================================================

@dataclass
class MaterialSettings:
    shininess: float = 32.0
    specular_strength: float = 0.5
    ambient: float = 0.3
    use_specular: bool = True


_material = MaterialSettings()


def set_material_shiny(shiny: bool = True, shininess: float = 32.0, specular: float = 0.5):
    """Configure material appearance."""
    _material.use_specular = shiny
    _material.shininess = max(1.0, min(128.0, shininess))
    _material.specular_strength = max(0.0, min(1.0, specular))


def set_material_matte():
    """Set matte (non-reflective) material."""
    set_material_shiny(shiny=False)


# ============================================================
# SHADER PROGRAMS
# ============================================================

def _get_program(name: str) -> moderngl.Program:
    """Get or create shader program."""
    global _ctx, _programs

    if name not in _programs:
        if name == 'instanced_solid':
            # Instanced rendering with per-instance transform and color
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform vec3 light_dir;

                    in vec3 in_position;
                    in vec3 in_normal;

                    // Per-instance: 4x4 transform matrix + RGBA color
                    in vec4 inst_row0;
                    in vec4 inst_row1;
                    in vec4 inst_row2;
                    in vec4 inst_row3;
                    in vec4 inst_color;

                    out vec3 v_normal;
                    out vec3 v_position;
                    out vec4 v_color;
                    out vec3 v_light_dir;

                    void main() {
                        mat4 model = mat4(inst_row0, inst_row1, inst_row2, inst_row3);
                        vec4 world_pos = model * vec4(in_position, 1.0);
                        v_position = world_pos.xyz;
                        v_normal = mat3(transpose(inverse(model))) * in_normal;
                        v_color = inst_color;
                        v_light_dir = light_dir;
                        gl_Position = projection * view * world_pos;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform float shininess;
                    uniform float specular_strength;
                    uniform bool use_specular;
                    uniform vec3 cam_pos;

                    in vec3 v_normal;
                    in vec3 v_position;
                    in vec4 v_color;
                    in vec3 v_light_dir;

                    out vec4 frag_color;

                    void main() {
                        vec3 norm = normalize(v_normal);
                        vec3 light = normalize(v_light_dir);

                        float diff = max(dot(norm, light), 0.0);

                        float spec = 0.0;
                        if (use_specular && diff > 0.0) {
                            vec3 view_dir = normalize(cam_pos - v_position);
                            vec3 halfway = normalize(light + view_dir);
                            spec = pow(max(dot(norm, halfway), 0.0), shininess);
                        }

                        float ambient = 0.3;
                        vec3 result = v_color.rgb * (ambient + diff * 0.7);
                        if (use_specular) {
                            result += vec3(1.0) * spec * specular_strength;
                        }

                        frag_color = vec4(result, v_color.a);
                    }
                '''
            )

        elif name == 'batched_lines':
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

        elif name == 'batched_triangles':
            # Batched triangle rendering with per-vertex color and lighting
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;

                    in vec3 in_position;
                    in vec3 in_normal;
                    in vec4 in_color;

                    out vec3 v_normal;
                    out vec3 v_position;
                    out vec4 v_color;

                    void main() {
                        v_position = in_position;
                        v_normal = in_normal;
                        v_color = in_color;
                        gl_Position = projection * view * vec4(in_position, 1.0);
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform vec3 light_dir;
                    uniform float shininess;
                    uniform float specular_strength;
                    uniform bool use_specular;
                    uniform vec3 cam_pos;

                    in vec3 v_normal;
                    in vec3 v_position;
                    in vec4 v_color;

                    out vec4 frag_color;

                    void main() {
                        vec3 norm = normalize(v_normal);
                        vec3 light = normalize(light_dir);

                        float diff = max(dot(norm, light), 0.0);

                        // Also light from behind for two-sided lighting
                        float diff_back = max(dot(-norm, light), 0.0);
                        diff = max(diff, diff_back * 0.5);

                        float spec = 0.0;
                        if (use_specular && diff > 0.0) {
                            vec3 view_dir = normalize(cam_pos - v_position);
                            vec3 halfway = normalize(light + view_dir);
                            spec = pow(max(dot(norm, halfway), 0.0), shininess);
                        }

                        float ambient = 0.3;
                        vec3 result = v_color.rgb * (ambient + diff * 0.7);
                        if (use_specular) {
                            result += vec3(1.0) * spec * specular_strength;
                        }

                        frag_color = vec4(result, v_color.a);
                    }
                '''
            )

    return _programs[name]


# ============================================================
# GEOMETRY GENERATION (Called once, cached forever)
# ============================================================

def _generate_sphere_geometry(detail: int = 12) -> np.ndarray:
    """Generate unit sphere vertices (position + normal)."""
    vertices = []
    for i in range(detail):
        lat0 = math.pi * (-0.5 + float(i) / detail)
        lat1 = math.pi * (-0.5 + float(i + 1) / detail)

        for j in range(detail):
            lon0 = 2 * math.pi * float(j) / detail
            lon1 = 2 * math.pi * float(j + 1) / detail

            def p(lat, lon):
                x = math.cos(lat) * math.cos(lon)
                y = math.cos(lat) * math.sin(lon)
                z = math.sin(lat)
                return [x, y, z, x, y, z]  # pos, normal (unit sphere)

            vertices.extend(p(lat0, lon0))
            vertices.extend(p(lat0, lon1))
            vertices.extend(p(lat1, lon1))
            vertices.extend(p(lat0, lon0))
            vertices.extend(p(lat1, lon1))
            vertices.extend(p(lat1, lon0))

    return np.array(vertices, dtype='f4')


def _generate_cube_geometry() -> np.ndarray:
    """Generate unit cube vertices."""
    s = 0.5
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
    return np.array(vertices, dtype='f4')


def _generate_cylinder_geometry(detail: int = 16) -> np.ndarray:
    """Generate unit cylinder (length=1 along Z, radius=1)."""
    vertices = []

    # Side faces
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)

        # Two triangles per segment
        # Bottom-left, bottom-right, top-right
        vertices.extend([c0, s0, 0, c0, s0, 0])
        vertices.extend([c1, s1, 0, c1, s1, 0])
        vertices.extend([c1, s1, 1, c1, s1, 0])
        # Bottom-left, top-right, top-left
        vertices.extend([c0, s0, 0, c0, s0, 0])
        vertices.extend([c1, s1, 1, c1, s1, 0])
        vertices.extend([c0, s0, 1, c0, s0, 0])

    # Top cap
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        vertices.extend([0, 0, 1, 0, 0, 1])
        vertices.extend([math.cos(a0), math.sin(a0), 1, 0, 0, 1])
        vertices.extend([math.cos(a1), math.sin(a1), 1, 0, 0, 1])

    # Bottom cap
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        vertices.extend([0, 0, 0, 0, 0, -1])
        vertices.extend([math.cos(a1), math.sin(a1), 0, 0, 0, -1])
        vertices.extend([math.cos(a0), math.sin(a0), 0, 0, 0, -1])

    return np.array(vertices, dtype='f4')


def _get_cached_sphere(detail: int = 12) -> Tuple[moderngl.Buffer, int]:
    """Get or create cached sphere geometry."""
    global _cached_sphere_vbo, _ctx
    if detail not in _cached_sphere_vbo:
        data = _generate_sphere_geometry(detail)
        vbo = _ctx.buffer(data.tobytes())
        _cached_sphere_vbo[detail] = (vbo, len(data) // 6)
    return _cached_sphere_vbo[detail]


def _get_cached_cube() -> Tuple[moderngl.Buffer, int]:
    """Get or create cached cube geometry."""
    global _cached_cube_vbo, _ctx
    if _cached_cube_vbo is None:
        data = _generate_cube_geometry()
        vbo = _ctx.buffer(data.tobytes())
        _cached_cube_vbo = (vbo, len(data) // 6)
    return _cached_cube_vbo


def _get_cached_cylinder(detail: int = 16) -> Tuple[moderngl.Buffer, int]:
    """Get or create cached cylinder geometry."""
    global _cached_unit_cylinder_vbo, _ctx
    if detail not in _cached_unit_cylinder_vbo:
        data = _generate_cylinder_geometry(detail)
        vbo = _ctx.buffer(data.tobytes())
        _cached_unit_cylinder_vbo[detail] = (vbo, len(data) // 6)
    return _cached_unit_cylinder_vbo[detail]


# ============================================================
# TRANSFORM UTILITIES
# ============================================================

def _make_transform_matrix(center, scale, rotation_matrix=None) -> np.ndarray:
    """Create 4x4 transform matrix."""
    m = np.eye(4, dtype='f4')
    if rotation_matrix is not None:
        m[:3, :3] = rotation_matrix * np.array(scale, dtype='f4')
    else:
        m[0, 0], m[1, 1], m[2, 2] = scale
    m[0, 3], m[1, 3], m[2, 3] = center
    return m


def _cylinder_transform(start, end, radius) -> np.ndarray:
    """Create transform for unit cylinder to go from start to end."""
    start = np.array(start, dtype='f4')
    end = np.array(end, dtype='f4')
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return np.eye(4, dtype='f4')

    axis_norm = axis / length

    # Build rotation matrix to align Z-axis with cylinder axis
    z_axis = np.array([0, 0, 1], dtype='f4')
    if abs(np.dot(axis_norm, z_axis)) > 0.999:
        # Nearly parallel - use simple scaling
        rot = np.eye(3, dtype='f4')
        if axis_norm[2] < 0:
            rot[2, 2] = -1
    else:
        # Rodrigues rotation
        v = np.cross(z_axis, axis_norm)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, axis_norm)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype='f4')
        rot = np.eye(3, dtype='f4') + vx + vx @ vx * ((1 - c) / (s * s + 1e-8))

    # Scale: radius in XY, length in Z
    scale_mat = np.diag([radius, radius, length]).astype('f4')
    rot_scale = rot @ scale_mat

    # Build 4x4 matrix
    m = np.eye(4, dtype='f4')
    m[:3, :3] = rot_scale
    m[0, 3], m[1, 3], m[2, 3] = start
    return m


def _ensure_rgba(color):
    """Ensure color has alpha channel."""
    if len(color) == 3:
        return (*color, 1.0)
    return color


# ============================================================
# PUBLIC API: Draw Functions (Deferred)
# ============================================================

def _init_context(ctx, view, proj):
    """Initialize context (called by engine each frame)."""
    global _ctx, _current_view, _current_proj
    _ctx = ctx
    _current_view = view
    _current_proj = proj


def draw_sphere(center=(0, 0, 0), radius=0.5, color=(0.7, 0.7, 0.7), detail=12):
    """Queue a sphere for batched rendering."""
    _sphere_queue.append((center, radius, _ensure_rgba(color), detail))


def draw_cube(center=(0, 0, 0), size=1.0, color=(0.7, 0.7, 0.7), rotation=(0, 0, 0)):
    """Queue a cube for batched rendering."""
    if isinstance(size, (int, float)):
        size = (size, size, size)
    _cube_queue.append((center, size, _ensure_rgba(color), rotation))


def draw_cylinder(start=(0, 0, 0), end=(0, 0, 1), radius=0.2, color=(0.7, 0.7, 0.7), detail=16):
    """Queue a cylinder for batched rendering."""
    _cylinder_queue.append((start, end, radius, _ensure_rgba(color), detail))


def draw_cone(base=(0, 0, 0), tip=(0, 0, 1), radius=0.3, color=(0.7, 0.7, 0.7), detail=16):
    """Queue a cone for batched rendering."""
    _cone_queue.append((base, tip, radius, _ensure_rgba(color), detail))


def draw_line(start, end, color=(1, 1, 1), width=1.0):
    """Queue a line for batched rendering."""
    _line_queue.append((start, end, _ensure_rgba(color), width))


def draw_arrow(start, end, color=(1, 1, 1), head_size=0.1, head_radius=None, width_radius=0.03):
    """Queue an arrow (cylinder + cone)."""
    start = np.array(start, dtype='f4')
    end = np.array(end, dtype='f4')
    d = end - start
    length = np.linalg.norm(d)
    if length < 0.001:
        return
    d /= length
    hl = min(head_size if head_size > 0.1 else length * 0.2, length)
    hr = head_radius if head_radius else width_radius * 2.5
    split = end - d * hl
    draw_cylinder(tuple(start), tuple(split), radius=width_radius, color=color)
    draw_cone(tuple(split), tuple(end), radius=hr, color=color)


def draw_triangle(v1, v2, v3, color=(0.7, 0.7, 0.7)):
    """Queue a triangle for batched rendering."""
    _triangle_queue.append((v1, v2, v3, _ensure_rgba(color)))


def draw_plane(size=(5, 5), color=(0.5, 0.5, 0.5), center=(0, 0, 0), normal=(0, 0, 1)):
    """Draw a plane as two triangles."""
    w, h = size[0] / 2, size[1] / 2
    n = np.array(normal, dtype='f4')
    n /= np.linalg.norm(n)

    if abs(n[2]) < 0.9:
        right = np.cross(n, [0, 0, 1])
    else:
        right = np.cross(n, [1, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, n)

    c = np.array(center, dtype='f4')
    v1 = c - right * w - up * h
    v2 = c + right * w - up * h
    v3 = c + right * w + up * h
    v4 = c - right * w + up * h

    draw_triangle(tuple(v1), tuple(v2), tuple(v3), color)
    draw_triangle(tuple(v1), tuple(v3), tuple(v4), color)


def draw_point(position, color=(1, 1, 1), size=5.0):
    """Draw a single point (rendered immediately for simplicity)."""
    global _ctx
    if _ctx is None:
        return
    prog = _get_program('batched_lines')
    color = _ensure_rgba(color)
    vertices = np.array([*position, *color], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    _ctx.point_size = size
    _ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    vao.render(moderngl.POINTS)
    vbo.release()
    vao.release()


def draw_face(v1, v2, v3, c1=(1, 0, 0), c2=(0, 1, 0), c3=(0, 0, 1)):
    """Draw a triangle with per-vertex colors (immediate mode)."""
    global _ctx
    if _ctx is None:
        return
    # For now, render immediately - could be batched later
    prog = _get_program('batched_lines')
    c1, c2, c3 = _ensure_rgba(c1), _ensure_rgba(c2), _ensure_rgba(c3)
    vertices = np.array([*v1, *c1, *v2, *c2, *v3, *c3], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    _ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    _ctx.enable(moderngl.CULL_FACE)
    vbo.release()
    vao.release()


def draw_path(points, color=(1, 1, 1), width=1.0):
    """Draw a connected path through points."""
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        draw_line(points[i], points[i + 1], color, width)


# ============================================================
# PARTICLES (Already optimized - single draw call)
# ============================================================

def draw_particles(positions, colors, sizes=1.0):
    """Draw particles (already batched - renders immediately)."""
    global _ctx
    if _ctx is None:
        return
    prog = _get_program('particles')

    if not isinstance(positions, np.ndarray):
        positions = np.array(positions, dtype='f4')
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f4')

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

    vbo_pos.release()
    vbo_col.release()
    vbo_size.release()
    vao.release()


# ============================================================
# BATCH RENDERING (Called by engine at end of frame)
# ============================================================

def _ensure_instance_buffer(needed_bytes: int):
    """Ensure instance buffer is large enough."""
    global _instance_buffer, _instance_buffer_size, _ctx

    if _instance_buffer is None or _instance_buffer_size < needed_bytes:
        if _instance_buffer is not None:
            _instance_buffer.release()
        # Grow by 2x to avoid frequent reallocations
        new_size = max(needed_bytes, _instance_buffer_size * 2, 1024 * 1024)  # Min 1MB
        _instance_buffer = _ctx.buffer(reserve=new_size)
        _instance_buffer_size = new_size


def _ensure_line_buffer(needed_bytes: int):
    """Ensure line buffer is large enough."""
    global _line_buffer, _line_buffer_size, _ctx

    if _line_buffer is None or _line_buffer_size < needed_bytes:
        if _line_buffer is not None:
            _line_buffer.release()
        new_size = max(needed_bytes, _line_buffer_size * 2, 256 * 1024)  # Min 256KB
        _line_buffer = _ctx.buffer(reserve=new_size)
        _line_buffer_size = new_size


def _ensure_triangle_buffer(needed_bytes: int):
    """Ensure triangle buffer is large enough."""
    global _triangle_buffer, _triangle_buffer_size, _ctx

    if _triangle_buffer is None or _triangle_buffer_size < needed_bytes:
        if _triangle_buffer is not None:
            _triangle_buffer.release()
        new_size = max(needed_bytes, _triangle_buffer_size * 2, 512 * 1024)  # Min 512KB
        _triangle_buffer = _ctx.buffer(reserve=new_size)
        _triangle_buffer_size = new_size


def _render_instanced_shapes(queue: List, get_geometry_fn, shape_name: str):
    """Render a batch of shapes using instanced rendering."""
    global _ctx, _current_view, _current_proj

    if not queue:
        return

    prog = _get_program('instanced_solid')

    # Group by detail level (for spheres/cylinders)
    by_detail: Dict[int, List] = {}
    for item in queue:
        detail = item[-1] if len(item) > 3 else 12
        if detail not in by_detail:
            by_detail[detail] = []
        by_detail[detail].append(item)

    for detail, items in by_detail.items():
        geo_vbo, vertex_count = get_geometry_fn(detail)

        # Build instance data: 4x4 matrix (16 floats) + RGBA color (4 floats) = 20 floats
        instance_data = np.zeros((len(items), 20), dtype='f4')

        for i, item in enumerate(items):
            if shape_name == 'sphere':
                center, radius, color, _ = item
                m = _make_transform_matrix(center, (radius, radius, radius))
            elif shape_name == 'cube':
                center, size, color, rotation = item
                m = _make_transform_matrix(center, size)
                # TODO: Add rotation support
            elif shape_name == 'cylinder':
                start, end, radius, color, _ = item
                m = _cylinder_transform(start, end, radius)
            else:
                continue

            instance_data[i, :16] = m.T.flatten()
            instance_data[i, 16:20] = color

        # Upload instance data
        _ensure_instance_buffer(instance_data.nbytes)
        _instance_buffer.write(instance_data.tobytes())

        # Create VAO
        vao = _ctx.vertex_array(prog, [
            (geo_vbo, '3f 3f', 'in_position', 'in_normal'),
            (_instance_buffer, '4f 4f 4f 4f 4f/i', 'inst_row0', 'inst_row1', 'inst_row2', 'inst_row3', 'inst_color'),
        ])

        # Set uniforms
        prog['view'].write(_current_view.T.tobytes())
        prog['projection'].write(_current_proj.T.tobytes())
        prog['light_dir'].value = (0.5, 0.3, 0.8)
        prog['shininess'].value = _material.shininess
        prog['specular_strength'].value = _material.specular_strength
        prog['use_specular'].value = _material.use_specular

        # Extract camera position from view matrix
        inv_view = np.linalg.inv(_current_view)
        cam_pos = inv_view[:3, 3]
        prog['cam_pos'].value = tuple(cam_pos)

        # Render all instances in one call!
        vao.render(moderngl.TRIANGLES, instances=len(items))
        vao.release()


def _render_lines():
    """Render all queued lines in a single draw call."""
    global _ctx, _line_queue

    if not _line_queue:
        return

    prog = _get_program('batched_lines')

    # Build vertex data: 2 vertices per line, 7 floats each (pos3 + color4)
    vertex_data = np.zeros((len(_line_queue) * 2, 7), dtype='f4')

    for i, (start, end, color, width) in enumerate(_line_queue):
        vertex_data[i * 2, :3] = start
        vertex_data[i * 2, 3:7] = color
        vertex_data[i * 2 + 1, :3] = end
        vertex_data[i * 2 + 1, 3:7] = color

    _ensure_line_buffer(vertex_data.nbytes)
    _line_buffer.write(vertex_data.tobytes())

    vao = _ctx.vertex_array(prog, [
        (_line_buffer, '3f 4f', 'in_position', 'in_color')
    ])

    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())

    # Use average width (could be improved)
    avg_width = sum(w for _, _, _, w in _line_queue) / len(_line_queue)
    _ctx.line_width = avg_width

    vao.render(moderngl.LINES, vertices=len(_line_queue) * 2)
    vao.release()


def _render_triangles():
    """Render all queued triangles in a single draw call."""
    global _ctx, _triangle_queue

    if not _triangle_queue:
        return

    prog = _get_program('batched_triangles')

    # Build vertex data: 3 vertices per triangle, 7 floats each (pos3 + color4)
    vertex_data = np.zeros((len(_triangle_queue) * 3, 10), dtype='f4')  # pos3 + normal3 + color4

    for i, (v1, v2, v3, color) in enumerate(_triangle_queue):
        v1, v2, v3 = np.array(v1, dtype='f4'), np.array(v2, dtype='f4'), np.array(v3, dtype='f4')

        # Calculate normal
        edge1 = v2 - v1
        edge2 = v3 - v1
        norm = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(norm)
        if norm_len > 1e-6:
            norm /= norm_len
        else:
            norm = np.array([0, 0, 1], dtype='f4')

        # Vertex 1
        vertex_data[i * 3, :3] = v1
        vertex_data[i * 3, 3:6] = norm
        vertex_data[i * 3, 6:10] = color

        # Vertex 2
        vertex_data[i * 3 + 1, :3] = v2
        vertex_data[i * 3 + 1, 3:6] = norm
        vertex_data[i * 3 + 1, 6:10] = color

        # Vertex 3
        vertex_data[i * 3 + 2, :3] = v3
        vertex_data[i * 3 + 2, 3:6] = norm
        vertex_data[i * 3 + 2, 6:10] = color

    # Use persistent buffer
    _ensure_triangle_buffer(vertex_data.nbytes)
    _triangle_buffer.write(vertex_data.tobytes())

    vao = _ctx.vertex_array(prog, [
        (_triangle_buffer, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')
    ])

    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    prog['light_dir'].value = (0.5, 0.3, 0.8)
    prog['shininess'].value = _material.shininess
    prog['specular_strength'].value = _material.specular_strength
    prog['use_specular'].value = _material.use_specular

    # Extract camera position
    inv_view = np.linalg.inv(_current_view)
    cam_pos = inv_view[:3, 3]
    prog['cam_pos'].value = tuple(cam_pos)

    _ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES, vertices=len(_triangle_queue) * 3)
    _ctx.enable(moderngl.CULL_FACE)
    vao.release()


# Keep the old function as fallback but it won't be used
def _render_single_triangle(v1, v2, v3, color):
    """Render a single triangle (fallback - not used with batching)."""
    global _ctx

    prog = _get_program('batched_lines')
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
    vertices = np.array([*v1, *color, *v2, *color, *v3, *color], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())
    _ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    _ctx.enable(moderngl.CULL_FACE)
    vbo.release()
    vao.release()


def flush_all():
    """
    Flush all queued primitives to GPU.
    Called automatically by the engine at end of each frame.
    """
    global _sphere_queue, _cube_queue, _cylinder_queue, _cone_queue, _line_queue, _triangle_queue

    # Render each type
    _render_instanced_shapes(_sphere_queue, _get_cached_sphere, 'sphere')
    _render_instanced_shapes(_cube_queue, lambda d: _get_cached_cube(), 'cube')
    _render_instanced_shapes(_cylinder_queue, _get_cached_cylinder, 'cylinder')
    # Cones use cylinder geometry with different transform (TODO: proper cone)
    _render_instanced_shapes(_cone_queue, _get_cached_cylinder, 'cylinder')
    _render_lines()
    _render_triangles()

    # Clear queues for next frame
    _sphere_queue.clear()
    _cube_queue.clear()
    _cylinder_queue.clear()
    _cone_queue.clear()
    _line_queue.clear()
    _triangle_queue.clear()


def clear_cache():
    """Clear all cached geometry (call on context recreation)."""
    global _cached_sphere_vbo, _cached_cube_vbo, _cached_unit_cylinder_vbo
    global _instance_buffer, _line_buffer

    for vbo, _ in _cached_sphere_vbo.values():
        try:
            vbo.release()
        except:
            pass
    _cached_sphere_vbo.clear()

    if _cached_cube_vbo is not None:
        try:
            _cached_cube_vbo[0].release()
        except:
            pass
        _cached_cube_vbo = None

    for vbo, _ in _cached_unit_cylinder_vbo.values():
        try:
            vbo.release()
        except:
            pass
    _cached_unit_cylinder_vbo.clear()

    if _instance_buffer is not None:
        try:
            _instance_buffer.release()
        except:
            pass
        _instance_buffer = None

    if _line_buffer is not None:
        try:
            _line_buffer.release()
        except:
            pass
        _line_buffer = None

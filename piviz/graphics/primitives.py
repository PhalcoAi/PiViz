# piviz/graphics/primitives.py
"""
PiViz High-Performance Primitives (v1.0.1)
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
from typing import Tuple, Optional, Dict, List, Union
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
# Triangle queue stores: (v1, v2, v3, c1, c2, c3)
# If c2/c3 are None, c1 is used for all vertices
_triangle_queue: List[Tuple] = []

# Cached geometry (created once, reused forever)
_cached_sphere_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}
_cached_cube_vbo: Optional[Tuple[moderngl.Buffer, int]] = None
_cached_unit_cylinder_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}
_cached_cone_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}

# Persistent instance buffers (grow as needed, never shrink)
_instance_buffer: Optional[moderngl.Buffer] = None
_instance_buffer_size: int = 0
_line_buffer: Optional[moderngl.Buffer] = None
_line_buffer_size: int = 0
_triangle_buffer: Optional[moderngl.Buffer] = None
_triangle_buffer_size: int = 0

# VAO Cache to avoid per-frame object creation overhead
# Keys: (type_id, detail_level) or just type_id
_cached_vaos: Dict[str, moderngl.VertexArray] = {}

# Bounds tracking
_current_min = np.full(3, np.inf, dtype='f4')
_current_max = np.full(3, -np.inf, dtype='f4')
_last_frame_bounds = ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))


def get_scene_bounds():
    """Get the bounding box of the last rendered frame."""
    return _last_frame_bounds


def _update_bounds(p_min, p_max):
    """Update current frame bounds."""
    global _current_min, _current_max
    _current_min = np.minimum(_current_min, p_min)
    _current_max = np.maximum(_current_max, p_max)


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
        
        elif name == 'instanced_directional':
            # Optimized shader for cylinders/cones defined by start/end points
            # Calculates transform on GPU to save CPU time
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform vec3 light_dir;

                    in vec3 in_position;
                    in vec3 in_normal;

                    // Per-instance attributes
                    in vec3 inst_start;
                    in vec3 inst_end;
                    in float inst_radius;
                    in vec4 inst_color;

                    out vec3 v_normal;
                    out vec3 v_position;
                    out vec4 v_color;
                    out vec3 v_light_dir;

                    mat3 make_rotation(vec3 direction) {
                        vec3 z = normalize(direction);
                        // Handle case where direction is parallel to up vector
                        vec3 ref = abs(z.z) > 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
                        vec3 x = normalize(cross(ref, z));
                        vec3 y = cross(z, x);
                        return mat3(x, y, z);
                    }

                    void main() {
                        vec3 axis = inst_end - inst_start;
                        float len = length(axis);
                        
                        // Avoid degenerate geometry
                        if (len < 0.0001) {
                            gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                            return;
                        }

                        mat3 rot = make_rotation(axis);
                        
                        // Scale: XY by radius, Z by length
                        vec3 scaled_pos = in_position;
                        scaled_pos.xy *= inst_radius;
                        scaled_pos.z *= len;
                        
                        // Rotate and Translate
                        vec3 world_pos = (rot * scaled_pos) + inst_start;
                        
                        v_position = world_pos;
                        
                        // Normal transformation
                        // Scale normal by inverse scale factors to maintain orthogonality
                        vec3 scaled_normal = in_normal;
                        scaled_normal.xy /= inst_radius;
                        scaled_normal.z /= len;
                        v_normal = rot * normalize(scaled_normal);
                        
                        v_color = inst_color;
                        v_light_dir = light_dir;
                        gl_Position = projection * view * vec4(world_pos, 1.0);
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


def _generate_cone_geometry(detail: int = 16) -> np.ndarray:
    """Generate unit cone (base at z=0, tip at z=1, base radius=1)."""
    vertices = []

    # Calculate the slope angle for proper normals
    # For a cone with height 1 and radius 1, the slope is 45 degrees
    slope = 1.0 / math.sqrt(2.0)  # cos(45°) = sin(45°)

    # Side faces - triangles from base edge to tip
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)

        # Normal for side face - average of the two edge normals
        # For a cone, the normal points outward and upward
        mid_angle = (a0 + a1) / 2
        nx = math.cos(mid_angle) * slope
        ny = math.sin(mid_angle) * slope
        nz = slope

        # Triangle: base vertex 0, base vertex 1, tip
        # Vertex at base edge 0
        vertices.extend([c0, s0, 0, c0 * slope, s0 * slope, slope])
        # Vertex at base edge 1
        vertices.extend([c1, s1, 0, c1 * slope, s1 * slope, slope])
        # Tip vertex (use averaged normal for smooth shading)
        vertices.extend([0, 0, 1, nx, ny, nz])

    # Bottom cap
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        # Center vertex
        vertices.extend([0, 0, 0, 0, 0, -1])
        # Edge vertices (wound clockwise when viewed from below)
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


def _get_cached_cone(detail: int = 16) -> Tuple[moderngl.Buffer, int]:
    """Get or create cached cone geometry."""
    global _cached_cone_vbo, _ctx
    if detail not in _cached_cone_vbo:
        data = _generate_cone_geometry(detail)
        vbo = _ctx.buffer(data.tobytes())
        _cached_cone_vbo[detail] = (vbo, len(data) // 6)
    return _cached_cone_vbo[detail]


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


def _make_rotation_matrix_xyz(rotation) -> np.ndarray:
    """Create 3x3 rotation matrix from Euler angles (x, y, z) in radians."""
    rx, ry, rz = rotation

    # Rotation around X axis
    cx, sx = math.cos(rx), math.sin(rx)
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype='f4')

    # Rotation around Y axis
    cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype='f4')

    # Rotation around Z axis
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype='f4')

    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


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


def _cone_transform(base, tip, radius) -> np.ndarray:
    """Create transform for unit cone to go from base to tip."""
    base = np.array(base, dtype='f4')
    tip = np.array(tip, dtype='f4')
    axis = tip - base
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return np.eye(4, dtype='f4')

    axis_norm = axis / length

    # Build rotation matrix to align Z-axis with cone axis
    z_axis = np.array([0, 0, 1], dtype='f4')
    if abs(np.dot(axis_norm, z_axis)) > 0.999:
        # Nearly parallel - use simple scaling
        rot = np.eye(3, dtype='f4')
        if axis_norm[2] < 0:
            # If pointing down (-Z), we need to flip the Z axis
            # But just setting rot[2,2] = -1 flips Z, which turns a right-handed system into left-handed
            # Better to rotate 180 degrees around X or Y
            rot[0, 0] = 1
            rot[1, 1] = -1
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
    m[0, 3], m[1, 3], m[2, 3] = base
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
    
    c = np.array(center, dtype='f4')
    _update_bounds(c - radius, c + radius)


def draw_cube(center=(0, 0, 0), size=1.0, color=(0.7, 0.7, 0.7), rotation=(0, 0, 0)):
    """Queue a cube for batched rendering."""
    if isinstance(size, (int, float)):
        size = (size, size, size)
    _cube_queue.append((center, size, _ensure_rgba(color), rotation))
    
    c = np.array(center, dtype='f4')
    s = np.array(size, dtype='f4')
    if any(r != 0 for r in rotation):
        # Bounding sphere approximation for rotated cube
        r = np.linalg.norm(s) * 0.5
        _update_bounds(c - r, c + r)
    else:
        half = s * 0.5
        _update_bounds(c - half, c + half)


def draw_cylinder(start=(0, 0, 0), end=(0, 0, 1), radius=0.2, color=(0.7, 0.7, 0.7), detail=16):
    """Queue a cylinder for batched rendering."""
    _cylinder_queue.append((start, end, radius, _ensure_rgba(color), detail))
    
    p1 = np.array(start, dtype='f4')
    p2 = np.array(end, dtype='f4')
    b_min = np.minimum(p1, p2) - radius
    b_max = np.maximum(p1, p2) + radius
    _update_bounds(b_min, b_max)


def draw_cone(base=(0, 0, 0), tip=(0, 0, 1), radius=0.3, color=(0.7, 0.7, 0.7), detail=16):
    """Queue a cone for batched rendering."""
    _cone_queue.append((base, tip, radius, _ensure_rgba(color), detail))
    
    p1 = np.array(base, dtype='f4')
    p2 = np.array(tip, dtype='f4')
    b_min = np.minimum(p1, p2) - radius
    b_max = np.maximum(p1, p2) + radius
    _update_bounds(b_min, b_max)


def draw_line(start, end, color=(1, 1, 1), width=1.0):
    """Queue a line for batched rendering."""
    _line_queue.append((start, end, _ensure_rgba(color), width))
    
    p1 = np.array(start, dtype='f4')
    p2 = np.array(end, dtype='f4')
    _update_bounds(np.minimum(p1, p2), np.maximum(p1, p2))


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
    # Store as (v1, v2, v3, c1, None, None) to indicate uniform color
    _triangle_queue.append((v1, v2, v3, _ensure_rgba(color), None, None))
    
    pts = np.array([v1, v2, v3], dtype='f4')
    _update_bounds(np.min(pts, axis=0), np.max(pts, axis=0))


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
    
    # Update bounds
    p = np.array(position, dtype='f4')
    _update_bounds(p, p)
    
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
    """Draw a triangle with per-vertex colors (batched)."""
    # Store as (v1, v2, v3, c1, c2, c3)
    _triangle_queue.append((v1, v2, v3, _ensure_rgba(c1), _ensure_rgba(c2), _ensure_rgba(c3)))
    
    # Update bounds
    pts = np.array([v1, v2, v3], dtype='f4')
    _update_bounds(np.min(pts, axis=0), np.max(pts, axis=0))


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
    
    # Update bounds
    if len(positions) > 0:
        if not isinstance(positions, np.ndarray):
             pos = np.array(positions, dtype='f4')
        else:
             pos = positions
        _update_bounds(np.min(pos, axis=0), np.max(pos, axis=0))
        
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
        # Start with 4MB to reduce early churn
        new_size = max(needed_bytes, _instance_buffer_size * 2, 4 * 1024 * 1024)
        _instance_buffer = _ctx.buffer(reserve=new_size)
        _instance_buffer_size = new_size
        
        # Invalidate VAO cache because buffer handle changed
        _clear_vao_cache_for_buffer('instance')


def _ensure_line_buffer(needed_bytes: int):
    """Ensure line buffer is large enough."""
    global _line_buffer, _line_buffer_size, _ctx
    if _line_buffer is None or _line_buffer_size < needed_bytes:
        if _line_buffer is not None:
            _line_buffer.release()
        new_size = max(needed_bytes, _line_buffer_size * 2, 1024 * 1024)  # Min 1MB
        _line_buffer = _ctx.buffer(reserve=new_size)
        _line_buffer_size = new_size
        
        # Invalidate VAO cache
        _clear_vao_cache_for_buffer('line')


def _ensure_triangle_buffer(needed_bytes: int):
    """Ensure triangle buffer is large enough."""
    global _triangle_buffer, _triangle_buffer_size, _ctx
    if _triangle_buffer is None or _triangle_buffer_size < needed_bytes:
        if _triangle_buffer is not None:
            _triangle_buffer.release()
        new_size = max(needed_bytes, _triangle_buffer_size * 2, 2 * 1024 * 1024)  # Min 2MB
        _triangle_buffer = _ctx.buffer(reserve=new_size)
        _triangle_buffer_size = new_size
        
        # Invalidate VAO cache
        _clear_vao_cache_for_buffer('triangle')


def _clear_vao_cache_for_buffer(buffer_type: str):
    """Clear cached VAOs associated with a specific buffer type."""
    global _cached_vaos
    keys_to_remove = []
    
    for key in _cached_vaos:
        # Key structure: (type_id, detail) or (type_id,)
        # type_id is 'sphere', 'cube', 'cylinder', 'cone', 'lines', 'triangles'
        type_id = key[0]
        
        if buffer_type == 'instance':
            if type_id in ('sphere', 'cube', 'cylinder', 'cone'):
                keys_to_remove.append(key)
        elif buffer_type == 'line':
            if type_id == 'lines':
                keys_to_remove.append(key)
        elif buffer_type == 'triangle':
            if type_id == 'triangles':
                keys_to_remove.append(key)
                
    for key in keys_to_remove:
        if key in _cached_vaos:
            _cached_vaos[key].release()
            del _cached_vaos[key]


def _get_cached_vao(key, create_fn):
    """Get or create a cached VAO."""
    global _cached_vaos
    if key not in _cached_vaos:
        _cached_vaos[key] = create_fn()
    return _cached_vaos[key]


def _render_instanced_shapes(queue: List, get_geometry_fn, shape_name: str):
    """Render a batch of shapes using instanced rendering."""
    global _ctx, _current_view, _current_proj
    if not queue:
        return

    # Use optimized shader for cylinders and cones
    use_directional_shader = shape_name in ('cylinder', 'cone')
    prog_name = 'instanced_directional' if use_directional_shader else 'instanced_solid'
    prog = _get_program(prog_name)

    # Group by detail level (for spheres/cylinders/cones)
    by_detail: Dict[int, List] = {}
    for item in queue:
        detail = item[-1] if len(item) > 3 else 12
        if detail not in by_detail:
            by_detail[detail] = []
        by_detail[detail].append(item)

    for detail, items in by_detail.items():
        geo_vbo, vertex_count = get_geometry_fn(detail)
        count = len(items)

        if use_directional_shader:
            # Optimized path for cylinders/cones: 11 floats per instance
            # start(3), end(3), radius(1), color(4)
            instance_data = np.zeros((count, 11), dtype='f4')
            
            # Fast vectorized copy if possible, or simple loop
            # Since items is a list of tuples, we iterate
            for i, (start, end, radius, color, _) in enumerate(items):
                instance_data[i, 0:3] = start
                instance_data[i, 3:6] = end
                instance_data[i, 6] = radius
                instance_data[i, 7:11] = color
                
            # Upload instance data
            _ensure_instance_buffer(instance_data.nbytes)
            _instance_buffer.write(instance_data.tobytes())
            
            # Get or create cached VAO
            vao_key = (shape_name, detail)
            
            def create_vao():
                return _ctx.vertex_array(prog, [
                    (geo_vbo, '3f 3f', 'in_position', 'in_normal'),
                    (_instance_buffer, '3f 3f 1f 4f/i', 'inst_start', 'inst_end', 'inst_radius', 'inst_color'),
                ])
                
            vao = _get_cached_vao(vao_key, create_vao)
            
        else:
            # Standard path for spheres/cubes: 20 floats per instance
            # matrix(16), color(4)
            instance_data = np.zeros((count, 20), dtype='f4')

            for i, item in enumerate(items):
                if shape_name == 'sphere':
                    center, radius, color, _ = item
                    m = _make_transform_matrix(center, (radius, radius, radius))
                elif shape_name == 'cube':
                    center, size, color, rotation = item
                    if any(r != 0 for r in rotation):
                        rot_matrix = _make_rotation_matrix_xyz(rotation)
                        scale_mat = np.diag(size).astype('f4')
                        rot_scale = rot_matrix @ scale_mat
                        m = np.eye(4, dtype='f4')
                        m[:3, :3] = rot_scale
                        m[0, 3], m[1, 3], m[2, 3] = center
                    else:
                        m = _make_transform_matrix(center, size)
                else:
                    continue

                instance_data[i, :16] = m.T.flatten()
                instance_data[i, 16:20] = color

            # Upload instance data
            _ensure_instance_buffer(instance_data.nbytes)
            _instance_buffer.write(instance_data.tobytes())

            # Get or create cached VAO
            vao_key = (shape_name, detail)
            
            def create_vao():
                return _ctx.vertex_array(prog, [
                    (geo_vbo, '3f 3f', 'in_position', 'in_normal'),
                    (_instance_buffer, '4f 4f 4f 4f 4f/i', 'inst_row0', 'inst_row1', 'inst_row2', 'inst_row3', 'inst_color'),
                ])
                
            vao = _get_cached_vao(vao_key, create_vao)

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

    # Get or create cached VAO
    vao_key = ('lines',)
    
    def create_vao():
        return _ctx.vertex_array(prog, [
            (_line_buffer, '3f 4f', 'in_position', 'in_color')
        ])
        
    vao = _get_cached_vao(vao_key, create_vao)

    prog['view'].write(_current_view.T.tobytes())
    prog['projection'].write(_current_proj.T.tobytes())

    # Use average width (could be improved)
    avg_width = sum(w for _, _, _, w in _line_queue) / len(_line_queue)
    _ctx.line_width = avg_width

    vao.render(moderngl.LINES, vertices=len(_line_queue) * 2)


def _render_triangles():
    """Render all queued triangles in a single draw call."""
    global _ctx, _triangle_queue
    if not _triangle_queue:
        return

    prog = _get_program('batched_triangles')

    # Build vertex data: 3 vertices per triangle, 7 floats each (pos3 + normal3 + color4)
    vertex_data = np.zeros((len(_triangle_queue) * 3, 10), dtype='f4')  # pos3 + normal3 + color4

    for i, (v1, v2, v3, c1, c2, c3) in enumerate(_triangle_queue):
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

        # If c2/c3 are None, use c1 for all vertices (uniform color)
        if c2 is None:
            c2 = c1
            c3 = c1

        # Vertex 1
        vertex_data[i * 3, :3] = v1
        vertex_data[i * 3, 3:6] = norm
        vertex_data[i * 3, 6:10] = c1

        # Vertex 2
        vertex_data[i * 3 + 1, :3] = v2
        vertex_data[i * 3 + 1, 3:6] = norm
        vertex_data[i * 3 + 1, 6:10] = c2

        # Vertex 3
        vertex_data[i * 3 + 2, :3] = v3
        vertex_data[i * 3 + 2, 3:6] = norm
        vertex_data[i * 3 + 2, 6:10] = c3

    # Use persistent buffer
    _ensure_triangle_buffer(vertex_data.nbytes)
    _triangle_buffer.write(vertex_data.tobytes())

    # Get or create cached VAO
    vao_key = ('triangles',)
    
    def create_vao():
        return _ctx.vertex_array(prog, [
            (_triangle_buffer, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')
        ])
        
    vao = _get_cached_vao(vao_key, create_vao)

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
    # Cones use proper cone geometry with dedicated transform
    _render_instanced_shapes(_cone_queue, _get_cached_cone, 'cone')
    _render_lines()
    _render_triangles()

    # Update global last frame bounds
    global _last_frame_bounds, _current_min, _current_max
    if not np.all(np.isinf(_current_min)):
        _last_frame_bounds = (tuple(_current_min), tuple(_current_max))
    
    # Reset for next frame
    _current_min = np.full(3, np.inf, dtype='f4')
    _current_max = np.full(3, -np.inf, dtype='f4')

    # Clear queues for next frame
    _sphere_queue.clear()
    _cube_queue.clear()
    _cylinder_queue.clear()
    _cone_queue.clear()
    _line_queue.clear()
    _triangle_queue.clear()


def clear_cache():
    """Clear all cached geometry (call on context recreation)."""
    global _cached_sphere_vbo, _cached_cube_vbo, _cached_unit_cylinder_vbo, _cached_cone_vbo
    global _instance_buffer, _line_buffer, _cached_vaos
    
    # Clear VAO cache
    for vao in _cached_vaos.values():
        try:
            vao.release()
        except:
            pass
    _cached_vaos.clear()

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

    for vbo, _ in _cached_cone_vbo.values():
        try:
            vbo.release()
        except:
            pass
    _cached_cone_vbo.clear()

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

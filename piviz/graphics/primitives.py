"""
PiViz High-Performance Primitives (v1.2.0)
========================================

ARCHITECTURE:
- All draw_*() calls write directly into pre-allocated numpy arrays
- Actual GPU rendering happens in flush_all() called by the engine
- Geometry is cached and reused across frames
- Buffers are persistent and grow as needed (no per-frame allocations)

OPTIMIZATIONS (v1.2.0 over v1.1.0):
- Struct-of-arrays: draw calls write into numpy arrays, flush is zero-loop
- Dedicated instanced_sphere shader: 7 floats/instance vs 20 (center+radius+color)
- Directional shader for cylinders/cones: 11 floats/instance, GPU-side transform
- Vectorized triangle normal computation
- Cached inverse view matrix + uniform locations
- Fixed two-sided triangle lighting (gl_FrontFacing normal flip)

Author: Yogesh Phalak
"""

import os
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

# Cached per-frame derived matrices
_cached_view_bytes: Optional[bytes] = None
_cached_proj_bytes: Optional[bytes] = None
_cached_cam_pos: Optional[tuple] = None

# Cached uniform locations: program_name -> {uniform_name: UniformLocation}
_uniform_cache: Dict[str, Dict] = {}

# ============================================================
# STRUCT-OF-ARRAYS QUEUES
# ============================================================
# Pre-allocated numpy arrays that grow as needed.
# draw_*() writes directly into these — no tuple/list overhead.
# flush reads them with zero Python loops.

_INITIAL_CAPACITY = 4096

# -- Spheres: 7 floats per instance (center3 + radius1 + color3_alpha1 = 3+1+4 = 8? no: center3, radius, rgba4 = 8)
# Actually: we group by detail, so we need detail too. But 95%+ of spheres share
# the same detail. Strategy: store detail separately, fast-path single-detail case.
# Layout: [cx, cy, cz, radius, r, g, b, a] = 8 floats
_sphere_data: np.ndarray = np.empty((_INITIAL_CAPACITY, 8), dtype='f4')
_sphere_details: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype='i4')
_sphere_count: int = 0

# -- Cubes: need center(3) + size(3) + color(4) + rotation(3) = 13 floats
# Rotation requires slow path, but we track a flag to skip it when possible
_cube_data: np.ndarray = np.empty((_INITIAL_CAPACITY, 13), dtype='f4')
_cube_has_rotation: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype='bool')
_cube_count: int = 0

# -- Cylinders: 11 floats (start3 + end3 + radius + color4)
_cylinder_data: np.ndarray = np.empty((_INITIAL_CAPACITY, 11), dtype='f4')
_cylinder_details: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype='i4')
_cylinder_count: int = 0

# -- Cones: 11 floats (base3 + tip3 + radius + color4)
_cone_data: np.ndarray = np.empty((_INITIAL_CAPACITY, 11), dtype='f4')
_cone_details: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype='i4')
_cone_count: int = 0

# -- Lines: 11 floats (start3 + end3 + color4 + width1)
_line_data: np.ndarray = np.empty((_INITIAL_CAPACITY, 11), dtype='f4')
_line_count: int = 0

# -- Triangles: 22 floats (v1_3 + v2_3 + v3_3 + c1_4 + c2_4 + c3_4 + has_per_vertex_color)
# Actually store vertices and colors separately for vectorized normal computation
_tri_v1: np.ndarray = np.empty((_INITIAL_CAPACITY, 3), dtype='f4')
_tri_v2: np.ndarray = np.empty((_INITIAL_CAPACITY, 3), dtype='f4')
_tri_v3: np.ndarray = np.empty((_INITIAL_CAPACITY, 3), dtype='f4')
_tri_c1: np.ndarray = np.empty((_INITIAL_CAPACITY, 4), dtype='f4')
_tri_c2: np.ndarray = np.empty((_INITIAL_CAPACITY, 4), dtype='f4')
_tri_c3: np.ndarray = np.empty((_INITIAL_CAPACITY, 4), dtype='f4')
_tri_count: int = 0

# -- Meshes: 13 floats per instance (pos3 + scale3 + rot3 + color4)
# Grouped by mesh path — each path maps to its own instance array
_mesh_instances: Dict[str, np.ndarray] = {}  # mesh_key -> (N, 13) f4
_mesh_counts: Dict[str, int] = {}            # mesh_key -> count


def _ensure_capacity(current_arr, needed, cols=None):
    """Double array capacity if needed. Returns (possibly new) array."""
    if needed <= len(current_arr):
        return current_arr
    new_cap = max(needed, len(current_arr) * 2)
    if current_arr.ndim == 1:
        new_arr = np.empty(new_cap, dtype=current_arr.dtype)
    else:
        new_arr = np.empty((new_cap, current_arr.shape[1]), dtype=current_arr.dtype)
    new_arr[:len(current_arr)] = current_arr
    return new_arr


# Cached geometry (created once, reused forever)
_cached_sphere_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}
_cached_cube_vbo: Optional[Tuple[moderngl.Buffer, int]] = None
_cached_unit_cylinder_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}
_cached_cone_vbo: Dict[int, Tuple[moderngl.Buffer, int]] = {}
_cached_mesh_vbo: Dict[str, List[Tuple[moderngl.Buffer, int, Optional[moderngl.Texture]]]] = {}

# tex_path → moderngl.Texture | None (None = load failed)
_cached_textures: Dict[str, Optional[moderngl.Texture]] = {}

# Persistent GPU buffers (grow as needed, never shrink)
_instance_buffer: Optional[moderngl.Buffer] = None
_instance_buffer_size: int = 0
_line_buffer: Optional[moderngl.Buffer] = None
_line_buffer_size: int = 0
_triangle_buffer: Optional[moderngl.Buffer] = None
_triangle_buffer_size: int = 0

# VAO Cache
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
# UNIFORM CACHING
# ============================================================

def _get_uniform(prog, prog_name, uniform_name):
    """Get a cached uniform location."""
    global _uniform_cache
    if prog_name not in _uniform_cache:
        _uniform_cache[prog_name] = {}
    cache = _uniform_cache[prog_name]
    if uniform_name not in cache:
        cache[uniform_name] = prog[uniform_name]
    return cache[uniform_name]


def _set_common_uniforms(prog, prog_name):
    """Set view/projection uniforms using cached values."""
    _get_uniform(prog, prog_name, 'view').write(_cached_view_bytes)
    _get_uniform(prog, prog_name, 'projection').write(_cached_proj_bytes)


def _set_lighting_uniforms(prog, prog_name):
    """Set lighting-related uniforms."""
    _get_uniform(prog, prog_name, 'light_dir').value = (0.5, 0.3, 0.8)
    _get_uniform(prog, prog_name, 'shininess').value = _material.shininess
    _get_uniform(prog, prog_name, 'specular_strength').value = _material.specular_strength
    _get_uniform(prog, prog_name, 'use_specular').value = _material.use_specular
    _get_uniform(prog, prog_name, 'cam_pos').value = _cached_cam_pos


# ============================================================
# SHADER PROGRAMS
# ============================================================

def _get_program(name: str) -> moderngl.Program:
    """Get or create shader program."""
    global _ctx, _programs

    if name not in _programs:
        if name == 'instanced_solid':
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform vec3 light_dir;

                    in vec3 in_position;
                    in vec3 in_normal;

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

        elif name == 'instanced_sphere':
            # Optimized shader for spheres: only needs center, radius, color
            # Builds the model matrix on GPU from just 8 floats instead of 20
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform vec3 light_dir;

                    in vec3 in_position;
                    in vec3 in_normal;

                    in vec3 inst_center;
                    in float inst_radius;
                    in vec4 inst_color;

                    out vec3 v_normal;
                    out vec3 v_position;
                    out vec4 v_color;
                    out vec3 v_light_dir;

                    void main() {
                        // Sphere: uniform scale, no rotation needed
                        vec3 world_pos = in_position * inst_radius + inst_center;
                        v_position = world_pos;
                        v_normal = in_normal;  // Unit sphere normals are already correct
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

        elif name == 'instanced_directional':
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform vec3 light_dir;

                    in vec3 in_position;
                    in vec3 in_normal;

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
                        vec3 ref = abs(z.z) > 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
                        vec3 x = normalize(cross(ref, z));
                        vec3 y = cross(z, x);
                        return mat3(x, y, z);
                    }

                    void main() {
                        vec3 axis = inst_end - inst_start;
                        float len = length(axis);

                        if (len < 0.0001) {
                            gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                            return;
                        }

                        mat3 rot = make_rotation(axis);

                        vec3 scaled_pos = in_position;
                        scaled_pos.xy *= inst_radius;
                        scaled_pos.z *= len;

                        vec3 world_pos = (rot * scaled_pos) + inst_start;

                        v_position = world_pos;

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

                        // Fix patchy lighting: flip normal to face camera
                        if (!gl_FrontFacing) norm = -norm;

                        vec3 light = normalize(light_dir);
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

        elif name == 'instanced_mesh':
            # Mesh instancing: 13 floats/instance (pos3 + scale3 + rot3 + color4)
            # Supports optional GPU texture sampling via sampler2D + has_texture uniform
            _programs[name] = _ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform vec3 light_dir;

                    in vec3 in_position;
                    in vec3 in_normal;
                    in vec4 in_color;   // vertex color (white when textured)
                    in vec2 in_uv;      // texture UV coords

                    in vec3 inst_pos;
                    in vec3 inst_scale;
                    in vec3 inst_rot;
                    in vec4 inst_color;  // instance color tint

                    out vec3 v_normal;
                    out vec3 v_position;
                    out vec4 v_color;
                    out vec3 v_light_dir;
                    out vec2 v_uv;

                    void main() {
                        float cx = cos(inst_rot.x), sx = sin(inst_rot.x);
                        float cy = cos(inst_rot.y), sy = sin(inst_rot.y);
                        float cz = cos(inst_rot.z), sz = sin(inst_rot.z);

                        mat3 rot = mat3(
                            cy*cz,              cy*sz,              -sy,
                            sx*sy*cz - cx*sz,   sx*sy*sz + cx*cz,   sx*cy,
                            cx*sy*cz + sx*sz,   cx*sy*sz - sx*cz,   cx*cy
                        );

                        vec3 scaled = in_position * inst_scale;
                        vec3 world_pos = rot * scaled + inst_pos;
                        v_position = world_pos;
                        v_normal = normalize(rot * (in_normal / inst_scale));

                        v_color = in_color * inst_color;
                        v_light_dir = light_dir;
                        v_uv = in_uv;
                        gl_Position = projection * view * vec4(world_pos, 1.0);
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform float shininess;
                    uniform float specular_strength;
                    uniform bool use_specular;
                    uniform vec3 cam_pos;
                    uniform sampler2D tex0;
                    uniform bool has_texture;

                    in vec3 v_normal;
                    in vec3 v_position;
                    in vec4 v_color;
                    in vec3 v_light_dir;
                    in vec2 v_uv;

                    out vec4 frag_color;

                    void main() {
                        vec4 base = has_texture
                            ? texture(tex0, v_uv) * v_color
                            : v_color;

                        vec3 norm = normalize(v_normal);
                        if (!gl_FrontFacing) norm = -norm;

                        vec3 light = normalize(v_light_dir);
                        float diff = max(dot(norm, light), 0.0);

                        float spec = 0.0;
                        if (use_specular && diff > 0.0) {
                            vec3 view_dir = normalize(cam_pos - v_position);
                            vec3 halfway = normalize(light + view_dir);
                            spec = pow(max(dot(norm, halfway), 0.0), shininess);
                        }

                        float ambient = 0.3;
                        vec3 result = base.rgb * (ambient + diff * 0.7);
                        if (use_specular) result += vec3(1.0) * spec * specular_strength;
                        frag_color = vec4(result, base.a);
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
                return [x, y, z, x, y, z]

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
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        vertices.extend([c0, s0, 0, c0, s0, 0])
        vertices.extend([c1, s1, 0, c1, s1, 0])
        vertices.extend([c1, s1, 1, c1, s1, 0])
        vertices.extend([c0, s0, 0, c0, s0, 0])
        vertices.extend([c1, s1, 1, c1, s1, 0])
        vertices.extend([c0, s0, 1, c0, s0, 0])
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        vertices.extend([0, 0, 1, 0, 0, 1])
        vertices.extend([math.cos(a0), math.sin(a0), 1, 0, 0, 1])
        vertices.extend([math.cos(a1), math.sin(a1), 1, 0, 0, 1])
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
    slope = 1.0 / math.sqrt(2.0)
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        mid_angle = (a0 + a1) / 2
        nx = math.cos(mid_angle) * slope
        ny = math.sin(mid_angle) * slope
        nz = slope
        vertices.extend([c0, s0, 0, c0 * slope, s0 * slope, slope])
        vertices.extend([c1, s1, 0, c1 * slope, s1 * slope, slope])
        vertices.extend([0, 0, 1, nx, ny, nz])
    for i in range(detail):
        a0 = 2 * math.pi * i / detail
        a1 = 2 * math.pi * (i + 1) / detail
        vertices.extend([0, 0, 0, 0, 0, -1])
        vertices.extend([math.cos(a1), math.sin(a1), 0, 0, 0, -1])
        vertices.extend([math.cos(a0), math.sin(a0), 0, 0, 0, -1])
    return np.array(vertices, dtype='f4')


def _get_cached_sphere(detail: int = 12) -> Tuple[moderngl.Buffer, int]:
    global _cached_sphere_vbo, _ctx
    if detail not in _cached_sphere_vbo:
        data = _generate_sphere_geometry(detail)
        vbo = _ctx.buffer(data.tobytes())
        _cached_sphere_vbo[detail] = (vbo, len(data) // 6)
    return _cached_sphere_vbo[detail]


def _get_cached_cube() -> Tuple[moderngl.Buffer, int]:
    global _cached_cube_vbo, _ctx
    if _cached_cube_vbo is None:
        data = _generate_cube_geometry()
        vbo = _ctx.buffer(data.tobytes())
        _cached_cube_vbo = (vbo, len(data) // 6)
    return _cached_cube_vbo


def _get_cached_cylinder(detail: int = 16) -> Tuple[moderngl.Buffer, int]:
    global _cached_unit_cylinder_vbo, _ctx
    if detail not in _cached_unit_cylinder_vbo:
        data = _generate_cylinder_geometry(detail)
        vbo = _ctx.buffer(data.tobytes())
        _cached_unit_cylinder_vbo[detail] = (vbo, len(data) // 6)
    return _cached_unit_cylinder_vbo[detail]


def _get_cached_cone(detail: int = 16) -> Tuple[moderngl.Buffer, int]:
    global _cached_cone_vbo, _ctx
    if detail not in _cached_cone_vbo:
        data = _generate_cone_geometry(detail)
        vbo = _ctx.buffer(data.tobytes())
        _cached_cone_vbo[detail] = (vbo, len(data) // 6)
    return _cached_cone_vbo[detail]


def _get_or_load_texture(tex_path: str) -> Optional[moderngl.Texture]:
    """Upload an image to the GPU as a mipmapped RGBA texture (cached)."""
    global _cached_textures, _ctx
    if tex_path not in _cached_textures:
        try:
            from PIL import Image
            img = Image.open(tex_path).convert('RGBA').transpose(Image.FLIP_TOP_BOTTOM)
            tex = _ctx.texture(img.size, 4, img.tobytes())
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.build_mipmaps()
            _cached_textures[tex_path] = tex
        except Exception as e:
            print(f"[PiViz] Texture load failed {tex_path}: {e}")
            _cached_textures[tex_path] = None
    return _cached_textures[tex_path]


def _mesh_key(path: str, mtl: str = None, texture_dir: str = None) -> str:
    """Stable string key for a (path, mtl, texture_dir) combination.

    mtl=None  → auto-detect from OBJ's mtllib directive (default).
    mtl=''    → skip all material loading (renders white).
    mtl=<str> → use explicit .mtl file path.
    """
    if mtl is None and not texture_dir:
        return path  # fast path — backwards compatible
    mtl_part = '\x01' if mtl == '' else (mtl or '')  # '' != None in cache
    return f"{path}\x00{mtl_part}\x00{texture_dir or ''}"


def _get_cached_mesh(
    path: str,
    mtl: str = None,
    texture_dir: str = None,
) -> List[Tuple[moderngl.Buffer, int, Optional[moderngl.Texture]]]:
    """Get or create cached mesh groups. Returns list of (vbo, vertex_count, texture).
    VBO format: 12 floats per vertex (pos3 + normal3 + rgba4 + uv2).
    One entry per material group."""
    global _cached_mesh_vbo, _ctx
    key = _mesh_key(path, mtl, texture_dir)
    if key not in _cached_mesh_vbo:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.stl':
            from .stl_loader import load_stl
            groups = load_stl(path)
        else:
            from .obj_loader import load_obj
            groups = load_obj(path, mtl_override=mtl, texture_dir=texture_dir)
        if not groups:
            _cached_mesh_vbo[key] = []
        else:
            entries = []
            for arr, tex_path in groups:
                vbo = _ctx.buffer(arr.tobytes())
                tex = _get_or_load_texture(tex_path) if tex_path else None
                entries.append((vbo, len(arr), tex))
            _cached_mesh_vbo[key] = entries
    return _cached_mesh_vbo[key]


# ============================================================
# TRANSFORM UTILITIES (only needed for rotated cubes now)
# ============================================================

def _make_rotation_matrix_xyz(rotation) -> np.ndarray:
    rx, ry, rz = rotation
    cx, sx = math.cos(rx), math.sin(rx)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype='f4')
    cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype='f4')
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype='f4')
    return Rz @ Ry @ Rx


def _ensure_rgba(color):
    if len(color) == 3:
        return (*color, 1.0)
    return color


# ============================================================
# PUBLIC API: Draw Functions (Direct array writes)
# ============================================================

def _init_context(ctx, view, proj):
    """Initialize context (called by engine each frame)."""
    global _ctx, _current_view, _current_proj
    global _cached_view_bytes, _cached_proj_bytes, _cached_cam_pos
    _ctx = ctx
    _current_view = view
    _current_proj = proj
    _cached_view_bytes = view.T.tobytes()
    _cached_proj_bytes = proj.T.tobytes()
    inv_view = np.linalg.inv(view)
    _cached_cam_pos = tuple(inv_view[:3, 3])


def draw_sphere(center=(0, 0, 0), radius=0.5, color=(0.7, 0.7, 0.7), detail=12):
    """Queue a sphere for batched rendering."""
    global _sphere_data, _sphere_details, _sphere_count
    i = _sphere_count
    _sphere_data = _ensure_capacity(_sphere_data, i + 1)
    _sphere_details = _ensure_capacity(_sphere_details, i + 1)

    c = _ensure_rgba(color)
    _sphere_data[i, 0] = center[0]
    _sphere_data[i, 1] = center[1]
    _sphere_data[i, 2] = center[2]
    _sphere_data[i, 3] = radius
    _sphere_data[i, 4] = c[0]
    _sphere_data[i, 5] = c[1]
    _sphere_data[i, 6] = c[2]
    _sphere_data[i, 7] = c[3]
    _sphere_details[i] = detail
    _sphere_count = i + 1

    cn = np.array(center, dtype='f4')
    _update_bounds(cn - radius, cn + radius)


def draw_cube(center=(0, 0, 0), size=1.0, color=(0.7, 0.7, 0.7), rotation=(0, 0, 0)):
    """Queue a cube for batched rendering."""
    global _cube_data, _cube_has_rotation, _cube_count
    if isinstance(size, (int, float)):
        size = (size, size, size)

    i = _cube_count
    _cube_data = _ensure_capacity(_cube_data, i + 1)
    _cube_has_rotation = _ensure_capacity(_cube_has_rotation, i + 1)

    c = _ensure_rgba(color)
    _cube_data[i, 0:3] = center
    _cube_data[i, 3:6] = size
    _cube_data[i, 6:10] = c
    _cube_data[i, 10:13] = rotation
    _cube_has_rotation[i] = any(r != 0 for r in rotation)
    _cube_count = i + 1

    cn = np.array(center, dtype='f4')
    s = np.array(size, dtype='f4')
    if _cube_has_rotation[i]:
        r = np.linalg.norm(s) * 0.5
        _update_bounds(cn - r, cn + r)
    else:
        half = s * 0.5
        _update_bounds(cn - half, cn + half)


def draw_cylinder(start=(0, 0, 0), end=(0, 0, 1), radius=0.2, color=(0.7, 0.7, 0.7), detail=16):
    """Queue a cylinder for batched rendering."""
    global _cylinder_data, _cylinder_details, _cylinder_count
    i = _cylinder_count
    _cylinder_data = _ensure_capacity(_cylinder_data, i + 1)
    _cylinder_details = _ensure_capacity(_cylinder_details, i + 1)

    c = _ensure_rgba(color)
    _cylinder_data[i, 0:3] = start
    _cylinder_data[i, 3:6] = end
    _cylinder_data[i, 6] = radius
    _cylinder_data[i, 7:11] = c
    _cylinder_details[i] = detail
    _cylinder_count = i + 1

    p1 = np.array(start, dtype='f4')
    p2 = np.array(end, dtype='f4')
    _update_bounds(np.minimum(p1, p2) - radius, np.maximum(p1, p2) + radius)


def draw_cone(base=(0, 0, 0), tip=(0, 0, 1), radius=0.3, color=(0.7, 0.7, 0.7), detail=16):
    """Queue a cone for batched rendering."""
    global _cone_data, _cone_details, _cone_count
    i = _cone_count
    _cone_data = _ensure_capacity(_cone_data, i + 1)
    _cone_details = _ensure_capacity(_cone_details, i + 1)

    c = _ensure_rgba(color)
    _cone_data[i, 0:3] = base
    _cone_data[i, 3:6] = tip
    _cone_data[i, 6] = radius
    _cone_data[i, 7:11] = c
    _cone_details[i] = detail
    _cone_count = i + 1

    p1 = np.array(base, dtype='f4')
    p2 = np.array(tip, dtype='f4')
    _update_bounds(np.minimum(p1, p2) - radius, np.maximum(p1, p2) + radius)


def draw_line(start, end, color=(1, 1, 1), width=1.0):
    """Queue a line for batched rendering."""
    global _line_data, _line_count
    i = _line_count
    _line_data = _ensure_capacity(_line_data, i + 1)

    c = _ensure_rgba(color)
    _line_data[i, 0:3] = start
    _line_data[i, 3:6] = end
    _line_data[i, 6:10] = c
    _line_data[i, 10] = width
    _line_count = i + 1

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
    global _tri_v1, _tri_v2, _tri_v3, _tri_c1, _tri_c2, _tri_c3, _tri_count
    i = _tri_count
    _tri_v1 = _ensure_capacity(_tri_v1, i + 1)
    _tri_v2 = _ensure_capacity(_tri_v2, i + 1)
    _tri_v3 = _ensure_capacity(_tri_v3, i + 1)
    _tri_c1 = _ensure_capacity(_tri_c1, i + 1)
    _tri_c2 = _ensure_capacity(_tri_c2, i + 1)
    _tri_c3 = _ensure_capacity(_tri_c3, i + 1)

    c = _ensure_rgba(color)
    _tri_v1[i] = v1
    _tri_v2[i] = v2
    _tri_v3[i] = v3
    _tri_c1[i] = c
    _tri_c2[i] = c
    _tri_c3[i] = c
    _tri_count = i + 1

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
    """Draw a single point (rendered immediately)."""
    global _ctx
    if _ctx is None:
        return
    p = np.array(position, dtype='f4')
    _update_bounds(p, p)
    prog = _get_program('batched_lines')
    color = _ensure_rgba(color)
    vertices = np.array([*position, *color], dtype='f4')
    vbo = _ctx.buffer(vertices.tobytes())
    vao = _ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
    _set_common_uniforms(prog, 'batched_lines')
    _ctx.point_size = size
    _ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    vao.render(moderngl.POINTS)
    vbo.release()
    vao.release()


def draw_face(v1, v2, v3, c1=(1, 0, 0), c2=(0, 1, 0), c3=(0, 0, 1)):
    """Draw a triangle with per-vertex colors (batched)."""
    global _tri_v1, _tri_v2, _tri_v3, _tri_c1, _tri_c2, _tri_c3, _tri_count
    i = _tri_count
    _tri_v1 = _ensure_capacity(_tri_v1, i + 1)
    _tri_v2 = _ensure_capacity(_tri_v2, i + 1)
    _tri_v3 = _ensure_capacity(_tri_v3, i + 1)
    _tri_c1 = _ensure_capacity(_tri_c1, i + 1)
    _tri_c2 = _ensure_capacity(_tri_c2, i + 1)
    _tri_c3 = _ensure_capacity(_tri_c3, i + 1)

    _tri_v1[i] = v1
    _tri_v2[i] = v2
    _tri_v3[i] = v3
    _tri_c1[i] = _ensure_rgba(c1)
    _tri_c2[i] = _ensure_rgba(c2)
    _tri_c3[i] = _ensure_rgba(c3)
    _tri_count = i + 1

    pts = np.array([v1, v2, v3], dtype='f4')
    _update_bounds(np.min(pts, axis=0), np.max(pts, axis=0))


def draw_path(points, color=(1, 1, 1), width=1.0):
    """Draw a connected path through points."""
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        draw_line(points[i], points[i + 1], color, width)


def draw_mesh(
    path,
    position=(0, 0, 0),
    scale=(1, 1, 1),
    rotation=(0, 0, 0),
    color=(1, 1, 1, 1),
    mtl=None,
    texture_dir=None,
):
    """
    Queue a mesh instance for batched rendering.

    Args:
        path:        Path to .obj file (geometry cached on first load)
        position:    (x, y, z) world position
        scale:       (sx, sy, sz) or scalar
        rotation:    (rx, ry, rz) Euler angles in radians (Rz * Ry * Rx order)
        color:       (r, g, b) or (r, g, b, a) — multiplied with vertex colors
        mtl:         Optional explicit .mtl path (overrides 'mtllib' in OBJ)
        texture_dir: Optional directory to search for texture images
    """
    global _mesh_instances, _mesh_counts

    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    c = _ensure_rgba(color)

    key = _mesh_key(path, mtl, texture_dir)
    if key not in _mesh_instances:
        _mesh_instances[key] = np.empty((_INITIAL_CAPACITY, 13), dtype='f4')
        _mesh_counts[key] = 0

    i = _mesh_counts[key]
    _mesh_instances[key] = _ensure_capacity(_mesh_instances[key], i + 1)

    _mesh_instances[key][i, 0:3] = position
    _mesh_instances[key][i, 3:6] = scale
    _mesh_instances[key][i, 6:9] = rotation
    _mesh_instances[key][i, 9:13] = c
    _mesh_counts[key] = i + 1

    p = np.array(position, dtype='f4')
    s_max = max(abs(scale[0]), abs(scale[1]), abs(scale[2]))
    _update_bounds(p - s_max, p + s_max)


def draw_meshes_batch(
    path,
    positions,
    scales,
    rotations,
    colors,
    mtl=None,
    texture_dir=None,
):
    """
    Queue many mesh instances from numpy arrays — zero per-mesh loop.

    Args:
        path:        Path to .obj file
        positions:   (N, 3) float32 array
        scales:      (N, 3) float32 array, or scalar, or (3,) broadcast
        rotations:   (N, 3) float32 array, or (3,) broadcast, or None
        colors:      (N, 3) or (N, 4) float32 array
        mtl:         Optional explicit .mtl path (overrides 'mtllib' in OBJ)
        texture_dir: Optional directory to search for texture images
    """
    global _mesh_instances, _mesh_counts

    if not isinstance(positions, np.ndarray):
        positions = np.array(positions, dtype='f4')
    n = len(positions)
    if n == 0:
        return

    if isinstance(scales, (int, float)):
        scales = np.full((n, 3), scales, dtype='f4')
    elif not isinstance(scales, np.ndarray):
        scales = np.array(scales, dtype='f4')
    if scales.ndim == 1:
        scales = np.broadcast_to(scales, (n, 3)).copy()

    if rotations is None:
        rotations = np.zeros((n, 3), dtype='f4')
    elif not isinstance(rotations, np.ndarray):
        rotations = np.array(rotations, dtype='f4')
    if rotations.ndim == 1:
        rotations = np.broadcast_to(rotations, (n, 3)).copy()

    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f4')

    key = _mesh_key(path, mtl, texture_dir)
    if key not in _mesh_instances:
        _mesh_instances[key] = np.empty((max(_INITIAL_CAPACITY, n), 13), dtype='f4')
        _mesh_counts[key] = 0

    needed = _mesh_counts[key] + n
    _mesh_instances[key] = _ensure_capacity(_mesh_instances[key], needed)

    s = _mesh_counts[key]
    _mesh_instances[key][s:s + n, 0:3] = positions
    _mesh_instances[key][s:s + n, 3:6] = scales
    _mesh_instances[key][s:s + n, 6:9] = rotations
    if colors.shape[-1] == 3:
        _mesh_instances[key][s:s + n, 9:12] = colors
        _mesh_instances[key][s:s + n, 12] = 1.0
    else:
        _mesh_instances[key][s:s + n, 9:13] = colors
    _mesh_counts[key] = s + n

    s_max = float(np.max(np.abs(scales)))
    _update_bounds(np.min(positions, axis=0) - s_max, np.max(positions, axis=0) + s_max)


# ============================================================
# BULK BATCH API (numpy arrays in, zero Python loops)
# ============================================================

def draw_spheres_batch(centers, radii, colors, detail=12):
    """
    Draw many spheres from numpy arrays — zero per-sphere Python overhead.

    Args:
        centers: (N, 3) float32 array of sphere centers
        radii: (N,) float32 array of radii, or scalar
        colors: (N, 3) or (N, 4) float32 array of colors
        detail: tessellation level (shared by all spheres)
    """
    global _sphere_data, _sphere_details, _sphere_count

    if not isinstance(centers, np.ndarray):
        centers = np.array(centers, dtype='f4')
    n = len(centers)
    if n == 0:
        return

    if isinstance(radii, (int, float)):
        radii = np.full(n, radii, dtype='f4')
    elif not isinstance(radii, np.ndarray):
        radii = np.array(radii, dtype='f4')

    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f4')

    # Ensure capacity
    needed = _sphere_count + n
    _sphere_data = _ensure_capacity(_sphere_data, needed)
    _sphere_details = _ensure_capacity(_sphere_details, needed)

    s = _sphere_count
    # Bulk write — no Python loop
    _sphere_data[s:s + n, 0:3] = centers
    _sphere_data[s:s + n, 3] = radii
    if colors.shape[-1] == 3:
        _sphere_data[s:s + n, 4:7] = colors
        _sphere_data[s:s + n, 7] = 1.0  # alpha
    else:
        _sphere_data[s:s + n, 4:8] = colors
    _sphere_details[s:s + n] = detail
    _sphere_count = s + n

    # Bounds
    r_max = float(np.max(radii))
    _update_bounds(np.min(centers, axis=0) - r_max, np.max(centers, axis=0) + r_max)


def draw_cylinders_batch(starts, ends, radii, colors, detail=8):
    """
    Draw many cylinders from numpy arrays — zero per-cylinder Python overhead.

    Args:
        starts: (N, 3) float32 array of start points
        ends: (N, 3) float32 array of end points
        radii: (N,) float32 array or scalar
        colors: (N, 3) or (N, 4) float32 array of colors
        detail: tessellation level (shared)
    """
    global _cylinder_data, _cylinder_details, _cylinder_count

    if not isinstance(starts, np.ndarray):
        starts = np.array(starts, dtype='f4')
    if not isinstance(ends, np.ndarray):
        ends = np.array(ends, dtype='f4')
    n = len(starts)
    if n == 0:
        return

    if isinstance(radii, (int, float)):
        radii = np.full(n, radii, dtype='f4')
    elif not isinstance(radii, np.ndarray):
        radii = np.array(radii, dtype='f4')

    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f4')

    needed = _cylinder_count + n
    _cylinder_data = _ensure_capacity(_cylinder_data, needed)
    _cylinder_details = _ensure_capacity(_cylinder_details, needed)

    s = _cylinder_count
    _cylinder_data[s:s + n, 0:3] = starts
    _cylinder_data[s:s + n, 3:6] = ends
    _cylinder_data[s:s + n, 6] = radii
    if colors.shape[-1] == 3:
        _cylinder_data[s:s + n, 7:10] = colors
        _cylinder_data[s:s + n, 10] = 1.0  # alpha
    else:
        _cylinder_data[s:s + n, 7:11] = colors
    _cylinder_details[s:s + n] = detail
    _cylinder_count = s + n

    r_max = float(np.max(radii))
    all_pts = np.concatenate([starts, ends], axis=0)
    _update_bounds(np.min(all_pts, axis=0) - r_max, np.max(all_pts, axis=0) + r_max)


def draw_lines_batch(starts, ends, colors, width=1.0):
    """
    Draw many lines from numpy arrays — zero per-line Python overhead.

    Args:
        starts: (N, 3) float32 array
        ends: (N, 3) float32 array
        colors: (N, 3) or (N, 4) float32 array
        width: line width (scalar, shared)
    """
    global _line_data, _line_count

    if not isinstance(starts, np.ndarray):
        starts = np.array(starts, dtype='f4')
    if not isinstance(ends, np.ndarray):
        ends = np.array(ends, dtype='f4')
    n = len(starts)
    if n == 0:
        return

    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f4')

    needed = _line_count + n
    _line_data = _ensure_capacity(_line_data, needed)

    s = _line_count
    _line_data[s:s + n, 0:3] = starts
    _line_data[s:s + n, 3:6] = ends
    if colors.shape[-1] == 3:
        _line_data[s:s + n, 6:9] = colors
        _line_data[s:s + n, 9] = 1.0
    else:
        _line_data[s:s + n, 6:10] = colors
    _line_data[s:s + n, 10] = width
    _line_count = s + n

    all_pts = np.concatenate([starts, ends], axis=0)
    _update_bounds(np.min(all_pts, axis=0), np.max(all_pts, axis=0))


def draw_triangles_batch(v1s, v2s, v3s, colors):
    """
    Draw many flat-colored triangles from numpy arrays — zero per-triangle loop.

    Args:
        v1s: (N, 3) float32 array — first vertices
        v2s: (N, 3) float32 array — second vertices
        v3s: (N, 3) float32 array — third vertices
        colors: (N, 3) or (N, 4) float32 array — one color per triangle
    """
    global _tri_v1, _tri_v2, _tri_v3, _tri_c1, _tri_c2, _tri_c3, _tri_count

    if not isinstance(v1s, np.ndarray):
        v1s = np.array(v1s, dtype='f4')
    n = len(v1s)
    if n == 0:
        return
    if not isinstance(v2s, np.ndarray):
        v2s = np.array(v2s, dtype='f4')
    if not isinstance(v3s, np.ndarray):
        v3s = np.array(v3s, dtype='f4')
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f4')

    needed = _tri_count + n
    _tri_v1 = _ensure_capacity(_tri_v1, needed)
    _tri_v2 = _ensure_capacity(_tri_v2, needed)
    _tri_v3 = _ensure_capacity(_tri_v3, needed)
    _tri_c1 = _ensure_capacity(_tri_c1, needed)
    _tri_c2 = _ensure_capacity(_tri_c2, needed)
    _tri_c3 = _ensure_capacity(_tri_c3, needed)

    s = _tri_count
    _tri_v1[s:s + n] = v1s
    _tri_v2[s:s + n] = v2s
    _tri_v3[s:s + n] = v3s

    # Expand 3-component color to 4
    if colors.ndim == 1:
        # Single color for all triangles
        c = _ensure_rgba(tuple(colors))
        _tri_c1[s:s + n] = c
        _tri_c2[s:s + n] = c
        _tri_c3[s:s + n] = c
    elif colors.shape[-1] == 3:
        _tri_c1[s:s + n, :3] = colors
        _tri_c1[s:s + n, 3] = 1.0
        _tri_c2[s:s + n, :3] = colors
        _tri_c2[s:s + n, 3] = 1.0
        _tri_c3[s:s + n, :3] = colors
        _tri_c3[s:s + n, 3] = 1.0
    else:
        _tri_c1[s:s + n] = colors
        _tri_c2[s:s + n] = colors
        _tri_c3[s:s + n] = colors

    _tri_count = s + n

    all_pts = np.concatenate([v1s, v2s, v3s], axis=0)
    _update_bounds(np.min(all_pts, axis=0), np.max(all_pts, axis=0))


def draw_faces_batch(v1s, v2s, v3s, c1s, c2s, c3s):
    """
    Draw many per-vertex-colored triangles — zero per-face loop.

    Args:
        v1s, v2s, v3s: (N, 3) float32 arrays — triangle vertices
        c1s, c2s, c3s: (N, 3) or (N, 4) float32 arrays — per-vertex colors
    """
    global _tri_v1, _tri_v2, _tri_v3, _tri_c1, _tri_c2, _tri_c3, _tri_count

    if not isinstance(v1s, np.ndarray):
        v1s = np.array(v1s, dtype='f4')
    n = len(v1s)
    if n == 0:
        return
    if not isinstance(v2s, np.ndarray):
        v2s = np.array(v2s, dtype='f4')
    if not isinstance(v3s, np.ndarray):
        v3s = np.array(v3s, dtype='f4')
    if not isinstance(c1s, np.ndarray):
        c1s = np.array(c1s, dtype='f4')
    if not isinstance(c2s, np.ndarray):
        c2s = np.array(c2s, dtype='f4')
    if not isinstance(c3s, np.ndarray):
        c3s = np.array(c3s, dtype='f4')

    needed = _tri_count + n
    _tri_v1 = _ensure_capacity(_tri_v1, needed)
    _tri_v2 = _ensure_capacity(_tri_v2, needed)
    _tri_v3 = _ensure_capacity(_tri_v3, needed)
    _tri_c1 = _ensure_capacity(_tri_c1, needed)
    _tri_c2 = _ensure_capacity(_tri_c2, needed)
    _tri_c3 = _ensure_capacity(_tri_c3, needed)

    s = _tri_count
    _tri_v1[s:s + n] = v1s
    _tri_v2[s:s + n] = v2s
    _tri_v3[s:s + n] = v3s

    if c1s.shape[-1] == 3:
        _tri_c1[s:s + n, :3] = c1s
        _tri_c1[s:s + n, 3] = 1.0
        _tri_c2[s:s + n, :3] = c2s
        _tri_c2[s:s + n, 3] = 1.0
        _tri_c3[s:s + n, :3] = c3s
        _tri_c3[s:s + n, 3] = 1.0
    else:
        _tri_c1[s:s + n] = c1s
        _tri_c2[s:s + n] = c2s
        _tri_c3[s:s + n] = c3s

    _tri_count = s + n

    all_pts = np.concatenate([v1s, v2s, v3s], axis=0)
    _update_bounds(np.min(all_pts, axis=0), np.max(all_pts, axis=0))


# ============================================================
# PARTICLES (Already optimized - single draw call)
# ============================================================

def draw_particles(positions, colors, sizes=1.0):
    """Draw particles (already batched - renders immediately)."""
    global _ctx
    if _ctx is None:
        return
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
    _set_common_uniforms(prog, 'particles')
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
# GPU BUFFER MANAGEMENT
# ============================================================

def _ensure_instance_buffer(needed_bytes: int):
    global _instance_buffer, _instance_buffer_size, _ctx
    if _instance_buffer is None or _instance_buffer_size < needed_bytes:
        if _instance_buffer is not None:
            _instance_buffer.release()
        new_size = max(needed_bytes, _instance_buffer_size * 2, 4 * 1024 * 1024)
        _instance_buffer = _ctx.buffer(reserve=new_size)
        _instance_buffer_size = new_size
        _clear_vao_cache_for_buffer('instance')


def _ensure_line_buffer(needed_bytes: int):
    global _line_buffer, _line_buffer_size, _ctx
    if _line_buffer is None or _line_buffer_size < needed_bytes:
        if _line_buffer is not None:
            _line_buffer.release()
        new_size = max(needed_bytes, _line_buffer_size * 2, 1024 * 1024)
        _line_buffer = _ctx.buffer(reserve=new_size)
        _line_buffer_size = new_size
        _clear_vao_cache_for_buffer('line')


def _ensure_triangle_buffer(needed_bytes: int):
    global _triangle_buffer, _triangle_buffer_size, _ctx
    if _triangle_buffer is None or _triangle_buffer_size < needed_bytes:
        if _triangle_buffer is not None:
            _triangle_buffer.release()
        new_size = max(needed_bytes, _triangle_buffer_size * 2, 2 * 1024 * 1024)
        _triangle_buffer = _ctx.buffer(reserve=new_size)
        _triangle_buffer_size = new_size
        _clear_vao_cache_for_buffer('triangle')


def _clear_vao_cache_for_buffer(buffer_type: str):
    global _cached_vaos
    keys_to_remove = []
    for key in _cached_vaos:
        type_id = key[0]
        if buffer_type == 'instance' and type_id in ('sphere', 'cube', 'cylinder', 'cone'):
            keys_to_remove.append(key)
        elif buffer_type == 'line' and type_id == 'lines':
            keys_to_remove.append(key)
        elif buffer_type == 'triangle' and type_id == 'triangles':
            keys_to_remove.append(key)
    for key in keys_to_remove:
        if key in _cached_vaos:
            _cached_vaos[key].release()
            del _cached_vaos[key]


def _get_cached_vao(key, create_fn):
    global _cached_vaos
    if key not in _cached_vaos:
        _cached_vaos[key] = create_fn()
    return _cached_vaos[key]


# ============================================================
# BATCH FLUSH RENDERERS (zero Python loops in hot path)
# ============================================================

def _flush_spheres():
    """Render all queued spheres — zero-loop flush using dedicated sphere shader."""
    global _sphere_count
    if _sphere_count == 0:
        return

    n = _sphere_count
    data = _sphere_data[:n]  # (n, 8): cx,cy,cz, radius, r,g,b,a
    details = _sphere_details[:n]  # (n,)

    prog_name = 'instanced_sphere'
    prog = _get_program(prog_name)

    # Group by detail level — fast path for single detail
    unique_details = np.unique(details)

    for detail in unique_details:
        detail = int(detail)
        geo_vbo, vertex_count = _get_cached_sphere(detail)

        if len(unique_details) == 1:
            # Fast path: all same detail, use data directly
            subset = data
        else:
            mask = details == detail
            subset = data[mask]

        count = len(subset)

        # Data is already in the right layout: center(3), radius(1), color(4) = 8 floats
        # which maps to: inst_center(3f), inst_radius(1f), inst_color(4f)
        _ensure_instance_buffer(subset.nbytes)
        _instance_buffer.write(subset.tobytes())

        vao_key = ('sphere', detail)

        def create_vao(gvbo=geo_vbo):
            return _ctx.vertex_array(prog, [
                (gvbo, '3f 3f', 'in_position', 'in_normal'),
                (_instance_buffer, '3f 1f 4f/i', 'inst_center', 'inst_radius', 'inst_color'),
            ])

        vao = _get_cached_vao(vao_key, create_vao)

        _set_common_uniforms(prog, prog_name)
        _set_lighting_uniforms(prog, prog_name)
        vao.render(moderngl.TRIANGLES, instances=count)


def _flush_cubes():
    """Render all queued cubes."""
    global _cube_count
    if _cube_count == 0:
        return

    n = _cube_count
    data = _cube_data[:n]  # (n, 13): center3, size3, color4, rotation3
    has_rot = _cube_has_rotation[:n]

    prog_name = 'instanced_solid'
    prog = _get_program(prog_name)
    geo_vbo, vertex_count = _get_cached_cube()

    # Build 20-float instance data: matrix(16) + color(4)
    instance_data = np.zeros((n, 20), dtype='f4')

    # Fast path: cubes without rotation (majority case)
    no_rot_mask = ~has_rot
    if np.any(no_rot_mask):
        idx = np.where(no_rot_mask)[0]
        # Build diagonal matrices directly with vectorized indexing
        instance_data[idx, 0] = data[idx, 3]  # size_x -> m[0,0]
        instance_data[idx, 5] = data[idx, 4]  # size_y -> m[1,1]
        instance_data[idx, 10] = data[idx, 5]  # size_z -> m[2,2]
        instance_data[idx, 15] = 1.0  # m[3,3]
        instance_data[idx, 12] = data[idx, 0]  # cx -> m[3,0] col-major
        instance_data[idx, 13] = data[idx, 1]  # cy
        instance_data[idx, 14] = data[idx, 2]  # cz
        instance_data[idx, 16] = data[idx, 6]  # r
        instance_data[idx, 17] = data[idx, 7]  # g
        instance_data[idx, 18] = data[idx, 8]  # b
        instance_data[idx, 19] = data[idx, 9]  # a

    # Slow path: cubes with rotation (rare)
    if np.any(has_rot):
        for i in np.where(has_rot)[0]:
            center = data[i, 0:3]
            size = data[i, 3:6]
            color = data[i, 6:10]
            rotation = data[i, 10:13]
            rot_matrix = _make_rotation_matrix_xyz(rotation)
            scale_mat = np.diag(size).astype('f4')
            rot_scale = rot_matrix @ scale_mat
            m = np.eye(4, dtype='f4')
            m[:3, :3] = rot_scale
            m[0, 3], m[1, 3], m[2, 3] = center
            instance_data[i, :16] = m.T.flatten()
            instance_data[i, 16:20] = color

    _ensure_instance_buffer(instance_data.nbytes)
    _instance_buffer.write(instance_data.tobytes())

    vao_key = ('cube', 0)

    def create_vao():
        return _ctx.vertex_array(prog, [
            (geo_vbo, '3f 3f', 'in_position', 'in_normal'),
            (_instance_buffer, '4f 4f 4f 4f 4f/i', 'inst_row0', 'inst_row1', 'inst_row2', 'inst_row3', 'inst_color'),
        ])

    vao = _get_cached_vao(vao_key, create_vao)

    _set_common_uniforms(prog, prog_name)
    _set_lighting_uniforms(prog, prog_name)
    vao.render(moderngl.TRIANGLES, instances=n)


def _flush_directional(shape_name: str):
    """Render cylinders or cones — zero-loop flush, data already in GPU layout."""
    if shape_name == 'cylinder':
        count = _cylinder_count
        if count == 0:
            return
        data = _cylinder_data[:count]  # (n, 11)
        details = _cylinder_details[:count]
        get_geo = _get_cached_cylinder
    else:
        count = _cone_count
        if count == 0:
            return
        data = _cone_data[:count]
        details = _cone_details[:count]
        get_geo = _get_cached_cone

    prog_name = 'instanced_directional'
    prog = _get_program(prog_name)

    unique_details = np.unique(details)

    for detail in unique_details:
        detail = int(detail)
        geo_vbo, vertex_count = get_geo(detail)

        if len(unique_details) == 1:
            subset = data
        else:
            mask = details == detail
            subset = data[mask]

        n = len(subset)

        # Data layout already matches shader: start(3), end(3), radius(1), color(4)
        _ensure_instance_buffer(subset.nbytes)
        _instance_buffer.write(subset.tobytes())

        vao_key = (shape_name, detail)

        def create_vao(gvbo=geo_vbo):
            return _ctx.vertex_array(prog, [
                (gvbo, '3f 3f', 'in_position', 'in_normal'),
                (_instance_buffer, '3f 3f 1f 4f/i', 'inst_start', 'inst_end', 'inst_radius', 'inst_color'),
            ])

        vao = _get_cached_vao(vao_key, create_vao)

        _set_common_uniforms(prog, prog_name)
        _set_lighting_uniforms(prog, prog_name)
        vao.render(moderngl.TRIANGLES, instances=n)


def _flush_lines():
    """Render all queued lines — vectorized interleave, zero per-line loop."""
    global _line_count
    if _line_count == 0:
        return

    prog = _get_program('batched_lines')
    n = _line_count
    data = _line_data[:n]  # (n, 11): start3, end3, color4, width

    # Build interleaved vertex data: 2 verts per line, 7 floats each
    vertex_data = np.empty((n * 2, 7), dtype='f4')
    vertex_data[0::2, :3] = data[:, 0:3]  # starts
    vertex_data[0::2, 3:7] = data[:, 6:10]  # colors
    vertex_data[1::2, :3] = data[:, 3:6]  # ends
    vertex_data[1::2, 3:7] = data[:, 6:10]  # colors

    _ensure_line_buffer(vertex_data.nbytes)
    _line_buffer.write(vertex_data.tobytes())

    vao_key = ('lines',)

    def create_vao():
        return _ctx.vertex_array(prog, [
            (_line_buffer, '3f 4f', 'in_position', 'in_color')
        ])

    vao = _get_cached_vao(vao_key, create_vao)

    _set_common_uniforms(prog, 'batched_lines')

    avg_width = float(np.mean(data[:, 10]))
    _ctx.line_width = avg_width

    vao.render(moderngl.LINES, vertices=n * 2)


def _flush_triangles():
    """Render all queued triangles — fully vectorized, zero per-triangle loop."""
    global _tri_count
    if _tri_count == 0:
        return

    prog = _get_program('batched_triangles')
    n = _tri_count

    # Slice the pre-filled arrays (no copy needed for computation)
    v1 = _tri_v1[:n]  # (n, 3)
    v2 = _tri_v2[:n]
    v3 = _tri_v3[:n]
    c1 = _tri_c1[:n]  # (n, 4)
    c2 = _tri_c2[:n]
    c3 = _tri_c3[:n]

    # Vectorized normal computation — single numpy call for ALL triangles
    edge1 = v2 - v1
    edge2 = v3 - v1
    normals = np.cross(edge1, edge2)
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lengths = np.maximum(norm_lengths, 1e-6)
    normals /= norm_lengths

    # Build GPU vertex data via bulk slicing (zero per-triangle loop)
    vertex_data = np.empty((n * 3, 10), dtype='f4')
    vertex_data[0::3, :3] = v1
    vertex_data[0::3, 3:6] = normals
    vertex_data[0::3, 6:10] = c1
    vertex_data[1::3, :3] = v2
    vertex_data[1::3, 3:6] = normals
    vertex_data[1::3, 6:10] = c2
    vertex_data[2::3, :3] = v3
    vertex_data[2::3, 3:6] = normals
    vertex_data[2::3, 6:10] = c3

    _ensure_triangle_buffer(vertex_data.nbytes)
    _triangle_buffer.write(vertex_data.tobytes())

    vao_key = ('triangles',)

    def create_vao():
        return _ctx.vertex_array(prog, [
            (_triangle_buffer, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')
        ])

    vao = _get_cached_vao(vao_key, create_vao)

    _set_common_uniforms(prog, 'batched_triangles')
    _set_lighting_uniforms(prog, 'batched_triangles')

    _ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES, vertices=n * 3)
    _ctx.enable(moderngl.CULL_FACE)


# ============================================================
# MAIN FLUSH (called by engine at end of frame)
# ============================================================

def _flush_meshes():
    """Render all queued mesh instances — one instanced draw call per (mesh key, group)."""
    global _mesh_counts

    for key, count in _mesh_counts.items():
        if count == 0:
            continue

        # Decode key back to (path, mtl, texture_dir)
        parts = key.split('\x00')
        path = parts[0]
        if len(parts) > 1:
            raw = parts[1]
            mtl = '' if raw == '\x01' else (raw or None)
        else:
            mtl = None
        texture_dir = parts[2] if len(parts) > 2 and parts[2] else None

        groups = _get_cached_mesh(path, mtl, texture_dir)
        if not groups:
            continue

        prog_name = 'instanced_mesh'
        prog = _get_program(prog_name)

        data = _mesh_instances[key][:count]  # (count, 13): pos3 + scale3 + rot3 + color4
        _ensure_instance_buffer(data.nbytes)
        _instance_buffer.write(data.tobytes())

        _set_common_uniforms(prog, prog_name)
        _set_lighting_uniforms(prog, prog_name)

        for grp_idx, (geo_vbo, vertex_count, texture) in enumerate(groups):
            if vertex_count == 0:
                continue

            has_tex = texture is not None
            prog['has_texture'].value = has_tex
            if has_tex:
                texture.use(location=0)
                prog['tex0'].value = 0

            vao_key = ('mesh', key, grp_idx)

            def create_vao(gvbo=geo_vbo):
                return _ctx.vertex_array(prog, [
                    (gvbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv'),
                    (_instance_buffer, '3f 3f 3f 4f/i', 'inst_pos', 'inst_scale', 'inst_rot', 'inst_color'),
                ])

            vao = _get_cached_vao(vao_key, create_vao)
            vao.render(moderngl.TRIANGLES, instances=count)


def flush_all():
    """
    Flush all queued primitives to GPU.
    Called automatically by the engine at end of each frame.
    """
    global _sphere_count, _cube_count, _cylinder_count, _cone_count
    global _line_count, _tri_count, _mesh_counts

    _flush_spheres()
    _flush_cubes()
    _flush_directional('cylinder')
    _flush_directional('cone')
    _flush_lines()
    _flush_triangles()
    _flush_meshes()

    # Update global last frame bounds
    global _last_frame_bounds, _current_min, _current_max
    if not np.all(np.isinf(_current_min)):
        _last_frame_bounds = (tuple(_current_min), tuple(_current_max))

    # Reset for next frame (just reset counters — arrays stay allocated)
    _current_min = np.full(3, np.inf, dtype='f4')
    _current_max = np.full(3, -np.inf, dtype='f4')
    _sphere_count = 0
    _cube_count = 0
    _cylinder_count = 0
    _cone_count = 0
    _line_count = 0
    _tri_count = 0
    for path in _mesh_counts:
        _mesh_counts[path] = 0


def clear_cache():
    """Clear all cached geometry (call on context recreation)."""
    global _cached_sphere_vbo, _cached_cube_vbo, _cached_unit_cylinder_vbo, _cached_cone_vbo
    global _cached_mesh_vbo, _cached_textures
    global _instance_buffer, _line_buffer, _triangle_buffer
    global _cached_vaos, _uniform_cache
    global _sphere_count, _cube_count, _cylinder_count, _cone_count
    global _line_count, _tri_count, _mesh_instances, _mesh_counts

    _uniform_cache.clear()

    # Reset counters
    _sphere_count = 0
    _cube_count = 0
    _cylinder_count = 0
    _cone_count = 0
    _line_count = 0
    _tri_count = 0
    _mesh_instances.clear()
    _mesh_counts.clear()

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

    for groups in _cached_mesh_vbo.values():
        for vbo, _, tex in groups:
            try:
                if vbo is not None:
                    vbo.release()
            except:
                pass
    _cached_mesh_vbo.clear()

    for tex in _cached_textures.values():
        try:
            if tex is not None:
                tex.release()
        except:
            pass
    _cached_textures.clear()

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

    if _triangle_buffer is not None:
        try:
            _triangle_buffer.release()
        except:
            pass
        _triangle_buffer = None

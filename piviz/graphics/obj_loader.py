"""
Fast OBJ Loader for PiViz
=========================

Parses geometry + materials and returns one buffer per material group so the
GPU can bind the correct texture for each draw call.

Return value of load_obj()
--------------------------
A list of (array, texture_path) tuples — one entry per material group:
  array        np.ndarray  shape (N, 12) float32
               columns: [x y z  nx ny nz  r g b a  u v]
  texture_path str | None  absolute path to the diffuse texture, or None

When a material has a texture the vertex color (cols 6-10) is set to white
(1,1,1,1) so the GPU fragment shader can tint it cleanly:
    final = texture(tex, uv) * inst_color

When there is no texture the vertex color is the MTL Kd value.

mtl_override semantics
----------------------
  None   → auto-detect from the OBJ's 'mtllib' directive (default)
  ''     → skip all material loading; every face renders white
  <path> → use this .mtl file; ignore 'mtllib' inside the OBJ
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file(filename: str, *search_dirs) -> str | None:
    for d in search_dirs:
        if d:
            c = os.path.join(d, filename)
            if os.path.exists(c):
                return c
    if os.path.exists(filename):
        return filename
    return None


# ---------------------------------------------------------------------------
# MTL loader
# ---------------------------------------------------------------------------

def load_mtl(mtl_path: str, texture_dir: str = None, obj_dir: str = None) -> dict:
    """
    Parse a .mtl file.

    Returns
    -------
    dict  {material_name: {'color': [r,g,b,a], 'texture': abs_path_or_None}}
    """
    materials: dict = {}
    current: str | None = None
    mtl_dir = os.path.dirname(os.path.abspath(mtl_path))

    if not os.path.exists(mtl_path):
        print(f"[PiViz] MTL not found: {mtl_path}")
        return materials

    with open(mtl_path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            kw = tokens[0].lower()

            if kw == 'newmtl':
                current = tokens[1] if len(tokens) > 1 else ''
                materials[current] = {'color': [1.0, 1.0, 1.0, 1.0], 'texture': None}

            elif kw == 'kd' and current is not None and len(tokens) >= 4:
                c = materials[current]['color']
                c[0], c[1], c[2] = float(tokens[1]), float(tokens[2]), float(tokens[3])

            elif kw == 'd' and current is not None and len(tokens) >= 2:
                materials[current]['color'][3] = float(tokens[1])

            elif kw == 'tr' and current is not None and len(tokens) >= 2:
                materials[current]['color'][3] = 1.0 - float(tokens[1])

            elif kw == 'map_kd' and current is not None and len(tokens) >= 2:
                tex_name = ' '.join(tokens[1:])
                tex_path = _find_file(tex_name, texture_dir, mtl_dir, obj_dir)
                if tex_path:
                    materials[current]['texture'] = os.path.abspath(tex_path)
                else:
                    print(f"[PiViz] Texture not found: {tex_name}")

    return materials


# ---------------------------------------------------------------------------
# OBJ loader
# ---------------------------------------------------------------------------

def load_obj(
    path: str,
    mtl_override: str = None,
    texture_dir: str = None,
) -> list:
    """
    Load an OBJ file.

    Returns a list of (array, texture_path) tuples — one per material group.
    Each array has shape (N, 12) float32: [x y z  nx ny nz  r g b a  u v].

    Args
    ----
    path          Path to the .obj file.
    mtl_override  '' → no materials (white).  '<path>' → explicit MTL.
                  None (default) → auto from 'mtllib' in the OBJ.
    texture_dir   Extra directory to search for texture images.
    """
    if not os.path.exists(path):
        print(f"[PiViz] OBJ not found: {path}")
        return []

    skip_mtl  = (mtl_override == '')
    base_dir  = os.path.dirname(os.path.abspath(path))
    obj_name  = os.path.splitext(os.path.basename(path))[0]

    # ---- parse pass --------------------------------------------------------
    v_pos: list  = []   # [x, y, z]
    v_uv:  list  = []   # [u, v]
    v_nrm: list  = []   # [nx, ny, nz]

    # Per-vertex index lists built as flat arrays (3 entries per triangle)
    vi_list  = []   # vertex position indices
    vti_list = []   # UV indices (-1 = absent)
    vni_list = []   # normal indices (-1 = absent)
    grp_list = []   # material group index per vertex

    materials:   dict = {}    # loaded MTL data
    mat_info_by_name: dict = {}   # mat_name → {color, texture}
    mat_names:   list = []    # stable insertion order
    mat_name_to_idx: dict = {}
    current_grp: int  = -1    # current material group index

    has_uvs    = False
    has_normals = False
    mtl_loaded  = False

    def _get_or_add_mat(name: str) -> int:
        if name not in mat_name_to_idx:
            mat_name_to_idx[name] = len(mat_names)
            mat_names.append(name)
        return mat_name_to_idx[name]

    # Start with a default group so faces before any usemtl are captured
    current_grp = _get_or_add_mat('__default__')

    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            kw = tokens[0]

            if kw == 'v':
                v_pos.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])

            elif kw == 'vt':
                v_uv.append([float(tokens[1]), float(tokens[2])])
                has_uvs = True

            elif kw == 'vn':
                v_nrm.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                has_normals = True

            elif kw == 'mtllib' and not skip_mtl and not mtl_loaded:
                mtl_filename = ' '.join(tokens[1:])
                if mtl_override:
                    mtl_path = mtl_override
                else:
                    mtl_path = os.path.join(base_dir, mtl_filename)
                    if not os.path.exists(mtl_path):
                        fb = os.path.join(base_dir, obj_name + '.mtl')
                        if os.path.exists(fb):
                            mtl_path = fb
                materials = load_mtl(mtl_path, texture_dir=texture_dir, obj_dir=base_dir)
                mtl_loaded = True

            elif kw == 'usemtl':
                mat_name = ' '.join(tokens[1:])
                current_grp = _get_or_add_mat(mat_name)

            elif kw == 'f':
                verts = []
                for tok in tokens[1:]:
                    p  = tok.split('/')
                    vi  = int(p[0]) - 1
                    vti = int(p[1]) - 1 if len(p) > 1 and p[1] else -1
                    vni = int(p[2]) - 1 if len(p) > 2 and p[2] else -1
                    verts.append((vi, vti, vni))
                # fan triangulation
                for i in range(1, len(verts) - 1):
                    for vi, vti, vni in (verts[0], verts[i], verts[i + 1]):
                        vi_list.append(vi)
                        vti_list.append(vti)
                        vni_list.append(vni)
                        grp_list.append(current_grp)

    if not vi_list:
        print(f"[PiViz] OBJ has no geometry: {path}")
        return []

    # ---- numpy arrays -------------------------------------------------------
    v_pos_np = np.array(v_pos, dtype='f4')
    v_uv_np  = np.array(v_uv,  dtype='f4') if has_uvs    else None
    v_nrm_np = np.array(v_nrm, dtype='f4') if has_normals else None

    vi_np  = np.array(vi_list,  dtype='i4')
    vti_np = np.array(vti_list, dtype='i4')
    vni_np = np.array(vni_list, dtype='i4')
    grp_np = np.array(grp_list, dtype='i4')

    # ---- build per-group buffers (vectorized) --------------------------------
    result = []

    for grp_idx, mat_name in enumerate(mat_names):
        mask = (grp_np == grp_idx)
        if not mask.any():
            continue

        vi  = vi_np[mask]
        vti = vti_np[mask]
        vni = vni_np[mask]
        n   = len(vi)

        buf = np.zeros((n, 12), dtype='f4')

        # positions
        buf[:, 0:3] = v_pos_np[vi]

        # normals
        if has_normals:
            valid = vni >= 0
            if valid.any():
                buf[valid, 3:6] = v_nrm_np[vni[valid]]

        # vertex color + texture
        mat_data = materials.get(mat_name, {})
        tex_path = mat_data.get('texture') if not skip_mtl else None
        if tex_path:
            buf[:, 6:10] = [1.0, 1.0, 1.0, 1.0]  # texture provides color
        else:
            color = mat_data.get('color', [1.0, 1.0, 1.0, 1.0]) if not skip_mtl else [1.0, 1.0, 1.0, 1.0]
            buf[:, 6:10] = color

        # UVs
        if has_uvs and v_uv_np is not None:
            valid = vti >= 0
            if valid.any():
                buf[valid, 10:12] = v_uv_np[vti[valid]]

        # flat normals if none in file
        if not has_normals:
            p1 = buf[0::3, 0:3]
            p2 = buf[1::3, 0:3]
            p3 = buf[2::3, 0:3]
            nrm = np.cross(p2 - p1, p3 - p1)
            ln  = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm /= np.where(ln > 0, ln, 1.0)
            buf[0::3, 3:6] = nrm
            buf[1::3, 3:6] = nrm
            buf[2::3, 3:6] = nrm

        result.append((buf, tex_path))

    return result
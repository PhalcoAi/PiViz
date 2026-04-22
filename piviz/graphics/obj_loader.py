"""
Fast OBJ Loader for PiViz
=========================

A lightweight, efficient OBJ parser optimized for ModernGL buffers.
Returns interleaved vertex data (position + normal + color).
"""

import numpy as np
import os


def load_mtl(path: str) -> dict:
    """Load material library and return a dict of material names to RGBA colors."""
    materials = {}
    current_mtl = None

    if not os.path.exists(path):
        return materials

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'newmtl':
                current_mtl = values[1]
                materials[current_mtl] = [1.0, 1.0, 1.0, 1.0]  # Default white
            elif values[0] == 'Kd' and current_mtl:
                # Diffuse color
                color = [float(x) for x in values[1:4]]
                if len(color) == 3: color.append(1.0)  # Add alpha
                materials[current_mtl] = color

    return materials


def load_obj(path: str) -> np.ndarray:
    """
    Load an OBJ file and return a numpy array of vertices.
    
    Returns:
        np.ndarray: Float32 array of shape (N, 10) containing [x, y, z, nx, ny, nz, r, g, b, a].
    """
    if not os.path.exists(path):
        print(f"Error: Mesh file not found: {path}")
        return np.array([], dtype='f4')

    # Data containers
    v_data = []
    vn_data = []
    faces = []  # (v_idx, vn_idx, mtl_idx)

    materials = {}
    current_mtl_idx = -1
    mtl_list = []  # List of colors corresponding to indices

    base_dir = os.path.dirname(path)

    has_normals = False

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                v_data.append([float(x) for x in values[1:4]])
            elif values[0] == 'vn':
                vn_data.append([float(x) for x in values[1:4]])
                has_normals = True
            elif values[0] == 'mtllib':
                mtl_path = os.path.join(base_dir, values[1])
                materials = load_mtl(mtl_path)
                # Reset list
                mtl_list = []
                # We need a way to map name to index. 
                # Let's rebuild mtl_list as we encounter usemtl
            elif values[0] == 'usemtl':
                mtl_name = values[1]
                if mtl_name in materials:
                    color = materials[mtl_name]
                    # Check if this color is already in our list to reuse index? 
                    # Or just append. Simple append is safer for state machine.
                    mtl_list.append(color)
                    current_mtl_idx = len(mtl_list) - 1
                else:
                    # Unknown material, use white
                    mtl_list.append([1.0, 1.0, 1.0, 1.0])
                    current_mtl_idx = len(mtl_list) - 1
            elif values[0] == 'f':
                # Handle f v1/vt1/vn1 v2/vt2/vn2 ...
                face_verts = []
                for v in values[1:]:
                    w = v.split('/')
                    vi = int(w[0]) - 1
                    vni = int(w[2]) - 1 if len(w) > 2 and w[2] else -1
                    face_verts.append((vi, vni))

                # Triangulate
                for i in range(1, len(face_verts) - 1):
                    faces.append((face_verts[0], face_verts[i], face_verts[i + 1], current_mtl_idx))

    # Convert to numpy
    v_np = np.array(v_data, dtype='f4')
    vn_np = np.array(vn_data, dtype='f4') if has_normals else None

    # Default color if no materials
    default_color = np.array([1.0, 1.0, 1.0, 1.0], dtype='f4')

    # Build final buffer
    num_vertices = len(faces) * 3
    buffer_data = np.zeros((num_vertices, 10), dtype='f4')

    idx = 0
    for v1, v2, v3, mtl_idx in faces:
        # Get color
        if mtl_idx >= 0 and mtl_idx < len(mtl_list):
            color = mtl_list[mtl_idx]
        else:
            color = default_color

        # Vertex 1
        buffer_data[idx, 0:3] = v_np[v1[0]]
        if v1[1] >= 0 and has_normals: buffer_data[idx, 3:6] = vn_np[v1[1]]
        buffer_data[idx, 6:10] = color
        idx += 1

        # Vertex 2
        buffer_data[idx, 0:3] = v_np[v2[0]]
        if v2[1] >= 0 and has_normals: buffer_data[idx, 3:6] = vn_np[v2[1]]
        buffer_data[idx, 6:10] = color
        idx += 1

        # Vertex 3
        buffer_data[idx, 0:3] = v_np[v3[0]]
        if v3[1] >= 0 and has_normals: buffer_data[idx, 3:6] = vn_np[v3[1]]
        buffer_data[idx, 6:10] = color
        idx += 1

    # Auto-generate normals if missing
    if not has_normals:
        for i in range(0, num_vertices, 3):
            p1 = buffer_data[i, 0:3]
            p2 = buffer_data[i + 1, 0:3]
            p3 = buffer_data[i + 2, 0:3]

            u = p2 - p1
            v = p3 - p1

            n = np.cross(u, v)
            l = np.linalg.norm(n)
            if l > 0: n /= l

            buffer_data[i, 3:6] = n
            buffer_data[i + 1, 3:6] = n
            buffer_data[i + 2, 3:6] = n

    return buffer_data

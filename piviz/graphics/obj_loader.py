"""
Fast OBJ Loader for PiViz
=========================

A lightweight, efficient OBJ parser optimized for ModernGL buffers.
Returns interleaved vertex data (position + normal).
"""

import numpy as np
import os

def load_obj(path: str) -> np.ndarray:
    """
    Load an OBJ file and return a numpy array of vertices.
    
    Returns:
        np.ndarray: Float32 array of shape (N, 6) containing [x, y, z, nx, ny, nz].
    """
    if not os.path.exists(path):
        print(f"Error: Mesh file not found: {path}")
        return np.array([], dtype='f4')

    vertices = []
    normals = []
    faces = []

    # Temporary lists for processing
    v_data = []
    vn_data = []
    
    # Check if file has normals
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
            elif values[0] == 'f':
                # Handle f v1/vt1/vn1 v2/vt2/vn2 ...
                face_verts = []
                for v in values[1:]:
                    w = v.split('/')
                    # OBJ indices are 1-based
                    vi = int(w[0]) - 1
                    vni = int(w[2]) - 1 if len(w) > 2 and w[2] else -1
                    face_verts.append((vi, vni))
                
                # Triangulate polygon (fan)
                for i in range(1, len(face_verts) - 1):
                    faces.append((face_verts[0], face_verts[i], face_verts[i+1]))

    # Convert to numpy for speed
    v_np = np.array(v_data, dtype='f4')
    if has_normals:
        vn_np = np.array(vn_data, dtype='f4')
    
    # Build final buffer
    # 3 vertices per face, 6 floats per vertex (pos + normal)
    num_vertices = len(faces) * 3
    buffer_data = np.zeros((num_vertices, 6), dtype='f4')
    
    idx = 0
    for v1, v2, v3 in faces:
        # Vertex 1
        buffer_data[idx, 0:3] = v_np[v1[0]]
        if v1[1] >= 0:
            buffer_data[idx, 3:6] = vn_np[v1[1]]
        idx += 1
        
        # Vertex 2
        buffer_data[idx, 0:3] = v_np[v2[0]]
        if v2[1] >= 0:
            buffer_data[idx, 3:6] = vn_np[v2[1]]
        idx += 1
        
        # Vertex 3
        buffer_data[idx, 0:3] = v_np[v3[0]]
        if v3[1] >= 0:
            buffer_data[idx, 3:6] = vn_np[v3[1]]
        idx += 1

    # Auto-generate normals if missing
    if not has_normals:
        # Flat shading normals
        for i in range(0, num_vertices, 3):
            p1 = buffer_data[i, 0:3]
            p2 = buffer_data[i+1, 0:3]
            p3 = buffer_data[i+2, 0:3]
            
            u = p2 - p1
            v = p3 - p1
            
            n = np.cross(u, v)
            l = np.linalg.norm(n)
            if l > 0:
                n /= l
                
            buffer_data[i, 3:6] = n
            buffer_data[i+1, 3:6] = n
            buffer_data[i+2, 3:6] = n

    return buffer_data

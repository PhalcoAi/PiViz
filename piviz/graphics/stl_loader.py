"""
STL Loader for PiViz
====================

Supports both binary and ASCII STL files.

Return value of load_stl()
---------------------------
A list containing one (array, None) tuple — same shape as load_obj() so the
same rendering pipeline handles both formats without any special-casing:

  array        np.ndarray  shape (N, 12) float32
               columns: [x y z  nx ny nz  r g b a  u v]
  texture_path None  (STL carries no material or texture data)

Vertex colors are set to white (1,1,1,1) so the instance color tint applied
at render time is the only colour input, matching mtl='' OBJ behaviour.

Normals are always recomputed from vertex positions — STL face normals can be
zero or inconsistent across exporters.
"""

import os
import struct
import numpy as np


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_binary(path: str) -> bool:
    """True if the file is a binary STL (vs ASCII)."""
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        f.read(80)                          # header
        cb = f.read(4)
        if len(cb) < 4:
            return False
    count = struct.unpack('<I', cb)[0]
    return file_size == 80 + 4 + count * 50


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_binary(path: str) -> tuple:
    """Return (v1, v2, v3) each (N, 3) float32 from a binary STL."""
    with open(path, 'rb') as f:
        f.read(80)
        count = struct.unpack('<I', f.read(4))[0]
        dtype = np.dtype([
            ('normal', np.float32, (3,)),
            ('v1',     np.float32, (3,)),
            ('v2',     np.float32, (3,)),
            ('v3',     np.float32, (3,)),
            ('attr',   np.uint16),
        ])
        raw = np.frombuffer(f.read(count * 50), dtype=dtype)
    return raw['v1'], raw['v2'], raw['v3']


def _load_ascii(path: str) -> tuple:
    """Return (v1, v2, v3) each (N, 3) float32 from an ASCII STL."""
    v1s, v2s, v3s, buf = [], [], [], []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            t = line.strip()
            if t.startswith('vertex'):
                p = t.split()
                buf.append([float(p[1]), float(p[2]), float(p[3])])
                if len(buf) == 3:
                    v1s.append(buf[0])
                    v2s.append(buf[1])
                    v3s.append(buf[2])
                    buf = []
    if not v1s:
        empty = np.empty((0, 3), dtype='f4')
        return empty, empty, empty
    return (
        np.array(v1s, dtype='f4'),
        np.array(v2s, dtype='f4'),
        np.array(v3s, dtype='f4'),
    )


# ---------------------------------------------------------------------------
# Buffer assembly
# ---------------------------------------------------------------------------

def _build_buffer(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    n = len(v1)
    if n == 0:
        return np.empty((0, 12), dtype='f4')

    e1 = v2 - v1
    e2 = v3 - v1
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm /= np.where(ln > 0, ln, 1.0)

    buf = np.zeros((n * 3, 12), dtype='f4')
    buf[0::3, 0:3] = v1
    buf[1::3, 0:3] = v2
    buf[2::3, 0:3] = v3
    buf[0::3, 3:6] = nrm
    buf[1::3, 3:6] = nrm
    buf[2::3, 3:6] = nrm
    buf[:, 6:10] = [1.0, 1.0, 1.0, 1.0]   # white — tinted by instance color
    # cols 10-12 (UV) stay 0 — STL has no texture coordinates
    return buf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_stl(path: str) -> list:
    """
    Load a binary or ASCII STL file.

    Returns a list containing one (array, None) tuple, matching the load_obj()
    return contract so primitives.py needs no format-specific logic.
    """
    if not os.path.exists(path):
        print(f"[PiViz] STL not found: {path}")
        return []

    try:
        if _is_binary(path):
            v1, v2, v3 = _load_binary(path)
        else:
            v1, v2, v3 = _load_ascii(path)
    except Exception as e:
        print(f"[PiViz] STL load failed {path}: {e}")
        return []

    buf = _build_buffer(v1, v2, v3)
    if len(buf) == 0:
        print(f"[PiViz] STL has no geometry: {path}")
        return []

    return [(buf, None)]

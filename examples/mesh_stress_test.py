"""
Mesh Rendering Stress Test
==========================

Tests the performance of the new OBJ mesh loader and batch rendering.
Requires a 'teapot.obj' or similar file in the current directory or resources.
"""

import piviz as pz
import numpy as np
import os
import time

# Configuration
MESH_FILE = "teapot.obj"  # Will fallback to generating a dummy file if missing
GRID_SIZE = 20  # Initial grid size
SPACING = 3.0

def create_dummy_obj():
    """Create a simple pyramid OBJ file for testing."""
    with open(MESH_FILE, 'w') as f:
        f.write("""
v 1.0 0.0 1.0
v 1.0 0.0 -1.0
v -1.0 0.0 -1.0
v -1.0 0.0 1.0
v 0.0 2.0 0.0
f 1 2 5
f 2 3 5
f 3 4 5
f 4 1 5
f 1 4 3 2
""")
    print(f"Created dummy mesh: {MESH_FILE}")

def set_grid_size(val):
    global GRID_SIZE
    GRID_SIZE = int(val)
    # Update label
    # Note: In a real app we'd want a way to update the label text dynamically
    # For now, the console print confirms the change
    print(f"Grid Size set to: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE**3} meshes")

def setup():
    if not os.path.exists(MESH_FILE):
        create_dummy_obj()
        
    print(f"Rendering grid of meshes...")
    
    # Setup UI
    pz.add_label("Mesh Stress Test")
    pz.add_slider("Grid Size", 5, 50, GRID_SIZE, set_grid_size)

def update(dt):
    t = time.time()
    
    # Render a massive grid of meshes
    count = 0
    limit = GRID_SIZE
    
    # Center the grid
    offset = (limit * SPACING) / 2.0
    
    for x in range(limit):
        for y in range(limit):
            for z in range(limit):
                # Calculate position
                px = x * SPACING - offset
                py = y * SPACING - offset
                pz_val = z * SPACING - offset
                
                # Animate rotation and color
                rot_x = t + x * 0.1
                rot_y = t * 0.5 + y * 0.1
                
                r = (np.sin(t + x * 0.2) + 1) / 2
                g = (np.cos(t + y * 0.2) + 1) / 2
                b = (np.sin(t + z * 0.2) + 1) / 2
                
                pz.mesh(
                    path=MESH_FILE,
                    pos=(px, py, pz_val),
                    scale=0.8,
                    rotation=(rot_x, rot_y, 0),
                    color=(r, g, b)
                )
                count += 1
                
                # Safety break for extreme values
                if count > 125000: break

if __name__ == '__main__':
    pz.run(setup, update)

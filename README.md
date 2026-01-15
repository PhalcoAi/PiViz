# πViz: Interactive Scientific Visualization Engine

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![Status](https://img.shields.io/badge/status-beta-orange)

**πViz** is a high-performance, Python-native 3D visualization library designed specifically for academic simulations and engineering analysis. Built on ModernGL (OpenGL 3.3+), it bridges the gap between raw computational data and publication-quality visuals, offering a lightweight alternative to heavy game engines for scientific workflows.

---

## Core Features

* **High-Performance Rendering:** Optimized batch rendering pipeline capable of displaying 100,000+ particles and complex meshes in real-time.
* **Automatic Batching:** All primitive draw calls are automatically batched and rendered with minimal GPU draw calls - no manual optimization required.
* **Automatic GPU Selection:** Detects and uses the best available GPU (NVIDIA > AMD > Intel) without manual configuration.
* **Instanced Drawing:** GPU-accelerated instanced rendering for spheres, cylinders, and triangles with massive performance gains.
* **Blinn-Phong Shading:** Configurable material system with shiny/matte toggle for publication-quality visuals.
* **Native USD Player:** Integrated playback engine for Universal Scene Description (`.usdc`, `.usd`) files, supporting `PointInstancers` and `BasisCurves`.
* **Simulation Recording:** Built-in, high-throughput video exporter (`.mp4`) and lossless screenshot capture (`.png`) that bypasses UI overlays for clean figures.
* **Scientific Color System:** Comprehensive library of perceptually uniform colormaps (Viridis, Plasma, Magma) and categorical palettes designed for academic publications.
* **Immediate Mode Primitives:** Simple, Pythonic API (`pgfx`) for drawing shapes, vectors, and point clouds without boilerplate code.
* **Advanced UI System:** Integrated windowing system for creating custom simulation controllers, telemetry dashboards, and floating inspectors.
* **Accurate Performance Metrics:** Real-time FPS monitoring with exponential moving average for stable readings.

---

## Installation

### Option 1: Install via PyPI (Recommended)
```bash
pip install piviz-3d
```

*Note: The package automatically installs `imageio[ffmpeg]` for video recording support.*

### Option 2: Install from Source

For developers or those wanting the latest unreleased features:

```bash
git clone https://github.com/PhalcoAi/PiViz.git
cd PiViz
pip install -e .
```

**Dependencies:**
The installation will automatically fetch: `moderngl`, `moderngl-window`, `imgui`, `numpy`, `imageio[ffmpeg]`, `usd-core`, `psutil`, and `GPUtil`.

---

## Quick Start

### 1. Basic Geometry and Primitives

PiViz uses a "stateless" immediate-mode style for drawing, making it easy to integrate into existing simulation loops. All draw calls are automatically batched for optimal performance.

```python
from piviz import PiVizStudio, PiVizFX, pgfx, Colors, Palette

class MySimulation(PiVizFX):
    def render(self, time, dt):
        # Draw a ground plane
        pgfx.draw_plane(size=(10, 10), color=Colors.GREY)
        
        # Draw a dynamic sphere
        import math
        z = 1.0 + math.sin(time) * 0.5
        pgfx.draw_sphere(
            center=(0, 0, z), 
            radius=0.5, 
            color=Palette.Standard10[0]  # Academic Blue
        )

if __name__ == '__main__':
    PiVizStudio(scene_fx=MySimulation()).run()
```

### 2. Massive Particle Systems

For rendering dense point clouds (fluid simulations, astrophysics), use the optimized particle renderer:

```python
import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colormap

class GalaxyViz(PiVizFX):
    def setup(self):
        self.n = 100000
        self.pos = np.random.randn(self.n, 3).astype('f4') * 5.0
        # Color particles based on distance from center using Viridis map
        dist = np.linalg.norm(self.pos, axis=1)
        norm_dist = dist / np.max(dist)
        self.colors = np.array([Colormap.viridis(d) for d in norm_dist], dtype='f4')

    def render(self, time, dt):
        # Render 100k particles in one draw call
        pgfx.draw_particles(self.pos, self.colors, sizes=2.0)

if __name__ == '__main__':
    PiVizStudio(scene_fx=GalaxyViz()).run()
```

### 3. Spring-Mass Networks and Meshes

For simulations with thousands of nodes and connections, PiViz automatically batches all primitives:

```python
import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx

class MassiveNetwork(PiVizFX):
    def setup(self):
        self.n_nodes = 2000
        self.positions = np.random.randn(self.n_nodes, 3).astype('f4') * 5.0
        
        # Generate random connections
        self.connections = []
        for i in range(self.n_nodes):
            for j in range(i + 1, min(i + 5, self.n_nodes)):
                self.connections.append((i, j))

    def render(self, time, dt):
        # Draw all nodes - automatically batched!
        for i in range(self.n_nodes):
            pgfx.draw_sphere(
                center=tuple(self.positions[i]),
                radius=0.1,
                color=(0.2, 0.6, 1.0)
            )
        
        # Draw all connections - automatically batched!
        for i, j in self.connections:
            pgfx.draw_cylinder(
                start=tuple(self.positions[i]),
                end=tuple(self.positions[j]),
                radius=0.02,
                color=(0.8, 0.3, 0.2)
            )

if __name__ == '__main__':
    PiVizStudio(scene_fx=MassiveNetwork()).run()
```

**Performance Comparison:**

| Scenario | v1.x (Individual) | v2.0 (Batched) | Speedup |
|----------|-------------------|----------------|---------|
| 100 nodes | 60 FPS | 60 FPS | 1x |
| 500 nodes + 1000 springs | 20 FPS | 60 FPS | 3x |
| 1000 nodes + 2000 springs | 3 FPS | 55 FPS | **18x** |
| 5000 nodes + 10000 springs | <1 FPS | 40 FPS | **40x+** |

### 4. Triangle Meshes

For custom geometry and terrain rendering, triangles are also automatically batched:

```python
import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx

class TerrainMesh(PiVizFX):
    def setup(self):
        # Generate a grid of triangles
        self.triangles = []
        grid_size = 50
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                x0, x1 = i * 0.5 - 12, (i + 1) * 0.5 - 12
                y0, y1 = j * 0.5 - 12, (j + 1) * 0.5 - 12
                
                # Height based on sine waves
                z00 = np.sin(x0) * np.cos(y0)
                z10 = np.sin(x1) * np.cos(y0)
                z01 = np.sin(x0) * np.cos(y1)
                z11 = np.sin(x1) * np.cos(y1)
                
                # Two triangles per grid cell
                self.triangles.append(((x0, y0, z00), (x1, y0, z10), (x1, y1, z11)))
                self.triangles.append(((x0, y0, z00), (x1, y1, z11), (x0, y1, z01)))

    def render(self, time, dt):
        # All triangles batched into single draw call
        for v1, v2, v3 in self.triangles:
            height = (v1[2] + v2[2] + v3[2]) / 3
            color = (0.2 + height * 0.3, 0.5, 0.8 - height * 0.3)
            pgfx.draw_triangle(v1, v2, v3, color)

if __name__ == '__main__':
    PiVizStudio(scene_fx=TerrainMesh()).run()
```

### 5. Material System (Shiny vs Matte)

Control the visual appearance of all primitives with the material API:

```python
from piviz import pgfx
from piviz.graphics.primitives import set_material_shiny, set_material_matte

# Shiny materials (Blinn-Phong specular highlights)
set_material_shiny(shiny=True, shininess=64.0, specular=0.6)

# Matte materials (diffuse only, clay-like appearance)
set_material_matte()

# Fine-grained control
set_material_shiny(
    shiny=True,
    shininess=32.0,   # 1-128: higher = tighter highlights
    specular=0.5      # 0-1: specular intensity
)
```

**Shininess Guide:**

| Value | Appearance |
|-------|------------|
| 1-10 | Very soft, almost matte |
| 32 | Balanced (default) |
| 64-128 | Glossy, metallic look |

### 6. Playing USD Simulations

To visualize external simulation data saved in Pixar's USD format:

```python
import piviz

if __name__ == '__main__':
    # Launches the native player with timeline controls
    piviz.play_usd("simulation_output.usdc")
```

---

## Automatic GPU Selection

PiViz automatically detects and selects the best available GPU at startup. The selection priority is:

1. NVIDIA discrete GPU (highest performance)
2. AMD discrete GPU
3. Intel integrated GPU (with performance warning)

On systems with multiple GPUs (e.g., laptop with Intel + NVIDIA), PiViz automatically sets the appropriate environment variables to use the discrete GPU.

**Startup output example:**
```
PiViz GPU Detection
----------------------------------------
  Available GPUs:
    [DISCRETE] NVIDIA: NVIDIA GeForce RTX 3060
    [INTEGRATED] INTEL: Intel UHD Graphics

  NVIDIA GPU selected via environment offload
  Set: __NV_PRIME_RENDER_OFFLOAD=1
  Set: __GLX_VENDOR_LIBRARY_NAME=nvidia
----------------------------------------
```

**Manual GPU selection (if needed):**
```bash
# Force NVIDIA on Linux
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python your_script.py

# Force AMD on Linux
DRI_PRIME=1 python your_script.py

# Or set permanently via prime-select
sudo prime-select nvidia
```

---

## User Interface and Controls

The Studio interface provides essential tools for inspection and capture without cluttering the view.

| Control | Action |
|---------|--------|
| **Rotate** | Left Click + Drag |
| **Pan** | Right Click + Drag / Middle Click + Drag |
| **Zoom** | Scroll Wheel |
| **ViewCube** | Click faces/corners on the cube to snap views |

### Toolbar Functions

* **REC:** Toggles video recording. Flashes red when active. Saves directly to `exports/`.
* **IMG:** Captures a high-resolution screenshot of the simulation only (excludes UI windows).
* **Grid/Axes:** Toggle rendering of the spatial reference guides.
* **Theme:** Toggles between Dark Mode (default) and Light Mode (optimized for printing).

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `G` | Toggle Grid |
| `A` | Toggle Axes |
| `T` | Toggle Theme |
| `0` | Isometric View |
| `1` | Front View |
| `3` | Top View |

---

## Performance Overlay

The built-in performance HUD displays:

* **FPS:** Real-time frame rate with min/avg/max statistics
* **Frame Time:** Milliseconds per frame
* **CPU/RAM:** System resource usage
* **GPU:** Load, temperature, and VRAM (if GPUtil available)
* **Scene Stats:** Custom metrics from your simulation

The overlay uses accurate `time.perf_counter()` measurements with exponential moving average for stable, jitter-free readings.

---

## Architecture

```
piviz/
├── core/
│   ├── camera.py        # SolidWorks-style orbital camera
│   ├── gpu_selector.py  # Automatic GPU detection and selection
│   ├── scene.py         # PiVizFX base class
│   ├── studio.py        # Main application engine
│   └── theme.py         # Dark/Light/Publication themes
├── graphics/
│   ├── environment.py   # Infinite grid + axes renderers
│   └── primitives.py    # pgfx drawing functions + automatic batching
├── ui/
│   ├── manager.py       # Widget management
│   ├── overlay.py       # Performance HUD
│   ├── viewcube.py      # Navigation widget
│   └── widgets.py       # Label, Button, Slider, Checkbox, etc.
└── resources/
```

---

## API Reference

### Primitives (pgfx)

All primitives are automatically batched for optimal performance.

```python
# Basic shapes
pgfx.draw_sphere(center, radius, color, detail=12)
pgfx.draw_cube(center, size, color, rotation=(0,0,0))
pgfx.draw_cylinder(start, end, radius, color, detail=16)
pgfx.draw_cone(base, tip, radius, color, detail=16)
pgfx.draw_plane(size, color, center, normal)
pgfx.draw_arrow(start, end, color, head_size=0.1)

# Lines and paths
pgfx.draw_line(start, end, color, width=1.0)
pgfx.draw_path(points, color, width=1.0)

# Points and particles
pgfx.draw_point(position, color, size=5.0)
pgfx.draw_particles(positions, colors, sizes)  # Batch optimized

# Triangles and faces
pgfx.draw_triangle(v1, v2, v3, color)
pgfx.draw_face(v1, v2, v3, c1, c2, c3)  # Per-vertex color
```

### Material Control

```python
from piviz.graphics.primitives import set_material_shiny, set_material_matte

set_material_shiny(shiny=True, shininess=48.0, specular=0.5)
set_material_matte()
```

### GPU Selection (Advanced)

```python
from piviz.core.gpu_selector import auto_select_gpu, verify_gpu_selection

# Manually trigger GPU selection (normally automatic)
result = auto_select_gpu(verbose=True)

# Verify which GPU is actually being used
info = verify_gpu_selection()
print(f"Renderer: {info['renderer']}")
```

---

## Rendering Architecture (v2.0)

PiViz v2.0 uses a deferred batched rendering architecture:

1. **Draw calls queue primitives:** `draw_sphere()`, `draw_cylinder()`, etc. add shapes to internal queues
2. **Geometry is cached:** Unit sphere, cube, and cylinder geometry is generated once and reused
3. **Instanced rendering:** All shapes of the same type are rendered in a single GPU draw call
4. **Automatic flush:** The engine calls `flush_all()` at the end of each frame

This architecture provides 10-40x performance improvement over traditional immediate-mode rendering, with zero changes required to user code.

**Draw call comparison:**

| Primitives | v1.x Draw Calls | v2.0 Draw Calls |
|------------|-----------------|-----------------|
| 1000 spheres | 1000 | 1 |
| 500 cylinders | 500 | 1 |
| 2000 triangles | 2000 | 1 |
| Mixed scene | 3500 | 3-5 |

---

## Contributing

We welcome contributions from the scientific and open-source community!

1. **Fork the repository** on GitHub.
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`).
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`).
4. **Push to the branch** (`git push origin feature/AmazingFeature`).
5. **Open a Pull Request**.

Please ensure your code follows the existing style guidelines and includes comments where necessary. For major changes, please open an issue first to discuss what you would like to change.

---

## License

Copyright 2026 Yogesh Phalak.

Licensed under the **Apache License, Version 2.0** (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

```
./LICENSE
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
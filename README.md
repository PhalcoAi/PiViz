# πviz: High-Performance Scientific Visualization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/piviz-3d.svg)](https://pypi.org/project/piviz-3d/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![OpenGL](https://img.shields.io/badge/OpenGL-3.3+-red.svg)](https://www.opengl.org/)
[![ModernGL](https://img.shields.io/badge/ModernGL-latest-orange.svg)](https://github.com/moderngl/moderngl)
[![ImGui](https://img.shields.io/badge/ImGui-UI-blueviolet.svg)](https://github.com/ocornut/imgui)
[![NumPy](https://img.shields.io/badge/NumPy-accelerated-success.svg)](https://numpy.org/)

**πviz** is a Python-native 3D visualization library designed for academic simulations and engineering analysis. Built on ModernGL (OpenGL 3.3+), it delivers publication-quality rendering with minimal overhead, making it ideal for computational mechanics, physics simulations, and scientific computing workflows.

**Backend:** ModernGL • OpenGL 3.3+ • GPU-Accelerated Batched Rendering • MSAA 4x

![PiViz Demo](exports/piviz_galaxy.gif)

---

## Key Features

- **Batched Rendering Pipeline** – Automatic primitive batching delivers 10-40x performance improvement over traditional immediate-mode rendering
- **GPU-Accelerated Instancing** – Render 100,000+ particles and complex meshes at 60 FPS
- **Scientific Colormaps** – Perceptually uniform colormaps (Viridis, Plasma, Magma, Coolwarm) and categorical palettes
- **Publication-Quality Export** – High-resolution screenshots and MP4 video recording with clean, UI-free output
- **Blinn-Phong Shading** – Configurable material system with adjustable shininess and specular highlights
- **Immediate-Mode API** – Simple, Pythonic drawing interface with zero boilerplate
- **Automatic GPU Selection** – Detects and uses the best available GPU (NVIDIA > AMD > Intel)
- **USD File Playback** – Native support for Pixar Universal Scene Description files
- **Integrated UI System** – Built-in widgets for simulation control and parameter adjustment

---

## Installation

```bash
pip install piviz-3d
```

Dependencies are automatically installed: `moderngl`, `moderngl-window`, `imgui`, `numpy`, `imageio[ffmpeg]`, `usd-core`

---

## Quick Start

### Simple Script Syntax

πviz provides a script-like interface for rapid prototyping:

```python
import piviz as pz

# Simulation state
ball_pos = [0, 0, 0]
ball_vel = [0, 0, 0]
gravity = -9.8

def setup():
    """Initialize simulation"""
    pz.add_button("Reset", lambda: reset_ball())
    pz.add_slider("Gravity", 1.0, 20.0, 9.8, lambda v: set_gravity(v))

def update(dt):
    """Update physics and render"""
    # Physics
    ball_vel[2] += gravity * dt
    for i in range(3):
        ball_pos[i] += ball_vel[i] * dt
    
    # Collision
    if ball_pos[2] < 0:
        ball_pos[2] = 0
        ball_vel[2] *= -0.8
    
    # Render
    pz.box(pos=(0, 0, -0.1), size=(10, 10, 0.2), color=(0.3, 0.3, 0.3))
    pz.sphere(pos=tuple(ball_pos), radius=0.5, color=(1, 0.2, 0.2))

if __name__ == '__main__':
    pz.run(setup, update)
```

### Class-Based Approach

For complex simulations with state management:

```python
from piviz import PiVizStudio, PiVizFX, pgfx
import numpy as np

class SpringMassNetwork(PiVizFX):
    def setup(self):
        self.n_nodes = 1000
        self.positions = np.random.randn(self.n_nodes, 3).astype('f4') * 5.0
        self.connections = [(i, (i+1) % self.n_nodes) for i in range(self.n_nodes)]
    
    def render(self, time, dt):
        # All primitives automatically batched - single draw call per type
        for pos in self.positions:
            pgfx.draw_sphere(center=tuple(pos), radius=0.1, color=(0.2, 0.6, 1.0))
        
        for i, j in self.connections:
            pgfx.draw_cylinder(
                start=tuple(self.positions[i]),
                end=tuple(self.positions[j]),
                radius=0.02,
                color=(0.8, 0.3, 0.2)
            )

if __name__ == '__main__':
    PiVizStudio(scene_fx=SpringMassNetwork()).run()
```

### Massive Particle Systems

Optimized for dense point clouds and large-scale datasets:

```python
import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colormap

class ParticleSimulation(PiVizFX):
    def setup(self):
        self.n = 100000
        self.positions = np.random.randn(self.n, 3).astype('f4') * 5.0
        
        # Color by distance using scientific colormap
        dist = np.linalg.norm(self.positions, axis=1)
        norm_dist = dist / np.max(dist)
        self.colors = np.array([Colormap.viridis(d) for d in norm_dist], dtype='f4')
    
    def render(self, time, dt):
        pgfx.draw_particles(self.positions, self.colors, sizes=2.0)

if __name__ == '__main__':
    PiVizStudio(scene_fx=ParticleSimulation()).run()
```

---

## Performance

**Automatic Batching System** (v2.0+)

| Configuration | Draw Calls (v1.x) | Draw Calls (v2.0) | Performance Gain |
|---------------|-------------------|-------------------|------------------|
| 100 nodes | 100 | 1 | 1x (baseline) |
| 500 nodes + 1000 springs | 1,500 | 2 | 3x faster |
| 1000 nodes + 2000 springs | 3,000 | 2 | 18x faster |
| 5000 nodes + 10000 springs | 15,000 | 2 | 40x+ faster |

The rendering pipeline automatically batches all primitives of the same type into single instanced draw calls, eliminating per-object GPU overhead.

---

## Controls

| Action | Input |
|--------|-------|
| Rotate View | Left Click + Drag |
| Pan | Right Click + Drag / Middle Click + Drag |
| Zoom | Scroll Wheel |
| Toggle Grid | `G` |
| Toggle Axes | `A` |
| Toggle Theme | `T` |
| Isometric View | `0` |
| Front View | `1` |
| Top View | `3` |

---

## Material System

Control surface appearance with the material API:

```python
from piviz.graphics.primitives import set_material_shiny, set_material_matte

# Glossy materials with specular highlights
set_material_shiny(shiny=True, shininess=64.0, specular=0.6)

# Matte/clay appearance
set_material_matte()
```

**Shininess scale:** 1-10 (soft) | 32 (balanced) | 64-128 (metallic)

---

## USD File Playback

Visualize simulation data stored in Pixar's Universal Scene Description format:

```python
import piviz

piviz.play_usd("simulation_output.usdc")
```

Supports `PointInstancers` and `BasisCurves` primitives with timeline controls.

---

## Export Capabilities

**Screenshot:** Click the camera icon or press `IMG` to capture high-resolution PNG

**Video Recording:** Click the record button to start/stop MP4 export (saved to `exports/`)

All exports exclude UI overlays for publication-ready output.

---

## Architecture

```
piviz/
├── core/
│   ├── studio.py        # Main application engine (ModernGL 3.3+, MSAA 4x)
│   ├── camera.py        # Orbital camera with SolidWorks-style navigation
│   ├── gpu_selector.py  # Automatic GPU detection and selection
│   ├── scene.py         # PiVizFX base class for simulations
│   └── theme.py         # Dark/Light/Publication color schemes
├── graphics/
│   ├── primitives.py    # Batched drawing API (pgfx)
│   └── environment.py   # Grid and axes renderers
├── ui/
│   ├── manager.py       # Widget system
│   ├── overlay.py       # Performance HUD
│   └── viewcube.py      # Navigation cube
└── resources/
```

**Rendering Pipeline:** Deferred batched rendering with automatic geometry caching and GPU instancing

---

## API Reference

### Drawing Primitives (pgfx)

```python
# Shapes (automatically batched)
pgfx.draw_sphere(center, radius, color, detail=12)
pgfx.draw_cube(center, size, color, rotation=(0,0,0))
pgfx.draw_cylinder(start, end, radius, color, detail=16)
pgfx.draw_plane(size, color, center=(0,0,0), normal=(0,0,1))

# Lines and paths
pgfx.draw_line(start, end, color, width=1.0)
pgfx.draw_arrow(start, end, color, head_size=0.1)

# Particles (optimized batch renderer)
pgfx.draw_particles(positions, colors, sizes)

# Triangular meshes
pgfx.draw_triangle(v1, v2, v3, color)
```

### Scientific Colormaps

```python
from piviz import Colormap, Palette, Colors

# Perceptually uniform colormaps
color = Colormap.viridis(t)      # t ∈ [0, 1]
color = Colormap.plasma(t)
color = Colormap.magma(t)
color = Colormap.coolwarm(t)

# Categorical palettes
colors = Palette.Standard10       # 10 distinct colors
primary = Colors.BLUE            # Named colors
```

---

## Use Cases

- Computational mechanics and structural analysis
- Particle-based physics simulations (SPH, DEM, MD)
- Soft robotics and origami kinematics
- Reservoir computing and neural networks
- Spring-mass networks and lattice structures
- Educational physics demonstrations

---

## Requirements

- Python 3.8+
- OpenGL 3.3+ capable GPU
- Windows, Linux, or macOS

---

## License

Copyright 2026 Yogesh Phalak

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.

---

## Citation

If you use πviz in academic work, please cite:

```bibtex
@software{phalak2026piviz,
  author = {Phalak, Yogesh},
  title = {PiViz: High-Performance Scientific Visualization},
  year = {2026},
  url = {https://github.com/PhalcoAi/PiViz}
}
```
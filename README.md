# πViz: Interactive Scientific Visualization Engine

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![Status](https://img.shields.io/badge/status-beta-orange)

**πViz** (PiViz) is a high-performance, Python-native 3D visualization library designed specifically for academic simulations and engineering analysis. Built on ModernGL (OpenGL 3.3+), it bridges the gap between raw computational data and publication-quality visuals, offering a lightweight alternative to heavy game engines for scientific workflows.

##  Core Features

* **High-Performance Rendering:** Optimized batch rendering pipeline capable of displaying 100,000+ particles and complex paths in real-time.
* **Native USD Player:** Integrated playback engine for Universal Scene Description (`.usdc`, `.usd`) files, supporting `PointInstancers` and `BasisCurves`.
* **Simulation Recording:** Built-in, high-throughput video exporter (`.mp4`) and lossless screenshot capture (`.png`) that bypasses UI overlays for clean figures.
* **Scientific Color System:** Comprehensive library of perceptually uniform colormaps (Viridis, Plasma, Magma) and categorical palettes designed for academic publications.
* **Immediate Mode Primitives:** Simple, Pythonic API (`pgfx`) for drawing shapes, vectors, and point clouds without boilerplate code.
* **Advanced UI System:** Integrated windowing system for creating custom simulation controllers, telemetry dashboards, and floating inspectors.

---

##  Installation

### Option 1: Install via PyPI (Recommended)
Once released, you can install πViz directly into your environment:

```bash
pip install piviz

```

*Note: The package automatically installs `imageio[ffmpeg]` for video recording support.*

### Option 2: Install from Source

For developers or those wanting the latest unreleased features:

1. Clone the repository:
```bash
git clone [https://github.com/PhalcoAi/PiViz.git](https://github.com/PhalcoAi/PiViz.git)
cd PiViz

```


2. Install in editable mode:
```bash
pip install -e .

```



**Dependencies:**
The installation will automatically fetch: `moderngl`, `moderngl-window`, `imgui`, `numpy`, `imageio[ffmpeg]`, `usd-core`, `psutil`, and `GPUtil`.

---

##  Quick Start

### 1. Basic Geometry & Primitives

πViz uses a "stateless" immediate-mode style for drawing, making it easy to integrate into existing simulation loops.

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
            color=Palette.Standard10[0] # Academic Blue
        )

if __name__ == '__main__':
    PiVizStudio(scene_fx=MySimulation()).run()

```

### 2. Massive Particle Systems

For rendering dense point clouds (fluid simulations, astrophysics), use the optimized batch renderer:

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

### 3. Playing USD Simulations

To visualize external simulation data saved in Pixar's USD format:

```python
import piviz

if __name__ == '__main__':
    # Launches the native player with timeline controls
    piviz.play_usd("simulation_output.usdc")

```

---

##  User Interface & Controls

The Studio interface provides essential tools for inspection and capture without cluttering the view.

| Control | Action |
| --- | --- |
| **Rotate** | Left Click + Drag |
| **Pan** | Right Click + Drag / Middle Click + Drag |
| **Zoom** | Scroll Wheel |
| **ViewCube** | Click faces/corners on the cube to snap views |

### Toolbar Functions

* 🔴 **REC:** Toggles video recording. Flashes red when active. Saves directly to `exports/`.
* 📷 **IMG:** Captures a high-resolution screenshot of the *simulation only* (excludes UI windows).
* **Grid/Axes:** Toggle rendering of the spatial reference guides.
* **Theme (Sun/Moon):** Toggles between Dark Mode (default) and Light Mode (optimized for printing).

### Keyboard Shortcuts

* `G`: Toggle Grid
* `A`: Toggle Axes
* `T`: Toggle Theme
* `0`: Isometric View
* `1`: Front View
* `3`: Top View

---

##  Contributing

We welcome contributions from the scientific and open-source community!

1. **Fork the repository** on GitHub.
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`).
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`).
4. **Push to the branch** (`git push origin feature/AmazingFeature`).
5. **Open a Pull Request**.

Please ensure your code follows the existing style guidelines and includes comments where necessary. For major changes, please open an issue first to discuss what you would like to change.

---

##  License

Copyright © 2026 Yogesh Phalak.

Licensed under the **Apache License, Version 2.0** (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```
./LICENSE

```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.




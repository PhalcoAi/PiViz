"""
Theme System for PiViz
======================

Consistent color palettes for dark, light, and publication modes.
"""

from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class Theme:
    """Color theme definition."""
    name: str

    # Viewport
    background: Tuple[float, float, float, float]

    # Grid
    grid_major: Tuple[float, float, float, float]
    grid_minor: Tuple[float, float, float, float]
    grid_origin: Tuple[float, float, float, float]

    # Axes
    axis_x: Tuple[float, float, float, float]
    axis_y: Tuple[float, float, float, float]
    axis_z: Tuple[float, float, float, float]
    axis_label: Tuple[float, float, float, float]

    # UI — text_primary must always contrast against panel
    text_primary: Tuple[float, float, float, float]
    text_secondary: Tuple[float, float, float, float]
    accent: Tuple[float, float, float, float]
    accent_hover: Tuple[float, float, float, float]
    panel: Tuple[float, float, float, float]

    # Primitives
    default_color: Tuple[float, float, float]


# ── Dark ─────────────────────────────────────────────────────────────────────

DARK_THEME = Theme(
    name="dark",

    background=(0.07, 0.08, 0.10, 1.0),  # deep navy-black

    grid_major=(0.25, 0.26, 0.31, 0.78),
    grid_minor=(0.15, 0.16, 0.20, 0.42),
    grid_origin=(0.40, 0.42, 0.48, 1.0),

    axis_x=(0.95, 0.28, 0.28, 1.0),
    axis_y=(0.28, 0.92, 0.38, 1.0),
    axis_z=(0.28, 0.50, 0.95, 1.0),
    axis_label=(0.70, 0.70, 0.74, 1.0),

    text_primary=(0.92, 0.92, 0.95, 1.0),  # near-white — works on dark panels
    text_secondary=(0.48, 0.50, 0.57, 1.0),  # muted
    accent=(0.98, 0.52, 0.08, 1.0),  # vibrant orange
    accent_hover=(1.00, 0.65, 0.22, 1.0),
    panel=(0.10, 0.11, 0.14, 0.93),  # dark glass

    default_color=(0.60, 0.60, 0.64),
)

# ── Light ─────────────────────────────────────────────────────────────────────
# Panels are light; text is near-black; accent is a darker orange for contrast.

LIGHT_THEME = Theme(
    name="light",

    background=(0.90, 0.91, 0.93, 1.0),  # silver-white

    grid_major=(0.44, 0.44, 0.48, 0.68),
    grid_minor=(0.60, 0.60, 0.64, 0.42),
    grid_origin=(0.24, 0.24, 0.32, 0.85),

    axis_x=(0.80, 0.14, 0.14, 1.0),
    axis_y=(0.10, 0.54, 0.14, 1.0),
    axis_z=(0.14, 0.28, 0.80, 1.0),
    axis_label=(0.10, 0.10, 0.13, 1.0),

    text_primary=(0.08, 0.08, 0.10, 1.0),  # near-black — works on light panels
    text_secondary=(0.34, 0.35, 0.40, 1.0),  # medium grey
    accent=(0.80, 0.30, 0.04, 1.0),  # dark orange — readable on light bg
    accent_hover=(0.90, 0.40, 0.10, 1.0),
    panel=(0.87, 0.88, 0.90, 0.96),  # light panel

    default_color=(0.20, 0.20, 0.23),
)

# ── Publication ───────────────────────────────────────────────────────────────

PUBLICATION_THEME = Theme(
    name="publication",

    background=(1.0, 1.0, 1.0, 1.0),

    grid_major=(0.68, 0.68, 0.70, 0.48),
    grid_minor=(0.82, 0.82, 0.84, 0.28),
    grid_origin=(0.48, 0.48, 0.52, 0.65),

    axis_x=(0.72, 0.12, 0.12, 1.0),
    axis_y=(0.08, 0.52, 0.12, 1.0),
    axis_z=(0.12, 0.22, 0.72, 1.0),
    axis_label=(0.18, 0.18, 0.18, 1.0),

    text_primary=(0.05, 0.05, 0.05, 1.0),
    text_secondary=(0.38, 0.38, 0.40, 1.0),
    accent=(0.15, 0.15, 0.15, 1.0),
    accent_hover=(0.35, 0.35, 0.35, 1.0),
    panel=(0.93, 0.93, 0.95, 0.95),

    default_color=(0.22, 0.22, 0.22),
)

# ─────────────────────────────────────────────────────────────────────────────

THEMES: Dict[str, Theme] = {
    'dark': DARK_THEME,
    'light': LIGHT_THEME,
    'publication': PUBLICATION_THEME,
}


def get_theme(name: str) -> Theme:
    return THEMES.get(name.lower(), DARK_THEME)

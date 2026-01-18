# piviz/core/theme.py
"""
Theme System for PiViz
======================

Provides consistent color schemes for dark and light modes.
Useful for academic figures where white backgrounds are preferred.
"""

from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class Theme:
    """Color theme definition."""
    name: str

    # Background
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

    # UI
    text_primary: Tuple[float, float, float, float]
    text_secondary: Tuple[float, float, float, float]
    accent: Tuple[float, float, float, float]
    accent_hover: Tuple[float, float, float, float]
    panel: Tuple[float, float, float, float]

    # Defaults for primitives
    default_color: Tuple[float, float, float]


# === Predefined Themes ===

DARK_THEME = Theme(
    name="dark",

    # Deep blue-grey background
    background=(0.08, 0.09, 0.11, 1.0),

    # Subtle grid
    grid_major=(0.28, 0.28, 0.32, 0.85),
    grid_minor=(0.18, 0.18, 0.22, 0.55),
    grid_origin=(0.45, 0.45, 0.5, 1.0),

    # Vivid axes
    axis_x=(0.95, 0.3, 0.3, 1.0),
    axis_y=(0.3, 0.95, 0.4, 1.0),
    axis_z=(0.3, 0.5, 0.95, 1.0),
    axis_label=(0.8, 0.8, 0.8, 1.0),

    # UI colors - light text on dark panels
    text_primary=(0.95, 0.95, 0.97, 1.0),
    text_secondary=(0.6, 0.6, 0.65, 1.0),
    accent=(1.0, 0.45, 0.1, 1.0),  # Orange
    accent_hover=(1.0, 0.55, 0.2, 1.0),
    panel=(0.12, 0.12, 0.15, 0.92),

    default_color=(0.7, 0.7, 0.7),
)

LIGHT_THEME = Theme(
    name="light",

    # Clean off-white background
    background=(0.92, 0.93, 0.94, 1.0),

    # Visible grid with good contrast
    grid_major=(0.5, 0.5, 0.55, 0.8),
    grid_minor=(0.68, 0.68, 0.72, 0.55),
    grid_origin=(0.3, 0.3, 0.38, 0.95),

    # Saturated axes for visibility on light background
    axis_x=(0.82, 0.18, 0.18, 1.0),
    axis_y=(0.12, 0.58, 0.18, 1.0),
    axis_z=(0.18, 0.32, 0.82, 1.0),
    axis_label=(0.12, 0.12, 0.15, 1.0),

    # High contrast UI - dark panels with light text (inverted from background)
    text_primary=(0.95, 0.95, 0.97, 1.0),  # Light text on dark panels
    text_secondary=(0.7, 0.7, 0.75, 1.0),
    accent=(1.0, 0.5, 0.15, 1.0),  # Bright orange
    accent_hover=(1.0, 0.6, 0.25, 1.0),
    panel=(0.15, 0.15, 0.18, 0.94),  # Dark panels for contrast

    default_color=(0.25, 0.25, 0.28),
)

PUBLICATION_THEME = Theme(
    name="publication",

    # Pure white for papers
    background=(1.0, 1.0, 1.0, 1.0),

    # Very light grid
    grid_major=(0.7, 0.7, 0.72, 0.5),
    grid_minor=(0.85, 0.85, 0.87, 0.3),
    grid_origin=(0.5, 0.5, 0.55, 0.7),

    # Muted professional axes
    axis_x=(0.75, 0.15, 0.15, 1.0),
    axis_y=(0.1, 0.55, 0.15, 1.0),
    axis_z=(0.15, 0.25, 0.75, 1.0),
    axis_label=(0.2, 0.2, 0.2, 1.0),

    # Minimal UI
    text_primary=(0.0, 0.0, 0.0, 1.0),
    text_secondary=(0.4, 0.4, 0.4, 1.0),
    accent=(0.2, 0.2, 0.2, 1.0),
    accent_hover=(0.4, 0.4, 0.4, 1.0),
    panel=(0.92, 0.92, 0.94, 0.9),

    default_color=(0.25, 0.25, 0.25),
)

THEMES: Dict[str, Theme] = {
    'dark': DARK_THEME,
    'light': LIGHT_THEME,
    'publication': PUBLICATION_THEME,
}


def get_theme(name: str) -> Theme:
    """Get theme by name."""
    return THEMES.get(name.lower(), DARK_THEME)

# piviz/graphics/colors.py
"""
Academic Color Palettes & Colormaps
===================================
Provides scientific, perceptually uniform colormaps (Viridis, Magma, etc.)
and categorical palettes suitable for publications.
"""

from typing import Tuple, List

Color = Tuple[float, float, float, float]  # RGBA


class Colors:
    """Standard named colors (RGBA)."""
    WHITE = (1.0, 1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0, 1.0)
    RED = (1.0, 0.0, 0.0, 1.0)
    GREEN = (0.0, 1.0, 0.0, 1.0)
    BLUE = (0.0, 0.0, 1.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0, 1.0)
    CYAN = (0.0, 1.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0, 1.0)
    GREY = (0.5, 0.5, 0.5, 1.0)
    ORANGE = (1.0, 0.5, 0.0, 1.0)
    PURPLE = (0.5, 0.0, 0.5, 1.0)

    # Soft variants (good for backgrounds)
    SOFT_RED = (0.9, 0.4, 0.4, 1.0)
    SOFT_GREEN = (0.4, 0.8, 0.4, 1.0)
    SOFT_BLUE = (0.4, 0.6, 0.9, 1.0)


class Palette:
    """Standard Categorical Palettes for Academic Figures."""

    # "Tab10" style - distinct, high contrast
    Standard10: List[Color] = [
        (0.12, 0.47, 0.71, 1.0),  # Blue
        (1.00, 0.50, 0.05, 1.0),  # Orange
        (0.17, 0.63, 0.17, 1.0),  # Green
        (0.84, 0.15, 0.16, 1.0),  # Red
        (0.58, 0.40, 0.74, 1.0),  # Purple
        (0.55, 0.34, 0.29, 1.0),  # Brown
        (0.89, 0.47, 0.76, 1.0),  # Pink
        (0.50, 0.50, 0.50, 1.0),  # Grey
        (0.74, 0.74, 0.13, 1.0),  # Olive
        (0.09, 0.75, 0.81, 1.0),  # Cyan
    ]

    # "Dark" - Good for light backgrounds in papers
    Dark8: List[Color] = [
        (0.11, 0.11, 0.11, 1.0),
        (0.85, 0.37, 0.01, 1.0),
        (0.46, 0.44, 0.70, 1.0),
        (0.91, 0.16, 0.54, 1.0),
        (0.40, 0.65, 0.12, 1.0),
        (0.90, 0.67, 0.01, 1.0),
        (0.65, 0.46, 0.11, 1.0),
        (0.96, 0.51, 0.75, 1.0),
    ]

    @staticmethod
    def get(index: int, palette: List[Color] = Standard10) -> Color:
        """Get a color from a palette by index (loops automatically)."""
        return palette[index % len(palette)]


class Colormap:
    """Scientific Colormaps (approximated for lightweight usage)."""

    @staticmethod
    def viridis(t: float, alpha: float = 1.0) -> Color:
        """Perceptually uniform Blue-Green-Yellow."""
        t = max(0.0, min(1.0, t))
        # Polynomial approximation of Viridis
        r = 0.28 + 2.9 * t - 9.1 * t ** 2 + 7.0 * t ** 3
        g = 0.05 + 2.2 * t - 3.8 * t ** 2 + 2.5 * t ** 3
        b = 0.45 - 1.2 * t + 3.7 * t ** 2 - 2.8 * t ** 3
        return (min(r, 1.0), min(g, 1.0), min(b, 1.0), alpha)

    @staticmethod
    def plasma(t: float, alpha: float = 1.0) -> Color:
        """High contrast Blue-Red-Yellow."""
        t = max(0.0, min(1.0, t))
        r = 0.06 + 2.5 * t - 1.5 * t ** 2
        g = 0.02 + 0.1 * t + 3.0 * t ** 2 - 2.2 * t ** 3
        b = 0.55 + 0.6 * t - 4.5 * t ** 2 + 3.5 * t ** 3
        return (min(r, 1.0), min(g, 1.0), min(b, 1.0), alpha)

    @staticmethod
    def magma(t: float, alpha: float = 1.0) -> Color:
        """Black-Red-White (High intensity)."""
        t = max(0.0, min(1.0, t))
        r = 0.0 + 2.8 * t - 2.0 * t ** 2
        g = 0.0 + 1.0 * t + 0.5 * t ** 2
        b = 0.0 + 1.2 * t - 0.5 * t ** 2
        return (min(r, 1.0), min(g, 1.0), min(b, 1.0), alpha)

    @staticmethod
    def coolwarm(t: float, alpha: float = 1.0) -> Color:
        """Diverging Blue-White-Red."""
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            # Blue to White
            f = t * 2.0
            return (f, f, 1.0, alpha)
        else:
            # White to Red
            f = (t - 0.5) * 2.0
            return (1.0, 1.0 - f, 1.0 - f, alpha)

    @staticmethod
    def jet(t: float, alpha: float = 1.0) -> Color:
        """Classic Jet colormap (Blue-Green-Yellow-Red)."""
        t = max(0.0, min(1.0, t))
        r = min(max(1.5 - abs(4.0 * t - 3.0), 0.0), 1.0)
        g = min(max(1.5 - abs(4.0 * t - 2.0), 0.0), 1.0)
        b = min(max(1.5 - abs(4.0 * t - 1.0), 0.0), 1.0)
        return (r, g, b, alpha)

    @staticmethod
    def map(value: float, min_v: float, max_v: float, cmap_func=viridis, alpha: float = 1.0) -> Color:
        """Map a value to a color."""
        if max_v == min_v:
            t = 0.0
        else:
            t = (value - min_v) / (max_v - min_v)
        return cmap_func(t, alpha)

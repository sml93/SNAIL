from __future__ import annotations

import math
from typing import Tuple

# ---------------------------------------------------------------------------
# Waypoint generators
# ---------------------------------------------------------------------------
def multipoint_waypoints(
    *points: Tuple[float, float],
) -> list[Tuple[float, float]]:
    """
    Define an explicit sequence of (x, y) waypoints for custom motion.

    Args:
        *points: Any number of (x, y) tuples in metres, in traversal order.

    Returns:
        List of (x, y) waypoints passed directly to simulate().

    Example:
        waypoints = multipoint_waypoints(
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5),
        )
    """
    if not points:
        raise ValueError("multipoint_waypoints requires at least one point.")
    return list(points)


def circle_waypoints(
    cx: float = 0.0,
    cy: float = 0.0,
    radius: float = 0.30,
    n_points: int = 12,
    start_angle_deg: float = 0.0,
) -> list[Tuple[float, float]]:
    """
    Generate evenly-spaced waypoints around a circle.

    Args:
        cx, cy:          Centre of the circle (m).
        radius:          Circle radius (m).
        n_points:        Number of waypoints (resolution of the circle).
        start_angle_deg: Starting angle in degrees (0 = rightmost point).

    Returns:
        List of (x, y) tuples progressing counter-clockwise.
    """
    start = math.radians(start_angle_deg)
    return [
        (
            cx + radius * math.cos(start + 2 * math.pi * i / n_points),
            cy + radius * math.sin(start + 2 * math.pi * i / n_points),
        )
        for i in range(n_points)
    ]


def figure8_waypoints(
    cx: float = 0.0,
    cy: float = 0.0,
    radius: float = 0.20,
    n_points: int = 24,
) -> list[Tuple[float, float]]:
    """
    Generate waypoints tracing a figure-8 (lemniscate) path.

    Uses a parametric lemniscate of Bernoulli:
        x(t) = cx + radius * cos(t) / (1 + sin²(t))
        y(t) = cy + radius * sin(t) * cos(t) / (1 + sin²(t))

    The path crosses the centre at t=0 and t=π, forming two symmetric loops.

    Args:
        cx, cy:   Centre of the figure-8 (m).
        radius:   Half-width of each lobe (m).
        n_points: Number of waypoints (must be even for symmetric loops).

    Returns:
        List of (x, y) tuples for one full traversal of the figure-8.
    """
    pts = []
    for i in range(n_points):
        t = 2 * math.pi * i / n_points
        denom = 1 + math.sin(t) ** 2
        x = cx + radius * math.cos(t) / denom
        y = cy + radius * math.sin(t) * math.cos(t) / denom
        pts.append((x, y))
    return pts



def boustrophedon_waypoints(
    x0: float = 0.0,
    y0: float = 0.0,
    width: float = 2.0,
    height: float = 1.0,
    lane_width: float = 0.20,
) -> list[Tuple[float, float]]:
    """
    Generate a boustrophedon (lawnmower) coverage path within a rectangle.

    The robot sweeps left-to-right on even lanes and right-to-left on odd
    lanes, stepping up by lane_width between each pass.  Only the two
    endpoints of each lane are emitted as waypoints; the PID controller
    handles straight-line travel between them.

    Args:
        x0, y0:     Bottom-left corner of the coverage rectangle (m).
        width:      Rectangle width  along x (m).  Default 2.0 m.
        height:     Rectangle height along y (m).  Default 1.0 m.
        lane_width: Spacing between parallel lanes (m).

    Returns:
        List of (x, y) waypoints tracing the full boustrophedon path.
    """
    pts: list[Tuple[float, float]] = []
    n_lanes = max(1, round(height / lane_width))
    for i in range(n_lanes):
        y = y0 + i * lane_width
        if i % 2 == 0:          # even lane: left → right
            pts.append((x0,         y))
            pts.append((x0 + width, y))
        else:                   # odd lane: right → left
            pts.append((x0 + width, y))
            pts.append((x0,         y))
    return pts
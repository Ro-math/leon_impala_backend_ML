import math
from typing import Tuple, List

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_point_in_triangle(pt: Tuple[float, float], v1: Tuple[float, float], v2: Tuple[float, float], v3: Tuple[float, float]) -> bool:
    """
    Check if a point pt is inside the triangle defined by vertices v1, v2, v3.
    Using barycentric coordinates.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def get_interpolated_points(p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get integer points along the line segment between p1 and p2.
    Useful for checking line of sight or movement paths if needed.
    """
    points = []
    x1, y1 = p1
    x2, y2 = p2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
            
    return points

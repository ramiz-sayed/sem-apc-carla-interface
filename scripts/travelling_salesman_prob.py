#!/usr/bin/env python3
"""
TSP planner in 3D with optional orientation costs and visualization.

- Distances: 3D Euclidean
- Orientation (optional): penalizes initial heading mismatch (start pose vs first leg)
                         and sharp turns between legs (segment-to-segment direction changes).
- Solvers:
    * Exact (brute force) for small waypoint sets
    * Heuristic (Nearest Neighbor + optional 2-opt) for larger sets
- Returns: (sequence_of_indices, total_cost)
- Visualization: 3D matplotlib plot of the chosen route

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Data types ------------------------------------

@dataclass(frozen=True)
class Pose:
    """Start pose (angles in degrees)."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

# ----------------------------- Helpers ---------------------------------------

def _to_vec_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit direction vector from yaw (Z axis) and pitch (X-Y plane tilt). Roll is ignored for direction."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    # Aerospace convention: yaw about +Z, pitch positive up; here we define
    # forward vector in world XYZ (x forward, y left, z up).
    vx = math.cos(pitch) * math.cos(yaw)
    vy = math.cos(pitch) * math.sin(yaw)
    vz = math.sin(pitch)
    v = np.array([vx, vy, vz], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])

def _segment_dir(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit direction vector from a->b (handles zero length)."""
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])

def _angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    """Unsigned angle (radians) between two unit vectors."""
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.acos(dot)

def _euclid3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _build_points_array(waypoints_xyz: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(waypoints_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("waypoints must be an iterable of (x,y,z)")
    return arr

# ----------------------- Orientation-aware path cost -------------------------

@dataclass
class OrientationWeights:
    """Weights for orientation penalty terms."""
    yaw: float = 0.0     # weight on heading mismatch (radians) vs 1 meter
    pitch: float = 0.0   # included via same heading mismatch angle
    roll: float = 0.0    # not used by geometry; kept for API symmetry
    turn: float = 0.0    # weight on turn angle between consecutive legs (radians)

def path_cost(points: np.ndarray,
              order: List[int],
              start_pose: Pose,
              ori_w: OrientationWeights) -> float:
    """
    Total cost = sum of 3D distances + optional orientation penalties:
      - Initial heading mismatch (start orientation vs first segment)
      - Turn penalties (angle between consecutive segments)
    All angles are multiplied by their weights (meters/radian) so cost is in 'meters'.
    """
    if not order:
        return 0.0

    p0 = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)
    cost = 0.0

    # First leg
    a = p0
    b = points[order[0]]
    cost += _euclid3(a, b)

    # Orientation penalty for first leg
    if any([ori_w.yaw, ori_w.pitch, ori_w.roll]):  # roll unused; included for API completeness
        start_dir = _to_vec_yaw_pitch(start_pose.yaw, start_pose.pitch)
        seg_dir = _segment_dir(a, b)
        ang = _angle_between(start_dir, seg_dir)  # radians
        cost += ang * max(ori_w.yaw, ori_w.pitch, ori_w.roll)

    # Remaining legs
    for i in range(len(order) - 1):
        a = points[order[i]]
        b = points[order[i + 1]]
        cost += _euclid3(a, b)

    # Turn penalties between consecutive segments
    if ori_w.turn > 0 and len(order) >= 2:
        # segment 0: start->wp0
        prev_a, prev_b = p0, points[order[0]]
        prev_dir = _segment_dir(prev_a, prev_b)
        for i in range(len(order) - 1):
            cur_a, cur_b = points[order[i]], points[order[i + 1]]
            cur_dir = _segment_dir(cur_a, cur_b)
            cost += _angle_between(prev_dir, cur_dir) * ori_w.turn
            prev_dir = cur_dir

    return float(cost)

# ----------------------------- Solvers ---------------------------------------

def tsp_exact(points: np.ndarray,
              start_pose: Pose,
              ori_w: OrientationWeights) -> Tuple[List[int], float]:
    """Brute force (exact) TSP (start fixed at the start pose). O(n!)."""
    n = len(points)
    best_order: Optional[List[int]] = None
    best_cost = float("inf")
    for order in itertools.permutations(range(n)):
        order = list(order)
        c = path_cost(points, order, start_pose, ori_w)
        if c < best_cost:
            best_cost, best_order = c, order
    return best_order or [], best_cost

def _nearest_neighbor(points: np.ndarray,
                      start_xyz: np.ndarray) -> List[int]:
    """Build a tour with nearest neighbor from the start position."""
    n = len(points)
    if n == 0:
        return []
    remaining = set(range(n))
    # pick first as nearest to start pose position
    d0 = np.linalg.norm(points - start_xyz, axis=1)
    cur = int(np.argmin(d0))
    order = [cur]
    remaining.remove(cur)
    while remaining:
        d = np.linalg.norm(points[list(remaining)] - points[cur], axis=1)
        j = int(np.argmin(d))
        cur = list(remaining)[j]
        order.append(cur)
        remaining.remove(cur)
    return order

def _two_opt(points: np.ndarray,
             order: List[int],
             start_pose: Pose,
             ori_w: OrientationWeights,
             max_iter: int = 100) -> List[int]:
    """Simple 2-opt local improvement."""
    if len(order) < 4:
        return order
    improved = True
    it = 0
    best = order[:]
    best_cost = path_cost(points, best, start_pose, ori_w)

    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i:k+1][::-1] + best[k+1:]
                c = path_cost(points, candidate, start_pose, ori_w)
                if c + 1e-9 < best_cost:
                    best, best_cost = candidate, c
                    improved = True
        # early exit if no change in this outer pass
    return best

def tsp_heuristic(points: np.ndarray,
                  start_pose: Pose,
                  ori_w: OrientationWeights,
                  use_two_opt: bool = True) -> Tuple[List[int], float]:
    """Nearest neighbor (from start pose) + optional 2-opt refinement."""
    start_xyz = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)
    order = _nearest_neighbor(points, start_xyz)
    if use_two_opt:
        order = _two_opt(points, order, start_pose, ori_w)
    return order, path_cost(points, order, start_pose, ori_w)

# ---------------------------- Public API -------------------------------------

def plan_tsp_path(start_pose: Pose,
                  waypoints_xyz: Iterable[Iterable[float]],
                  orientation_weights: Optional[Dict[str, float]] = None,
                  exact_threshold: int = 9,
                  use_two_opt: bool = True) -> Tuple[List[int], float]:
    """
    Compute a visiting order for waypoints that minimizes path cost.

    Args:
        start_pose: Pose(x,y,z, roll,pitch,yaw), angles in degrees.
        waypoints_xyz: iterable of (x, y, z).
        orientation_weights: dict with optional keys 'yaw','pitch','roll','turn' (meters per radian).
                             Default is zeros (no orientation penalty).
        exact_threshold: use exact brute force when len(waypoints) <= this value.
        use_two_opt: when heuristic is used, apply 2-opt improvement.

    Returns:
        (order, total_cost)
        - order: list of indices into the original waypoints list, visit order from start.
        - total_cost: total distance + orientation penalties (meters).
    """
    points = _build_points_array(waypoints_xyz)
    n = len(points)

    # Edge cases
    if n == 0:
        return [], 0.0
    if n == 1:
        order = [0]
        cost = path_cost(points, order, start_pose,
                         OrientationWeights(**(orientation_weights or {})))
        return order, cost

    ori_w = OrientationWeights(**(orientation_weights or {}))

    if n <= max(0, exact_threshold):
        return tsp_exact(points, start_pose, ori_w)
    else:
        return tsp_heuristic(points, start_pose, ori_w, use_two_opt=use_two_opt)

# --------------------------- Visualization -----------------------------------

def plot_path_3d(start_pose: Pose,
                 waypoints_xyz: Iterable[Iterable[float]],
                 order: List[int],
                 title: str = "TSP Path (3D)") -> None:
    """Render a 3D plot of the path."""
    pts = _build_points_array(waypoints_xyz)
    p0 = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)

    # Points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40, label="Waypoints")
    ax.scatter([p0[0]], [p0[1]], [p0[2]], s=70, marker='^', label="Start")

    # Path lines
    path_xyz = [p0] + [pts[i] for i in order]
    path_xyz = np.vstack(path_xyz)
    ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], linewidth=2, label="Path")

    # Labels in visiting order
    for step, idx in enumerate(order, start=1):
        ax.text(pts[idx, 0], pts[idx, 1], pts[idx, 2], f"{step}", fontsize=9)

    # Make axes equal-ish
    mins = np.min(np.vstack([pts, p0]), axis=0)
    maxs = np.max(np.vstack([pts, p0]), axis=0)
    ranges = maxs - mins
    center = (maxs + mins) / 2
    r = max(ranges) * 0.6 if np.any(ranges > 0) else 1.0
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

# --------------------------- Example usage -----------------------------------
if __name__ == "__main__":
    # === Your Start Pose ===
    start = Pose(
        x=280.363739,
        y=-129.306351,
        z=0.101746,
        roll=0.0,
        pitch=0.0,
        yaw=180.0
    )

    # === Your Goal Waypoints ===
    waypoints = [
        [334.949799,-161.106171,0.001736],
        [339.100037,-258.568939,0.001679],
        [396.295319,-183.195740,0.001678],
        [267.657074,-1.983160,0.001678],
        [153.868896,-26.115866,0.001678],
        [290.515564,-56.175072,0.001677],
        [92.325722,-86.063644,0.001677],
        [88.384346,-287.468567,0.001728],
        [177.594101,-326.386902,0.001677],
        [-1.646942,-197.501282,0.001555],
        [59.701321,-1.970804,0.001467],
        [122.100121,-55.142044,0.001596],
        [161.030975,-129.313187,0.001679],
        [184.758713,-199.424271,0.001680]
    ]

    # === Orientation weights (optional) ===
    ori_weights = {"yaw": 1.0, "pitch": 0.0, "roll": 0.0, "turn": 0.5}

    # === Solve TSP (heuristic, since > 9 waypoints) ===
    order, total = plan_tsp_path(
        start,
        waypoints,
        orientation_weights=ori_weights,
        exact_threshold=9
    )

    # Print visiting sequence
    print("Optimal visiting order (indices):", order)
    print("Total path cost (m):", round(total, 3))
    print("Visiting waypoints in order:")
    sequence = [waypoints[i] for i in order]
    print(sequence)

    # === 2D plot (top-down view XY) ===
    pts = np.array(waypoints)
    p0 = np.array([start.x, start.y, start.z])

    plt.figure(figsize=(8, 8))
    plt.title("TSP Path (2D Top-down XY)")

    # Plot waypoints
    plt.scatter(pts[:, 0], pts[:, 1], c="blue", s=40, label="Waypoints")
    plt.scatter([p0[0]], [p0[1]], c="red", s=80, marker="^", label="Start")

    # Draw path
    path_xy = np.vstack([p0[:2]] + [pts[i, :2] for i in order])
    plt.plot(path_xy[:, 0], path_xy[:, 1], "k-", linewidth=2, label="Path")

    # Annotate visiting order
    for step, idx in enumerate(order, start=1):
        plt.text(pts[idx, 0], pts[idx, 1], str(step), fontsize=9, color="black")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.show()

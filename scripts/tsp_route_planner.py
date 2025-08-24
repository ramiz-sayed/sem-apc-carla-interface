#!/usr/bin/env python3
"""
TSP route planner for 3D waypoints with:
- Exact & heuristic solvers (nearest-neighbor + optional 2-opt)
- 3D Euclidean distances (primary metric)
- Optional orientation penalties (start heading & turn smoothness)
- 2D & 3D visualization
- Physics-based energy estimation (kWh) using provided parameters
- Clean, readable output of the ordered waypoint list
- Drop-in hook to integrate with an existing CARLA "drive between points" loop

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt

# ============================== Data types ===================================

@dataclass(frozen=True)
class Pose:
    """Start pose (angles in degrees)."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

@dataclass
class OrientationWeights:
    """
    Weights for orientation penalty terms (meters per radian).
    - yaw/pitch/roll: penalty on initial heading mismatch (start pose vs first leg)
      (roll is kept for API symmetry but not used to form a direction vector)
    - turn: penalty on turn angle between consecutive legs (path smoothness)
    """
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    turn: float = 0.0

# ============================== Helpers ======================================

def _to_vec_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit direction vector from yaw (about +Z) and pitch (tilt in X-Z)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
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

def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
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

# ====================== Orientation-aware path cost ==========================

def path_cost(points: np.ndarray,
              order: List[int],
              start_pose: Pose,
              ori_w: OrientationWeights) -> float:
    """
    Cost = sum of 3D distances + optional orientation penalties:
      - Initial heading mismatch (start orientation vs first segment)
      - Turn penalties (angle between consecutive segments)
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
    if ori_w.yaw > 0 or ori_w.pitch > 0 or ori_w.roll > 0:
        start_dir = _to_vec_yaw_pitch(start_pose.yaw, start_pose.pitch)
        seg_dir = _segment_dir(a, b)
        ang = _angle_between(start_dir, seg_dir)  # radians
        # Use the largest of yaw/pitch/roll weights as a simple scalar weight
        cost += ang * max(ori_w.yaw, ori_w.pitch, ori_w.roll)

    # Remaining legs
    for i in range(len(order) - 1):
        a = points[order[i]]
        b = points[order[i + 1]]
        cost += _euclid3(a, b)

    # Turn penalties between consecutive segments (including start->first)
    if ori_w.turn > 0 and len(order) >= 2:
        prev_a, prev_b = p0, points[order[0]]
        prev_dir = _segment_dir(prev_a, prev_b)
        for i in range(len(order) - 1):
            cur_a, cur_b = points[order[i]], points[order[i + 1]]
            cur_dir = _segment_dir(cur_a, cur_b)
            cost += _angle_between(prev_dir, cur_dir) * ori_w.turn
            prev_dir = cur_dir

    return float(cost)

# ============================== Solvers ======================================

def tsp_exact(points: np.ndarray,
              start_pose: Pose,
              ori_w: OrientationWeights) -> Tuple[List[int], float]:
    """Brute force (exact) TSP visiting order (start fixed at start pose). O(n!)."""
    n = len(points)
    best_order: Optional[List[int]] = None
    best_cost = float("inf")
    for order in itertools.permutations(range(n)):
        order = list(order)
        c = path_cost(points, order, start_pose, ori_w)
        if c < best_cost:
            best_cost, best_order = c, order
    return best_order or [], best_cost

def _nearest_neighbor(points: np.ndarray, start_xyz: np.ndarray) -> List[int]:
    """Nearest Neighbor tour seeded from the start pose position."""
    n = len(points)
    if n == 0:
        return []
    remaining = set(range(n))
    d0 = np.linalg.norm(points - start_xyz, axis=1)
    cur = int(np.argmin(d0))
    order = [cur]
    remaining.remove(cur)
    while remaining:
        rem_list = list(remaining)
        d = np.linalg.norm(points[rem_list] - points[cur], axis=1)
        nxt = rem_list[int(np.argmin(d))]
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order

def _two_opt(points: np.ndarray,
             order: List[int],
             start_pose: Pose,
             ori_w: OrientationWeights,
             max_iter: int = 100) -> List[int]:
    """2-opt local refinement."""
    if len(order) < 4:
        return order
    best = order[:]
    best_cost = path_cost(points, best, start_pose, ori_w)
    improved = True
    it = 0
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
    return best

def tsp_heuristic(points: np.ndarray,
                  start_pose: Pose,
                  ori_w: OrientationWeights,
                  use_two_opt: bool = True) -> Tuple[List[int], float]:
    """Nearest Neighbor + optional 2-opt."""
    start_xyz = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)
    order = _nearest_neighbor(points, start_xyz)
    if use_two_opt:
        order = _two_opt(points, order, start_pose, ori_w)
    return order, path_cost(points, order, start_pose, ori_w)

# ============================= Public API ====================================

def plan_tsp_path(start_pose: Pose,
                  waypoints_xyz: Iterable[Iterable[float]],
                  orientation_weights: Optional[Dict[str, float]] = None,
                  exact_threshold: int = 9,
                  use_two_opt: bool = True) -> Tuple[List[int], float]:
    """
    Compute visiting order for waypoints that minimizes the path cost.

    Returns:
        (order, total_cost_m)
        - order: list of indices into the original waypoint list, visit order from start
        - total_cost_m: meters (plus orientation penalty in "meter-equivalents")
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

# ============================ Visualization ==================================

def plot_path_2d(start_pose: Pose,
                 waypoints_xyz: Iterable[Iterable[float]],
                 order: List[int],
                 title: str = "TSP Path (Top-Down)") -> None:
    """2D top-down plot (X/Y), labeling visiting order, and printing sequence."""
    pts = _build_points_array(waypoints_xyz)
    p0 = np.array([start_pose.x, start_pose.y], dtype=float)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    # points
    ax.scatter(pts[:, 0], pts[:, 1], s=40, label="Waypoints")
    ax.scatter([p0[0]], [p0[1]], s=70, marker='^', label="Start")

    # path
    path_xy = np.vstack([p0, pts[order][:, :2]])
    ax.plot(path_xy[:, 0], path_xy[:, 1], linewidth=2, label="Path")

    # numbers on points in visiting order
    for step, idx in enumerate(order, start=1):
        ax.text(pts[idx, 0], pts[idx, 1], f"{step}", fontsize=9)

    # formatting
    all_xy = np.vstack([pts[:, :2], p0])
    mins = np.min(all_xy, axis=0)
    maxs = np.max(all_xy, axis=0)
    pad = 0.05 * (maxs - mins + 1e-6)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_path_3d(start_pose: Pose,
                 waypoints_xyz: Iterable[Iterable[float]],
                 order: List[int],
                 title: str = "TSP Path (3D)") -> None:
    """3D plot."""
    pts = _build_points_array(waypoints_xyz)
    p0 = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40, label="Waypoints")
    ax.scatter([p0[0]], [p0[1]], [p0[2]], s=70, marker='^', label="Start")

    path_xyz = np.vstack([p0, pts[order]])
    ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], linewidth=2, label="Path")

    for step, idx in enumerate(order, start=1):
        ax.text(pts[idx, 0], pts[idx, 1], pts[idx, 2], f"{step}", fontsize=9)

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

# ========================= Energy Estimation =================================

@dataclass
class VehicleParams:
    Cd: float = 0.15000000596       # Drag coefficient
    mass: float = 1845.0            # kg
    g: float = 9.81                  # m/s^2
    rho: float = 1.2                 # kg/m^3 (air density)
    area: float = 2.22               # m^2 (frontal area)
    Crr: float = 0.01                # rolling resistance coefficient

def estimate_energy_kwh(start_pose: Pose,
                        waypoints_xyz: Iterable[Iterable[float]],
                        order: List[int],
                        cruise_speed: float = 5.0,
                        accel: float = 0.0,
                        params: VehicleParams = VehicleParams()) -> Tuple[float, List[float]]:
    """
    Physics-based energy estimate for the route defined by `order`.

    Force model (your formula):
      F = m*g*Crr*cos(theta) + 0.5*rho*Cd*A*v^2 + m*a + m*g*sin(theta)
    Energy per segment: E = F * d  [J]
    Route energy: sum(E) converted to kWh via / 3.6e6

    Notes:
      - theta is computed from segment slope: atan2(dz, horizontal_distance)
      - v is assumed constant = cruise_speed (m/s) per segment
      - a can be set to 0 (cruise), or any constant if you want to bias for accel
    """
    pts = _build_points_array(waypoints_xyz)
    p0 = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)

    energy_j_segments: List[float] = []
    prev = p0
    v = cruise_speed
    a = accel

    for idx in order:
        p = pts[idx]
        vec = p - prev
        d = float(np.linalg.norm(vec))
        if d < 1e-9:
            energy_j_segments.append(0.0)
            prev = p
            continue

        dxdy = math.hypot(vec[0], vec[1])
        dz = vec[2]
        theta = math.atan2(dz, max(1e-9, dxdy))  # slope angle

        # Forces
        rolling = params.mass * params.g * params.Crr * math.cos(theta)
        aero = 0.5 * params.rho * params.Cd * params.area * (v ** 2)
        inertial = params.mass * a
        grade = params.mass * params.g * math.sin(theta)
        F = rolling + aero + inertial + grade

        E = F * d  # Joules
        energy_j_segments.append(E)
        prev = p

    total_kwh = sum(energy_j_segments) / 3.6e6
    return total_kwh, energy_j_segments

# ======================== CARLA Integration Hook =============================

def waypoints_in_order(order: List[int],
                       waypoints_xyz: Iterable[Iterable[float]]) -> List[Tuple[float, float, float]]:
    """Return the ordered list of waypoints (x,y,z) according to TSP order."""
    pts = _build_points_array(waypoints_xyz)
    return [tuple(pts[i]) for i in order]

# Example stub showing how you’d use the ordered points inside your CARLA loop:
#
# def drive_route_in_carla(world, vehicle, ordered_xyz, control_vehicle_fn):
#     """
#     For each (x,y,z) in ordered_xyz:
#         - plan a local path from current pose to that waypoint (your A* code)
#         - follow that path with your control loop (control_vehicle_fn)
#     """
#     for gx, gy, gz in ordered_xyz:
#         # Convert world (x,y) to grid indices, re-run your A* from current grid start to (gx,gy)
#         # Then consume the path using your existing `control_vehicle()` loop.
#         pass

# ============================== Example / CLI ================================

if __name__ == "__main__":
    # --------- Your specific numbers (from your message) ----------
    start = Pose(
        x=280.363739, y=-129.306351, z=0.101746,
        roll=0.0, pitch=0.0, yaw=180.0
    )

    waypoints = [
        [334.949799, -161.106171, 0.001736],
        [339.100037, -258.568939, 0.001679],
        [396.295319, -183.195740, 0.001678],
        [267.657074, -1.983160,   0.001678],
        [153.868896, -26.115866,  0.001678],
        [290.515564, -56.175072,  0.001677],
        [92.325722,  -86.063644,  0.001677],
        [88.384346,  -287.468567, 0.001728],
        [177.594101, -326.386902, 0.001677],
        [-1.646942,  -197.501282, 0.001555],
        [59.701321,  -1.970804,   0.001467],
        [122.100121, -55.142044,  0.001596],
        [161.030975, -129.313187, 0.001679],
        [184.758713, -199.424271, 0.001680],
    ]

    # (Optional) orientation weighting (meters per radian). Set all zeros to ignore orientation.
    ori_weights = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "turn": 0.0}

    # Choose solver automatically (exact if N <= exact_threshold)
    order, total_cost_m = plan_tsp_path(
        start_pose=start,
        waypoints_xyz=waypoints,
        orientation_weights=ori_weights,
        exact_threshold=9,      # exact for <= 9 points; heuristic for more
        use_two_opt=True
    )

    # Print the sequence as requested (from start to end)
    ordered_points = waypoints_in_order(order, waypoints)
    print("\n--- Optimal visit order (indices) ---")
    print(order)
    print("\n--- Ordered waypoints (start -> ... -> end) ---")
    seq = [(start.x, start.y, start.z)] + ordered_points
    for i, p in enumerate(seq):
        print(f"{i:02d}: [{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]")

    print(f"\nTotal path length (m, incl. orientation penalties in meter-equivalents): {total_cost_m:.3f}")

    # Plot (2D top-down). Numbers indicate visiting order.
    plot_path_2d(start, waypoints, order, title="Top-Down TSP Path with Visit Order")

    # Optional 3D plot
    # plot_path_3d(start, waypoints, order, title="3D TSP Path with Visit Order")

    # Energy estimate (simple cruise model)
    kwh, per_seg_j = estimate_energy_kwh(
        start_pose=start,
        waypoints_xyz=waypoints,
        order=order,
        cruise_speed=5.0,     # m/s
        accel=0.0             # assume no net accel across segments for the estimate
    )
    print(f"\nEstimated route energy: {kwh:.6f} kWh\n")
    sequence = [waypoints[i] for i in order]
    print(sequence)

    # === 2D plot (top-down view XY) ===
    pts = np.array(waypoints)
    p0 = np.array([start.x, start.y, start.z])

#!/usr/bin/env python3
"""
Combined TSP Route Planner + A* Pathfinding for CARLA Autonomous Driving

This integrated system provides:
1. TSP optimization for visiting order of 3D waypoints with:
   - Exact & heuristic solvers (nearest-neighbor + optional 2-opt)
   - 3D Euclidean distances (primary metric)
   - Optional orientation penalties (start heading & turn smoothness)
   - Physics-based energy estimation (kWh) using provided parameters

2. A* pathfinding with obstacle avoidance for navigation between waypoints:
   - Real-time obstacle grid generation from CARLA map
   - 8-direction movement with diagonal cost adjustment
   - Improved vehicle control with PID-like behavior
   - Debug visualization of grid, obstacles, and planned path

3. Complete CARLA integration:
   - Automatic vehicle spawning and control
   - Real-time obstacle detection and grid mapping
   - Seamless integration between TSP optimization and A* navigation
   - Clean shutdown and resource management

Author: Combined from TSP planner and A* CARLA navigation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import carla
import random
import time
import queue

# ============================== TSP Data Types ==============================

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

@dataclass
class VehicleParams:
    """Vehicle parameters for energy estimation."""
    Cd: float = 0.15000000596       # Drag coefficient
    mass: float = 1845.0            # kg
    g: float = 9.81                 # m/s^2
    rho: float = 1.2                # kg/m^3 (air density)
    area: float = 2.22              # m^2 (frontal area)
    Crr: float = 0.01               # rolling resistance coefficient

# ============================== CARLA Parameters ==============================

GRID_SIZE = 1.0  # meters per grid cell
TARGET_SPEED = 5.0  # m/s (reduced for precision)
GRID_WIDTH = 200  # cells
GRID_HEIGHT = 200  # cells
OBSTACLE_BUFFER = 0.5  # buffer around obstacles
DEBUG = True  # Enable debugging visualizations

# Color codes for CARLA debug visualization
RED = carla.Color(255, 0, 0)
GREEN = carla.Color(0, 255, 0)
BLUE = carla.Color(0, 0, 255)
YELLOW = carla.Color(255, 255, 0)
WHITE = carla.Color(255, 255, 255)
CYAN = carla.Color(0, 255, 255)
MAGENTA = carla.Color(255, 0, 255)

# Global variables for cleanup
vehicle = None
debug_objects = []

# ============================== TSP Helper Functions ========================

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

# ====================== TSP Orientation-aware Path Cost ===================

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

# ============================== TSP Solvers ===============================

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

# ============================= TSP Public API =============================

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

def waypoints_in_order(order: List[int],
                       waypoints_xyz: Iterable[Iterable[float]]) -> List[Tuple[float, float, float]]:
    """Return the ordered list of waypoints (x,y,z) according to TSP order."""
    pts = _build_points_array(waypoints_xyz)
    return [tuple(pts[i]) for i in order]

# ========================= TSP Energy Estimation ==========================

def estimate_energy_kwh(start_pose: Pose,
                        waypoints_xyz: Iterable[Iterable[float]],
                        order: List[int],
                        cruise_speed: float = 5.0,
                        accel: float = 0.0,
                        params: VehicleParams = VehicleParams()) -> Tuple[float, List[float]]:
    """
    Physics-based energy estimate for the route defined by `order`.

    Force model:
      F = m*g*Crr*cos(theta) + 0.5*rho*Cd*A*v^2 + m*a + m*g*sin(theta)
    Energy per segment: E = F * d  [J]
    Route energy: sum(E) converted to kWh via / 3.6e6
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

# ============================ TSP Visualization ===========================

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

# ===================== A* Pathfinding Functions ===========================

def get_obstacle_grid(world, center_location):
    """Create an occupancy grid based on static obstacles in the map"""
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    
    # Calculate grid origin (top-left corner)
    origin = carla.Location(
        x=center_location.x - (GRID_WIDTH * GRID_SIZE) / 2,
        y=center_location.y - (GRID_HEIGHT * GRID_SIZE) / 2,
        z=0
    )
    
    # Conversion function
    def world_to_grid(location):
        x = int((location.x - origin.x) / GRID_SIZE)
        y = int((location.y - origin.y) / GRID_SIZE)
        return max(0, min(GRID_WIDTH - 1, x)), max(0, min(GRID_HEIGHT - 1, y))
    
    # Get all static obstacles in the world
    obstacle_tags = [
        carla.CityObjectLabel.Buildings,
        carla.CityObjectLabel.Walls,
        carla.CityObjectLabel.Fences,
        carla.CityObjectLabel.Poles,
        carla.CityObjectLabel.Static,
        carla.CityObjectLabel.TrafficSigns,
        carla.CityObjectLabel.Vegetation
    ]
    
    # Add regular obstacles with reduced buffer
    for obj_type in obstacle_tags:
        for obj in world.get_environment_objects(obj_type):
            transform = obj.transform
            bbox = obj.bounding_box
            bbox_ext = carla.Vector3D(
                x=bbox.extent.x + OBSTACLE_BUFFER,
                y=bbox.extent.y + OBSTACLE_BUFFER,
                z=bbox.extent.z
            )
            
            # Get all 8 corners of the bounding box
            corners = [
                transform.transform(carla.Location(x=bbox_ext.x, y=bbox_ext.y)),
                transform.transform(carla.Location(x=-bbox_ext.x, y=bbox_ext.y)),
                transform.transform(carla.Location(x=bbox_ext.x, y=-bbox_ext.y)),
                transform.transform(carla.Location(x=-bbox_ext.x, y=-bbox_ext.y)),
                transform.transform(carla.Location(x=bbox_ext.x, y=bbox_ext.y, z=-bbox_ext.z)),
                transform.transform(carla.Location(x=-bbox_ext.x, y=bbox_ext.y, z=-bbox_ext.z)),
                transform.transform(carla.Location(x=bbox_ext.x, y=-bbox_ext.y, z=-bbox_ext.z)),
                transform.transform(carla.Location(x=-bbox_ext.x, y=-bbox_ext.y, z=-bbox_ext.z))
            ]
            
            # Find grid boundaries for this obstacle
            grid_corners = [world_to_grid(c) for c in corners]
            min_x = min(c[0] for c in grid_corners)
            max_x = max(c[0] for c in grid_corners)
            min_y = min(c[1] for c in grid_corners)
            max_y = max(c[1] for c in grid_corners)
            
            # Mark all cells within these bounds as occupied
            for x in range(max(0, min_x), min(GRID_WIDTH, max_x + 1)):
                for y in range(max(0, min_y), min(GRID_HEIGHT, max_y + 1)):
                    grid[x][y] = 1
                    
    return grid, origin

def a_star(grid, start, goal):
    """Improved A* pathfinding algorithm with 8-direction movement"""
    directions = [(-1,0), (1,0), (0,-1), (0,1),
                 (-1,-1), (-1,1), (1,-1), (1,1)]
    
    # Check if start or goal is blocked
    if grid[start[0]][start[1]] == 1:
        print("Start position is blocked!")
        return []
    if grid[goal[0]][goal[1]] == 1:
        print("Goal position is blocked!")
        return []
    
    # Initialize data structures
    open_set = queue.PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    g_score = {start: 0}
    
    while not open_set.empty():
        current = open_set.get()[1]
        
        # Early exit if we reach the goal
        if current == goal:
            break
            
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check bounds and walkability
            if (0 <= neighbor[0] < len(grid)) and (0 <= neighbor[1] < len(grid[0])):
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
                
                # Diagonal movement costs more
                move_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    open_set.put((f_score, neighbor))
    
    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            print("Path reconstruction failed - no valid path exists")
            return []
    path.reverse()
    return path

def heuristic(a, b):
    """Diagonal distance heuristic"""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

# ===================== Vehicle Control Functions ===========================

def control_vehicle(vehicle, target_loc):
    """Improved vehicle control with PID-like behavior"""
    vehicle_location = vehicle.get_location()
    vehicle_transform = vehicle.get_transform()
    
    # Convert to numpy arrays for easier math
    current_pos = np.array([vehicle_location.x, vehicle_location.y])
    target_pos = np.array([target_loc.x, target_loc.y])
    
    # Calculate direction and distance
    direction = target_pos - current_pos
    distance = np.linalg.norm(direction)
    
    if distance > 0.5:  # Only move if we're reasonably far
        # Normalize direction
        direction /= distance
        
        # Calculate target yaw (convert to degrees)
        target_yaw = math.degrees(math.atan2(direction[1], direction[0]))
        
        # Calculate angle difference (normalized to [-180, 180])
        yaw_diff = (target_yaw - vehicle_transform.rotation.yaw + 180) % 360 - 180
        
        # Calculate steering (more aggressive when farther from target)
        steering = np.clip(yaw_diff / 45.0, -1.0, 1.0)
        
        # Calculate current speed
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        
        # PID-like throttle control
        throttle = np.clip((TARGET_SPEED - speed) / TARGET_SPEED, 0.0, 1.0)
        
        # Apply control
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steering
        control.brake = 0.0 if distance > 2.0 else min(0.5, (2.0 - distance) / 2.0)
        vehicle.apply_control(control)
        
        return distance
    return 0

def navigate_to_waypoint(world, vehicle, start_loc, target_loc, grid, origin):
    """Navigate from start to target using A* pathfinding"""
    # Convert world coordinates to grid coordinates
    def world_to_grid(location):
        x = int((location.x - origin.x) / GRID_SIZE)
        y = int((location.y - origin.y) / GRID_SIZE)
        return max(0, min(GRID_WIDTH - 1, x)), max(0, min(GRID_HEIGHT - 1, y))
    
    start_grid = world_to_grid(start_loc)
    goal_grid = world_to_grid(target_loc)
    
    print(f"Navigating from {start_grid} to {goal_grid}")
    
    # Find path using A*
    path = a_star(grid, start_grid, goal_grid)
    
    if not path:
        print("No path found to waypoint!")
        return False
    
    print(f"Found A* path with {len(path)} nodes")
    
    # Draw debug visualization for this segment
    if DEBUG:
        draw_debug_segment(world, grid, origin, path, start_grid, goal_grid)
    
    # Follow the A* path
    for node in path:
        target_loc_grid = carla.Location(
            x=origin.x + node[0] * GRID_SIZE + GRID_SIZE / 2,
            y=origin.y + node[1] * GRID_SIZE + GRID_SIZE / 2,
            z=0.3
        )
        
        # Keep moving toward this grid point until we're close enough
        while True:
            dist = control_vehicle(vehicle, target_loc_grid)
            time.sleep(0.05)
            if dist < 2.0:  # Close enough to move to next grid point
                break
    
    return True

# ===================== Debug Visualization ==============================

def draw_debug_segment(world, grid, origin, path, start, goal):
    """Draw debug visualization for a single A* path segment"""
    global debug_objects
    
    # Clear previous debug objects for this segment
    for obj in debug_objects[-100:]:  # Only clear recent objects
        try:
            obj.destroy()
        except:
            pass
    debug_objects = debug_objects[:-100] if len(debug_objects) > 100 else []
    
    # Draw path for this segment
    if path:
        for i in range(len(path)-1):
            start_p = carla.Location(
                x=origin.x + path[i][0] * GRID_SIZE + GRID_SIZE / 2,
                y=origin.y + path[i][1] * GRID_SIZE + GRID_SIZE / 2,
                z=0.3
            )
            end_p = carla.Location(
                x=origin.x + path[i+1][0] * GRID_SIZE + GRID_SIZE / 2,
                y=origin.y + path[i+1][1] * GRID_SIZE + GRID_SIZE / 2,
                z=0.3
            )
            line = world.debug.draw_line(
                start_p, 
                end_p, 
                thickness=0.15, 
                color=CYAN, 
                life_time=10.0,
                persistent_lines=False
            )
            debug_objects.append(line)

def draw_tsp_waypoints(world, waypoints_ordered, start_pose):
    """Draw TSP waypoints and connection order"""
    global debug_objects
    
    # Draw start position
    start_loc = carla.Location(x=start_pose.x, y=start_pose.y, z=start_pose.z + 2.0)
    start_point = world.debug.draw_point(
        start_loc, 
        size=0.5, 
        color=BLUE, 
        life_time=60.0,
        persistent_lines=False
    )
    debug_objects.append(start_point)
    
    # Draw waypoints in order with numbers
    prev_loc = start_loc
    for i, (x, y, z) in enumerate(waypoints_ordered):
        wp_loc = carla.Location(x=x, y=y, z=z + 2.0)
        
        # Draw waypoint
        wp_point = world.debug.draw_point(
            wp_loc, 
            size=0.3, 
            color=YELLOW, 
            life_time=60.0,
            persistent_lines=False
        )
        debug_objects.append(wp_point)
        
        # Draw connection from previous waypoint
        line = world.debug.draw_line(
            prev_loc, 
            wp_loc, 
            thickness=0.2, 
            color=MAGENTA, 
            life_time=60.0,
            persistent_lines=False
        )
        debug_objects.append(line)
        
        # Draw number label
        text_loc = carla.Location(x=x, y=y, z=z + 3.0)
        text = world.debug.draw_string(
            text_loc, 
            str(i + 1), 
            draw_shadow=True, 
            color=WHITE, 
            life_time=60.0,
            persistent_lines=False
        )
        debug_objects.append(text)
        
        prev_loc = wp_loc

def draw_debug_full(world, grid, origin, waypoints_ordered, start_pose):
    """Enhanced debug visualization showing both grid and TSP route"""
    global debug_objects
    
    # Clear all previous debug objects
    for obj in debug_objects:
        try:
            obj.destroy()
        except:
            pass
    debug_objects = []
    
    # Draw obstacle grid (sample every 5th cell to reduce clutter)
    for x in range(0, len(grid), 5):
        for y in range(0, len(grid[0]), 5):
            loc = carla.Location(
                x=origin.x + x * GRID_SIZE + GRID_SIZE / 2,
                y=origin.y + y * GRID_SIZE + GRID_SIZE / 2,
                z=0.1
            )
            if grid[x][y] == 1:
                point = world.debug.draw_point(
                    loc, 
                    size=0.05, 
                    color=RED, 
                    life_time=60.0,
                    persistent_lines=False
                )
                debug_objects.append(point)
    
    # Draw TSP waypoints and route
    draw_tsp_waypoints(world, waypoints_ordered, start_pose)

def cleanup():
    """Clean up all created objects"""
    global vehicle, debug_objects
    
    # Destroy vehicle
    if vehicle and vehicle.is_alive:
        try:
            vehicle.destroy()
            print("Vehicle destroyed")
        except:
            pass
    
    # Clear debug objects
    for obj in debug_objects:
        try:
            obj.destroy()
        except:
            pass
    debug_objects = []

# ===================== Main Integration Function ========================

def run_combined_planner(world, waypoints, start_pose_dict, orientation_weights=None):
    """
    Main function that combines TSP optimization with A* navigation in CARLA
    
    Args:
        world: CARLA world object
        waypoints: List of [x, y, z] waypoint coordinates
        start_pose_dict: Dict with keys x, y, z, roll, pitch, yaw
        orientation_weights: Optional dict for TSP orientation penalties
    
    Returns:
        Tuple of (success: bool, total_distance: float, energy_kwh: float)
    """
    global vehicle
    
    try:
        # Create Pose object from dictionary
        start_pose = Pose(**start_pose_dict)
        
        print("=== TSP Route Optimization ===")
        
        # 1. Optimize waypoint visiting order using TSP
        order, total_cost_m = plan_tsp_path(
            start_pose=start_pose,
            waypoints_xyz=waypoints,
            orientation_weights=orientation_weights,
            exact_threshold=9,
            use_two_opt=True
        )
        
        # Get ordered waypoints
        ordered_waypoints = waypoints_in_order(order, waypoints)
        
        print(f"Optimal visit order (indices): {order}")
        print(f"Total optimized path length: {total_cost_m:.3f} meters")
        
        # Print the sequence
        print("\n--- Optimized waypoint sequence ---")
        sequence = [(start_pose.x, start_pose.y, start_pose.z)] + ordered_waypoints
        for i, p in enumerate(sequence):
            print(f"{i:02d}: [{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]")
        
        # 2. Energy estimation
        kwh, per_seg_j = estimate_energy_kwh(
            start_pose=start_pose,
            waypoints_xyz=waypoints,
            order=order,
            cruise_speed=TARGET_SPEED,
            accel=0.0
        )
        print(f"\nEstimated route energy: {kwh:.6f} kWh")
        
        # 3. Spawn vehicle in CARLA
        print("\n=== CARLA Vehicle Setup ===")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        
        spawn_transform = carla.Transform(
            carla.Location(x=start_pose.x, y=start_pose.y, z=start_pose.z),
            carla.Rotation(yaw=start_pose.yaw, pitch=start_pose.pitch, roll=start_pose.roll)
        )
        
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
        if not vehicle:
            print("Failed to spawn vehicle!")
            return False, 0.0, 0.0
            
        print(f"Vehicle spawned at {spawn_transform.location}")
        
        # Set spectator view
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            spawn_transform.location + carla.Location(z=50),
            carla.Rotation(pitch=-90)
        ))
        
        # 4. Generate obstacle grid
        print("\n=== Obstacle Grid Generation ===")
        current_loc = vehicle.get_location()
        grid, origin = get_obstacle_grid(world, current_loc)
        print(f"Generated {GRID_WIDTH}x{GRID_HEIGHT} obstacle grid")
        
        # 5. Draw debug visualization
        if DEBUG:
            print("Drawing debug visualization...")
            draw_debug_full(world, grid, origin, ordered_waypoints, start_pose)
        
        # 6. Navigate to each waypoint using A*
        print("\n=== A* Navigation Execution ===")
        success_count = 0
        
        for i, (wx, wy, wz) in enumerate(ordered_waypoints):
            print(f"\nNavigating to waypoint {i+1}/{len(ordered_waypoints)}: ({wx:.2f}, {wy:.2f}, {wz:.2f})")
            
            current_location = vehicle.get_location()
            target_location = carla.Location(x=wx, y=wy, z=wz)
            
            # Navigate using A* pathfinding
            success = navigate_to_waypoint(
                world, vehicle, current_location, target_location, grid, origin
            )
            
            if success:
                success_count += 1
                print(f"Successfully reached waypoint {i+1}")
                
                # Brief pause at each waypoint
                time.sleep(0.5)
            else:
                print(f"Failed to reach waypoint {i+1}")
        
        # 7. Final stop
        print("\n=== Mission Complete ===")
        control = carla.VehicleControl()
        control.brake = 1.0
        vehicle.apply_control(control)
        time.sleep(1)
        
        success_rate = success_count / len(ordered_waypoints) if ordered_waypoints else 0
        print(f"Successfully reached {success_count}/{len(ordered_waypoints)} waypoints ({success_rate*100:.1f}%)")
        print(f"Total planned distance: {total_cost_m:.3f} meters")
        print(f"Estimated energy consumption: {kwh:.6f} kWh")
        
        return success_rate > 0.5, total_cost_m, kwh
        
    except Exception as e:
        print(f"Error in combined planner: {str(e)}")
        return False, 0.0, 0.0

# ===================== Example Usage and Main ===========================

def main():
    """Main function demonstrating the combined TSP + A* system"""
    global vehicle
    
    try:
        # Connect to CARLA
        print("Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        client.load_world('Town01')
        world = client.get_world()
        
        # Example waypoints (from your original data)
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
        
        # Starting pose (from your original data)
        start_pose = {
            'x': 280.363739,
            'y': -129.306351,
            'z': 0.101746,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 180.0
        }
        
        # Optional orientation weights (set to 0 to ignore orientation penalties)
        orientation_weights = {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "turn": 0.0
        }
        
        # Run the combined planner
        success, distance, energy = run_combined_planner(
            world=world,
            waypoints=waypoints,
            start_pose_dict=start_pose,
            orientation_weights=orientation_weights
        )
        
        if success:
            print(f"\n✅ Mission completed successfully!")
            print(f"📏 Total distance: {distance:.3f} meters")
            print(f"⚡ Energy consumption: {energy:.6f} kWh")
        else:
            print(f"\n❌ Mission failed or partially completed")
            
        # Optional: Generate matplotlib plots for analysis
        print("\nGenerating analysis plots...")
        start_pose_obj = Pose(**start_pose)
        
        # Get optimized order for plotting
        order, _ = plan_tsp_path(
            start_pose=start_pose_obj,
            waypoints_xyz=waypoints,
            orientation_weights=orientation_weights
        )
        
        # Create 2D visualization
        plot_path_2d(start_pose_obj, waypoints, order, 
                    title="Optimized TSP Route (executed in CARLA)")
        
        # Uncomment for 3D plot
        # plot_path_3d(start_pose_obj, waypoints, order, 
        #             title="3D TSP Route Visualization")
        
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
    finally:
        cleanup()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    # Alternative simplified usage example
    print("="*60)
    print("🚗 Combined TSP + A* CARLA Path Planner")
    print("="*60)
    print("\nThis system provides:")
    print("1. 📊 TSP optimization for waypoint visiting order")
    print("2. 🧭 A* pathfinding for obstacle avoidance")  
    print("3. 🎮 Full CARLA integration with vehicle control")
    print("4. 📈 Energy estimation and route analysis")
    print("5. 🔍 Debug visualization and monitoring")
    print("\nStarting main execution...")
    print("="*60)
    
    main()
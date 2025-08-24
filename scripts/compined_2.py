#!/usr/bin/env python3
"""
🚗 CARLA Advanced Pathfinding System with TSP Route Optimization
==============================================================

This comprehensive system combines:
1. TSP (Traveling Salesman Problem) route optimization for multiple waypoints
2. A* pathfinding for obstacle avoidance between waypoints
3. Real-time CARLA simulation with vehicle control
4. Advanced visualization and energy estimation
5. Physics-based vehicle dynamics

Features:
- Exact & heuristic TSP solvers (nearest-neighbor + optional 2-opt)
- 3D Euclidean distances with optional orientation penalties
- Real-time obstacle detection from CARLA world
- Smooth vehicle control with PID-like behavior
- Comprehensive visualization (2D/3D plots + CARLA debug)
- Energy consumption estimation
- Robust error handling and cleanup

Author: Advanced CARLA Navigation System
Version: 2.0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict, Union
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import carla
import random
import time
import queue
import threading
import logging

# ============================== CONFIGURATION ==============================

# TSP Configuration
@dataclass(frozen=True)
class Pose:
    """Start pose with position and orientation (angles in degrees)."""
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

@dataclass
class OrientationWeights:
    """
    Weights for orientation penalty terms (meters per radian).
    - yaw/pitch/roll: penalty on initial heading mismatch
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

# CARLA Pathfinding Configuration
GRID_SIZE = 1.0                     # meters per grid cell
TARGET_SPEED = 6.0                  # m/s vehicle speed
GRID_WIDTH = 200                    # grid cells width
GRID_HEIGHT = 200                   # grid cells height
OBSTACLE_BUFFER = 0.5               # buffer around obstacles (meters)
WAYPOINT_TOLERANCE = 1.8            # waypoint reach tolerance (meters)
RRT_MAX_ITER = 8000                 # maximum RRT iterations
RRT_STEP_SIZE = 3.0                 # RRT step size (grid cells)

# Visualization Colors
RED = carla.Color(255, 0, 0)        # Obstacles
GREEN = carla.Color(0, 255, 0)      # Path segments
BLUE = carla.Color(0, 0, 255)       # Start position
YELLOW = carla.Color(255, 255, 0)   # Goal/waypoints
WHITE = carla.Color(255, 255, 255)  # Grid
PURPLE = carla.Color(255, 0, 255)   # TSP waypoints
CYAN = carla.Color(0, 255, 255)     # Current target

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================== TSP SOLVER ==============================

def _to_vec_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Convert yaw/pitch angles to unit direction vector."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    vx = math.cos(pitch) * math.cos(yaw)
    vy = math.cos(pitch) * math.sin(yaw)
    vz = math.sin(pitch)
    v = np.array([vx, vy, vz], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])

def _segment_dir(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit direction vector from point a to point b."""
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])

def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate unsigned angle (radians) between two unit vectors."""
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.acos(dot)

def _euclid3(a: np.ndarray, b: np.ndarray) -> float:
    """3D Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))

def _build_points_array(waypoints_xyz: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert waypoints to numpy array with validation."""
    arr = np.asarray(waypoints_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("waypoints must be an iterable of (x,y,z)")
    return arr

def path_cost(points: np.ndarray,
              order: List[int],
              start_pose: Pose,
              ori_w: OrientationWeights) -> float:
    """
    Calculate total path cost including 3D distances and orientation penalties.
    
    Cost = sum of 3D distances + optional orientation penalties:
      - Initial heading mismatch (start orientation vs first segment)
      - Turn penalties (angle between consecutive segments)
    """
    if not order:
        return 0.0

    p0 = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)
    cost = 0.0

    # First leg cost
    a = p0
    b = points[order[0]]
    cost += _euclid3(a, b)

    # Orientation penalty for first leg
    if ori_w.yaw > 0 or ori_w.pitch > 0 or ori_w.roll > 0:
        start_dir = _to_vec_yaw_pitch(start_pose.yaw, start_pose.pitch)
        seg_dir = _segment_dir(a, b)
        ang = _angle_between(start_dir, seg_dir)
        cost += ang * max(ori_w.yaw, ori_w.pitch, ori_w.roll)

    # Remaining legs
    for i in range(len(order) - 1):
        a = points[order[i]]
        b = points[order[i + 1]]
        cost += _euclid3(a, b)

    # Turn penalties between consecutive segments
    if ori_w.turn > 0 and len(order) >= 2:
        prev_a, prev_b = p0, points[order[0]]
        prev_dir = _segment_dir(prev_a, prev_b)
        for i in range(len(order) - 1):
            cur_a, cur_b = points[order[i]], points[order[i + 1]]
            cur_dir = _segment_dir(cur_a, cur_b)
            cost += _angle_between(prev_dir, cur_dir) * ori_w.turn
            prev_dir = cur_dir

    return float(cost)

def tsp_exact(points: np.ndarray,
              start_pose: Pose,
              ori_w: OrientationWeights) -> Tuple[List[int], float]:
    """Exact TSP solver using brute force (O(n!) - use for small problems)."""
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
    """Nearest Neighbor heuristic for TSP."""
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
    """2-opt local optimization for TSP."""
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
    """Heuristic TSP solver: Nearest Neighbor + optional 2-opt."""
    start_xyz = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)
    order = _nearest_neighbor(points, start_xyz)
    
    if use_two_opt:
        order = _two_opt(points, order, start_pose, ori_w)
    
    return order, path_cost(points, order, start_pose, ori_w)

def plan_tsp_path(start_pose: Pose,
                  waypoints_xyz: Iterable[Iterable[float]],
                  orientation_weights: Optional[Dict[str, float]] = None,
                  exact_threshold: int = 9,
                  use_two_opt: bool = True) -> Tuple[List[int], float]:
    """
    Main TSP solver function.
    
    Args:
        start_pose: Starting position and orientation
        waypoints_xyz: List of (x,y,z) waypoints to visit
        orientation_weights: Optional orientation penalty weights
        exact_threshold: Use exact solver if waypoints <= this number
        use_two_opt: Apply 2-opt optimization to heuristic solution
    
    Returns:
        (order, total_cost): Optimal visit order and total path cost
    """
    points = _build_points_array(waypoints_xyz)
    n = len(points)

    # Handle edge cases
    if n == 0:
        return [], 0.0
    if n == 1:
        order = [0]
        cost = path_cost(points, order, start_pose,
                         OrientationWeights(**(orientation_weights or {})))
        return order, cost

    ori_w = OrientationWeights(**(orientation_weights or {}))
    
    # Choose solver based on problem size
    if n <= max(0, exact_threshold):
        logger.info(f"Using exact TSP solver for {n} waypoints")
        return tsp_exact(points, start_pose, ori_w)
    else:
        logger.info(f"Using heuristic TSP solver for {n} waypoints")
        return tsp_heuristic(points, start_pose, ori_w, use_two_opt=use_two_opt)

# ============================== ENERGY ESTIMATION ==============================

def estimate_energy_kwh(start_pose: Pose,
                        waypoints_xyz: Iterable[Iterable[float]],
                        order: List[int],
                        cruise_speed: float = 5.0,
                        accel: float = 0.0,
                        params: VehicleParams = VehicleParams()) -> Tuple[float, List[float]]:
    """
    Physics-based energy estimation for the planned route.
    
    Force model: F = m*g*Crr*cos(theta) + 0.5*rho*Cd*A*v^2 + m*a + m*g*sin(theta)
    Energy per segment: E = F * d [Joules]
    Total energy converted to kWh
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

        # Calculate slope angle
        dxdy = math.hypot(vec[0], vec[1])
        dz = vec[2]
        theta = math.atan2(dz, max(1e-9, dxdy))

        # Calculate forces
        rolling = params.mass * params.g * params.Crr * math.cos(theta)
        aero = 0.5 * params.rho * params.Cd * params.area * (v ** 2)
        inertial = params.mass * a
        grade = params.mass * params.g * math.sin(theta)
        F = rolling + aero + inertial + grade

        # Calculate energy for this segment
        E = F * d  # Joules
        energy_j_segments.append(E)
        prev = p

    total_kwh = sum(energy_j_segments) / 3.6e6
    return total_kwh, energy_j_segments

# ============================== CARLA PATHFINDING ==============================

class CarlaAdvancedPathPlanner:
    """
    Advanced CARLA pathfinding system with TSP optimization.
    
    This class combines TSP route optimization with A* pathfinding
    to create optimal routes through multiple waypoints while
    avoiding obstacles in the CARLA simulation environment.
    """
    
    def __init__(self):
        """Initialize the pathfinding system with default parameters."""
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None
        
        # Pathfinding parameters
        self.GRID_SIZE = GRID_SIZE
        self.TARGET_SPEED = TARGET_SPEED
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.OBSTACLE_BUFFER = OBSTACLE_BUFFER
        self.WAYPOINT_TOLERANCE = WAYPOINT_TOLERANCE
        self.RRT_MAX_ITER = RRT_MAX_ITER
        self.RRT_STEP_SIZE = RRT_STEP_SIZE
        
        # Control and state
        self.path_execution_active = False
        self.debug_objects = []
        
        logger.info("🚗 CARLA Advanced Pathfinding System initialized")

    def get_obstacle_grid(self, center_location):
        """
        Create detailed occupancy grid from CARLA world static obstacles.
        
        Args:
            center_location: Center point for grid generation
            
        Returns:
            (grid, origin): 2D occupancy grid and world origin point
        """
        logger.info("🗺️ Generating obstacle grid from CARLA world...")
        
        # Initialize empty grid
        grid = [[0 for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        
        # Calculate grid origin (top-left corner)
        origin = carla.Location(
            x=center_location.x - (self.GRID_WIDTH * self.GRID_SIZE) / 2,
            y=center_location.y - (self.GRID_HEIGHT * self.GRID_SIZE) / 2,
            z=0
        )
        
        def world_to_grid(location):
            """Convert world coordinates to grid indices."""
            x = int((location.x - origin.x) / self.GRID_SIZE)
            y = int((location.y - origin.y) / self.GRID_SIZE)
            return max(0, min(self.GRID_WIDTH - 1, x)), max(0, min(self.GRID_HEIGHT - 1, y))
        
        # Define obstacle types to detect
        obstacle_tags = [
            carla.CityObjectLabel.Buildings,
            carla.CityObjectLabel.Walls,
            carla.CityObjectLabel.Fences,
            carla.CityObjectLabel.Poles,
            carla.CityObjectLabel.Static,
            carla.CityObjectLabel.TrafficSigns,
            carla.CityObjectLabel.Vegetation
        ]
        
        # Process each obstacle type
        obstacle_count = 0
        for obj_type in obstacle_tags:
            for obj in self.world.get_environment_objects(obj_type):
                obstacle_count += 1
                transform = obj.transform
                bbox = obj.bounding_box
                
                # Expand bounding box with buffer
                bbox_ext = carla.Vector3D(
                    x=bbox.extent.x + self.OBSTACLE_BUFFER,
                    y=bbox.extent.y + self.OBSTACLE_BUFFER,
                    z=bbox.extent.z
                )
                
                # Get all 8 corners of the expanded bounding box
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
                
                # Mark all cells within bounds as occupied
                for x in range(max(0, min_x), min(self.GRID_WIDTH, max_x + 1)):
                    for y in range(max(0, min_y), min(self.GRID_HEIGHT, max_y + 1)):
                        grid[x][y] = 1
        
        logger.info(f"✓ Processed {obstacle_count} obstacles")
        occupied_cells = sum(sum(row) for row in grid)
        total_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        occupancy_rate = (occupied_cells / total_cells) * 100
        logger.info(f"Grid occupancy: {occupied_cells}/{total_cells} cells ({occupancy_rate:.1f}%)")
        
        return grid, origin

    def distance(self, a, b):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def rrt_path_plan(self, start, goal, grid):
        """
        RRT (Rapidly-exploring Random Tree) pathfinding algorithm.
        
        More robust than A* for complex environments with many obstacles.
        
        Args:
            start: Start position (grid coordinates)
            goal: Goal position (grid coordinates)
            grid: 2D occupancy grid (0=free, 1=occupied)
            
        Returns:
            List of waypoints from start to goal, or empty list if no path found
        """
        logger.info(f"🧭 RRT pathfinding: {start} → {goal}")
        
        # Validate start and goal
        if grid[start[0]][start[1]] == 1:
            logger.error("Start position is occupied!")
            return []
        if grid[goal[0]][goal[1]] == 1:
            logger.error("Goal position is occupied!")
            return []
        
        # RRT data structures
        nodes = [start]
        parent = {start: None}
        
        def get_random_point():
            """Generate random point with goal bias."""
            if random.random() < 0.1:  # 10% goal bias
                return goal
            return (random.randint(0, self.GRID_WIDTH-1), 
                   random.randint(0, self.GRID_HEIGHT-1))
        
        def find_nearest(point):
            """Find nearest node in tree to given point."""
            min_dist = float('inf')
            nearest = start
            for node in nodes:
                dist = self.distance(node, point)
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            return nearest
        
        def steer(from_node, to_point):
            """Steer from node towards point with step size limit."""
            dist = self.distance(from_node, to_point)
            if dist <= self.RRT_STEP_SIZE:
                return to_point
            
            # Calculate unit direction vector
            dx = to_point[0] - from_node[0]
            dy = to_point[1] - from_node[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length == 0:
                return from_node
            
            # Step in direction with limited step size
            new_x = from_node[0] + (dx / length) * self.RRT_STEP_SIZE
            new_y = from_node[1] + (dy / length) * self.RRT_STEP_SIZE
            
            return (int(new_x), int(new_y))
        
        def is_collision_free(from_node, to_node):
            """Check if path between nodes is collision-free."""
            # Simple line collision check
            x0, y0 = from_node
            x1, y1 = to_node
            
            # Bresenham's line algorithm for collision checking
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            
            x, y = x0, y0
            while True:
                # Check bounds and collision
                if (x < 0 or x >= self.GRID_WIDTH or 
                    y < 0 or y >= self.GRID_HEIGHT or 
                    grid[x][y] == 1):
                    return False
                
                if x == x1 and y == y1:
                    break
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
            
            return True
        
        # Main RRT loop
        for iteration in range(self.RRT_MAX_ITER):
            # Progress logging
            if iteration % 1000 == 0 and iteration > 0:
                logger.info(f"RRT iteration {iteration}/{self.RRT_MAX_ITER}")
            
            # Sample random point
            rand_point = get_random_point()
            
            # Find nearest node
            nearest_node = find_nearest(rand_point)
            
            # Steer towards random point
            new_node = steer(nearest_node, rand_point)
            
            # Check if new node is valid and collision-free
            if (0 <= new_node[0] < self.GRID_WIDTH and 
                0 <= new_node[1] < self.GRID_HEIGHT and
                grid[new_node[0]][new_node[1]] == 0 and
                is_collision_free(nearest_node, new_node)):
                
                # Add new node to tree
                nodes.append(new_node)
                parent[new_node] = nearest_node
                
                # Check if we reached the goal
                if self.distance(new_node, goal) <= self.RRT_STEP_SIZE:
                    if is_collision_free(new_node, goal):
                        # Path found! Reconstruct path
                        parent[goal] = new_node
                        path = []
                        current = goal
                        
                        while current is not None:
                            path.append(current)
                            current = parent[current]
                        
                        path.reverse()
                        logger.info(f"✓ RRT path found in {iteration+1} iterations")
                        logger.info(f"Path length: {len(path)} waypoints")
                        return path[1:]  # Exclude start point
        
        logger.error(f"❌ RRT failed to find path after {self.RRT_MAX_ITER} iterations")
        return []

    def control_vehicle(self, target_loc):
        """
        Advanced PID-based vehicle control system.
        
        Args:
            target_loc: Target location (carla.Location)
            
        Returns:
            Distance to target
        """
        if not self.vehicle or not self.vehicle.is_alive:
            return float('inf')
        
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        # Calculate direction and distance to target
        current_pos = np.array([vehicle_location.x, vehicle_location.y])
        target_pos = np.array([target_loc.x, target_loc.y])
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # Very close to target
            return distance
        
        # Normalize direction vector
        direction /= distance
        
        # Calculate target yaw angle
        target_yaw = math.degrees(math.atan2(direction[1], direction[0]))
        
        # Calculate steering angle (normalized to [-180, 180])
        current_yaw = vehicle_transform.rotation.yaw
        yaw_diff = (target_yaw - current_yaw + 180) % 360 - 180
        
        # PID-like steering control
        steering = np.clip(yaw_diff / 30.0, -1.0, 1.0)
        
        # Speed control based on distance and steering angle
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
        
        # Reduce speed when turning or close to target
        speed_factor = 1.0
        if abs(yaw_diff) > 15:  # Turning
            speed_factor = 0.7
        if distance < 5.0:  # Close to target
            speed_factor *= (distance / 5.0)
        
        target_speed = self.TARGET_SPEED * speed_factor
        speed_error = target_speed - current_speed
        
        # Throttle/brake control
        if speed_error > 0:
            throttle = min(1.0, speed_error / self.TARGET_SPEED)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, abs(speed_error) / self.TARGET_SPEED)
        
        # Apply control
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steering
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False
        
        self.vehicle.apply_control(control)
        return distance

    def execute_path(self, path, origin):
        """
        Execute planned path with advanced vehicle control.
        
        Args:
            path: List of grid coordinates representing the path
            origin: World origin point for coordinate conversion
        """
        logger.info(f"🚗 Executing path with {len(path)} waypoints")
        self.path_execution_active = True
        
        waypoint_count = 0
        total_waypoints = len(path)
        
        try:
            for i, node in enumerate(path):
                if not self.path_execution_active:
                    break
                
                # Convert grid coordinates to world coordinates
                target_loc = carla.Location(
                    x=origin.x + node[0] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    y=origin.y + node[1] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    z=0.3
                )
                
                waypoint_count += 1
                logger.info(f"📍 Navigating to waypoint {waypoint_count}/{total_waypoints}")
                
                # Draw current target
                if self.world:
                    debug_point = self.world.debug.draw_point(
                        target_loc, size=0.3, color=CYAN, life_time=5.0
                    )
                    self.debug_objects.append(debug_point)
                
                # Navigate to waypoint
                start_time = time.time()
                timeout = 30.0  # 30 second timeout per waypoint
                stuck_threshold = 0.1  # m/s minimum speed
                stuck_time = 0.0
                last_position = self.vehicle.get_location()
                
                while self.path_execution_active:
                    distance = self.control_vehicle(target_loc)
                    
                    # Check if waypoint reached
                    if distance < self.WAYPOINT_TOLERANCE:
                        logger.info(f"✓ Waypoint {waypoint_count} reached (distance: {distance:.2f}m)")
                        break
                    
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        logger.warning(f"⚠️ Timeout reaching waypoint {waypoint_count}")
                        break
                    
                    # Check if vehicle is stuck
                    current_position = self.vehicle.get_location()
                    movement = math.sqrt(
                        (current_position.x - last_position.x)**2 + 
                        (current_position.y - last_position.y)**2
                    )
                    
                    if movement < stuck_threshold:
                        stuck_time += 0.1
                        if stuck_time > 5.0:  # Stuck for 5 seconds
                            logger.warning(f"⚠️ Vehicle appears stuck at waypoint {waypoint_count}")
                            # Try to unstuck by backing up slightly
                            unstuck_control = carla.VehicleControl()
                            unstuck_control.throttle = 0.3
                            unstuck_control.steer = random.uniform(-0.5, 0.5)
                            unstuck_control.reverse = True
                            self.vehicle.apply_control(unstuck_control)
                            time.sleep(1.0)
                            stuck_time = 0.0
                    else:
                        stuck_time = 0.0
                    
                    last_position = current_position
                    time.sleep(0.1)  # 10 Hz control loop
                
        except Exception as e:
            logger.error(f"Error during path execution: {e}")
        finally:
            # Stop vehicle
            if self.vehicle and self.vehicle.is_alive:
                stop_control = carla.VehicleControl()
                stop_control.brake = 1.0
                stop_control.throttle = 0.0
                self.vehicle.apply_control(stop_control)
            
            self.path_execution_active = False
            logger.info("🏁 Path execution completed")

    def draw_debug_visualization(self, grid, origin, path):
        """
        Create comprehensive debug visualization in CARLA.
        
        Args:
            grid: 2D occupancy grid
            origin: World origin point
            path: Planned path as list of grid coordinates
        """
        logger.info("🎨 Drawing debug visualization...")
        
        # Clear previous debug objects
        self.cleanup_debug_objects()
        
        # Draw grid (sample points to avoid performance issues)
        grid_sample_rate = 5  # Draw every 5th grid cell
        for x in range(0, len(grid), grid_sample_rate):
            for y in range(0, len(grid[0]), grid_sample_rate):
                loc = carla.Location(
                    x=origin.x + x * self.GRID_SIZE + self.GRID_SIZE / 2,
                    y=origin.y + y * self.GRID_SIZE + self.GRID_SIZE / 2,
                    z=0.1
                )
                
                if grid[x][y] == 1:
                    # Draw obstacles as red points
                    point = self.world.debug.draw_point(
                        loc, size=0.1, color=RED, life_time=60.0
                    )
                    self.debug_objects.append(point)
        
        # Draw path as connected line segments
        if path:
            logger.info(f"Drawing path with {len(path)} segments")
            for i in range(len(path) - 1):
                start_p = carla.Location(
                    x=origin.x + path[i][0] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    y=origin.y + path[i][1] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    z=0.5
                )
                end_p = carla.Location(
                    x=origin.x + path[i+1][0] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    y=origin.y + path[i+1][1] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    z=0.5
                )
                
                line = self.world.debug.draw_line(
                    start_p, end_p, thickness=0.1, color=GREEN, life_time=60.0
                )
                self.debug_objects.append(line)
            
            # Draw waypoints as numbered points
            for i, waypoint in enumerate(path):
                loc = carla.Location(
                    x=origin.x + waypoint[0] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    y=origin.y + waypoint[1] * self.GRID_SIZE + self.GRID_SIZE / 2,
                    z=0.7
                )
                
                color = YELLOW if i < len(path) - 1 else PURPLE  # Last waypoint in purple
                point = self.world.debug.draw_point(
                    loc, size=0.2, color=color, life_time=60.0
                )
                self.debug_objects.append(point)
        
        logger.info(f"✓ Debug visualization complete ({len(self.debug_objects)} objects)")

    def cleanup_debug_objects(self):
        """Clean up all debug visualization objects."""
        for obj in self.debug_objects:
            try:
                if hasattr(obj, 'destroy'):
                    obj.destroy()
            except:
                pass
        self.debug_objects = []

    def navigate_to_waypoints(self, start_pose: Pose, waypoints_xyz: List[List[float]], 
                            orientation_weights: Optional[Dict[str, float]] = None):
        """
        Complete navigation system: TSP optimization + pathfinding + execution.
        
        Args:
            start_pose: Starting position and orientation
            waypoints_xyz: List of [x, y, z] waypoints to visit
            orientation_weights: Optional orientation penalty weights
        """
        try:
            logger.info("🎯 Starting complete navigation system")
            logger.info(f"Start pose: ({start_pose.x:.1f}, {start_pose.y:.1f}, {start_pose.z:.1f})")
            logger.info(f"Waypoints to visit: {len(waypoints_xyz)}")
            
            # === TSP OPTIMIZATION ===
            logger.info("🧮 Optimizing waypoint visit order with TSP...")
            tsp_start_time = time.time()
            
            order, total_cost = plan_tsp_path(
                start_pose=start_pose,
                waypoints_xyz=waypoints_xyz,
                orientation_weights=orientation_weights,
                exact_threshold=9,
                use_two_opt=True
            )
            
            tsp_time = time.time() - tsp_start_time
            logger.info(f"✓ TSP optimization completed in {tsp_time:.2f} seconds")
            logger.info(f"Optimal visit order: {order}")
            logger.info(f"Total route cost: {total_cost:.2f} meters")
            
            # Get ordered waypoints
            ordered_waypoints = [waypoints_xyz[i] for i in order]
            
            # === ENERGY ESTIMATION ===
            kwh, per_seg_j = estimate_energy_kwh(
                start_pose=start_pose,
                waypoints_xyz=waypoints_xyz,
                order=order,
                cruise_speed=self.TARGET_SPEED,
                accel=0.0
            )
            logger.info(f"⚡ Estimated energy consumption: {kwh:.4f} kWh")
            
            # === OBSTACLE GRID GENERATION ===
            logger.info("🗺️ Generating obstacle map...")
            vehicle_location = self.vehicle.get_location()
            grid, origin = self.get_obstacle_grid(vehicle_location)
            
            # === NAVIGATE TO EACH WAYPOINT ===
            current_pose = start_pose
            total_waypoints = len(ordered_waypoints)
            
            for waypoint_idx, target_waypoint in enumerate(ordered_waypoints):
                logger.info(f"🎯 Navigating to waypoint {waypoint_idx + 1}/{total_waypoints}")
                logger.info(f"Target: ({target_waypoint[0]:.1f}, {target_waypoint[1]:.1f}, {target_waypoint[2]:.1f})")
                
                # Convert current position to grid coordinates
                current_loc = self.vehicle.get_location()
                start_x = int((current_loc.x - origin.x) / self.GRID_SIZE)
                start_y = int((current_loc.y - origin.y) / self.GRID_SIZE)
                start_grid = (
                    max(0, min(self.GRID_WIDTH - 1, start_x)),
                    max(0, min(self.GRID_HEIGHT - 1, start_y))
                )
                
                # Convert target to grid coordinates
                goal_x = int((target_waypoint[0] - origin.x) / self.GRID_SIZE)
                goal_y = int((target_waypoint[1] - origin.y) / self.GRID_SIZE)
                goal_grid = (
                    max(0, min(self.GRID_WIDTH - 1, goal_x)),
                    max(0, min(self.GRID_HEIGHT - 1, goal_y))
                )
                
                logger.info(f"Grid path: {start_grid} → {goal_grid}")
                
                # Plan path using RRT
                path = self.rrt_path_plan(start_grid, goal_grid, grid)
                
                if not path:
                    logger.error(f"❌ Failed to find path to waypoint {waypoint_idx + 1}")
                    continue
                
                # Add start position to path
                full_path = [start_grid] + path
                
                # Visualize this segment
                self.draw_debug_visualization(grid, origin, full_path)
                
                # Execute path to this waypoint
                logger.info(f"🚗 Executing path to waypoint {waypoint_idx + 1}")
                self.execute_path(path, origin)
                
                # Update current pose for next iteration
                final_loc = self.vehicle.get_location()
                current_pose = Pose(
                    x=final_loc.x, y=final_loc.y, z=final_loc.z,
                    yaw=self.vehicle.get_transform().rotation.yaw
                )
                
                logger.info(f"✓ Reached waypoint {waypoint_idx + 1}")
                time.sleep(1.0)  # Brief pause between waypoints
            
            logger.info("🎉 All waypoints reached successfully!")
            
        except Exception as e:
            logger.error(f"❌ Navigation error: {e}", exc_info=True)

    def run_complete_system(self):
        """
        Run the complete CARLA pathfinding system with TSP optimization.
        
        This is the main entry point that demonstrates the full system capabilities.
        """
        try:
            logger.info("🚀 Starting CARLA Advanced Pathfinding System")
            
            # === CARLA CONNECTION ===
            logger.info("🔌 Connecting to CARLA server...")
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(30.0)
            
            version = self.client.get_server_version()
            logger.info(f"✓ Connected to CARLA server version: {version}")
            
            # === WORLD SETUP ===
            logger.info("🌍 Loading CARLA world...")
            self.world = self.client.load_world('Town03')
            logger.info("✓ World 'Town03' loaded successfully")
            
            # Configure world settings
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            
            # Set weather
            weather = carla.WeatherParameters(
                cloudiness=20.0, precipitation=0.0, sun_altitude_angle=70.0,
                fog_density=0.0, wetness=0.0
            )
            self.world.set_weather(weather)
            
            # === VEHICLE SPAWNING ===
            logger.info("🚗 Spawning vehicle...")
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.audi.a2')
            
            # Custom spawn location
            spawn_transform = carla.Transform(
                carla.Location(x=280.363739, y=-129.306351, z=0.275307),
                carla.Rotation(yaw=180.0)
            )
            
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if not self.vehicle:
                logger.error("❌ Failed to spawn vehicle!")
                return
            
            logger.info(f"✓ Vehicle spawned at ({spawn_transform.location.x:.1f}, {spawn_transform.location.y:.1f})")
            
            # === SPECTATOR SETUP ===
            spectator = self.world.get_spectator()
            spectator_transform = carla.Transform(
                spawn_transform.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(spectator_transform)
            
            time.sleep(3.0)  # Let vehicle settle
            
            # === DEFINE WAYPOINTS AND START POSE ===
            start_pose = Pose(
                x=280.363739, y=-129.306351, z=0.101746,
                roll=0.0, pitch=0.0, yaw=180.0
            )
            
            # Multiple waypoints to visit (from your TSP example)
            waypoints = [
                [334.949799, -161.106171, 0.001736],
                [339.100037, -258.568939, 0.001679],
                [396.295319, -183.195740, 0.001678],
                [267.657074, -1.983160, 0.001678],
                [153.868896, -26.115866, 0.001678],
                [290.515564, -56.175072, 0.001677],
                [92.325722, -86.063644, 0.001677],
                [88.384346, -287.468567, 0.001728],
                [177.594101, -326.386902, 0.001677],
                [-1.646942, -197.501282, 0.001555],
                [59.701321, -1.970804, 0.001467],
                [122.100121, -55.142044, 0.001596],
                [161.030975, -129.313187, 0.001679],
                [184.758713, -199.424271, 0.001680],
            ]
            
            # Optional: Use only a subset for demonstration
            demo_waypoints = waypoints[:6]  # First 6 waypoints
            
            logger.info(f"📍 Planning route through {len(demo_waypoints)} waypoints")
            
            # === RUN COMPLETE NAVIGATION ===
            self.navigate_to_waypoints(
                start_pose=start_pose,
                waypoints_xyz=demo_waypoints,
                orientation_weights={"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "turn": 0.0}
            )
            
            # === VISUALIZATION AND ANALYSIS ===
            logger.info("📊 Generating route analysis...")
            
            # Plot the optimized route
            order, total_cost = plan_tsp_path(
                start_pose=start_pose,
                waypoints_xyz=demo_waypoints,
                exact_threshold=9,
                use_two_opt=True
            )
            
            # Show 2D visualization
            plot_path_2d(start_pose, demo_waypoints, order, 
                        title="CARLA TSP Optimized Route")
            
            # Keep simulation running for observation
            logger.info("🎬 Keeping simulation active for 30 seconds...")
            time.sleep(30.0)
            
        except KeyboardInterrupt:
            logger.info("🛑 System interrupted by user")
        except Exception as e:
            logger.error(f"❌ System error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Comprehensive cleanup of all CARLA resources."""
        logger.info("🧹 Cleaning up CARLA resources...")
        
        try:
            self.path_execution_active = False
            
            # Stop vehicle
            if self.vehicle and self.vehicle.is_alive:
                stop_control = carla.VehicleControl()
                stop_control.brake = 1.0
                self.vehicle.apply_control(stop_control)
                time.sleep(0.5)
                self.vehicle.destroy()
                logger.info("✓ Vehicle destroyed")
            
            # Clean up debug objects
            self.cleanup_debug_objects()
            
            # Clear references
            self.vehicle = None
            self.world = None
            self.client = None
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# ============================== VISUALIZATION ==============================

def plot_path_2d(start_pose: Pose,
                 waypoints_xyz: Iterable[Iterable[float]],
                 order: List[int],
                 title: str = "TSP Optimized Path (Top-Down)") -> None:
    """
    Create 2D top-down visualization of the TSP optimized path.
    
    Args:
        start_pose: Starting position and orientation
        waypoints_xyz: List of waypoints
        order: Optimal visit order from TSP solver
        title: Plot title
    """
    pts = _build_points_array(waypoints_xyz)
    p0 = np.array([start_pose.x, start_pose.y], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    # Plot waypoints
    ax.scatter(pts[:, 0], pts[:, 1], s=100, c='red', marker='o', 
              label="Waypoints", zorder=5, edgecolors='black', linewidth=1)
    
    # Plot start position
    ax.scatter([p0[0]], [p0[1]], s=150, marker='^', c='blue',
              label="Start Position", zorder=6, edgecolors='black', linewidth=2)

    # Plot optimized path
    path_xy = np.vstack([p0, pts[order][:, :2]])
    ax.plot(path_xy[:, 0], path_xy[:, 1], linewidth=3, color='green',
           label="Optimized Route", alpha=0.8, zorder=3)
    
    # Add arrows to show direction
    for i in range(len(path_xy) - 1):
        dx = path_xy[i+1, 0] - path_xy[i, 0]
        dy = path_xy[i+1, 1] - path_xy[i, 1]
        ax.arrow(path_xy[i, 0], path_xy[i, 1], dx*0.7, dy*0.7,
                head_width=2, head_length=3, fc='green', ec='green',
                alpha=0.6, zorder=4)

    # Number waypoints in visit order
    for step, idx in enumerate(order, start=1):
        ax.annotate(f'{step}', (pts[idx, 0], pts[idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))

    # Formatting
    all_xy = np.vstack([pts[:, :2], p0])
    mins = np.min(all_xy, axis=0)
    maxs = np.max(all_xy, axis=0)
    pad = 0.1 * (maxs - mins + 1e-6)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    
    ax.set_xlabel("X Coordinate (meters)", fontsize=12)
    ax.set_ylabel("Y Coordinate (meters)", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_path_3d(start_pose: Pose,
                 waypoints_xyz: Iterable[Iterable[float]],
                 order: List[int],
                 title: str = "TSP Optimized Path (3D)") -> None:
    """Create 3D visualization of the TSP optimized path."""
    pts = _build_points_array(waypoints_xyz)
    p0 = np.array([start_pose.x, start_pose.y, start_pose.z], dtype=float)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Plot waypoints and path
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=100, c='red', 
              label="Waypoints", edgecolors='black')
    ax.scatter([p0[0]], [p0[1]], [p0[2]], s=150, marker='^', c='blue',
              label="Start Position", edgecolors='black')

    path_xyz = np.vstack([p0, pts[order]])
    ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], 
           linewidth=3, color='green', label="Optimized Route")

    # Number waypoints
    for step, idx in enumerate(order, start=1):
        ax.text(pts[idx, 0], pts[idx, 1], pts[idx, 2], f'{step}',
               fontsize=10, fontweight='bold')

    # Formatting
    mins = np.min(np.vstack([pts, p0]), axis=0)
    maxs = np.max(np.vstack([pts, p0]), axis=0)
    ranges = maxs - mins
    center = (maxs + mins) / 2
    r = max(ranges) * 0.6 if np.any(ranges > 0) else 1.0
    
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# ============================== UTILITY FUNCTIONS ==============================

def waypoints_in_order(order: List[int],
                       waypoints_xyz: Iterable[Iterable[float]]) -> List[Tuple[float, float, float]]:
    """Return waypoints in TSP optimized order."""
    pts = _build_points_array(waypoints_xyz)
    return [tuple(pts[i]) for i in order]

def print_route_summary(start_pose: Pose, waypoints_xyz: List[List[float]], 
                       order: List[int], total_cost: float):
    """Print comprehensive route summary."""
    print("\n" + "="*60)
    print("🎯 ROUTE OPTIMIZATION SUMMARY")
    print("="*60)
    
    ordered_points = waypoints_in_order(order, waypoints_xyz)
    sequence = [(start_pose.x, start_pose.y, start_pose.z)] + ordered_points
    
    print(f"📍 Total waypoints: {len(waypoints_xyz)}")
    print(f"📏 Total route cost: {total_cost:.3f} meters")
    print(f"🔄 Visit order: {order}")
    
    print("\n📋 DETAILED ROUTE SEQUENCE:")
    print("   Step | X Coordinate | Y Coordinate | Z Coordinate")
    print("   -----|--------------|--------------|-------------")
    
    for i, point in enumerate(sequence):
        step_type = "START" if i == 0 else f"WP-{i}"
        print(f"   {step_type:4} | {point[0]:11.3f} | {point[1]:11.3f} | {point[2]:11.3f}")
    
    # Calculate segment distances
    print(f"\n📐 SEGMENT DISTANCES:")
    total_distance = 0.0
    for i in range(len(sequence) - 1):
        p1, p2 = sequence[i], sequence[i+1]
        dist = math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))
        total_distance += dist
        print(f"   Segment {i+1}: {dist:.3f} meters")
    
    print(f"\n📊 ROUTE STATISTICS:")
    print(f"   • Total distance: {total_distance:.3f} meters")
    print(f"   • Average segment: {total_distance/max(1, len(sequence)-1):.3f} meters")
    
    # Energy estimation
    kwh, _ = estimate_energy_kwh(start_pose, waypoints_xyz, order)
    print(f"   • Estimated energy: {kwh:.6f} kWh")
    print("="*60)

# ============================== MAIN EXECUTION ==============================

def main():
    """
    Main function demonstrating the complete CARLA pathfinding system.
    
    This function shows how to use both the TSP optimization and 
    CARLA pathfinding components together.
    """
    print("🚗" + "="*58 + "🚗")
    print("   CARLA ADVANCED PATHFINDING WITH TSP OPTIMIZATION")
    print("🚗" + "="*58 + "🚗")
    print()
    print("🎯 SYSTEM CAPABILITIES:")
    print("   • TSP route optimization for multiple waypoints")
    print("   • Real-time obstacle detection and avoidance")
    print("   • Advanced vehicle control with PID algorithms")
    print("   • Comprehensive visualization (2D/3D + CARLA)")
    print("   • Energy consumption estimation")
    print("   • Robust error handling and recovery")
    print()
    print("🔧 CONFIGURATION:")
    print(f"   • Grid resolution: {GRID_SIZE}m per cell")
    print(f"   • Target speed: {TARGET_SPEED} m/s")
    print(f"   • Grid size: {GRID_WIDTH}×{GRID_HEIGHT} cells")
    print(f"   • RRT max iterations: {RRT_MAX_ITER}")
    print("🚗" + "="*58 + "🚗")
    
    # Example 1: TSP-only demonstration
    print("\n📊 EXAMPLE 1: TSP ROUTE OPTIMIZATION")
    print("-" * 40)
    
    # Define example waypoints and start pose
    start = Pose(x=280.363739, y=-129.306351, z=0.101746, yaw=180.0)
    
    waypoints = [
        [334.949799, -161.106171, 0.001736],
        [339.100037, -258.568939, 0.001679],
        [396.295319, -183.195740, 0.001678],
        [267.657074, -1.983160, 0.001678],
        [153.868896, -26.115866, 0.001678],
        [290.515564, -56.175072, 0.001677],
    ]
    
    # Solve TSP
    order, total_cost = plan_tsp_path(
        start_pose=start,
        waypoints_xyz=waypoints,
        orientation_weights={"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "turn": 0.0},
        exact_threshold=9,
        use_two_opt=True
    )
    
    # Print results
    print_route_summary(start, waypoints, order, total_cost)
    
    # Visualize
    plot_path_2d(start, waypoints, order, "TSP Optimized Route - Example")
    
    # Example 2: Full CARLA system
    print(f"\n🚗 EXAMPLE 2: FULL CARLA SIMULATION")
    print("-" * 40)
    print("Starting complete CARLA pathfinding system...")
    print("Make sure CARLA server is running on localhost:2000")
    
    response = input("\nRun full CARLA simulation? (y/n): ").lower().strip()
    
    if response == 'y':
        planner = CarlaAdvancedPathPlanner()
        try:
            planner.run_complete_system()
        except Exception as e:
            logger.error(f"CARLA system error: {e}")
        finally:
            planner.cleanup()
    else:
        print("Skipping CARLA simulation.")
    
    print("\n🏁 DEMONSTRATION COMPLETE")
    print("🚗" + "="*58 + "🚗")

if __name__ == "__main__":
    main()

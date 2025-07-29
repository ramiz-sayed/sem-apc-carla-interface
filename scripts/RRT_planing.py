#!/usr/bin/env python3
"""
Perfect CARLA Pathfinding System with RRT Algorithm
==================================================
This script implements an advanced pathfinding system for CARLA simulator using:
- RRT (Rapidly-exploring Random Tree) algorithm for path planning
- Enhanced obstacle detection and avoidance
- Smooth vehicle control with PID-like behavior
- Comprehensive visualization of obstacles and paths
- Robust error handling and logging
"""

import carla
import random
import time
import numpy as np
import math
from collections import defaultdict
import logging
import threading
from typing import List, Tuple, Optional

# Configure comprehensive logging system
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carla_pathfinding.log'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

class CarlaPathPlanner:
    """
    Advanced pathfinding system for CARLA simulator
    Implements RRT algorithm with obstacle avoidance and smooth vehicle control
    """
    
    def __init__(self):
        """Initialize the pathfinding system with optimized parameters"""
        
        # === GRID AND WORLD PARAMETERS ===
        self.GRID_SIZE = 1.0                    # Size of each grid cell in meters
        self.GRID_WIDTH = 200                   # Number of grid cells in X direction
        self.GRID_HEIGHT = 200                  # Number of grid cells in Y direction
        self.OBSTACLE_BUFFER = 1.2              # Safety buffer around obstacles (meters)
        
        # === VEHICLE CONTROL PARAMETERS ===
        self.TARGET_SPEED = 6.0                 # Target driving speed (m/s)
        self.WAYPOINT_TOLERANCE = 1.8           # Distance to consider waypoint reached
        self.CONTROL_FREQUENCY = 20             # Control loop frequency (Hz)
        
        # === RRT ALGORITHM PARAMETERS ===
        self.RRT_STEP_SIZE = 3.0               # Maximum step size for RRT expansion
        self.RRT_MAX_ITER = 8000               # Maximum RRT iterations
        self.GOAL_BIAS = 0.2                   # Probability of sampling toward goal
        self.SMOOTHING_ENABLED = True          # Enable path smoothing
        
        # === VISUALIZATION PARAMETERS ===
        self.DEBUG = True                      # Enable debug visualizations
        self.OBSTACLE_SAMPLE_RATE = 4          # Sample every Nth obstacle for visualization
        self.MAX_OBSTACLE_POINTS = 800         # Maximum obstacle points to visualize
        
        # === CARLA COLOR DEFINITIONS ===
        self.RED = carla.Color(255, 0, 0)      # Obstacles
        self.GREEN = carla.Color(0, 255, 0)    # Path segments
        self.BLUE = carla.Color(0, 0, 255)     # Start position
        self.YELLOW = carla.Color(255, 255, 0) # Current waypoint
        self.WHITE = carla.Color(200, 200, 200)# Grid points
        self.PURPLE = carla.Color(255, 0, 255) # Goal position
        self.ORANGE = carla.Color(255, 165, 0) # Vehicle trajectory
        
        # === INTERNAL STATE VARIABLES ===
        self.vehicle = None                    # CARLA vehicle actor
        self.debug_objects = []               # List of debug visualization objects
        self.world = None                     # CARLA world instance
        self.client = None                    # CARLA client connection
        self.current_waypoint_index = 0       # Index of current target waypoint
        self.path_execution_active = False    # Flag for path execution status

    def cleanup_debug_objects(self):
        """
        Clean up all debug visualization objects to prevent memory leaks
        Called before creating new visualizations
        """
        logger.debug(f"Cleaning up {len(self.debug_objects)} debug objects")
        
        for obj in self.debug_objects:
            try:
                # Check if object has destroy method and call it
                if hasattr(obj, 'destroy'):
                    obj.destroy()
            except Exception as e:
                # Ignore cleanup errors as objects might already be destroyed
                logger.debug(f"Error cleaning debug object: {e}")
        
        # Clear the list after cleanup
        self.debug_objects.clear()
        logger.debug("Debug objects cleanup completed")

    def get_obstacle_grid(self, center_location: carla.Location) -> Tuple[np.ndarray, carla.Location]:
        """
        Create a 2D occupancy grid from CARLA world obstacles
        
        Args:
            center_location: Center point for grid generation
            
        Returns:
            Tuple of (grid array, origin location)
            grid[x][y] = 1 means obstacle, 0 means free space
        """
        logger.info("Starting obstacle grid generation...")
        
        # Initialize empty grid (0 = free space, 1 = obstacle)
        grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.uint8)
        
        # Calculate grid origin (bottom-left corner in world coordinates)
        origin = carla.Location(
            x=center_location.x - (self.GRID_WIDTH * self.GRID_SIZE) / 2,
            y=center_location.y - (self.GRID_HEIGHT * self.GRID_SIZE) / 2,
            z=center_location.z
        )
        logger.info(f"Grid origin set to: {origin}")
        
        def world_to_grid(location: carla.Location) -> Tuple[int, int]:
            """Convert world coordinates to grid indices"""
            x = int((location.x - origin.x) / self.GRID_SIZE)
            y = int((location.y - origin.y) / self.GRID_SIZE)
            # Clamp to grid boundaries
            return np.clip(x, 0, self.GRID_WIDTH - 1), np.clip(y, 0, self.GRID_HEIGHT - 1)
        
        # Define obstacle types to detect
        obstacle_types = [
            carla.CityObjectLabel.Buildings,    # Buildings and structures
            carla.CityObjectLabel.Walls,        # Walls and barriers
            carla.CityObjectLabel.Fences,       # Fences
            carla.CityObjectLabel.Poles,        # Light poles, signs
            carla.CityObjectLabel.TrafficSigns, # Traffic signs
            carla.CityObjectLabel.Vegetation,   # Trees and bushes
            carla.CityObjectLabel.Static        # Other static objects
        ]
        
        obstacle_count = 0
        total_objects = 0
        
        # Process each obstacle type
        for obj_type in obstacle_types:
            try:
                # Get all objects of this type from CARLA world
                objects = self.world.get_environment_objects(obj_type)
                logger.info(f"Processing {len(objects)} objects of type {obj_type}")
                
                for obj in objects:
                    total_objects += 1
                    transform = obj.transform
                    bbox = obj.bounding_box
                    
                    # Calculate object bounds with safety buffer
                    min_x = transform.location.x - bbox.extent.x - self.OBSTACLE_BUFFER
                    max_x = transform.location.x + bbox.extent.x + self.OBSTACLE_BUFFER
                    min_y = transform.location.y - bbox.extent.y - self.OBSTACLE_BUFFER
                    max_y = transform.location.y + bbox.extent.y + self.OBSTACLE_BUFFER
                    
                    # Convert world bounds to grid coordinates
                    gx_min, gy_min = world_to_grid(carla.Location(min_x, min_y, 0))
                    gx_max, gy_max = world_to_grid(carla.Location(max_x, max_y, 0))
                    
                    # Mark all cells within bounds as obstacles
                    for x in range(gx_min, gx_max + 1):
                        for y in range(gy_min, gy_max + 1):
                            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                                if grid[x][y] == 0:  # Only count new obstacles
                                    grid[x][y] = 1
                                    obstacle_count += 1
                                    
            except Exception as e:
                logger.warning(f"Error processing obstacle type {obj_type}: {e}")
        
        # Log statistics
        total_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        obstacle_percentage = (obstacle_count / total_cells) * 100
        logger.info(f"Obstacle grid generated: {obstacle_count} obstacle cells "
                   f"({obstacle_percentage:.1f}%) from {total_objects} objects")
        
        return grid, origin

    def bresenham_line(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm - returns all grid points on line between start and end
        Used for collision detection along paths
        
        Args:
            start: Starting grid coordinates (x, y)
            end: Ending grid coordinates (x, y)
            
        Returns:
            List of all grid points on the line
        """
        x0, y0 = start
        x1, y1 = end
        points = []
        
        # Calculate deltas and step directions
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1  # Step direction in x
        sy = 1 if y0 < y1 else -1  # Step direction in y
        err = dx - dy              # Error term
        
        # Generate all points on the line
        while True:
            points.append((x0, y0))
            
            # Check if we've reached the end point
            if x0 == x1 and y0 == y1:
                break
                
            # Update error and coordinates
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points

    def check_collision_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                           grid: np.ndarray) -> bool:
        """
        Check if a straight line path between two points collides with obstacles
        
        Args:
            start: Starting grid coordinates
            end: Ending grid coordinates
            grid: Obstacle grid (1 = obstacle, 0 = free)
            
        Returns:
            True if collision detected, False if path is clear
        """
        # Get all points on the line using Bresenham's algorithm
        points = self.bresenham_line(start, end)
        
        # Check each point for collisions
        for x, y in points:
            # Check if point is outside grid boundaries
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True  # Out of bounds = collision
            
            # Check if point is an obstacle
            if grid[x][y] == 1:
                return True  # Obstacle found = collision
        
        return False  # No collision detected

    def distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two grid points
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            Euclidean distance as float
        """
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_nearest_node(self, nodes: List[Tuple[int, int]], 
                         target: Tuple[int, int]) -> Tuple[int, int]:
        """
        Find the nearest node to a target point (optimized for large node sets)
        
        Args:
            nodes: List of existing nodes
            target: Target point to find nearest neighbor for
            
        Returns:
            Nearest node coordinates
        """
        if not nodes:
            return None
        
        # Use numpy for vectorized distance calculation (faster for large sets)
        if len(nodes) > 100:
            nodes_array = np.array(nodes)
            target_array = np.array(target)
            distances = np.sum((nodes_array - target_array)**2, axis=1)
            nearest_idx = np.argmin(distances)
            return tuple(nodes_array[nearest_idx])
        else:
            # Use simple loop for small sets
            return min(nodes, key=lambda p: self.distance(p, target))

    def rrt_path_plan(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        RRT (Rapidly-exploring Random Tree) pathfinding algorithm
        
        Args:
            start: Starting grid coordinates
            goal: Goal grid coordinates  
            grid: Obstacle grid
            
        Returns:
            List of waypoints from start to goal, empty if no path found
        """
        logger.info(f"Starting RRT pathfinding from {start} to {goal}")
        
        # Validate start and goal positions
        if grid[start[0]][start[1]] == 1:
            logger.error("Start position is in obstacle!")
            return []
        if grid[goal[0]][goal[1]] == 1:
            logger.error("Goal position is in obstacle!")
            return []
        
        # Initialize RRT data structures
        tree = {tuple(start): None}  # Dictionary: node -> parent
        nodes = [start]              # List of all nodes for fast nearest neighbor search
        
        logger.info(f"RRT parameters: max_iter={self.RRT_MAX_ITER}, "
                   f"step_size={self.RRT_STEP_SIZE}, goal_bias={self.GOAL_BIAS}")
        
        # Main RRT loop
        for iteration in range(self.RRT_MAX_ITER):
            # Adaptive goal biasing - increase bias as iterations progress
            current_goal_bias = min(self.GOAL_BIAS + (iteration / self.RRT_MAX_ITER) * 0.15, 0.4)
            
            # Sample random point (with goal biasing)
            if random.random() < current_goal_bias:
                sample = goal  # Sample goal directly
            else:
                # Sample random point in grid
                sample = (
                    random.randint(0, self.GRID_WIDTH - 1),
                    random.randint(0, self.GRID_HEIGHT - 1)
                )
            
            # Find nearest existing node
            nearest = self.find_nearest_node(nodes, sample)
            if nearest is None:
                continue
            
            # Calculate direction and distance to sample
            dist_to_sample = self.distance(nearest, sample)
            if dist_to_sample == 0:
                continue  # Skip if sample is same as nearest
            
            # Calculate unit direction vector
            direction = (
                (sample[0] - nearest[0]) / dist_to_sample,
                (sample[1] - nearest[1]) / dist_to_sample
            )
            
            # Apply step size limit
            step_size = min(self.RRT_STEP_SIZE, dist_to_sample)
            new_node = (
                int(round(nearest[0] + direction[0] * step_size)),
                int(round(nearest[1] + direction[1] * step_size))
            )
            
            # Validate new node
            if (0 <= new_node[0] < self.GRID_WIDTH and 
                0 <= new_node[1] < self.GRID_HEIGHT and
                new_node not in tree and
                not self.check_collision_line(nearest, new_node, grid)):
                
                # Add new node to tree
                tree[new_node] = nearest
                nodes.append(new_node)
                
                # Check if we can connect directly to goal
                goal_distance = self.distance(new_node, goal)
                if goal_distance <= self.RRT_STEP_SIZE * 1.5:  # Allow slightly larger step to goal
                    if not self.check_collision_line(new_node, goal, grid):
                        # Path to goal found!
                        tree[tuple(goal)] = new_node
                        logger.info(f"Path found in {iteration + 1} iterations with {len(nodes)} nodes")
                        break
            
            # Progress logging every 1000 iterations
            if iteration % 1000 == 0 and iteration > 0:
                logger.info(f"RRT progress: iteration {iteration}, nodes: {len(nodes)}")
        
        # Check if goal was reached
        if tuple(goal) not in tree:
            logger.error(f"RRT failed to find path after {self.RRT_MAX_ITER} iterations")
            return []
        
        # Reconstruct path by following parent pointers
        path = []
        current = tuple(goal)
        while current is not None:
            path.append(current)
            current = tree[current]
        
        # Reverse to get start-to-goal order
        path.reverse()
        
        logger.info(f"Raw path reconstructed with {len(path)} waypoints")
        
        # Apply path smoothing if enabled
        if self.SMOOTHING_ENABLED:
            path = self.smooth_path(path, grid)
        
        return path

    def smooth_path(self, path: List[Tuple[int, int]], 
                   grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary intermediate waypoints
        Uses line-of-sight optimization
        
        Args:
            path: Original path waypoints
            grid: Obstacle grid for collision checking
            
        Returns:
            Smoothed path with fewer waypoints
        """
        if len(path) <= 2:
            return path  # Can't smooth very short paths
        
        logger.info(f"Smoothing path with {len(path)} waypoints...")
        
        smoothed = [path[0]]  # Always keep start point
        i = 0
        
        while i < len(path) - 1:
            # Try to connect current point to furthest visible point
            furthest_reachable = i + 1  # At minimum, can reach next point
            
            # Check line of sight to increasingly distant points
            for j in range(len(path) - 1, i + 1, -1):
                if not self.check_collision_line(path[i], path[j], grid):
                    furthest_reachable = j
                    break
            
            # Add the furthest reachable point
            if furthest_reachable < len(path):
                smoothed.append(path[furthest_reachable])
                i = furthest_reachable
            else:
                break
        
        # Ensure goal is included
        if smoothed[-1] != path[-1]:
            smoothed.append(path[-1])
        
        logger.info(f"Path smoothed from {len(path)} to {len(smoothed)} waypoints "
                   f"({((len(path) - len(smoothed)) / len(path) * 100):.1f}% reduction)")
        
        return smoothed

    def enhanced_vehicle_control(self, target_location: carla.Location) -> float:
        """
        Advanced PID-based vehicle control system with smooth steering and speed management
        
        Args:
            target_location: Target position in world coordinates
            
        Returns:
            Distance to target location
        """
        if not self.vehicle:
            return float('inf')
        
        # Get current vehicle state
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
        
        # Calculate vector to target
        dx = target_location.x - vehicle_location.x
        dy = target_location.y - vehicle_location.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # If very close to target, minimal control needed
        if distance < 0.2:
            return distance
        
        # Calculate target heading angle
        target_yaw = math.degrees(math.atan2(dy, dx))
        current_yaw = vehicle_rotation.yaw
        
        # Normalize yaw difference to [-180, 180] range
        yaw_diff = target_yaw - current_yaw
        while yaw_diff > 180:
            yaw_diff -= 360
        while yaw_diff < -180:
            yaw_diff += 360
        
        # === STEERING CONTROL ===
        # Base steering proportional to yaw error
        steering_sensitivity = 35.0  # Degrees per full steering input
        base_steer = np.clip(yaw_diff / steering_sensitivity, -1.0, 1.0)
        
        # Speed-dependent steering adjustment (less steering at high speed)
        speed_factor = max(0.3, 1.0 - (current_speed / 10.0))
        steer = base_steer * speed_factor
        
        # === THROTTLE CONTROL ===
        # Base throttle from speed error
        speed_error = self.TARGET_SPEED - current_speed
        base_throttle = np.clip(speed_error / self.TARGET_SPEED, 0.0, 0.9)
        
        # Distance-based throttle modulation (slow down when approaching waypoint)
        if distance < 8.0:
            distance_factor = max(0.3, distance / 8.0)
            base_throttle *= distance_factor
        
        # Steering-based throttle reduction (slow down for sharp turns)
        steering_penalty = 1.0 - (abs(steer) * 0.4)
        throttle = base_throttle * steering_penalty
        
        # === BRAKE CONTROL ===
        brake = 0.0
        
        # Distance-based braking
        if distance < 4.0 and current_speed > 3.0:
            brake = min(0.5, (4.0 - distance) / 4.0)
        
        # Sharp turn braking
        elif abs(yaw_diff) > 60 and current_speed > 5.0:
            brake = 0.3
        
        # High speed braking
        elif current_speed > self.TARGET_SPEED * 1.5:
            brake = 0.2
        
        # === APPLY CONTROL ===
        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = float(max(0.0, throttle))
        control.brake = float(brake)
        control.hand_brake = False
        control.reverse = False
        
        # Apply control to vehicle
        self.vehicle.apply_control(control)
        
        # Log detailed control info occasionally
        if random.random() < 0.01:  # 1% chance to log
            logger.debug(f"Control: steer={control.steer:.2f}, throttle={control.throttle:.2f}, "
                        f"brake={control.brake:.2f}, speed={current_speed:.1f}, "
                        f"distance={distance:.1f}, yaw_diff={yaw_diff:.1f}")
        
        return distance

    def draw_debug_visualization(self, grid: np.ndarray, origin: carla.Location, 
                               path: List[Tuple[int, int]]):
        """
        Create comprehensive debug visualization in CARLA world
        Shows obstacles, path, start/goal positions with different colors
        
        Args:
            grid: Obstacle grid
            origin: Grid origin in world coordinates
            path: Planned path waypoints
        """
        if not self.DEBUG:
            return
        
        logger.info("Creating debug visualization...")
        
        # Clean up previous visualizations
        self.cleanup_debug_objects()
        
        # === VISUALIZE OBSTACLES ===
        obstacle_points = []
        
        # Sample obstacles for visualization (performance optimization)
        for x in range(0, self.GRID_WIDTH, self.OBSTACLE_SAMPLE_RATE):
            for y in range(0, self.GRID_HEIGHT, self.OBSTACLE_SAMPLE_RATE):
                if grid[x][y] == 1:
                    obstacle_points.append((x, y))
        
        # Limit number of obstacle points for performance
        if len(obstacle_points) > self.MAX_OBSTACLE_POINTS:
            obstacle_points = random.sample(obstacle_points, self.MAX_OBSTACLE_POINTS)
        
        logger.info(f"Visualizing {len(obstacle_points)} obstacle points")
        
        # Draw obstacle points
        for x, y in obstacle_points:
            loc = carla.Location(
                x=origin.x + x * self.GRID_SIZE,
                y=origin.y + y * self.GRID_SIZE,
                z=0.3  # Slightly above ground
            )
            # Draw red points for obstacles
            self.world.debug.draw_point(loc, 0.15, self.RED, 20.0)
        
        # === VISUALIZE PATH ===
        if path and len(path) > 1:
            logger.info(f"Visualizing path with {len(path)} waypoints")
            
            # Draw path segments
            for i in range(len(path) - 1):
                start_loc = carla.Location(
                    x=origin.x + path[i][0] * self.GRID_SIZE,
                    y=origin.y + path[i][1] * self.GRID_SIZE,
                    z=0.8  # Above obstacles
                )
                end_loc = carla.Location(
                    x=origin.x + path[i + 1][0] * self.GRID_SIZE,
                    y=origin.y + path[i + 1][1] * self.GRID_SIZE,
                    z=0.8
                )
                # Draw green lines for path segments
                self.world.debug.draw_line(start_loc, end_loc, 0.2, self.GREEN, 20.0)
            
            # Draw waypoint markers
            for i, waypoint in enumerate(path):
                loc = carla.Location(
                    x=origin.x + waypoint[0] * self.GRID_SIZE,
                    y=origin.y + waypoint[1] * self.GRID_SIZE,
                    z=1.2
                )
                # Different colors for different waypoint types
                if i == 0:
                    color = self.BLUE    # Start waypoint
                    size = 0.4
                elif i == len(path) - 1:
                    color = self.PURPLE  # Goal waypoint
                    size = 0.4
                else:
                    color = self.YELLOW  # Intermediate waypoints
                    size = 0.25
                
                self.world.debug.draw_point(loc, size, color, 20.0)
        
        # === VISUALIZE GRID BOUNDS ===
        # Draw grid boundary for reference
        corners = [
            carla.Location(origin.x, origin.y, 0.1),
            carla.Location(origin.x + self.GRID_WIDTH * self.GRID_SIZE, origin.y, 0.1),
            carla.Location(origin.x + self.GRID_WIDTH * self.GRID_SIZE, 
                          origin.y + self.GRID_HEIGHT * self.GRID_SIZE, 0.1),
            carla.Location(origin.x, origin.y + self.GRID_HEIGHT * self.GRID_SIZE, 0.1)
        ]
        
        # Draw grid boundary lines
        for i in range(4):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]
            self.world.debug.draw_line(start_corner, end_corner, 0.1, self.WHITE, 20.0)
        
        logger.info("Debug visualization completed")

    def execute_path(self, path: List[Tuple[int, int]], origin: carla.Location):
        """
        Execute the planned path by controlling the vehicle through each waypoint
        
        Args:
            path: List of waypoints in grid coordinates
            origin: Grid origin for coordinate conversion
        """
        if not path:
            logger.error("Cannot execute empty path")
            return
        
        logger.info(f"Starting path execution with {len(path)} waypoints")
        self.path_execution_active = True
        self.current_waypoint_index = 0
        
        # Execute each waypoint in sequence
        for i, waypoint in enumerate(path):
            self.current_waypoint_index = i
            
            # Convert grid coordinates to world coordinates
            target_location = carla.Location(
                x=origin.x + waypoint[0] * self.GRID_SIZE,
                y=origin.y + waypoint[1] * self.GRID_SIZE,
                z=0.5
            )
            
            logger.info(f"Moving to waypoint {i+1}/{len(path)}: "
                       f"grid({waypoint[0]}, {waypoint[1]}) -> "
                       f"world({target_location.x:.1f}, {target_location.y:.1f})")
            
            # Move toward waypoint with timeout
            start_time = time.time()
            timeout_duration = 45.0  # 45 seconds per waypoint
            min_distance_achieved = float('inf')
            stuck_counter = 0
            
            while time.time() - start_time < timeout_duration:
                # Control vehicle toward target
                distance = self.enhanced_vehicle_control(target_location)
                
                # Track minimum distance for stuck detection
                if distance < min_distance_achieved:
                    min_distance_achieved = distance
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                
                # Check if waypoint reached
                if distance < self.WAYPOINT_TOLERANCE:
                    logger.info(f"✓ Reached waypoint {i+1} (distance: {distance:.2f}m)")
                    break
                
                # Check if vehicle is stuck
                if stuck_counter > 100:  # 5 seconds at 20Hz
                    logger.warning(f"Vehicle may be stuck at waypoint {i+1}, "
                                 f"minimum distance: {min_distance_achieved:.2f}m")
                    if min_distance_achieved < self.WAYPOINT_TOLERANCE * 2:
                        logger.info("Close enough to waypoint, continuing...")
                        break
                
                # Control loop timing
                time.sleep(1.0 / self.CONTROL_FREQUENCY)
            else:
                # Timeout occurred
                logger.warning(f"⚠ Timeout reaching waypoint {i+1} "
                             f"(final distance: {min_distance_achieved:.2f}m)")
                
                # Decide whether to continue or abort
                if min_distance_achieved > self.WAYPOINT_TOLERANCE * 3:
                    logger.error("Vehicle too far from waypoint, aborting path execution")
                    break
        
        # Stop the vehicle at the end
        logger.info("Path execution completed, stopping vehicle")
        self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
        self.path_execution_active = False

    def run(self):
        """
        Main execution function - orchestrates the entire pathfinding process
        """
        try:
            # === CARLA CONNECTION SETUP ===
            logger.info("🚀 Starting CARLA Pathfinding System")
            logger.info("Connecting to CARLA server...")
            # Connect to CARLA server
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(30.0)  # Increased timeout for stability
            
            # Test connection
            version = self.client.get_server_version()
            logger.info(f"Connected to CARLA server version: {version}")
            
            # === WORLD SETUP ===
            logger.info("Loading CARLA world...")
            available_maps = self.client.get_available_maps()
            logger.info(f"Available maps: {available_maps}")
            
            # Load Town03 (good for pathfinding demonstrations)
            self.world = self.client.load_world('Town03')
            logger.info("World 'Town03' loaded successfully")
            
            # Configure world settings for optimal performance
            settings = self.world.get_settings()
            settings.synchronous_mode = False  # Asynchronous for real-time control
            settings.fixed_delta_seconds = None  # Variable time step
            self.world.apply_settings(settings)
            
            # Set optimal weather conditions for visibility
            weather = carla.WeatherParameters(
                cloudiness=20.0,           # Light clouds
                precipitation=0.0,         # No rain
                sun_altitude_angle=70.0,   # High sun
                fog_density=0.0,           # Clear visibility
                wetness=0.0               # Dry roads
            )
            self.world.set_weather(weather)
            logger.info("Weather conditions optimized")
            
            # === VEHICLE SPAWNING ===
            logger.info("Spawning vehicle...")
            blueprint_library = self.world.get_blueprint_library()
            
            # Choose a suitable vehicle (Audi A2 - good handling)
            vehicle_bp = blueprint_library.find('vehicle.audi.a2')
            if vehicle_bp.has_attribute('color'):
                # Set distinctive color for easy identification
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
                logger.info(f"Vehicle color set to: {color}")
            
            # Get available spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            logger.info(f"Found {len(spawn_points)} available spawn points")
            
            # Try to spawn vehicle at different locations
            spawn_attempts = 0
            max_spawn_attempts = 10
            
            for spawn_point in random.sample(spawn_points, min(max_spawn_attempts, len(spawn_points))):
                spawn_attempts += 1
                try:
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if self.vehicle:
                        logger.info(f"✓ Vehicle spawned successfully at attempt {spawn_attempts}")
                        logger.info(f"Vehicle location: {spawn_point.location}")
                        break
                except Exception as e:
                    logger.debug(f"Spawn attempt {spawn_attempts} failed: {e}")
                    continue
            
            if not self.vehicle:
                logger.error("❌ Failed to spawn vehicle after all attempts!")
                return
            
            # Wait for vehicle to settle in the world
            logger.info("Waiting for vehicle to settle...")
            time.sleep(3.0)
            
            # === CAMERA/SPECTATOR SETUP ===
            logger.info("Setting up spectator view...")
            spectator = self.world.get_spectator()
            vehicle_transform = self.vehicle.get_transform()
            
            # Position spectator above vehicle for bird's eye view
            spectator_transform = carla.Transform(
                vehicle_transform.location + carla.Location(z=60),  # 60m above vehicle
                carla.Rotation(pitch=-90)  # Looking straight down
            )
            spectator.set_transform(spectator_transform)
            logger.info("Spectator positioned for optimal view")
            
            # === OBSTACLE GRID GENERATION ===
            logger.info("🗺️ Generating obstacle map...")
            start_time = time.time()
            
            vehicle_location = self.vehicle.get_location()
            grid, origin = self.get_obstacle_grid(vehicle_location)
            
            generation_time = time.time() - start_time
            logger.info(f"Obstacle grid generated in {generation_time:.2f} seconds")
            
            # === COORDINATE CONVERSION ===
            # Convert vehicle position to grid coordinates
            vehicle_loc = self.vehicle.get_location()
            start_x = int((vehicle_loc.x - origin.x) / self.GRID_SIZE)
            start_y = int((vehicle_loc.y - origin.y) / self.GRID_SIZE)
            
            # Clamp to grid boundaries
            start = (
                max(0, min(self.GRID_WIDTH - 1, start_x)), 
                max(0, min(self.GRID_HEIGHT - 1, start_y))
            )
            
            logger.info(f"Vehicle start position: world({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f}) "
                       f"-> grid{start}")
            
            # Validate start position
            if grid[start[0]][start[1]] == 1:
                logger.error("❌ Vehicle spawned inside obstacle! Trying to find nearby free space...")
                
                # Search for nearby free space
                found_free_space = False
                search_radius = 5
                
                for radius in range(1, search_radius + 1):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            new_x = start[0] + dx
                            new_y = start[1] + dy
                            
                            if (0 <= new_x < self.GRID_WIDTH and 
                                0 <= new_y < self.GRID_HEIGHT and
                                grid[new_x][new_y] == 0):
                                
                                start = (new_x, new_y)
                                logger.info(f"✓ Found free space at grid{start}")
                                found_free_space = True
                                break
                        if found_free_space:
                            break
                    if found_free_space:
                        break
                
                if not found_free_space:
                    logger.error("Could not find free space near vehicle!")
                    return
            
            # === GOAL SELECTION ===
            # Define goal position (you can modify these coordinates)
            # These coordinates work well for Town03
            goal_candidates = [
                (150, 150),  # Far corner
                (50, 150),   # Different corner
                (150, 50),   # Another corner
                (100, 100),  # Center area
                (75, 125),   # Intermediate position
            ]
            
            goal = None
            for candidate in goal_candidates:
                # Validate goal position
                if (0 <= candidate[0] < self.GRID_WIDTH and 
                    0 <= candidate[1] < self.GRID_HEIGHT and
                    grid[candidate[0]][candidate[1]] == 0):
                    
                    goal = candidate
                    logger.info(f"✓ Goal set to grid{goal}")
                    break
            
            if not goal:
                logger.error("❌ Could not find valid goal position!")
                return
            
            # Convert goal to world coordinates for logging
            goal_world = carla.Location(
                x=origin.x + goal[0] * self.GRID_SIZE,
                y=origin.y + goal[1] * self.GRID_SIZE,
                z=0
            )
            logger.info(f"Goal position: grid{goal} -> world({goal_world.x:.1f}, {goal_world.y:.1f})")
            
            # Calculate straight-line distance
            straight_distance = self.distance(start, goal) * self.GRID_SIZE
            logger.info(f"Straight-line distance to goal: {straight_distance:.1f} meters")
            
            # === PATH PLANNING ===
            logger.info("🧭 Starting path planning with RRT algorithm...")
            planning_start_time = time.time()
            
            path = self.rrt_path_plan(start, goal, grid)
            
            planning_time = time.time() - planning_start_time
            logger.info(f"Path planning completed in {planning_time:.2f} seconds")
            
            if not path:
                logger.error("❌ Path planning failed! No path found to goal.")
                return
            
            # Calculate path statistics
            path_length_grid = sum(self.distance(path[i], path[i+1]) for i in range(len(path)-1))
            path_length_meters = path_length_grid * self.GRID_SIZE
            efficiency = (straight_distance / path_length_meters) * 100
            
            logger.info(f"✓ Path found successfully!")
            logger.info(f"  - Waypoints: {len(path)}")
            logger.info(f"  - Path length: {path_length_meters:.1f} meters")
            logger.info(f"  - Efficiency: {efficiency:.1f}% (vs straight line)")
            
            # === VISUALIZATION ===
            logger.info("🎨 Creating visualization...")
            self.draw_debug_visualization(grid, origin, path)
            
            # === PATH EXECUTION ===
            logger.info("🚗 Starting autonomous navigation...")
            
            # Update spectator to follow vehicle during execution
            def update_spectator():
                """Update spectator position to follow vehicle"""
                while self.path_execution_active:
                    try:
                        if self.vehicle and self.vehicle.is_alive:
                            vehicle_transform = self.vehicle.get_transform()
                            spectator_transform = carla.Transform(
                                vehicle_transform.location + carla.Location(z=40),
                                carla.Rotation(pitch=-60)  # Slightly angled view
                            )
                            spectator.set_transform(spectator_transform)
                        time.sleep(0.5)  # Update every 0.5 seconds
                    except:
                        break
            
            # Start spectator update thread
            spectator_thread = threading.Thread(target=update_spectator, daemon=True)
            spectator_thread.start()
            
            # Execute the planned path
            execution_start_time = time.time()
            self.execute_path(path, origin)
            execution_time = time.time() - execution_start_time
            
            # === COMPLETION STATISTICS ===
            logger.info("🎉 Navigation completed!")
            logger.info(f"📊 Performance Summary:")
            logger.info(f"  - Total execution time: {execution_time:.1f} seconds")
            logger.info(f"  - Average speed: {(path_length_meters / execution_time):.1f} m/s")
            logger.info(f"  - Planning efficiency: {efficiency:.1f}%")
            
            # Final vehicle position
            final_location = self.vehicle.get_location()
            final_distance_to_goal = math.sqrt(
                (final_location.x - goal_world.x)**2 + 
                (final_location.y - goal_world.y)**2
            )
            logger.info(f"  - Final distance to goal: {final_distance_to_goal:.1f} meters")
            
            if final_distance_to_goal < self.WAYPOINT_TOLERANCE * 2:
                logger.info("✅ Goal reached successfully!")
            else:
                logger.warning("⚠️ Goal not fully reached, but navigation completed")
            
            # Keep visualization active for observation
            logger.info("Keeping visualization active for 30 seconds...")
            time.sleep(30.0)
            
        except KeyboardInterrupt:
            logger.info("🛑 Navigation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Critical error occurred: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Comprehensive cleanup of all CARLA resources
        Ensures no memory leaks or hanging resources
        """
        logger.info("🧹 Starting cleanup process...")
        
        try:
            # Stop vehicle if it exists
            if self.vehicle and self.vehicle.is_alive:
                logger.info("Stopping and destroying vehicle...")
                self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                time.sleep(0.5)  # Let vehicle stop
                self.vehicle.destroy()
                logger.info("✓ Vehicle destroyed")
            
            # Clean up debug visualizations
            self.cleanup_debug_objects()
            
            # Reset world settings if needed
            if self.world:
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self.world.apply_settings(settings)
                    logger.info("✓ World settings reset")
                except:
                    pass
            
            # Clear references
            self.vehicle = None
            self.world = None
            self.client = None
            self.path_execution_active = False
            
            logger.info("✅ Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """
    Main entry point for the CARLA pathfinding system
    """
    print("=" * 60)
    print("🚗 CARLA Advanced Pathfinding System")
    print("=" * 60)
    print("Features:")
    print("  • RRT pathfinding algorithm")
    print("  • Advanced obstacle detection")
    print("  • Smooth vehicle control")
    print("  • Real-time visualization")
    print("  • Comprehensive logging")
    print("=" * 60)
    
    # Create and run the pathfinding system
    planner = CarlaPathPlanner()
    
    try:
        planner.run()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
    finally:
        print("\n" + "=" * 60)
        print("🏁 CARLA Pathfinding System Terminated")
        print("=" * 60)

if __name__ == '__main__':
    main()

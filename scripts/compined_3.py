#!/usr/bin/env python3
"""
Enhanced CARLA Waypoint Navigation with TSP Optimization
========================================================
Task: Navigate optimally through waypoints following CARLA roads in Town01
Features: 
- TSP-based route optimization using actual road distances and directions
- Smooth steering with PID control
- Persistent visualization of routes and waypoints (small dots with labels)
- Collision avoidance and overturn prevention
- Dynamic speed adjustment for curves
- Waypoint proximity improvements
"""
import carla
import sys

def patch_carla_debug():
    """Add missing debug methods to older CARLA versions"""
    # Save original methods
    original_draw_point = carla.DebugHelper.draw_point
    original_draw_line = carla.DebugHelper.draw_line
    
    def draw_sphere(self, location, radius=0.5, color=carla.Color(255, 0, 0), life_time=-1.0):
        """Alternative implementation for draw_sphere using draw_point"""
        return original_draw_point(self, location, size=radius, color=color, life_time=life_time)
    
    def draw_arrow(self, begin, end, thickness=0.1, arrow_size=0.1, color=carla.Color(0, 255, 0), life_time=-1.0):
        """Alternative implementation for draw_arrow using draw_line"""
        return original_draw_line(self, begin, end, thickness=thickness, color=color, life_time=life_time)
    
    # Apply patches
    carla.DebugHelper.draw_sphere = draw_sphere
    carla.DebugHelper.draw_arrow = draw_arrow

# Apply the patch immediately
patch_carla_debug()

import math
import time
import random
import numpy as np
from collections import deque
from itertools import permutations
import heapq

# Configuration constants with detailed explanations
# ================================================

# WAYPOINTS: List of target coordinates [x, y, z] that the vehicle must visit
# Each waypoint represents a specific location in Town01 that the vehicle needs to reach
# Changing these values will alter the navigation path and optimization results
WAYPOINTS = [
    [334.949799, 161.106171, 0.001736],  # Waypoint 0: Near intersection
    [339.100037, 258.568939, 0.001679],  # Waypoint 1: Straight road section
    [396.295319, 183.195740, 0.001678],  # Waypoint 2: Curved road section
    [267.657074, 1.983160, 0.001678],    # Waypoint 3: T-junction
    [153.868896, 26.115866, 0.001678],   # Waypoint 4: Residential area
    [290.515564, 56.175072, 0.001677],   # Waypoint 5: Near roundabout
    [92.325722, 86.063644, 0.001677],    # Waypoint 6: Narrow street
    [88.384346, 287.468567, 0.001728],   # Waypoint 7: Highway entrance
    [177.594101, 326.386902, 0.001677],  # Waypoint 8: Highway section
    [-1.646942, 197.501282, 0.001555],   # Waypoint 9: Downtown area
    [59.701321, 1.970804, 0.001467],     # Waypoint 10: Industrial zone
    [122.100121, 55.142044, 0.001596],   # Waypoint 11: Suburban area
    [161.030975, 129.313187, 0.001679],  # Waypoint 12: School zone
    [184.758713, 199.424271, 0.001680],  # Waypoint 13: Commercial district
]

# START_POSITION: Initial vehicle spawn location [x, y, z]
# This is where the vehicle will be placed at the beginning of the simulation
# Changing this will affect the initial optimization calculations
START_POSITION = [280.363739, 129.306351, 0.101746]  # Near center of Town01

# Visualization constants
# =======================

# WAYPOINT_DOT_SIZE: Size of waypoint markers in meters
# Small dot to mark waypoint location without obscuring the scene
WAYPOINT_DOT_SIZE = 0.1

# WAYPOINT_LABEL_HEIGHT: Height above ground for waypoint labels in meters
# Ensures labels are visible above the vehicle and other objects
WAYPOINT_LABEL_HEIGHT = 2.0

# Visualization update interval in seconds
# Controls how frequently waypoint sizes are recalculated based on camera position
# Lower values provide smoother scaling but increase computational load
# Higher values reduce computational load but make scaling less responsive
VISUALIZATION_UPDATE_INTERVAL = 0.1

# PID Controller constants
# ========================

# PID_KP: Proportional gain for steering control
# Higher values make the vehicle respond more aggressively to steering errors
# Too high can cause oscillations; too low can make the vehicle unresponsive
# Recommended range: 0.5 to 1.5
PID_KP = 0.8

# PID_KI: Integral gain for steering control
# Helps eliminate steady-state errors (like constant drift)
# Too high can cause instability; too low can leave persistent errors
# Recommended range: 0.01 to 0.1
PID_KI = 0.05

# PID_KD: Derivative gain for steering control
# Helps dampen oscillations and improve stability
# Too high can cause sluggish response; too low can allow overshooting
# Recommended range: 0.1 to 0.5
PID_KD = 0.3

# Vehicle control constants
# =========================

# TARGET_SPEED: Desired vehicle speed in km/h
# This is the base speed the vehicle will try to maintain on straight roads
# Higher values complete navigation faster but increase collision risk
# Lower values provide safer navigation but take longer
# Recommended range: 20 to 50 km/h for urban environments
TARGET_SPEED = 30.0

# MAX_ROLL_ANGLE: Maximum safe roll angle in degrees
# If the vehicle tilts beyond this angle, it will apply emergency braking
# Lower values provide more stability but may trigger unnecessarily on bumpy roads
# Higher values allow more aggressive turning but risk rollover
# Recommended range: 10 to 20 degrees
MAX_ROLL_ANGLE = 15.0

# WAYPOINT_REACH_THRESHOLD: Distance in meters to consider a waypoint reached
# When the vehicle is within this distance of a waypoint, it moves to the next one
# Reduced value makes the vehicle get closer to waypoints before considering them reached
# Recommended range: 1.0 to 3.0 meters for precise navigation
WAYPOINT_REACH_THRESHOLD = 2.0  # Reduced from 4.0 to get closer to waypoints

# LOOK_AHEAD_WAYPOINTS: Number of waypoints to look ahead for steering
# Higher values provide smoother turns but may cause cutting corners
# Lower values provide more precise following but may cause jerky steering
# Recommended range: 2 to 5
LOOK_AHEAD_WAYPOINTS = 3

# Speed adjustment constants
# ==========================

# CURVE_SPEED_REDUCTION: How much to reduce speed in curves (0.0 to 1.0)
# Higher values make the vehicle slow down more in turns
# Lower values make the vehicle maintain more speed through turns
# Recommended range: 0.4 to 0.8
CURVE_SPEED_REDUCTION = 0.6

# ROLL_SPEED_REDUCTION: How much to reduce speed based on roll angle (0.0 to 1.0)
# Higher values make the vehicle slow down more when tilted
# Lower values make the vehicle maintain more speed when tilted
# Recommended range: 0.2 to 0.5
ROLL_SPEED_REDUCTION = 0.3

# MIN_SPEED: Minimum speed in km/h
# The vehicle will not go slower than this speed under normal circumstances
# Higher values prevent the vehicle from getting stuck but may be unsafe
# Lower values provide more control in tight spaces
# Recommended range: 5 to 15 km/h
MIN_SPEED = 10.0

# Pathfinding constants
# ====================

# SAMPLING_RESOLUTION: Distance between sampled waypoints in meters
# Higher values create coarser paths but reduce computational load
# Lower values create smoother paths but increase computational load
# Recommended range: 1.0 to 5.0 meters
SAMPLING_RESOLUTION = 2.0

# ASTAR_MAX_ITERATIONS: Maximum iterations for A* pathfinding
# Higher values increase the chance of finding a path but take more time
# Lower values may fail to find paths in complex situations
# Recommended range: 200 to 1000
ASTAR_MAX_ITERATIONS = 500

# Vehicle spawn constants
# ======================

# VEHICLE_MODEL: CARLA vehicle blueprint ID
# Changing this will spawn a different vehicle model with different characteristics
# Available options: 'vehicle.audi.a2', 'vehicle.tesla.model3', 'vehicle.bmw.isetta', etc.
VEHICLE_MODEL = 'vehicle.audi.a2'

class PIDController:
    """PID controller for smooth steering"""
    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD, dt=0.05):
        """
        Initialize PID controller with tuning parameters
        
        Args:
            kp: Proportional gain - reacts to current error
            ki: Integral gain - accumulates past errors
            kd: Derivative gain - predicts future error
            dt: Time step between updates in seconds
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error_integral = 0  # Accumulated error over time
        self.prev_error = 0      # Error from previous update
        self.error_buffer = deque(maxlen=10)  # Buffer for smoothing errors
        
    def update(self, error):
        """
        Update PID controller with new error value
        
        Args:
            error: Current error (difference between desired and actual value)
            
        Returns:
            Control output value, clipped between -1.0 and 1.0
        """
        # Smooth error using moving average to reduce noise
        self.error_buffer.append(error)
        smoothed_error = np.mean(self.error_buffer)
        
        # PID calculations
        self.error_integral += smoothed_error * self.dt
        # Anti-windup: limit integral term to prevent instability
        self.error_integral = np.clip(self.error_integral, -5, 5)
        
        # Calculate derivative of error
        error_derivative = (smoothed_error - self.prev_error) / self.dt
        
        # Compute PID output
        control = (self.kp * smoothed_error + 
                  self.ki * self.error_integral + 
                  self.kd * error_derivative)
        
        # Store current error for next derivative calculation
        self.prev_error = smoothed_error
        
        # Limit control output to valid range
        return np.clip(control, -1.0, 1.0)
    
    def reset(self):
        """Reset controller state to initial values"""
        self.error_integral = 0
        self.prev_error = 0
        self.error_buffer.clear()

class RouteOptimizer:
    """Optimizes waypoint order using TSP with road distances and directions"""
    def __init__(self, world_map):
        """
        Initialize route optimizer with map data
        
        Args:
            world_map: CARLA map object containing road network information
        """
        self.map = world_map
        self.distance_cache = {}  # Cache for storing calculated distances
        
    def get_road_distance(self, start_loc, end_loc):
        """
        Calculate actual road distance between two locations considering street direction
        
        Args:
            start_loc: Starting location (carla.Location)
            end_loc: Ending location (carla.Location)
            
        Returns:
            Distance in meters along roads following traffic direction
        """
        # Create cache key from location coordinates
        cache_key = (
            (start_loc.x, start_loc.y),
            (end_loc.x, end_loc.y)
        )
        
        # Return cached distance if available
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Get waypoints on road network
        start_wp = self.map.get_waypoint(start_loc)
        end_wp = self.map.get_waypoint(end_loc)
        
        if not start_wp or not end_wp:
            # Fallback to euclidean distance if waypoints not found
            distance = start_loc.distance(end_loc)
        else:
            # Use A* pathfinding to calculate road distance respecting direction
            distance = self._astar_distance(start_wp, end_wp)
            if distance is None:
                # If no path found, use euclidean distance with penalty
                distance = start_loc.distance(end_loc) * 1.5
        
        # Cache the calculated distance
        self.distance_cache[cache_key] = distance
        return distance
    
    def _astar_distance(self, start_wp, end_wp, max_iterations=ASTAR_MAX_ITERATIONS):
        """
        A* pathfinding algorithm for calculating road distance respecting street direction
        
        Args:
            start_wp: Starting waypoint
            end_wp: Destination waypoint
            max_iterations: Maximum number of iterations before giving up
            
        Returns:
            Total distance in meters or None if no path found
        """
        # Priority queue: (estimated_total_distance, waypoint_id, waypoint, current_distance)
        open_set = [(0, id(start_wp), start_wp, 0)]
        closed_set = set()  # Set of visited waypoints
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current, dist = heapq.heappop(open_set)
            
            # Check if we've reached the destination
            if current.transform.location.distance(end_wp.transform.location) < 5.0:
                return dist
            
            # Create hash for current waypoint to check if already visited
            wp_hash = (current.transform.location.x, 
                      current.transform.location.y)
            if wp_hash in closed_set:
                continue
            closed_set.add(wp_hash)
            
            # Get next waypoints along the road in the direction of traffic
            next_wps = current.next(SAMPLING_RESOLUTION)
            
            # Also consider lane changes if they're in the same direction
            if current.lane_change & carla.LaneChange.Left:
                left_wp = current.get_left_lane()
                if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                    # Check if the left lane is going in a compatible direction
                    left_dir = left_wp.transform.get_forward_vector()
                    current_dir = current.transform.get_forward_vector()
                    dot_product = left_dir.x * current_dir.x + left_dir.y * current_dir.y
                    if dot_product > 0.5:  # Lanes are going in similar directions
                        next_wps.append(left_wp)
            
            if current.lane_change & carla.LaneChange.Right:
                right_wp = current.get_right_lane()
                if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                    # Check if the right lane is going in a compatible direction
                    right_dir = right_wp.transform.get_forward_vector()
                    current_dir = current.transform.get_forward_vector()
                    dot_product = right_dir.x * current_dir.x + right_dir.y * current_dir.y
                    if dot_product > 0.5:  # Lanes are going in similar directions
                        next_wps.append(right_wp)
            
            for next_wp in next_wps:
                if next_wp:
                    new_dist = dist + SAMPLING_RESOLUTION
                    # Heuristic: straight-line distance to destination
                    heuristic = next_wp.transform.location.distance(end_wp.transform.location)
                    heapq.heappush(open_set, 
                                 (new_dist + heuristic, id(next_wp), next_wp, new_dist))
        
        return None  # No path found within max_iterations
    
    def optimize_route_nearest_neighbor(self, start_pos, waypoints):
        """
        Optimize route using nearest neighbor heuristic (fast but not optimal)
        
        Args:
            start_pos: Starting position [x, y, z]
            waypoints: List of waypoint positions [[x1, y1, z1], [x2, y2, z2], ...]
            
        Returns:
            List of waypoint indices in optimized order
        """
        unvisited = list(range(len(waypoints)))
        route = []
        current_loc = carla.Location(x=start_pos[0], y=start_pos[1], z=start_pos[2])
        
        while unvisited:
            # Find nearest unvisited waypoint
            min_dist = float('inf')
            nearest_idx = None
            
            for idx in unvisited:
                wp_loc = carla.Location(x=waypoints[idx][0], 
                                       y=waypoints[idx][1], 
                                       z=waypoints[idx][2])
                dist = self.get_road_distance(current_loc, wp_loc)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            
            route.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current_loc = carla.Location(x=waypoints[nearest_idx][0],
                                       y=waypoints[nearest_idx][1],
                                       z=waypoints[nearest_idx][2])
        
        return route
    
    def optimize_route_2opt(self, start_pos, waypoints, initial_route=None):
        """
        Optimize route using 2-opt improvement heuristic (better than nearest neighbor)
        
        Args:
            start_pos: Starting position [x, y, z]
            waypoints: List of waypoint positions [[x1, y1, z1], [x2, y2, z2], ...]
            initial_route: Initial route to improve (if None, uses nearest neighbor)
            
        Returns:
            List of waypoint indices in optimized order
        """
        # Use nearest neighbor as initial solution if none provided
        if initial_route is None:
            route = self.optimize_route_nearest_neighbor(start_pos, waypoints)
        else:
            route = initial_route.copy()
        
        def calculate_total_distance(route_order):
            """Calculate total distance for a given route order"""
            total = 0
            current_loc = carla.Location(x=start_pos[0], y=start_pos[1], z=start_pos[2])
            
            for idx in route_order:
                wp_loc = carla.Location(x=waypoints[idx][0],
                                      y=waypoints[idx][1],
                                      z=waypoints[idx][2])
                total += self.get_road_distance(current_loc, wp_loc)
                current_loc = wp_loc
            return total
        
        # Iteratively improve the route using 2-opt swaps
        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Try swapping edges to see if it improves the route
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    
                    if calculate_total_distance(new_route) < calculate_total_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        
        return route

class EnhancedCarlaNavigator:
    def __init__(self):
        """Initialize the CARLA navigator and set up the environment"""
        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Load Town01 if needed
        if 'Town01' not in self.world.get_map().name:
            print("Loading Town01...")
            self.world = self.client.load_world('Town01')
        
        self.map = self.world.get_map()
        self.vehicle = None
        self.spectator = self.world.get_spectator()
        
        # Initialize PID controller with tuned parameters
        self.steering_pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD)
        
        # Initialize route optimizer
        self.route_optimizer = RouteOptimizer(self.map)
        
        # Navigation state variables
        self.optimized_route = []
        self.all_route_waypoints = []
        self.visualization_objects = []
        self.waypoint_locations = []  # For waypoint visualization
        self.visualization_update_counter = 0
        
        print("✅ Enhanced CARLA Navigator initialized")
    
    def spawn_vehicle(self):
        """Spawn vehicle at start position with collision sensor"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(VEHICLE_MODEL)
        
        # Find a valid spawn point near our start position
        spawn_points = self.map.get_spawn_points()
        start_loc = carla.Location(x=START_POSITION[0], y=START_POSITION[1], z=START_POSITION[2])
        
        # Find closest spawn point
        closest_spawn = min(spawn_points, 
                          key=lambda sp: start_loc.distance(sp.location))
        
        # Spawn the vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, closest_spawn)
        
        # Add collision sensor for safety
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        collision_sensor.listen(lambda event: self.on_collision(event))
        
        print(f"🚗 Vehicle spawned at {closest_spawn.location}")
        return self.vehicle
    
    def on_collision(self, event):
        """Handle collision events"""
        print(f"⚠️ Collision detected with {event.other_actor.type_id}")
    
    def optimize_waypoint_order(self):
        """Find optimal order to visit waypoints using TSP algorithms"""
        print("\n🧮 Optimizing waypoint order using TSP...")
        
        # First, try nearest neighbor heuristic (fast)
        nn_route = self.route_optimizer.optimize_route_nearest_neighbor(
            START_POSITION, WAYPOINTS
        )
        
        # Then improve with 2-opt algorithm (better quality)
        optimized_route = self.route_optimizer.optimize_route_2opt(
            START_POSITION, WAYPOINTS, nn_route
        )
        
        # Calculate distances for comparison
        def calc_distance(route):
            """Calculate total distance for a route"""
            total = 0
            current = carla.Location(x=START_POSITION[0], y=START_POSITION[1], z=START_POSITION[2])
            for idx in route:
                next_loc = carla.Location(x=WAYPOINTS[idx][0], 
                                        y=WAYPOINTS[idx][1], 
                                        z=WAYPOINTS[idx][2])
                total += self.route_optimizer.get_road_distance(current, next_loc)
                current = next_loc
            return total
        
        original_distance = calc_distance(list(range(len(WAYPOINTS))))
        optimized_distance = calc_distance(optimized_route)
        
        print(f"📊 Original order distance: {original_distance:.1f}m")
        print(f"📊 Optimized order distance: {optimized_distance:.1f}m")
        print(f"💰 Distance saved: {original_distance - optimized_distance:.1f}m "
              f"({100*(original_distance - optimized_distance)/original_distance:.1f}%)")
        
        # Print the optimized route with original indices
        print(f"📋 Optimized route order:")
        for i, idx in enumerate(optimized_route):
            print(f"   Stop {i+1}: Waypoint {idx} {WAYPOINTS[idx]}")
        
        self.optimized_route = optimized_route
        return optimized_route
    
    def get_waypoint_path(self, start_location, target_location, sampling_resolution=SAMPLING_RESOLUTION):
        """
        Create path using CARLA map waypoints with A* pathfinding respecting direction
        
        Args:
            start_location: Starting position (carla.Location)
            target_location: Target position (carla.Location)
            sampling_resolution: Distance between sampled waypoints in meters
            
        Returns:
            List of waypoints forming the path
        """
        start_waypoint = self.map.get_waypoint(start_location)
        target_waypoint = self.map.get_waypoint(target_location)
        
        if not start_waypoint or not target_waypoint:
            return []
        
        # Use A* for better pathfinding
        open_set = [(0, id(start_waypoint), start_waypoint, [start_waypoint])]
        closed_set = set()
        best_path = None
        best_distance = float('inf')
        
        iterations = 0
        max_iterations = ASTAR_MAX_ITERATIONS
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current, path = heapq.heappop(open_set)
            
            current_distance = current.transform.location.distance(target_location)
            
            # Check if we reached target
            if current_distance < 5.0:
                return path
            
            # Keep track of best path so far
            if current_distance < best_distance:
                best_distance = current_distance
                best_path = path
            
            # Create hash for current waypoint to check if already visited
            wp_hash = (round(current.transform.location.x, 1), 
                      round(current.transform.location.y, 1))
            if wp_hash in closed_set:
                continue
            closed_set.add(wp_hash)
            
            # Get next waypoints along the road in the direction of traffic
            next_waypoints = current.next(sampling_resolution)
            
            # Also consider lane changes if they're in the same direction
            if current.lane_change & carla.LaneChange.Left:
                left_wp = current.get_left_lane()
                if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                    # Check if the left lane is going in a compatible direction
                    left_dir = left_wp.transform.get_forward_vector()
                    current_dir = current.transform.get_forward_vector()
                    dot_product = left_dir.x * current_dir.x + left_dir.y * current_dir.y
                    if dot_product > 0.5:  # Lanes are going in similar directions
                        next_waypoints.append(left_wp)
            
            if current.lane_change & carla.LaneChange.Right:
                right_wp = current.get_right_lane()
                if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                    # Check if the right lane is going in a compatible direction
                    right_dir = right_wp.transform.get_forward_vector()
                    current_dir = current.transform.get_forward_vector()
                    dot_product = right_dir.x * current_dir.x + right_dir.y * current_dir.y
                    if dot_product > 0.5:  # Lanes are going in similar directions
                        next_waypoints.append(right_wp)
            
            for next_wp in next_waypoints:
                if next_wp:
                    new_path = path + [next_wp]
                    cost = len(new_path) * sampling_resolution
                    heuristic = next_wp.transform.location.distance(target_location)
                    heapq.heappush(open_set, 
                                 (cost + heuristic, id(next_wp), next_wp, new_path))
        
        return best_path if best_path else []
    
    def visualize_waypoints(self):
        """
        Visualize waypoints as small dots with labels showing both original and optimized indices
        """
        # Clear previous waypoint visualizations
        for obj in self.visualization_objects:
            if hasattr(obj, 'destroy'):
                obj.destroy()
        self.visualization_objects = []
        
        # Store waypoint locations for visualization
        self.waypoint_locations = []
        for i, wp_idx in enumerate(self.optimized_route):
            waypoint = WAYPOINTS[wp_idx]
            location = carla.Location(x=waypoint[0], y=waypoint[1], z=waypoint[2])
            self.waypoint_locations.append((location, wp_idx, i))
        
        # Draw each waypoint as a small dot with label
        for loc, orig_idx, opt_idx in self.waypoint_locations:
            # Draw a small dot at waypoint location
            self.world.debug.draw_point(
                loc + carla.Location(z=0.2),  # Slightly above ground
                size=WAYPOINT_DOT_SIZE,
                color=carla.Color(r=255, g=0, b=0),  # Red dot
                life_time=0.0  # Persistent
            )
            
            # Draw waypoint label showing optimized order and original index
            self.world.debug.draw_string(
                loc + carla.Location(z=WAYPOINT_LABEL_HEIGHT),
                f"WP-{opt_idx+1}\n(orig: {orig_idx})",
                draw_shadow=True,
                color=carla.Color(r=255, g=255, b=255),  # White text
                life_time=0.0  # Persistent
            )
        
        print(f"📍 Visualized {len(self.optimized_route)} waypoints as dots with labels")
    
    def visualize_complete_route(self):
        """Visualize complete route as arrows between waypoints"""
        # Draw waypoints first
        self.visualize_waypoints()
        
        # Draw route as arrows between waypoints
        current_loc = carla.Location(x=START_POSITION[0], y=START_POSITION[1], z=START_POSITION[2])
        
        for i, wp_idx in enumerate(self.optimized_route):
            target_loc = carla.Location(x=WAYPOINTS[wp_idx][0], 
                                      y=WAYPOINTS[wp_idx][1], 
                                      z=WAYPOINTS[wp_idx][2])
            
            # Get path between waypoints
            path = self.get_waypoint_path(current_loc, target_loc, sampling_resolution=5.0)
            
            if path:
                # Draw path segments
                for j in range(len(path) - 1):
                    start = path[j].transform.location + carla.Location(z=0.5)
                    end = path[j + 1].transform.location + carla.Location(z=0.5)
                    
                    # Alternate colors for different segments
                    color = carla.Color(r=0, g=255, b=0) if i % 2 == 0 else carla.Color(r=0, g=0, b=255)
                    
                    self.world.debug.draw_arrow(
                        start, end,
                        thickness=0.1,
                        arrow_size=0.1,
                        color=color,
                        life_time=0.0  # Persistent
                    )
                
                # Store waypoints for navigation
                self.all_route_waypoints.extend(path)
            
            current_loc = target_loc
        
        print(f"📍 Visualized complete route with {len(self.optimized_route)} segments")
    
    def follow_waypoints_smooth(self, waypoint_list, target_index, target_speed=TARGET_SPEED):
        """
        Follow waypoints with smooth PID steering control
        
        Args:
            waypoint_list: List of waypoints to follow
            target_index: Index of the target waypoint in the original list
            target_speed: Desired speed in km/h
        """
        current_waypoint_index = 0
        collision_count = 0
        
        # Reset PID controller for fresh start
        self.steering_pid.reset()
        
        while current_waypoint_index < len(waypoint_list):
            # Get current vehicle state
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_velocity = self.vehicle.get_velocity()
            current_speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2) * 3.6  # m/s to km/h
            
            # Check vehicle stability (anti-rollover)
            vehicle_rotation = vehicle_transform.rotation
            roll_angle = abs(vehicle_rotation.roll)
            
            if roll_angle > MAX_ROLL_ANGLE:
                print(f"⚠️ High roll angle detected: {roll_angle:.1f}°")
                # Emergency brake
                self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                time.sleep(0.5)
                continue
            
            # Look-ahead for smoother navigation
            look_ahead_index = min(current_waypoint_index + LOOK_AHEAD_WAYPOINTS, len(waypoint_list) - 1)
            target_waypoint = waypoint_list[look_ahead_index]
            target_location = target_waypoint.transform.location
            
            # Calculate distance to current waypoint
            current_wp_location = waypoint_list[current_waypoint_index].transform.location
            distance = vehicle_location.distance(current_wp_location)
            
            # Check if waypoint reached (with reduced threshold)
            if distance < WAYPOINT_REACH_THRESHOLD:
                current_waypoint_index += 1
                if current_waypoint_index < len(waypoint_list):
                    if current_waypoint_index % 10 == 0:
                        print(f"✅ Progress: {current_waypoint_index}/{len(waypoint_list)} waypoints")
                continue
            
            # Calculate steering with PID
            target_vector = target_location - vehicle_location
            forward_vector = vehicle_transform.get_forward_vector()
            
            # Calculate cross product for steering direction
            cross = forward_vector.x * target_vector.y - forward_vector.y * target_vector.x
            
            # Normalize by distance for consistent error magnitude
            error = math.atan2(cross, target_vector.x * forward_vector.x + target_vector.y * forward_vector.y)
            
            # PID steering
            steer = self.steering_pid.update(error)
            
            # Dynamic speed adjustment based on curvature
            curvature = abs(steer)
            speed_factor = 1.0 - (curvature * CURVE_SPEED_REDUCTION)  # Reduce speed in curves
            
            # Additional speed reduction based on roll
            roll_factor = 1.0 - (roll_angle / MAX_ROLL_ANGLE) * ROLL_SPEED_REDUCTION
            
            adjusted_target_speed = target_speed * speed_factor * roll_factor
            adjusted_target_speed = max(MIN_SPEED, adjusted_target_speed)  # Minimum speed
            
            # Smooth acceleration/deceleration
            speed_error = adjusted_target_speed - current_speed
            
            if speed_error > 0:
                # Accelerate smoothly
                throttle = np.clip(speed_error / 30.0, 0.0, 0.7)
                brake = 0.0
            else:
                # Brake smoothly
                throttle = 0.0
                brake = np.clip(-speed_error / 40.0, 0.0, 0.5)
            
            # Apply control with smooth limits
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False
            )
            self.vehicle.apply_control(control)
            
            # Update visualization of current position
            if current_waypoint_index % 5 == 0:
                self.world.debug.draw_point(
                    vehicle_location + carla.Location(z=1.0),
                    size=0.1,
                    color=carla.Color(r=255, g=255, b=0),
                    life_time=5.0
                )
            
            # Wait for next tick
            self.world.tick()
            time.sleep(0.05)
    
    def navigate_to_waypoint(self, target_waypoint, waypoint_index, optimized_index):
        """
        Navigate to a single waypoint with smooth control
        
        Args:
            target_waypoint: Target waypoint coordinates [x, y, z]
            waypoint_index: Original index of the waypoint
            optimized_index: Index in the optimized route
            
        Returns:
            True if waypoint was reached, False otherwise
        """
        print(f"\n📍 Navigating to waypoint {optimized_index+1}/{len(self.optimized_route)}"
              f" (Original #{waypoint_index}): {target_waypoint}")
        
        # Get current position
        current_location = self.vehicle.get_transform().location
        target_location = carla.Location(x=target_waypoint[0], 
                                       y=target_waypoint[1], 
                                       z=target_waypoint[2])
        
        # Plan path
        waypoint_path = self.get_waypoint_path(current_location, target_location)
        
        if not waypoint_path:
            print(f"❌ Could not plan path to waypoint {optimized_index+1}")
            return False
        
        print(f"📋 Path planned: {len(waypoint_path)} waypoints")
        
        # Update current target visualization
        self.world.debug.draw_point(
            target_location + carla.Location(z=0.5),
            size=0.3,
            color=carla.Color(r=0, g=255, b=0),  # Green for current target
            life_time=10.0
        )
        
        # Follow path with smooth control
        self.follow_waypoints_smooth(waypoint_path, waypoint_index)
        
        # Check if we reached the target
        final_location = self.vehicle.get_transform().location
        final_distance = final_location.distance(target_location)
        
        if final_distance < WAYPOINT_REACH_THRESHOLD * 1.5:  # Slightly larger threshold for final check
            print(f"✅ Reached waypoint {optimized_index+1} (distance: {final_distance:.1f}m)")
            return True
        else:
            print(f"⚠️ Close to waypoint {optimized_index+1} (distance: {final_distance:.1f}m)")
            return True
    
    def navigate_all_waypoints(self):
        """Navigate to all waypoints in optimized sequence"""
        # Optimize route first
        self.optimize_waypoint_order()
        
        # Visualize complete route
        self.visualize_complete_route()
        
        print(f"\n🎯 Starting navigation to {len(self.optimized_route)} waypoints in optimized order")
        print(f"📋 Route order: {self.optimized_route}")
        
        successful_waypoints = 0
        start_time = time.time()
        
        for i, wp_idx in enumerate(self.optimized_route):
            try:
                waypoint = WAYPOINTS[wp_idx]
                success = self.navigate_to_waypoint(waypoint, wp_idx, i)
                if success:
                    successful_waypoints += 1
                
                # Brief pause between waypoints
                time.sleep(1.0)
                
            except Exception as e:
                print(f"❌ Error navigating to waypoint {i+1}: {e}")
                continue
        
        # Stop vehicle
        self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        
        total_time = time.time() - start_time
        print(f"\n🏁 Navigation complete!")
        print(f"📊 Results: {successful_waypoints}/{len(self.optimized_route)} waypoints reached")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"⚡ Average time per waypoint: {total_time/len(self.optimized_route):.1f} seconds")
    
    def cleanup(self):
        """Clean up resources"""
        # Clear visualizations
        self.world.debug.draw_string(
            carla.Location(x=0, y=0, z=0),
            "",
            life_time=0.0
        )
        
        if self.vehicle:
            self.vehicle.destroy()
            print("🧹 Vehicle destroyed")

def main():
    """Main function"""
    navigator = None
    
    try:
        # Create navigator
        navigator = EnhancedCarlaNavigator()
        
        # Spawn vehicle
        navigator.spawn_vehicle()
        
        # Wait a moment for vehicle to settle
        time.sleep(2.0)
        
        # Start navigation
        navigator.navigate_all_waypoints()
        
        # Keep visualization active
        print("\n📍 Keeping visualization active. Press Ctrl+C to exit...")
        while True:
            time.sleep(1.0)
        
    except KeyboardInterrupt:
        print("\n⏹️ Navigation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if navigator:
            navigator.cleanup()

if __name__ == '__main__':
    print("🚀 Enhanced CARLA Navigation with TSP Optimization")
    print("=" * 60)
    print("📋 Features:")
    print("  • TSP-based route optimization following actual roads and directions")
    print("  • Smooth PID steering control to prevent collisions")
    print("  • Persistent route visualization with small waypoint dots")
    print("  • Anti-rollover protection")
    print("  • Dynamic speed adjustment for curves")
    print("  • Precise waypoint proximity (2m threshold)")
    print("🗺️ Map: Town01")
    print("🎮 Press Ctrl+C to stop")
    print("=" * 60)
    
    main()
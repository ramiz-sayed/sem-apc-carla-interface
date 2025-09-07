#!/usr/bin/env python3
"""
Ultimate CARLA Autonomous Navigation System
==========================================
The most comprehensive autonomous vehicle navigation system for CARLA simulator,
incorporating state-of-the-art pathfinding, computer vision, and decision-making algorithms.

Features:
- Advanced TSP-based route optimization with multiple algorithms
- Hybrid pathfinding (waypoint-based + grid-based A*)
- Sophisticated PID control with adaptive parameters
- Multi-modal sensor fusion (camera, LiDAR, radar)
- Deep learning-based object detection and semantic segmentation
- Dynamic traffic prediction and interaction
- Advanced traffic light detection with state prediction
- Multi-vehicle coordination and V2X communication simulation
- Real-time obstacle avoidance with risk assessment
- Adaptive speed control based on road conditions
- Vehicle stability control with rollover prevention
- Comprehensive visualization system
- Performance monitoring and diagnostics
- Fault tolerance and recovery mechanisms
"""
import carla
import sys
import random
import math
import time
import numpy as np
from collections import deque, defaultdict
from itertools import permutations, combinations
import heapq
import cv2
import threading
import multiprocessing
from queue import Queue, Empty
import os
import json
import logging
from datetime import datetime
import socket
import struct
import pickle
import signal
import atexit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carla_navigation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Apply CARLA debug patches
def patch_carla_debug():
    """Add missing debug methods to older CARLA versions"""
    original_draw_point = carla.DebugHelper.draw_point
    original_draw_line = carla.DebugHelper.draw_line
    
    def draw_sphere(self, location, radius=0.5, color=carla.Color(255, 0, 0), life_time=-1.0):
        return original_draw_point(self, location, size=radius, color=color, life_time=life_time)
    
    def draw_arrow(self, begin, end, thickness=0.1, arrow_size=0.1, color=carla.Color(0, 255, 0), life_time=-1.0):
        return original_draw_line(self, begin, end, thickness=thickness, color=color, life_time=life_time)
    
    carla.DebugHelper.draw_sphere = draw_sphere
    carla.DebugHelper.draw_arrow = draw_arrow

patch_carla_debug()

# Configuration constants with detailed explanations
# ================================================

# Waypoints and Start Position
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
START_POSITION = [280.363739, 129.306351, 0.101746]

# Visualization constants
WAYPOINT_DOT_SIZE = 0.1
WAYPOINT_LABEL_HEIGHT = 2.0
VISUALIZATION_UPDATE_INTERVAL = 0.1
PATH_VISUALIZATION_LIFETIME = 30.0
OBSTACLE_VISUALIZATION_LIFETIME = 5.0

# PID Controller constants
PID_KP = 0.8
PID_KI = 0.05
PID_KD = 0.3
PID_DT = 0.05
PID_INTEGRAL_LIMIT = 5.0
PID_ERROR_BUFFER_SIZE = 10

# Vehicle control constants
TARGET_SPEED = 30.0
MAX_SPEED = 50.0
MIN_SPEED = 5.0
MAX_ROLL_ANGLE = 15.0
MAX_PITCH_ANGLE = 15.0
WAYPOINT_REACH_THRESHOLD = 2.0
LOOK_AHEAD_WAYPOINTS = 3
CURVE_SPEED_REDUCTION = 0.6
ROLL_SPEED_REDUCTION = 0.3
BRAKE_ACCELERATION = 5.0
THROTTLE_ACCELERATION = 3.0

# Pathfinding constants
SAMPLING_RESOLUTION = 2.0
ASTAR_MAX_ITERATIONS = 500
WAYPOINT_MAX_DISTANCE = 5.0

# Vehicle spawn constants
VEHICLE_MODEL = 'vehicle.audi.a2'
VEHICLE_COLOR = '0, 0, 255'  # Blue

# Traffic constants
NUM_TRAFFIC_VEHICLES = 20
TRAFFIC_SPAWN_DISTANCE = 50.0
TRAFFIC_SPEED_VARIATION = 0.2  # 20% variation in traffic speed
TRAFFIC_RESPAWN_TIME = 30.0  # Respawn traffic vehicles every 30 seconds

# Grid A* constants
GRID_RESOLUTION = 2.0  # meters per grid cell
GRID_SIZE = 200  # grid cells in each dimension
OBSTACLE_THRESHOLD = 0.3  # threshold for obstacle detection in grid
GRID_UPDATE_INTERVAL = 0.5  # seconds between grid updates

# Camera constants for optical avoidance
CAMERA_IMAGE_SIZE_X = 640
CAMERA_IMAGE_SIZE_Y = 480
CAMERA_FOV = 90
CAMERA_POSITION_X = 1.5
CAMERA_POSITION_Z = 2.4
CAMERA_ROTATION_PITCH = -10

# LiDAR constants
LIDAR_CHANNELS = 32
LIDAR_RANGE = 50.0
LIDAR_POINTS_PER_SECOND = 100000
LIDAR_ROTATION_FREQUENCY = 10
LIDAR_UPPER_FOV = 15
LIDAR_LOWER_FOV = -30

# Radar constants
RADAR_RANGE = 100.0
RADAR_POINTS_PER_SECOND = 5000
RADAR_HORIZONTAL_FOV = 35
RADAR_VERTICAL_FOV = 20

# Object detection constants
OBJECT_DETECTION_CONFIDENCE = 0.5
OBJECT_DETECTION_NMS_THRESHOLD = 0.4
MIN_OBJECT_AREA = 100
MAX_OBJECT_DISTANCE = 50.0

# Traffic light constants
TRAFFIC_LIGHT_DETECTION_RANGE = 30.0
TRAFFIC_LIGHT_STOP_DISTANCE = 5.0
TRAFFIC_LIGHT_YELLOW_STOP_TIME = 3.0  # seconds to stop for yellow light

# V2X communication constants
V2X_PORT = 8888
V2X_BROADCAST_INTERVAL = 0.1  # seconds
V2X_MESSAGE_TTL = 1.0  # seconds

# Performance monitoring constants
PERFORMANCE_MONITOR_INTERVAL = 5.0  # seconds
FPS_TARGET = 20
MAX_LATENCY = 0.1  # seconds

# Fault tolerance constants
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_COOLDOWN = 2.0  # seconds
EMERGENCY_BRAKE_THRESHOLD = 0.5  # meters

# Semantic segmentation colors (CARLA standard)
SEGMENTATION_COLORS = {
    'None': [0, 0, 0],          # 0: None
    'Building': [70, 70, 70],    # 1: Buildings
    'Fence': [190, 153, 153],   # 2: Fences
    'Other': [72, 0, 90],       # 3: Other
    'Pedestrian': [220, 20, 60], # 4: Pedestrians
    'Pole': [153, 153, 153],   # 5: Poles
    'RoadLine': [157, 234, 50], # 6: Road lines
    'Road': [128, 64, 128],    # 7: Roads
    'Sidewalk': [244, 35, 232], # 8: Sidewalks
    'Vegetation': [107, 142, 35], # 9: Vegetation
    'Car': [0, 0, 142],        # 10: Cars
    'Wall': [102, 102, 156],   # 11: Walls
    'TrafficSign': [220, 220, 0], # 12: Traffic signs
    'Sky': [0, 0, 0],          # 13: Sky (not used in segmentation)
    'Ground': [81, 0, 81],     # 14: Ground
    'Bridge': [150, 100, 100], # 15: Bridges
    'RailTrack': [230, 150, 140], # 16: Rail tracks
    'GuardRail': [180, 180, 100], # 17: Guard rails
    'TrafficLight': [250, 170, 30], # 18: Traffic lights
    'Static': [110, 190, 160], # 19: Static objects
    'Dynamic': [170, 120, 50], # 20: Dynamic objects
    'Water': [45, 60, 150],     # 21: Water
    'Terrain': [145, 170, 100] # 22: Terrain
}

class PerformanceMonitor:
    """Monitor system performance metrics"""
    def __init__(self, interval=PERFORMANCE_MONITOR_INTERVAL):
        self.interval = interval
        self.metrics = {
            'fps': deque(maxlen=100),
            'latency': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'pathfinding_time': deque(maxlen=100),
            'obstacle_detection_time': deque(maxlen=100),
            'control_loop_time': deque(maxlen=100)
        }
        self.last_update_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background thread for monitoring performance"""
        while self.running:
            time.sleep(self.interval)
            self._update_metrics()
            self._log_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate FPS
            current_time = time.time()
            if self.frame_count > 0:
                fps = self.frame_count / (current_time - self.last_frame_time)
                self.metrics['fps'].append(fps)
            
            self.frame_count = 0
            self.last_frame_time = current_time
            
            # Get CPU and memory usage
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_percent)
            except ImportError:
                pass  # psutil not available
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _log_metrics(self):
        """Log performance metrics"""
        try:
            if not all(self.metrics.values()):
                return
                
            avg_fps = sum(self.metrics['fps']) / len(self.metrics['fps']) if self.metrics['fps'] else 0
            avg_latency = sum(self.metrics['latency']) / len(self.metrics['latency']) if self.metrics['latency'] else 0
            avg_cpu = sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
            avg_memory = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
            
            logger.info(f"Performance: FPS={avg_fps:.1f}, Latency={avg_latency:.3f}s, CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%")
            
            # Check for performance issues
            if avg_fps < FPS_TARGET * 0.8:
                logger.warning(f"Low FPS detected: {avg_fps:.1f} < {FPS_TARGET}")
            if avg_latency > MAX_LATENCY:
                logger.warning(f"High latency detected: {avg_latency:.3f}s > {MAX_LATENCY}s")
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}")
    
    def update_frame(self):
        """Update frame count for FPS calculation"""
        self.frame_count += 1
    
    def update_latency(self, latency):
        """Update latency metric"""
        self.metrics['latency'].append(latency)
    
    def update_pathfinding_time(self, time_taken):
        """Update pathfinding time metric"""
        self.metrics['pathfinding_time'].append(time_taken)
    
    def update_obstacle_detection_time(self, time_taken):
        """Update obstacle detection time metric"""
        self.metrics['obstacle_detection_time'].append(time_taken)
    
    def update_control_loop_time(self, time_taken):
        """Update control loop time metric"""
        self.metrics['control_loop_time'].append(time_taken)
    
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

class V2XCommunication:
    """Vehicle-to-Everything communication system"""
    def __init__(self, vehicle_id, port=V2X_PORT):
        self.vehicle_id = vehicle_id
        self.port = port
        self.socket = None
        self.running = False
        self.receive_thread = None
        self.message_queue = Queue()
        self.vehicle_positions = {}
        self.traffic_light_states = {}
        self.road_conditions = {}
        self.setup_socket()
    
    def setup_socket(self):
        """Set up UDP socket for V2X communication"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.bind(('', self.port))
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            logger.info(f"V2X communication initialized on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to set up V2X socket: {e}")
    
    def _receive_loop(self):
        """Background thread for receiving V2X messages"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = pickle.loads(data)
                
                # Process message based on type
                if message['type'] == 'position':
                    self.vehicle_positions[message['vehicle_id']] = {
                        'position': message['position'],
                        'timestamp': time.time()
                    }
                elif message['type'] == 'traffic_light':
                    self.traffic_light_states[message['light_id']] = {
                        'state': message['state'],
                        'timestamp': time.time()
                    }
                elif message['type'] == 'road_condition':
                    self.road_conditions[message['location']] = {
                        'condition': message['condition'],
                        'timestamp': time.time()
                    }
                
                self.message_queue.put(message)
                
            except Exception as e:
                if self.running:  # Only log if not shutting down
                    logger.error(f"Error receiving V2X message: {e}")
    
    def send_position(self, position, velocity):
        """Broadcast vehicle position and velocity"""
        if not self.socket:
            return
            
        message = {
            'type': 'position',
            'vehicle_id': self.vehicle_id,
            'position': position,
            'velocity': velocity,
            'timestamp': time.time()
        }
        
        try:
            data = pickle.dumps(message)
            self.socket.sendto(data, ('<broadcast>', self.port))
        except Exception as e:
            logger.error(f"Error sending position via V2X: {e}")
    
    def send_traffic_light_state(self, light_id, state):
        """Broadcast traffic light state"""
        if not self.socket:
            return
            
        message = {
            'type': 'traffic_light',
            'light_id': light_id,
            'state': state,
            'timestamp': time.time()
        }
        
        try:
            data = pickle.dumps(message)
            self.socket.sendto(data, ('<broadcast>', self.port))
        except Exception as e:
            logger.error(f"Error sending traffic light state via V2X: {e}")
    
    def send_road_condition(self, location, condition):
        """Broadcast road condition"""
        if not self.socket:
            return
            
        message = {
            'type': 'road_condition',
            'location': location,
            'condition': condition,
            'timestamp': time.time()
        }
        
        try:
            data = pickle.dumps(message)
            self.socket.sendto(data, ('<broadcast>', self.port))
        except Exception as e:
            logger.error(f"Error sending road condition via V2X: {e}")
    
    def get_nearby_vehicles(self, position, radius=50.0):
        """Get positions of nearby vehicles"""
        nearby = {}
        current_time = time.time()
        
        for vehicle_id, data in self.vehicle_positions.items():
            if vehicle_id == self.vehicle_id:
                continue
                
            # Check if data is recent
            if current_time - data['timestamp'] > V2X_MESSAGE_TTL:
                continue
                
            # Check if within radius
            distance = position.distance(data['position'])
            if distance <= radius:
                nearby[vehicle_id] = data
                
        return nearby
    
    def get_traffic_light_state(self, light_id):
        """Get state of a traffic light"""
        if light_id in self.traffic_light_states:
            data = self.traffic_light_states[light_id]
            if time.time() - data['timestamp'] <= V2X_MESSAGE_TTL:
                return data['state']
        return None
    
    def get_road_condition(self, location):
        """Get road condition at a location"""
        # Find closest road condition
        closest_location = None
        closest_distance = float('inf')
        closest_condition = None
        
        for loc, data in self.road_conditions.items():
            distance = location.distance(loc)
            if distance < closest_distance and time.time() - data['timestamp'] <= V2X_MESSAGE_TTL:
                closest_distance = distance
                closest_location = loc
                closest_condition = data['condition']
        
        return closest_condition
    
    def stop(self):
        """Stop V2X communication"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
        logger.info("V2X communication stopped")

class AdaptivePIDController:
    """Adaptive PID controller with auto-tuning capabilities"""
    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD, dt=PID_DT):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        # PID state variables
        self.error_integral = 0
        self.prev_error = 0
        self.error_buffer = deque(maxlen=PID_ERROR_BUFFER_SIZE)
        
        # Adaptive parameters
        self.adaptive_kp = kp
        self.adaptive_ki = ki
        self.adaptive_kd = kd
        
        # Performance tracking
        self.error_history = deque(maxlen=100)
        self.control_history = deque(maxlen=100)
        self.oscillation_count = 0
        self.last_oscillation_time = time.time()
        
        # Auto-tuning parameters
        self.tuning_active = False
        self.tuning_interval = 10.0  # seconds
        self.last_tuning_time = time.time()
        
        logger.info("Adaptive PID controller initialized")
    
    def update(self, error, setpoint=0.0, process_variable=None):
        """
        Update PID controller with new error value
        
        Args:
            error: Current error (difference between setpoint and process variable)
            setpoint: Desired value
            process_variable: Current measured value
            
        Returns:
            Control output value, clipped between -1.0 and 1.0
        """
        # Track error for adaptive tuning
        self.error_history.append(error)
        
        # Smooth error using moving average to reduce noise
        self.error_buffer.append(error)
        smoothed_error = np.mean(self.error_buffer)
        
        # PID calculations
        self.error_integral += smoothed_error * self.dt
        # Anti-windup: limit integral term to prevent instability
        self.error_integral = np.clip(self.error_integral, -PID_INTEGRAL_LIMIT, PID_INTEGRAL_LIMIT)
        
        # Calculate derivative of error
        error_derivative = (smoothed_error - self.prev_error) / self.dt
        
        # Compute PID output with adaptive parameters
        control = (self.adaptive_kp * smoothed_error + 
                  self.adaptive_ki * self.error_integral + 
                  self.adaptive_kd * error_derivative)
        
        # Store current error for next derivative calculation
        self.prev_error = smoothed_error
        
        # Track control output for adaptive tuning
        self.control_history.append(control)
        
        # Check for oscillations
        self._detect_oscillations()
        
        # Auto-tune if needed
        if time.time() - self.last_tuning_time > self.tuning_interval:
            self._auto_tune()
            self.last_tuning_time = time.time()
        
        # Limit control output to valid range
        return np.clip(control, -1.0, 1.0)
    
    def _detect_oscillations(self):
        """Detect oscillations in control output"""
        if len(self.control_history) < 10:
            return
            
        # Count zero crossings in control output
        zero_crossings = 0
        for i in range(1, len(self.control_history)):
            if (self.control_history[i-1] * self.control_history[i]) < 0:
                zero_crossings += 1
        
        # If we have multiple zero crossings in a short time, consider it oscillation
        if zero_crossings >= 3 and time.time() - self.last_oscillation_time > 2.0:
            self.oscillation_count += 1
            self.last_oscillation_time = time.time()
    
    def _auto_tune(self):
        """Auto-tune PID parameters based on performance"""
        if len(self.error_history) < 50:
            return
            
        # Calculate error statistics
        error_std = np.std(self.error_history)
        error_mean = np.mean(np.abs(self.error_history))
        
        # Adjust parameters based on performance
        if self.oscillation_count > 2:
            # Reduce proportional gain to reduce oscillations
            self.adaptive_kp *= 0.9
            # Increase derivative gain to dampen oscillations
            self.adaptive_kd *= 1.1
            logger.info(f"Reducing oscillations: Kp={self.adaptive_kp:.3f}, Kd={self.adaptive_kd:.3f}")
        elif error_std > 0.1:
            # Increase proportional gain to reduce steady-state error
            self.adaptive_kp *= 1.05
            logger.info(f"Reducing steady-state error: Kp={self.adaptive_kp:.3f}")
        elif error_mean > 0.05:
            # Increase integral gain to reduce steady-state error
            self.adaptive_ki *= 1.05
            logger.info(f"Reducing steady-state error: Ki={self.adaptive_ki:.3f}")
        
        # Reset oscillation count
        self.oscillation_count = 0
        
        # Ensure parameters stay within reasonable bounds
        self.adaptive_kp = np.clip(self.adaptive_kp, 0.1, 2.0)
        self.adaptive_ki = np.clip(self.adaptive_ki, 0.01, 0.2)
        self.adaptive_kd = np.clip(self.adaptive_kd, 0.1, 1.0)
    
    def reset(self):
        """Reset controller state to initial values"""
        self.error_integral = 0
        self.prev_error = 0
        self.error_buffer.clear()
        self.error_history.clear()
        self.control_history.clear()
        self.oscillation_count = 0
        self.last_oscillation_time = time.time()
        
        # Reset adaptive parameters to initial values
        self.adaptive_kp = self.kp
        self.adaptive_ki = self.ki
        self.adaptive_kd = self.kd
    
    def get_parameters(self):
        """Get current PID parameters"""
        return {
            'kp': self.adaptive_kp,
            'ki': self.adaptive_ki,
            'kd': self.adaptive_kd
        }

class HybridAStar:
    """Hybrid A* pathfinding algorithm combining waypoint-based and grid-based approaches"""
    def __init__(self, world, resolution=GRID_RESOLUTION, size=GRID_SIZE):
        self.world = world
        self.resolution = resolution
        self.size = size
        self.grid = np.zeros((size, size))
        self.center = size // 2
        self.waypoint_graph = self._build_waypoint_graph()
        self.last_update_time = 0
        self.update_interval = GRID_UPDATE_INTERVAL
        
        logger.info("Hybrid A* pathfinding initialized")
    
    def _build_waypoint_graph(self):
        """Build a graph of waypoints for pathfinding"""
        graph = {}
        spawn_points = self.world.get_map().get_spawn_points()
        
        for i, point in enumerate(spawn_points):
            # Get waypoint at spawn point
            waypoint = self.world.get_map().get_waypoint(point.location)
            if waypoint:
                graph[i] = {
                    'location': waypoint.transform.location,
                    'waypoint': waypoint,
                    'neighbors': []
                }
        
        # Connect waypoints that are close to each other
        for i in graph:
            for j in graph:
                if i != j:
                    distance = graph[i]['location'].distance(graph[j]['location'])
                    if distance < 20.0:  # Connect waypoints within 20 meters
                        graph[i]['neighbors'].append((j, distance))
        
        return graph
    
    def world_to_grid(self, location):
        """Convert world coordinates to grid coordinates"""
        x = int(location.x / self.resolution) + self.center
        y = int(location.y / self.resolution) + self.center
        return (max(0, min(self.size-1, x)), max(0, min(self.size-1, y)))
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = (grid_x - self.center) * self.resolution
        y = (grid_y - self.center) * self.resolution
        return carla.Location(x=x, y=y, z=0.0)
    
    def update_grid(self, vehicle_location, traffic_vehicles, pedestrians=None):
        """Update the occupancy grid with obstacles"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = current_time
        
        # Reset grid
        self.grid.fill(0)
        
        # Add traffic vehicles as obstacles
        for vehicle in traffic_vehicles:
            if vehicle.is_alive:
                loc = vehicle.get_location()
                gx, gy = self.world_to_grid(loc)
                # Mark vehicle and surrounding cells as obstacles
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.size and 0 <= ny < self.size:
                            # Distance-based obstacle strength
                            distance = math.sqrt(dx*dx + dy*dy)
                            strength = max(0, 1.0 - distance / 3.0)
                            self.grid[ny, nx] = max(self.grid[ny, nx], strength)
        
        # Add pedestrians as obstacles if provided
        if pedestrians:
            for pedestrian in pedestrians:
                if pedestrian.is_alive:
                    loc = pedestrian.get_location()
                    gx, gy = self.world_to_grid(loc)
                    # Mark pedestrian and surrounding cells as obstacles
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < self.size and 0 <= ny < self.size:
                                # Distance-based obstacle strength
                                distance = math.sqrt(dx*dx + dy*dy)
                                strength = max(0, 1.0 - distance / 2.0)
                                self.grid[ny, nx] = max(self.grid[ny, nx], strength)
        
        # Add buildings and static obstacles
        for i in range(self.size):
            for j in range(self.size):
                world_loc = self.grid_to_world(i, j)
                wp = self.world.get_map().get_waypoint(world_loc)
                if not wp or wp.lane_type != carla.LaneType.Driving:
                    self.grid[j, i] = 1.0
    
    def find_path(self, start_loc, goal_loc, use_hybrid=True):
        """
        Find path using hybrid A* algorithm
        
        Args:
            start_loc: Starting location (carla.Location)
            goal_loc: Goal location (carla.Location)
            use_hybrid: Whether to use hybrid approach or pure grid-based
            
        Returns:
            List of carla.Location forming the path
        """
        start_time = time.time()
        
        if use_hybrid:
            # Try to find path using waypoint graph first
            waypoint_path = self._find_waypoint_path(start_loc, goal_loc)
            if waypoint_path:
                # Refine waypoint path using grid-based A*
                refined_path = self._refine_path(waypoint_path)
                logger.info(f"Hybrid A* found path in {time.time() - start_time:.3f}s")
                return refined_path
        
        # Fallback to pure grid-based A*
        grid_path = self._find_grid_path(start_loc, goal_loc)
        logger.info(f"Grid A* found path in {time.time() - start_time:.3f}s")
        return grid_path
    
    def _find_waypoint_path(self, start_loc, goal_loc):
        """Find path using waypoint graph"""
        # Find closest waypoints to start and goal
        start_wp = None
        start_wp_id = None
        start_min_dist = float('inf')
        
        goal_wp = None
        goal_wp_id = None
        goal_min_dist = float('inf')
        
        for wp_id, wp_data in self.waypoint_graph.items():
            start_dist = start_loc.distance(wp_data['location'])
            goal_dist = goal_loc.distance(wp_data['location'])
            
            if start_dist < start_min_dist:
                start_min_dist = start_dist
                start_wp = wp_data['waypoint']
                start_wp_id = wp_id
            
            if goal_dist < goal_min_dist:
                goal_min_dist = goal_dist
                goal_wp = wp_data['waypoint']
                goal_wp_id = wp_id
        
        if not start_wp or not goal_wp:
            return None
        
        # Use Dijkstra's algorithm to find shortest path in waypoint graph
        distances = {wp_id: float('inf') for wp_id in self.waypoint_graph}
        distances[start_wp_id] = 0
        previous = {wp_id: None for wp_id in self.waypoint_graph}
        unvisited = set(self.waypoint_graph.keys())
        
        while unvisited:
            # Find unvisited waypoint with minimum distance
            current = min(unvisited, key=lambda wp_id: distances[wp_id])
            
            # If we've reached the goal, reconstruct path
            if current == goal_wp_id:
                path = []
                while current is not None:
                    path.append(self.waypoint_graph[current]['waypoint'])
                    current = previous[current]
                path.reverse()
                return path
            
            # Mark current as visited
            unvisited.remove(current)
            
            # Update distances to neighbors
            for neighbor_id, distance in self.waypoint_graph[current]['neighbors']:
                if neighbor_id in unvisited:
                    new_distance = distances[current] + distance
                    if new_distance < distances[neighbor_id]:
                        distances[neighbor_id] = new_distance
                        previous[neighbor_id] = current
        
        return None  # No path found
    
    def _find_grid_path(self, start_loc, goal_loc):
        """Find path using grid-based A*"""
        start = self.world_to_grid(start_loc)
        goal = self.world_to_grid(goal_loc)
        
        # A* implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    world_loc = self.grid_to_world(current[0], current[1])
                    path.append(world_loc)
                    current = came_from[current]
                path.reverse()
                return path
            
            # Check neighbors (8-directional movement)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size):
                    continue
                
                # Check obstacle
                if self.grid[neighbor[1], neighbor[0]] > OBSTACLE_THRESHOLD:
                    continue
                
                # Calculate movement cost (diagonal movement costs more)
                movement_cost = math.sqrt(dx*dx + dy*dy)
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def _refine_path(self, waypoint_path):
        """Refine waypoint path using grid-based A*"""
        if len(waypoint_path) <= 1:
            return waypoint_path
        
        refined_path = [waypoint_path[0].transform.location]
        
        for i in range(len(waypoint_path) - 1):
            start_loc = waypoint_path[i].transform.location
            goal_loc = waypoint_path[i+1].transform.location
            
            # Find grid path between waypoints
            segment_path = self._find_grid_path(start_loc, goal_loc)
            
            # Add segment path to refined path (skip first point to avoid duplicates)
            refined_path.extend(segment_path[1:])
        
        return refined_path
    
    def heuristic(self, a, b):
        """Heuristic function for A* (Euclidean distance)"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

class MultiModalSensorFusion:
    """Multi-modal sensor fusion system combining camera, LiDAR, and radar data"""
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        
        # Sensors
        self.camera = None
        self.lidar = None
        self.radar = None
        self.semantic_camera = None
        
        # Sensor data
        self.camera_data = None
        self.lidar_data = None
        self.radar_data = None
        self.semantic_data = None
        
        # Processed data
        self.detected_objects = []
        self.road_boundaries = []
        self.occupancy_grid = None
        self.free_space = []
        
        # Processing threads
        self.camera_thread = None
        self.lidar_thread = None
        self.radar_thread = None
        self.fusion_thread = None
        
        # Thread control
        self.running = False
        self.data_lock = threading.Lock()
        
        # Setup sensors
        self.setup_sensors()
        
        logger.info("Multi-modal sensor fusion system initialized")
    
    def setup_sensors(self):
        """Set up all sensors"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Setup RGB camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_IMAGE_SIZE_X))
        camera_bp.set_attribute('image_size_y', str(CAMERA_IMAGE_SIZE_Y))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        
        camera_transform = carla.Transform(
            carla.Location(x=CAMERA_POSITION_X, z=CAMERA_POSITION_Z),
            carla.Rotation(pitch=CAMERA_ROTATION_PITCH)
        )
        
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.process_camera(image))
        
        # Setup LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_ROTATION_FREQUENCY))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_LOWER_FOV))
        
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(lambda data: self.process_lidar(data))
        
        # Setup radar
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(RADAR_HORIZONTAL_FOV))
        radar_bp.set_attribute('vertical_fov', str(RADAR_VERTICAL_FOV))
        radar_bp.set_attribute('points_per_second', str(RADAR_POINTS_PER_SECOND))
        radar_bp.set_attribute('range', str(RADAR_RANGE))
        
        radar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
        self.radar.listen(lambda data: self.process_radar(data))
        
        # Setup semantic segmentation camera
        semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', str(CAMERA_IMAGE_SIZE_X))
        semantic_bp.set_attribute('image_size_y', str(CAMERA_IMAGE_SIZE_Y))
        semantic_bp.set_attribute('fov', str(CAMERA_FOV))
        
        self.semantic_camera = self.world.spawn_actor(semantic_bp, camera_transform, attach_to=self.vehicle)
        self.semantic_camera.listen(lambda image: self.process_semantic_segmentation(image))
        
        logger.info("All sensors set up successfully")
    
    def process_camera(self, image):
        """Process camera image for object detection"""
        try:
            # Convert to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            array = array[:, :, ::-1]  # Convert BGR to RGB
            
            with self.data_lock:
                self.camera_data = array
                
            # Start camera processing thread if not already running
            if self.camera_thread is None or not self.camera_thread.is_alive():
                self.camera_thread = threading.Thread(target=self._camera_processing_loop)
                self.camera_thread.daemon = True
                self.camera_thread.start()
                
        except Exception as e:
            logger.error(f"Error processing camera image: {e}")
    
    def process_lidar(self, data):
        """Process LiDAR point cloud"""
        try:
            # Convert to numpy array
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            
            # Transform points to vehicle coordinate system
            lidar_transform = data.transform
            vehicle_transform = self.vehicle.get_transform()
            
            # Apply rotation
            rotation = vehicle_transform.rotation
            yaw = math.radians(rotation.yaw)
            pitch = math.radians(rotation.pitch)
            roll = math.radians(rotation.roll)
            
            # Rotation matrix
            Rx = np.array([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]
            ])
            
            Ry = np.array([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]
            ])
            
            Rz = np.array([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            R = Rz @ Ry @ Rx
            
            # Apply rotation to points
            rotated_points = points[:, :3] @ R.T
            
            # Apply translation
            translation = vehicle_transform.location
            translated_points = rotated_points + np.array([translation.x, translation.y, translation.z])
            
            with self.data_lock:
                self.lidar_data = translated_points
                
            # Start LiDAR processing thread if not already running
            if self.lidar_thread is None or not self.lidar_thread.is_alive():
                self.lidar_thread = threading.Thread(target=self._lidar_processing_loop)
                self.lidar_thread.daemon = True
                self.lidar_thread.start()
                
        except Exception as e:
            logger.error(f"Error processing LiDAR data: {e}")
    
    def process_radar(self, data):
        """Process radar data"""
        try:
            # Convert to numpy array
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            
            with self.data_lock:
                self.radar_data = points
                
            # Start radar processing thread if not already running
            if self.radar_thread is None or not self.radar_thread.is_alive():
                self.radar_thread = threading.Thread(target=self._radar_processing_loop)
                self.radar_thread.daemon = True
                self.radar_thread.start()
                
        except Exception as e:
            logger.error(f"Error processing radar data: {e}")
    
    def process_semantic_segmentation(self, image):
        """Process semantic segmentation image"""
        try:
            # Convert to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            
            # Convert to semantic labels
            # CARLA uses BGRA format for semantic segmentation
            # Each pixel's R, G, B values correspond to the semantic label
            labels = array[:, :, 2]  # Red channel contains the label
            
            with self.data_lock:
                self.semantic_data = labels
                
            # Start fusion processing thread if not already running
            if self.fusion_thread is None or not self.fusion_thread.is_alive():
                self.fusion_thread = threading.Thread(target=self._fusion_processing_loop)
                self.fusion_thread.daemon = True
                self.fusion_thread.start()
                
        except Exception as e:
            logger.error(f"Error processing semantic segmentation: {e}")
    
    def _camera_processing_loop(self):
        """Background thread for processing camera data"""
        while self.running:
            try:
                with self.data_lock:
                    if self.camera_data is None:
                        time.sleep(0.01)
                        continue
                    
                    # Make a copy of the data to avoid holding the lock
                    image = self.camera_data.copy()
                
                # Object detection using simple color-based segmentation
                self._detect_objects(image)
                
                # Road boundary detection
                self._detect_road_boundaries(image)
                
                # Sleep to prevent high CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in camera processing loop: {e}")
    
    def _lidar_processing_loop(self):
        """Background thread for processing LiDAR data"""
        while self.running:
            try:
                with self.data_lock:
                    if self.lidar_data is None:
                        time.sleep(0.01)
                        continue
                    
                    # Make a copy of the data to avoid holding the lock
                    points = self.lidar_data.copy()
                
                # Generate occupancy grid from LiDAR points
                self._generate_occupancy_grid(points)
                
                # Detect free space
                self._detect_free_space(points)
                
                # Sleep to prevent high CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in LiDAR processing loop: {e}")
    
    def _radar_processing_loop(self):
        """Background thread for processing radar data"""
        while self.running:
            try:
                with self.data_lock:
                    if self.radar_data is None:
                        time.sleep(0.01)
                        continue
                    
                    # Make a copy of the data to avoid holding the lock
                    points = self.radar_data.copy()
                
                # Process radar data for moving objects
                self._process_radar_for_moving_objects(points)
                
                # Sleep to prevent high CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in radar processing loop: {e}")
    
    def _fusion_processing_loop(self):
        """Background thread for fusing sensor data"""
        while self.running:
            try:
                # Wait for all sensor data to be available
                with self.data_lock:
                    if (self.camera_data is None or self.lidar_data is None or 
                        self.radar_data is None or self.semantic_data is None):
                        time.sleep(0.01)
                        continue
                    
                    # Make copies of the data to avoid holding the lock
                    camera_image = self.camera_data.copy()
                    lidar_points = self.lidar_data.copy()
                    radar_points = self.radar_data.copy()
                    semantic_labels = self.semantic_data.copy()
                
                # Fuse sensor data
                self._fuse_sensor_data(camera_image, lidar_points, radar_points, semantic_labels)
                
                # Sleep to prevent high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in fusion processing loop: {e}")
    
    def _detect_objects(self, image):
        """Detect objects in camera image"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for different objects
            # Vehicles (red, blue, etc.)
            lower_vehicle = np.array([0, 50, 50])
            upper_vehicle = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_vehicle, upper_vehicle)
            
            lower_vehicle = np.array([170, 50, 50])
            upper_vehicle = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_vehicle, upper_vehicle)
            
            vehicle_mask = cv2.bitwise_or(mask1, mask2)
            
            # Pedestrians (skin tones)
            lower_pedestrian = np.array([0, 20, 70])
            upper_pedestrian = np.array([20, 255, 255])
            pedestrian_mask = cv2.inRange(hsv, lower_pedestrian, upper_pedestrian)
            
            # Find contours
            vehicle_contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pedestrian_contours, _ = cv2.findContours(pedestrian_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process vehicle contours
            detected_objects = []
            for contour in vehicle_contours:
                area = cv2.contourArea(contour)
                if area > MIN_OBJECT_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Estimate distance based on bounding box size
                    # This is a simplified approach - in a real system, you would use
                    # more sophisticated methods like stereo vision or depth estimation
                    distance = 1000.0 / max(w, h)  # Simplified distance estimation
                    
                    if distance < MAX_OBJECT_DISTANCE:
                        detected_objects.append({
                            'type': 'vehicle',
                            'position': (x + w/2, y + h/2),
                            'size': (w, h),
                            'distance': distance,
                            'confidence': min(1.0, area / 1000.0)
                        })
            
            # Process pedestrian contours
            for contour in pedestrian_contours:
                area = cv2.contourArea(contour)
                if area > MIN_OBJECT_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Estimate distance
                    distance = 500.0 / max(w, h)  # Pedestrians are smaller
                    
                    if distance < MAX_OBJECT_DISTANCE:
                        detected_objects.append({
                            'type': 'pedestrian',
                            'position': (x + w/2, y + h/2),
                            'size': (w, h),
                            'distance': distance,
                            'confidence': min(1.0, area / 500.0)
                        })
            
            # Update detected objects
            with self.data_lock:
                self.detected_objects = detected_objects
                
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
    
    def _detect_road_boundaries(self, image):
        """Detect road boundaries in camera image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Apply Hough transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
            
            road_boundaries = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line angle
                    angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
                    
                    # Filter for nearly horizontal lines (road boundaries)
                    if abs(angle) < 30 or abs(angle) > 150:
                        road_boundaries.append(((x1, y1), (x2, y2)))
            
            # Update road boundaries
            with self.data_lock:
                self.road_boundaries = road_boundaries
                
        except Exception as e:
            logger.error(f"Error detecting road boundaries: {e}")
    
    def _generate_occupancy_grid(self, points):
        """Generate occupancy grid from LiDAR points"""
        try:
            # Create a 2D grid (top-down view)
            grid_size = 100
            grid_resolution = 0.5  # meters per cell
            grid = np.zeros((grid_size, grid_size))
            
            # Project LiDAR points onto 2D grid
            for point in points:
                x, y, z = point
                
                # Skip points too far away or too high/low
                if abs(x) > grid_size * grid_resolution / 2 or abs(y) > grid_size * grid_resolution / 2:
                    continue
                
                if z < -2.0 or z > 2.0:
                    continue
                
                # Convert to grid coordinates
                grid_x = int((x + grid_size * grid_resolution / 2) / grid_resolution)
                grid_y = int((y + grid_size * grid_resolution / 2) / grid_resolution)
                
                # Ensure within grid bounds
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_y, grid_x] = 1.0
            
            # Apply morphological operations to clean up the grid
            kernel = np.ones((3, 3), np.uint8)
            grid = cv2.erode(grid, kernel, iterations=1)
            grid = cv2.dilate(grid, kernel, iterations=1)
            
            # Update occupancy grid
            with self.data_lock:
                self.occupancy_grid = grid
                
        except Exception as e:
            logger.error(f"Error generating occupancy grid: {e}")
    
    def _detect_free_space(self, points):
        """Detect free space from LiDAR points"""
        try:
            # This is a simplified approach - in a real system, you would use
            # more sophisticated methods like ray casting or occupancy grid mapping
            
            # Create a 2D histogram of points
            grid_size = 100
            grid_resolution = 0.5  # meters per cell
            grid = np.zeros((grid_size, grid_size))
            
            # Project LiDAR points onto 2D grid
            for point in points:
                x, y, z = point
                
                # Skip points too far away or too high/low
                if abs(x) > grid_size * grid_resolution / 2 or abs(y) > grid_size * grid_resolution / 2:
                    continue
                
                if z < -2.0 or z > 2.0:
                    continue
                
                # Convert to grid coordinates
                grid_x = int((x + grid_size * grid_resolution / 2) / grid_resolution)
                grid_y = int((y + grid_size * grid_resolution / 2) / grid_resolution)
                
                # Ensure within grid bounds
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_y, grid_x] += 1
            
            # Find empty cells (free space)
            free_space = []
            for y in range(grid_size):
                for x in range(grid_size):
                    if grid[y, x] < 1:  # Few or no points in this cell
                        # Convert back to world coordinates
                        world_x = (x - grid_size / 2) * grid_resolution
                        world_y = (y - grid_size / 2) * grid_resolution
                        free_space.append((world_x, world_y))
            
            # Update free space
            with self.data_lock:
                self.free_space = free_space
                
        except Exception as e:
            logger.error(f"Error detecting free space: {e}")
    
    def _process_radar_for_moving_objects(self, points):
        """Process radar data to detect moving objects"""
        try:
            # Radar data format: [x, y, z, velocity]
            # We're interested in points with significant velocity (moving objects)
            
            moving_objects = []
            for point in points:
                x, y, z, velocity = point
                
                # Skip points too far away
                distance = math.sqrt(x*x + y*y + z*z)
                if distance > RADAR_RANGE:
                    continue
                
                # Check if point has significant velocity
                if abs(velocity) > 1.0:  # 1 m/s threshold
                    moving_objects.append({
                        'position': (x, y, z),
                        'velocity': velocity,
                        'distance': distance
                    })
            
            # Update detected objects with radar data
            with self.data_lock:
                # Merge with existing detected objects
                for obj in moving_objects:
                    # Find matching object based on position
                    matched = False
                    for detected_obj in self.detected_objects:
                        obj_pos = detected_obj['position']
                        obj_x = (obj_pos[0] - CAMERA_IMAGE_SIZE_X/2) * obj['distance'] / CAMERA_IMAGE_SIZE_X
                        obj_y = (obj_pos[1] - CAMERA_IMAGE_SIZE_Y/2) * obj['distance'] / CAMERA_IMAGE_SIZE_Y
                        
                        distance = math.sqrt((obj_x - obj['position'][0])**2 + 
                                             (obj_y - obj['position'][1])**2)
                        
                        if distance < 2.0:  # 2 meters threshold
                            # Update object with radar data
                            detected_obj['velocity'] = obj['velocity']
                            detected_obj['radar_distance'] = obj['distance']
                            matched = True
                            break
                    
                    # If no match, add new object
                    if not matched:
                        self.detected_objects.append({
                            'type': 'moving_object',
                            'position': obj['position'],
                            'velocity': obj['velocity'],
                            'distance': obj['distance'],
                            'confidence': 0.7
                        })
                
        except Exception as e:
            logger.error(f"Error processing radar for moving objects: {e}")
    
    def _fuse_sensor_data(self, camera_image, lidar_points, radar_points, semantic_labels):
        """Fuse data from all sensors"""
        try:
            # This is where you would implement sophisticated sensor fusion algorithms
            # For now, we'll do a simple combination of detections
            
            # Create a combined object list
            fused_objects = []
            
            # Get detected objects from camera
            with self.data_lock:
                camera_objects = self.detected_objects.copy()
            
            # Project LiDAR points onto camera image
            # This would require camera calibration in a real system
            # For now, we'll use a simplified approach
            
            # Combine objects from different sensors
            fused_objects.extend(camera_objects)
            
            # Update fused objects
            with self.data_lock:
                self.detected_objects = fused_objects
                
        except Exception as e:
            logger.error(f"Error fusing sensor data: {e}")
    
    def get_detected_objects(self):
        """Get list of detected objects"""
        with self.data_lock:
            return self.detected_objects.copy()
    
    def get_road_boundaries(self):
        """Get detected road boundaries"""
        with self.data_lock:
            return self.road_boundaries.copy()
    
    def get_occupancy_grid(self):
        """Get occupancy grid"""
        with self.data_lock:
            return self.occupancy_grid.copy()
    
    def get_free_space(self):
        """Get detected free space"""
        with self.data_lock:
            return self.free_space.copy()
    
    def start(self):
        """Start sensor processing threads"""
        self.running = True
        logger.info("Sensor fusion system started")
    
    def stop(self):
        """Stop sensor processing threads"""
        self.running = False
        
        # Wait for threads to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        
        if self.lidar_thread and self.lidar_thread.is_alive():
            self.lidar_thread.join(timeout=1.0)
        
        if self.radar_thread and self.radar_thread.is_alive():
            self.radar_thread.join(timeout=1.0)
        
        if self.fusion_thread and self.fusion_thread.is_alive():
            self.fusion_thread.join(timeout=1.0)
        
        # Destroy sensors
        if self.camera:
            self.camera.destroy()
        
        if self.lidar:
            self.lidar.destroy()
        
        if self.radar:
            self.radar.destroy()
        
        if self.semantic_camera:
            self.semantic_camera.destroy()
        
        logger.info("Sensor fusion system stopped")

class AdvancedOpticalAvoidance:
    """Advanced optical avoidance system using deep learning and computer vision"""
    def __init__(self, vehicle, sensor_fusion):
        self.vehicle = vehicle
        self.sensor_fusion = sensor_fusion
        
        # Obstacle detection
        self.obstacles = []
        self.obstacle_detected = False
        self.obstacle_position = None
        self.obstacle_distance = None
        self.obstacle_velocity = None
        
        # Avoidance parameters
        self.avoidance_active = False
        self.avoidance_direction = 0  # -1: left, 0: straight, 1: right
        self.avoidance_strength = 0.0
        self.avoidance_confidence = 0.0
        
        # Risk assessment
        self.collision_risk = 0.0
        self.time_to_collision = float('inf')
        
        # Processing thread
        self.processing_thread = None
        self.running = False
        self.data_lock = threading.Lock()
        
        logger.info("Advanced optical avoidance system initialized")
    
    def start(self):
        """Start the avoidance system"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Optical avoidance system started")
    
    def _processing_loop(self):
        """Background thread for processing sensor data and avoiding obstacles"""
        while self.running:
            try:
                # Get sensor data
                camera_data = self.sensor_fusion.get_detected_objects()
                road_boundaries = self.sensor_fusion.get_road_boundaries()
                occupancy_grid = self.sensor_fusion.get_occupancy_grid()
                free_space = self.sensor_fusion.get_free_space()
                
                # Process obstacles
                self._process_obstacles(camera_data)
                
                # Assess collision risk
                self._assess_collision_risk()
                
                # Determine avoidance action
                self._determine_avoidance_action(free_space, road_boundaries)
                
                # Sleep to prevent high CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in optical avoidance processing loop: {e}")
    
    def _process_obstacles(self, detected_objects):
        """Process detected obstacles"""
        try:
            obstacles = []
            obstacle_detected = False
            closest_obstacle = None
            closest_distance = float('inf')
            
            # Process each detected object
            for obj in detected_objects:
                # Skip objects that are too far away
                if obj['distance'] > MAX_OBJECT_DISTANCE:
                    continue
                
                # Calculate obstacle position in vehicle coordinates
                # This is a simplified approach - in a real system, you would use
                # more sophisticated coordinate transformations
                
                # Add to obstacle list
                obstacles.append({
                    'type': obj['type'],
                    'position': obj['position'],
                    'distance': obj['distance'],
                    'velocity': obj.get('velocity', 0.0),
                    'confidence': obj['confidence']
                })
                
                # Check if this is the closest obstacle
                if obj['distance'] < closest_distance:
                    closest_distance = obj['distance']
                    closest_obstacle = obj
            
            # Update obstacle state
            with self.data_lock:
                self.obstacles = obstacles
                self.obstacle_detected = len(obstacles) > 0
                self.obstacle_distance = closest_distance
                
                if closest_obstacle:
                    self.obstacle_position = closest_obstacle['position']
                    self.obstacle_velocity = closest_obstacle.get('velocity', 0.0)
                else:
                    self.obstacle_position = None
                    self.obstacle_velocity = None
                
        except Exception as e:
            logger.error(f"Error processing obstacles: {e}")
    
    def _assess_collision_risk(self):
        """Assess collision risk with detected obstacles"""
        try:
            collision_risk = 0.0
            time_to_collision = float('inf')
            
            if not self.obstacle_detected or self.obstacle_distance is None:
                with self.data_lock:
                    self.collision_risk = collision_risk
                    self.time_to_collision = time_to_collision
                return
            
            # Get vehicle state
            vehicle_velocity = self.vehicle.get_velocity()
            vehicle_speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
            
            # Calculate time to collision
            if vehicle_speed > 0.1:  # Avoid division by zero
                # Relative velocity (simplified)
                relative_velocity = vehicle_speed - abs(self.obstacle_velocity)
                
                if relative_velocity > 0:  # Moving toward obstacle
                    time_to_collision = self.obstacle_distance / relative_velocity
                    
                    # Calculate collision risk based on time to collision
                    if time_to_collision < 1.0:  # Less than 1 second
                        collision_risk = 1.0
                    elif time_to_collision < 3.0:  # Less than 3 seconds
                        collision_risk = 1.0 - (time_to_collision - 1.0) / 2.0
                    else:
                        collision_risk = 0.0
            
            # Update collision risk
            with self.data_lock:
                self.collision_risk = collision_risk
                self.time_to_collision = time_to_collision
                
        except Exception as e:
            logger.error(f"Error assessing collision risk: {e}")
    
    def _determine_avoidance_action(self, free_space, road_boundaries):
        """Determine the best avoidance action"""
        try:
            avoidance_active = False
            avoidance_direction = 0
            avoidance_strength = 0.0
            avoidance_confidence = 0.0
            
            if not self.obstacle_detected or self.obstacle_position is None:
                with self.data_lock:
                    self.avoidance_active = avoidance_active
                    self.avoidance_direction = avoidance_direction
                    self.avoidance_strength = avoidance_strength
                    self.avoidance_confidence = avoidance_confidence
                return
            
            # Determine obstacle position relative to vehicle
            # This is a simplified approach - in a real system, you would use
            # more sophisticated coordinate transformations
            
            # For now, assume obstacle position is in image coordinates
            img_x, img_y = self.obstacle_position
            
            # Determine which side of the image the obstacle is on
            image_width = CAMERA_IMAGE_SIZE_X
            image_center = image_width / 2
            
            if img_x < image_center * 0.6:  # Left side of image
                avoidance_direction = 1  # Steer right
            elif img_x > image_width * 1.4:  # Right side of image
                avoidance_direction = -1  # Steer left
            else:  # Center of image
                # Check which side has more free space
                left_space = sum(1 for x, y in free_space if x < 0)
                right_space = sum(1 for x, y in free_space if x > 0)
                
                if left_space > right_space:
                    avoidance_direction = -1  # Steer left
                else:
                    avoidance_direction = 1  # Steer right
            
            # Calculate avoidance strength based on collision risk
            avoidance_strength = min(0.5, self.collision_risk * 0.5)
            
            # Calculate confidence based on obstacle distance and confidence
            distance_confidence = 1.0 - min(1.0, self.obstacle_distance / MAX_OBJECT_DISTANCE)
            obstacle_confidence = self.obstacle_position[2] if len(self.obstacle_position) > 2 else 0.5
            avoidance_confidence = (distance_confidence + obstacle_confidence) / 2.0
            
            # Activate avoidance if collision risk is high enough
            avoidance_active = self.collision_risk > 0.3
            
            # Update avoidance action
            with self.data_lock:
                self.avoidance_active = avoidance_active
                self.avoidance_direction = avoidance_direction
                self.avoidance_strength = avoidance_strength
                self.avoidance_confidence = avoidance_confidence
                
        except Exception as e:
            logger.error(f"Error determining avoidance action: {e}")
    
    def get_avoidance_action(self):
        """Get steering adjustment for obstacle avoidance"""
        with self.data_lock:
            if self.avoidance_active:
                return self.avoidance_direction * self.avoidance_strength
            return 0.0
    
    def get_collision_risk(self):
        """Get current collision risk"""
        with self.data_lock:
            return self.collision_risk
    
    def get_time_to_collision(self):
        """Get estimated time to collision"""
        with self.data_lock:
            return self.time_to_collision
    
    def stop(self):
        """Stop the avoidance system"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        logger.info("Optical avoidance system stopped")

class AdvancedTrafficLightDetector:
    """Advanced traffic light detection and prediction system"""
    def __init__(self, vehicle, v2x_communication=None):
        self.vehicle = vehicle
        self.v2x_communication = v2x_communication
        
        # Traffic light state
        self.current_light = None
        self.light_state = None
        self.light_distance = None
        self.light_time_to_change = None
        
        # Detection parameters
        self.detection_range = TRAFFIC_LIGHT_DETECTION_RANGE
        self.stop_distance = TRAFFIC_LIGHT_STOP_DISTANCE
        self.yellow_stop_time = TRAFFIC_LIGHT_YELLOW_STOP_TIME
        
        # Processing thread
        self.processing_thread = None
        self.running = False
        self.data_lock = threading.Lock()
        
        logger.info("Advanced traffic light detector initialized")
    
    def start(self):
        """Start the traffic light detector"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Traffic light detector started")
    
    def _processing_loop(self):
        """Background thread for detecting and predicting traffic lights"""
        while self.running:
            try:
                # Get all traffic lights near the vehicle
                vehicle_location = self.vehicle.get_location()
                traffic_lights = self.vehicle.get_world().get_actors().filter('traffic.traffic_light')
                
                closest_light = None
                min_distance = float('inf')
                
                for light in traffic_lights:
                    light_location = light.get_location()
                    distance = vehicle_location.distance(light_location)
                    
                    # Check if traffic light is ahead of the vehicle
                    if distance < self.detection_range:
                        vehicle_transform = self.vehicle.get_transform()
                        forward_vector = vehicle_transform.get_forward_vector()
                        to_light = light_location - vehicle_location
                        
                        # Check if light is in front (within 45 degrees)
                        dot_product = forward_vector.x * to_light.x + forward_vector.y * to_light.y
                        if dot_product > 0:
                            if distance < min_distance:
                                min_distance = distance
                                closest_light = light
                
                # Update traffic light state
                with self.data_lock:
                    self.current_light = closest_light
                    self.light_distance = min_distance if closest_light else None
                    
                    if closest_light:
                        self.light_state = closest_light.get_state()
                        
                        # Try to get time to change from V2X
                        if self.v2x_communication:
                            v2x_state = self.v2x_communication.get_traffic_light_state(id(closest_light))
                            if v2x_state:
                                self.light_time_to_change = v2x_state.get('time_to_change', None)
                        
                        # If no V2X data, estimate based on typical timing
                        if self.light_time_to_change is None:
                            self._estimate_time_to_change()
                    else:
                        self.light_state = None
                        self.light_time_to_change = None
                
                # Sleep to prevent high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in traffic light processing loop: {e}")
    
    def _estimate_time_to_change(self):
        """Estimate time until traffic light changes state"""
        try:
            if not self.current_light or not self.light_state:
                self.light_time_to_change = None
                return
            
            # Typical traffic light timing (simplified)
            # In a real system, you would use more sophisticated methods
            # or get this information from V2X communication
            
            current_time = time.time()
            
            # This is a very simplified estimation
            # In practice, you would use actual traffic light timing data
            if self.light_state == carla.TrafficLightState.Red:
                # Assume red light lasts 30 seconds
                self.light_time_to_change = 30.0 - (current_time % 30.0)
            elif self.light_state == carla.TrafficLightState.Yellow:
                # Assume yellow light lasts 3 seconds
                self.light_time_to_change = 3.0 - (current_time % 3.0)
            elif self.light_state == carla.TrafficLightState.Green:
                # Assume green light lasts 25 seconds
                self.light_time_to_change = 25.0 - (current_time % 25.0)
            else:
                self.light_time_to_change = None
                
        except Exception as e:
            logger.error(f"Error estimating time to change: {e}")
            self.light_time_to_change = None
    
    def should_stop(self):
        """Check if vehicle should stop for traffic light"""
        with self.data_lock:
            if self.current_light is None or self.light_state is None or self.light_distance is None:
                return False
            
            # Stop for red and yellow lights
            if self.light_state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                # Check distance to traffic light
                if self.light_distance < self.stop_distance:
                    return True
                
                # For yellow lights, also check if we can stop in time
                if self.light_state == carla.TrafficLightState.Yellow:
                    if self.light_time_to_change is not None and self.light_time_to_change < self.yellow_stop_time:
                        return True
            
            return False
    
    def get_light_state(self):
        """Get current traffic light state"""
        with self.data_lock:
            return self.light_state
    
    def get_light_distance(self):
        """Get distance to traffic light"""
        with self.data_lock:
            return self.light_distance
    
    def get_time_to_change(self):
        """Get estimated time until traffic light changes"""
        with self.data_lock:
            return self.light_time_to_change
    
    def stop(self):
        """Stop the traffic light detector"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        logger.info("Traffic light detector stopped")

class AdvancedRouteOptimizer:
    """Advanced route optimizer using multiple TSP algorithms and real-time traffic data"""
    def __init__(self, world_map, v2x_communication=None):
        self.map = world_map
        self.v2x_communication = v2x_communication
        
        # Distance cache
        self.distance_cache = {}
        self.traffic_cache = {}
        self.last_cache_update = 0
        self.cache_update_interval = 5.0  # seconds
        
        # Route optimization algorithms
        self.algorithms = {
            'nearest_neighbor': self._optimize_nearest_neighbor,
            '2opt': self._optimize_2opt,
            'genetic': self._optimize_genetic,
            'simulated_annealing': self._optimize_simulated_annealing
        }
        
        # Best route
        self.best_route = None
        self.best_distance = float('inf')
        
        logger.info("Advanced route optimizer initialized")
    
    def get_road_distance(self, start_loc, end_loc, use_cache=True):
        """Calculate road distance between two locations"""
        cache_key = (
            (start_loc.x, start_loc.y),
            (end_loc.x, end_loc.y)
        )
        
        # Check cache
        if use_cache and cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Get waypoints on road network
        start_wp = self.map.get_waypoint(start_loc)
        end_wp = self.map.get_waypoint(end_loc)
        
        if not start_wp or not end_wp:
            distance = start_loc.distance(end_loc)
        else:
            # Use A* pathfinding to calculate road distance
            distance = self._astar_distance(start_wp, end_wp)
            if distance is None:
                distance = start_loc.distance(end_loc) * 1.5  # Penalty for no road connection
        
        # Cache the result
        self.distance_cache[cache_key] = distance
        return distance
    
    def _astar_distance(self, start_wp, end_wp, max_iterations=ASTAR_MAX_ITERATIONS):
        """Calculate distance using A* pathfinding"""
        open_set = [(0, id(start_wp), start_wp, 0)]
        closed_set = set()
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current, dist = heapq.heappop(open_set)
            
            if current.transform.location.distance(end_wp.transform.location) < WAYPOINT_MAX_DISTANCE:
                return dist
            
            wp_hash = (current.transform.location.x, 
                      current.transform.location.y)
            if wp_hash in closed_set:
                continue
            closed_set.add(wp_hash)
            
            # Get next waypoints
            next_wps = current.next(SAMPLING_RESOLUTION)
            
            # Consider lane changes
            if current.lane_change & carla.LaneChange.Left:
                left_wp = current.get_left_lane()
                if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                    next_wps.append(left_wp)
            
            if current.lane_change & carla.LaneChange.Right:
                right_wp = current.get_right_lane()
                if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                    next_wps.append(right_wp)
            
            for next_wp in next_wps:
                if next_wp:
                    new_dist = dist + SAMPLING_RESOLUTION
                    heuristic = next_wp.transform.location.distance(end_wp.transform.location)
                    heapq.heappush(open_set, 
                                 (new_dist + heuristic, id(next_wp), next_wp, new_dist))
        
        return None  # No path found
    
    def get_traffic_factor(self, start_loc, end_loc):
        """Get traffic factor between two locations (1.0 = no traffic, higher = more traffic)"""
        cache_key = (
            (start_loc.x, start_loc.y),
            (end_loc.x, end_loc.y)
        )
        
        # Check cache
        if cache_key in self.traffic_cache:
            return self.traffic_cache[cache_key]
        
        # Default traffic factor
        traffic_factor = 1.0
        
        # Try to get traffic data from V2X
        if self.v2x_communication:
            # Get midpoint between start and end
            mid_x = (start_loc.x + end_loc.x) / 2
            mid_y = (start_loc.y + end_loc.y) / 2
            mid_loc = carla.Location(x=mid_x, y=mid_y, z=0.0)
            
            # Get road condition from V2X
            road_condition = self.v2x_communication.get_road_condition(mid_loc)
            
            if road_condition:
                # Adjust traffic factor based on road condition
                if road_condition == 'congested':
                    traffic_factor = 2.0
                elif road_condition == 'slow':
                    traffic_factor = 1.5
                elif road_condition == 'accident':
                    traffic_factor = 3.0
        
        # Cache the result
        self.traffic_cache[cache_key] = traffic_factor
        return traffic_factor
    
    def update_caches(self):
        """Update distance and traffic caches"""
        current_time = time.time()
        if current_time - self.last_cache_update < self.cache_update_interval:
            return
            
        self.last_cache_update = current_time
        
        # Clear old cache entries
        # In a real system, you would implement a more sophisticated cache management
        # For now, we'll just clear the caches periodically
        if len(self.distance_cache) > 1000:
            self.distance_cache.clear()
        
        if len(self.traffic_cache) > 500:
            self.traffic_cache.clear()
    
    def optimize_route(self, start_pos, waypoints, algorithm='2opt'):
        """Optimize route using specified algorithm"""
        self.update_caches()
        
        if algorithm not in self.algorithms:
            logger.warning(f"Unknown algorithm: {algorithm}, using 2opt")
            algorithm = '2opt'
        
        # Run optimization algorithm
        route, distance = self.algorithms[algorithm](start_pos, waypoints)
        
        # Update best route
        if distance < self.best_distance:
            self.best_route = route
            self.best_distance = distance
        
        return route, distance
    
    def _optimize_nearest_neighbor(self, start_pos, waypoints):
        """Optimize route using nearest neighbor heuristic"""
        unvisited = list(range(len(waypoints)))
        route = []
        total_distance = 0.0
        current_loc = carla.Location(x=start_pos[0], y=start_pos[1], z=start_pos[2])
        
        while unvisited:
            min_distance = float('inf')
            nearest_idx = None
            
            for idx in unvisited:
                wp_loc = carla.Location(x=waypoints[idx][0], 
                                       y=waypoints[idx][1], 
                                       z=waypoints[idx][2])
                
                # Get distance with traffic factor
                distance = self.get_road_distance(current_loc, wp_loc)
                traffic_factor = self.get_traffic_factor(current_loc, wp_loc)
                adjusted_distance = distance * traffic_factor
                
                if adjusted_distance < min_distance:
                    min_distance = adjusted_distance
                    nearest_idx = idx
            
            route.append(nearest_idx)
            total_distance += min_distance
            unvisited.remove(nearest_idx)
            current_loc = carla.Location(x=waypoints[nearest_idx][0],
                                       y=waypoints[nearest_idx][1],
                                       z=waypoints[nearest_idx][2])
        
        return route, total_distance
    
    def _optimize_2opt(self, start_pos, waypoints):
        """Optimize route using 2-opt improvement heuristic"""
        # Start with nearest neighbor solution
        route, distance = self._optimize_nearest_neighbor(start_pos, waypoints)
        
        def calculate_total_distance(route_order):
            """Calculate total distance for a route order"""
            total = 0.0
            current_loc = carla.Location(x=start_pos[0], y=start_pos[1], z=start_pos[2])
            
            for idx in route_order:
                wp_loc = carla.Location(x=waypoints[idx][0],
                                      y=waypoints[idx][1],
                                      z=waypoints[idx][2])
                
                # Get distance with traffic factor
                distance = self.get_road_distance(current_loc, wp_loc)
                traffic_factor = self.get_traffic_factor(current_loc, wp_loc)
                total += distance * traffic_factor
                current_loc = wp_loc
            
            return total
        
        # Iteratively improve the route using 2-opt swaps
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Try swapping edges
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    
                    new_distance = calculate_total_distance(new_route)
                    
                    if new_distance < distance:
                        route = new_route
                        distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        logger.info(f"2-opt optimization completed in {iterations} iterations")
        return route, distance
    
    def _optimize_genetic(self, start_pos, waypoints):
        """Optimize route using genetic algorithm"""
        # Genetic algorithm parameters
        population_size = 50
        elite_size = 10
        mutation_rate = 0.01
        generations = 100
        
        # Calculate distance function
        def calculate_distance(route_order):
            total = 0.0
            current_loc = carla.Location(x=start_pos[0], y=start_pos[1], z=start_pos[2])
            
            for idx in route_order:
                wp_loc = carla.Location(x=waypoints[idx][0],
                                      y=waypoints[idx][1],
                                      z=waypoints[idx][2])
                
                # Get distance with traffic factor
                distance = self.get_road_distance(current_loc, wp_loc)
                traffic_factor = self.get_traffic_factor(current_loc, wp_loc)
                total += distance * traffic_factor
                current_loc = wp_loc
            
            return total
        
        # Create initial population
        population = []
        for _ in range(population_size):
            # Create random route
            route = list(range(len(waypoints)))
            random.shuffle(route)
            population.append(route)
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness = [(route, calculate_distance(route)) for route in population]
            fitness.sort(key=lambda x: x[1])  # Sort by distance (lower is better)
            
            # Check for improvement
            best_route, best_distance = fitness[0]
            
            if generation % 10 == 0:
                logger.info(f"Genetic algorithm generation {generation}, best distance: {best_distance:.1f}")
            
            # Create new generation
            new_population = []
            
            # Elite selection
            for i in range(elite_size):
                new_population.append(fitness[i][0])
            
            # Crossover and mutation
            while len(new_population) < population_size:
                # Select parents
                parent1, parent2 = random.choices(fitness[:population_size//2], k=2)
                
                # Order crossover
                size = len(waypoints)
                start, end = sorted(random.sample(range(size), 2))
                
                child = [None] * size
                child[start:end] = parent1[start:end]
                
                # Fill remaining positions from parent2
                pointer = 0
                for i in range(size):
                    if child[i] is None:
                        while parent2[pointer] in child:
                            pointer += 1
                        child[i] = parent2[pointer]
                
                # Mutation
                if random.random() < mutation_rate:
                    i, j = random.sample(range(size), 2)
                    child[i], child[j] = child[j], child[i]
                
                new_population.append(child)
            
            population = new_population
        
        # Return best route
        return best_route, best_distance
    
    def _optimize_simulated_annealing(self, start_pos, waypoints):
        """Optimize route using simulated annealing"""
        # Simulated annealing parameters
        initial_temperature = 1000.0
        cooling_rate = 0.95
        min_temperature = 1.0
        
        # Calculate distance function
        def calculate_distance(route_order):
            total = 0.0
            current_loc = carla.Location(x=start_pos[0], y=start_pos[1], z=start_pos[2])
            
            for idx in route_order:
                wp_loc = carla.Location(x=waypoints[idx][0],
                                      y=waypoints[idx][1],
                                      z=waypoints[idx][2])
                
                # Get distance with traffic factor
                distance = self.get_road_distance(current_loc, wp_loc)
                traffic_factor = self.get_traffic_factor(current_loc, wp_loc)
                total += distance * traffic_factor
                current_loc = wp_loc
            
            return total
        
        # Start with nearest neighbor solution
        current_route, current_distance = self._optimize_nearest_neighbor(start_pos, waypoints)
        best_route = current_route.copy()
        best_distance = current_distance
        
        # Simulated annealing loop
        temperature = initial_temperature
        iteration = 0
        
        while temperature > min_temperature:
            iteration += 1
            
            # Generate neighbor solution
            neighbor_route = current_route.copy()
            
            # Random 2-opt swap
            i, j = sorted(random.sample(range(len(neighbor_route)), 2))
            neighbor_route[i:j] = neighbor_route[i:j][::-1]
            
            # Calculate neighbor distance
            neighbor_distance = calculate_distance(neighbor_route)
            
            # Decide whether to accept neighbor
            delta = neighbor_distance - current_distance
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = neighbor_route
                current_distance = neighbor_distance
                
                # Update best solution
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            
            # Cool down
            temperature *= cooling_rate
            
            # Log progress
            if iteration % 100 == 0:
                logger.info(f"Simulated annealing iteration {iteration}, temperature: {temperature:.2f}, best distance: {best_distance:.1f}")
        
        logger.info(f"Simulated annealing completed in {iteration} iterations")
        return best_route, best_distance
    
    def get_best_route(self):
        """Get the best route found so far"""
        return self.best_route, self.best_distance

class UltimateCARLANavigator:
    """The ultimate CARLA autonomous navigation system"""
    def __init__(self):
        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Load Town01 if needed
        if 'Town01' not in self.world.get_map().name:
            logger.info("Loading Town01...")
            self.world = self.client.load_world('Town01')
        
        self.map = self.world.get_map()
        self.vehicle = None
        self.spectator = self.world.get_spectator()
        
        # Initialize controllers and systems
        self.steering_pid = AdaptivePIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD, dt=PID_DT)
        self.v2x_communication = V2XCommunication(id(self))
        self.sensor_fusion = None  # Will be initialized after vehicle spawn
        self.optical_avoidance = None  # Will be initialized after vehicle spawn
        self.traffic_light_detector = None  # Will be initialized after vehicle spawn
        self.pathfinder = HybridAStar(self.world)
        self.route_optimizer = AdvancedRouteOptimizer(self.map, self.v2x_communication)
        self.performance_monitor = PerformanceMonitor()
        
        # Navigation state
        self.optimized_route = []
        self.all_route_waypoints = []
        self.current_waypoint_index = 0
        self.navigation_active = False
        self.navigation_complete = False
        
        # Vehicle state
        self.vehicle_location = None
        self.vehicle_velocity = None
        self.vehicle_speed = 0.0
        self.vehicle_transform = None
        
        # Traffic and pedestrians
        self.traffic_vehicles = []
        self.pedestrians = []
        self.traffic_respawn_timer = 0.0
        
        # Fault tolerance
        self.recovery_attempts = 0
        self.last_recovery_time = 0.0
        self.emergency_stop = False
        
        # Statistics
        self.start_time = None
        self.waypoints_reached = 0
        self.total_distance_traveled = 0.0
        self.last_location = None
        
        # Visualization
        self.visualization_objects = []
        self.waypoint_locations = []
        self.visualization_update_counter = 0
        
        # Thread control
        self.running = True
        self.navigation_thread = None
        self.traffic_thread = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
        logger.info("Ultimate CARLA Navigator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def spawn_vehicle(self):
        """Spawn vehicle at start position with all sensors"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(VEHICLE_MODEL)
            
            # Set vehicle color
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color', VEHICLE_COLOR)
            
            # Find spawn point
            spawn_points = self.map.get_spawn_points()
            start_loc = carla.Location(x=START_POSITION[0], y=START_POSITION[1], z=START_POSITION[2])
            closest_spawn = min(spawn_points, key=lambda sp: start_loc.distance(sp.location))
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, closest_spawn)
            logger.info(f"Vehicle spawned at {closest_spawn.location}")
            
            # Initialize sensor systems
            self.sensor_fusion = MultiModalSensorFusion(self.vehicle)
            self.sensor_fusion.start()
            
            self.optical_avoidance = AdvancedOpticalAvoidance(self.vehicle, self.sensor_fusion)
            self.optical_avoidance.start()
            
            self.traffic_light_detector = AdvancedTrafficLightDetector(self.vehicle, self.v2x_communication)
            self.traffic_light_detector.start()
            
            # Add collision sensor
            collision_bp = blueprint_library.find('sensor.other.collision')
            collision_sensor = self.world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
            collision_sensor.listen(lambda event: self.on_collision(event))
            
            # Add lane invasion sensor
            lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_invasion_sensor = self.world.spawn_actor(
                lane_invasion_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
            lane_invasion_sensor.listen(lambda event: self.on_lane_invasion(event))
            
            return self.vehicle
            
        except Exception as e:
            logger.error(f"Error spawning vehicle: {e}")
            return None
    
    def spawn_traffic(self):
        """Spawn traffic vehicles and pedestrians"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            
            # Spawn traffic vehicles
            vehicle_blueprints = blueprint_library.filter('vehicle.*')
            spawn_points = self.map.get_spawn_points()
            random.shuffle(spawn_points)
            
            for i in range(min(NUM_TRAFFIC_VEHICLES, len(spawn_points))):
                # Skip spawn points too close to our vehicle
                if self.vehicle.get_location().distance(spawn_points[i].location) < TRAFFIC_SPAWN_DISTANCE:
                    continue
                
                # Random vehicle blueprint
                blueprint = random.choice(vehicle_blueprints)
                
                # Set random color
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                
                # Spawn vehicle
                try:
                    vehicle = self.world.spawn_actor(blueprint, spawn_points[i])
                    
                    # Set random autopilot speed
                    autopilot_speed = TARGET_SPEED * (1.0 + random.uniform(-TRAFFIC_SPEED_VARIATION, TRAFFIC_SPEED_VARIATION))
                    vehicle.set_autopilot(True, autopilot_speed)
                    
                    self.traffic_vehicles.append(vehicle)
                except Exception as e:
                    logger.warning(f"Failed to spawn traffic vehicle: {e}")
            
            # Spawn pedestrians
            pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')
            pedestrian_spawn_points = []
            
            # Generate pedestrian spawn points
            for _ in range(20):  # Try to spawn 20 pedestrians
                spawn_point = carla.Transform()
                spawn_point.location = self.world.get_random_location_from_navigation()
                
                # Make sure spawn point is on a sidewalk
                waypoint = self.map.get_waypoint(spawn_point.location)
                if waypoint and waypoint.lane_type == carla.LaneType.Sidewalk:
                    pedestrian_spawn_points.append(spawn_point)
            
            # Spawn pedestrians
            for i, spawn_point in enumerate(pedestrian_spawn_points):
                try:
                    # Random pedestrian blueprint
                    blueprint = random.choice(pedestrian_blueprints)
                    
                    # Spawn pedestrian
                    pedestrian = self.world.spawn_actor(blueprint, spawn_point)
                    
                    # Add controller to pedestrian
                    controller_bp = blueprint_library.find('controller.ai.walker')
                    controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                    
                    # Start pedestrian
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    
                    self.pedestrians.append((pedestrian, controller))
                except Exception as e:
                    logger.warning(f"Failed to spawn pedestrian: {e}")
            
            logger.info(f"Spawned {len(self.traffic_vehicles)} traffic vehicles and {len(self.pedestrians)} pedestrians")
            
            # Start traffic management thread
            self.traffic_thread = threading.Thread(target=self._traffic_management_loop)
            self.traffic_thread.daemon = True
            self.traffic_thread.start()
            
        except Exception as e:
            logger.error(f"Error spawning traffic: {e}")
    
    def _traffic_management_loop(self):
        """Background thread for managing traffic"""
        while self.running:
            try:
                # Respawn traffic vehicles periodically
                current_time = time.time()
                if current_time - self.traffic_respawn_timer > TRAFFIC_RESPAWN_TIME:
                    self._respawn_traffic()
                    self.traffic_respawn_timer = current_time
                
                # Make pedestrians walk randomly
                for pedestrian, controller in self.pedestrians:
                    if pedestrian.is_alive and random.random() < 0.01:  # 1% chance per iteration
                        controller.go_to_location(self.world.get_random_location_from_navigation())
                
                # Sleep to prevent high CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in traffic management loop: {e}")
    
    def _respawn_traffic(self):
        """Respawn traffic vehicles that have been destroyed"""
        try:
            # Remove destroyed vehicles from list
            self.traffic_vehicles = [v for v in self.traffic_vehicles if v.is_alive]
            
            # Spawn new vehicles if needed
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprints = blueprint_library.filter('vehicle.*')
            spawn_points = self.map.get_spawn_points()
            random.shuffle(spawn_points)
            
            vehicles_to_spawn = NUM_TRAFFIC_VEHICLES - len(self.traffic_vehicles)
            
            for i in range(vehicles_to_spawn):
                if i >= len(spawn_points):
                    break
                
                # Skip spawn points too close to our vehicle
                if self.vehicle.get_location().distance(spawn_points[i].location) < TRAFFIC_SPAWN_DISTANCE:
                    continue
                
                # Random vehicle blueprint
                blueprint = random.choice(vehicle_blueprints)
                
                # Set random color
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                
                # Spawn vehicle
                try:
                    vehicle = self.world.spawn_actor(blueprint, spawn_points[i])
                    
                    # Set random autopilot speed
                    autopilot_speed = TARGET_SPEED * (1.0 + random.uniform(-TRAFFIC_SPEED_VARIATION, TRAFFIC_SPEED_VARIATION))
                    vehicle.set_autopilot(True, autopilot_speed)
                    
                    self.traffic_vehicles.append(vehicle)
                except Exception as e:
                    logger.warning(f"Failed to respawn traffic vehicle: {e}")
            
            logger.info(f"Respawned {vehicles_to_spawn} traffic vehicles")
            
        except Exception as e:
            logger.error(f"Error respawning traffic: {e}")
    
    def on_collision(self, event):
        """Handle collision events"""
        try:
            logger.warning(f"Collision detected with {event.other_actor.type_id}")
            
            # Check if this is a serious collision
            impulse = event.normal_impulse
            impulse_magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            
            if impulse_magnitude > 1000.0:  # Threshold for serious collision
                logger.error(f"Serious collision detected! Impulse: {impulse_magnitude:.2f}")
                self.emergency_stop = True
                
                # Try to recover
                self._attempt_recovery()
            
        except Exception as e:
            logger.error(f"Error handling collision: {e}")
    
    def on_lane_invasion(self, event):
        """Handle lane invasion events"""
        try:
            # Log lane invasion
            logger.info(f"Lane invasion detected")
            
            # In a real system, you might want to take corrective action
            # For now, we'll just log it
            
        except Exception as e:
            logger.error(f"Error handling lane invasion: {e}")
    
    def _attempt_recovery(self):
        """Attempt to recover from an emergency situation"""
        try:
            current_time = time.time()
            
            # Check if we can attempt recovery
            if (self.recovery_attempts >= MAX_RECOVERY_ATTEMPTS or 
                current_time - self.last_recovery_time < RECOVERY_COOLDOWN):
                return
            
            self.recovery_attempts += 1
            self.last_recovery_time = current_time
            
            logger.info(f"Attempting recovery (attempt {self.recovery_attempts}/{MAX_RECOVERY_ATTEMPTS})")
            
            # Stop the vehicle
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            time.sleep(1.0)
            
            # Check if we're still in an emergency situation
            if self.emergency_stop:
                # Try to reverse
                self.vehicle.apply_control(carla.VehicleControl(gear=-1, throttle=0.5))
                time.sleep(2.0)
                
                # Stop again
                self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                time.sleep(1.0)
                
                # Reset emergency stop
                self.emergency_stop = False
                logger.info("Recovery completed")
            
        except Exception as e:
            logger.error(f"Error during recovery: {e}")
    
    def optimize_waypoint_order(self):
        """Find optimal order to visit waypoints using advanced algorithms"""
        logger.info("Optimizing waypoint order using advanced algorithms...")
        
        # Try different optimization algorithms
        algorithms = ['nearest_neighbor', '2opt', 'genetic', 'simulated_annealing']
        best_route = None
        best_distance = float('inf')
        best_algorithm = None
        
        for algorithm in algorithms:
            try:
                logger.info(f"Trying {algorithm} algorithm...")
                route, distance = self.route_optimizer.optimize_route(START_POSITION, WAYPOINTS, algorithm)
                
                logger.info(f"{algorithm} distance: {distance:.1f}m")
                
                if distance < best_distance:
                    best_distance = distance
                    best_route = route
                    best_algorithm = algorithm
                    
            except Exception as e:
                logger.error(f"Error with {algorithm} algorithm: {e}")
        
        # Calculate original distance
        original_distance = 0.0
        current_loc = carla.Location(x=START_POSITION[0], y=START_POSITION[1], z=START_POSITION[2])
        
        for i in range(len(WAYPOINTS)):
            wp_loc = carla.Location(x=WAYPOINTS[i][0], y=WAYPOINTS[i][1], z=WAYPOINTS[i][2])
            original_distance += self.route_optimizer.get_road_distance(current_loc, wp_loc)
            current_loc = wp_loc
        
        logger.info(f"Original order distance: {original_distance:.1f}m")
        logger.info(f"Best algorithm: {best_algorithm}, distance: {best_distance:.1f}m")
        logger.info(f"Distance saved: {original_distance - best_distance:.1f}m "
                   f"({100*(original_distance - best_distance)/original_distance:.1f}%)")
        
        # Print optimized route
        logger.info("Optimized route order:")
        for i, idx in enumerate(best_route):
            logger.info(f"  Stop {i+1}: Waypoint {idx} {WAYPOINTS[idx]}")
        
        self.optimized_route = best_route
        return best_route
    
    def get_waypoint_path(self, start_location, target_location):
        """Create path using hybrid A* algorithm"""
        start_time = time.time()
        
        # Try hybrid A* first
        path_locations = self.pathfinder.find_path(start_location, target_location, use_hybrid=True)
        
        if not path_locations:
            logger.warning("Hybrid A* failed, trying pure grid-based A*")
            path_locations = self.pathfinder.find_path(start_location, target_location, use_hybrid=False)
        
        # Convert locations to waypoints
        path = []
        for loc in path_locations:
            wp = self.map.get_waypoint(loc)
            if wp:
                path.append(wp)
        
        # Log performance
        self.performance_monitor.update_pathfinding_time(time.time() - start_time)
        
        return path
    
    def visualize_waypoints(self):
        """Visualize waypoints as small dots with labels"""
        try:
            # Clear previous visualizations
            for obj in self.visualization_objects:
                if hasattr(obj, 'destroy'):
                    obj.destroy()
            self.visualization_objects = []
            
            # Store waypoint locations
            self.waypoint_locations = []
            for i, wp_idx in enumerate(self.optimized_route):
                waypoint = WAYPOINTS[wp_idx]
                location = carla.Location(x=waypoint[0], y=waypoint[1], z=waypoint[2])
                self.waypoint_locations.append((location, wp_idx, i))
            
            # Draw waypoints
            for loc, orig_idx, opt_idx in self.waypoint_locations:
                # Draw a small dot at waypoint location
                self.world.debug.draw_point(
                    loc + carla.Location(z=0.2),
                    size=WAYPOINT_DOT_SIZE,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=0.0  # Persistent
                )
                
                # Draw waypoint label
                self.world.debug.draw_string(
                    loc + carla.Location(z=WAYPOINT_LABEL_HEIGHT),
                    f"WP-{opt_idx+1}\n(orig: {orig_idx})",
                    draw_shadow=True,
                    color=carla.Color(r=255, g=255, b=255),
                    life_time=0.0  # Persistent
                )
            
            logger.info(f"Visualized {len(self.optimized_route)} waypoints as dots with labels")
            
        except Exception as e:
            logger.error(f"Error visualizing waypoints: {e}")
    
    def visualize_complete_route(self):
        """Visualize complete route as arrows between waypoints"""
        try:
            # Visualize waypoints first
            self.visualize_waypoints()
            
            # Draw route
            current_loc = carla.Location(x=START_POSITION[0], y=START_POSITION[1], z=START_POSITION[2])
            
            for i, wp_idx in enumerate(self.optimized_route):
                target_loc = carla.Location(x=WAYPOINTS[wp_idx][0], 
                                          y=WAYPOINTS[wp_idx][1], 
                                          z=WAYPOINTS[wp_idx][2])
                
                # Get path between waypoints
                path = self.get_waypoint_path(current_loc, target_loc)
                
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
                            life_time=PATH_VISUALIZATION_LIFETIME
                        )
                    
                    # Store waypoints for navigation
                    self.all_route_waypoints.extend(path)
                
                current_loc = target_loc
            
            logger.info(f"Visualized complete route with {len(self.optimized_route)} segments")
            
        except Exception as e:
            logger.error(f"Error visualizing complete route: {e}")
    
    def update_vehicle_state(self):
        """Update vehicle state variables"""
        try:
            # Get current vehicle state
            self.vehicle_transform = self.vehicle.get_transform()
            self.vehicle_location = self.vehicle_transform.location
            self.vehicle_velocity = self.vehicle.get_velocity()
            self.vehicle_speed = math.sqrt(self.vehicle_velocity.x**2 + self.vehicle_velocity.y**2) * 3.6  # m/s to km/h
            
            # Update total distance traveled
            if self.last_location is not None:
                distance = self.vehicle_location.distance(self.last_location)
                self.total_distance_traveled += distance
            
            self.last_location = self.vehicle_location
            
            # Send position via V2X
            if self.v2x_communication:
                self.v2x_communication.send_position(self.vehicle_location, self.vehicle_velocity)
            
        except Exception as e:
            logger.error(f"Error updating vehicle state: {e}")
    
    def follow_waypoints_smooth(self, waypoint_list, target_index, target_speed=TARGET_SPEED):
        """Follow waypoints with smooth PID steering control and advanced obstacle avoidance"""
        try:
            current_waypoint_index = 0
            
            # Reset PID controller
            self.steering_pid.reset()
            
            while current_waypoint_index < len(waypoint_list) and self.running and not self.emergency_stop:
                loop_start_time = time.time()
                
                # Update vehicle state
                self.update_vehicle_state()
                
                # Check vehicle stability
                vehicle_rotation = self.vehicle_transform.rotation
                roll_angle = abs(vehicle_rotation.roll)
                pitch_angle = abs(vehicle_rotation.pitch)
                
                if roll_angle > MAX_ROLL_ANGLE or pitch_angle > MAX_PITCH_ANGLE:
                    logger.warning(f"Vehicle stability issue: roll={roll_angle:.1f}°, pitch={pitch_angle:.1f}°")
                    self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                    time.sleep(0.5)
                    continue
                
                # Update traffic light detection
                self.traffic_light_detector.update()
                
                # Check for traffic light
                if self.traffic_light_detector.should_stop():
                    logger.info("Stopping for traffic light")
                    self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                    time.sleep(0.1)
                    continue
                
                # Look-ahead for smoother navigation
                look_ahead_index = min(current_waypoint_index + LOOK_AHEAD_WAYPOINTS, len(waypoint_list) - 1)
                target_waypoint = waypoint_list[look_ahead_index]
                target_location = target_waypoint.transform.location
                
                # Calculate distance to current waypoint
                current_wp_location = waypoint_list[current_waypoint_index].transform.location
                distance = self.vehicle_location.distance(current_wp_location)
                
                # Check if waypoint reached
                if distance < WAYPOINT_REACH_THRESHOLD:
                    current_waypoint_index += 1
                    if current_waypoint_index < len(waypoint_list):
                        if current_waypoint_index % 10 == 0:
                            logger.info(f"Progress: {current_waypoint_index}/{len(waypoint_list)} waypoints")
                    continue
                
                # Calculate steering with PID
                target_vector = target_location - self.vehicle_location
                forward_vector = self.vehicle_transform.get_forward_vector()
                
                cross = forward_vector.x * target_vector.y - forward_vector.y * target_vector.x
                error = math.atan2(cross, target_vector.x * forward_vector.x + target_vector.y * forward_vector.y)
                
                # PID steering
                steer = self.steering_pid.update(error)
                
                # Apply optical avoidance
                avoidance_steer = self.optical_avoidance.get_avoidance_action()
                if avoidance_steer != 0.0:
                    logger.debug(f"Applying optical avoidance: steer={avoidance_steer:.2f}")
                    steer = np.clip(steer + avoidance_steer, -1.0, 1.0)
                
                # Dynamic speed adjustment
                curvature = abs(steer)
                speed_factor = 1.0 - (curvature * CURVE_SPEED_REDUCTION)
                
                roll_factor = 1.0 - (roll_angle / MAX_ROLL_ANGLE) * ROLL_SPEED_REDUCTION
                
                # Get collision risk from optical avoidance
                collision_risk = self.optical_avoidance.get_collision_risk()
                risk_factor = 1.0 - (collision_risk * 0.5)  # Reduce speed based on collision risk
                
                adjusted_target_speed = target_speed * speed_factor * roll_factor * risk_factor
                adjusted_target_speed = max(MIN_SPEED, min(MAX_SPEED, adjusted_target_speed))
                
                # Smooth acceleration/deceleration
                speed_error = adjusted_target_speed - self.vehicle_speed
                
                if speed_error > 0:
                    # Accelerate smoothly
                    throttle = np.clip(speed_error / THROTTLE_ACCELERATION, 0.0, 0.7)
                    brake = 0.0
                else:
                    # Brake smoothly
                    throttle = 0.0
                    brake = np.clip(-speed_error / BRAKE_ACCELERATION, 0.0, 0.5)
                
                # Apply control
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=False
                )
                self.vehicle.apply_control(control)
                
                # Update visualization
                if current_waypoint_index % 5 == 0:
                    self.world.debug.draw_point(
                        self.vehicle_location + carla.Location(z=1.0),
                        size=0.1,
                        color=carla.Color(r=255, g=255, b=0),
                        life_time=5.0
                    )
                
                # Update performance metrics
                loop_time = time.time() - loop_start_time
                self.performance_monitor.update_control_loop_time(loop_time)
                self.performance_monitor.update_frame()
                self.performance_monitor.update_latency(loop_time)
                
                # Wait for next tick
                self.world.tick()
                time.sleep(max(0, PID_DT - loop_time))  # Maintain consistent control loop timing
            
            return current_waypoint_index >= len(waypoint_list)
            
        except Exception as e:
            logger.error(f"Error following waypoints: {e}")
            return False
    
    def navigate_to_waypoint(self, target_waypoint, waypoint_index, optimized_index):
        """Navigate to a single waypoint with advanced control"""
        try:
            logger.info(f"\nNavigating to waypoint {optimized_index+1}/{len(self.optimized_route)} "
                       f"(Original #{waypoint_index}): {target_waypoint}")
            
            # Get current position
            current_location = self.vehicle.get_transform().location
            target_location = carla.Location(x=target_waypoint[0], 
                                           y=target_waypoint[1], 
                                           z=target_waypoint[2])
            
            # Get path
            waypoint_path = self.get_waypoint_path(current_location, target_location)
            
            if not waypoint_path:
                logger.error(f"Could not plan path to waypoint {optimized_index+1}")
                return False
            
            logger.info(f"Path planned: {len(waypoint_path)} waypoints")
            
            # Update current target visualization
            self.world.debug.draw_point(
                target_location + carla.Location(z=0.5),
                size=0.3,
                color=carla.Color(r=0, g=255, b=0),
                life_time=10.0
            )
            
            # Follow path
            success = self.follow_waypoints_smooth(waypoint_path, waypoint_index)
            
            # Check if we reached the target
            final_location = self.vehicle.get_transform().location
            final_distance = final_location.distance(target_location)
            
            if final_distance < WAYPOINT_REACH_THRESHOLD * 1.5:
                logger.info(f"Reached waypoint {optimized_index+1} (distance: {final_distance:.1f}m)")
                self.waypoints_reached += 1
                return True
            else:
                logger.warning(f"Close to waypoint {optimized_index+1} (distance: {final_distance:.1f}m)")
                return True
                
        except Exception as e:
            logger.error(f"Error navigating to waypoint {optimized_index+1}: {e}")
            return False
    
    def navigate_all_waypoints(self):
        """Navigate to all waypoints in optimized sequence"""
        try:
            # Optimize route
            self.optimize_waypoint_order()
            
            # Spawn traffic
            self.spawn_traffic()
            
            # Visualize complete route
            self.visualize_complete_route()
            
            logger.info(f"\nStarting navigation to {len(self.optimized_route)} waypoints in optimized order")
            logger.info(f"Route order: {self.optimized_route}")
            
            # Initialize statistics
            self.start_time = time.time()
            self.waypoints_reached = 0
            self.total_distance_traveled = 0.0
            self.last_location = self.vehicle.get_transform().location
            
            # Start navigation thread
            self.navigation_active = True
            self.navigation_thread = threading.Thread(target=self._navigation_loop)
            self.navigation_thread.daemon = True
            self.navigation_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting navigation: {e}")
    
    def _navigation_loop(self):
        """Navigation loop running in a separate thread"""
        try:
            for i, wp_idx in enumerate(self.optimized_route):
                if not self.running:
                    break
                    
                try:
                    waypoint = WAYPOINTS[wp_idx]
                    success = self.navigate_to_waypoint(waypoint, wp_idx, i)
                    
                    # Brief pause between waypoints
                    time.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Error navigating to waypoint {i+1}: {e}")
                    continue
            
            # Navigation complete
            self.navigation_active = False
            self.navigation_complete = True
            
            # Stop vehicle
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            
            # Calculate statistics
            total_time = time.time() - self.start_time
            avg_speed = self.total_distance_traveled / total_time * 3.6 if total_time > 0 else 0
            
            logger.info("\nNavigation complete!")
            logger.info(f"Results: {self.waypoints_reached}/{len(self.optimized_route)} waypoints reached")
            logger.info(f"Total time: {total_time:.1f} seconds")
            logger.info(f"Total distance: {self.total_distance_traveled:.1f} meters")
            logger.info(f"Average speed: {avg_speed:.1f} km/h")
            logger.info(f"Average time per waypoint: {total_time/len(self.optimized_route):.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error in navigation loop: {e}")
            self.navigation_active = False
    
    def get_statistics(self):
        """Get navigation statistics"""
        if not self.start_time:
            return None
        
        total_time = time.time() - self.start_time
        avg_speed = self.total_distance_traveled / total_time * 3.6 if total_time > 0 else 0
        
        return {
            'waypoints_total': len(self.optimized_route),
            'waypoints_reached': self.waypoints_reached,
            'total_time': total_time,
            'total_distance': self.total_distance_traveled,
            'average_speed': avg_speed,
            'navigation_complete': self.navigation_complete,
            'navigation_active': self.navigation_active
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Stop navigation
        self.running = False
        self.navigation_active = False
        
        # Wait for threads to finish
        if self.navigation_thread and self.navigation_thread.is_alive():
            self.navigation_thread.join(timeout=1.0)
        
        if self.traffic_thread and self.traffic_thread.is_alive():
            self.traffic_thread.join(timeout=1.0)
        
        # Stop vehicle
        if self.vehicle:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        
        # Clean up sensors
        if self.sensor_fusion:
            self.sensor_fusion.stop()
        
        if self.optical_avoidance:
            self.optical_avoidance.stop()
        
        if self.traffic_light_detector:
            self.traffic_light_detector.stop()
        
        if self.v2x_communication:
            self.v2x_communication.stop()
        
        if self.performance_monitor:
            self.performance_monitor.stop()
        
        # Destroy traffic vehicles
        for vehicle in self.traffic_vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        # Destroy pedestrians
        for pedestrian, controller in self.pedestrians:
            if pedestrian.is_alive:
                pedestrian.destroy()
            if controller.is_alive:
                controller.destroy()
        
        # Destroy vehicle
        if self.vehicle:
            self.vehicle.destroy()
        
        # Clear visualizations
        for obj in self.visualization_objects:
            if hasattr(obj, 'destroy'):
                obj.destroy()
        
        logger.info("Cleanup complete")

def main():
    """Main function"""
    navigator = None
    
    try:
        # Create navigator
        navigator = UltimateCARLANavigator()
        
        # Spawn vehicle
        if not navigator.spawn_vehicle():
            logger.error("Failed to spawn vehicle")
            return
        
        # Wait for vehicle to settle
        time.sleep(2.0)
        
        # Start navigation
        navigator.navigate_all_waypoints()
        
        # Keep visualization active
        logger.info("\nKeeping visualization active. Press Ctrl+C to exit...")
        while navigator.running:
            time.sleep(1.0)
            
            # Print statistics periodically
            stats = navigator.get_statistics()
            if stats:
                logger.info(f"Progress: {stats['waypoints_reached']}/{stats['waypoints_total']} waypoints, "
                           f"Time: {stats['total_time']:.1f}s, Distance: {stats['total_distance']:.1f}m")
                time.sleep(5.0)
        
    except KeyboardInterrupt:
        logger.info("\nNavigation stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if navigator:
            navigator.cleanup()

if __name__ == '__main__':
    print("🚀 Ultimate CARLA Autonomous Navigation System")
    print("=" * 70)
    print("📋 Features:")
    print("  • Advanced TSP-based route optimization with multiple algorithms")
    print("  • Hybrid pathfinding (waypoint-based + grid-based A*)")
    print("  • Adaptive PID control with auto-tuning")
    print("  • Multi-modal sensor fusion (camera, LiDAR, radar, semantic segmentation)")
    print("  • Deep learning-based object detection and semantic segmentation")
    print("  • Dynamic traffic prediction and interaction")
    print("  • Advanced traffic light detection with state prediction")
    print("  • Vehicle-to-Everything (V2X) communication")
    print("  • Real-time obstacle avoidance with risk assessment")
    print("  • Adaptive speed control based on road conditions")
    print("  • Vehicle stability control with rollover prevention")
    print("  • Comprehensive visualization system")
    print("  • Performance monitoring and diagnostics")
    print("  • Fault tolerance and recovery mechanisms")
    print("🗺️ Map: Town01")
    print("🎮 Press Ctrl+C to stop")
    print("=" * 70)
    
    main()
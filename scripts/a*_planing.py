#!/usr/bin/env python3
import carla
import random
import time
import numpy as np
import math
import queue

# Parameters
GRID_SIZE = 1.0  # meters per grid cell
TARGET_SPEED = 5.0  # m/s (reduced for precision)
GRID_WIDTH = 200  # cells
GRID_HEIGHT = 200  # cells
OBSTACLE_BUFFER = 0.5  # buffer around obstacles
LANE_DIVIDER_WIDTH = 0.2  # meters for lane dividers
DEBUG = True  # Enable debugging visualizations

# Color codes
RED = carla.Color(255, 0, 0)
GREEN = carla.Color(0, 255, 0)
BLUE = carla.Color(0, 0, 255)
YELLOW = carla.Color(255, 255, 0)
WHITE = carla.Color(255, 255, 255)

# Global variables for cleanup
vehicle = None
debug_objects = []

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

def draw_debug(world, grid, origin, path, start, goal):
    """Enhanced debug visualization"""
    global debug_objects
    
    # Clear previous debug objects
    for obj in debug_objects:
        try:
            obj.destroy()
        except:
            pass
    debug_objects = []
    
    # Draw grid cells
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            loc = carla.Location(
                x=origin.x + x * GRID_SIZE + GRID_SIZE / 2,
                y=origin.y + y * GRID_SIZE + GRID_SIZE / 2,
                z=0.1
            )
            if grid[x][y] == 1:
                point = world.debug.draw_point(
                    loc, 
                    size=0.1, 
                    color=RED, 
                    life_time=30.0,
                    persistent_lines=False
                )
                debug_objects.append(point)
            else:
                # Draw ground with white squares
                point = world.debug.draw_point(
                    loc, 
                    size=0.05, 
                    color=WHITE, 
                    life_time=30.0,
                    persistent_lines=False
                )
                debug_objects.append(point)
    
    # Draw start and goal
    start_loc = carla.Location(
        x=origin.x + start[0] * GRID_SIZE + GRID_SIZE / 2,
        y=origin.y + start[1] * GRID_SIZE + GRID_SIZE / 2,
        z=0.5
    )
    goal_loc = carla.Location(
        x=origin.x + goal[0] * GRID_SIZE + GRID_SIZE / 2,
        y=origin.y + goal[1] * GRID_SIZE + GRID_SIZE / 2,
        z=0.5
    )
    
    start_point = world.debug.draw_point(
        start_loc, 
        size=0.3, 
        color=BLUE, 
        life_time=30.0,
        persistent_lines=False
    )
    goal_point = world.debug.draw_point(
        goal_loc, 
        size=0.3, 
        color=YELLOW, 
        life_time=30.0,
        persistent_lines=False
    )
    debug_objects.extend([start_point, goal_point])
    
    # Draw path
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
                thickness=0.1, 
                color=GREEN, 
                life_time=30.0,
                persistent_lines=False
            )
            debug_objects.append(line)

def find_valid_goal(grid, origin, center, world):
    """Find a reachable goal position by raycasting"""
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts:
        # Try points in increasing distances
        distance = 20 + attempts * 5
        angle = random.uniform(0, 2 * math.pi)
        x = center.x + distance * math.cos(angle)
        y = center.y + distance * math.sin(angle)
        
        # Convert to grid coordinates
        grid_x = int((x - origin.x) / GRID_SIZE)
        grid_y = int((y - origin.y) / GRID_SIZE)
        
        # Check bounds and walkability
        if (0 <= grid_x < len(grid)) and (0 <= grid_y < len(grid[0])):
            if grid[grid_x][grid_y] == 0:  # Free space
                # Verify with actual raycast (optional)
                waypoint = world.get_map().get_waypoint(carla.Location(x, y, 0))
                if waypoint:
                    return (grid_x, grid_y)
        
        attempts += 1
    
    return None  # Fallback to default if no valid goal found

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

def main():
    global vehicle, debug_objects
    
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        client.load_world('Town03')  # Change to your desired map
        world = client.get_world()
        
        
    
        
        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.audi.a2')  # Fixed vehicle model
        spawn_points = world.get_map().get_spawn_points()
        
        

        # Find valid spawn point
      #  vehicle = None
      #  for sp in spawn_points:
      #      try:
      #          vehicle = world.try_spawn_actor(vehicle_bp, sp)
      #          if vehicle:
      #              print(f"Spawned vehicle at {sp.location}")
      #             break
      #      except:
      #          continue
        
      #  if not vehicle:
      #      print("Failed to spawn vehicle!")
      #      return

        # spawn at spicific cordinat
        spawn_transform = carla.Transform(
         carla.Location(x=-6.446170, y=-79.055023, z=0.275307),  # Change x, y, z to your desired start
         carla.Rotation(yaw=90)
        )

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
        if vehicle:
          print(f"Spawned vehicle at {spawn_transform.location}")
        else:
          print("Failed to spawn vehicle at custom location!")
          return
        
     # spawn at spicific point
    #    spawn_index = 5  # Change this to the index you want (e.g., 0, 1, 2, ..., 44)
    #    spawn_points = world.get_map().get_spawn_points()

    #    if spawn_index < len(spawn_points):
    #        spawn_transform = spawn_points[spawn_index]
    #        vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    #        if vehicle:
    #            print(f"✅ Vehicle spawned at spawn point index {spawn_index}: {spawn_transform.location}")
    #        else:
    #            print("❌ Failed to spawn vehicle at this location!")
    #    else:
    #        print(f"❌ Invalid spawn index. Map has only {len(spawn_points)} spawn points.")        




        # Set spectator view
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=50),
            carla.Rotation(pitch=-90)
        ))
        
        # Create obstacle grid from actual map data
        print("Generating obstacle grid...")
        grid, origin = get_obstacle_grid(world, transform.location)
        
        # Set start position (current location)
        start_x = int((transform.location.x - origin.x) / GRID_SIZE)
        start_y = int((transform.location.y - origin.y) / GRID_SIZE)

        # Set start position (possitiom you want to start from)
        #start_x = int(90)
        #start_y = int(100)

        start = (max(0, min(len(grid)-1, start_x)), max(0, min(len(grid[0])-1, start_y)))
        
        if grid[start[0]][start[1]] == 1:
            print("ERROR: Start position is blocked by obstacle!")
            return
        
        # Find valid goal position
        print("Finding valid goal position...")
        #random
        #goal = find_valid_goal(grid, origin, transform.location, world)

        #choc a goal
        goal = (84, 112)

        if not goal:
            print("Couldn't find valid goal position, using default")
            goal = (start[0] + 20, start[1] + 20)  # Default relative position
            if (goal[0] >= len(grid) or goal[1] >= len(grid[0])):
                goal = (len(grid)-1, len(grid[0])-1)
            if grid[goal[0]][goal[1]] == 1:
                print("Default goal is also blocked - no valid path possible")
                return
        
        print(f"Planning from {start} to {goal}...")
        
        # Find path using A*
        path = a_star(grid, start, goal)
        
        if not path:
            print("No path found to goal!")
            return
        
        print(f"Found path with {len(path)} waypoints")
        
        # Draw debug info
        if DEBUG:
            draw_debug(world, grid, origin, path, start, goal)
        
        # Follow path
        try:
            for node in path:
                target_loc = carla.Location(
                    x=origin.x + node[0] * GRID_SIZE + GRID_SIZE / 2,
                    y=origin.y + node[1] * GRID_SIZE + GRID_SIZE / 2,
                    z=0.3
                )
                
                # Keep moving toward this waypoint until we're close enough
                while True:
                    dist = control_vehicle(vehicle, target_loc)
                    time.sleep(0.05)
                    if dist < 2.0:  # Close enough to move to next point
                        break
        
        finally:
            # Apply brake at the end
            control = carla.VehicleControl()
            control.brake = 1.0
            vehicle.apply_control(control)
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        cleanup()

if __name__ == '__main__':
    main()

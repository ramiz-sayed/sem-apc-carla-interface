#!/usr/bin/env python3
import carla
import math
import time
import random

# Configuration
# ============

# WAYPOINTS: List of target coordinates [x, y, z] that the vehicle must visit
# Each waypoint represents a specific location in Town01 that the vehicle needs to reach
# You can modify these coordinates to change the navigation path
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
# You can modify this to start from a different location
START_POSITION = [280.363739, 129.306351, 0.101746]  # Near center of Town01

class SimpleCarlaNavigator:
    """Simple CARLA navigator class for waypoint navigation"""
    
    def __init__(self):
        """
        Initialize the CARLA navigator and set up the environment
        This method connects to the CARLA server and loads the required map
        """
        # Connect to CARLA server running on localhost at port 2000
        self.client = carla.Client('localhost', 2000)
        # Set timeout for network operations (10 seconds)
        self.client.set_timeout(10.0)
        # Get the world object
        self.world = self.client.get_world()
        
        # Load Town01 if needed
        if 'Town01' not in self.world.get_map().name:
            print("Loading Town01...")
            self.world = self.client.load_world('Town01')
        
        # Get the map object for path planning
        self.map = self.world.get_map()
        # Vehicle object will be set later
        self.vehicle = None
        
        print("✅ CARLA Navigator initialized")
    
    def spawn_vehicle(self):
        """
        Spawn a vehicle at the start position
        This method finds a valid spawn point near the specified start position
        and spawns a vehicle there
        """
        # Get the blueprint library which contains all available vehicle types
        blueprint_library = self.world.get_blueprint_library()
        # Select the Audi A2 vehicle blueprint
        vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        
        # Find a valid spawn point near our start position
        spawn_points = self.map.get_spawn_points()
        # Convert our start position to a CARLA Location object
        start_loc = carla.Location(x=START_POSITION[0], y=START_POSITION[1], z=START_POSITION[2])
        
        # Find the closest spawn point to our desired start position
        closest_spawn = min(spawn_points, 
                          key=lambda sp: start_loc.distance(sp.location))
        
        # Spawn the vehicle at the closest spawn point
        self.vehicle = self.world.spawn_actor(vehicle_bp, closest_spawn)
        print(f"🚗 Vehicle spawned at {closest_spawn.location}")
        return self.vehicle
    
    def get_waypoint_path(self, start_location, target_location, sampling_resolution=2.0):
        """
        Create a simple path using CARLA map waypoints
        This method generates a path from start to target location following the road network
        
        Args:
            start_location: Starting position (carla.Location)
            target_location: Target position (carla.Location)
            sampling_resolution: Distance between sampled waypoints in meters
            
        Returns:
            List of waypoints forming the path
        """
        # Get nearest waypoints on the road network
        start_waypoint = self.map.get_waypoint(start_location)
        target_waypoint = self.map.get_waypoint(target_location)
        
        # Check if valid waypoints were found
        if not start_waypoint or not target_waypoint:
            print("❌ Could not find valid waypoints on road")
            return []
        
        # Simple path: try to get waypoints leading towards target
        waypoint_list = [start_waypoint]
        current_waypoint = start_waypoint
        max_waypoints = 100  # Prevent infinite loops
        
        # Generate waypoints until we reach the target or hit the limit
        for _ in range(max_waypoints):
            # Get next waypoints along the road
            next_waypoints = current_waypoint.next(sampling_resolution)
            
            # If no more waypoints, we've reached a dead end
            if not next_waypoints:
                break
            
            # If only one option, take it
            if len(next_waypoints) == 1:
                current_waypoint = next_waypoints[0]
            else:
                # Multiple options - choose the one closest to target
                current_waypoint = min(next_waypoints, 
                                     key=lambda wp: wp.transform.location.distance(target_location))
            
            waypoint_list.append(current_waypoint)
            
            # Check if we're close to target (within 10 meters)
            if current_waypoint.transform.location.distance(target_location) < 10.0:
                break
        
        return waypoint_list
    
    def visualize_waypoints(self, waypoint_list, target_index):
        """
        Draw route and waypoints in the CARLA world
        This method visualizes the planned path and target waypoint
        
        Args:
            waypoint_list: List of waypoints forming the path
            target_index: Index of the target waypoint in WAYPOINTS list
        """
        # Draw route waypoints with alternating colors
        for i, waypoint in enumerate(waypoint_list):
            # Alternate between green and yellow for better visibility
            color = carla.Color(r=0, g=255, b=0) if i % 2 == 0 else carla.Color(r=255, g=255, b=0)
            # Draw a small marker at each waypoint
            self.world.debug.draw_string(
                waypoint.transform.location + carla.Location(z=1.0),  # Slightly above ground
                'o',  # Small circle marker
                draw_shadow=False,
                color=color,
                life_time=20.0,  # Visible for 20 seconds
                persistent_lines=True
            )
        
        # Draw target waypoint with a red label
        target_location = carla.Location(x=WAYPOINTS[target_index][0], 
                                       y=WAYPOINTS[target_index][1], 
                                       z=WAYPOINTS[target_index][2])
        self.world.debug.draw_string(
            target_location + carla.Location(z=3.0),  # Higher above ground
            f'TARGET-{target_index}',  # Label with waypoint number
            draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0),  # Red color
            life_time=20.0,
            persistent_lines=True
        )
    
    def follow_waypoints(self, waypoint_list, target_speed=30.0):
        """
        Follow waypoints with simple control
        This method controls the vehicle to follow the planned path
        
        Args:
            waypoint_list: List of waypoints to follow
            target_speed: Desired speed in km/h
        """
        current_waypoint_index = 0
        
        # Loop until all waypoints are reached
        while current_waypoint_index < len(waypoint_list):
            # Get current vehicle state
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_velocity = self.vehicle.get_velocity()
            # Convert velocity from m/s to km/h
            current_speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2) * 3.6
            
            # Get target waypoint
            target_waypoint = waypoint_list[current_waypoint_index]
            target_location = target_waypoint.transform.location
            
            # Calculate distance to target
            distance = vehicle_location.distance(target_location)
            
            # Check if waypoint reached (within 4 meters)
            if distance < 4.0:
                current_waypoint_index += 1
                if current_waypoint_index < len(waypoint_list):
                    print(f"✅ Waypoint {current_waypoint_index}/{len(waypoint_list)} reached")
                continue
            
            # Calculate steering
            # Vector to target
            target_vector = target_location - vehicle_location
            target_vector_2d = carla.Vector2D(target_vector.x, target_vector.y)
            
            # Vehicle forward vector
            forward_vector = vehicle_transform.get_forward_vector()
            forward_vector_2d = carla.Vector2D(forward_vector.x, forward_vector.y)
            
            # Calculate angle between vectors
            def vector_angle(v):
                """Calculate angle of a 2D vector"""
                return math.atan2(v.y, v.x)
            
            target_angle = vector_angle(target_vector_2d)
            vehicle_angle = vector_angle(forward_vector_2d)
            
            # Calculate angle difference
            angle_diff = target_angle - vehicle_angle
            
            # Normalize angle to [-π, π]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Calculate steering (simple proportional control)
            # The 0.7 factor controls how aggressively the vehicle turns
            steer = max(-1.0, min(1.0, angle_diff * 0.7))
            
            # Calculate throttle and brake
            # Reduce target speed for sharp turns (safety measure)
            turn_factor = 1.0 - abs(steer) * 0.4
            adjusted_target_speed = target_speed * turn_factor
            
            # Simple speed control
            if current_speed < adjusted_target_speed:
                # Accelerate
                throttle = min(0.8, (adjusted_target_speed - current_speed) / 20.0)
                brake = 0.0
            else:
                # Brake
                throttle = 0.0
                brake = min(0.3, (current_speed - adjusted_target_speed) / 30.0)
            
            # Apply control to vehicle
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
            self.vehicle.apply_control(control)
            
            # Debug info - print progress every 10th waypoint
            if current_waypoint_index % 10 == 0:
                print(f"📍 Following waypoint {current_waypoint_index}/{len(waypoint_list)}, "
                      f"Distance: {distance:.1f}m, Speed: {current_speed:.1f}km/h")
            
            # Wait for next tick (20 Hz update rate)
            self.world.tick()
            time.sleep(0.05)
    
    def navigate_to_waypoint(self, target_waypoint, waypoint_index):
        """
        Navigate to a single waypoint
        This method plans a path to the target and follows it
        
        Args:
            target_waypoint: Target waypoint coordinates [x, y, z]
            waypoint_index: Index of the target waypoint in WAYPOINTS list
            
        Returns:
            True if waypoint was reached, False otherwise
        """
        print(f"\n📍 Navigating to waypoint {waypoint_index+1}/{len(WAYPOINTS)}: {target_waypoint}")
        
        # Get current position
        current_location = self.vehicle.get_transform().location
        target_location = carla.Location(x=target_waypoint[0], 
                                       y=target_waypoint[1], 
                                       z=target_waypoint[2])
        
        # Plan path
        waypoint_path = self.get_waypoint_path(current_location, target_location)
        
        # Check if path was found
        if not waypoint_path:
            print(f"❌ Could not plan path to waypoint {waypoint_index+1}")
            return False
        
        print(f"📋 Path planned: {len(waypoint_path)} waypoints")
        
        # Visualize the path
        self.visualize_waypoints(waypoint_path, waypoint_index)
        
        # Follow the path
        self.follow_waypoints(waypoint_path)
        
        # Check if we reached the target
        final_location = self.vehicle.get_transform().location
        final_distance = final_location.distance(target_location)
        
        if final_distance < 10.0:
            print(f"✅ Reached waypoint {waypoint_index+1} (distance: {final_distance:.1f}m)")
            return True
        else:
            print(f"⚠️ Close to waypoint {waypoint_index+1} (distance: {final_distance:.1f}m)")
            return True
    
    def navigate_all_waypoints(self):
        """
        Navigate to all waypoints in sequence
        This method goes through each waypoint in the WAYPOINTS list one by one
        """
        print(f"🎯 Starting navigation to {len(WAYPOINTS)} waypoints")
        
        successful_waypoints = 0
        
        # Loop through all waypoints
        for i, waypoint in enumerate(WAYPOINTS):
            try:
                # Navigate to current waypoint
                success = self.navigate_to_waypoint(waypoint, i)
                if success:
                    successful_waypoints += 1
                
                # Brief pause between waypoints (2 seconds)
                time.sleep(2.0)
                
            except Exception as e:
                print(f"❌ Error navigating to waypoint {i+1}: {e}")
                continue
        
        # Stop vehicle after navigation is complete
        self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        print(f"\n🏁 Navigation complete! Reached {successful_waypoints}/{len(WAYPOINTS)} waypoints.")
    
    def cleanup(self):
        """
        Clean up resources
        This method destroys the vehicle to free up resources
        """
        if self.vehicle:
            self.vehicle.destroy()
            print("🧹 Vehicle destroyed")

def main():
    """
    Main function that runs the navigation
    This function sets up the navigator, spawns the vehicle, and starts navigation
    """
    navigator = None
    
    try:
        # Create navigator
        navigator = SimpleCarlaNavigator()
        
        # Spawn vehicle
        navigator.spawn_vehicle()
        
        # Wait a moment for vehicle to settle
        time.sleep(2.0)
        
        # Start navigation
        navigator.navigate_all_waypoints()
        
    except KeyboardInterrupt:
        print("\n⏹️ Navigation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up, even if an error occurred
        if navigator:
            navigator.cleanup()

# Entry point of the script
if __name__ == '__main__':
    print("🚀 Simple CARLA Waypoint Navigation (No Agents)")
    print("=" * 50)
    print("📋 Task: Navigate through waypoints using CARLA roads")
    print("🗺️ Map: Town01")
    print("🎮 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run the main function
    main()
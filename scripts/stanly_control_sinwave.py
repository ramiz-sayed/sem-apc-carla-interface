#!/usr/bin/env python3

import carla
import math
import time
import numpy as np


def get_ego_vehicle(world, role_name="ego_vehicle"):
    """Search for the ego vehicle by its role name."""
    for _ in range(10):
        for actor in world.get_actors():
            if actor.type_id.startswith('vehicle.') and actor.attributes.get('role_name') == role_name:
                return actor
        time.sleep(1)
    return None


def get_speed(vehicle):
    """Return the magnitude of the vehicle's velocity vector (m/s)."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def get_pose(vehicle):
    """Return the (x, y, yaw) of the vehicle in global coordinates."""
    tf = vehicle.get_transform()
    return tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)


def stanley_control(x, y, yaw, speed, path_points, gain=1.0):
    """
    Stanley Controller for steering angle calculation.

    - x, y: current vehicle position
    - yaw: current heading (radians)
    - speed: current forward speed (m/s)
    - path_points: list of (x, y) coordinates representing the path
    - gain: Stanley control gain (default=1.0)
    """
    # 1. Find closest path point
    nearest_idx = np.argmin([math.hypot(px - x, py - y) for px, py in path_points])
    target_x, target_y = path_points[nearest_idx]

    # 2. Heading to the next point
    path_heading = math.atan2(target_y - y, target_x - x)

    # 3. Heading error (angle between path and vehicle heading)
    heading_error = math.atan2(math.sin(path_heading - yaw), math.cos(path_heading - yaw))

    # 4. Cross-track error (distance to the path)
    cross_track_error = math.hypot(target_x - x, target_y - y)

    # 5. Stanley control law: steer_correction = atan(k * e / v)
    steer_correction = math.atan2(gain * cross_track_error, speed + 1e-3)

    # 6. Final steering command (bounded later)
    return heading_error + steer_correction


def reset_vehicle(vehicle, transform):
    """Teleport the vehicle back to original transform and reset its motion."""
    vehicle.set_transform(transform)  # Reset position and heading
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))  # Reset velocity
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))  # Reset rotation
    print("🔁 Ego vehicle reset to original location.")


def main():
    # === Connect to CARLA ===
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # === Get Ego Vehicle ===
    ego_vehicle = get_ego_vehicle(world)
    if not ego_vehicle:
        print("❌ Ego vehicle not found.")
        return

    # Save original position for reset later
    original_transform = ego_vehicle.get_transform()

    # === Sine Path Generation ===
    start_x = original_transform.location.x
    sine_amplitude = 10        # amplitude of sine wave in meters
    sine_frequency = 0.1       # frequency of sine wave (rad/meter)
    path_length = 10           # total path length in meters
    num_points = 20            # number of path points

    path = [(x, sine_amplitude * math.sin(sine_frequency * x))
            for x in np.linspace(start_x, start_x + path_length, num_points)]

    # === Path Following Loop ===
    try:
        for _ in range(50):  # Run control loop for 50 iterations
            x, y, yaw = get_pose(ego_vehicle)
            speed = get_speed(ego_vehicle)

            steer = stanley_control(x, y, yaw, speed, path, gain=1.0)
            steer = max(-1.0, min(1.0, steer))  # clamp steer to [-1, 1]
            print(f"Pos: x={x:.2f}, y={y:.2f}, speed={speed:.2f} m/s, steer={steer:.2f}")


            control = carla.VehicleControl(throttle=0.6, steer=steer)
            ego_vehicle.apply_control(control)
            time.sleep(0.05)  # 50 ms loop delay

        print("✅ Finished path following.")

    finally:
        # Stop the car
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1)

        # Reset vehicle back to starting position
        reset_vehicle(ego_vehicle, original_transform)


if __name__ == '__main__':
    main()

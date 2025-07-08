#!/usr/bin/env python3

import carla
import math
import time
import numpy as np

# === Utility Functions ===

def get_speed(vehicle):
    """Returns speed in m/s from velocity vector."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def get_pose(vehicle):
    """Returns vehicle position (x, y) and heading (yaw in radians)."""
    tf = vehicle.get_transform()
    return tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)

def stanley_control(x, y, yaw, speed, path_points, gain=1.5, max_cte=3.0):
    """
    Stanley controller for steering.
    gain: control gain (higher = more aggressive turning)
    max_cte: maximum cross-track error clamp (in meters)
    """
    # Find nearest point on path
    nearest_idx = np.argmin([math.hypot(px - x, py - y) for px, py in path_points])
    lookahead_idx = min(nearest_idx + 5, len(path_points) - 1)  # 5-point lookahead
    target_x, target_y = path_points[lookahead_idx]

    # Calculate heading error
    path_heading = math.atan2(target_y - y, target_x - x)
    heading_error = math.atan2(math.sin(path_heading - yaw), math.cos(path_heading - yaw))

    # Cross-track error (perpendicular distance)
    dx = target_x - x
    dy = target_y - y
    heading_vector = np.array([math.cos(yaw), math.sin(yaw)])
    error_vector = np.array([dx, dy])
    cross_track_error = np.cross(heading_vector, error_vector)
    cross_track_error = max(-max_cte, min(max_cte, cross_track_error))  # clamp

    # Stanley correction term
    steer_correction = math.atan2(gain * cross_track_error, speed + 1e-3) if speed > 0.1 else 0.0
    return heading_error + steer_correction

# === Main Function ===

def main():
    # === CARLA Connection ===
    client = carla.Client("localhost", 2000)     # host:port
    client.set_timeout(5.0)                      # 5 second timeout
    world = client.get_world()
    debug = world.debug                          # for drawing debug markers
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # === Spawn Vehicle ===
    npc_bp = blueprint_library.find('vehicle.ford.mustang')  # vehicle type
    npc_bp.set_attribute('role_name', 'npc_vehicle')
    spawn_point = spawn_points[96]                           # choose spawn point
    npc_vehicle = world.spawn_actor(npc_bp, spawn_point)
    print(f"🚙 Spawned Ford Mustang: ID={npc_vehicle.id}, Type={npc_vehicle.type_id}")

    # === Sine Path Generation ===
    start_y = spawn_point.location.y
    base_x = spawn_point.location.x
    sine_amplitude = 2              # vertical curve height (meters)
    sine_frequency = 0.2            # wave frequency (tightness of sine)
    path_length = 200               # total forward path in Y (meters)
    num_points = 400                # resolution of path points

    path = [(base_x + sine_amplitude * math.sin(sine_frequency * y), y)
            for y in np.linspace(start_y, start_y + path_length, num_points)]

    # === PID Controller Parameters ===
    target_speed = 5.0              # target velocity (m/s)
    Kp = 0.3                        # Proportional gain
    Ki = 0.05                       # Integral gain
    Kd = 0.1                        # Derivative gain
    dt = 0.04                       # control loop time step (seconds)

    # === Initialize PID State ===
    integral = 0.0
    prev_error = 0.0

    # === Steering Filter ===
    previous_steer = 0.0
    steer_smoothing = 0.6           # smoothing factor (0 = no smoothing, 1 = full smoothing)

    try:
        for step in range(600):  # run for 600 steps (~24 seconds)
            # === Vehicle State ===
            x, y, yaw = get_pose(npc_vehicle)
            speed = get_speed(npc_vehicle)

            # === Steering Control ===
            steer = stanley_control(x, y, yaw, speed, path)
            steer = steer_smoothing * previous_steer + (1 - steer_smoothing) * steer  # low-pass filter
            steer = max(-0.6, min(0.6, steer))  # steering range clamp [-0.6, 0.6]
            previous_steer = steer

            # === Throttle Control (PID) ===
            error = target_speed - speed
            integral += error * dt
            derivative = (error - prev_error) / dt
            prev_error = error

            throttle = Kp * error + Ki * integral + Kd * derivative
            throttle = max(0.0, min(1.0, throttle))  # throttle range clamp [0.0, 1.0]

            # === Apply Control ===
            control = carla.VehicleControl(throttle=throttle, steer=steer)
            npc_vehicle.apply_control(control)

            # === Debug Info ===
            print(f"Step {step:03d} | "
                  f"Pos: ({x:.1f}, {y:.1f}) | "
                  f"Speed: {speed:.2f} m/s | "
                  f"Target: {target_speed:.2f} | "
                  f"Error: {error:.2f} | "
                  f"Throttle: {throttle:.2f} | "
                  f"Steer: {steer:.2f}")

            # === Draw Full Path Points (optional) ===
            for px, py in path:
             debug.draw_point(carla.Location(x=px, y=py, z=spawn_point.location.z + 0.5),
                     size=0.05, color=carla.Color(0, 255, 0), life_time=0.2)

            debug.draw_point(carla.Location(x=x, y=y, z=spawn_point.location.z + 1.0),
                 size=0.08, color=carla.Color(255, 0, 0), life_time=10.0)


            # === Draw Current Position (red dot) ===
            debug.draw_point(carla.Location(x=x, y=y, z=spawn_point.location.z + 1.0),
                             size=0.08, color=carla.Color(255, 0, 0), life_time=0.3)

            time.sleep(dt)

        print("✅ Path following completed.")

    finally:
        # === Stop and Cleanup ===
        npc_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1)
        print("🗑️ Destroying vehicle...")
        npc_vehicle.destroy()
        print("✅ Vehicle destroyed.")


# === Entry Point ===
if __name__ == '__main__':
    main()

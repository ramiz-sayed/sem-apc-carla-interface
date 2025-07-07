# stanley_straight_line.py

import carla
import math
import time
import numpy as np

def get_ego_vehicle(world):
    for _ in range(10):
        actors = world.get_actors()
        for actor in actors:
            if actor.type_id.startswith('vehicle.') and actor.attributes.get('role_name') == 'ego_vehicle':
                return actor
        time.sleep(1)
    return None

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def get_pose(vehicle):
    tf = vehicle.get_transform()
    return tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)

def stanley_control(x, y, yaw, speed, path, k=1.0):
    nearest = np.argmin([math.hypot(px - x, py - y) for px, py in path])
    tx, ty = path[nearest]
    heading = math.atan2(ty - y, tx - x)
    heading_error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    cross_track = math.hypot(tx - x, ty - y)
    steer_correction = math.atan2(k * cross_track, speed + 1e-3)
    return heading_error + steer_correction

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()
vehicle = get_ego_vehicle(world)
if not vehicle:
    print("Ego vehicle not found.")
    exit()

start_x = vehicle.get_transform().location.x
start_y = vehicle.get_transform().location.y
path = [(start_x + i, start_y) for i in range(100)]

try:
    for _ in range(500):
        x, y, yaw = get_pose(vehicle)
        speed = get_speed(vehicle)
        steer = stanley_control(x, y, yaw, speed, path)
        steer = max(-1.0, min(1.0, steer))
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=steer))
        time.sleep(0.05)
finally:
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    print("✅ Done moving in a straight line.")

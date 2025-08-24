#!/usr/bin/env python3
import carla
import time
import math

# ========================= CONFIG ============================
WAYPOINTS = [
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

START_POSE = {
    'x': 280.363739,
    'y': -129.306351,
    'z': 0.101746,
    'roll': 0.0,
    'pitch': 0.0,
    'yaw': 180.0
}

# =============================================================

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.load_world("Town01")
    blueprint_library = world.get_blueprint_library()

    # Choose vehicle blueprint (Tesla Model 3)
    vehicle_bp = blueprint_library.filter("model3")[0]

    # Convert start pose to Carla transform
    start_transform = carla.Transform(
        carla.Location(x=START_POSE['x'], y=START_POSE['y'], z=START_POSE['z'] + 2.0),
        carla.Rotation(pitch=START_POSE['pitch'], yaw=START_POSE['yaw'], roll=START_POSE['roll'])
    )

    # Spawn vehicle
    vehicle = world.spawn_actor(vehicle_bp, start_transform)
    vehicle.set_autopilot(False)

    print("Spawned vehicle at:", start_transform)

    # Draw all waypoints in the world
    for i, (x, y, z) in enumerate(WAYPOINTS):
        loc = carla.Location(x=x, y=y, z=z+1.0)
        world.debug.draw_string(loc, str(i+1), draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=0)
        world.debug.draw_point(loc, size=0.1, color=carla.Color(r=0, g=255, b=0), life_time=0)

        if i > 0:
            prev = carla.Location(x=WAYPOINTS[i-1][0], y=WAYPOINTS[i-1][1], z=WAYPOINTS[i-1][2]+1.0)
            world.debug.draw_line(prev, loc, thickness=0.2,
                                  color=carla.Color(r=0, g=0, b=255), life_time=0)

    # Simple sequential navigation (no avoidance yet)
    for wp in WAYPOINTS:
        target_location = carla.Location(x=wp[0], y=wp[1], z=wp[2]+2.0)
        vehicle.set_transform(carla.Transform(target_location,
                                              carla.Rotation(yaw=START_POSE['yaw'])))
        print(f"Moved to waypoint: {wp}")
        time.sleep(1.5)

    print("Finished visiting all waypoints.")
    time.sleep(5)

    # Cleanup
    vehicle.destroy()

if __name__ == "__main__":
    main()

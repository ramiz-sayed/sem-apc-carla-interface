import carla
import time
import math

# PID coefficients
Kp, Ki, Kd = 0.4, 0.05, 0.1
target_speed = 10.0  # m/s (~36 km/h)
target_distance = 70.0  # meters

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Find ego vehicle
ego_vehicle = None
for _ in range(10):
    actors = world.get_actors()
    ego_list = [a for a in actors if a.type_id.startswith('vehicle.') and a.attributes.get('role_name') == 'ego_vehicle']
    if ego_list:
        ego_vehicle = ego_list[0]
        break
    time.sleep(1)

if ego_vehicle is None:
    print("Ego vehicle not found. Exiting.")
    exit()

print(f"Ego vehicle found: ID={ego_vehicle.id}, Type={ego_vehicle.type_id}")

# Spawn and attach IMU sensor to ego vehicle
blueprint_library = world.get_blueprint_library()
imu_bp = blueprint_library.find('sensor.other.imu')
imu_transform = carla.Transform(carla.Location(x=0.0, z=0.0))  # adjust if needed

imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle)

# Variables to hold IMU data
acceleration_x = 0.0

# Velocity and distance estimation variables
velocity = 0.0
distance_traveled = 0.0
last_time = time.time()

# Callback to update acceleration
def imu_callback(imu_data):
    global acceleration_x
    # Assume forward acceleration is along X-axis in sensor frame
    acceleration_x = imu_data.accelerometer.x

imu_sensor.listen(imu_callback)

prev_error = 0.0
integral = 0.0

try:
    while distance_traveled < target_distance:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Integrate acceleration to get velocity (simple Euler integration)
        velocity += acceleration_x * dt

        # Clamp velocity to non-negative (vehicle doesn't move backward in this control)
        if velocity < 0:
            velocity = 0

        # Integrate velocity to get distance
        distance_traveled += velocity * dt

        error = target_speed - velocity
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        output = Kp * error + Ki * integral + Kd * derivative

        throttle = max(0.0, min(output, 1.0))
        brake = 0.0 if output > 0 else min(-output, 1.0)

        ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))

        print(f"Distance: {distance_traveled:.2f} m | Velocity: {velocity:.2f} m/s | Throttle: {throttle:.2f} | Brake: {brake:.2f}")

        prev_error = error
        time.sleep(0.05)

finally:
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    imu_sensor.stop()
    imu_sensor.destroy()
    print(f"Target distance {target_distance}m reached. Vehicle stopped.")

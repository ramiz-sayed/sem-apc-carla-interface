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

prev_error = 0.0
integral = 0.0
last_time = time.time()
distance_traveled = 0.0

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

try:
    while distance_traveled < target_distance:
        #time.time() real time world
        #world.get_snapshot().timestamp.elapsed_seconds simulation time 
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        current_speed = get_speed(ego_vehicle)
        distance_traveled += current_speed * dt

        error = target_speed - current_speed
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        output = Kp * error + Ki * integral + Kd * derivative

        throttle = max(0.0, min(output, 1.0))
        brake = 0.0 if output > 0 else min(-output, 1.0)

        ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))

        print(f"Distance: {distance_traveled:.2f} m | Speed: {current_speed:.2f} m/s | Throttle: {throttle:.2f} | Brake: {brake:.2f}")

        prev_error = error
        time.sleep(0.05)

finally:
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    print(f"Target distance {target_distance}m reached. Vehicle stopped.")
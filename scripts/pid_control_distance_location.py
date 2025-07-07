import carla
import time
import math

# PID coefficients for distance control
Kp = 0.8
Ki = 0.01
Kd = 0.1

target_distance = 70.0  # meters

# Connect to Carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Retry loop to find ego vehicle
ego_vehicle = None
for attempt in range(10):
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith('vehicle.') and actor.attributes.get('role_name') == 'ego_vehicle':
            ego_vehicle = actor
            break
    if ego_vehicle:
        break
    print(f"⏳ Attempt {attempt+1}/10: Ego vehicle not found. Retrying...")
    time.sleep(1)

if not ego_vehicle:
    print("❌ Ego vehicle not found after retries.")
    exit()

print(f"✅ Found ego vehicle: ID={ego_vehicle.id}, Type={ego_vehicle.type_id}")

# Record start location
start_location = ego_vehicle.get_location()

# PID control state
prev_error = 0.0
integral = 0.0
last_time = time.time()

def get_distance(start, current):
    dx = current.x - start.x
    dy = current.y - start.y
    dz = current.z - start.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

try:
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        current_location = ego_vehicle.get_location()
        distance_traveled = get_distance(start_location, current_location)
        error = target_distance - distance_traveled

        if error <= 0:
            print("🎯 Target distance reached!")
            break

        # PID calculations
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        output = Kp * error + Ki * integral + Kd * derivative

        # Throttle/brake logic
        throttle = max(0.0, min(output, 1.0))
        brake = 0.0 if output > 0 else min(-output, 1.0)

        ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))
        
        print(f"Distance: {distance_traveled:.2f} m, Error: {error:.2f}, Throttle: {throttle:.2f}, Brake: {brake:.2f}")

        prev_error = error
        time.sleep(0.05)

finally:
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    print("✅ Vehicle stopped safely.")

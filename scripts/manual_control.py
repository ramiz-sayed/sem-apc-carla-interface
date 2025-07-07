import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()


# Retry loop for finding the ego vehicle
ego_vehicle = None
#print("📋 Searching for vehicle with role_name='ego_vehicle'...")

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


#if ego_vehicle is None:
#    print("❌ Ego vehicle not found after retries. Please check your scenario or spawn configuration.")
#    exit()

#print(f"✅ Found ego vehicle: ID={ego_vehicle.id}, Type={ego_vehicle.type_id}")

# Spawn second vehicle (Mustang)
npc_bp = blueprint_library.find('vehicle.ford.mustang')
npc_bp.set_attribute('role_name', 'npc_vehicle')
npc_vehicle = world.spawn_actor(npc_bp, spawn_points[96])
print(f"🚙 Spawned second vehicle: ID={npc_vehicle.id}, Type={npc_vehicle.type_id}")

# Control the vehicle
try:
    print("↩️ Turning right...")
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=-0.5 ,reverse=True))
    time.sleep(5)

    print("➡️ Moving backward...")
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0 , reverse=True))
    time.sleep(3)

    print("🛑 Stopping...")
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    time.sleep(2)

    print("↩️ Turning left...")
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.5))
    time.sleep(2)

    print("➡️ Moving forward...")
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0 , reverse=False))
    time.sleep(3)

    print("⏹️ Straight and stop...")
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

finally:
    print("✅ Finished manual control.")
    print("🗑️ Destroying npc vehicle...")
    npc_vehicle.destroy()
    print("✅ Vehicle destroyed.")
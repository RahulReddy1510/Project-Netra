import blenderproc as bproc
import numpy as np
import os
import random

def setup_scene():
    bproc.init()
    # Create a basic ground plane (Construction Site Floor)
    ground = bproc.object.create_primitive(shape='PLANE', scale=[50, 50, 1])
    ground_mat = bproc.material.create("GroundMat")
    # In a real scenario, we'd load a concrete texture here
    ground_mat.set_principled_shader_value("Base Color", [0.3, 0.3, 0.3, 1])
    ground.replace_materials(ground_mat)
    
    # Add some walls/perimeter
    wall = bproc.object.create_primitive(shape='CUBE', location=[0, 20, 5], scale=[20, 0.5, 5])

def create_lights():
    # Randomize between Noon (Hard Shadows) and Overcast (Soft Light)
    scenario = random.choice(['sunny', 'overcast'])
    
    if scenario == 'sunny':
        # Sun Light
        light = bproc.types.Light()
        light.set_type("SUN")
        light.set_location([0, 0, 50])
        light.set_energy(5)
        # Random rotation for time of day
        light.set_rotation_euler(np.random.uniform([0,0,0], [0.5, 0.5, 0]))
        
    else:
        # Overcast (Area Light dome)
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_location([0, 0, 20])
        light.set_energy(500)
        light.set_scale([20, 20, 1])

def load_worker_model(asset_path):
    # This is a placeholder. In reality, you download a rigged worker .obj/.fbx
    # For now, we use a basic shape to represent a worker
    worker = bproc.object.create_primitive(shape='CUBE', scale=[0.3, 0.3, 0.9])
    worker.set_name("Worker")
    
    # Apply PPE Material (Yellow Vest)
    vest_mat = bproc.material.create("VestMat")
    vest_mat.set_principled_shader_value("Base Color", [1.0, 1.0, 0.0, 1]) # Neon Yellow
    vest_mat.set_principled_shader_value("Emission", [0.5, 0.5, 0.0, 1])   # Reflective
    worker.replace_materials(vest_mat)
    
    return worker

def scenario_fall_event(worker):
    """ Simulate a worker falling from height """
    # Position in air
    worker.set_location([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(3, 8)])
    # Rotation (Tumbling)
    worker.set_rotation_euler(np.random.uniform([0,0,0], [3.14, 3.14, 3.14]))

def scenario_intrusion(worker):
    """ Simulate a worker entering a Red Zone """
    # Red Zone is near the 'wall' at y=20
    worker.set_location([random.uniform(-5, 5), 18, 1])
    worker.set_rotation_euler([0, 0, random.uniform(0, 3.14)])

def generate_digital_twin_data(output_dir="digital_twin/output", num_samples=10):
    setup_scene()
    
    for i in range(num_samples):
        bproc.utility.reset_keyframes()
        
        create_lights()
        
        # Spawn Worker
        worker = load_worker_model(None)
        
        # Pick a Rare Event
        event_type = random.choice(['fall', 'intrusion', 'normal'])
        if event_type == 'fall':
            scenario_fall_event(worker)
        elif event_type == 'intrusion':
            scenario_intrusion(worker)
        else:
            worker.set_location([random.uniform(-10, 10), random.uniform(-10, 10), 0.9])
            
        # Camera Setup
        cam_pose = bproc.math.build_transformation_mat(
            [0, -15, 5], # Location
            [1.2, 0, 0]  # Rotation (looking slightly down)
        )
        bproc.camera.add_camera_pose(cam_pose)
        
        # Render
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(output_dir, data)
        # Also write COCO annotations (bounding boxes)
        bproc.writer.write_coco_annotations(os.path.join(output_dir, "coco_data"), 
                                            instance_segmentation_data=data["instance_segmaps"],
                                            instance_attribute_maps=data["instance_attribute_maps"],
                                            colors=data["colors"])
        
        print(f"Generated Sample {i+1}/{num_samples} - Type: {event_type}")
        
        # Cleanup for next loop
        worker.delete()

if __name__ == "__main__":
    generate_digital_twin_data()

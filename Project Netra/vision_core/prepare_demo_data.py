import os
import urllib.request
import cv2
import numpy as np
import yaml

# Configuration
DATA_ROOT = os.path.join("vision_core", "data", "demo_dataset")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
LABELS_DIR = os.path.join(DATA_ROOT, "labels")
SPLITS = ["train", "val"]

# Sample Images (Public Domain / Creative Commons valid for demo)
SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Construction_worker_in_safety_vest_and_hard_hat.jpg/640px-Construction_worker_in_safety_vest_and_hard_hat.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Construction_Worker_at_Work.jpg/640px-Construction_Worker_at_Work.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Construction_workers_building_arch.jpg/640px-Construction_workers_building_arch.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Construction_Worker.jpg/640px-Construction_Worker.jpg"
]

def setup_directories():
    for split in SPLITS:
        os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)
    print(f"Created directories in {DATA_ROOT}")

def download_images():
    print("Generating synthetic demo images (Noise)...")
    for i in range(4):
        # Save to train (mostly) and val (last one)
        split = "train" if i < 3 else "val"
        filename = f"demo_img_{i}.jpg"
        filepath = os.path.join(IMAGES_DIR, split, filename)
        
        # Generate random noise image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Draw some rectangles to mimic objects
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2) # Fake 'Vest'
        cv2.rectangle(img, (300, 300), (400, 400), (0, 0, 255), 2) # Fake 'Helmet'
        
        cv2.imwrite(filepath, img)
        print(f"  Created {filename} -> {split}")
        
        # Create a Dummy Label for this image
        # Class 0 (Helmet), Center X, Center Y, Width, Height
        label_path = os.path.join(LABELS_DIR, split, f"demo_img_{i}.txt")
        with open(label_path, "w") as f:
            # 0: Helmet, 1: Vest. YOLO format normalized (0-1)
            # Box at 350,350 size 100x100 -> center 350,350 -> norm 350/640=0.54
            f.write("0 0.54 0.54 0.15 0.15\n") 
            f.write("1 0.23 0.23 0.15 0.15\n")

def create_yaml_config():
    yaml_path = os.path.join("vision_core", "data", "demo.yaml")
    
    # Absolute paths are safer for YOLO sometimes, but relative works if cwd is correct
    # We will use relative to the execution path (Project Netra root)
    data = {
        'path': os.path.abspath(DATA_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
             0: 'Helmet',
             1: 'Vest',
             2: 'Person',
             3: 'No-Helmet',
             4: 'No-Vest'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created config: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    setup_directories()
    download_images()
    create_yaml_config()
    print("\n✅ Demo Data Ready.")
    print("You can now run 'train_yolo.py' (it will default to demo.yaml if data.yaml data implies empty dirs).")

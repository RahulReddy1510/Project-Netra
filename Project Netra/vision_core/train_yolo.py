from ultralytics import YOLO
import torch
import os

def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return 0 # Device 0
    else:
        print("⚠️ No GPU detected. Training will be extremely slow on CPU.")
        print("Recommendation: Use an NVIDIA Jetson Orin or Cloud GPU instance.")
        return 'cpu'

def train_netra_core():
    print("="*60)
    print("🚀 Project Netra: Vision Core Training Initialization")
    print("="*60)

    # 1. Device Setup
    device = check_gpu()

    # 2. Model Initialization
    # We use YOLOv8-m (Medium) as per the plan for the balance of speed/accuracy
    print("\n[1/4] Loading YOLOv8-Medium Architecture...")
    model = YOLO('yolov8m.pt') 

    # 3. Training Configuration
    parser.add_argument('--evolve', action='store_true', help='Enable Hyperparameter Evolution (Genetic Algorithm)')
    args, unknown = parser.parse_known_args()

    if args.config:
        dataset_yaml = args.config
    elif os.path.exists(os.path.join("vision_core", "data", "demo.yaml")):
        print("ℹ️  Using DEMO dataset for verification.")
        dataset_yaml = os.path.join("vision_core", "data", "demo.yaml")
    else:
        dataset_yaml = os.path.join("data", "data.yaml")
    
    print(f"\n[2/4] Configuring Training Parameters for {dataset_yaml}...")
    
    # Base Args
    training_args = {
        'data': dataset_yaml,
        'epochs': 100,           
        'imgsz': 640,            
        'batch': 16,             
        'device': device,
        'workers': 8,
        'project': 'Netra_Vision_Core',
        'name': 'v1_meta_enhanced',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',    
        'lr0': 0.001,            
        'cos_lr': True,          
        'hsv_h': 0.015,
        'hsv_s': 0.7, 
        'hsv_v': 0.4,    
        'degrees': 10.0, 
        'translate': 0.1,
        'scale': 0.5,    
        'mosaic': 1.0,   
        'mixup': 0.1,    
    }

    if args.evolve:
        print("\n🧬 MODE: HYPERPARAMETER EVOLUTION ENABLED")
        print("   This will simulate 300 generations of training to find the perfect hyperparameters.")
        print("   Warning: This can take days on a single GPU.")
        # Evolve overides standard train
        model.tune(data=dataset_yaml, epochs=10, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)
        print("✅ Evolution Complete. Best params saved to runs/tune")
        return

    print("\n[3/4] Ready to Launch.")

    print("Ensure you have downloaded the following datasets into 'vision_core/datasets':")
    print("  - HardHat-Vest Dataset")
    print("  - Pictor-PPE Dataset")
    
    user_input = input("\nStart Training? (y/n): ")
    if user_input.lower() == 'y':
        print("\n[4/4] Starting Training Loop...")
        try:
            results = model.train(**training_args)
            print("\n✅ Training Complete. Best model saved in 'Netra_Vision_Core/v1_meta_enhanced/weights/best.pt'")
        except Exception as e:
            print(f"\n❌ Training Failed: {e}")
            print("Tip: Check if 'data.yaml' paths exist and contain images.")

if __name__ == "__main__":
    train_netra_core()

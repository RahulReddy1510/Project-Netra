from ultralytics import YOLO
import argparse
import os

def export_to_onnx(model_path, output_name):
    print(f"🔄 Loading Model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("🚀 Starting export to ONNX...")
    # opset=12 is widely supported by TensorRT
    # dynamic=True allows variable batch sizes (good for multi-stream)
    path = model.export(format='onnx', dynamic=True, opset=12)
    
    print(f"✅ Export Successful!")
    print(f"💾 ONNX Model saved at: {path}")
    print("\nNext Step: Run 'tensorrt_compile.py' on the Jetson Orin to convert this ONNX to a .engine file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the 'best.pt' if it exists, otherwise the standard yolov8m.pt
    default_model = "vision_core/Netra_Vision_Core/v1_meta_enhanced/weights/best.pt"
    if not os.path.exists(default_model):
        default_model = "yolov8m.pt"
        
    parser.add_argument('--weights', type=str, default=default_model, help='Path to .pt model')
    args = parser.parse_args()
    
    export_to_onnx(args.weights, "netra_core_v1")

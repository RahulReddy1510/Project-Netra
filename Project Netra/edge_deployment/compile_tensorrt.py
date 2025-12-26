import os

# NOTE: This script is intended to be run on the NVIDIA Jetson Orin.
# You must have TensorRT installed (part of JetPack).

def compile_engine(onnx_path, engine_output_path):
    print("⚙️  Initializing TensorRT Compilation...")
    
    # We use 'trtexec', the standard command-line tool for TensorRT
    # Flags explained:
    # --onnx: Input model
    # --saveEngine: Output engine file
    # --fp16: Enable Float16 precision (huge speedup on Orin)
    # --int8: Enable INT8 quantization (requires calibration cache, omitted for simplicity here)
    # --workspace=4096: Allocate memory for builder
    
    cmd = (
        f"trtexec --onnx={onnx_path} "
        f"--saveEngine={engine_output_path} "
        f"--fp16 "
        "--workspace=4096"
    )
    
    print(f"RUNNING: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    onnx_file = "../vision_core/yolov8m.onnx" # Example path
    engine_file = "netra_core_v1.engine"
    
    if not os.path.exists(onnx_file):
        print(f"⚠️  ONNX file not found at {onnx_file}. Run export_onnx.py first.")
    else:
        compile_engine(onnx_file, engine_file)

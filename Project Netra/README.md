# Project Netra: AI Safety Monitoring System

**Version**: 1.0 (Alpha)
**Target Hardware**: NVIDIA Jetson Orin Nano / AGX
**Architecture**: YOLOv8 + TensorRT + PyQt6

## 🏗️ Phase 1: Vision Core
The central detection engine.
1.  **Dependencies**: `pip install ultralytics albumentations`
2.  **Augmentation**: Run `vision_core/augmentation_pipeline.py` to test hazardous effects.
3.  **Training**:
    *   Download Datasets (See `vision_core/DATASET_SETUP.md`).
    *   Run `python vision_core/train_yolo.py`.
    *   *Note: Includes a demo mode with synthetic data for testing.*

## 🏭 Phase 2: Digital Twin
Synthetic data factory.
1.  **Setup**: Install Blender & `pip install blenderproc`.
2.  **Generate**: `blenderproc run digital_twin/generate_scenarios.py`.
3.  **Output**: Creates photorealistic "Falling Worker" & "Intrusion" events.

## ⚡ Phase 3: Edge Deployment
Optimization for Jetson.
1.  **Export ONNX**: `python edge_deployment/export_onnx.py`
2.  **Compile TensorRT**: `python edge_deployment/compile_tensorrt.py` (Run on Jetson).
3.  **Inference**: `python edge_deployment/inference_loop.py` (Standalone logic test).

## 🖥️ Phase 4: Netra Command Interface
The Operator Dashboard.
1.  **Launch**: `python netra_command/main.py`
2.  **Features**:
    *   Real-time video feed with Bounding Boxes & Danger Zones.
    *   Event Log for "PPE Violations" and "Intrusions".
    *   System Health Monitor.

## ⚠ Troubleshooting
*   **No Camera?**: The system will crash or hang. Ensure a webcam is connected or modify `edge_deployment/inference_loop.py` to use a video file path.
*   **Slow FPS?**: Running YOLOv8m on a CPU is slow (~3-5 FPS). Use an NVIDIA GPU or Jetson for real speeds (30+ FPS).

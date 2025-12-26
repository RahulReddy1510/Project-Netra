# Digital Twin Factory Setup

## 1. Prerequisites
*   **Blender**: Install Blender 3.0+ from [blender.org](https://www.blender.org/).
*   **BlenderProc**: A Python pipeline for photorealistic rendering.

## 2. Installation
```bash
pip install blenderproc
```

## 3. Running the Generator
The generation script `generate_scenarios.py` uses BlenderProc to simulate:
*   Falling Workers (Motion Blur & Pose)
*   Intrusions (Red Zone Proximity)
*   Lighting Shifts (Noon vs Overcast)

Run it with:
```bash
blenderproc run digital_twin/generate_scenarios.py
```

## 4. Custom Assets
To make this realistic, you need 3D models.
*   Download "Worker" and "Excavator" models (OBJ/FBX) from Sketchfab/TurboSquid.
*   Update `load_worker_model` in `generate_scenarios.py` to `bproc.loader.load_obj("path/to/worker.obj")`.

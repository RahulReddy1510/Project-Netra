# Project Netra Dataset Setup

To train the production-grade model, you need to download real datasets. The `demo_dataset` is only for testing the code pipeline.

## 1. Directory Structure
Ensure your folders look like this:
```
Project Netra/
└── vision_core/
    └── datasets/
        ├── hardhat_vest/
        │   ├── train/
        │   ├── valid/
        │   └── data.yaml  <-- Included with dataset usually
        └── pictor_ppe/
            ├── train/
            ├── valid/
            └── ...
```

## 2. Recommended Datasets (Downloads)

### A. HardHat-Vest Dataset (Primary)
*   **Source**: [Kaggle - Hard Hat Workers Dataset](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) or [Roboflow Universe](https://universe.roboflow.com/search?q=hard%20hat%20vest)
*   **Action**: Download the "YOLOv8" format version. Unzip into `vision_core/datasets/hardhat_vest`.

### B. Pictor-PPE (Calibration/Hard Cases)
*   **Source**: [GitHub (Pictor-PPE)](https://github.com/ciber-lab/pictor-ppe) or search on Roboflow.
*   **Action**: Download and unzip into `vision_core/datasets/pictor_ppe`.

## 3. Updating Configuration
Once downloaded:
1.  Open `vision_core/data/data.yaml`.
2.  Update the `path` variable to point to your new dataset folder (e.g., `../datasets/hardhat_vest`).

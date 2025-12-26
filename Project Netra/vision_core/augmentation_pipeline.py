import cv2
import albumentations as A
import os
import random
import numpy as np

class ConstructionAugmentor:
    """
    Custom Augmentation Pipeline for 'Project Netra'.
    Simulates hazardous construction environments:
    - Heavy machinery vibration (Motion Blur)
    - Outdoor weather (Rain, Fog, Shadows)
    - Feature occlusion (Coarse Dropout for scaffolding/beams)
    """

    def __init__(self):
        self.transform = A.Compose([
            # 1. Geometric Warps (Simulate camera lens distortion/angle shifts)
            A.Perspective(scale=(0.05, 0.1), p=0.2),
            
            # 2. Environmental Effects (The "Messiness")
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=1),
                A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1),
            ], p=0.4),

            # 3. Sensor/Camera Defects
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            ], p=0.3),

            # 4. Occlusions (CRITICAL for construction sites)
            # Simulates workers behind bars, scaffolding, or pillars
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=2,
                min_height=8,
                min_width=8,
                fill_value=0, 
                p=0.3
            ),
            
            # 5. Lighting Conditions (Dawn/Dusk/Floodlights)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ToGray(p=0.05) # Test color invariance
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def augment_image(self, image_path, bboxes, class_labels, output_dir=None):
        """
        Applies pipeline to a single image.
        Args:
            image_path (str): Path to input image.
            bboxes (list): List of bounding boxes [x_center, y_center, width, height] normalized.
            class_labels (list): List of class IDs.
            output_dir (str, optional): If provided, saves the result.
        Returns:
            dict: Augmented 'image', 'bboxes', 'class_labels'
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"aug_{filename}")
            cv2.imwrite(output_path, transformed['image'])
            # Note: You would also need to save the new labels in a real pipeline
            
        return transformed

# Demo Usage
if __name__ == "__main__":
    print("Initializing Netra Construction Augmentor...")
    augmentor = ConstructionAugmentor()
    print("Pipeline Ready. Import this class in your data preprocessing script.")

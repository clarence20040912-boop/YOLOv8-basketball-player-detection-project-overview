"""
YOLOv8 Pose Estimation Module
Extracts player skeleton keypoints for action recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# COCO 17 keypoint definitions
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections (for visualization)
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # upper limbs
    (5, 11), (6, 12), (11, 12),                    # torso
    (11, 13), (13, 15), (12, 14), (14, 16)         # lower limbs
]


@dataclass
class PoseResult:
    """Pose estimation result"""
    bbox: Tuple[int, int, int, int]          # person bounding box
    keypoints: np.ndarray                     # keypoint coordinates (17, 3) [x, y, conf]
    confidence: float                         # detection confidence
    
    def get_keypoint(self, name: str) -> Tuple[float, float, float]:
        """Get the specified keypoint (x, y, confidence)"""
        idx = KEYPOINT_NAMES.index(name)
        return tuple(self.keypoints[idx])
    
    def get_angle(self, p1: str, p2: str, p3: str) -> float:
        """
        Compute the angle formed by three keypoints.
        Used for action detection (e.g. arm bend angle).

        Args:
            p1, p2, p3: keypoint names; p2 is the vertex of the angle

        Returns:
            angle in degrees
        """
        a = np.array(self.get_keypoint(p1)[:2])
        b = np.array(self.get_keypoint(p2)[:2])
        c = np.array(self.get_keypoint(p3)[:2])
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        return angle


class PoseEstimator:
    """YOLOv8 Pose Estimator"""
    
    def __init__(self, model_path: str = "yolov8n-pose.pt",
                 confidence_threshold: float = 0.5):
        """
        Initialize the pose estimator.

        Args:
            model_path: YOLOv8-Pose model path
            confidence_threshold: confidence threshold
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"✅ Pose estimator loaded: {model_path}")
    
    def estimate(self, image: np.ndarray) -> List[PoseResult]:
        """
        Run pose estimation on people in the image.

        Args:
            image: input image (BGR, numpy)

        Returns:
            list of pose estimation results
        """
        results = self.model(
            image, 
            conf=self.confidence_threshold,
            verbose=False
        )
        
        poses = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            
            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = tuple(boxes.xyxy[i].cpu().numpy().astype(int))
                conf = float(boxes.conf[i].cpu().numpy())
                kps = keypoints_data[i]  # (17, 3)
                
                poses.append(PoseResult(
                    bbox=bbox,
                    keypoints=kps,
                    confidence=conf
                ))
        
        return poses
    
    def draw_poses(self, image: np.ndarray, 
                   poses: List[PoseResult]) -> np.ndarray:
        """Draw skeleton keypoints"""
        output = image.copy()
        
        for pose in poses:
            kps = pose.keypoints
            
            # draw keypoints
            for j in range(17):
                x, y, conf = kps[j]
                if conf > 0.5:
                    cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # draw skeleton connections
            for (idx1, idx2) in SKELETON_CONNECTIONS:
                x1, y1, c1 = kps[idx1]
                x2, y2, c2 = kps[idx2]
                if c1 > 0.5 and c2 > 0.5:
                    cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)),
                            (255, 0, 0), 2)
        
        return output

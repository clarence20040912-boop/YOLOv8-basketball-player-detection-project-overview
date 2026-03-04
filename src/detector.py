"""
YOLOv8 Basketball Player Detection Module
Detects players, basketballs, hoops, and other targets
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Detection:
    """Detection result dataclass"""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float                  # confidence score
    class_id: int                      # class ID
    class_name: str                    # class name
    track_id: Optional[int] = None     # tracking ID


class BasketballDetector:
    """Basketball scene object detector"""
    
    # Default class mapping
    DEFAULT_CLASSES = {
        0: "player",
        1: "basketball",
        2: "hoop",
        3: "referee"
    }
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 confidence_threshold: float = 0.5,
                 custom_classes: Optional[Dict[int, str]] = None):
        """
        Initialize the detector.

        Args:
            model_path: YOLOv8 model weights path (pretrained or custom trained)
            confidence_threshold: confidence threshold
            custom_classes: custom class mapping
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.classes = custom_classes or self.DEFAULT_CLASSES
        print(f"✅ Detector loaded: {model_path}")
    
    def detect(self, image: np.ndarray, 
               use_tracking: bool = False) -> List[Detection]:
        """
        Run object detection on a single image.

        Args:
            image: input image (BGR format, numpy array)
            use_tracking: whether to enable object tracking

        Returns:
            list of detection results
        """
        if use_tracking:
            results = self.model.track(
                image, 
                conf=self.confidence_threshold,
                persist=True,
                verbose=False
            )
        else:
            results = self.model(
                image, 
                conf=self.confidence_threshold,
                verbose=False
            )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                bbox = tuple(boxes.xyxy[i].cpu().numpy().astype(int))
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.classes.get(cls_id, 
                           result.names.get(cls_id, f"class_{cls_id}"))
                
                track_id = None
                if use_tracking and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                
                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=track_id
                ))
        
        return detections
    
    def detect_players(self, image: np.ndarray) -> List[Detection]:
        """Detect players only"""
        all_detections = self.detect(image)
        return [d for d in all_detections if d.class_name == "player"]
    
    def detect_ball(self, image: np.ndarray) -> Optional[Detection]:
        """Detect the basketball"""
        all_detections = self.detect(image)
        balls = [d for d in all_detections if d.class_name == "basketball"]
        return balls[0] if balls else None
    
    def draw_detections(self, image: np.ndarray, 
                        detections: List[Detection]) -> np.ndarray:
        """
        Draw detection results on the image.

        Args:
            image: input image
            detections: list of detection results

        Returns:
            annotated image
        """
        output = image.copy()
        
        # Class color mapping
        colors = {
            "player": (0, 255, 0),       # green
            "basketball": (0, 165, 255),  # orange
            "hoop": (255, 0, 0),          # blue
            "referee": (0, 255, 255),     # yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            
            # bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # label
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"
            
            # label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return output

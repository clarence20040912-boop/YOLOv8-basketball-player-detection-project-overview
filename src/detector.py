"""
YOLOv8 篮球运动员检测模块
检测球员、篮球、篮筐等目标
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float                  # 置信度
    class_id: int                      # 类别ID
    class_name: str                    # 类别名称
    track_id: Optional[int] = None     # 跟踪ID


class BasketballDetector:
    """篮球场景目标检测器"""
    
    # 默认类别映射
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
        初始化检测器
        
        Args:
            model_path: YOLOv8模型权重路径 (预训练或自定义训练)
            confidence_threshold: 置信度阈值
            custom_classes: 自定义类别映射
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.classes = custom_classes or self.DEFAULT_CLASSES
        print(f"✅ 检测器已加载: {model_path}")
    
    def detect(self, image: np.ndarray, 
               use_tracking: bool = False) -> List[Detection]:
        """
        对单张图片进行目标检测
        
        Args:
            image: 输入图片 (BGR格式, numpy数组)
            use_tracking: 是否启用目标跟踪
            
        Returns:
            检测结果列表
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
        """仅检测球员"""
        all_detections = self.detect(image)
        return [d for d in all_detections if d.class_name == "player"]
    
    def detect_ball(self, image: np.ndarray) -> Optional[Detection]:
        """检测篮球"""
        all_detections = self.detect(image)
        balls = [d for d in all_detections if d.class_name == "basketball"]
        return balls[0] if balls else None
    
    def draw_detections(self, image: np.ndarray, 
                        detections: List[Detection]) -> np.ndarray:
        """
        在图片上绘制检测结果
        
        Args:
            image: 输入图片
            detections: 检测结果列表
            
        Returns:
            绘制后的图片
        """
        output = image.copy()
        
        # 类别颜色映射
        colors = {
            "player": (0, 255, 0),       # 绿色
            "basketball": (0, 165, 255),  # 橙色
            "hoop": (255, 0, 0),          # 蓝色
            "referee": (0, 255, 255),     # 黄色
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            
            # 画框
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"
            
            # 标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return output

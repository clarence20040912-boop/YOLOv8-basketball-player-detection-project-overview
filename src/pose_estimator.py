"""
YOLOv8 Pose 姿态估计模块
提取球员骨骼关键点，用于动作识别
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# COCO 17个关键点定义
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 骨骼连接 (用于可视化)
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # 上肢
    (5, 11), (6, 12), (11, 12),                    # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16)         # 下肢
]


@dataclass
class PoseResult:
    """姿态估计结果"""
    bbox: Tuple[int, int, int, int]          # 人物边界框
    keypoints: np.ndarray                     # 关键点坐标 (17, 3) [x, y, conf]
    confidence: float                         # 检测置信度
    
    def get_keypoint(self, name: str) -> Tuple[float, float, float]:
        """获取指定关键点 (x, y, confidence)"""
        idx = KEYPOINT_NAMES.index(name)
        return tuple(self.keypoints[idx])
    
    def get_angle(self, p1: str, p2: str, p3: str) -> float:
        """
        计算三个关键点之间的角度
        用于判断动作（如手臂弯曲角度）
        
        Args:
            p1, p2, p3: 关键点名称, p2为角的顶点
            
        Returns:
            角度 (度)
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
    """YOLOv8 姿态估计器"""
    
    def __init__(self, model_path: str = "yolov8n-pose.pt",
                 confidence_threshold: float = 0.5):
        """
        初始化姿态估计器
        
        Args:
            model_path: YOLOv8-Pose模型路径
            confidence_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"✅ 姿态估计器已加载: {model_path}")
    
    def estimate(self, image: np.ndarray) -> List[PoseResult]:
        """
        对图片中的人物进行姿态估计
        
        Args:
            image: 输入图片 (BGR, numpy)
            
        Returns:
            姿态估计结果列表
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
        """绘制骨骼关键点"""
        output = image.copy()
        
        for pose in poses:
            kps = pose.keypoints
            
            # 画关键点
            for j in range(17):
                x, y, conf = kps[j]
                if conf > 0.5:
                    cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # 画骨骼连接
            for (idx1, idx2) in SKELETON_CONNECTIONS:
                x1, y1, c1 = kps[idx1]
                x2, y2, c2 = kps[idx2]
                if c1 > 0.5 and c2 > 0.5:
                    cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)),
                            (255, 0, 0), 2)
        
        return output

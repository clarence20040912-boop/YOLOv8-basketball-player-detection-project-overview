"""
视频处理模块
负责视频读取、关键帧提取、帧序列处理
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class FrameInfo:
    """帧信息"""
    frame: np.ndarray       # 帧图像
    frame_id: int           # 帧编号
    timestamp: float        # 时间戳（秒）
    is_keyframe: bool       # 是否为关键帧


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, video_path: str):
        """
        初始化视频处理器
        
        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"✅ 视频已加载: {video_path}")
        print(f"   分辨率: {self.width}x{self.height}, FPS: {self.fps:.1f}, "
              f"时长: {self.duration:.1f}s, 总帧数: {self.total_frames}")
    
    def read_frames(self, skip: int = 1) -> Generator[FrameInfo, None, None]:
        """
        逐帧读取视频
        
        Args:
            skip: 每隔skip帧读取一帧（用于加速处理）
            
        Yields:
            FrameInfo对象
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_id % skip == 0:
                yield FrameInfo(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=frame_id / self.fps if self.fps > 0 else 0,
                    is_keyframe=(frame_id % skip == 0)
                )
            
            frame_id += 1
    
    def extract_keyframes(self, method: str = "interval", 
                          interval: float = 1.0,
                          threshold: float = 30.0) -> List[FrameInfo]:
        """
        提取关键帧
        
        Args:
            method: 提取方法
                - "interval": 按时间间隔提取
                - "diff": 基于帧差法提取（场景变化时）
            interval: 时间间隔（秒），用于interval方法
            threshold: 帧差阈值，用于diff方法
            
        Returns:
            关键帧列表
        """
        keyframes = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if method == "interval":
            frame_interval = max(1, int(self.fps * interval))
            frame_id = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if frame_id % frame_interval == 0:
                    keyframes.append(FrameInfo(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=frame_id / self.fps,
                        is_keyframe=True
                    ))
                
                frame_id += 1
        
        elif method == "diff":
            prev_gray = None
            frame_id = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > threshold:
                        keyframes.append(FrameInfo(
                            frame=frame,
                            frame_id=frame_id,
                            timestamp=frame_id / self.fps,
                            is_keyframe=True
                        ))
                else:
                    # 第一帧总是关键帧
                    keyframes.append(FrameInfo(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=frame_id / self.fps,
                        is_keyframe=True
                    ))
                
                prev_gray = gray
                frame_id += 1
        
        print(f"✅ 提取了 {len(keyframes)} 个关键帧 (方法: {method})")
        return keyframes
    
    def get_frame_at(self, timestamp: float) -> Optional[np.ndarray]:
        """获取指定时间戳的帧"""
        frame_id = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """释放视频资源"""
        if self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        self.release()

"""
Video Processing Module
Handles video reading, keyframe extraction, and frame sequence processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class FrameInfo:
    """Frame information"""
    frame: np.ndarray       # frame image
    frame_id: int           # frame number
    timestamp: float        # timestamp (seconds)
    is_keyframe: bool       # whether this is a keyframe


class VideoProcessor:
    """Video processor"""
    
    def __init__(self, video_path: str):
        """
        Initialize the video processor.

        Args:
            video_path: path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"✅ Video loaded: {video_path}")
        print(f"   Resolution: {self.width}x{self.height}, FPS: {self.fps:.1f}, "
              f"Duration: {self.duration:.1f}s, Total frames: {self.total_frames}")
    
    def read_frames(self, skip: int = 1) -> Generator[FrameInfo, None, None]:
        """
        Read frames from the video one by one.

        Args:
            skip: read one frame every `skip` frames (for faster processing)

        Yields:
            FrameInfo objects
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
        Extract keyframes from the video.

        Args:
            method: extraction method
                - "interval": extract at fixed time intervals
                - "diff": extract on scene changes (frame difference)
            interval: time interval in seconds, used for the "interval" method
            threshold: frame difference threshold, used for the "diff" method

        Returns:
            list of keyframes
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
                    # first frame is always a keyframe
                    keyframes.append(FrameInfo(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=frame_id / self.fps,
                        is_keyframe=True
                    ))
                
                prev_gray = gray
                frame_id += 1
        
        print(f"✅ Extracted {len(keyframes)} keyframes (method: {method})")
        return keyframes
    
    def get_frame_at(self, timestamp: float) -> Optional[np.ndarray]:
        """Get the frame at the specified timestamp"""
        frame_id = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release video resources"""
        if self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        self.release()

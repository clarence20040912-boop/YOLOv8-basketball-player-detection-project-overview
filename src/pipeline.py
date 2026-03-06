"""
Main pipeline
Chains all modules: detection → pose estimation → action recognition → commentary generation
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.detector import BasketballDetector, Detection
from src.pose_estimator import PoseEstimator, PoseResult
from src.action_recognizer import RuleBasedActionRecognizer, ActionResult, ActionType
from src.commentary_generator import (
    TemplateCommentaryGenerator, 
    LLMCommentaryGenerator, 
    CommentaryResult
)
from src.video_processor import VideoProcessor, FrameInfo


@dataclass
class FrameAnalysisResult:
    """Single frame analysis result"""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    poses: List[PoseResult]
    actions: List[ActionResult]
    commentaries: List[CommentaryResult]
    annotated_frame: Optional[np.ndarray] = None


class BasketballCommentaryPipeline:
    """Basketball commentary system main pipeline"""
    
    def __init__(self, 
                 detector_model: str = "yolov8n.pt",
                 pose_model: str = "yolov8n-pose.pt",
                 use_llm: bool = False,
                 llm_api_key: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 language: str = "en"):
        """
        Initialize the pipeline.

        Args:
            detector_model: detection model path
            pose_model: pose model path
            use_llm: whether to use LLM for commentary generation
            llm_api_key: LLM API key
            confidence_threshold: confidence threshold
            language: commentary language
        """
        print("=" * 60)
        print("🏀 Initializing Basketball Commentary System")
        print("=" * 60)
        
        # initialize modules
        self.detector = BasketballDetector(
            detector_model, confidence_threshold
        )
        self.pose_estimator = PoseEstimator(
            pose_model, confidence_threshold
        )
        self.action_recognizer = RuleBasedActionRecognizer()
        
        if use_llm:
            self.commentary_generator = LLMCommentaryGenerator(
                api_key=llm_api_key
            )
        else:
            self.commentary_generator = TemplateCommentaryGenerator()
        
        self.language = language
        print("=" * 60)
        print("✅ System initialization complete!")
        print("=" * 60)
    
    def analyze_image(self, image: np.ndarray, 
                      draw: bool = True) -> FrameAnalysisResult:
        """
        Analyze a single image.

        Args:
            image: input image
            draw: whether to draw annotations

        Returns:
            analysis result
        """
        # Step 1: object detection
        detections = self.detector.detect(image)
        
        # Step 2: locate basketball
        ball_det = next((d for d in detections if d.class_name == "basketball"), None)
        ball_pos = None
        if ball_det:
            bx = (ball_det.bbox[0] + ball_det.bbox[2]) // 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) // 2
            ball_pos = (bx, by)
        
        # Step 3: pose estimation
        poses = self.pose_estimator.estimate(image)
        
        # Step 4: action recognition
        actions = []
        for pose in poses:
            action = self.action_recognizer.recognize(pose, ball_pos)
            actions.append(action)
        
        # Step 5: commentary generation
        commentaries = []
        for i, action in enumerate(actions):
            player_name = f"Player #{i+1}"
            commentary = self.commentary_generator.generate(
                action, player_name, self.language
            )
            commentaries.append(commentary)
        
        # Step 6: draw annotations
        annotated = None
        if draw:
            annotated = self._draw_results(image, detections, poses, actions, commentaries)
        
        return FrameAnalysisResult(
            frame_id=0,
            timestamp=0.0,
            detections=detections,
            poses=poses,
            actions=actions,
            commentaries=commentaries,
            annotated_frame=annotated
        )
    
    def analyze_video(self, video_path: str,
                      output_path: Optional[str] = None,
                      keyframe_interval: float = 1.0,
                      show_progress: bool = True) -> List[FrameAnalysisResult]:
        """
        Analyze a video.

        Args:
            video_path: video file path
            output_path: output video path (optional)
            keyframe_interval: keyframe extraction interval in seconds
            show_progress: whether to show a progress bar

        Returns:
            analysis results for all keyframes
        """
        from tqdm import tqdm
        
        processor = VideoProcessor(video_path)
        keyframes = processor.extract_keyframes(
            method="interval", 
            interval=keyframe_interval
        )
        
        # output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, fourcc, 
                1.0 / keyframe_interval,  # output frame rate
                (processor.width, processor.height)
            )
        
        results = []
        iterator = tqdm(keyframes, desc="Analyzing") if show_progress else keyframes
        need_draw = output_path is not None
        
        try:
            for frame_info in iterator:
                result = self.analyze_image(frame_info.frame, draw=need_draw)
                result.frame_id = frame_info.frame_id
                result.timestamp = frame_info.timestamp
                results.append(result)
                
                # write to output video
                if writer and result.annotated_frame is not None:
                    writer.write(result.annotated_frame)
                
                # free heavy frame data to reduce memory usage
                result.annotated_frame = None
                frame_info.frame = None
                
                # print commentary
                if result.commentaries:
                    timestamp_str = f"[{frame_info.timestamp:.1f}s]"
                    for comm in result.commentaries:
                        if show_progress:
                            tqdm.write(f"  {timestamp_str} {comm.text}")
        finally:
            if writer:
                writer.release()
            processor.release()
        
        if output_path:
            print(f"✅ Output video saved: {output_path}")
        
        print(f"\n📊 Analysis complete: {len(results)} keyframes, "
              f"{sum(len(r.actions) for r in results)} actions detected")
        
        return results
    
    def _draw_results(self, image: np.ndarray,
                      detections: List[Detection],
                      poses: List[PoseResult],
                      actions: List[ActionResult],
                      commentaries: List[CommentaryResult]) -> np.ndarray:
        """Draw combined results"""
        output = image.copy()
        
        # draw detection boxes
        output = self.detector.draw_detections(output, detections)
        
        # draw skeleton
        output = self.pose_estimator.draw_poses(output, poses)
        
        # draw action labels
        for i, (pose, action) in enumerate(zip(poses, actions)):
            x1, y1, x2, y2 = pose.bbox
            label = f"{action.action_en} ({action.confidence:.0%})"
            
            cv2.putText(output, label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # draw commentary text at the bottom
        if commentaries:
            h, w = output.shape[:2]
            # semi-transparent bar
            overlay = output.copy()
            bar_height = 40 * len(commentaries) + 20
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
            output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)
            
            for i, comm in enumerate(commentaries):
                y = h - bar_height + 30 + i * 40
                truncated = comm.text[:57] + "..." if len(comm.text) > 60 else comm.text
                cv2.putText(output, truncated, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return output

"""
主流水线
串联所有模块：检测 → 姿态估计 → 动作识别 → 解说生成
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
    """单帧分析结果"""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    poses: List[PoseResult]
    actions: List[ActionResult]
    commentaries: List[CommentaryResult]
    annotated_frame: Optional[np.ndarray] = None


class BasketballCommentaryPipeline:
    """篮球解说系统主流水线"""
    
    def __init__(self, 
                 detector_model: str = "yolov8n.pt",
                 pose_model: str = "yolov8n-pose.pt",
                 use_llm: bool = False,
                 llm_api_key: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 language: str = "cn"):
        """
        初始化流水线
        
        Args:
            detector_model: 检测模型路径
            pose_model: 姿态模型路径
            use_llm: 是否使用LLM生成解说
            llm_api_key: LLM API密钥
            confidence_threshold: 置信度阈值
            language: 解说语言
        """
        print("=" * 60)
        print("🏀 初始化篮球解说系统")
        print("=" * 60)
        
        # 初始化各模块
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
        print("✅ 系统初始化完成！")
        print("=" * 60)
    
    def analyze_image(self, image: np.ndarray, 
                      draw: bool = True) -> FrameAnalysisResult:
        """
        分析单张图片
        
        Args:
            image: 输入图片
            draw: 是否绘制标注
            
        Returns:
            分析结果
        """
        # Step 1: 目标检测
        detections = self.detector.detect(image)
        
        # Step 2: 获取篮球位置
        ball_det = next((d for d in detections if d.class_name == "basketball"), None)
        ball_pos = None
        if ball_det:
            bx = (ball_det.bbox[0] + ball_det.bbox[2]) // 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) // 2
            ball_pos = (bx, by)
        
        # Step 3: 姿态估计
        poses = self.pose_estimator.estimate(image)
        
        # Step 4: 动作识别
        actions = []
        for pose in poses:
            action = self.action_recognizer.recognize(pose, ball_pos)
            actions.append(action)
        
        # Step 5: 生成解说
        commentaries = []
        for i, action in enumerate(actions):
            player_name = f"{'球员' if self.language == 'cn' else 'Player'} #{i+1}"
            commentary = self.commentary_generator.generate(
                action, player_name, self.language
            )
            commentaries.append(commentary)
        
        # Step 6: 绘制标注
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
        分析视频
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径（可选）
            keyframe_interval: 关键帧提取间隔（秒）
            show_progress: 是否显示进度
            
        Returns:
            所有关键帧的分析结果
        """
        from tqdm import tqdm
        
        processor = VideoProcessor(video_path)
        keyframes = processor.extract_keyframes(
            method="interval", 
            interval=keyframe_interval
        )
        
        # 输出视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, fourcc, 
                1.0 / keyframe_interval,  # 输出帧率
                (processor.width, processor.height)
            )
        
        results = []
        iterator = tqdm(keyframes, desc="分析中") if show_progress else keyframes
        
        for frame_info in iterator:
            result = self.analyze_image(frame_info.frame, draw=True)
            result.frame_id = frame_info.frame_id
            result.timestamp = frame_info.timestamp
            results.append(result)
            
            # 写入输出视频
            if writer and result.annotated_frame is not None:
                writer.write(result.annotated_frame)
            
            # 打印解说
            if result.commentaries:
                timestamp_str = f"[{frame_info.timestamp:.1f}s]"
                for comm in result.commentaries:
                    if show_progress:
                        tqdm.write(f"  {timestamp_str} {comm.text}")
        
        if writer:
            writer.release()
            print(f"✅ 输出视频已保存: {output_path}")
        
        processor.release()
        
        print(f"\n📊 分析完成: {len(results)} 个关键帧, "
              f"检测到 {sum(len(r.actions) for r in results)} 个动作")
        
        return results
    
    def _draw_results(self, image: np.ndarray,
                      detections: List[Detection],
                      poses: List[PoseResult],
                      actions: List[ActionResult],
                      commentaries: List[CommentaryResult]) -> np.ndarray:
        """绘制综合结果"""
        output = image.copy()
        
        # 绘制检测框
        output = self.detector.draw_detections(output, detections)
        
        # 绘制骨骼
        output = self.pose_estimator.draw_poses(output, poses)
        
        # 绘制动作标签
        for i, (pose, action) in enumerate(zip(poses, actions)):
            x1, y1, x2, y2 = pose.bbox
            label = f"{action.action_cn} ({action.confidence:.0%})"
            
            cv2.putText(output, label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 在底部绘制解说文字
        if commentaries:
            h, w = output.shape[:2]
            # 半透明底栏
            overlay = output.copy()
            bar_height = 40 * len(commentaries) + 20
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
            output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)
            
            for i, comm in enumerate(commentaries):
                y = h - bar_height + 30 + i * 40
                # 注意: OpenCV不直接支持中文，实际使用中可用PIL绘制
                cv2.putText(output, comm.text[:60], (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return output

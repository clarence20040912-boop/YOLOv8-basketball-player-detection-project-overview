"""
篮球动作识别模块
基于姿态关键点特征，识别球员动作
支持：投篮、运球、传球、扣篮、防守、跑动、站立
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.pose_estimator import PoseResult, KEYPOINT_NAMES


class ActionType(Enum):
    """篮球动作类型"""
    SHOOTING = "shooting"       # 投篮
    DRIBBLING = "dribbling"     # 运球
    PASSING = "passing"         # 传球
    DUNKING = "dunking"         # 扣篮
    BLOCKING = "blocking"       # 盖帽/防守
    REBOUNDING = "rebounding"   # 抢篮板
    RUNNING = "running"         # 跑动
    STANDING = "standing"       # 站立
    UNKNOWN = "unknown"         # 未知


# 动作中文名映射
ACTION_CN_NAMES = {
    ActionType.SHOOTING: "投篮",
    ActionType.DRIBBLING: "运球",
    ActionType.PASSING: "传球",
    ActionType.DUNKING: "扣篮",
    ActionType.BLOCKING: "盖帽",
    ActionType.REBOUNDING: "抢篮板",
    ActionType.RUNNING: "跑动",
    ActionType.STANDING: "站立",
    ActionType.UNKNOWN: "未知动作",
}


@dataclass
class ActionResult:
    """动作识别结果"""
    action: ActionType
    confidence: float
    action_cn: str = ""
    details: Dict = None
    
    def __post_init__(self):
        self.action_cn = ACTION_CN_NAMES.get(self.action, "未知")
        if self.details is None:
            self.details = {}


class RuleBasedActionRecognizer:
    """
    基于规则的动作识别器
    利用关键点几何关系判断动作类型
    
    适用于快速原型和毕业设计展示
    """
    
    def __init__(self):
        print("✅ 规则动作识别器已初始化")
    
    def recognize(self, pose: PoseResult, 
                  ball_position: Optional[Tuple[int, int]] = None) -> ActionResult:
        """
        识别单个球员的动作
        
        Args:
            pose: 姿态估计结果
            ball_position: 篮球位置 (x, y)，可选
            
        Returns:
            动作识别结果
        """
        kps = pose.keypoints
        features = self._extract_features(kps, ball_position)
        
        # 按优先级判断动作
        action, confidence, details = self._classify_action(features, kps)
        
        return ActionResult(
            action=action,
            confidence=confidence,
            details=details
        )
    
    def _extract_features(self, keypoints: np.ndarray, 
                          ball_pos: Optional[Tuple[int, int]]) -> Dict:
        """提取姿态特征"""
        features = {}
        
        # 关键点坐标
        left_wrist = keypoints[KEYPOINT_NAMES.index("left_wrist")]
        right_wrist = keypoints[KEYPOINT_NAMES.index("right_wrist")]
        left_shoulder = keypoints[KEYPOINT_NAMES.index("left_shoulder")]
        right_shoulder = keypoints[KEYPOINT_NAMES.index("right_shoulder")]
        left_elbow = keypoints[KEYPOINT_NAMES.index("left_elbow")]
        right_elbow = keypoints[KEYPOINT_NAMES.index("right_elbow")]
        left_hip = keypoints[KEYPOINT_NAMES.index("left_hip")]
        right_hip = keypoints[KEYPOINT_NAMES.index("right_hip")]
        left_knee = keypoints[KEYPOINT_NAMES.index("left_knee")]
        right_knee = keypoints[KEYPOINT_NAMES.index("right_knee")]
        left_ankle = keypoints[KEYPOINT_NAMES.index("left_ankle")]
        right_ankle = keypoints[KEYPOINT_NAMES.index("right_ankle")]
        nose = keypoints[KEYPOINT_NAMES.index("nose")]
        
        # 1. 手臂高度（相对于肩膀）
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        body_height = abs(hip_y - shoulder_y) + 1e-6
        
        features["left_hand_raised"] = (shoulder_y - left_wrist[1]) / body_height
        features["right_hand_raised"] = (shoulder_y - right_wrist[1]) / body_height
        features["max_hand_raised"] = max(features["left_hand_raised"], 
                                          features["right_hand_raised"])
        features["both_hands_raised"] = min(features["left_hand_raised"], 
                                            features["right_hand_raised"])
        
        # 2. 手腕高度（绝对）
        features["left_wrist_y"] = left_wrist[1]
        features["right_wrist_y"] = right_wrist[1]
        features["min_wrist_y"] = min(left_wrist[1], right_wrist[1])  # 越小越高
        
        # 3. 手臂弯曲角度
        features["left_elbow_angle"] = self._calculate_angle(
            left_shoulder[:2], left_elbow[:2], left_wrist[:2])
        features["right_elbow_angle"] = self._calculate_angle(
            right_shoulder[:2], right_elbow[:2], right_wrist[:2])
        
        # 4. 膝盖弯曲角度
        features["left_knee_angle"] = self._calculate_angle(
            left_hip[:2], left_knee[:2], left_ankle[:2])
        features["right_knee_angle"] = self._calculate_angle(
            right_hip[:2], right_knee[:2], right_ankle[:2])
        
        # 5. 手腕到臀部的距离（判断运球）
        features["left_wrist_to_hip"] = (left_wrist[1] - hip_y) / body_height
        features["right_wrist_to_hip"] = (right_wrist[1] - hip_y) / body_height
        
        # 6. 手臂展开宽度（判断传球/防守）
        features["arm_spread"] = abs(left_wrist[0] - right_wrist[0]) / body_height
        
        # 7. 身体倾斜度
        features["body_lean"] = abs(nose[0] - (left_hip[0] + right_hip[0]) / 2) / body_height
        
        # 8. 与篮球的距离
        if ball_pos is not None:
            bx, by = ball_pos
            features["left_hand_to_ball"] = np.sqrt(
                (left_wrist[0] - bx)**2 + (left_wrist[1] - by)**2) / body_height
            features["right_hand_to_ball"] = np.sqrt(
                (right_wrist[0] - bx)**2 + (right_wrist[1] - by)**2) / body_height
            features["min_hand_to_ball"] = min(features["left_hand_to_ball"], 
                                               features["right_hand_to_ball"])
            features["ball_above_head"] = (nose[1] - by) / body_height
        
        return features
    
    def _classify_action(self, features: Dict, 
                         keypoints: np.ndarray) -> Tuple[ActionType, float, Dict]:
        """基于规则的动作分类"""
        scores = {}
        
        # === 投篮 ===
        # 特征: 单手或双手高举过头，手臂从弯曲到伸展
        shooting_score = 0.0
        if features["max_hand_raised"] > 0.8:
            shooting_score += 0.4
        if features["max_hand_raised"] > 1.2:
            shooting_score += 0.2
        # 投篮手肘角度通常在90-160度
        max_elbow = max(features["left_elbow_angle"], features["right_elbow_angle"])
        if 80 < max_elbow < 170:
            shooting_score += 0.2
        # 膝盖微弯
        avg_knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2
        if 130 < avg_knee < 175:
            shooting_score += 0.2
        scores[ActionType.SHOOTING] = min(shooting_score, 1.0)
        
        # === 运球 ===
        # 特征: 手在腰部以下，身体微弯
        dribbling_score = 0.0
        if features["left_wrist_to_hip"] > 0.3 or features["right_wrist_to_hip"] > 0.3:
            dribbling_score += 0.4
        if features["max_hand_raised"] < 0.2:
            dribbling_score += 0.3
        if features["body_lean"] > 0.2:
            dribbling_score += 0.3
        scores[ActionType.DRIBBLING] = min(dribbling_score, 1.0)
        
        # === 传球 ===
        # 特征: 双手在胸前位置，手臂展开
        passing_score = 0.0
        if 0.0 < features["max_hand_raised"] < 0.6:
            passing_score += 0.3
        if features["arm_spread"] > 1.0:
            passing_score += 0.4
        if features["both_hands_raised"] > -0.2:
            passing_score += 0.3
        scores[ActionType.PASSING] = min(passing_score, 1.0)
        
        # === 扣篮 ===
        # 特征: 单手高举，身体大幅跳起
        dunking_score = 0.0
        if features["max_hand_raised"] > 1.5:
            dunking_score += 0.5
        # 单手为主
        hand_diff = abs(features["left_hand_raised"] - features["right_hand_raised"])
        if hand_diff > 0.5:
            dunking_score += 0.3
        if features["body_lean"] > 0.3:
            dunking_score += 0.2
        scores[ActionType.DUNKING] = min(dunking_score, 1.0)
        
        # === 防守/盖帽 ===
        # 特征: 双手展开，身体略蹲
        blocking_score = 0.0
        if features["arm_spread"] > 1.5:
            blocking_score += 0.4
        if features["both_hands_raised"] > 0.3:
            blocking_score += 0.3
        avg_knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2
        if avg_knee < 150:
            blocking_score += 0.3
        scores[ActionType.BLOCKING] = min(blocking_score, 1.0)
        
        # === 抢篮板 ===
        # 特征: 双手高举，身体伸展
        rebounding_score = 0.0
        if features["both_hands_raised"] > 0.8:
            rebounding_score += 0.5
        if features["arm_spread"] < 1.0:
            rebounding_score += 0.3
        scores[ActionType.REBOUNDING] = min(rebounding_score, 0.95)
        
        # === 站立 ===
        standing_score = 0.0
        avg_knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2
        if avg_knee > 160:
            standing_score += 0.3
        if abs(features["max_hand_raised"]) < 0.3:
            standing_score += 0.3
        if features["body_lean"] < 0.15:
            standing_score += 0.2
        scores[ActionType.STANDING] = min(standing_score, 0.8)
        
        # 选择最高分的动作
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        
        if best_score < 0.3:
            best_action = ActionType.UNKNOWN
            best_score = 0.0
        
        return best_action, best_score, {"all_scores": {k.value: round(v, 3) for k, v in scores.items()}}
    
    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """计算三点角度, b为顶点"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

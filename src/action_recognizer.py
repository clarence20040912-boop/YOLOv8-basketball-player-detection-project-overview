"""
Basketball Action Recognition Module
Recognizes player actions from pose keypoint features.
Supports: shooting, dribbling, passing, dunking, blocking, rebounding, running, standing
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.pose_estimator import PoseResult, KEYPOINT_NAMES


class ActionType(Enum):
    """Basketball action types"""
    SHOOTING = "shooting"
    DRIBBLING = "dribbling"
    PASSING = "passing"
    DUNKING = "dunking"
    BLOCKING = "blocking"
    REBOUNDING = "rebounding"
    RUNNING = "running"
    STANDING = "standing"
    UNKNOWN = "unknown"


# Action English name mapping
ACTION_EN_NAMES = {
    ActionType.SHOOTING: "Shooting",
    ActionType.DRIBBLING: "Dribbling",
    ActionType.PASSING: "Passing",
    ActionType.DUNKING: "Dunking",
    ActionType.BLOCKING: "Blocking",
    ActionType.REBOUNDING: "Rebounding",
    ActionType.RUNNING: "Running",
    ActionType.STANDING: "Standing",
    ActionType.UNKNOWN: "Unknown",
}


@dataclass
class ActionResult:
    """Action recognition result"""
    action: ActionType
    confidence: float
    action_en: str = ""
    details: Dict = None
    
    def __post_init__(self):
        self.action_en = ACTION_EN_NAMES.get(self.action, "Unknown")
        if self.details is None:
            self.details = {}


class RuleBasedActionRecognizer:
    """
    Rule-based action recognizer.
    Uses keypoint geometry to classify player actions.

    Suitable for quick prototyping and project demonstrations.
    """
    
    def __init__(self):
        print("✅ Rule-based action recognizer initialized")
    
    def recognize(self, pose: PoseResult, 
                  ball_position: Optional[Tuple[int, int]] = None) -> ActionResult:
        """
        Recognize the action of a single player.

        Args:
            pose: pose estimation result
            ball_position: basketball position (x, y), optional

        Returns:
            action recognition result
        """
        kps = pose.keypoints
        features = self._extract_features(kps, ball_position)
        
        # classify action by priority
        action, confidence, details = self._classify_action(features, kps)
        
        return ActionResult(
            action=action,
            confidence=confidence,
            details=details
        )
    
    def _extract_features(self, keypoints: np.ndarray, 
                          ball_pos: Optional[Tuple[int, int]]) -> Dict:
        """Extract pose features"""
        features = {}
        
        # keypoint coordinates
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
        
        # 1. Hand height (relative to shoulder)
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        body_height = abs(hip_y - shoulder_y) + 1e-6
        
        features["left_hand_raised"] = (shoulder_y - left_wrist[1]) / body_height
        features["right_hand_raised"] = (shoulder_y - right_wrist[1]) / body_height
        features["max_hand_raised"] = max(features["left_hand_raised"], 
                                          features["right_hand_raised"])
        features["both_hands_raised"] = min(features["left_hand_raised"], 
                                            features["right_hand_raised"])
        
        # 2. Wrist height (absolute)
        features["left_wrist_y"] = left_wrist[1]
        features["right_wrist_y"] = right_wrist[1]
        features["min_wrist_y"] = min(left_wrist[1], right_wrist[1])  # smaller = higher
        
        # 3. Elbow bend angle
        features["left_elbow_angle"] = self._calculate_angle(
            left_shoulder[:2], left_elbow[:2], left_wrist[:2])
        features["right_elbow_angle"] = self._calculate_angle(
            right_shoulder[:2], right_elbow[:2], right_wrist[:2])
        
        # 4. Knee bend angle
        features["left_knee_angle"] = self._calculate_angle(
            left_hip[:2], left_knee[:2], left_ankle[:2])
        features["right_knee_angle"] = self._calculate_angle(
            right_hip[:2], right_knee[:2], right_ankle[:2])
        
        # 5. Wrist-to-hip distance (for dribbling detection)
        features["left_wrist_to_hip"] = (left_wrist[1] - hip_y) / body_height
        features["right_wrist_to_hip"] = (right_wrist[1] - hip_y) / body_height
        
        # 6. Arm spread width (for passing/blocking detection)
        features["arm_spread"] = abs(left_wrist[0] - right_wrist[0]) / body_height
        
        # 7. Body lean
        features["body_lean"] = abs(nose[0] - (left_hip[0] + right_hip[0]) / 2) / body_height
        
        # 8. Distance to basketball
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
        """Rule-based action classification"""
        scores = {}
        
        # === SHOOTING ===
        # Features: one or both hands raised high, arm going from bent to extended
        shooting_score = 0.0
        if features["max_hand_raised"] > 0.8:
            shooting_score += 0.4
        if features["max_hand_raised"] > 1.2:
            shooting_score += 0.2
        # shooting elbow angle typically 90-160 degrees
        max_elbow = max(features["left_elbow_angle"], features["right_elbow_angle"])
        if 80 < max_elbow < 170:
            shooting_score += 0.2
        # slight knee bend
        avg_knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2
        if 130 < avg_knee < 175:
            shooting_score += 0.2
        scores[ActionType.SHOOTING] = min(shooting_score, 1.0)
        
        # === DRIBBLING ===
        # Features: hand below hip, body slightly bent
        dribbling_score = 0.0
        if features["left_wrist_to_hip"] > 0.3 or features["right_wrist_to_hip"] > 0.3:
            dribbling_score += 0.4
        if features["max_hand_raised"] < 0.2:
            dribbling_score += 0.3
        if features["body_lean"] > 0.2:
            dribbling_score += 0.3
        scores[ActionType.DRIBBLING] = min(dribbling_score, 1.0)
        
        # === PASSING ===
        # Features: both hands at chest level, arms spread
        passing_score = 0.0
        if 0.0 < features["max_hand_raised"] < 0.6:
            passing_score += 0.3
        if features["arm_spread"] > 1.0:
            passing_score += 0.4
        if features["both_hands_raised"] > -0.2:
            passing_score += 0.3
        scores[ActionType.PASSING] = min(passing_score, 1.0)
        
        # === DUNKING ===
        # Features: one hand raised very high, body fully extended
        dunking_score = 0.0
        if features["max_hand_raised"] > 1.5:
            dunking_score += 0.5
        # predominantly one-handed
        hand_diff = abs(features["left_hand_raised"] - features["right_hand_raised"])
        if hand_diff > 0.5:
            dunking_score += 0.3
        if features["body_lean"] > 0.3:
            dunking_score += 0.2
        scores[ActionType.DUNKING] = min(dunking_score, 1.0)
        
        # === BLOCKING ===
        # Features: both hands spread wide, body slightly crouched
        blocking_score = 0.0
        if features["arm_spread"] > 1.5:
            blocking_score += 0.4
        if features["both_hands_raised"] > 0.3:
            blocking_score += 0.3
        avg_knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2
        if avg_knee < 150:
            blocking_score += 0.3
        scores[ActionType.BLOCKING] = min(blocking_score, 1.0)
        
        # === REBOUNDING ===
        # Features: both hands raised, body fully extended
        rebounding_score = 0.0
        if features["both_hands_raised"] > 0.8:
            rebounding_score += 0.5
        if features["arm_spread"] < 1.0:
            rebounding_score += 0.3
        scores[ActionType.REBOUNDING] = min(rebounding_score, 0.95)
        
        # === STANDING ===
        standing_score = 0.0
        avg_knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2
        if avg_knee > 160:
            standing_score += 0.3
        if abs(features["max_hand_raised"]) < 0.3:
            standing_score += 0.3
        if features["body_lean"] < 0.15:
            standing_score += 0.2
        scores[ActionType.STANDING] = min(standing_score, 0.8)
        
        # select action with the highest score
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        
        if best_score < 0.3:
            best_action = ActionType.UNKNOWN
            best_score = 0.0
        
        return best_action, best_score, {"all_scores": {k.value: round(v, 3) for k, v in scores.items()}}
    
    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Compute the angle at vertex b formed by points a-b-c."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

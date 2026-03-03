"""
可视化工具
用于结果展示和调试
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from PIL import Image, ImageDraw, ImageFont


def put_chinese_text(image: np.ndarray, text: str, 
                     position: tuple, font_size: int = 24,
                     color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    在图片上绘制中文文字
    OpenCV不支持中文，需要用PIL
    
    Args:
        image: BGR图片
        text: 中文文本
        position: (x, y) 位置
        font_size: 字体大小
        color: BGR颜色
        
    Returns:
        绘制后的图片
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    try:
        # 尝试加载中文字体
        font = ImageFont.truetype("simhei.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
    
    # BGR → RGB颜色
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def plot_action_distribution(actions: List[str], save_path: str = None):
    """绘制动作分布统计图"""
    from collections import Counter
    
    counts = Counter(actions)
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    plt.bar(labels, values, color=colors, edgecolor='black')
    plt.xlabel("动作类型", fontsize=12)
    plt.ylabel("次数", fontsize=12)
    plt.title("🏀 篮球动作识别分布", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ 统计图已保存: {save_path}")
    
    plt.show()

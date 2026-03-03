"""
训练YOLOv8篮球检测模型
使用自定义数据集微调预训练模型
"""

import argparse
from ultralytics import YOLO


def train_detector(
    data_yaml: str = "config/dataset.yaml",
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    name: str = "basketball_detector",
    device: str = "",
    patience: int = 20,
    lr0: float = 0.01
):
    """
    训练篮球目标检测模型
    
    Args:
        data_yaml: 数据集配置文件路径
        model: 预训练模型 (yolov8n/s/m/l/x.pt)
        epochs: 训练轮次
        imgsz: 输入图片大小
        batch: 批次大小
        name: 实验名称
        device: 训练设备 ("" 自动, "0" GPU0, "cpu" CPU)
        patience: 早停耐心值
        lr0: 初始学习率
    """
    print("=" * 60)
    print("🏀 开始训练篮球目标检测模型")
    print("=" * 60)
    print(f"  模型: {model}")
    print(f"  数据集: {data_yaml}")
    print(f"  轮次: {epochs}")
    print(f"  图片大小: {imgsz}")
    print(f"  批次大小: {batch}")
    print(f"  设备: {device or '自动'}")
    print("=" * 60)
    
    # 加载预训练模型
    yolo = YOLO(model)
    
    # 开始训练
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        device=device if device else None,
        patience=patience,
        lr0=lr0,
        # 数据增强
        hsv_h=0.015,     # 色调
        hsv_s=0.7,       # 饱和度
        hsv_v=0.4,       # 亮度
        degrees=10.0,     # 旋转
        translate=0.1,    # 平移
        scale=0.5,        # 缩放
        fliplr=0.5,       # 水平翻转
        mosaic=1.0,       # 马赛克增强
        mixup=0.1,        # MixUp增强
        # 保存
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print(f"  最佳模型: runs/detect/{name}/weights/best.pt")
    print(f"  最后模型: runs/detect/{name}/weights/last.pt")
    print(f"  训练日志: runs/detect/{name}/")
    print("=" * 60)
    
    # 验证
    print("\n📊 在验证集上评估...")
    val_results = yolo.val()
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练YOLOv8篮球检测模型")
    parser.add_argument("--data", type=str, default="config/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="预训练模型: yolov8n/s/m/l/x.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--name", type=str, default="basketball_detector")
    args = parser.parse_args()
    
    train_detector(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name
    )

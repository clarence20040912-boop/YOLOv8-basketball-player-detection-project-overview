"""
数据集下载脚本
支持从Roboflow下载篮球检测数据集
"""

import os
import argparse


def download_roboflow_dataset(api_key: str, workspace: str, project: str, 
                                version: int, save_dir: str = "data"):
    """
    从Roboflow下载数据集
    
    Args:
        api_key: Roboflow API密钥
        workspace: Roboflow工作空间名
        project: 项目名
        version: 数据集版本号
        save_dir: 保存目录
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("请先安装roboflow: pip install roboflow")
        return
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=save_dir)
    print(f"✅ 数据集已下载到: {save_dir}")
    return dataset


def download_default_basketball_dataset(save_dir: str = "data"):
    """
    下载默认的篮球检测数据集（Roboflow公开数据集）
    
    使用前请到 https://roboflow.com 注册获取API Key
    推荐数据集:
    1. https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2
    2. https://universe.roboflow.com 搜索 "basketball player detection"
    """
    print("=" * 60)
    print("🏀 篮球运动员检测数据集下载工具")
    print("=" * 60)
    print()
    print("推荐数据集来源:")
    print("1. Roboflow Universe (推荐，已标注，支持YOLOv8格式导出)")
    print("   https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2")
    print()
    print("2. Ultralytics Platform")
    print("   https://platform.ultralytics.com/tomer-yadgarov/datasets/basketball-player-detection-2v4iyolov8")
    print()
    print("3. GitHub开源项目 (含数据+代码)")
    print("   https://github.com/LittleFish-Coder/basketball-sports-ai")
    print()
    print("4. SpaceJam Dataset (动作识别专用)")
    print("   https://github.com/hkair/Basketball-Action-Recognition")
    print()
    
    api_key = input("请输入你的Roboflow API Key (没有请按Enter跳过): ").strip()
    
    if api_key:
        download_roboflow_dataset(
            api_key=api_key,
            workspace="roboflow-universe-projects",
            project="basketball-players-fy4c2",
            version=16,
            save_dir=save_dir
        )
    else:
        print("\n⚠️ 请手动下载数据集:")
        print(f"  1. 访问上述链接下载数据集")
        print(f"  2. 选择 'YOLOv8' 格式导出")
        print(f"  3. 解压到 '{save_dir}/' 目录下")
        print(f"  4. 确保目录结构如下:")
        print(f"     {save_dir}/")
        print(f"     ├── images/")
        print(f"     │   ├── train/")
        print(f"     │   └── val/")
        print(f"     └── labels/")
        print(f"         ├── train/")
        print(f"         └── val/")
    
    # 创建必要的目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(save_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "labels", split), exist_ok=True)
    
    print(f"\n✅ 目录结构已创建: {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载篮球检测数据集")
    parser.add_argument("--save_dir", type=str, default="data", help="保存目录")
    parser.add_argument("--api_key", type=str, default="", help="Roboflow API Key")
    args = parser.parse_args()
    
    if args.api_key:
        download_roboflow_dataset(
            api_key=args.api_key,
            workspace="roboflow-universe-projects",
            project="basketball-players-fy4c2",
            version=16,
            save_dir=args.save_dir
        )
    else:
        download_default_basketball_dataset(args.save_dir)

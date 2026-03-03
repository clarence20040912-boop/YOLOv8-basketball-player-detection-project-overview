"""
训练动作分类器
基于姿态关键点特征的动作分类模型
可替代/增强基于规则的动作识别
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import argparse


class ActionClassifierNet(nn.Module):
    """
    基于关键点特征的动作分类网络
    输入: 17个关键点 x 3 (x, y, confidence) = 51维特征
    输出: 8类动作
    """
    
    ACTION_CLASSES = [
        "shooting", "dribbling", "passing", "dunking",
        "blocking", "rebounding", "running", "standing"
    ]
    
    def __init__(self, input_dim: int = 51, num_classes: int = 8, 
                 hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PoseActionDataset(Dataset):
    """姿态-动作数据集"""
    
    def __init__(self, keypoints: np.ndarray, labels: np.ndarray):
        """
        Args:
            keypoints: 关键点数据 (N, 17, 3)
            labels: 动作标签 (N,)
        """
        self.keypoints = torch.FloatTensor(keypoints.reshape(-1, 51))
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.keypoints[idx], self.labels[idx]


def generate_synthetic_data(num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成训练数据（用于演示/初始训练）
    实际项目中应使用真实标注数据替换
    
    Returns:
        keypoints: (N, 17, 3)
        labels: (N,)
    """
    print("📊 生成合成训练数据...")
    
    keypoints_list = []
    labels_list = []
    
    for _ in range(num_samples):
        label = np.random.randint(0, 8)
        kps = np.random.randn(17, 3) * 0.1  # 基础噪声
        
        # 根据动作类别调整关键点模式
        if label == 0:  # shooting - 手高举
            kps[9, 1] -= 0.8   # right_wrist 高
            kps[10, 1] -= 0.8  # left_wrist 高
            kps[7, 1] -= 0.5   # elbow
        elif label == 1:  # dribbling - 手在下方
            kps[9, 1] += 0.6   # wrist 低
            kps[10, 1] += 0.6
        elif label == 2:  # passing - 手在胸前展开
            kps[9, 0] -= 0.5   # 手臂展开
            kps[10, 0] += 0.5
        elif label == 3:  # dunking - 单手极高
            kps[9, 1] -= 1.2
            kps[0, 1] -= 0.3   # 身体跳起
        elif label == 4:  # blocking - 双手展开高举
            kps[9, 0] -= 0.7
            kps[10, 0] += 0.7
            kps[9, 1] -= 0.5
            kps[10, 1] -= 0.5
        elif label == 5:  # rebounding - 双手高举
            kps[9, 1] -= 0.9
            kps[10, 1] -= 0.9
        elif label == 6:  # running - 腿部交叉
            kps[15, 0] += 0.3
            kps[16, 0] -= 0.3
        # label == 7: standing - 保持默认
        
        kps[:, 2] = np.random.uniform(0.7, 1.0, 17)  # 置信度
        kps += np.random.randn(17, 3) * 0.05  # 添加噪声
        
        keypoints_list.append(kps)
        labels_list.append(label)
    
    return np.array(keypoints_list), np.array(labels_list)


def train_action_classifier(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = "models/action_classifier.pth",
    use_synthetic: bool = True
):
    """训练动作分类器"""
    
    print("=" * 60)
    print("🏀 训练动作分类器")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    
    # 准备数据
    if use_synthetic:
        keypoints, labels = generate_synthetic_data(5000)
    else:
        # 从真实数据加载
        # keypoints = np.load("data/action_keypoints.npy")
        # labels = np.load("data/action_labels.npy")
        raise NotImplementedError("请准备真实数据集")
    
    # 划分训练/验证集
    split = int(len(labels) * 0.8)
    train_dataset = PoseActionDataset(keypoints[:split], labels[:split])
    val_dataset = PoseActionDataset(keypoints[split:], labels[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = ActionClassifierNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_kps, batch_labels in train_loader:
            batch_kps = batch_kps.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_kps)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        train_acc = correct / total
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_kps, batch_labels in val_loader:
                batch_kps = batch_kps.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_kps)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {train_loss/len(train_loader):.4f} "
                  f"Train Acc: {train_acc:.4f} "
                  f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': ActionClassifierNet.ACTION_CLASSES,
                'best_acc': best_acc,
                'epoch': epoch
            }, save_path)
    
    print(f"\n✅ 训练完成! 最佳验证准确率: {best_acc:.4f}")
    print(f"  模型已保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="models/action_classifier.pth")
    args = parser.parse_args()
    
    train_action_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )

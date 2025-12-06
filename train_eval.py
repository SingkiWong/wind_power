"""
Training and Evaluation Script for CCP Wind Power Prediction System
CCP系统训练与评估脚本

功能：
1. 模型训练（支持多种骨干网络）
2. 评估指标计算
3. 可视化分析
4. 模型保存与加载
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import json
import os

from ccp_framework import CCPSystem, CCPConfig, create_ccp_system


class WindPowerDataset(Dataset):
    """
    风电功率预测数据集
    
    支持：
    - SCADA数据格式
    - 多变量时间序列
    - 滑动窗口采样
    """
    
    def __init__(
        self,
        data: np.ndarray,
        input_len: int = 96,
        output_len: int = 24,
        stride: int = 1
    ):
        """
        Args:
            data: [time_steps, num_features] 原始数据
            input_len: 输入序列长度
            output_len: 输出序列长度
            stride: 采样步长
        """
        self.data = torch.FloatTensor(data)
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride
        
        # 计算有效样本数
        total_len = input_len + output_len
        self.num_samples = (len(data) - total_len) // stride + 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        mid = start + self.input_len
        end = mid + self.output_len
        
        x = self.data[start:mid]
        y = self.data[mid:end]
        
        # 假设第一列是功率，第二列是风速，第三列是风向
        wind_speed = self.data[start:mid, 1].mean() if self.data.shape[1] > 1 else torch.tensor(10.0)
        wind_direction = self.data[mid-1, 2] if self.data.shape[1] > 2 else torch.tensor(270.0)
        
        return {
            'x': x,
            'y': y,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction
        }


def generate_synthetic_data(
    num_samples: int = 10000,
    num_features: int = 5,
    rated_power: float = 2.0
) -> np.ndarray:
    """
    生成模拟风电数据
    
    特征：[功率, 风速, 风向, 温度, 气压]
    """
    t = np.arange(num_samples)
    
    # 风速：带有日周期和随机波动
    wind_speed = 8 + 4 * np.sin(2 * np.pi * t / 144) + np.random.randn(num_samples) * 2
    wind_speed = np.clip(wind_speed, 0, 30)
    
    # 风向：缓慢变化
    wind_direction = 270 + 30 * np.sin(2 * np.pi * t / 1000) + np.random.randn(num_samples) * 10
    wind_direction = wind_direction % 360
    
    # 温度
    temperature = 15 + 5 * np.sin(2 * np.pi * t / 144) + np.random.randn(num_samples) * 2
    
    # 气压
    pressure = 1013 + 5 * np.sin(2 * np.pi * t / 500) + np.random.randn(num_samples) * 3
    
    # 功率：基于物理模型 + 噪声
    power = np.zeros(num_samples)
    for i in range(num_samples):
        ws = wind_speed[i]
        if ws < 3:
            power[i] = 0
        elif ws < 12:
            power[i] = rated_power * ((ws - 3) / 9) ** 3
        elif ws < 25:
            power[i] = rated_power
        else:
            power[i] = 0
        power[i] += np.random.randn() * 0.1  # 添加噪声
    power = np.clip(power, 0, rated_power)
    
    # 组合数据
    data = np.column_stack([power, wind_speed, wind_direction, temperature, pressure])
    
    return data


class Trainer:
    """
    CCP系统训练器
    """
    
    def __init__(
        self,
        model: CCPSystem,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, current_epoch: Optional[int] = None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_data_loss = 0
        total_physics_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            wind_speed = batch['wind_speed'].to(self.device)
            wind_direction = batch['wind_direction'].to(self.device)
            causal_graph = batch.get('causal_graph')

            # 扩展wind_speed到序列长度
            if wind_speed.dim() == 1:
                wind_speed = wind_speed.unsqueeze(1).expand(-1, y.shape[1])

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(
                x,
                wind_direction,
                return_attention=causal_graph is not None
            )

            # 计算损失
            loss, loss_dict = self.model.compute_loss(
                outputs,
                y,
                wind_speed,
                causal_graph=causal_graph,
                current_epoch=current_epoch
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss_dict['total']
            total_data_loss += loss_dict['data']
            total_physics_loss += loss_dict['physics']
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'data_loss': total_data_loss / num_batches,
            'physics_loss': total_physics_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self, current_epoch: Optional[int] = None) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            wind_speed = batch['wind_speed'].to(self.device)
            wind_direction = batch['wind_direction'].to(self.device)
            causal_graph = batch.get('causal_graph')

            if wind_speed.dim() == 1:
                wind_speed = wind_speed.unsqueeze(1).expand(-1, y.shape[1])

            outputs = self.model(
                x,
                wind_direction,
                return_attention=causal_graph is not None
            )
            loss, _ = self.model.compute_loss(
                outputs,
                y,
                wind_speed,
                causal_graph=causal_graph,
                current_epoch=current_epoch
            )
            
            total_loss += loss.item()
            all_preds.append(outputs['predictions'].cpu())
            all_targets.append(y.cpu())
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        metrics = compute_metrics(preds, targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        完整训练流程
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(current_epoch=epoch)

            # 验证
            val_metrics = self.validate(current_epoch=epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['total_loss'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
            
            # 打印进度
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
                  f"Val RMSE: {val_metrics.get('rmse', 0):.4f} | "
                  f"Time: {elapsed:.1f}s")
            
            # 早停检查
            if val_metrics and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print("-" * 60)
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        
        return self.history
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config.__dict__,
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {path}")


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    计算评估指标
    
    指标：
    - MSE: 均方误差
    - RMSE: 均方根误差
    - MAE: 平均绝对误差
    - MAPE: 平均绝对百分比误差
    - R²: 决定系数
    """
    # 展平
    pred = predictions.flatten().numpy()
    target = targets.flatten().numpy()
    
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(pred - target))
    
    # MAPE (避免除零)
    mask = target != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100
    else:
        mape = 0.0
    
    # R²
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def run_experiment(
    backbone: str = "mamba",
    use_kan: bool = True,
    num_epochs: int = 50,
    batch_size: int = 32
):
    """
    运行完整实验
    """
    print("=" * 70)
    print(f"Running Experiment: {backbone.upper()} + {'KAN' if use_kan else 'MLP'}")
    print("=" * 70)
    
    # 生成数据
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(num_samples=5000, num_features=5)
    
    # 划分数据集
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 创建数据加载器
    train_dataset = WindPowerDataset(train_data, input_len=96, output_len=24)
    val_dataset = WindPowerDataset(val_data, input_len=96, output_len=24)
    test_dataset = WindPowerDataset(test_data, input_len=96, output_len=24)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型
    print("\n2. Creating CCP model...")
    model = create_ccp_system(
        backbone=backbone,
        use_kan=use_kan,
        input_len=96,
        output_len=24,
        num_features=5
    )
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {params:,}")
    
    # 训练
    print("\n3. Training...")
    trainer = Trainer(model, train_loader, val_loader, lr=1e-3)
    history = trainer.train(num_epochs=num_epochs, early_stopping_patience=10)
    
    # 测试评估
    print("\n4. Evaluating on test set...")
    model.eval()
    all_preds = []
    all_targets = []
    
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            wind_direction = batch['wind_direction'].to(device)
            
            outputs = model(x, wind_direction)
            all_preds.append(outputs['predictions'].cpu())
            all_targets.append(y.cpu())
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(preds, targets)
    
    print("\n5. Test Results:")
    print(f"   MSE:  {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE:  {metrics['mae']:.4f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   R²:   {metrics['r2']:.4f}")
    
    return {
        'backbone': backbone,
        'use_kan': use_kan,
        'params': params,
        'metrics': metrics,
        'history': history
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CCP System Training & Evaluation")
    print("=" * 70)
    
    # 运行不同配置的实验
    results = []
    
    # 测试Mamba + KAN（推荐配置）
    result = run_experiment(backbone="mamba", use_kan=True, num_epochs=20)
    results.append(result)
    
    # 可以添加更多实验配置...
    # result = run_experiment(backbone="ttm", use_kan=True, num_epochs=20)
    # results.append(result)
    
    print("\n" + "=" * 70)
    print("Experiment Summary")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['backbone'].upper()} + {'KAN' if r['use_kan'] else 'MLP'}:")
        print(f"  Parameters: {r['params']:,}")
        print(f"  Test RMSE: {r['metrics']['rmse']:.4f}")
        print(f"  Test R²: {r['metrics']['r2']:.4f}")

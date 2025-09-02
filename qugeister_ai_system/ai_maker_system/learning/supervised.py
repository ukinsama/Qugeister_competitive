#!/usr/bin/env python3
"""
教師あり学習システムモジュール
"""

import torch
import torch.nn as nn
import random
import numpy as np
from typing import Dict, List, Tuple
from ..core.base_modules import LearningSystem
from ..core.game_state import LearningConfig
from ..core.data_processor import DataProcessor


class SupervisedLearning(LearningSystem):
    """教師あり学習システム"""
    
    def __init__(self):
        self.training_history = []
    
    def train(
        self, 
        model: nn.Module, 
        training_data: List[Dict], 
        config: LearningConfig
    ) -> nn.Module:
        """教師あり学習を実行"""
        print(f"📚 教師あり学習開始: {len(training_data)}件のデータ")
        print(f"🔧 設定: バッチサイズ={config.batch_size}, 学習率={config.learning_rate}")
        
        device = torch.device(config.device)
        model.to(device)
        
        # オプティマイザとロス関数
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # データ分割
        n_val = int(len(training_data) * config.validation_split)
        val_data = training_data[:n_val]
        train_data = training_data[n_val:]
        
        print(f"📊 データ分割: 学習用={len(train_data)}件, 検証用={len(val_data)}件")
        
        model.train()
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(config.supervised_epochs):
            random.shuffle(train_data)
            
            total_loss = 0
            correct = 0
            total = 0
            
            # 学習フェーズ
            for i in range(0, len(train_data), config.batch_size):
                batch = train_data[i:i + config.batch_size]
                inputs, labels = self._prepare_batch(batch)
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total if total > 0 else 0.0
            avg_batches = max(1, len(train_data) // config.batch_size)
            train_loss = total_loss / avg_batches
            
            # 検証フェーズ
            val_acc, val_loss = self._validate(model, val_data, config.batch_size, device, criterion)
            
            # 履歴記録
            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            
            print(f"Epoch {epoch+1}/{config.supervised_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}%")
            
            # Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print("🎯 新しいベストモデル！")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"⏹️ Early Stopping: {patience_counter}エポック改善なし")
                    break
            
            # 学習率調整
            if epoch > 0 and epoch % 30 == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.8
                print(f"📉 学習率調整: {param_group['lr']:.6f}")
        
        print(f"🎉 教師あり学習完了! ベスト検証精度: {best_val_acc:.1f}%")
        return model
    
    def evaluate(self, model: nn.Module, test_data: List[Dict]) -> Dict[str, float]:
        """モデル評価"""
        device = next(model.parameters()).device
        model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(test_data), 32):  # バッチサイズ32で評価
                batch = test_data[i:i + 32]
                inputs, labels = self._prepare_batch(batch)
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(1, len(test_data) // 32)
        
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "total_samples": total
        }
    
    def _prepare_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """バッチデータ準備"""
        inputs = []
        labels = []
        
        for sample in batch:
            tensor = DataProcessor.prepare_7channel_tensor(
                board=sample["board"],
                player=sample["player"],
                my_pieces=sample["my_pieces"],
                turn=sample["turn"]
            )
            inputs.append(tensor.squeeze(0))
            
            # ラベル生成（善玉/悪玉分類）
            if sample.get("true_labels"):
                first_enemy_type = list(sample["true_labels"].values())[0]
                label = 0 if first_enemy_type == "good" else 1
            else:
                label = 0
            labels.append(label)
        
        batch_tensor = torch.stack(inputs)
        label_tensor = torch.LongTensor(labels)
        
        return batch_tensor, label_tensor
    
    def _validate(
        self, 
        model: nn.Module, 
        val_data: List[Dict], 
        batch_size: int,
        device: torch.device,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """検証実行"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                inputs, labels = self._prepare_batch(batch)
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        model.train()
        accuracy = 100 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(1, len(val_data) // batch_size)
        
        return accuracy, avg_loss
    
    def get_training_history(self) -> List[Dict]:
        """学習履歴を取得"""
        return self.training_history
    
    def save_model(self, model: nn.Module, filepath: str):
        """モデル保存"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': self.training_history,
            'model_class': model.__class__.__name__
        }, filepath)
        print(f"💾 モデル保存: {filepath}")
    
    def load_model(self, model: nn.Module, filepath: str) -> nn.Module:
        """モデル読み込み"""
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"📁 モデル読み込み: {filepath}")
        return model
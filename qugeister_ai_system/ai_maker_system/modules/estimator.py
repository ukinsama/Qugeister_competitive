#!/usr/bin/env python3
"""
敵駒推定器モジュール - CQCNN学習機能付き
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple
from collections import deque

from ..core.base_modules import PieceEstimator, CQCNNModel
from ..core.data_processor import DataProcessor
from ..core.game_state import LearningConfig


class CQCNNEstimator(PieceEstimator):
    """CQCNN推定器（学習機能付き）"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.model = CQCNNModel(n_qubits, n_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 強化学習用
        self.memory = deque(maxlen=10000)
        self.target_model = CQCNNModel(n_qubits, n_layers).to(self.device)
        self.update_target_model()
    
    def estimate_pieces(self, tensor: torch.Tensor) -> torch.Tensor:
        """駒の種類を推定"""
        self.model.eval()
        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
        return probabilities
    
    def train_estimator(self, training_data: List[Dict], config: LearningConfig):
        """推定器を訓練"""
        if hasattr(config, 'learning_mode') and config.learning_mode == 'reinforcement':
            self.train_reinforcement(training_data, config)
        else:
            self.train_supervised(training_data, config)
    
    def train_supervised(self, training_data: List[Dict], config: LearningConfig) -> None:
        """教師あり学習"""
        print(f"📚 CQCNN教師あり学習開始: {len(training_data)}件のデータ")
        print(f"🔧 設定: バッチサイズ={config.batch_size}, 学習率={config.learning_rate}")
        print(f"💻 デバイス: {self.device}")
        
        # データ分割
        n_val = int(len(training_data) * config.validation_split)
        val_data = training_data[:n_val]
        train_data = training_data[n_val:]
        
        print(f"📊 データ分割: 学習用={len(train_data)}件, 検証用={len(val_data)}件")
        
        self.model.train()
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(config.supervised_epochs):
            random.shuffle(train_data)
            
            total_loss = 0
            correct = 0
            total = 0
            
            for i in range(0, len(train_data), config.batch_size):
                batch = train_data[i:i + config.batch_size]
                inputs, labels = self._prepare_supervised_batch(batch)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total if total > 0 else 0.0
            avg_batches = max(1, len(train_data) // config.batch_size)
            train_loss = total_loss / avg_batches
            
            val_acc, val_loss = self._validate_supervised(val_data, config.batch_size)
            
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
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print("🎯 新しいベストモデル！")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"⏹️ Early Stopping: {patience_counter}エポック改善なし")
                    break
        
        self.is_trained = True
        print(f"🎉 教師あり学習完了! ベスト検証精度: {best_val_acc:.1f}%")
    
    def _prepare_supervised_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """教師あり学習用バッチデータ準備"""
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
            
            if sample.get("true_labels"):
                first_enemy_type = list(sample["true_labels"].values())[0]
                label = 0 if first_enemy_type == "good" else 1
            else:
                label = 0
            labels.append(label)
        
        batch_tensor = torch.stack(inputs)
        label_tensor = torch.LongTensor(labels)
        
        return batch_tensor, label_tensor
    
    def _validate_supervised(self, val_data: List[Dict], batch_size: int) -> Tuple[float, float]:
        """教師あり学習の検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                inputs, labels = self._prepare_supervised_batch(batch)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / max(1, len(val_data) // batch_size)
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        self.model.train()
        return accuracy, avg_loss
    
    def update_target_model(self):
        """ターゲットモデル更新（強化学習用）"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_name(self) -> str:
        return "CQCNN推定器"
    
    def get_config(self) -> Dict[str, any]:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "is_trained": self.is_trained,
            "device": str(self.device)
        }
    
    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.estimate_pieces(tensor)


class SimpleEstimator(PieceEstimator):
    """シンプルな推定器（ランダム）"""
    
    def estimate_pieces(self, tensor: torch.Tensor) -> torch.Tensor:
        """ランダムに推定"""
        batch_size = tensor.size(0)
        # ランダムな確率を返す
        return torch.rand(batch_size, 2)
    
    def train_estimator(self, training_data: List, config: Dict):
        """訓練は不要"""
        print("SimpleEstimator: 訓練不要")
    
    def get_name(self) -> str:
        return "シンプル推定器"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "random"}
    
    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.estimate_pieces(tensor)


# エスティメーターファクトリ
class EstimatorFactory:
    """推定器ファクトリ"""
    
    @staticmethod
    def create_estimator(estimator_type: str, **kwargs) -> PieceEstimator:
        """推定器を作成"""
        if estimator_type == "cqcnn":
            n_qubits = kwargs.get("n_qubits", 6)
            n_layers = kwargs.get("n_layers", 3)
            return CQCNNEstimator(n_qubits, n_layers)
        elif estimator_type == "simple":
            return SimpleEstimator()
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
    
    @staticmethod
    def get_available_estimators() -> list:
        """利用可能な推定器一覧"""
        return ["cqcnn", "simple"]
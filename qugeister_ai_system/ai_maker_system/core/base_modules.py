#!/usr/bin/env python3
"""
基本モジュール定義 - 抽象基底クラスとCQCNNモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any


class CQCNNModel(nn.Module):
    """Classical-Quantum CNN モデル（量子回路風処理）"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical CNN部分
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Quantum-inspired部分
        self.quantum_dim = n_qubits * n_layers
        self.quantum_linear = None
        self.quantum_layers = nn.ModuleList([
            nn.Linear(self.quantum_dim, self.quantum_dim) 
            for _ in range(n_layers)
        ])
        
        # 出力層
        self.classifier = nn.Sequential(
            nn.Linear(self.quantum_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),  # 善玉/悪玉の2クラス
        )
        
        # 入力サイズを計算
        self._initialize_linear_layers()
    
    def _initialize_linear_layers(self):
        """Linear層の入力サイズを動的に計算"""
        dummy_input = torch.randn(1, 7, 6, 6)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            
            flattened_size = x.view(x.size(0), -1).size(1)
            self.quantum_linear = nn.Linear(flattened_size, self.quantum_dim)
    
    def forward(self, x):
        # Classical CNN処理
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Quantum-inspired処理
        x = F.relu(self.quantum_linear(x))
        
        for quantum_layer in self.quantum_layers:
            x_new = quantum_layer(x)
            x = F.normalize(x_new + x, dim=1)  # 残差接続 + 正規化
        
        # 分類
        output = self.classifier(x)
        return output


# ========== 抽象基底クラス群 ==========

class AIModule(ABC):
    """AI機能モジュールの基底クラス"""
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """メイン処理"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """モジュール名を取得"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """設定を取得"""
        pass


class PlacementStrategy(AIModule):
    """初期配置戦略の基底クラス"""
    
    @abstractmethod
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        pass


class PieceEstimator(AIModule):
    """敵駒推定器の基底クラス"""
    
    @abstractmethod  
    def estimate_pieces(self, tensor: torch.Tensor) -> torch.Tensor:
        """駒の種類を推定"""
        pass
    
    @abstractmethod
    def train_estimator(self, training_data: List[Tuple], config: Dict):
        """推定器を訓練"""
        pass


class RewardFunction(AIModule):
    """報酬関数の基底クラス"""
    
    @abstractmethod
    def calculate_reward(
        self, 
        state_before: 'GameState', 
        action: Tuple, 
        state_after: 'GameState',
        player: str
    ) -> float:
        """報酬を計算"""
        pass


class QMapGenerator(AIModule):
    """Q値マップ生成器の基底クラス"""
    
    @abstractmethod
    def generate_qmap(
        self, 
        tensor: torch.Tensor, 
        valid_actions: List[Tuple]
    ) -> Dict[Tuple, float]:
        """Q値マップを生成"""
        pass


class ActionSelector(AIModule):
    """行動選択器の基底クラス"""
    
    @abstractmethod
    def select_action(
        self, 
        qmap: Dict[Tuple, float], 
        valid_actions: List[Tuple],
        epsilon: float = 0.0
    ) -> Tuple:
        """行動を選択"""
        pass


class LearningSystem(ABC):
    """学習システムの基底クラス"""
    
    @abstractmethod
    def train(
        self, 
        model: nn.Module, 
        training_data: List, 
        config: Dict
    ) -> nn.Module:
        """モデルを訓練"""
        pass
    
    @abstractmethod
    def evaluate(self, model: nn.Module, test_data: List) -> Dict[str, float]:
        """モデルを評価"""
        pass
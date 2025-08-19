#!/usr/bin/env python3
"""
敵駒推定器の共通インターフェース
教師あり学習、強化学習、その他の手法を統一的に扱うためのインターフェース
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime


# ================================================================================
# 学習方式の列挙型
# ================================================================================

class LearningMethod(Enum):
    """学習方式"""
    SUPERVISED = "supervised"          # 教師あり学習
    REINFORCEMENT = "reinforcement"    # 強化学習
    SELF_PLAY = "self_play"           # 自己対戦学習
    HYBRID = "hybrid"                  # ハイブリッド
    PRETRAINED = "pretrained"          # 事前学習済み
    RANDOM = "random"                  # ランダム（ベースライン）


# ================================================================================
# 学習設定
# ================================================================================

@dataclass
class EstimatorConfig:
    """推定器の設定"""
    # 共通設定
    n_qubits: int = 8                 # 量子ビット数
    n_layers: int = 3                 # 量子回路の層数
    learning_rate: float = 0.001      # 学習率
    batch_size: int = 32              # バッチサイズ
    device: str = "cpu"               # 計算デバイス
    
    # 教師あり学習用
    epochs: int = 100                 # エポック数
    validation_split: float = 0.2     # 検証データの割合
    
    # 強化学習用
    epsilon_start: float = 1.0        # 初期探索率
    epsilon_end: float = 0.01         # 最終探索率
    epsilon_decay: float = 0.995      # 探索率の減衰
    gamma: float = 0.95               # 割引率
    buffer_size: int = 10000          # リプレイバッファサイズ
    
    # 自己対戦用
    self_play_games: int = 100        # 自己対戦ゲーム数
    update_interval: int = 10         # モデル更新間隔
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'device': self.device,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'buffer_size': self.buffer_size,
            'self_play_games': self.self_play_games,
            'update_interval': self.update_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EstimatorConfig':
        """辞書から作成"""
        return cls(**data)


# ================================================================================
# 学習データ形式
# ================================================================================

@dataclass
class TrainingData:
    """学習データ"""
    board_state: np.ndarray           # ボード状態
    enemy_positions: List[Tuple[int, int]]  # 敵駒位置
    true_types: Dict[Tuple[int, int], str]  # 真の駒タイプ
    player_id: str                    # プレイヤーID
    
    # オプション（強化学習用）
    reward: Optional[float] = None
    next_state: Optional[np.ndarray] = None
    action: Optional[Tuple] = None
    done: Optional[bool] = False


@dataclass
class EstimationResult:
    """推定結果"""
    estimations: Dict[Tuple[int, int], Dict[str, float]]  # 位置ごとの確率分布
    confidence: float                  # 全体的な確信度
    computation_time: float            # 計算時間（ミリ秒）
    method_used: LearningMethod       # 使用した学習方式


# ================================================================================
# 共通インターフェース
# ================================================================================

class PieceEstimatorInterface(ABC):
    """
    敵駒推定器の共通インターフェース
    すべての推定器はこのインターフェースを実装する
    """
    
    def __init__(self, config: EstimatorConfig = None):
        """
        初期化
        
        Args:
            config: 推定器の設定
        """
        self.config = config or EstimatorConfig()
        self.learning_method = LearningMethod.RANDOM
        self.is_trained = False
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'rewards': [],
            'episodes': []
        }
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'total_training_time': 0.0,
            'total_samples_seen': 0
        }
    
    @abstractmethod
    def estimate(self, 
                board: np.ndarray,
                enemy_positions: List[Tuple[int, int]],
                player_id: str,
                return_confidence: bool = False) -> Union[Dict, EstimationResult]:
        """
        敵駒のタイプを推定
        
        Args:
            board: ボード状態 (6x5のnumpy配列)
            enemy_positions: 敵駒の位置リスト
            player_id: プレイヤーID ("A" or "B")
            return_confidence: 詳細な結果を返すかどうか
            
        Returns:
            Dict または EstimationResult: 各位置の駒タイプ確率分布
        """
        pass
    
    @abstractmethod
    def train(self, 
             training_data: Union[List[TrainingData], TrainingData],
             validation_data: Optional[List[TrainingData]] = None) -> Dict[str, float]:
        """
        モデルを学習
        
        Args:
            training_data: 学習データ
            validation_data: 検証データ（オプション）
            
        Returns:
            学習結果の統計情報
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                test_data: List[TrainingData]) -> Dict[str, float]:
        """
        モデルを評価
        
        Args:
            test_data: テストデータ
            
        Returns:
            評価指標（accuracy, loss, etc.）
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        モデルを保存
        
        Args:
            filepath: 保存先パス
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        モデルを読み込み
        
        Args:
            filepath: 読み込み元パス
        """
        pass
    
    def get_estimator_name(self) -> str:
        """推定器の名前を取得"""
        return f"{self.__class__.__name__}({self.learning_method.value})"
    
    def get_config(self) -> EstimatorConfig:
        """設定を取得"""
        return self.config
    
    def set_config(self, config: EstimatorConfig) -> None:
        """設定を更新"""
        self.config = config
    
    def is_ready(self) -> bool:
        """推定可能な状態かどうか"""
        return self.is_trained or self.learning_method == LearningMethod.RANDOM
    
    def get_training_history(self) -> Dict:
        """学習履歴を取得"""
        return self.training_history
    
    def get_metadata(self) -> Dict:
        """メタデータを取得"""
        return self.metadata
    
    def reset(self) -> None:
        """状態をリセット"""
        self.is_trained = False
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'rewards': [],
            'episodes': []
        }
        self.metadata['total_training_time'] = 0.0
        self.metadata['total_samples_seen'] = 0


# ================================================================================
# 基底実装クラス
# ================================================================================

class BaseCQCNNEstimator(PieceEstimatorInterface):
    """
    CQCNN推定器の基底クラス
    共通機能を実装
    """
    
    def __init__(self, config: EstimatorConfig = None):
        super().__init__(config)
        self.device = torch.device(self.config.device)
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # 駒タイプのマッピング
        self.piece_types = ["P", "K", "Q", "R", "B", "N"]
        self.piece_to_idx = {p: i for i, p in enumerate(self.piece_types)}
        self.idx_to_piece = {i: p for p, i in self.piece_to_idx.items()}
    
    def _prepare_input(self, 
                      board: np.ndarray,
                      position: Tuple[int, int]) -> torch.Tensor:
        """
        入力データを準備
        
        Args:
            board: ボード状態
            position: 対象位置
            
        Returns:
            モデル入力用テンソル
        """
        # 局所的な5x5領域を抽出
        x, y = position
        local_board = np.zeros((5, 5))
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 5 and 0 <= ny < 6:
                    local_board[dx+2, dy+2] = board[ny, nx]
        
        # テンソルに変換
        tensor = torch.tensor(local_board, dtype=torch.float32)
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def _calculate_confidence(self, 
                            probabilities: Dict[str, float]) -> float:
        """
        確信度を計算
        
        Args:
            probabilities: 確率分布
            
        Returns:
            確信度（0-1）
        """
        if not probabilities:
            return 0.0
        
        # エントロピーベースの確信度
        entropy = -sum(p * np.log(p + 1e-10) for p in probabilities.values())
        max_entropy = -np.log(1.0 / len(probabilities))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        return confidence
    
    def save_checkpoint(self, filepath: str, **kwargs) -> None:
        """
        チェックポイントを保存
        
        Args:
            filepath: 保存先パス
            **kwargs: 追加で保存する情報
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config.to_dict(),
            'learning_method': self.learning_method.value,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'metadata': self.metadata,
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 チェックポイント保存: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """
        チェックポイントを読み込み
        
        Args:
            filepath: 読み込み元パス
            
        Returns:
            追加情報
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"チェックポイントが見つかりません: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 設定を復元
        self.config = EstimatorConfig.from_dict(checkpoint['config'])
        self.learning_method = LearningMethod(checkpoint['learning_method'])
        self.is_trained = checkpoint['is_trained']
        self.training_history = checkpoint['training_history']
        self.metadata = checkpoint['metadata']
        
        # モデルとオプティマイザを復元
        if self.model and checkpoint['model_state_dict']:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✅ チェックポイント読み込み: {filepath}")
        
        # 追加情報を返す
        return {k: v for k, v in checkpoint.items() 
                if k not in ['model_state_dict', 'optimizer_state_dict', 
                            'config', 'learning_method', 'is_trained', 
                            'training_history', 'metadata']}


# ================================================================================
# 使用例とテストコード
# ================================================================================

def example_usage():
    """使用例"""
    print("=" * 70)
    print("敵駒推定器インターフェースの使用例")
    print("=" * 70)
    
    # 設定作成
    config = EstimatorConfig(
        n_qubits=8,
        n_layers=3,
        learning_rate=0.001,
        batch_size=32
    )
    
    print("\n📋 設定:")
    print(f"  量子ビット数: {config.n_qubits}")
    print(f"  量子回路層数: {config.n_layers}")
    print(f"  学習率: {config.learning_rate}")
    print(f"  バッチサイズ: {config.batch_size}")
    
    # 学習データの例
    training_data = TrainingData(
        board_state=np.random.randint(-1, 2, (6, 5)),
        enemy_positions=[(1, 1), (3, 2)],
        true_types={(1, 1): "Q", (3, 2): "P"},
        player_id="A"
    )
    
    print("\n📊 学習データ形式:")
    print(f"  ボードサイズ: {training_data.board_state.shape}")
    print(f"  敵駒位置: {training_data.enemy_positions}")
    print(f"  真のタイプ: {training_data.true_types}")
    
    # 推定結果の例
    estimation_result = EstimationResult(
        estimations={
            (1, 1): {"P": 0.1, "K": 0.1, "Q": 0.6, "R": 0.1, "B": 0.05, "N": 0.05},
            (3, 2): {"P": 0.7, "K": 0.05, "Q": 0.05, "R": 0.1, "B": 0.05, "N": 0.05}
        },
        confidence=0.85,
        computation_time=15.3,
        method_used=LearningMethod.SUPERVISED
    )
    
    print("\n🎯 推定結果形式:")
    print(f"  推定数: {len(estimation_result.estimations)}")
    print(f"  確信度: {estimation_result.confidence:.2%}")
    print(f"  計算時間: {estimation_result.computation_time:.1f}ms")
    print(f"  使用手法: {estimation_result.method_used.value}")
    
    print("\n✅ インターフェース定義完了！")
    print("\n次のステップ:")
    print("1. SupervisedCQCNNEstimator - 教師あり学習実装")
    print("2. RLCQCNNEstimator - 強化学習実装")
    print("3. HybridCQCNNEstimator - ハイブリッド実装")
    print("4. PretrainedCQCNNEstimator - 事前学習済みモデル")


if __name__ == "__main__":
    example_usage()
#!/usr/bin/env python3
"""
CQCNN対戦システム - モジュラー拡張版
量子回路とCNNを組み合わせた駒推定AIによる対戦システム

ユーザーが自由にモジュールを組み替えて、独自のAIを構築可能
- 初期配置戦略
- 敵駒推定器（量子/古典）
- 行動選択戦略
- Q値マップ生成器
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import json
import os
from datetime import datetime
from enum import Enum

# ================================================================================
# Part 1: 基本インターフェース
# ================================================================================

class ModuleType(Enum):
    """モジュールタイプ"""
    PLACEMENT = "placement"
    ESTIMATOR = "estimator"
    QMAP = "qmap"
    SELECTOR = "selector"


# ================================================================================
# Part 2: 初期配置戦略モジュール
# ================================================================================

class PlacementStrategy(ABC):
    """初期配置戦略の基底クラス"""
    
    @abstractmethod
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """戦略名を取得"""
        pass


class StandardPlacement(PlacementStrategy):
    """標準配置戦略"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            # 前列に悪玉、後列に善玉
            positions = [(0, j) for j in range(4)] + [(1, j) for j in range(4)]
            piece_types = ["bad"] * 4 + ["good"] * 4
        else:
            # 前列に悪玉、後列に善玉
            positions = [(4, j) for j in range(4)] + [(5, j) for j in range(4)]
            piece_types = ["bad"] * 4 + ["good"] * 4
        
        for pos, piece_type in zip(positions, piece_types):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "標準配置"


class DefensivePlacement(PlacementStrategy):
    """守備的配置戦略"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            # 善玉を後方に、悪玉を前方に配置
            positions = [(0, j) for j in range(4)] + [(1, j) for j in range(4)]
            piece_types = ["bad", "bad", "bad", "good"] + ["good", "good", "good", "bad"]
        else:
            positions = [(4, j) for j in range(4)] + [(5, j) for j in range(4)]
            piece_types = ["bad", "bad", "bad", "good"] + ["good", "good", "good", "bad"]
        
        for pos, piece_type in zip(positions, piece_types):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "守備的配置"


class AggressivePlacement(PlacementStrategy):
    """攻撃的配置戦略"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            # 善玉を前方に配置して早期脱出を狙う
            positions = [(0, j) for j in range(4)] + [(1, j) for j in range(4)]
            piece_types = ["good", "good", "bad", "bad"] + ["bad", "bad", "good", "good"]
        else:
            positions = [(4, j) for j in range(4)] + [(5, j) for j in range(4)]
            piece_types = ["good", "good", "bad", "bad"] + ["bad", "bad", "good", "good"]
        
        for pos, piece_type in zip(positions, piece_types):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "攻撃的配置"


class RandomPlacement(PlacementStrategy):
    """ランダム配置戦略"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            positions = [(i, j) for i in range(2) for j in range(4)]
        else:
            positions = [(i+4, j) for i in range(2) for j in range(4)]
        
        piece_types = ["good"] * 4 + ["bad"] * 4
        random.shuffle(piece_types)
        
        for pos, piece_type in zip(positions, piece_types):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "ランダム配置"


# ================================================================================
# Part 3: 敵駒推定器モジュール
# ================================================================================

class EstimatorModule(ABC):
    """敵駒推定器の基底クラス"""
    
    @abstractmethod
    def estimate(self, board: np.ndarray, 
                enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """敵駒を推定"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """推定器名を取得"""
        pass
    
    def train(self, data: List[Dict]) -> None:
        """学習（オプション）"""
        pass


class QuantumCircuitLayer(nn.Module):
    """量子回路層"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.rotation_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        self.entanglement_params = nn.Parameter(
            torch.randn(n_layers, n_qubits - 1) * 0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # 初期状態
        state_real = torch.ones(batch_size, self.n_qubits)
        state_imag = torch.zeros(batch_size, self.n_qubits)
        
        # データエンコーディング
        for i in range(min(self.n_qubits, x.shape[1])):
            angle = x[:, i] * np.pi
            new_state_real = state_real.clone()
            new_state_imag = state_imag.clone()
            new_state_real[:, i] = torch.cos(angle/2)
            new_state_imag[:, i] = torch.sin(angle/2)
            state_real = new_state_real
            state_imag = new_state_imag
        
        # 変分回路
        for layer in range(self.n_layers):
            new_state_real = state_real.clone()
            new_state_imag = state_imag.clone()
            
            # 回転ゲート
            for qubit in range(self.n_qubits):
                rx = self.rotation_params[layer, qubit, 0]
                q_real = state_real[:, qubit]
                q_imag = state_imag[:, qubit]
                
                cos_rx = torch.cos(rx/2)
                sin_rx = torch.sin(rx/2)
                new_q_real = cos_rx * q_real + sin_rx * q_imag
                new_q_imag = cos_rx * q_imag - sin_rx * q_real
                
                new_state_real[:, qubit] = new_q_real
                new_state_imag[:, qubit] = new_q_imag
            
            state_real = new_state_real
            state_imag = new_state_imag
            
            # エンタングルメント
            entangled_real = state_real.clone()
            for i in range(self.n_qubits - 1):
                strength = torch.sigmoid(self.entanglement_params[layer, i])
                avg = (state_real[:, i] + state_real[:, i+1]) / 2
                entangled_real[:, i] = (1 - strength) * state_real[:, i] + strength * avg
                entangled_real[:, i+1] = (1 - strength) * state_real[:, i+1] + strength * avg
            state_real = entangled_real
        
        measurements = torch.sqrt(state_real**2 + state_imag**2)
        return measurements


class CQCNNEstimator(EstimatorModule):
    """CQCNN（量子回路）推定器"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def _build_model(self):
        """CQCNNモデルを構築"""
        class CQCNNModel(nn.Module):
            def __init__(self, n_qubits, n_layers):
                super().__init__()
                
                # CNN特徴抽出
                self.cnn = nn.Sequential(
                    nn.Conv2d(5, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                )
                
                # 次元削減
                self.reduction = nn.Sequential(
                    nn.Linear(64 * 3 * 3, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, n_qubits),
                    nn.Tanh()
                )
                
                # 量子回路
                self.quantum = QuantumCircuitLayer(n_qubits, n_layers)
                
                # 出力層
                self.output = nn.Sequential(
                    nn.Linear(n_qubits, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )
            
            def forward(self, x):
                x = self.cnn(x)
                x = self.reduction(x)
                x = self.quantum(x)
                x = self.output(x)
                return x
        
        return CQCNNModel(self.n_qubits, self.n_layers)
    
    def estimate(self, board: np.ndarray, 
                enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        self.model.eval()
        results = {}
        
        # ボードをテンソルに変換
        board_tensor = self._prepare_board_tensor(board, player)
        
        with torch.no_grad():
            output = self.model(board_tensor)
            probs = F.softmax(output, dim=1)
            
            for i, pos in enumerate(enemy_positions):
                position_factor = (pos[0] / 5.0 + pos[1] / 5.0) / 2.0
                adjusted_probs = probs[0] * (0.8 + 0.2 * position_factor)
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                
                results[pos] = {
                    'good_prob': adjusted_probs[0].item(),
                    'bad_prob': adjusted_probs[1].item(),
                    'confidence': max(adjusted_probs.tolist())
                }
        
        return results
    
    def _prepare_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        tensor = torch.zeros(1, 5, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))
        
        if player == "A":
            tensor[0, 3, 0, 5] = 1.0
            tensor[0, 3, 5, 5] = 1.0
            tensor[0, 4, 0, 0] = 1.0
            tensor[0, 4, 5, 0] = 1.0
        else:
            tensor[0, 3, 0, 0] = 1.0
            tensor[0, 3, 5, 0] = 1.0
            tensor[0, 4, 0, 5] = 1.0
            tensor[0, 4, 5, 5] = 1.0
        
        return tensor
    
    def get_name(self) -> str:
        return f"CQCNN(q={self.n_qubits},l={self.n_layers})"


class SimpleCNNEstimator(EstimatorModule):
    """シンプルなCNN推定器（量子回路なし）"""
    
    def __init__(self):
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def _build_model(self):
        return nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def estimate(self, board: np.ndarray, 
                enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        self.model.eval()
        results = {}
        
        board_tensor = self._prepare_board_tensor(board, player)
        
        with torch.no_grad():
            output = self.model(board_tensor)
            probs = F.softmax(output, dim=1)
            
            for pos in enemy_positions:
                results[pos] = {
                    'good_prob': probs[0, 0].item(),
                    'bad_prob': probs[0, 1].item(),
                    'confidence': max(probs[0].tolist())
                }
        
        return results
    
    def _prepare_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        # CQCNNEstimatorと同じ処理
        tensor = torch.zeros(1, 5, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))
        
        if player == "A":
            tensor[0, 3, 0, 5] = 1.0
            tensor[0, 3, 5, 5] = 1.0
            tensor[0, 4, 0, 0] = 1.0
            tensor[0, 4, 5, 0] = 1.0
        else:
            tensor[0, 3, 0, 0] = 1.0
            tensor[0, 3, 5, 0] = 1.0
            tensor[0, 4, 0, 5] = 1.0
            tensor[0, 4, 5, 5] = 1.0
        
        return tensor
    
    def get_name(self) -> str:
        return "SimpleCNN"


class RandomEstimator(EstimatorModule):
    """ランダム推定器"""
    
    def estimate(self, board: np.ndarray, 
                enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        results = {}
        
        for pos in enemy_positions:
            good_prob = random.random()
            bad_prob = 1 - good_prob
            
            results[pos] = {
                'good_prob': good_prob,
                'bad_prob': bad_prob,
                'confidence': max(good_prob, bad_prob)
            }
        
        return results
    
    def get_name(self) -> str:
        return "ランダム"


# ================================================================================
# Part 4: Q値マップ生成器モジュール
# ================================================================================

class QMapGenerator(ABC):
    """Q値マップ生成器の基底クラス"""
    
    @abstractmethod
    def generate(self, board: np.ndarray, 
                estimations: Dict, 
                my_pieces: Dict,
                player: str) -> np.ndarray:
        """Q値マップを生成"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """生成器名を取得"""
        pass


class SimpleQMapGenerator(QMapGenerator):
    """シンプルなQ値マップ生成器"""
    
    def generate(self, board: np.ndarray, 
                estimations: Dict, 
                my_pieces: Dict,
                player: str) -> np.ndarray:
        q_map = np.zeros((6, 6, 4))  # 4方向
        
        for pos, piece_type in my_pieces.items():
            base_value = 1.0 if piece_type == "good" else 0.5
            
            # 各方向のQ値を設定
            for i, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                if 0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6:
                    # 基本Q値
                    q_value = base_value
                    
                    # 敵駒がいる場合
                    if new_pos in estimations:
                        est = estimations[new_pos]
                        if piece_type == "bad":
                            # 悪玉は敵を取る
                            q_value += est['good_prob'] * 2.0 + est['bad_prob'] * 1.0
                        else:
                            # 善玉は敵を避ける
                            q_value -= est['bad_prob'] * 0.5
                    
                    q_map[pos[0], pos[1], i] = q_value
        
        return q_map
    
    def get_name(self) -> str:
        return "シンプルQ値"


class StrategicQMapGenerator(QMapGenerator):
    """戦略的Q値マップ生成器"""
    
    def generate(self, board: np.ndarray, 
                estimations: Dict, 
                my_pieces: Dict,
                player: str) -> np.ndarray:
        q_map = np.zeros((6, 6, 4))
        
        # 脱出口の位置
        if player == "A":
            escape_positions = [(0, 5), (5, 5)]
        else:
            escape_positions = [(0, 0), (5, 0)]
        
        for pos, piece_type in my_pieces.items():
            for i, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                if not (0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6):
                    continue
                
                q_value = 0.0
                
                if piece_type == "good":
                    # 善玉: 脱出を優先
                    min_dist_before = min(abs(pos[0] - ep[0]) + abs(pos[1] - ep[1]) 
                                         for ep in escape_positions)
                    min_dist_after = min(abs(new_pos[0] - ep[0]) + abs(new_pos[1] - ep[1]) 
                                        for ep in escape_positions)
                    
                    if min_dist_after < min_dist_before:
                        q_value += 3.0
                    
                    if new_pos in escape_positions:
                        q_value += 10.0
                else:
                    # 悪玉: 敵駒撃破を優先
                    if new_pos in estimations:
                        est = estimations[new_pos]
                        q_value += est['good_prob'] * 3.0 + est['bad_prob'] * 1.5
                
                q_map[pos[0], pos[1], i] = q_value
        
        return q_map
    
    def get_name(self) -> str:
        return "戦略的Q値"


# ================================================================================
# Part 5: 行動選択器モジュール
# ================================================================================

class ActionSelector(ABC):
    """行動選択器の基底クラス"""
    
    @abstractmethod
    def select(self, q_map: np.ndarray, 
              legal_moves: List[Tuple],
              temperature: float = 1.0) -> Tuple:
        """行動を選択"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """選択器名を取得"""
        pass


class GreedySelector(ActionSelector):
    """貪欲選択器"""
    
    def select(self, q_map: np.ndarray, 
              legal_moves: List[Tuple],
              temperature: float = 1.0) -> Tuple:
        if not legal_moves:
            return None
        
        best_move = None
        best_value = -float('inf')
        
        for from_pos, to_pos in legal_moves:
            # 方向を計算
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            dir_map = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3}
            if (dx, dy) in dir_map:
                dir_idx = dir_map[(dx, dy)]
                value = q_map[from_pos[0], from_pos[1], dir_idx]
                
                if value > best_value:
                    best_value = value
                    best_move = (from_pos, to_pos)
        
        return best_move or random.choice(legal_moves)
    
    def get_name(self) -> str:
        return "貪欲選択"


class EpsilonGreedySelector(ActionSelector):
    """ε-貪欲選択器"""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
    
    def select(self, q_map: np.ndarray, 
              legal_moves: List[Tuple],
              temperature: float = 1.0) -> Tuple:
        if not legal_moves:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # 貪欲選択
        greedy = GreedySelector()
        return greedy.select(q_map, legal_moves, temperature)
    
    def get_name(self) -> str:
        return f"ε貪欲(ε={self.epsilon})"


class SoftmaxSelector(ActionSelector):
    """ソフトマックス選択器"""
    
    def select(self, q_map: np.ndarray, 
              legal_moves: List[Tuple],
              temperature: float = 1.0) -> Tuple:
        if not legal_moves:
            return None
        
        q_values = []
        dir_map = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3}
        
        for from_pos, to_pos in legal_moves:
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            if (dx, dy) in dir_map:
                dir_idx = dir_map[(dx, dy)]
                value = q_map[from_pos[0], from_pos[1], dir_idx]
            else:
                value = 0.0
            
            q_values.append(value / temperature)
        
        # ソフトマックス確率
        q_tensor = torch.tensor(q_values, dtype=torch.float32)
        probs = F.softmax(q_tensor, dim=0).numpy()
        
        # 確率的に選択
        idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[idx]
    
    def get_name(self) -> str:
        return "ソフトマックス"


# ================================================================================
# Part 6: モジュラーエージェント
# ================================================================================

@dataclass
class AgentConfig:
    """エージェント設定"""
    placement: PlacementStrategy
    estimator: EstimatorModule
    qmap_generator: QMapGenerator
    action_selector: ActionSelector


class ModularAgent:
    """モジュラー型AIエージェント"""
    
    def __init__(self, player_id: str, config: AgentConfig):
        self.player_id = player_id
        self.config = config
        self.name = self._generate_name()
        self.game_history = []
    
    def _generate_name(self) -> str:
        return f"Agent_{self.player_id}[{self.config.placement.get_name()[:3]}+" \
               f"{self.config.estimator.get_name()[:8]}+" \
               f"{self.config.qmap_generator.get_name()[:3]}+" \
               f"{self.config.action_selector.get_name()[:3]}]"
    
    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        return self.config.placement.get_placement(self.player_id)
    
    def get_move(self, game_state, legal_moves: List[Tuple]) -> Tuple:
        """次の手を取得"""
        if not legal_moves:
            return None
        
        try:
            # 敵駒位置を特定
            enemy_positions = self._find_enemy_positions(game_state)
            
            # 敵駒を推定
            estimations = {}
            if enemy_positions:
                estimations = self.config.estimator.estimate(
                    game_state.board,
                    enemy_positions,
                    self.player_id
                )
            
            # 自分の駒情報
            my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
            
            # Q値マップを生成
            q_map = self.config.qmap_generator.generate(
                game_state.board,
                estimations,
                my_pieces,
                self.player_id
            )
            
            # 行動を選択
            move = self.config.action_selector.select(q_map, legal_moves)
            
            # 履歴に記録
            self.game_history.append({
                'turn': game_state.turn,
                'move': move,
                'estimations': len(estimations),
                'q_max': np.max(q_map) if q_map is not None else 0
            })
            
            return move
            
        except Exception as e:
            print(f"⚠️ エラー in {self.name}: {e}")
            return random.choice(legal_moves)
    
    def _find_enemy_positions(self, game_state) -> List[Tuple[int, int]]:
        """敵駒の位置を特定"""
        enemy_val = -1 if self.player_id == "A" else 1
        positions = []
        
        for i in range(game_state.board.shape[0]):
            for j in range(game_state.board.shape[1]):
                if game_state.board[i, j] == enemy_val:
                    positions.append((i, j))
        
        return positions


# ================================================================================
# Part 7: ゲームシステム
# ================================================================================

class GameState:
    """ゲーム状態"""
    def __init__(self):
        self.board = np.zeros((6, 6), dtype=int)
        self.turn = 0
        self.player_a_pieces = {}
        self.player_b_pieces = {}
        self.winner = None
    
    def is_game_over(self):
        return self.winner is not None or self.turn >= 100


class GameEngine:
    """ゲームエンジン"""
    
    def __init__(self):
        self.state = GameState()
        self.move_history = []
    
    def get_legal_moves(self, player: str) -> List[Tuple]:
        """合法手のリストを取得"""
        legal_moves = []
        pieces = self.state.player_a_pieces if player == "A" else self.state.player_b_pieces
        
        for pos, piece_type in pieces.items():
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if self._is_valid_move(pos, new_pos, player):
                    legal_moves.append((pos, new_pos))
        
        return legal_moves
    
    def _is_valid_move(self, from_pos: Tuple, to_pos: Tuple, player: str) -> bool:
        if not (0 <= to_pos[0] < 6 and 0 <= to_pos[1] < 6):
            return False
        
        player_val = 1 if player == "A" else -1
        if self.state.board[to_pos] == player_val:
            return False
        
        return True
    
    def make_move(self, from_pos: Tuple, to_pos: Tuple, player: str):
        """手を実行"""
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        # 敵駒を取る場合
        if self.state.board[to_pos] == enemy_val:
            enemy_pieces = self.state.player_b_pieces if player == "A" else self.state.player_a_pieces
            if to_pos in enemy_pieces:
                del enemy_pieces[to_pos]
        
        # 移動
        self.state.board[from_pos] = 0
        self.state.board[to_pos] = player_val
        
        pieces = self.state.player_a_pieces if player == "A" else self.state.player_b_pieces
        piece_type = pieces.pop(from_pos)
        pieces[to_pos] = piece_type
        
        # 脱出判定
        escape_positions = [(0, 5), (5, 5)] if player == "A" else [(0, 0), (5, 0)]
        if to_pos in escape_positions and piece_type == "good":
            self.state.winner = player
        
        self.state.turn += 1


# ================================================================================
# Part 8: 競技システム
# ================================================================================

class CompetitionRunner:
    """競技実行システム"""
    
    def __init__(self):
        # 利用可能なモジュール
        self.modules = {
            'placement': [
                StandardPlacement(),
                DefensivePlacement(),
                AggressivePlacement(),
                RandomPlacement()
            ],
            'estimator': [
                CQCNNEstimator(n_qubits=4, n_layers=2),
                CQCNNEstimator(n_qubits=6, n_layers=3),
                SimpleCNNEstimator(),
                RandomEstimator()
            ],
            'qmap': [
                SimpleQMapGenerator(),
                StrategicQMapGenerator()
            ],
            'selector': [
                GreedySelector(),
                EpsilonGreedySelector(epsilon=0.1),
                EpsilonGreedySelector(epsilon=0.3),
                SoftmaxSelector()
            ]
        }
        
        self.match_results = []
    
    def show_modules(self):
        """利用可能なモジュールを表示"""
        print("=" * 70)
        print("🎮 CQCNN競技システム - 利用可能なモジュール")
        print("=" * 70)
        
        print("\n【1. 初期配置戦略】")
        for i, module in enumerate(self.modules['placement']):
            print(f"  {i}: {module.get_name()}")
        
        print("\n【2. 敵駒推定器】")
        for i, module in enumerate(self.modules['estimator']):
            print(f"  {i}: {module.get_name()}")
        
        print("\n【3. Q値マップ生成器】")
        for i, module in enumerate(self.modules['qmap']):
            print(f"  {i}: {module.get_name()}")
        
        print("\n【4. 行動選択器】")
        for i, module in enumerate(self.modules['selector']):
            print(f"  {i}: {module.get_name()}")
    
    def create_agent(self, player_id: str, 
                    placement_idx: int = 0,
                    estimator_idx: int = 0,
                    qmap_idx: int = 0,
                    selector_idx: int = 0) -> ModularAgent:
        """指定されたモジュールでエージェントを作成"""
        config = AgentConfig(
            placement=self.modules['placement'][placement_idx],
            estimator=self.modules['estimator'][estimator_idx],
            qmap_generator=self.modules['qmap'][qmap_idx],
            action_selector=self.modules['selector'][selector_idx]
        )
        
        return ModularAgent(player_id, config)
    
    def run_match(self, agent1: ModularAgent, agent2: ModularAgent, verbose: bool = False):
        """1試合を実行"""
        engine = GameEngine()
        
        # 初期配置
        placement1 = agent1.get_initial_placement()
        placement2 = agent2.get_initial_placement()
        
        for pos, piece_type in placement1.items():
            engine.state.board[pos] = 1
            engine.state.player_a_pieces[pos] = piece_type
        
        for pos, piece_type in placement2.items():
            engine.state.board[pos] = -1
            engine.state.player_b_pieces[pos] = piece_type
        
        if verbose:
            print(f"\n対戦: {agent1.name} vs {agent2.name}")
        
        # ゲームループ
        current_player = "A"
        current_agent = agent1
        
        while not engine.state.is_game_over():
            legal_moves = engine.get_legal_moves(current_player)
            
            if not legal_moves:
                break
            
            move = current_agent.get_move(engine.state, legal_moves)
            
            if move:
                engine.make_move(move[0], move[1], current_player)
            
            # 勝者判定
            if engine.state.winner:
                break
            
            # ターン交代
            current_player = "B" if current_player == "A" else "A"
            current_agent = agent2 if current_agent == agent1 else agent1
        
        result = {
            'winner': engine.state.winner or "Draw",
            'turns': engine.state.turn,
            'agent1': agent1.name,
            'agent2': agent2.name
        }
        
        self.match_results.append(result)
        
        if verbose:
            print(f"勝者: {result['winner']}, ターン数: {result['turns']}")
        
        return result
    
    def run_tournament(self, n_agents: int = 4):
        """トーナメントを実行"""
        print("\n🏆 トーナメント開始")
        print("=" * 70)
        
        # ランダムにエージェントを生成
        agents = []
        for i in range(n_agents):
            agent = self.create_agent(
                "A" if i % 2 == 0 else "B",
                placement_idx=random.randint(0, len(self.modules['placement'])-1),
                estimator_idx=random.randint(0, len(self.modules['estimator'])-1),
                qmap_idx=random.randint(0, len(self.modules['qmap'])-1),
                selector_idx=random.randint(0, len(self.modules['selector'])-1)
            )
            agents.append(agent)
            print(f"Agent {i+1}: {agent.name}")
        
        # 総当たり戦
        results = {agent.name: {'wins': 0, 'losses': 0, 'draws': 0} for agent in agents}
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i >= j:
                    continue
                
                # 先手後手を入れ替えて2試合
                for swap in [False, True]:
                    if swap:
                        a1, a2 = agent2, agent1
                    else:
                        a1, a2 = agent1, agent2
                    
                    result = self.run_match(a1, a2)
                    
                    if result['winner'] == "A":
                        results[a1.name]['wins'] += 1
                        results[a2.name]['losses'] += 1
                    elif result['winner'] == "B":
                        results[a1.name]['losses'] += 1
                        results[a2.name]['wins'] += 1
                    else:
                        results[a1.name]['draws'] += 1
                        results[a2.name]['draws'] += 1
        
        # 結果表示
        print("\n📊 トーナメント結果")
        print("=" * 70)
        
        sorted_results = sorted(results.items(), 
                              key=lambda x: (x[1]['wins'], -x[1]['losses']), 
                              reverse=True)
        
        for rank, (name, stats) in enumerate(sorted_results, 1):
            total = stats['wins'] + stats['losses'] + stats['draws']
            win_rate = stats['wins'] / total if total > 0 else 0
            print(f"{rank}位: {name}")
            print(f"    勝:{stats['wins']} 負:{stats['losses']} 分:{stats['draws']} (勝率:{win_rate:.1%})")
        
        return sorted_results[0][0]  # 優勝者を返す


# ================================================================================
# Part 9: メインプログラム
# ================================================================================

def main():
    """メインプログラム"""
    print("=" * 70)
    print("🌟 CQCNN対戦システム - モジュラー拡張版")
    print("=" * 70)
    
    runner = CompetitionRunner()
    
    while True:
        print("\n📋 メニュー:")
        print("1. 利用可能なモジュールを表示")
        print("2. カスタムエージェント対戦")
        print("3. ランダムトーナメント")
        print("4. 最強構成を探索")
        print("5. 終了")
        
        choice = input("\n選択 (1-5): ")
        
        if choice == "1":
            runner.show_modules()
        
        elif choice == "2":
            print("\n🎮 カスタムエージェント対戦")
            runner.show_modules()
            
            print("\n【Agent 1の構成】")
            p1 = int(input("配置戦略 (0-3): "))
            e1 = int(input("推定器 (0-3): "))
            q1 = int(input("Q値生成 (0-1): "))
            s1 = int(input("選択器 (0-3): "))
            
            print("\n【Agent 2の構成】")
            p2 = int(input("配置戦略 (0-3): "))
            e2 = int(input("推定器 (0-3): "))
            q2 = int(input("Q値生成 (0-1): "))
            s2 = int(input("選択器 (0-3): "))
            
            agent1 = runner.create_agent("A", p1, e1, q1, s1)
            agent2 = runner.create_agent("B", p2, e2, q2, s2)
            
            print(f"\n対戦設定完了:")
            print(f"Agent 1: {agent1.name}")
            print(f"Agent 2: {agent2.name}")
            
            result = runner.run_match(agent1, agent2, verbose=True)
        
        elif choice == "3":
            n = int(input("参加エージェント数 (2-8): "))
            winner = runner.run_tournament(n)
            print(f"\n🏆 優勝: {winner}")
        
        elif choice == "4":
            print("\n🔍 最強構成を探索中...")
            best_configs = []
            
            for _ in range(10):  # 10回トーナメント
                winner = runner.run_tournament(6)
                best_configs.append(winner)
            
            from collections import Counter
            counter = Counter(best_configs)
            print("\n📊 最強構成ランキング:")
            for config, count in counter.most_common(3):
                print(f"  {config}: {count}回優勝")
        
        elif choice == "5":
            print("\n👋 終了します")
            break
    
    print("\n" + "=" * 70)
    print("✨ システムの特徴:")
    print("  • モジュラー設計で自由な組み合わせ")
    print("  • 量子回路による高度な推定")
    print("  • 戦略的な行動選択")
    print("  • 競技による最適化")
    print("=" * 70)


if __name__ == "__main__":
    main()

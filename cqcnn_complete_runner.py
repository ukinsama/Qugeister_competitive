#!/usr/bin/env python3
"""
CQCNN競技フレームワーク - 完全版ランナー
すべてのモジュールを統合した実行可能な競技システム
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
import time
import json
from abc import ABC, abstractmethod


# ================================================================================
# 基本インターフェース定義
# ================================================================================

class InitialPlacementStrategy(ABC):
    """初期配置戦略の基底クラス"""
    
    @abstractmethod
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        """初期配置を返す"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """戦略名を返す"""
        pass


class PieceEstimator(ABC):
    """敵駒推定器の基底クラス"""
    
    @abstractmethod
    def estimate(self, board: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player_id: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """敵駒のタイプを推定"""
        pass
    
    @abstractmethod
    def get_estimator_name(self) -> str:
        """推定器名を返す"""
        pass


class QMapGenerator(ABC):
    """Q値マップ生成器の基底クラス"""
    
    @abstractmethod
    def generate(self, board: np.ndarray, piece_estimations: Dict,
                my_pieces: Dict, player_id: str) -> np.ndarray:
        """Q値マップを生成"""
        pass
    
    @abstractmethod
    def get_generator_name(self) -> str:
        """生成器名を返す"""
        pass


class ActionSelector(ABC):
    """行動選択器の基底クラス"""
    
    @abstractmethod
    def select_action(self, q_map: np.ndarray, legal_moves: List) -> Tuple:
        """Q値マップから行動を選択"""
        pass
    
    @abstractmethod
    def get_selector_name(self) -> str:
        """選択器名を返す"""
        pass


# ================================================================================
# 初期配置戦略の実装
# ================================================================================

class StandardPlacement(InitialPlacementStrategy):
    """標準的な初期配置"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        if player_id == "A":
            return {
                (0, 0): "P", (1, 0): "P", (2, 0): "P", (3, 0): "P", (4, 0): "P",
                (0, 1): "B", (1, 1): "N", (2, 1): "K", (3, 1): "R", (4, 1): "Q"
            }
        else:
            return {
                (0, 5): "P", (1, 5): "P", (2, 5): "P", (3, 5): "P", (4, 5): "P",
                (0, 4): "Q", (1, 4): "R", (2, 4): "K", (3, 4): "N", (4, 4): "B"
            }
    
    def get_strategy_name(self) -> str:
        return "標準配置"


class DefensivePlacement(InitialPlacementStrategy):
    """守備的な初期配置"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        if player_id == "A":
            return {
                (0, 0): "P", (1, 0): "B", (2, 0): "P", (3, 0): "B", (4, 0): "P",
                (0, 1): "R", (1, 1): "P", (2, 1): "K", (3, 1): "P", (4, 1): "R"
            }
        else:
            return {
                (0, 5): "P", (1, 5): "B", (2, 5): "P", (3, 5): "B", (4, 5): "P",
                (0, 4): "R", (1, 4): "P", (2, 4): "K", (3, 4): "P", (4, 4): "R"
            }
    
    def get_strategy_name(self) -> str:
        return "守備的配置"


class RandomPlacement(InitialPlacementStrategy):
    """ランダムな初期配置"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        pieces = ["P"] * 5 + ["K", "Q", "R", "B", "N"]
        random.shuffle(pieces)
        
        if player_id == "A":
            positions = [(x, y) for y in range(2) for x in range(5)]
        else:
            positions = [(x, y) for y in range(4, 6) for x in range(5)]
        
        return dict(zip(positions, pieces))
    
    def get_strategy_name(self) -> str:
        return "ランダム配置"


# ================================================================================
# 量子回路層（CQCNN核心部分）
# ================================================================================

class QuantumCircuitLayer(nn.Module):
    """量子回路層 - CQCNNの核心"""
    
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 量子回路パラメータ
        self.rotation_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        self.entanglement_params = nn.Parameter(
            torch.randn(n_layers, n_qubits - 1) * 0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # 入力を量子ビット数に調整
        if x.shape[1] != self.n_qubits:
            fc = nn.Linear(x.shape[1], self.n_qubits).to(x.device)
            x = fc(x)
        
        # 量子状態を初期化（|0>状態）
        quantum_state = torch.zeros(batch_size, self.n_qubits, 2)
        quantum_state[:, :, 0] = 1.0
        
        # 入力エンコーディング
        for i in range(self.n_qubits):
            angle = x[:, i].unsqueeze(1) * np.pi
            quantum_state[:, i, 0] = torch.cos(angle / 2).squeeze()
            quantum_state[:, i, 1] = torch.sin(angle / 2).squeeze()
        
        # パラメータ化量子回路
        for layer in range(self.n_layers):
            # 単一量子ビット回転
            for i in range(self.n_qubits):
                rx = self.rotation_params[layer, i, 0]
                ry = self.rotation_params[layer, i, 1]
                rz = self.rotation_params[layer, i, 2]
                
                # RZ回転
                phase = torch.exp(1j * rz / 2)
                quantum_state[:, i, 0] *= phase
                quantum_state[:, i, 1] *= torch.conj(phase)
                
                # RY回転
                c = torch.cos(ry / 2)
                s = torch.sin(ry / 2)
                temp0 = c * quantum_state[:, i, 0] - s * quantum_state[:, i, 1]
                temp1 = s * quantum_state[:, i, 0] + c * quantum_state[:, i, 1]
                quantum_state[:, i, 0] = temp0
                quantum_state[:, i, 1] = temp1
                
                # RX回転
                c = torch.cos(rx / 2)
                s = torch.sin(rx / 2) * 1j
                temp0 = c * quantum_state[:, i, 0] - s * quantum_state[:, i, 1]
                temp1 = -s * quantum_state[:, i, 0] + c * quantum_state[:, i, 1]
                quantum_state[:, i, 0] = temp0
                quantum_state[:, i, 1] = temp1
            
            # エンタングルメント（CNOT風の操作）
            for i in range(self.n_qubits - 1):
                strength = torch.sigmoid(self.entanglement_params[layer, i])
                control = quantum_state[:, i, 1].abs()
                quantum_state[:, i+1, :] = (1 - strength) * quantum_state[:, i+1, :] + \
                                          strength * control.unsqueeze(1) * quantum_state[:, i+1, :].roll(1, dims=1)
        
        # 測定（期待値）
        measurements = (quantum_state.abs() ** 2)[:, :, 1]
        
        return measurements


# ================================================================================
# 敵駒推定器の実装
# ================================================================================

class SimpleCQCNNEstimator(PieceEstimator):
    """シンプルなCQCNN推定器"""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.quantum_layer = QuantumCircuitLayer(n_qubits, n_layers)
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 25, 6)  # 6種類の駒
    
    def estimate(self, board: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player_id: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        estimations = {}
        
        for pos in enemy_positions:
            # 局所的なボード状態を抽出
            local_board = self._extract_local_board(board, pos)
            
            # CNN特徴抽出
            x = torch.tensor(local_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            x = F.relu(self.conv(x))
            x = x.flatten(start_dim=1)
            
            # 量子回路処理
            quantum_features = self.quantum_layer(x[:, :4])
            
            # 最終予測
            combined = torch.cat([x[:, :8], quantum_features], dim=1)
            fc_layer = nn.Linear(combined.shape[1], 6)
            predictions = F.softmax(fc_layer(combined), dim=1)
            
            piece_types = ["P", "K", "Q", "R", "B", "N"]
            estimations[pos] = {
                piece: float(predictions[0, i])
                for i, piece in enumerate(piece_types)
            }
        
        return estimations
    
    def _extract_local_board(self, board: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
        """位置周辺のボード状態を抽出"""
        x, y = pos
        local = np.zeros((5, 5))
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 5 and 0 <= ny < 6:
                    local[dx+2, dy+2] = board[ny, nx]
        
        return local
    
    def get_estimator_name(self) -> str:
        return "シンプルCQCNN推定器"


class AdvancedCQCNNEstimator(PieceEstimator):
    """高度なCQCNN推定器"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        self.quantum_layer = QuantumCircuitLayer(n_qubits, n_layers)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 25 + n_qubits, 6)
    
    def estimate(self, board: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player_id: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        estimations = {}
        
        for pos in enemy_positions:
            local_board = self._extract_local_board(board, pos)
            
            # 深いCNN特徴抽出
            x = torch.tensor(local_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.flatten(start_dim=1)
            
            # 量子回路処理
            quantum_features = self.quantum_layer(x[:, :6])
            
            # 組み合わせて予測
            combined = torch.cat([x[:, :32], quantum_features], dim=1)
            fc_layer = nn.Linear(combined.shape[1], 6)
            predictions = F.softmax(fc_layer(combined), dim=1)
            
            piece_types = ["P", "K", "Q", "R", "B", "N"]
            estimations[pos] = {
                piece: float(predictions[0, i])
                for i, piece in enumerate(piece_types)
            }
        
        return estimations
    
    def _extract_local_board(self, board: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
        x, y = pos
        local = np.zeros((5, 5))
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 5 and 0 <= ny < 6:
                    local[dx+2, dy+2] = board[ny, nx]
        
        return local
    
    def get_estimator_name(self) -> str:
        return "高度CQCNN推定器"


class RandomEstimator(PieceEstimator):
    """ランダム推定器（ベースライン）"""
    
    def estimate(self, board: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player_id: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        estimations = {}
        piece_types = ["P", "K", "Q", "R", "B", "N"]
        
        for pos in enemy_positions:
            probs = np.random.dirichlet([1] * 6)
            estimations[pos] = {
                piece: float(probs[i])
                for i, piece in enumerate(piece_types)
            }
        
        return estimations
    
    def get_estimator_name(self) -> str:
        return "ランダム推定器"


# ================================================================================
# Q値マップ生成器の実装
# ================================================================================

class SimpleQMapGenerator(QMapGenerator):
    """シンプルなQ値マップ生成器"""
    
    def generate(self, board: np.ndarray, piece_estimations: Dict,
                my_pieces: Dict, player_id: str) -> np.ndarray:
        q_map = np.zeros((6, 5, 8))  # 6x5ボード、8方向
        
        # 各自分の駒について
        for pos, piece_type in my_pieces.items():
            x, y = pos
            
            # 各方向のQ値を計算
            for dir_idx, (dx, dy) in enumerate([(0,1), (1,1), (1,0), (1,-1), 
                                                 (0,-1), (-1,-1), (-1,0), (-1,1)]):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < 5 and 0 <= ny < 6:
                    # 基本Q値
                    base_q = 0.5
                    
                    # 敵駒がいる場合
                    if (nx, ny) in piece_estimations:
                        enemy_probs = piece_estimations[(nx, ny)]
                        # 駒の価値に基づいて加算
                        piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 100}
                        for piece, prob in enemy_probs.items():
                            base_q += prob * piece_values.get(piece, 0) * 0.1
                    
                    # 空きマスの場合
                    elif board[ny, nx] == 0:
                        base_q += 0.2
                    
                    q_map[y, x, dir_idx] = base_q
        
        return q_map
    
    def get_generator_name(self) -> str:
        return "シンプルQ値生成"


class StrategicQMapGenerator(QMapGenerator):
    """戦略的Q値マップ生成器"""
    
    def generate(self, board: np.ndarray, piece_estimations: Dict,
                my_pieces: Dict, player_id: str) -> np.ndarray:
        q_map = np.zeros((6, 5, 8))
        
        # ボードの中心性を評価
        center_x, center_y = 2, 3
        
        for pos, piece_type in my_pieces.items():
            x, y = pos
            
            for dir_idx, (dx, dy) in enumerate([(0,1), (1,1), (1,0), (1,-1), 
                                                 (0,-1), (-1,-1), (-1,0), (-1,1)]):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < 5 and 0 <= ny < 6:
                    # 位置価値
                    position_value = 1.0 - (abs(nx - center_x) + abs(ny - center_y)) * 0.1
                    
                    # 駒タイプによる補正
                    piece_multiplier = {
                        "P": 0.8, "N": 1.2, "B": 1.1, 
                        "R": 1.3, "Q": 1.5, "K": 0.5
                    }.get(piece_type, 1.0)
                    
                    base_q = position_value * piece_multiplier
                    
                    # 敵駒評価
                    if (nx, ny) in piece_estimations:
                        enemy_probs = piece_estimations[(nx, ny)]
                        threat_level = sum(prob * {"P": 1, "N": 2, "B": 2, 
                                                  "R": 3, "Q": 5, "K": 10}.get(p, 0)
                                         for p, prob in enemy_probs.items())
                        base_q += threat_level * 0.15
                    
                    q_map[y, x, dir_idx] = base_q
        
        return q_map
    
    def get_generator_name(self) -> str:
        return "戦略的Q値生成"


class NeuralQMapGenerator(QMapGenerator):
    """ニューラルネットワークベースのQ値生成器"""
    
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 30, 240)  # 6x5x8 = 240
    
    def generate(self, board: np.ndarray, piece_estimations: Dict,
                my_pieces: Dict, player_id: str) -> np.ndarray:
        # 3チャンネル入力を作成
        input_tensor = np.zeros((3, 6, 5))
        
        # チャンネル1: ボード状態
        input_tensor[0] = board
        
        # チャンネル2: 自分の駒
        for pos in my_pieces:
            input_tensor[1, pos[1], pos[0]] = 1
        
        # チャンネル3: 敵駒推定
        for pos in piece_estimations:
            input_tensor[2, pos[1], pos[0]] = max(piece_estimations[pos].values())
        
        # ニューラルネットワーク処理
        x = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        # Q値マップに変形
        q_map = x.detach().numpy().reshape(6, 5, 8)
        
        return q_map
    
    def get_generator_name(self) -> str:
        return "ニューラルQ値生成"


# ================================================================================
# 行動選択器の実装
# ================================================================================

class GreedySelector(ActionSelector):
    """貪欲選択器"""
    
    def select_action(self, q_map: np.ndarray, legal_moves: List) -> Tuple:
        if not legal_moves:
            return None
        
        best_q = -float('inf')
        best_move = legal_moves[0]
        
        for move in legal_moves:
            from_pos, to_pos, *_ = move
            
            # 方向を計算
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            # 方向インデックスを取得
            directions = [(0,1), (1,1), (1,0), (1,-1), 
                         (0,-1), (-1,-1), (-1,0), (-1,1)]
            
            try:
                dir_idx = directions.index((dx, dy))
            except ValueError:
                continue
            
            q_value = q_map[from_pos[1], from_pos[0], dir_idx]
            
            if q_value > best_q:
                best_q = q_value
                best_move = move
        
        return best_move
    
    def get_selector_name(self) -> str:
        return "貪欲選択"


class EpsilonGreedySelector(ActionSelector):
    """ε-貪欲選択器"""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
    
    def select_action(self, q_map: np.ndarray, legal_moves: List) -> Tuple:
        if not legal_moves:
            return None
        
        # ε確率でランダム選択
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # それ以外は貪欲選択
        greedy = GreedySelector()
        return greedy.select_action(q_map, legal_moves)
    
    def get_selector_name(self) -> str:
        return f"ε-貪欲(ε={self.epsilon})"


class SoftmaxSelector(ActionSelector):
    """ソフトマックス選択器"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def select_action(self, q_map: np.ndarray, legal_moves: List) -> Tuple:
        if not legal_moves:
            return None
        
        q_values = []
        directions = [(0,1), (1,1), (1,0), (1,-1), 
                     (0,-1), (-1,-1), (-1,0), (-1,1)]
        
        for move in legal_moves:
            from_pos, to_pos, *_ = move
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            try:
                dir_idx = directions.index((dx, dy))
            except ValueError:
                q_values.append(-100)
                continue
            
            q_values.append(q_map[from_pos[1], from_pos[0], dir_idx])
        
        # ソフトマックス確率を計算
        q_tensor = torch.tensor(q_values, dtype=torch.float32) / self.temperature
        probs = F.softmax(q_tensor, dim=0).numpy()
        
        # 確率的に選択
        choice_idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[choice_idx]
    
    def get_selector_name(self) -> str:
        return f"ソフトマックス(T={self.temperature})"


# ================================================================================
# モジュール設定
# ================================================================================

@dataclass
class ModuleConfig:
    """モジュール設定"""
    placement_strategy: InitialPlacementStrategy
    piece_estimator: PieceEstimator
    qmap_generator: QMapGenerator
    action_selector: ActionSelector


# ================================================================================
# CQCNNエージェント
# ================================================================================

class CQCNNAgent:
    """モジュラー型CQCNNエージェント"""
    
    def __init__(self, player_id: str, config: ModuleConfig, name: str = None):
        self.player_id = player_id
        self.config = config
        self.name = name or self._generate_name()
        
        # 統計情報
        self.games_played = 0
        self.wins = 0
        self.move_history = []
        self.last_estimations = {}
        self.last_q_map = None
    
    def _generate_name(self) -> str:
        """エージェント名を生成"""
        return f"{self.config.placement_strategy.get_strategy_name()[:4]}+" \
               f"{self.config.piece_estimator.get_estimator_name()[:6]}+" \
               f"{self.config.qmap_generator.get_generator_name()[:4]}+" \
               f"{self.config.action_selector.get_selector_name()[:4]}"
    
    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        return self.config.placement_strategy.get_placement(self.player_id)
    
    def get_move(self, board: np.ndarray, legal_moves: List, 
                enemy_positions: List[Tuple[int, int]], 
                my_pieces: Dict[Tuple[int, int], str]) -> Optional[Tuple]:
        """手を選択"""
        if not legal_moves:
            return None
        
        try:
            # 1. 敵駒タイプを推定
            if enemy_positions:
                self.last_estimations = self.config.piece_estimator.estimate(
                    board, enemy_positions, self.player_id
                )
            else:
                self.last_estimations = {}
            
            # 2. Q値マップを生成
            self.last_q_map = self.config.qmap_generator.generate(
                board, self.last_estimations, my_pieces, self.player_id
            )
            
            # 3. 行動を選択
            action = self.config.action_selector.select_action(
                self.last_q_map, legal_moves
            )
            
            # 履歴記録
            self.move_history.append({
                'action': action,
                'estimations': len(self.last_estimations),
                'q_max': np.max(self.last_q_map) if self.last_q_map is not None else 0
            })
            
            return action
            
        except Exception as e:
            print(f"⚠️ エラー in {self.name}: {e}")
            return random.choice(legal_moves)
    
    def game_end(self, won: bool):
        """ゲーム終了処理"""
        self.games_played += 1
        if won:
            self.wins += 1
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        return {
            'name': self.name,
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': self.wins / max(self.games_played, 1),
            'total_moves': len(self.move_history)
        }


# ================================================================================
# 競技ランナー
# ================================================================================

class CQCNNCompetitionRunner:
    """CQCNN競技実行システム"""
    
    def __init__(self):
        # 利用可能なモジュール
        self.modules = {
            'placement': [
                StandardPlacement(),
                DefensivePlacement(),
                RandomPlacement()
            ],
            'estimator': [
                SimpleCQCNNEstimator(n_qubits=4, n_layers=2),
                AdvancedCQCNNEstimator(n_qubits=6, n_layers=3),
                RandomEstimator()
            ],
            'qmap': [
                SimpleQMapGenerator(),
                StrategicQMapGenerator(),
                NeuralQMapGenerator()
            ],
            'selector': [
                GreedySelector(),
                EpsilonGreedySelector(epsilon=0.1),
                EpsilonGreedySelector(epsilon=0.3),
                SoftmaxSelector(temperature=1.0)
            ]
        }
        
        # 対戦結果記録
        self.match_results = []
        self.agent_stats = {}
    
    def show_modules(self):
        """利用可能なモジュールを表示"""
        print("=" * 70)
        print("🎮 CQCNN競技システム - 利用可能なモジュール")
        print("=" * 70)
        
        print("\n【1. 初期配置戦略】")
        for i, module in enumerate(self.modules['placement']):
            print(f"  {i}: {module.get_strategy_name()}")
        
        print("\n【2. 敵駒推定器】")
        for i, module in enumerate(self.modules['estimator']):
            print(f"  {i}: {module.get_estimator_name()}")
        
        print("\n【3. Q値マップ生成器】")
        for i, module in enumerate(self.modules['qmap']):
            print(f"  {i}: {module.get_generator_name()}")
        
        print("\n【4. 行動選択器】")
        for i, module in enumerate(self.modules['selector']):
            print(f"  {i}: {module.get_selector_name()}")
    
    def create_agent(self, player_id: str, module_indices: Tuple[int, int, int, int],
                    name: str = None) -> CQCNNAgent:
        """エージェントを作成"""
        config = ModuleConfig(
            placement_strategy=self.modules['placement'][module_indices[0]],
            piece_estimator=self.modules['estimator'][module_indices[1]],
            qmap_generator=self.modules['qmap'][module_indices[2]],
            action_selector=self.modules['selector'][module_indices[3]]
        )
        
        agent = CQCNNAgent(player_id, config, name)
        
        # 統計初期化
        if agent.name not in self.agent_stats:
            self.agent_stats[agent.name] = {
                'games': 0,
                'wins': 0,
                'modules': module_indices
            }
        
        return agent
    
    def simulate_game(self, agent1: CQCNNAgent, agent2: CQCNNAgent, 
                     max_turns: int = 100) -> Dict:
        """ゲームをシミュレート"""
        # 簡易ゲームシミュレーション
        board = np.zeros((6, 5))
        turn = 0
        
        # 初期配置
        placement1 = agent1.get_initial_placement()
        placement2 = agent2.get_initial_placement()
        
        # ボードに配置
        for pos in placement1:
            board[pos[1], pos[0]] = 1
        for pos in placement2:
            board[pos[1], pos[0]] = -1
        
        # ゲームループ（簡易版）
        while turn < max_turns:
            turn += 1
            
            # エージェント1の手番
            my_pieces1 = {pos: "P" for pos in placement1}  # 簡略化
            enemy_positions1 = list(placement2.keys())
            legal_moves1 = self._generate_legal_moves(board, my_pieces1)
            
            move1 = agent1.get_move(board, legal_moves1, enemy_positions1, my_pieces1)
            
            # エージェント2の手番
            my_pieces2 = {pos: "P" for pos in placement2}  # 簡略化
            enemy_positions2 = list(placement1.keys())
            legal_moves2 = self._generate_legal_moves(board, my_pieces2)
            
            move2 = agent2.get_move(board, legal_moves2, enemy_positions2, my_pieces2)
            
            # 勝敗判定（簡易版 - ランダム）
            if turn > 20:
                winner = "A" if random.random() > 0.5 else "B"
                break
        else:
            winner = "Draw"
        
        # 統計更新
        agent1.game_end(winner == "A")
        agent2.game_end(winner == "B")
        
        if winner == "A":
            self.agent_stats[agent1.name]['wins'] += 1
        elif winner == "B":
            self.agent_stats[agent2.name]['wins'] += 1
        
        self.agent_stats[agent1.name]['games'] += 1
        self.agent_stats[agent2.name]['games'] += 1
        
        return {
            'winner': winner,
            'turns': turn,
            'agent1': agent1.name,
            'agent2': agent2.name
        }
    
    def _generate_legal_moves(self, board: np.ndarray, 
                             my_pieces: Dict) -> List[Tuple]:
        """合法手を生成（簡易版）"""
        moves = []
        directions = [(0,1), (1,1), (1,0), (1,-1), 
                     (0,-1), (-1,-1), (-1,0), (-1,1)]
        
        for pos in my_pieces:
            for dx, dy in directions:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if 0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 6:
                    moves.append((pos, new_pos))
        
        return moves if moves else [(list(my_pieces.keys())[0], (2, 3))]
    
    def run_tournament(self, agents: List[CQCNNAgent], games_per_pair: int = 3):
        """トーナメントを実行"""
        print("\n" + "=" * 70)
        print("🏆 トーナメント開始")
        print("=" * 70)
        
        total_games = 0
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i >= j:  # 重複を避ける
                    continue
                
                print(f"\n【{agent1.name} vs {agent2.name}】")
                
                for game_num in range(games_per_pair):
                    result = self.simulate_game(agent1, agent2)
                    total_games += 1
                    
                    winner_name = agent1.name if result['winner'] == "A" else agent2.name
                    if result['winner'] == "Draw":
                        winner_name = "引き分け"
                    
                    print(f"  ゲーム {game_num + 1}: {winner_name} ({result['turns']}ターン)")
        
        print(f"\n📊 総ゲーム数: {total_games}")
    
    def show_results(self):
        """結果を表示"""
        print("\n" + "=" * 70)
        print("📊 最終結果")
        print("=" * 70)
        
        if not self.agent_stats:
            print("まだ結果がありません")
            return
        
        # 勝率でソート
        sorted_agents = sorted(
            self.agent_stats.items(),
            key=lambda x: x[1]['wins'] / max(x[1]['games'], 1),
            reverse=True
        )
        
        print("\n【ランキング】")
        for rank, (name, stats) in enumerate(sorted_agents, 1):
            win_rate = stats['wins'] / max(stats['games'], 1) * 100
            modules = stats['modules']
            print(f"\n{rank}. {name}")
            print(f"   勝率: {win_rate:.1f}% ({stats['wins']}/{stats['games']})")
            print(f"   モジュール構成:")
            print(f"     配置: {self.modules['placement'][modules[0]].get_strategy_name()}")
            print(f"     推定: {self.modules['estimator'][modules[1]].get_estimator_name()}")
            print(f"     Q値: {self.modules['qmap'][modules[2]].get_generator_name()}")
            print(f"     選択: {self.modules['selector'][modules[3]].get_selector_name()}")


# ================================================================================
# メイン実行
# ================================================================================

def quick_demo():
    """クイックデモ実行"""
    print("🚀 CQCNN競技システム - クイックデモ")
    print("=" * 70)
    
    runner = CQCNNCompetitionRunner()
    
    # モジュール表示
    runner.show_modules()
    
    print("\n" + "=" * 70)
    print("📝 エージェント作成")
    print("=" * 70)
    
    # 4種類のエージェントを作成
    agents = [
        runner.create_agent("A", (0, 0, 0, 0), "標準型"),      # 全て基本
        runner.create_agent("B", (1, 1, 1, 1), "高度型"),      # 全て高度
        runner.create_agent("A", (2, 0, 1, 2), "混合型"),      # 混合
        runner.create_agent("B", (0, 2, 2, 3), "実験型")       # 実験的組み合わせ
    ]
    
    for agent in agents:
        modules = runner.agent_stats[agent.name]['modules']
        print(f"\n{agent.name}:")
        print(f"  配置: {runner.modules['placement'][modules[0]].get_strategy_name()}")
        print(f"  推定: {runner.modules['estimator'][modules[1]].get_estimator_name()}")
        print(f"  Q値: {runner.modules['qmap'][modules[2]].get_generator_name()}")
        print(f"  選択: {runner.modules['selector'][modules[3]].get_selector_name()}")
    
    # トーナメント実行
    runner.run_tournament(agents, games_per_pair=2)
    
    # 結果表示
    runner.show_results()
    
    print("\n✅ デモ完了！")


def interactive_mode():
    """インタラクティブモード"""
    runner = CQCNNCompetitionRunner()
    
    print("🎮 CQCNN競技システム - インタラクティブモード")
    print("=" * 70)
    
    while True:
        print("\n【メニュー】")
        print("1. モジュール一覧を表示")
        print("2. カスタムエージェントを作成")
        print("3. クイック対戦（プリセット）")
        print("4. トーナメント実行")
        print("5. 結果表示")
        print("0. 終了")
        
        choice = input("\n選択 (0-5): ").strip()
        
        if choice == "0":
            break
        
        elif choice == "1":
            runner.show_modules()
        
        elif choice == "2":
            print("\n📝 カスタムエージェント作成")
            print("各モジュールの番号を入力してください")
            
            try:
                placement = int(input("初期配置 (0-2): "))
                estimator = int(input("推定器 (0-2): "))
                qmap = int(input("Q値生成 (0-2): "))
                selector = int(input("行動選択 (0-3): "))
                name = input("エージェント名 (省略可): ").strip() or None
                
                agent = runner.create_agent("A", (placement, estimator, qmap, selector), name)
                print(f"✅ エージェント作成: {agent.name}")
                
            except (ValueError, IndexError) as e:
                print(f"❌ エラー: {e}")
        
        elif choice == "3":
            print("\n⚡ クイック対戦")
            
            # プリセットエージェント
            agent1 = runner.create_agent("A", (0, 0, 0, 0), "標準型")
            agent2 = runner.create_agent("B", (1, 1, 1, 1), "高度型")
            
            result = runner.simulate_game(agent1, agent2)
            
            winner_name = agent1.name if result['winner'] == "A" else agent2.name
            if result['winner'] == "Draw":
                winner_name = "引き分け"
            
            print(f"\n結果: {winner_name} の勝利！")
            print(f"ターン数: {result['turns']}")
        
        elif choice == "4":
            print("\n🏆 トーナメント設定")
            
            # プリセットエージェント群
            agents = [
                runner.create_agent("A", (0, 0, 0, 0), "標準Simple"),
                runner.create_agent("B", (1, 1, 1, 1), "守備Advanced"),
                runner.create_agent("A", (2, 2, 2, 2), "ランダム型"),
                runner.create_agent("B", (0, 1, 1, 0), "ハイブリッド")
            ]
            
            games = int(input("各ペアのゲーム数 (1-10): ") or "3")
            runner.run_tournament(agents, games)
        
        elif choice == "5":
            runner.show_results()
    
    print("\n👋 終了します")


def main():
    """メイン実行"""
    print("=" * 70)
    print("🎮 CQCNN競技実行システム")
    print("=" * 70)
    print("\n実行モードを選択してください:")
    print("1. クイックデモ（自動実行）")
    print("2. インタラクティブモード（対話型）")
    print("3. カスタム対戦（上級者向け）")
    
    mode = input("\n選択 (1-3): ").strip()
    
    if mode == "1":
        quick_demo()
    elif mode == "2":
        interactive_mode()
    elif mode == "3":
        print("\n📝 カスタム対戦モード")
        runner = CQCNNCompetitionRunner()
        runner.show_modules()
        
        print("\n2つのエージェントを作成して対戦させます")
        
        # エージェント1
        print("\n【エージェント1】")
        p1 = int(input("配置 (0-2): "))
        e1 = int(input("推定 (0-2): "))
        q1 = int(input("Q値 (0-2): "))
        s1 = int(input("選択 (0-3): "))
        
        # エージェント2  
        print("\n【エージェント2】")
        p2 = int(input("配置 (0-2): "))
        e2 = int(input("推定 (0-2): "))
        q2 = int(input("Q値 (0-2): "))
        s2 = int(input("選択 (0-3): "))
        
        agent1 = runner.create_agent("A", (p1, e1, q1, s1))
        agent2 = runner.create_agent("B", (p2, e2, q2, s2))
        
        games = int(input("\nゲーム数 (1-10): ") or "5")
        
        for i in range(games):
            print(f"\n--- ゲーム {i+1}/{games} ---")
            result = runner.simulate_game(agent1, agent2)
            
            winner_name = agent1.name if result['winner'] == "A" else agent2.name
            if result['winner'] == "Draw":
                winner_name = "引き分け"
            
            print(f"結果: {winner_name} ({result['turns']}ターン)")
        
        runner.show_results()
    else:
        print("無効な選択です")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 中断されました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
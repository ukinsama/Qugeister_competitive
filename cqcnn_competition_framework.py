#!/usr/bin/env python3
"""
CQCNN競技フレームワーク
ユーザーが各モジュールをカスタマイズして対戦できるシステム

モジュール構成:
1. 初期配置戦略
2. 敵駒推定器（CQCNN）
3. Q値マップ生成器
4. 行動選択エージェント
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import json
import random
import time
from dataclasses import dataclass

# ================================================================================
# Module 1: 初期配置戦略
# ================================================================================

class InitialPlacementStrategy(ABC):
    """初期配置戦略の基底クラス"""
    
    @abstractmethod
    def get_placement(self, player: str) -> Dict[Tuple[int, int], str]:
        """初期配置を返す"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """戦略名を返す"""
        pass


class StandardPlacement(InitialPlacementStrategy):
    """標準配置: 前列善玉、後列悪玉"""
    
    def get_placement(self, player: str) -> Dict[Tuple[int, int], str]:
        if player == "A":
            return {
                (1, 0): "good", (2, 0): "good", (3, 0): "good", (4, 0): "good",
                (1, 1): "bad", (2, 1): "bad", (3, 1): "bad", (4, 1): "bad"
            }
        else:
            return {
                (1, 5): "good", (2, 5): "good", (3, 5): "good", (4, 5): "good",
                (1, 4): "bad", (2, 4): "bad", (3, 4): "bad", (4, 4): "bad"
            }
    
    def get_strategy_name(self) -> str:
        return "標準配置"


class DefensivePlacement(InitialPlacementStrategy):
    """守備的配置: 善玉を後ろに隠す"""
    
    def get_placement(self, player: str) -> Dict[Tuple[int, int], str]:
        if player == "A":
            return {
                (1, 0): "bad", (2, 0): "bad", (3, 0): "bad", (4, 0): "bad",
                (1, 1): "good", (2, 1): "good", (3, 1): "good", (4, 1): "good"
            }
        else:
            return {
                (1, 5): "bad", (2, 5): "bad", (3, 5): "bad", (4, 5): "bad",
                (1, 4): "good", (2, 4): "good", (3, 4): "good", (4, 4): "good"
            }
    
    def get_strategy_name(self) -> str:
        return "守備的配置"


class RandomPlacement(InitialPlacementStrategy):
    """ランダム配置: 善玉と悪玉をランダムに配置"""
    
    def get_placement(self, player: str) -> Dict[Tuple[int, int], str]:
        if player == "A":
            positions = [(1, 0), (2, 0), (3, 0), (4, 0),
                        (1, 1), (2, 1), (3, 1), (4, 1)]
        else:
            positions = [(1, 5), (2, 5), (3, 5), (4, 5),
                        (1, 4), (2, 4), (3, 4), (4, 4)]
        
        random.shuffle(positions)
        placement = {}
        for i, pos in enumerate(positions):
            placement[pos] = "good" if i < 4 else "bad"
        
        return placement
    
    def get_strategy_name(self) -> str:
        return "ランダム配置"


class MixedPlacement(InitialPlacementStrategy):
    """混合配置: 善玉と悪玉を交互に配置"""
    
    def get_placement(self, player: str) -> Dict[Tuple[int, int], str]:
        if player == "A":
            return {
                (1, 0): "good", (2, 0): "bad", (3, 0): "good", (4, 0): "bad",
                (1, 1): "bad", (2, 1): "good", (3, 1): "bad", (4, 1): "good"
            }
        else:
            return {
                (1, 5): "good", (2, 5): "bad", (3, 5): "good", (4, 5): "bad",
                (1, 4): "bad", (2, 4): "good", (3, 4): "bad", (4, 4): "good"
            }
    
    def get_strategy_name(self) -> str:
        return "混合配置"


# ================================================================================
# Module 2: 敵駒推定器（CQCNN）
# ================================================================================

class PieceEstimator(ABC):
    """敵駒推定器の基底クラス"""
    
    @abstractmethod
    def estimate(self, board_state: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        敵駒の種類を推定
        Returns: {position: {'good_prob': float, 'bad_prob': float, 'confidence': float}}
        """
        pass
    
    @abstractmethod
    def train(self, training_data: List[Dict]) -> None:
        """学習を実行"""
        pass
    
    @abstractmethod
    def get_estimator_name(self) -> str:
        """推定器名を返す"""
        pass


class SimpleCQCNNEstimator(PieceEstimator):
    """シンプルなCQCNN推定器"""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def _build_model(self):
        """簡易CQCNNモデル構築"""
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # good/bad確率
        )
    
    def estimate(self, board_state: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """敵駒推定"""
        results = {}
        
        # ボード状態をテンソルに変換
        board_tensor = self._prepare_board_tensor(board_state, player)
        
        with torch.no_grad():
            output = self.model(board_tensor)
            probs = F.softmax(output, dim=1)
            
            # 各敵駒位置に対して同じ推定を返す（簡易版）
            for pos in enemy_positions:
                results[pos] = {
                    'good_prob': probs[0, 0].item(),
                    'bad_prob': probs[0, 1].item(),
                    'confidence': max(probs[0].tolist())
                }
        
        return results
    
    def _prepare_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        """ボード状態をテンソルに変換"""
        tensor = torch.zeros(1, 3, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))
        
        return tensor
    
    def train(self, training_data: List[Dict]) -> None:
        """簡易学習"""
        # 実装は省略（実際にはtraining_dataから学習）
        pass
    
    def get_estimator_name(self) -> str:
        return f"SimpleCQCNN({self.n_qubits}qubits)"


class AdvancedCQCNNEstimator(PieceEstimator):
    """高度なCQCNN推定器（量子回路付き）"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.quantum_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.cnn = self._build_cnn()
        self.optimizer = torch.optim.Adam(
            list(self.cnn.parameters()) + [self.quantum_params], 
            lr=0.001
        )
    
    def _build_cnn(self):
        """CNN部分の構築"""
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, self.n_qubits),
            nn.Tanh()
        )
    
    def _quantum_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """簡易量子回路シミュレーション"""
        batch_size = x.shape[0]
        state = x.clone()
        
        for layer in range(self.n_layers):
            # 回転ゲート
            for q in range(self.n_qubits):
                rotation = self.quantum_params[layer, q]
                state[:, q] = torch.cos(rotation[0]) * state[:, q] + \
                             torch.sin(rotation[1]) * torch.roll(state, 1, dims=1)[:, q]
        
        return state
    
    def estimate(self, board_state: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """高度な敵駒推定"""
        results = {}
        board_tensor = self._prepare_board_tensor(board_state, player)
        
        with torch.no_grad():
            # CNN特徴抽出
            features = self.cnn(board_tensor)
            
            # 量子回路処理
            quantum_output = self._quantum_circuit(features)
            
            # 各位置に対して異なる推定
            for i, pos in enumerate(enemy_positions):
                # 位置依存の推定
                position_factor = (pos[0] / 5.0 + pos[1] / 5.0) / 2.0
                
                # 量子出力から確率を計算
                q_val = quantum_output[0, i % self.n_qubits].item()
                good_prob = (1 + q_val * position_factor) / 2
                good_prob = max(0.0, min(1.0, good_prob))
                
                results[pos] = {
                    'good_prob': good_prob,
                    'bad_prob': 1 - good_prob,
                    'confidence': 0.5 + abs(q_val) * 0.3
                }
        
        return results
    
    def _prepare_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        """ボード状態をテンソルに変換"""
        tensor = torch.zeros(1, 3, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))
        
        return tensor
    
    def train(self, training_data: List[Dict]) -> None:
        """高度な学習（強化学習）"""
        # 実装は省略
        pass
    
    def get_estimator_name(self) -> str:
        return f"AdvancedCQCNN({self.n_qubits}qubits,{self.n_layers}layers)"


class RandomEstimator(PieceEstimator):
    """ランダム推定器（ベースライン）"""
    
    def estimate(self, board_state: np.ndarray, enemy_positions: List[Tuple[int, int]], 
                player: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """ランダムな推定"""
        results = {}
        for pos in enemy_positions:
            good_prob = random.random()
            results[pos] = {
                'good_prob': good_prob,
                'bad_prob': 1 - good_prob,
                'confidence': random.uniform(0.3, 0.7)
            }
        return results
    
    def train(self, training_data: List[Dict]) -> None:
        """学習なし"""
        pass
    
    def get_estimator_name(self) -> str:
        return "ランダム推定"


# ================================================================================
# Module 3: Q値マップ生成器
# ================================================================================

class QMapGenerator(ABC):
    """Q値マップ生成器の基底クラス"""
    
    @abstractmethod
    def generate(self, board_state: np.ndarray, 
                estimations: Dict[Tuple[int, int], Dict[str, float]],
                my_pieces: Dict[Tuple[int, int], str],
                player: str) -> np.ndarray:
        """
        Q値マップを生成
        Returns: (6, 6, 4) の配列 - 各位置の4方向への移動価値
        """
        pass
    
    @abstractmethod
    def get_generator_name(self) -> str:
        """生成器名を返す"""
        pass


class SimpleQMapGenerator(QMapGenerator):
    """シンプルなQ値マップ生成器"""
    
    def generate(self, board_state: np.ndarray, 
                estimations: Dict[Tuple[int, int], Dict[str, float]],
                my_pieces: Dict[Tuple[int, int], str],
                player: str) -> np.ndarray:
        """基本的なQ値マップ生成"""
        q_map = np.zeros((6, 6, 4))
        
        for piece_pos, piece_type in my_pieces.items():
            x, y = piece_pos
            
            # 4方向の評価（上右下左）
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            for dir_idx, (dx, dy) in enumerate(directions):
                new_x, new_y = x + dx, y + dy
                
                # 境界チェック
                if not (0 <= new_x < 6 and 0 <= new_y < 6):
                    q_map[y, x, dir_idx] = -100
                    continue
                
                # 基本スコア
                score = 0.0
                
                # 前進ボーナス
                if player == "A" and dy > 0:
                    score += 2.0
                elif player == "B" and dy < 0:
                    score += 2.0
                
                # 駒取り評価
                if (new_x, new_y) in estimations:
                    est = estimations[(new_x, new_y)]
                    # 善玉を取る価値 - 悪玉を取るリスク
                    score += est['good_prob'] * 5.0 - est['bad_prob'] * 3.0
                
                q_map[y, x, dir_idx] = score
        
        return q_map
    
    def get_generator_name(self) -> str:
        return "シンプルQ値生成"


class StrategicQMapGenerator(QMapGenerator):
    """戦略的Q値マップ生成器"""
    
    def __init__(self):
        self.weights = {
            'forward': 3.0,
            'capture_good': 8.0,
            'capture_bad': -4.0,
            'escape': 10.0,
            'center': 1.5,
            'protection': 2.0
        }
    
    def generate(self, board_state: np.ndarray, 
                estimations: Dict[Tuple[int, int], Dict[str, float]],
                my_pieces: Dict[Tuple[int, int], str],
                player: str) -> np.ndarray:
        """戦略的なQ値マップ生成"""
        q_map = np.zeros((6, 6, 4))
        
        # 脱出口の定義
        escape_positions = [(0, 5), (5, 5)] if player == "A" else [(0, 0), (5, 0)]
        
        for piece_pos, piece_type in my_pieces.items():
            x, y = piece_pos
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            for dir_idx, (dx, dy) in enumerate(directions):
                new_x, new_y = x + dx, y + dy
                
                if not (0 <= new_x < 6 and 0 <= new_y < 6):
                    q_map[y, x, dir_idx] = -100
                    continue
                
                score = 0.0
                
                # 前進評価
                if player == "A" and dy > 0:
                    score += self.weights['forward']
                elif player == "B" and dy < 0:
                    score += self.weights['forward']
                
                # 脱出評価（善玉のみ）
                if piece_type == "good" and (new_x, new_y) in escape_positions:
                    score += self.weights['escape']
                
                # 中央制御
                center_dist = abs(new_x - 2.5) + abs(new_y - 2.5)
                score += self.weights['center'] * (5 - center_dist) / 5
                
                # 駒取り評価（推定を考慮）
                if (new_x, new_y) in estimations:
                    est = estimations[(new_x, new_y)]
                    score += est['good_prob'] * self.weights['capture_good']
                    score += est['bad_prob'] * self.weights['capture_bad']
                    score *= est['confidence']  # 確信度で重み付け
                
                # 味方との連携
                for ally_pos in my_pieces:
                    if ally_pos != piece_pos:
                        dist = abs(new_x - ally_pos[0]) + abs(new_y - ally_pos[1])
                        if dist == 1:
                            score += self.weights['protection']
                
                q_map[y, x, dir_idx] = score
        
        return q_map
    
    def get_generator_name(self) -> str:
        return "戦略的Q値生成"


class NeuralQMapGenerator(QMapGenerator):
    """ニューラルネットワークベースのQ値マップ生成器"""
    
    def __init__(self):
        # 入力次元を正確に計算
        # ボード: 6*6 = 36
        # 推定特徴: 10
        # 合計: 46
        input_dim = 6 * 6 + 10
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # 46 -> 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6 * 6 * 4),  # 64 -> 144 (6*6*4)
            nn.Tanh()
        )
    
    def generate(self, board_state: np.ndarray, 
                estimations: Dict[Tuple[int, int], Dict[str, float]],
                my_pieces: Dict[Tuple[int, int], str],
                player: str) -> np.ndarray:
        """ニューラルネットワークでQ値マップ生成"""
        # 入力特徴量を作成
        board_flat = board_state.flatten()
        
        # 推定情報を集約
        est_features = np.zeros(10)
        if estimations:
            good_probs = [e['good_prob'] for e in estimations.values()]
            confidences = [e['confidence'] for e in estimations.values()]
            est_features[0] = np.mean(good_probs)
            est_features[1] = np.std(good_probs)
            est_features[2] = np.mean(confidences)
            est_features[3] = len(my_pieces)
            est_features[4] = len(estimations)
        
        # 入力結合
        input_tensor = torch.tensor(
            np.concatenate([board_flat, est_features]), 
            dtype=torch.float32
        ).unsqueeze(0)
        
        # ネットワーク推論
        with torch.no_grad():
            output = self.network(input_tensor)
            q_map = output.view(6, 6, 4).numpy()
        
        # スケーリング
        q_map = q_map * 10.0
        
        return q_map
    
    def get_generator_name(self) -> str:
        return "ニューラルQ値生成"


# ================================================================================
# Module 4: 行動選択エージェント
# ================================================================================

class ActionSelector(ABC):
    """行動選択エージェントの基底クラス"""
    
    @abstractmethod
    def select_action(self, q_map: np.ndarray, 
                     legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple:
        """Q値マップから行動を選択"""
        pass
    
    @abstractmethod
    def get_selector_name(self) -> str:
        """選択器名を返す"""
        pass


class GreedySelector(ActionSelector):
    """貪欲選択: 最大Q値の行動を選択"""
    
    def select_action(self, q_map: np.ndarray, 
                     legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple:
        """最大Q値の行動を選択"""
        best_move = None
        best_q = -float('inf')
        
        for move in legal_moves:
            from_pos, to_pos = move
            
            # 方向を計算
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            if dy == -1 and dx == 0:    dir_idx = 0
            elif dx == 1 and dy == 0:   dir_idx = 1
            elif dy == 1 and dx == 0:   dir_idx = 2
            elif dx == -1 and dy == 0:  dir_idx = 3
            else: continue
            
            q_value = q_map[from_pos[1], from_pos[0], dir_idx]
            
            if q_value > best_q:
                best_q = q_value
                best_move = move
        
        return best_move if best_move else random.choice(legal_moves)
    
    def get_selector_name(self) -> str:
        return "貪欲選択"


class EpsilonGreedySelector(ActionSelector):
    """ε-貪欲選択: 確率的に探索"""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
    
    def select_action(self, q_map: np.ndarray, 
                     legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple:
        """ε確率でランダム、それ以外は最大Q値"""
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Greedyと同じ処理
        best_move = None
        best_q = -float('inf')
        
        for move in legal_moves:
            from_pos, to_pos = move
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            if dy == -1 and dx == 0:    dir_idx = 0
            elif dx == 1 and dy == 0:   dir_idx = 1
            elif dy == 1 and dx == 0:   dir_idx = 2
            elif dx == -1 and dy == 0:  dir_idx = 3
            else: continue
            
            q_value = q_map[from_pos[1], from_pos[0], dir_idx]
            
            if q_value > best_q:
                best_q = q_value
                best_move = move
        
        return best_move if best_move else random.choice(legal_moves)
    
    def get_selector_name(self) -> str:
        return f"ε-貪欲(ε={self.epsilon})"


class SoftmaxSelector(ActionSelector):
    """ソフトマックス選択: Q値に基づく確率的選択"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def select_action(self, q_map: np.ndarray, 
                     legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple:
        """Q値をソフトマックスで確率化して選択"""
        q_values = []
        
        for move in legal_moves:
            from_pos, to_pos = move
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            if dy == -1 and dx == 0:    dir_idx = 0
            elif dx == 1 and dy == 0:   dir_idx = 1
            elif dy == 1 and dx == 0:   dir_idx = 2
            elif dx == -1 and dy == 0:  dir_idx = 3
            else:
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
# 統合: モジュラーCQCNN AIエージェント
# ================================================================================

@dataclass
class ModuleConfig:
    """モジュール設定"""
    placement_strategy: InitialPlacementStrategy
    piece_estimator: PieceEstimator
    qmap_generator: QMapGenerator
    action_selector: ActionSelector


class ModularCQCNNAgent:
    """モジュラー型CQCNN AIエージェント"""
    
    def __init__(self, player_id: str, config: ModuleConfig):
        self.player_id = player_id
        self.config = config
        self.name = self._generate_name()
        
        # 統計情報
        self.games_played = 0
        self.wins = 0
        self.training_data = []
    
    def _generate_name(self) -> str:
        """エージェント名を生成"""
        return f"Agent[{self.config.placement_strategy.get_strategy_name()[:3]}+" \
               f"{self.config.piece_estimator.get_estimator_name()[:6]}+" \
               f"{self.config.qmap_generator.get_generator_name()[:3]}+" \
               f"{self.config.action_selector.get_selector_name()[:3]}]"
    
    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        return self.config.placement_strategy.get_placement(self.player_id)
    
    def get_move(self, game_state: Any, legal_moves: List) -> Optional[Tuple]:
        """手を選択"""
        if not legal_moves:
            return None
        
        # 1. 敵駒位置を特定
        enemy_positions = self._find_enemy_positions(game_state)
        
        # 2. 敵駒タイプを推定
        estimations = self.config.piece_estimator.estimate(
            game_state.board,
            enemy_positions,
            self.player_id
        )
        
        # 3. Q値マップを生成
        my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
        q_map = self.config.qmap_generator.generate(
            game_state.board,
            estimations,
            my_pieces,
            self.player_id
        )
        
        # 4. 行動を選択
        action = self.config.action_selector.select_action(q_map, legal_moves)
        
        # 学習データを記録
        self.training_data.append({
            'board': game_state.board.copy(),
            'estimations': estimations,
            'q_map': q_map.copy(),
            'action': action
        })
        
        return action
    
    def _find_enemy_positions(self, game_state: Any) -> List[Tuple[int, int]]:
        """敵駒の位置を特定"""
        enemy_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces
        return list(enemy_pieces.keys())
    
    def record_game_result(self, won: bool):
        """ゲーム結果を記録"""
        self.games_played += 1
        if won:
            self.wins += 1
        
        # 学習データに結果を反映
        for data in self.training_data[-20:]:  # 最後の20手
            data['result'] = 1.0 if won else -1.0
    
    def train(self):
        """各モジュールを学習"""
        if self.training_data:
            # 推定器の学習
            self.config.piece_estimator.train(self.training_data)
            
            # 他のモジュールも学習可能なら実行
            # （実装は省略）
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        return {
            'name': self.name,
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': self.wins / max(self.games_played, 1),
            'modules': {
                'placement': self.config.placement_strategy.get_strategy_name(),
                'estimator': self.config.piece_estimator.get_estimator_name(),
                'qmap': self.config.qmap_generator.get_generator_name(),
                'selector': self.config.action_selector.get_selector_name()
            }
        }


# ================================================================================
# 競技システム
# ================================================================================

class CQCNNCompetition:
    """CQCNN競技システム"""
    
    def __init__(self):
        self.available_modules = {
            'placement': [
                StandardPlacement(),
                DefensivePlacement(),
                RandomPlacement(),
                MixedPlacement()
            ],
            'estimator': [
                SimpleCQCNNEstimator(n_qubits=4),
                AdvancedCQCNNEstimator(n_qubits=6),
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
                SoftmaxSelector(temperature=1.0)
            ]
        }
    
    def create_agent(self, player_id: str, 
                    placement_idx: int = 0,
                    estimator_idx: int = 0,
                    qmap_idx: int = 0,
                    selector_idx: int = 0) -> ModularCQCNNAgent:
        """指定されたモジュールでエージェントを作成"""
        config = ModuleConfig(
            placement_strategy=self.available_modules['placement'][placement_idx],
            piece_estimator=self.available_modules['estimator'][estimator_idx],
            qmap_generator=self.available_modules['qmap'][qmap_idx],
            action_selector=self.available_modules['selector'][selector_idx]
        )
        
        return ModularCQCNNAgent(player_id, config)
    
    def show_available_modules(self):
        """利用可能なモジュールを表示"""
        print("=" * 70)
        print("🎮 利用可能なモジュール")
        print("=" * 70)
        
        for category, modules in self.available_modules.items():
            print(f"\n【{category.upper()}】")
            for i, module in enumerate(modules):
                if category == 'placement':
                    name = module.get_strategy_name()
                elif category == 'estimator':
                    name = module.get_estimator_name()
                elif category == 'qmap':
                    name = module.get_generator_name()
                elif category == 'selector':
                    name = module.get_selector_name()
                print(f"  {i}: {name}")
    
    def run_match(self, agent1: ModularCQCNNAgent, agent2: ModularCQCNNAgent, 
                 verbose: bool = True) -> str:
        """対戦を実行（簡易版）"""
        if verbose:
            print(f"\n🎮 対戦: {agent1.name} vs {agent2.name}")
        
        # ここでは結果をランダムに返す（実際のゲームエンジンと統合が必要）
        winner = random.choice([agent1.player_id, agent2.player_id, "Draw"])
        
        if winner == agent1.player_id:
            agent1.record_game_result(True)
            agent2.record_game_result(False)
            if verbose:
                print(f"🏆 勝者: {agent1.name}")
        elif winner == agent2.player_id:
            agent1.record_game_result(False)
            agent2.record_game_result(True)
            if verbose:
                print(f"🏆 勝者: {agent2.name}")
        else:
            agent1.record_game_result(False)
            agent2.record_game_result(False)
            if verbose:
                print("🤝 引き分け")
        
        return winner


# ================================================================================
# デモ実行
# ================================================================================

def main():
    """デモ実行"""
    print("🚀 CQCNN競技フレームワーク")
    print("=" * 70)
    
    # 競技システム初期化
    competition = CQCNNCompetition()
    
    # 利用可能モジュール表示
    competition.show_available_modules()
    
    # エージェント作成例
    print("\n" + "=" * 70)
    print("📝 エージェント作成デモ")
    print("=" * 70)
    
    # エージェント1: 標準配置 + シンプルCQCNN + シンプルQ値 + 貪欲選択
    agent1 = competition.create_agent("A", 0, 0, 0, 0)
    print(f"\nAgent 1: {agent1.name}")
    print(f"  配置: {agent1.config.placement_strategy.get_strategy_name()}")
    print(f"  推定: {agent1.config.piece_estimator.get_estimator_name()}")
    print(f"  Q値: {agent1.config.qmap_generator.get_generator_name()}")
    print(f"  選択: {agent1.config.action_selector.get_selector_name()}")
    
    # エージェント2: 守備配置 + 高度CQCNN + 戦略Q値 + ε-貪欲
    agent2 = competition.create_agent("B", 1, 1, 1, 1)
    print(f"\nAgent 2: {agent2.name}")
    print(f"  配置: {agent2.config.placement_strategy.get_strategy_name()}")
    print(f"  推定: {agent2.config.piece_estimator.get_estimator_name()}")
    print(f"  Q値: {agent2.config.qmap_generator.get_generator_name()}")
    print(f"  選択: {agent2.config.action_selector.get_selector_name()}")
    
    # デモ対戦
    print("\n" + "=" * 70)
    print("🏆 デモ対戦")
    print("=" * 70)
    
    for i in range(3):
        competition.run_match(agent1, agent2)
    
    # 統計表示
    print("\n" + "=" * 70)
    print("📊 統計")
    print("=" * 70)
    
    for agent in [agent1, agent2]:
        stats = agent.get_statistics()
        print(f"\n{stats['name']}:")
        print(f"  勝率: {stats['win_rate']:.1%} ({stats['wins']}/{stats['games_played']})")
    
    print("\n✅ デモ完了！")
    print("\n💡 使い方:")
    print("  1. create_agent()でモジュール番号を指定してエージェントを作成")
    print("  2. 異なる組み合わせで性能を比較")
    print("  3. 学習により各モジュールを改善")


if __name__ == "__main__":
    main()

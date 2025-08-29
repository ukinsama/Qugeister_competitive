#!/usr/bin/env python3
"""
CQCNN対戦システム v2 - 完全モジュラー設計版
5つの独立モジュール + 7チャンネル入力対応

モジュール構成:
1. PlacementStrategy - 初期配置戦略
2. PieceEstimator - 敵駒推定器
3. RewardFunction - 報酬関数
4. QMapGenerator - Q値マップ生成器
5. ActionSelector - 行動選択器
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
# Part 1: 基本設定とデータ構造
# ================================================================================

class GameConfig:
    """ゲーム設定"""
    def __init__(self):
        self.board_size = (6, 6)
        self.max_turns = 100
        self.n_pieces = 8  # 各プレイヤーの駒数
        self.n_good = 4   # 善玉の数
        self.n_bad = 4    # 悪玉の数


class GameState:
    """ゲーム状態"""
    def __init__(self):
        self.board = np.zeros((6, 6), dtype=int)
        self.player_a_pieces = {}  # {位置: 駒タイプ}
        self.player_b_pieces = {}
        self.turn = 0
        self.winner = None
        
    def is_game_over(self):
        return self.winner is not None or self.turn >= 100


# ================================================================================
# Part 2: モジュール1 - 初期配置戦略
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
        
        # プレイヤーAは下側（行0-1）、Bは上側（行4-5）の中央4列に配置
        if player_id == "A":
            positions = [(0, 1), (0, 2), (0, 3), (0, 4),
                        (1, 1), (1, 2), (1, 3), (1, 4)]
        else:
            positions = [(4, 1), (4, 2), (4, 3), (4, 4),
                        (5, 1), (5, 2), (5, 3), (5, 4)]
        
        # ランダムに善玉と悪玉を配置
        piece_types = ['good'] * 4 + ['bad'] * 4
        random.shuffle(piece_types)
        
        for pos, piece_type in zip(positions, piece_types):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "標準配置"


class AggressivePlacement(PlacementStrategy):
    """攻撃的配置戦略（善玉を前線に）"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            # 前列に善玉を多く配置
            front_positions = [(1, 1), (1, 2), (1, 3), (1, 4)]
            back_positions = [(0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            front_positions = [(4, 1), (4, 2), (4, 3), (4, 4)]
            back_positions = [(5, 1), (5, 2), (5, 3), (5, 4)]
        
        # 前列に善玉3個、悪玉1個
        front_pieces = ['good', 'good', 'good', 'bad']
        random.shuffle(front_pieces)
        
        # 後列に善玉1個、悪玉3個
        back_pieces = ['good', 'bad', 'bad', 'bad']
        random.shuffle(back_pieces)
        
        for pos, piece_type in zip(front_positions, front_pieces):
            placement[pos] = piece_type
        for pos, piece_type in zip(back_positions, back_pieces):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "攻撃的配置"


class DefensivePlacement(PlacementStrategy):
    """防御的配置戦略（善玉を後方に）"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            front_positions = [(1, 1), (1, 2), (1, 3), (1, 4)]
            back_positions = [(0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            front_positions = [(4, 1), (4, 2), (4, 3), (4, 4)]
            back_positions = [(5, 1), (5, 2), (5, 3), (5, 4)]
        
        # 前列に悪玉3個、善玉1個
        front_pieces = ['bad', 'bad', 'bad', 'good']
        random.shuffle(front_pieces)
        
        # 後列に悪玉1個、善玉3個
        back_pieces = ['bad', 'good', 'good', 'good']
        random.shuffle(back_pieces)
        
        for pos, piece_type in zip(front_positions, front_pieces):
            placement[pos] = piece_type
        for pos, piece_type in zip(back_positions, back_pieces):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "防御的配置"


# ================================================================================
# Part 3: モジュール2 - 敵駒推定器
# ================================================================================

class PieceEstimator(ABC):
    """敵駒推定器の基底クラス"""
    
    @abstractmethod
    def estimate(self, board: np.ndarray, 
                enemy_positions: List[Tuple[int, int]], 
                player: str,
                my_pieces: Dict[Tuple[int, int], str],
                turn: int) -> Dict[Tuple[int, int], Dict[str, float]]:
        """敵駒タイプを推定"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """推定器名を取得"""
        pass
    
    def prepare_tensor_7ch(self, board: np.ndarray, player: str, 
                           my_pieces: Dict, turn: int) -> torch.Tensor:
        """7チャンネル入力テンソルを準備"""
        tensor = torch.zeros(1, 7, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        # Ch0: 自分の善玉
        for pos, piece_type in my_pieces.items():
            if piece_type == 'good' and board[pos] == player_val:
                tensor[0, 0, pos[0], pos[1]] = 1.0
        
        # Ch1: 自分の悪玉
        for pos, piece_type in my_pieces.items():
            if piece_type == 'bad' and board[pos] == player_val:
                tensor[0, 1, pos[0], pos[1]] = 1.0
        
        # Ch2: 相手の駒（種類不明）
        tensor[0, 2] = torch.from_numpy((board == enemy_val).astype(np.float32))
        
        # Ch3: 空きマス
        tensor[0, 3] = torch.from_numpy((board == 0).astype(np.float32))
        
        # Ch4: 自分の脱出口
        if player == "A":
            tensor[0, 4, 5, 0] = 1.0  # 左上
            tensor[0, 4, 5, 5] = 1.0  # 右上
        else:
            tensor[0, 4, 0, 0] = 1.0  # 左下
            tensor[0, 4, 0, 5] = 1.0  # 右下
        
        # Ch5: 相手の脱出口
        if player == "A":
            tensor[0, 5, 0, 0] = 1.0
            tensor[0, 5, 0, 5] = 1.0
        else:
            tensor[0, 5, 5, 0] = 1.0
            tensor[0, 5, 5, 5] = 1.0
        
        # Ch6: ターン進行度
        tensor[0, 6, :, :] = turn / 100.0
        
        return tensor


class RandomEstimator(PieceEstimator):
    """ランダム推定器"""
    
    def estimate(self, board: np.ndarray, 
                enemy_positions: List[Tuple[int, int]], 
                player: str,
                my_pieces: Dict[Tuple[int, int], str],
                turn: int) -> Dict[Tuple[int, int], Dict[str, float]]:
        
        results = {}
        for pos in enemy_positions:
            good_prob = random.random()
            results[pos] = {
                'good_prob': good_prob,
                'bad_prob': 1 - good_prob,
                'confidence': 0.5
            }
        return results
    
    def get_name(self) -> str:
        return "ランダム推定"


class SimpleCNNEstimator(PieceEstimator):
    """シンプルCNN推定器"""
    
    def __init__(self):
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def _build_model(self):
        """7チャンネル入力対応のCNNモデル"""
        return nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 善玉/悪玉の2クラス
        )
    
    def estimate(self, board: np.ndarray, 
                enemy_positions: List[Tuple[int, int]], 
                player: str,
                my_pieces: Dict[Tuple[int, int], str],
                turn: int) -> Dict[Tuple[int, int], Dict[str, float]]:
        
        self.model.eval()
        results = {}
        
        # 7チャンネルテンソルを準備
        tensor = self.prepare_tensor_7ch(board, player, my_pieces, turn)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            
            # 全ての敵駒に同じ推定を適用（簡略化）
            for pos in enemy_positions:
                results[pos] = {
                    'good_prob': probs[0, 0].item(),
                    'bad_prob': probs[0, 1].item(),
                    'confidence': max(probs[0].tolist())
                }
        
        return results
    
    def get_name(self) -> str:
        return "SimpleCNN"


class CQCNNEstimator(PieceEstimator):
    """CQCNN（量子回路付き）推定器"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def _build_model(self):
        """量子回路層を含むCQCNNモデル"""
        class QuantumLayer(nn.Module):
            def __init__(self, n_qubits, n_layers):
                super().__init__()
                self.n_qubits = n_qubits
                self.n_layers = n_layers
                self.rotation_params = nn.Parameter(
                    torch.randn(n_layers, n_qubits, 3) * 0.1
                )
                self.entangle_params = nn.Parameter(
                    torch.randn(n_layers, n_qubits - 1) * 0.1
                )
            
            def forward(self, x):
                batch_size = x.shape[0]
                
                # 簡略化された量子回路シミュレーション
                state_real = torch.ones(batch_size, self.n_qubits)
                state_imag = torch.zeros(batch_size, self.n_qubits)
                
                # データエンコーディング
                for i in range(min(self.n_qubits, x.shape[1])):
                    angle = x[:, i] * np.pi
                    state_real[:, i] = torch.cos(angle/2)
                    state_imag[:, i] = torch.sin(angle/2)
                
                # 変分回路
                for layer in range(self.n_layers):
                    # 回転ゲート
                    for q in range(self.n_qubits):
                        rx = self.rotation_params[layer, q, 0]
                        state_real[:, q] = state_real[:, q] * torch.cos(rx/2)
                        state_imag[:, q] = state_imag[:, q] * torch.sin(rx/2)
                    
                    # エンタングルメント（簡略化）
                    for i in range(self.n_qubits - 1):
                        strength = torch.sigmoid(self.entangle_params[layer, i])
                        avg = (state_real[:, i] + state_real[:, i+1]) / 2
                        state_real[:, i] = (1-strength) * state_real[:, i] + strength * avg
                        state_real[:, i+1] = (1-strength) * state_real[:, i+1] + strength * avg
                
                # 測定
                measurements = torch.sqrt(state_real**2 + state_imag**2)
                return measurements
        
        class CQCNNModel(nn.Module):
            def __init__(self, n_qubits, n_layers):
                super().__init__()
                # CNN特徴抽出（7チャンネル入力）
                self.cnn = nn.Sequential(
                    nn.Conv2d(7, 16, kernel_size=3, padding=1),
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
                
                # 量子回路層
                self.quantum = QuantumLayer(n_qubits, n_layers)
                
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
                player: str,
                my_pieces: Dict[Tuple[int, int], str],
                turn: int) -> Dict[Tuple[int, int], Dict[str, float]]:
        
        self.model.eval()
        results = {}
        
        # 7チャンネルテンソルを準備
        tensor = self.prepare_tensor_7ch(board, player, my_pieces, turn)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            
            # 位置による調整を加えた推定
            for pos in enemy_positions:
                # 位置による重み付け
                position_factor = (pos[0] / 5.0 + pos[1] / 5.0) / 2.0
                adjusted_probs = probs[0] * (0.7 + 0.3 * position_factor)
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                
                results[pos] = {
                    'good_prob': adjusted_probs[0].item(),
                    'bad_prob': adjusted_probs[1].item(),
                    'confidence': max(adjusted_probs.tolist())
                }
        
        return results
    
    def get_name(self) -> str:
        return f"CQCNN(q={self.n_qubits},l={self.n_layers})"


# ================================================================================
# Part 4: モジュール3 - 報酬関数
# ================================================================================

class RewardFunction(ABC):
    """報酬関数の基底クラス"""
    
    @abstractmethod
    def calculate_move_reward(self, game_state: GameState, move: Tuple,
                            player: str, piece_info: Dict) -> float:
        """移動に対する報酬を計算"""
        pass
    
    @abstractmethod
    def calculate_state_reward(self, game_state: GameState, player: str) -> float:
        """状態に対する報酬を計算"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """報酬関数名を取得"""
        pass


class StandardRewardFunction(RewardFunction):
    """標準報酬関数"""
    
    def calculate_move_reward(self, game_state: GameState, move: Tuple,
                            player: str, piece_info: Dict) -> float:
        from_pos, to_pos = move
        reward = 0.0
        
        # 駒タイプを取得
        piece_type = piece_info.get(from_pos, 'unknown')
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        if piece_type == 'good':
            # 善玉：脱出を最優先
            escape_positions = [(5, 0), (5, 5)] if player == "A" else [(0, 0), (0, 5)]
            
            # 脱出口への接近
            min_dist_before = min(abs(from_pos[0] - ep[0]) + abs(from_pos[1] - ep[1]) 
                                 for ep in escape_positions)
            min_dist_after = min(abs(to_pos[0] - ep[0]) + abs(to_pos[1] - ep[1]) 
                                for ep in escape_positions)
            
            if min_dist_after < min_dist_before:
                reward += 2.0 * (min_dist_before - min_dist_after)
            
            # 脱出成功
            if to_pos in escape_positions:
                reward += 100.0
            
            # リスク評価（敵に隣接）
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (to_pos[0] + dx, to_pos[1] + dy)
                if (0 <= check_pos[0] < 6 and 0 <= check_pos[1] < 6 and
                    game_state.board[check_pos] == enemy_val):
                    reward -= 1.0
        
        elif piece_type == 'bad':
            # 悪玉：攻撃を重視
            if game_state.board[to_pos] == enemy_val:
                reward += 5.0
        
        # 共通：前進ボーナス
        if player == "A":
            reward += (to_pos[0] - from_pos[0]) * 0.3
        else:
            reward += (from_pos[0] - to_pos[0]) * 0.3
        
        return reward
    
    def calculate_state_reward(self, game_state: GameState, player: str) -> float:
        reward = 0.0
        
        # 駒数の差
        if player == "A":
            my_pieces = len(game_state.player_a_pieces)
            enemy_pieces = len(game_state.player_b_pieces)
        else:
            my_pieces = len(game_state.player_b_pieces)
            enemy_pieces = len(game_state.player_a_pieces)
        
        reward += (my_pieces - enemy_pieces) * 2.0
        
        # 勝敗
        if game_state.winner == player:
            reward += 1000.0
        elif game_state.winner and game_state.winner != player:
            reward -= 1000.0
        
        return reward
    
    def get_name(self) -> str:
        return "標準報酬"


class AggressiveRewardFunction(RewardFunction):
    """攻撃的報酬関数"""
    
    def calculate_move_reward(self, game_state: GameState, move: Tuple,
                            player: str, piece_info: Dict) -> float:
        from_pos, to_pos = move
        reward = 0.0
        
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        # 敵駒を取ることを最重視
        if game_state.board[to_pos] == enemy_val:
            reward += 10.0
        
        # 積極的な前進
        if player == "A":
            reward += (to_pos[0] - from_pos[0]) * 1.0
        else:
            reward += (from_pos[0] - to_pos[0]) * 1.0
        
        return reward
    
    def calculate_state_reward(self, game_state: GameState, player: str) -> float:
        return StandardRewardFunction().calculate_state_reward(game_state, player)
    
    def get_name(self) -> str:
        return "攻撃的報酬"


class DefensiveRewardFunction(RewardFunction):
    """防御的報酬関数"""
    
    def calculate_move_reward(self, game_state: GameState, move: Tuple,
                            player: str, piece_info: Dict) -> float:
        from_pos, to_pos = move
        reward = 0.0
        
        piece_type = piece_info.get(from_pos, 'unknown')
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        if piece_type == 'good':
            # 善玉の安全を最優先
            safe_distance = self._calculate_safety(to_pos, game_state, player)
            reward += safe_distance * 1.0
            
            # 慎重な脱出
            escape_positions = [(5, 0), (5, 5)] if player == "A" else [(0, 0), (0, 5)]
            if to_pos in escape_positions:
                if self._is_safe_position(to_pos, game_state, player):
                    reward += 100.0
                else:
                    reward -= 5.0
        
        return reward
    
    def _calculate_safety(self, pos: Tuple, game_state: GameState, player: str) -> float:
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        min_enemy_dist = 10
        for i in range(6):
            for j in range(6):
                if game_state.board[i, j] == enemy_val:
                    dist = abs(pos[0] - i) + abs(pos[1] - j)
                    min_enemy_dist = min(min_enemy_dist, dist)
        
        return min_enemy_dist
    
    def _is_safe_position(self, pos: Tuple, game_state: GameState, player: str) -> bool:
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            check_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= check_pos[0] < 6 and 0 <= check_pos[1] < 6 and
                game_state.board[check_pos] == enemy_val):
                return False
        return True
    
    def calculate_state_reward(self, game_state: GameState, player: str) -> float:
        reward = StandardRewardFunction().calculate_state_reward(game_state, player)
        
        # 善玉の生存ボーナス
        if player == "A":
            my_pieces = game_state.player_a_pieces
        else:
            my_pieces = game_state.player_b_pieces
        
        good_count = sum(1 for piece_type in my_pieces.values() if piece_type == 'good')
        reward += good_count * 5.0
        
        return reward
    
    def get_name(self) -> str:
        return "防御的報酬"


# ================================================================================
# Part 5: モジュール4 - Q値マップ生成器
# ================================================================================

class QMapGenerator(ABC):
    """Q値マップ生成器の基底クラス"""
    
    @abstractmethod
    def generate(self, board: np.ndarray, estimations: Dict,
                my_pieces: Dict, player: str, 
                reward_function: RewardFunction = None,
                game_state: GameState = None) -> np.ndarray:
        """Q値マップを生成"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """生成器名を取得"""
        pass


class SimpleQMapGenerator(QMapGenerator):
    """シンプルなQ値マップ生成器"""
    
    def generate(self, board: np.ndarray, estimations: Dict,
                my_pieces: Dict, player: str,
                reward_function: RewardFunction = None,
                game_state: GameState = None) -> np.ndarray:
        
        q_map = np.zeros((6, 6, 4))
        
        for pos, piece_type in my_pieces.items():
            base_value = 1.0 if piece_type == "good" else 0.5
            
            for i, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                if not (0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6):
                    q_map[pos[0], pos[1], i] = -100
                    continue
                
                q_value = base_value
                
                # 推定結果を使用
                if new_pos in estimations:
                    est = estimations[new_pos]
                    if piece_type == "bad":
                        q_value += est['good_prob'] * 2.0
                        q_value += est['bad_prob'] * 1.0
                    else:
                        q_value -= est['bad_prob'] * 0.5
                
                # 報酬関数を適用
                if reward_function and game_state:
                    move = (pos, new_pos)
                    reward = reward_function.calculate_move_reward(
                        game_state, move, player, my_pieces
                    )
                    q_value += reward * 0.1
                
                q_map[pos[0], pos[1], i] = q_value
        
        return q_map
    
    def get_name(self) -> str:
        return "シンプルQ値"


class StrategicQMapGenerator(QMapGenerator):
    """戦略的Q値マップ生成器"""
    
    def generate(self, board: np.ndarray, estimations: Dict,
                my_pieces: Dict, player: str,
                reward_function: RewardFunction = None,
                game_state: GameState = None) -> np.ndarray:
        
        q_map = np.zeros((6, 6, 4))
        
        # 脱出口の位置
        if player == "A":
            escape_positions = [(5, 0), (5, 5)]
        else:
            escape_positions = [(0, 0), (0, 5)]
        
        for pos, piece_type in my_pieces.items():
            for i, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                if not (0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6):
                    q_map[pos[0], pos[1], i] = -100
                    continue
                
                q_value = 0.0
                
                if piece_type == "good":
                    # 善玉：脱出戦略
                    min_dist_before = min(abs(pos[0] - ep[0]) + abs(pos[1] - ep[1]) 
                                         for ep in escape_positions)
                    min_dist_after = min(abs(new_pos[0] - ep[0]) + abs(new_pos[1] - ep[1]) 
                                        for ep in escape_positions)
                    
                    if min_dist_after < min_dist_before:
                        q_value += 3.0 + (min_dist_before - min_dist_after) * 1.5
                    
                    if new_pos in escape_positions:
                        q_value += 10.0
                    
                    if new_pos in estimations:
                        est = estimations[new_pos]
                        q_value -= est['bad_prob'] * 2.0
                        q_value += est['good_prob'] * 1.0
                
                else:  # bad
                    # 悪玉：攻撃戦略
                    if new_pos in estimations:
                        est = estimations[new_pos]
                        q_value += est['good_prob'] * 3.0
                        q_value += est['bad_prob'] * 1.5
                        q_value += est['confidence'] * 0.5
                    
                    # 相手の脱出口を守る
                    enemy_escape = [(0, 0), (0, 5)] if player == "A" else [(5, 0), (5, 5)]
                    for ep in enemy_escape:
                        if abs(new_pos[0] - ep[0]) + abs(new_pos[1] - ep[1]) <= 2:
                            q_value += 1.0
                
                # 報酬関数を統合
                if reward_function and game_state:
                    move = (pos, new_pos)
                    reward = reward_function.calculate_move_reward(
                        game_state, move, player, my_pieces
                    )
                    q_value += reward * 0.2
                
                q_map[pos[0], pos[1], i] = q_value
        
        return q_map
    
    def get_name(self) -> str:
        return "戦略的Q値"


# ================================================================================
# Part 6: モジュール5 - 行動選択器
# ================================================================================

class ActionSelector(ABC):
    """行動選択器の基底クラス"""
    
    @abstractmethod
    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        """行動を選択"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """選択器名を取得"""
        pass


class GreedySelector(ActionSelector):
    """貪欲選択器"""
    
    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        if not legal_moves:
            return None
        
        best_move = None
        best_value = -float('inf')
        
        for from_pos, to_pos in legal_moves:
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
    
    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        if not legal_moves:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        return GreedySelector().select(q_map, legal_moves)
    
    def get_name(self) -> str:
        return f"ε貪欲(ε={self.epsilon})"


class SoftmaxSelector(ActionSelector):
    """ソフトマックス選択器"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
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
                value = -100
            
            q_values.append(value / self.temperature)
        
        # ソフトマックス確率
        q_tensor = torch.tensor(q_values, dtype=torch.float32)
        probs = F.softmax(q_tensor, dim=0).numpy()
        
        # 確率的に選択
        idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[idx]
    
    def get_name(self) -> str:
        return f"ソフトマックス(T={self.temperature})"


# ================================================================================
# Part 7: エージェント統合
# ================================================================================

@dataclass
class ModuleConfig:
    """5つのモジュールを統合する設定"""
    placement_strategy: PlacementStrategy
    piece_estimator: PieceEstimator
    reward_function: RewardFunction
    qmap_generator: QMapGenerator
    action_selector: ActionSelector


class ModularAgent:
    """5モジュール統合エージェント"""
    
    def __init__(self, player_id: str, config: ModuleConfig):
        self.player_id = player_id
        self.config = config
        self.name = self._generate_name()
        self.piece_info = {}  # 自分の駒タイプを保持
        self.game_history = []
    
    def _generate_name(self) -> str:
        return f"Agent_{self.player_id}[" \
               f"{self.config.placement_strategy.get_name()[:3]}+" \
               f"{self.config.piece_estimator.get_name()[:6]}+" \
               f"{self.config.reward_function.get_name()[:3]}+" \
               f"{self.config.qmap_generator.get_name()[:3]}+" \
               f"{self.config.action_selector.get_name()[:3]}]"
    
    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        placement = self.config.placement_strategy.get_placement(self.player_id)
        self.piece_info = placement.copy()
        return placement
    
    def get_move(self, game_state: GameState, legal_moves: List[Tuple]) -> Tuple:
        """次の手を取得"""
        if not legal_moves:
            return None
        
        try:
            # 自分の駒情報を更新
            self._update_piece_info(game_state)
            
            # 1. 敵駒位置を特定
            enemy_positions = self._find_enemy_positions(game_state)
            
            # 2. 敵駒を推定（7チャンネル入力）
            estimations = {}
            if enemy_positions:
                estimations = self.config.piece_estimator.estimate(
                    board=game_state.board,
                    enemy_positions=enemy_positions,
                    player=self.player_id,
                    my_pieces=self.piece_info,
                    turn=game_state.turn
                )
            
            # 3. Q値マップを生成（報酬関数も統合）
            q_map = self.config.qmap_generator.generate(
                board=game_state.board,
                estimations=estimations,
                my_pieces=self.piece_info,
                player=self.player_id,
                reward_function=self.config.reward_function,
                game_state=game_state
            )
            
            # 4. 行動を選択
            selected_move = self.config.action_selector.select(q_map, legal_moves)
            
            # 履歴に記録
            self.game_history.append({
                'turn': game_state.turn,
                'move': selected_move,
                'estimations': len(estimations),
                'q_max': np.max(q_map) if q_map is not None else 0
            })
            
            return selected_move
            
        except Exception as e:
            print(f"⚠️ エラー in {self.name}: {e}")
            return random.choice(legal_moves)
    
    def _update_piece_info(self, game_state: GameState):
        """駒タイプ情報を更新"""
        if self.player_id == "A":
            current_pieces = game_state.player_a_pieces
        else:
            current_pieces = game_state.player_b_pieces
        
        # 現在の駒位置に合わせて更新
        new_piece_info = {}
        for pos, piece_type in current_pieces.items():
            # 元の駒タイプを保持
            for old_pos, old_type in self.piece_info.items():
                if old_type == piece_type:
                    new_piece_info[pos] = old_type
                    break
        
        self.piece_info = new_piece_info
    
    def _find_enemy_positions(self, game_state: GameState) -> List[Tuple[int, int]]:
        """敵駒の位置を特定"""
        enemy_val = -1 if self.player_id == "A" else 1
        positions = []
        
        for i in range(6):
            for j in range(6):
                if game_state.board[i, j] == enemy_val:
                    positions.append((i, j))
        
        return positions


# ================================================================================
# Part 8: ゲームエンジンと対戦システム
# ================================================================================

class GameEngine:
    """ゲームエンジン"""
    
    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        self.state = GameState()
        self.move_history = []
    
    def start_new_game(self, agent1: ModularAgent, agent2: ModularAgent):
        """新しいゲームを開始"""
        self.state = GameState()
        self.move_history = []
        
        # 初期配置
        placement1 = agent1.get_initial_placement()
        placement2 = agent2.get_initial_placement()
        
        for pos, piece_type in placement1.items():
            self.state.board[pos] = 1
            self.state.player_a_pieces[pos] = piece_type
        
        for pos, piece_type in placement2.items():
            self.state.board[pos] = -1
            self.state.player_b_pieces[pos] = piece_type
        
        return self.state
    
    def get_legal_moves(self, player: str) -> List[Tuple]:
        """合法手のリストを取得"""
        legal_moves = []
        pieces = self.state.player_a_pieces if player == "A" else self.state.player_b_pieces
        
        for pos in pieces.keys():
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
        
        # 敵駒を取る
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
        escape_positions = [(5, 0), (5, 5)] if player == "A" else [(0, 0), (0, 5)]
        if to_pos in escape_positions and piece_type == 'good':
            self.state.winner = player
        
        self.state.turn += 1
        
        # 履歴に記録
        self.move_history.append({
            'turn': self.state.turn,
            'player': player,
            'from': from_pos,
            'to': to_pos,
            'piece_type': piece_type
        })
    
    def check_winner(self) -> Optional[str]:
        """勝者を判定"""
        if self.state.winner:
            return self.state.winner
        
        # 善玉が全滅
        a_good = sum(1 for t in self.state.player_a_pieces.values() if t == 'good')
        b_good = sum(1 for t in self.state.player_b_pieces.values() if t == 'good')
        
        if a_good == 0:
            return "B"
        if b_good == 0:
            return "A"
        
        # ターン制限
        if self.state.turn >= self.config.max_turns:
            if a_good > b_good:
                return "A"
            elif b_good > a_good:
                return "B"
            else:
                return "Draw"
        
        return None


class BattleSystem:
    """対戦システム"""
    
    def __init__(self):
        self.engine = GameEngine()
        self.results = []
    
    def run_match(self, agent1: ModularAgent, agent2: ModularAgent, 
                  verbose: bool = False) -> Dict:
        """1試合を実行"""
        # ゲーム開始
        self.engine.start_new_game(agent1, agent2)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"対戦: {agent1.name} vs {agent2.name}")
            print(f"{'='*60}")
        
        # ゲームループ
        current_player = "A"
        current_agent = agent1
        
        while not self.engine.state.is_game_over():
            # 合法手を取得
            legal_moves = self.engine.get_legal_moves(current_player)
            
            if not legal_moves:
                break
            
            # 手を選択
            move = current_agent.get_move(self.engine.state, legal_moves)
            
            if move:
                self.engine.make_move(move[0], move[1], current_player)
                
                if verbose and self.engine.state.turn % 10 == 0:
                    print(f"Turn {self.engine.state.turn}: "
                          f"Player {current_player} moved {move[0]}→{move[1]}")
            
            # 勝者判定
            winner = self.engine.check_winner()
            if winner:
                self.engine.state.winner = winner
                break
            
            # プレイヤー交代
            if current_player == "A":
                current_player = "B"
                current_agent = agent2
            else:
                current_player = "A"
                current_agent = agent1
        
        # 結果記録
        result = {
            'winner': self.engine.state.winner,
            'turns': self.engine.state.turn,
            'agent1': agent1.name,
            'agent2': agent2.name,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        if verbose:
            print(f"\n勝者: {result['winner']}, ターン数: {result['turns']}")
        
        return result


# ================================================================================
# Part 9: メインプログラム
# ================================================================================

def main():
    """メインプログラム"""
    print("="*70)
    print("🌟 CQCNN対戦システム v2 - 5モジュール設計")
    print("="*70)
    
    # 利用可能なモジュール
    modules = {
        'placement': [
            StandardPlacement(),
            AggressivePlacement(),
            DefensivePlacement()
        ],
        'estimator': [
            RandomEstimator(),
            SimpleCNNEstimator(),
            CQCNNEstimator(n_qubits=4, n_layers=2),
            CQCNNEstimator(n_qubits=6, n_layers=3)
        ],
        'reward': [
            StandardRewardFunction(),
            AggressiveRewardFunction(),
            DefensiveRewardFunction()
        ],
        'qmap': [
            SimpleQMapGenerator(),
            StrategicQMapGenerator()
        ],
        'selector': [
            GreedySelector(),
            EpsilonGreedySelector(epsilon=0.1),
            EpsilonGreedySelector(epsilon=0.3),
            SoftmaxSelector(temperature=1.0)
        ]
    }
    
    # サンプル構成
    print("\n📋 サンプル構成:")
    print("1. バランス型")
    print("2. 攻撃型")
    print("3. 防御型")
    print("4. カスタム")
    
    choice = input("\n選択 (1-4): ")
    
    if choice == "1":
        # バランス型
        config1 = ModuleConfig(
            placement_strategy=StandardPlacement(),
            piece_estimator=CQCNNEstimator(n_qubits=6, n_layers=3),
            reward_function=StandardRewardFunction(),
            qmap_generator=StrategicQMapGenerator(),
            action_selector=EpsilonGreedySelector(epsilon=0.1)
        )
        config2 = ModuleConfig(
            placement_strategy=StandardPlacement(),
            piece_estimator=SimpleCNNEstimator(),
            reward_function=StandardRewardFunction(),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=GreedySelector()
        )
        
    elif choice == "2":
        # 攻撃型
        config1 = ModuleConfig(
            placement_strategy=AggressivePlacement(),
            piece_estimator=CQCNNEstimator(n_qubits=4, n_layers=2),
            reward_function=AggressiveRewardFunction(),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=GreedySelector()
        )
        config2 = ModuleConfig(
            placement_strategy=StandardPlacement(),
            piece_estimator=RandomEstimator(),
            reward_function=StandardRewardFunction(),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=EpsilonGreedySelector(epsilon=0.3)
        )
        
    elif choice == "3":
        # 防御型
        config1 = ModuleConfig(
            placement_strategy=DefensivePlacement(),
            piece_estimator=CQCNNEstimator(n_qubits=6, n_layers=3),
            reward_function=DefensiveRewardFunction(),
            qmap_generator=StrategicQMapGenerator(),
            action_selector=SoftmaxSelector(temperature=1.0)
        )
        config2 = ModuleConfig(
            placement_strategy=AggressivePlacement(),
            piece_estimator=SimpleCNNEstimator(),
            reward_function=AggressiveRewardFunction(),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=GreedySelector()
        )
        
    else:
        # カスタム構成
        print("\n🔧 モジュールを選択:")
        
        print("\n初期配置戦略:")
        for i, m in enumerate(modules['placement']):
            print(f"  {i}: {m.get_name()}")
        p1 = int(input("Agent1の配置戦略: "))
        p2 = int(input("Agent2の配置戦略: "))
        
        print("\n敵駒推定器:")
        for i, m in enumerate(modules['estimator']):
            print(f"  {i}: {m.get_name()}")
        e1 = int(input("Agent1の推定器: "))
        e2 = int(input("Agent2の推定器: "))
        
        print("\n報酬関数:")
        for i, m in enumerate(modules['reward']):
            print(f"  {i}: {m.get_name()}")
        r1 = int(input("Agent1の報酬関数: "))
        r2 = int(input("Agent2の報酬関数: "))
        
        print("\nQ値マップ生成器:")
        for i, m in enumerate(modules['qmap']):
            print(f"  {i}: {m.get_name()}")
        q1 = int(input("Agent1のQ値生成器: "))
        q2 = int(input("Agent2のQ値生成器: "))
        
        print("\n行動選択器:")
        for i, m in enumerate(modules['selector']):
            print(f"  {i}: {m.get_name()}")
        s1 = int(input("Agent1の選択器: "))
        s2 = int(input("Agent2の選択器: "))
        
        config1 = ModuleConfig(
            placement_strategy=modules['placement'][p1],
            piece_estimator=modules['estimator'][e1],
            reward_function=modules['reward'][r1],
            qmap_generator=modules['qmap'][q1],
            action_selector=modules['selector'][s1]
        )
        config2 = ModuleConfig(
            placement_strategy=modules['placement'][p2],
            piece_estimator=modules['estimator'][e2],
            reward_function=modules['reward'][r2],
            qmap_generator=modules['qmap'][q2],
            action_selector=modules['selector'][s2]
        )
    
    # エージェント作成
    agent1 = ModularAgent("A", config1)
    agent2 = ModularAgent("B", config2)
    
    print(f"\n🎮 対戦設定完了:")
    print(f"Agent1: {agent1.name}")
    print(f"Agent2: {agent2.name}")
    
    # 対戦実行
    n_games = int(input("\n対戦数: "))
    
    battle_system = BattleSystem()
    wins = {'A': 0, 'B': 0, 'Draw': 0}
    
    for i in range(n_games):
        print(f"\nGame {i+1}/{n_games}")
        result = battle_system.run_match(agent1, agent2, verbose=(i == 0))
        
        if result['winner'] == 'A':
            wins['A'] += 1
        elif result['winner'] == 'B':
            wins['B'] += 1
        else:
            wins['Draw'] += 1
    
    # 結果表示
    print("\n" + "="*70)
    print("📊 最終結果:")
    print(f"Agent1勝利: {wins['A']} ({wins['A']/n_games*100:.1f}%)")
    print(f"Agent2勝利: {wins['B']} ({wins['B']/n_games*100:.1f}%)")
    print(f"引き分け: {wins['Draw']} ({wins['Draw']/n_games*100:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()

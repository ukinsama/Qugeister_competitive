#!/usr/bin/env python3
"""
ゲーム状態管理モジュール
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class GameConfig:
    """ゲーム設定"""
    board_size: tuple = (6, 6)
    max_turns: int = 100
    n_pieces: int = 8
    n_good: int = 4
    n_bad: int = 4


@dataclass 
class LearningConfig:
    """学習設定"""
    # 共通設定
    learning_rate: float = 0.001
    batch_size: int = 32
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    # 教師あり学習設定
    supervised_epochs: int = 100
    validation_split: float = 0.2
    
    # 強化学習設定
    rl_episodes: int = 1000
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 500
    gamma: float = 0.95
    memory_size: int = 10000
    target_update: int = 10


class GameState:
    """ゲーム状態管理"""
    
    def __init__(self):
        self.board = np.zeros((6, 6), dtype=int)
        self.player_a_pieces = {}
        self.player_b_pieces = {}
        self.turn = 0
        self.winner = None
    
    def is_game_over(self) -> bool:
        """ゲーム終了判定"""
        return self.winner is not None or self.turn >= 100
    
    def get_valid_actions(self, player_pieces: Dict[Tuple[int, int], str]) -> list:
        """有効な行動を取得"""
        valid_actions = []
        for pos, piece_type in player_pieces.items():
            r, c = pos
            # 4方向の移動をチェック
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < 6 and 0 <= new_c < 6:
                    valid_actions.append((pos, (new_r, new_c)))
        return valid_actions
    
    def apply_action(self, action: Tuple[Tuple[int, int], Tuple[int, int]], player: str):
        """行動を適用"""
        from_pos, to_pos = action
        player_pieces = self.player_a_pieces if player == 'A' else self.player_b_pieces
        
        if from_pos in player_pieces:
            piece_type = player_pieces.pop(from_pos)
            player_pieces[to_pos] = piece_type
            
            # ボード更新
            self.board[from_pos] = 0
            piece_value = 1 if (player == 'A' and piece_type == 'good') else \
                         2 if (player == 'A' and piece_type == 'bad') else \
                         3 if (player == 'B' and piece_type == 'good') else 4
            self.board[to_pos] = piece_value
            
        self.turn += 1
    
    def copy(self):
        """ゲーム状態をコピー"""
        new_state = GameState()
        new_state.board = self.board.copy()
        new_state.player_a_pieces = self.player_a_pieces.copy()
        new_state.player_b_pieces = self.player_b_pieces.copy()
        new_state.turn = self.turn
        new_state.winner = self.winner
        return new_state
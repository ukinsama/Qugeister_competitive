#!/usr/bin/env python3
"""
報酬関数モジュール
"""

import numpy as np
from typing import Dict, Tuple
from ..core.base_modules import RewardFunction
from ..core.game_state import GameState


class BasicReward(RewardFunction):
    """基本報酬関数"""
    
    def calculate_reward(
        self, 
        state_before: GameState, 
        action: Tuple, 
        state_after: GameState,
        player: str
    ) -> float:
        """基本的な報酬を計算"""
        reward = 0.0
        
        # 生存ボーナス
        reward += 0.1
        
        # ゲーム終了判定
        if state_after.is_game_over():
            if state_after.winner == player:
                reward += 100.0  # 勝利
            elif state_after.winner is not None:
                reward -= 100.0  # 敗北
            else:
                reward += 10.0   # 引き分け
        
        # 駒の捕獲
        my_pieces_before = len(state_before.player_a_pieces if player == 'A' else state_before.player_b_pieces)
        my_pieces_after = len(state_after.player_a_pieces if player == 'A' else state_after.player_b_pieces)
        
        if my_pieces_after < my_pieces_before:
            reward -= 10.0  # 駒を失った
        
        return reward
    
    def get_name(self) -> str:
        return "基本報酬"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "basic", "survival_bonus": 0.1}
    
    def process(self, state_before: GameState, action: Tuple, state_after: GameState, player: str) -> float:
        return self.calculate_reward(state_before, action, state_after, player)


class AggressiveReward(RewardFunction):
    """攻撃的報酬関数"""
    
    def calculate_reward(
        self, 
        state_before: GameState, 
        action: Tuple, 
        state_after: GameState,
        player: str
    ) -> float:
        """攻撃的な報酬を計算"""
        reward = 0.0
        
        # 基本報酬
        basic_reward = BasicReward()
        reward += basic_reward.calculate_reward(state_before, action, state_after, player)
        
        # 前進ボーナス
        from_pos, to_pos = action
        if player == 'A':
            if to_pos[0] > from_pos[0]:  # 下に移動
                reward += 2.0
        else:
            if to_pos[0] < from_pos[0]:  # 上に移動
                reward += 2.0
        
        # 中央制圧ボーナス
        center_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
        if to_pos in center_positions:
            reward += 1.5
        
        # 敵駒接近ボーナス
        enemy_pieces = state_after.player_b_pieces if player == 'A' else state_after.player_a_pieces
        for enemy_pos in enemy_pieces.keys():
            distance = abs(to_pos[0] - enemy_pos[0]) + abs(to_pos[1] - enemy_pos[1])
            if distance <= 2:
                reward += 1.0
        
        return reward
    
    def get_name(self) -> str:
        return "攻撃的報酬"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "aggressive", "forward_bonus": 2.0, "center_bonus": 1.5}
    
    def process(self, state_before: GameState, action: Tuple, state_after: GameState, player: str) -> float:
        return self.calculate_reward(state_before, action, state_after, player)


class DefensiveReward(RewardFunction):
    """防御的報酬関数"""
    
    def calculate_reward(
        self, 
        state_before: GameState, 
        action: Tuple, 
        state_after: GameState,
        player: str
    ) -> float:
        """防御的な報酬を計算"""
        reward = 0.0
        
        # 基本報酬
        basic_reward = BasicReward()
        reward += basic_reward.calculate_reward(state_before, action, state_after, player)
        
        # 後退ペナルティ軽減（防御的戦略では後退も有効）
        from_pos, to_pos = action
        if player == 'A':
            if to_pos[0] < from_pos[0]:  # 上に移動（後退）
                reward += 1.0
        else:
            if to_pos[0] > from_pos[0]:  # 下に移動（後退）
                reward += 1.0
        
        # 端への移動ボーナス（守備的位置）
        if to_pos[1] == 0 or to_pos[1] == 5:  # 左右の端
            reward += 1.5
        
        # 自駒同士の距離を保つボーナス
        my_pieces = state_after.player_a_pieces if player == 'A' else state_after.player_b_pieces
        total_distance = 0
        piece_count = 0
        
        for pos1 in my_pieces.keys():
            for pos2 in my_pieces.keys():
                if pos1 != pos2:
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    total_distance += distance
                    piece_count += 1
        
        if piece_count > 0:
            avg_distance = total_distance / piece_count
            if avg_distance > 3:  # 適度に分散
                reward += 1.0
        
        return reward
    
    def get_name(self) -> str:
        return "防御的報酬"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "defensive", "retreat_bonus": 1.0, "edge_bonus": 1.5}
    
    def process(self, state_before: GameState, action: Tuple, state_after: GameState, player: str) -> float:
        return self.calculate_reward(state_before, action, state_after, player)


class EscapeReward(RewardFunction):
    """脱出重視報酬関数"""
    
    def calculate_reward(
        self, 
        state_before: GameState, 
        action: Tuple, 
        state_after: GameState,
        player: str
    ) -> float:
        """脱出重視の報酬を計算"""
        reward = 0.0
        
        # 基本報酬
        basic_reward = BasicReward()
        reward += basic_reward.calculate_reward(state_before, action, state_after, player)
        
        from_pos, to_pos = action
        my_pieces = state_after.player_a_pieces if player == 'A' else state_after.player_b_pieces
        
        # 脱出口への接近ボーナス
        escape_positions = [(5, 0), (5, 5)] if player == 'A' else [(0, 0), (0, 5)]
        
        if from_pos in my_pieces and my_pieces[from_pos] == 'good':
            for escape_pos in escape_positions:
                distance_before = abs(from_pos[0] - escape_pos[0]) + abs(from_pos[1] - escape_pos[1])
                distance_after = abs(to_pos[0] - escape_pos[0]) + abs(to_pos[1] - escape_pos[1])
                
                if distance_after < distance_before:
                    reward += 5.0  # 脱出口に接近
                
                # 脱出成功の超高ボーナス
                if to_pos in escape_positions:
                    reward += 50.0
        
        # 善玉駒の生存ボーナス
        good_pieces = sum(1 for piece_type in my_pieces.values() if piece_type == 'good')
        reward += good_pieces * 0.5
        
        return reward
    
    def get_name(self) -> str:
        return "脱出重視報酬"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "escape", "approach_bonus": 5.0, "escape_bonus": 50.0}
    
    def process(self, state_before: GameState, action: Tuple, state_after: GameState, player: str) -> float:
        return self.calculate_reward(state_before, action, state_after, player)


# 報酬関数ファクトリ
class RewardFactory:
    """報酬関数ファクトリ"""
    
    @staticmethod
    def create_reward(reward_type: str, **kwargs) -> RewardFunction:
        """報酬関数を作成"""
        if reward_type == "basic":
            return BasicReward()
        elif reward_type == "aggressive":
            return AggressiveReward()
        elif reward_type == "defensive":
            return DefensiveReward()
        elif reward_type == "escape":
            return EscapeReward()
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
    
    @staticmethod
    def get_available_rewards() -> list:
        """利用可能な報酬関数一覧"""
        return ["basic", "aggressive", "defensive", "escape"]
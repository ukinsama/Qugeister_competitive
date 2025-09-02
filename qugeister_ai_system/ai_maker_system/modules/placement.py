#!/usr/bin/env python3
"""
初期配置戦略モジュール
"""

import random
from typing import Dict, Tuple
from ..core.base_modules import PlacementStrategy


class StandardPlacement(PlacementStrategy):
    """標準配置戦略"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            positions = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4)]
        else:
            positions = [(4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
        
        piece_types = ["good"] * 4 + ["bad"] * 4
        random.shuffle(piece_types)
        
        for pos, piece_type in zip(positions, piece_types):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "標準配置"
    
    def get_config(self) -> Dict[str, any]:
        return {"strategy": "standard", "randomize": True}
    
    def process(self, player_id: str):
        return self.get_placement(player_id)


class AggressivePlacement(PlacementStrategy):
    """攻撃的配置戦略（善玉を前線に）"""
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}
        
        if player_id == "A":
            front_positions = [(1, 1), (1, 2), (1, 3), (1, 4)]
            back_positions = [(0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            front_positions = [(4, 1), (4, 2), (4, 3), (4, 4)]
            back_positions = [(5, 1), (5, 2), (5, 3), (5, 4)]
        
        front_pieces = ["good", "good", "good", "bad"]
        random.shuffle(front_pieces)
        
        back_pieces = ["good", "bad", "bad", "bad"]
        random.shuffle(back_pieces)
        
        for pos, piece_type in zip(front_positions, front_pieces):
            placement[pos] = piece_type
        for pos, piece_type in zip(back_positions, back_pieces):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "攻撃的配置"
    
    def get_config(self) -> Dict[str, any]:
        return {"strategy": "aggressive", "good_forward_ratio": 0.75}
    
    def process(self, player_id: str):
        return self.get_placement(player_id)


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
        
        front_pieces = ["bad", "bad", "bad", "good"]
        random.shuffle(front_pieces)
        
        back_pieces = ["bad", "good", "good", "good"]
        random.shuffle(back_pieces)
        
        for pos, piece_type in zip(front_positions, front_pieces):
            placement[pos] = piece_type
        for pos, piece_type in zip(back_positions, back_pieces):
            placement[pos] = piece_type
        
        return placement
    
    def get_name(self) -> str:
        return "防御的配置"
    
    def get_config(self) -> Dict[str, any]:
        return {"strategy": "defensive", "good_back_ratio": 0.75}
    
    def process(self, player_id: str):
        return self.get_placement(player_id)


class CustomPlacement(PlacementStrategy):
    """カスタム配置戦略"""
    
    def __init__(self, custom_placement: Dict[Tuple[int, int], str]):
        self.custom_placement = custom_placement
    
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        return self.custom_placement.copy()
    
    def get_name(self) -> str:
        return "カスタム配置"
    
    def get_config(self) -> Dict[str, any]:
        return {"strategy": "custom", "placement": self.custom_placement}
    
    def process(self, player_id: str):
        return self.get_placement(player_id)


# プリセット配置のファクトリ
class PlacementFactory:
    """配置戦略ファクトリ"""
    
    @staticmethod
    def create_placement(strategy_type: str, **kwargs) -> PlacementStrategy:
        """配置戦略を作成"""
        if strategy_type == "standard":
            return StandardPlacement()
        elif strategy_type == "aggressive":
            return AggressivePlacement()
        elif strategy_type == "defensive":
            return DefensivePlacement()
        elif strategy_type == "custom":
            custom_placement = kwargs.get("placement", {})
            return CustomPlacement(custom_placement)
        else:
            raise ValueError(f"Unknown placement strategy: {strategy_type}")
    
    @staticmethod
    def get_available_strategies() -> list:
        """利用可能な戦略一覧"""
        return ["standard", "aggressive", "defensive", "custom"]
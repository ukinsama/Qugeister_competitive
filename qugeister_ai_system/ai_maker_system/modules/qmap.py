#!/usr/bin/env python3
"""
Q値マップ生成器モジュール
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple
from ..core.base_modules import QMapGenerator
from ..core.game_state import GameState


class SimpleQMapGenerator(QMapGenerator):
    """シンプルQ値マップ生成器"""
    
    def generate_qmap(
        self, 
        tensor: torch.Tensor, 
        valid_actions: List[Tuple]
    ) -> Dict[Tuple, float]:
        """シンプルなQ値マップを生成"""
        qmap = {}
        
        # 各行動に基本的なQ値を割り当て
        for action in valid_actions:
            from_pos, to_pos = action
            
            # 基本Q値（ランダム）
            base_q = random.uniform(0.0, 1.0)
            
            # 位置ベースの調整
            # 中央に近いほど高い値
            center_bonus = 1.0 - (abs(to_pos[0] - 2.5) + abs(to_pos[1] - 2.5)) / 5.0
            
            # 移動距離ペナルティ
            distance_penalty = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
            
            qvalue = base_q + center_bonus * 0.3 - distance_penalty * 0.1
            qmap[action] = qvalue
        
        return qmap
    
    def get_name(self) -> str:
        return "シンプルQ値マップ"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "simple", "center_bonus": 0.3}
    
    def process(self, tensor: torch.Tensor, valid_actions: List[Tuple]) -> Dict[Tuple, float]:
        return self.generate_qmap(tensor, valid_actions)


class StrategicQMapGenerator(QMapGenerator):
    """戦略的Q値マップ生成器"""
    
    def __init__(self, strategy: str = "balanced"):
        self.strategy = strategy
    
    def generate_qmap(
        self, 
        tensor: torch.Tensor, 
        valid_actions: List[Tuple]
    ) -> Dict[Tuple, float]:
        """戦略的なQ値マップを生成"""
        qmap = {}
        
        for action in valid_actions:
            from_pos, to_pos = action
            
            # 基本Q値
            base_q = random.uniform(0.5, 1.0)
            
            # 戦略別調整
            if self.strategy == "aggressive":
                qvalue = self._aggressive_adjustment(base_q, from_pos, to_pos)
            elif self.strategy == "defensive":
                qvalue = self._defensive_adjustment(base_q, from_pos, to_pos)
            elif self.strategy == "escape":
                qvalue = self._escape_adjustment(base_q, from_pos, to_pos)
            else:
                qvalue = base_q
            
            qmap[action] = qvalue
        
        return qmap
    
    def _aggressive_adjustment(self, base_q: float, from_pos: Tuple, to_pos: Tuple) -> float:
        """攻撃的戦略の調整"""
        qvalue = base_q
        
        # 前進ボーナス
        if to_pos[0] > from_pos[0]:  # 下に移動（プレイヤーAの場合）
            qvalue += 0.5
        
        # 中央制圧ボーナス
        center_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
        if to_pos in center_positions:
            qvalue += 0.3
        
        return qvalue
    
    def _defensive_adjustment(self, base_q: float, from_pos: Tuple, to_pos: Tuple) -> float:
        """防御的戦略の調整"""
        qvalue = base_q
        
        # 端への移動ボーナス
        if to_pos[1] == 0 or to_pos[1] == 5:
            qvalue += 0.4
        
        # 後退ペナルティ軽減
        if to_pos[0] < from_pos[0]:  # 上に移動（後退）
            qvalue += 0.2  # 通常はペナルティだが、防御戦略では軽減
        
        return qvalue
    
    def _escape_adjustment(self, base_q: float, from_pos: Tuple, to_pos: Tuple) -> float:
        """脱出戦略の調整"""
        qvalue = base_q
        
        # 脱出口への接近ボーナス
        escape_positions = [(5, 0), (5, 5), (0, 0), (0, 5)]  # 4つの角
        
        for escape_pos in escape_positions:
            distance_before = abs(from_pos[0] - escape_pos[0]) + abs(from_pos[1] - escape_pos[1])
            distance_after = abs(to_pos[0] - escape_pos[0]) + abs(to_pos[1] - escape_pos[1])
            
            if distance_after < distance_before:
                qvalue += 0.6  # 脱出口に近づく
            
            # 脱出成功の超高ボーナス
            if to_pos in escape_positions:
                qvalue += 2.0
        
        return qvalue
    
    def get_name(self) -> str:
        return f"戦略的Q値マップ ({self.strategy})"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "strategic", "strategy": self.strategy}
    
    def process(self, tensor: torch.Tensor, valid_actions: List[Tuple]) -> Dict[Tuple, float]:
        return self.generate_qmap(tensor, valid_actions)


class LearnedQMapGenerator(QMapGenerator):
    """学習済みQ値マップ生成器（ニューラルネットワーク使用）"""
    
    def __init__(self, model: torch.nn.Module = None):
        self.model = model
        self.is_trained = model is not None
    
    def generate_qmap(
        self, 
        tensor: torch.Tensor, 
        valid_actions: List[Tuple]
    ) -> Dict[Tuple, float]:
        """学習済みモデルでQ値マップを生成"""
        qmap = {}
        
        if not self.is_trained or self.model is None:
            # 訓練されていない場合はランダム
            for action in valid_actions:
                qmap[action] = random.uniform(0.0, 1.0)
            return qmap
        
        try:
            # モデルで予測
            with torch.no_grad():
                self.model.eval()
                output = self.model(tensor)
                
                # 出力をQ値に変換（簡略化）
                if output.dim() == 2:
                    q_values = torch.softmax(output, dim=1).cpu().numpy()[0]
                else:
                    q_values = [0.5] * len(valid_actions)
                
                # 各行動にQ値を割り当て
                for i, action in enumerate(valid_actions):
                    if i < len(q_values):
                        qmap[action] = float(q_values[i])
                    else:
                        qmap[action] = 0.5
        
        except Exception as e:
            print(f"学習済みQ値生成エラー: {e}")
            # エラー時はランダム値
            for action in valid_actions:
                qmap[action] = random.uniform(0.0, 1.0)
        
        return qmap
    
    def set_model(self, model: torch.nn.Module):
        """モデルを設定"""
        self.model = model
        self.is_trained = True
    
    def get_name(self) -> str:
        return "学習済みQ値マップ"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "learned", "is_trained": self.is_trained}
    
    def process(self, tensor: torch.Tensor, valid_actions: List[Tuple]) -> Dict[Tuple, float]:
        return self.generate_qmap(tensor, valid_actions)


# Q値マップ生成器ファクトリ
class QMapFactory:
    """Q値マップ生成器ファクトリ"""
    
    @staticmethod
    def create_qmap_generator(generator_type: str, **kwargs) -> QMapGenerator:
        """Q値マップ生成器を作成"""
        if generator_type == "simple":
            return SimpleQMapGenerator()
        elif generator_type == "strategic":
            strategy = kwargs.get("strategy", "balanced")
            return StrategicQMapGenerator(strategy)
        elif generator_type == "learned":
            model = kwargs.get("model", None)
            return LearnedQMapGenerator(model)
        else:
            raise ValueError(f"Unknown qmap generator type: {generator_type}")
    
    @staticmethod
    def get_available_generators() -> list:
        """利用可能な生成器一覧"""
        return ["simple", "strategic", "learned"]
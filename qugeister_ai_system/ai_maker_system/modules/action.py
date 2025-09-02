#!/usr/bin/env python3
"""
行動選択器モジュール
"""

import random
import numpy as np
from typing import Dict, List, Tuple
from ..core.base_modules import ActionSelector


class GreedySelector(ActionSelector):
    """貪欲選択器 - 常に最高Q値の行動を選択"""
    
    def select_action(
        self, 
        qmap: Dict[Tuple, float], 
        valid_actions: List[Tuple],
        epsilon: float = 0.0
    ) -> Tuple:
        """最高Q値の行動を選択"""
        if not valid_actions:
            return None
        
        if not qmap:
            return random.choice(valid_actions)
        
        # Q値が最大の行動を選択
        best_action = max(valid_actions, key=lambda action: qmap.get(action, 0.0))
        return best_action
    
    def get_name(self) -> str:
        return "貪欲選択"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "greedy"}
    
    def process(self, qmap: Dict[Tuple, float], valid_actions: List[Tuple], epsilon: float = 0.0) -> Tuple:
        return self.select_action(qmap, valid_actions, epsilon)


class EpsilonGreedySelector(ActionSelector):
    """ε-貪欲選択器 - 確率εでランダム、1-εで貪欲選択"""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
    
    def select_action(
        self, 
        qmap: Dict[Tuple, float], 
        valid_actions: List[Tuple],
        epsilon: float = None
    ) -> Tuple:
        """ε-貪欲で行動を選択"""
        if not valid_actions:
            return None
        
        # epsilonが指定されていない場合はデフォルト値を使用
        eps = epsilon if epsilon is not None else self.epsilon
        
        if random.random() < eps:
            # ランダム選択
            return random.choice(valid_actions)
        else:
            # 貪欲選択
            if not qmap:
                return random.choice(valid_actions)
            
            best_action = max(valid_actions, key=lambda action: qmap.get(action, 0.0))
            return best_action
    
    def get_name(self) -> str:
        return f"ε-貪欲選択 (ε={self.epsilon})"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "epsilon_greedy", "epsilon": self.epsilon}
    
    def process(self, qmap: Dict[Tuple, float], valid_actions: List[Tuple], epsilon: float = None) -> Tuple:
        return self.select_action(qmap, valid_actions, epsilon)


class BoltzmannSelector(ActionSelector):
    """ボルツマン選択器 - 温度パラメータに基づく確率的選択"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def select_action(
        self, 
        qmap: Dict[Tuple, float], 
        valid_actions: List[Tuple],
        epsilon: float = 0.0
    ) -> Tuple:
        """ボルツマン分布で行動を選択"""
        if not valid_actions:
            return None
        
        if not qmap:
            return random.choice(valid_actions)
        
        # Q値を取得
        q_values = [qmap.get(action, 0.0) for action in valid_actions]
        
        if self.temperature <= 0.001:
            # 温度が非常に低い場合は貪欲選択
            best_idx = np.argmax(q_values)
            return valid_actions[best_idx]
        
        # ボルツマン分布で確率計算
        exp_values = np.exp(np.array(q_values) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # 確率に基づいて選択
        chosen_idx = np.random.choice(len(valid_actions), p=probabilities)
        return valid_actions[chosen_idx]
    
    def get_name(self) -> str:
        return f"ボルツマン選択 (T={self.temperature})"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "boltzmann", "temperature": self.temperature}
    
    def process(self, qmap: Dict[Tuple, float], valid_actions: List[Tuple], epsilon: float = 0.0) -> Tuple:
        return self.select_action(qmap, valid_actions, epsilon)


class UCBSelector(ActionSelector):
    """UCB選択器 - Upper Confidence Bound"""
    
    def __init__(self, c: float = 1.0):
        self.c = c  # 探索パラメータ
        self.action_counts = {}  # 各行動の選択回数
        self.total_count = 0
    
    def select_action(
        self, 
        qmap: Dict[Tuple, float], 
        valid_actions: List[Tuple],
        epsilon: float = 0.0
    ) -> Tuple:
        """UCBで行動を選択"""
        if not valid_actions:
            return None
        
        if not qmap:
            return random.choice(valid_actions)
        
        self.total_count += 1
        
        # まだ選択されていない行動があれば優先
        for action in valid_actions:
            if action not in self.action_counts:
                self.action_counts[action] = 0
                return action
        
        # UCB値を計算
        ucb_values = {}
        for action in valid_actions:
            count = self.action_counts.get(action, 1)
            q_value = qmap.get(action, 0.0)
            
            # UCB = Q値 + c * sqrt(ln(総回数) / 行動回数)
            confidence = self.c * np.sqrt(np.log(self.total_count) / count)
            ucb_values[action] = q_value + confidence
        
        # UCB値が最大の行動を選択
        best_action = max(valid_actions, key=lambda action: ucb_values.get(action, 0.0))
        self.action_counts[best_action] = self.action_counts.get(best_action, 0) + 1
        
        return best_action
    
    def get_name(self) -> str:
        return f"UCB選択 (c={self.c})"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "ucb", "c": self.c, "total_count": self.total_count}
    
    def process(self, qmap: Dict[Tuple, float], valid_actions: List[Tuple], epsilon: float = 0.0) -> Tuple:
        return self.select_action(qmap, valid_actions, epsilon)


class RandomSelector(ActionSelector):
    """ランダム選択器 - 完全にランダムな選択"""
    
    def select_action(
        self, 
        qmap: Dict[Tuple, float], 
        valid_actions: List[Tuple],
        epsilon: float = 0.0
    ) -> Tuple:
        """ランダムに行動を選択"""
        if not valid_actions:
            return None
        
        return random.choice(valid_actions)
    
    def get_name(self) -> str:
        return "ランダム選択"
    
    def get_config(self) -> Dict[str, any]:
        return {"type": "random"}
    
    def process(self, qmap: Dict[Tuple, float], valid_actions: List[Tuple], epsilon: float = 0.0) -> Tuple:
        return self.select_action(qmap, valid_actions, epsilon)


# 行動選択器ファクトリ
class ActionFactory:
    """行動選択器ファクトリ"""
    
    @staticmethod
    def create_selector(selector_type: str, **kwargs) -> ActionSelector:
        """行動選択器を作成"""
        if selector_type == "greedy":
            return GreedySelector()
        elif selector_type == "epsilon_greedy":
            epsilon = kwargs.get("epsilon", 0.1)
            return EpsilonGreedySelector(epsilon)
        elif selector_type == "boltzmann":
            temperature = kwargs.get("temperature", 1.0)
            return BoltzmannSelector(temperature)
        elif selector_type == "ucb":
            c = kwargs.get("c", 1.0)
            return UCBSelector(c)
        elif selector_type == "random":
            return RandomSelector()
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")
    
    @staticmethod
    def get_available_selectors() -> list:
        """利用可能な選択器一覧"""
        return ["greedy", "epsilon_greedy", "boltzmann", "ucb", "random"]
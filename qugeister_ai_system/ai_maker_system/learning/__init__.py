#!/usr/bin/env python3
"""
学習システムモジュール
"""

from .supervised import SupervisedLearning
from .reinforcement import DQNReinforcementLearning

__all__ = [
    'SupervisedLearning',
    'DQNReinforcementLearning'
]
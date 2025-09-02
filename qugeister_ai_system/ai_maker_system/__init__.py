#!/usr/bin/env python3
"""
AI Maker System - モジュール化されたAI制作システム
cqcnn_battle_learning_systemをベースにしたモジュール式AI生成

構成:
- core/: 基本システム・データ処理
- modules/: 5つの機能モジュール
- learning/: 学習システム
- ai_builder.py: AI組み立てシステム
"""

from .core.base_modules import *
from .core.data_processor import DataProcessor
from .core.game_state import GameState, GameConfig
from .ai_builder import AIBuilder

__version__ = "1.0.0"
__author__ = "3Step AI System"

__all__ = [
    'AIBuilder',
    'DataProcessor', 
    'GameState',
    'GameConfig'
]
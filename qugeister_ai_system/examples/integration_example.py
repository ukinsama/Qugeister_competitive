#!/usr/bin/env python3
"""
統合3step → ai_maker_system ワークフロー
3stepで設計 → ai_maker_systemでビルド → 学習 → トーナメント
"""

import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_maker_system.ai_builder import AIBuilder

def step_to_config(step_config):
    """3stepの出力をai_maker_system設定に変換"""
    
    # 戦略マッピング
    strategy_map = {
        "aggressive": {
            "placement": {"strategy": "aggressive", "good_forward_ratio": 0.75},
            "reward": {"type": "aggressive", "forward_bonus": 2.0, "center_bonus": 1.5},
            "qmap": {"type": "strategic", "strategy": "aggressive"},
            "action": {"type": "epsilon_greedy", "epsilon": 0.1}
        },
        "defensive": {
            "placement": {"strategy": "defensive", "good_back_ratio": 0.75},
            "reward": {"type": "defensive", "retreat_bonus": 1.0, "edge_bonus": 1.5},
            "qmap": {"type": "strategic", "strategy": "defensive"},
            "action": {"type": "boltzmann", "temperature": 0.8}
        },
        "escape": {
            "placement": {"strategy": "standard", "randomize": True},
            "reward": {"type": "escape", "approach_bonus": 5.0, "escape_bonus": 50.0},
            "qmap": {"type": "strategic", "strategy": "escape"},
            "action": {"type": "ucb", "c": 1.5, "total_count": 0}
        }
    }
    
    # 基本設定から戦略を判定
    strategy = step_config.get("strategy", "balanced")
    
    if strategy in strategy_map:
        config = strategy_map[strategy].copy()
    else:
        # デフォルト設定
        config = {
            "placement": {"strategy": "standard"},
            "reward": {"type": "basic"},
            "qmap": {"type": "simple"},
            "action": {"type": "epsilon_greedy", "epsilon": 0.1}
        }
    
    # 推定器設定（量子or古典）
    if step_config.get("use_quantum", True):
        config["estimator"] = {
            "type": "cqcnn",
            "n_layers": step_config.get("quantum_layers", 2),
            "learning_rate": step_config.get("learning_rate", 0.001)
        }
    else:
        config["estimator"] = {"type": "simple"}
    
    return config

def create_ai_from_3step(ai_name, step_config):
    """3step設定からAIを作成"""
    print(f"🎯 3step設定からAI作成: {ai_name}")
    
    # 設定変換
    ai_config = step_to_config(step_config)
    ai_config['name'] = ai_name  # AI名を追加
    print(f"📋 変換された設定: {json.dumps(ai_config, indent=2)}")
    
    # AIビルド
    builder = AIBuilder("integrated_ais")
    ai_info = builder.create_ai(ai_config)
    
    # 保存パスを取得
    save_path = f"integrated_ais/{ai_name}"
    
    print(f"✅ AI保存完了: {save_path}")
    return save_path, ai_config

def create_learning_recipe(ai_name, ai_config, save_path):
    """学習用レシピファイル作成"""
    
    strategy = "balanced"
    if "aggressive" in ai_config.get("placement", {}).get("strategy", ""):
        strategy = "aggressive"
    elif "defensive" in ai_config.get("placement", {}).get("strategy", ""):
        strategy = "defensive"
    elif "escape" in ai_config.get("reward", {}).get("type", ""):
        strategy = "escape"
    
    # レシピファイル作成
    recipe_content = f'''#!/usr/bin/env python3
"""
Recipe for {ai_name} AI
Generated from 3step → ai_maker_system workflow
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '{save_path}'))

from {ai_name}_ai import {ai_name}AI
from ai_maker_system.learning.supervised import SupervisedLearner

def train_ai(episodes=1000):
    """統合学習システム"""
    print(f"🎓 Training {{ai_name}} ({strategy} strategy)")
    
    # AI読み込み
    ai = {ai_name}AI()
    
    # 学習システム
    learner = SupervisedLearner(
        model=ai.estimator,
        config={{
            "learning_rate": {ai_config.get("estimator", {}).get("learning_rate", 0.001)},
            "batch_size": 32,
            "epochs": episodes
        }}
    )
    
    # 学習実行
    learner.train()
    print(f"✅ {{ai_name}} Training completed")
    
    return ai

if __name__ == "__main__":
    train_ai()
'''
    
    recipe_path = f"{ai_name.lower()}_integrated_recipe.py"
    with open(recipe_path, 'w', encoding='utf-8') as f:
        f.write(recipe_content)
    
    print(f"📝 レシピファイル作成: {recipe_path}")
    return recipe_path

def integrated_workflow_demo():
    """統合ワークフロー実演"""
    print("🚀 統合3step → ai_maker_system ワークフロー開始")
    print("="*60)
    
    # サンプル3step設定
    step_configs = {
        "AggressiveAI": {
            "strategy": "aggressive",
            "use_quantum": True,
            "quantum_layers": 2,
            "learning_rate": 0.001
        },
        "DefensiveAI": {
            "strategy": "defensive", 
            "use_quantum": True,
            "quantum_layers": 2,
            "learning_rate": 0.001
        },
        "EscapeAI": {
            "strategy": "escape",
            "use_quantum": True,
            "quantum_layers": 2,
            "learning_rate": 0.001
        }
    }
    
    created_ais = []
    
    for ai_name, config in step_configs.items():
        print(f"\\n🔨 Step 1: AI作成 - {ai_name}")
        save_path, ai_config = create_ai_from_3step(ai_name, config)
        
        print(f"📝 Step 2: 学習レシピ作成 - {ai_name}")
        recipe_path = create_learning_recipe(ai_name, ai_config, save_path)
        
        created_ais.append({
            "name": ai_name,
            "path": save_path,
            "recipe": recipe_path,
            "config": ai_config
        })
    
    print(f"\\n✅ 統合AI作成完了: {len(created_ais)}個")
    print("\\n📚 作成されたAI:")
    for ai in created_ais:
        print(f"  - {ai['name']}: {ai['path']}")
        print(f"    レシピ: {ai['recipe']}")
    
    print("\\n🏆 次のステップ:")
    print("1. 各レシピファイルで学習実行")
    for ai in created_ais:
        print(f"   python {ai['recipe']}")
    print("2. トーナメント実行")
    print("   python tournament/tournament_manager.py")
    
    return created_ais

if __name__ == "__main__":
    # ディレクトリ作成
    os.makedirs("integrated_ais", exist_ok=True)
    
    # 統合ワークフロー実行
    integrated_workflow_demo()
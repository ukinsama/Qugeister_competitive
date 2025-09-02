#!/usr/bin/env python3
"""
段階2: AI学習システム
Recipe Trainer - AI Recipe Learning System
"""

import sys
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# パス設定
sys.path.append(str(Path(__file__).parent.parent))

def find_ai_recipes():
    """利用可能なAIレシピを発見"""
    recipe_files = [
        '../quick_aggressive_recipe.py',
        '../quick_defensive_recipe.py', 
        '../quick_escape_recipe.py',
        '../aggressiveai_integrated_recipe.py',
        '../defensiveai_integrated_recipe.py',
        '../escapeai_integrated_recipe.py'
    ]
    
    available_recipes = []
    for recipe in recipe_files:
        if Path(recipe).exists():
            available_recipes.append(recipe)
            
    return available_recipes

def load_recipe_config(recipe_path):
    """レシピファイルからAI設定を読み込み"""
    try:
        # レシピファイル実行でAI設定を取得
        import importlib.util
        spec = importlib.util.spec_from_file_location("recipe", recipe_path)
        recipe_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recipe_module)
        
        # AI設定情報を取得
        if hasattr(recipe_module, 'get_ai_config'):
            return recipe_module.get_ai_config()
        else:
            # デフォルト設定
            return {
                'name': Path(recipe_path).stem,
                'type': 'quantum_cqcnn',
                'learning_rate': 0.001,
                'epochs': 100
            }
    except Exception as e:
        print(f"⚠️  レシピ読み込みエラー: {recipe_path} - {e}")
        return None

def simple_training_loop(ai_config, save_dir):
    """簡単な学習ループ"""
    print(f"🧠 AI学習開始: {ai_config['name']}")
    
    # サンプルモデル（実際の実装では各レシピのモデルを使用）
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 4)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=ai_config.get('learning_rate', 0.001))
    criterion = nn.MSELoss()
    
    epochs = ai_config.get('epochs', 50)
    
    for epoch in range(epochs):
        # ダミー訓練データ（実際の実装では対戦データを使用）
        inputs = torch.randn(32, 32)
        targets = torch.randn(32, 4)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # モデル保存
    model_path = save_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)
    
    # AI情報保存
    ai_info = {
        'name': ai_config['name'],
        'type': ai_config.get('type', 'neural_network'),
        'trained_at': datetime.now().isoformat(),
        'epochs': epochs,
        'final_loss': loss.item()
    }
    
    with open(save_dir / 'ai_info.json', 'w', encoding='utf-8') as f:
        json.dump(ai_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 学習完了: {model_path}")
    return model_path

def train_single_recipe(recipe_path):
    """単一レシピの学習"""
    config = load_recipe_config(recipe_path)
    if not config:
        return None
        
    # 保存ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("trained_models") / f"{config['name']}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return simple_training_loop(config, save_dir)

def batch_training():
    """バッチ学習"""
    print("🏭 バッチ学習モード")
    print("=" * 50)
    
    recipes = find_ai_recipes()
    if not recipes:
        print("❌ AIレシピが見つかりません")
        return
    
    results = []
    for recipe in recipes:
        print(f"\n📚 学習中: {recipe}")
        model_path = train_single_recipe(recipe)
        if model_path:
            results.append(model_path)
    
    print(f"\n🎉 バッチ学習完了! {len(results)}個のモデルを作成")
    for result in results:
        print(f"  ✅ {result}")

def interactive_training():
    """インタラクティブ学習"""
    print("🎯 インタラクティブ学習モード")
    print("=" * 50)
    
    recipes = find_ai_recipes()
    if not recipes:
        print("❌ AIレシピが見つかりません")
        return
    
    print("利用可能なレシピ:")
    for i, recipe in enumerate(recipes, 1):
        print(f"  {i}. {Path(recipe).name}")
    
    try:
        choice = int(input("\n学習するレシピを選択してください (番号): ")) - 1
        if 0 <= choice < len(recipes):
            selected_recipe = recipes[choice]
            print(f"\n選択: {selected_recipe}")
            model_path = train_single_recipe(selected_recipe)
            if model_path:
                print(f"✅ 学習完了: {model_path}")
        else:
            print("❌ 無効な選択です")
    except ValueError:
        print("❌ 数字を入力してください")

def main():
    """メイン実行"""
    print("🧠 Qugeister AI Learning System")
    print("段階2: AI学習システム")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch':
            batch_training()
        elif sys.argv[1] == '--help':
            print_help()
        else:
            print("❌ 無効なオプション")
            print_help()
    else:
        interactive_training()

def print_help():
    """ヘルプ表示"""
    print("""
🧠 AI学習システム使用方法:

📚 バッチ学習 (すべてのレシピを自動学習):
   python learning/recipe_trainer.py --batch

🎯 インタラクティブ学習 (レシピ選択):
   python learning/recipe_trainer.py

💡 学習後の実行手順:
   1. python learning/recipe_trainer.py --batch
   2. python tournament_system/tournament_manager.py
""")

if __name__ == "__main__":
    main()
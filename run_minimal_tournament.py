#!/usr/bin/env python3
"""
ミニマル大会実行スクリプト
Qugeister Competition - Minimal Tournament Runner

簡単に大会を開催するためのスクリプト
"""

import sys
import os
from pathlib import Path

def check_environment():
    """環境チェック"""
    try:
        import torch
        import numpy as np
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        return True
    except ImportError as e:
        print(f"❌ 依存関係エラー: {e}")
        print("💡 解決方法: pip install torch numpy")
        return False

def find_tournament_system():
    """利用可能なトーナメントシステムを探す"""
    possible_paths = [
        'qugeister_ai_system/tournament_system/tournament_manager.py',
        'tournament_system/tournament_runner.py',
        'simple_integrated_tournament.py',
        'cqcnn_tournament_system.py'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ トーナメントシステム発見: {path}")
            return path
    
    print("❌ トーナメントシステムが見つかりません")
    return None

def find_trained_models():
    """学習済みモデルを探す"""
    model_dirs = [
        'trained_models/',
        'integrated_ais/',
        'quick_demo_ais/',
        'tournament_system/ai_configs/'
    ]
    
    total_models = 0
    for dir_path in model_dirs:
        path = Path(dir_path)
        if path.exists():
            models = list(path.glob('**/model.pth'))
            if models:
                print(f"✅ {dir_path}: {len(models)}個のモデル")
                total_models += len(models)
    
    return total_models

def run_tournament(tournament_path):
    """トーナメント実行"""
    print(f"\n🏆 トーナメント開始: {tournament_path}")
    try:
        # パスを追加
        sys.path.append(str(Path(tournament_path).parent))
        
        # 実行
        os.system(f"python {tournament_path}")
        return True
    except Exception as e:
        print(f"❌ トーナメント実行エラー: {e}")
        return False

def main():
    """メイン実行"""
    print("🏆 Qugeister Competition - Minimal Tournament")
    print("=" * 50)
    
    # 1. 環境チェック
    print("📋 Step 1: 環境チェック")
    if not check_environment():
        return False
    print()
    
    # 2. トーナメントシステム検索
    print("🔍 Step 2: トーナメントシステム検索")
    tournament_path = find_tournament_system()
    if not tournament_path:
        print("\n💡 代替方法:")
        print("   1. python qugeister_ai_system/tournament_system/tournament_manager.py")
        print("   2. python cqcnn_tournament_system.py")
        print("   3. python simple_integrated_tournament.py")
        return False
    print()
    
    # 3. モデル検索
    print("🤖 Step 3: 学習済みモデル検索")
    model_count = find_trained_models()
    if model_count == 0:
        print("❌ 学習済みモデルが見つかりません")
        print("💡 AIを作成してから再実行してください:")
        print("   python qugeister_ai_system/examples/integration_example.py")
        return False
    print(f"✅ 合計 {model_count}個のモデルを発見")
    print()
    
    # 4. トーナメント実行
    print("🚀 Step 4: トーナメント実行")
    success = run_tournament(tournament_path)
    
    if success:
        print("\n🎉 大会完了!")
        print("📊 結果は tournament_results/ フォルダにあります")
    else:
        print("\n⚠️  大会実行に問題がありました")
    
    return success

def quick_help():
    """クイックヘルプ"""
    print("""
🏆 Qugeister Competition Quick Start

📋 基本セットアップ:
   pip install torch numpy

🤖 AI作成:
   python qugeister_ai_system/examples/integration_example.py

🏃 大会実行:
   python run_minimal_tournament.py

🔍 環境確認:
   python environment_check.py

📖 詳細ガイド:
   SETUP_GUIDE.md を参照
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        quick_help()
    else:
        success = main()
        sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
ノートブックとGUIの統合システム
"""

import json
import os


def test_notebook_connection():
    """ノートブック接続テスト"""
    print("📓 Jupyter Notebook接続テスト")

    try:
        import jupyter

        print("✅ Jupyter利用可能")
    except ImportError:
        print("❌ Jupyter未インストール: pip install jupyter")
        return False

    # ノートブックファイル存在確認
    if os.path.exists("ai_design_notebook.ipynb"):
        print("✅ AIデザインノートブック存在")
    else:
        print("❌ ai_design_notebook.ipynb が見つかりません")
        return False

    return True


def test_gui_connection():
    """GUI接続テスト"""
    print("🖥️  GUI接続テスト")

    try:
        import pygame

        print("✅ Pygame利用可能")
    except ImportError:
        print("❌ Pygame未インストール: pip install pygame")
        return False

    # GUIファイル存在確認
    if os.path.exists("gui/fixed_game_viewer.py"):
        print("✅ 修正版GUI存在")
    else:
        print("❌ gui/fixed_game_viewer.py が見つかりません")
        return False

    return True


def create_sample_config():
    """サンプル設定ファイル作成"""
    print("⚙️  サンプル設定作成")

    sample_configs = {
        "beginner_ai": {
            "strategy": "balanced",
            "risk_level": 0.4,
            "exploration_rate": 0.5,
            "memory_depth": 3,
            "bluff_probability": 0.1,
        },
        "intermediate_ai": {
            "strategy": "aggressive",
            "risk_level": 0.7,
            "exploration_rate": 0.3,
            "memory_depth": 6,
            "bluff_probability": 0.3,
        },
        "expert_ai": {
            "strategy": "balanced",
            "risk_level": 0.8,
            "exploration_rate": 0.2,
            "memory_depth": 10,
            "bluff_probability": 0.4,
        },
    }

    os.makedirs("saved_configs", exist_ok=True)

    for name, config in sample_configs.items():
        filename = f"saved_configs/{name}.json"
        with open(filename, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  ✅ {filename}")

    return True


def run_integration_test():
    """統合テスト実行"""
    print("🧪 統合システムテスト開始")
    print("=" * 40)

    success = True

    # 各コンポーネントテスト
    if not test_notebook_connection():
        success = False

    if not test_gui_connection():
        success = False

    if not create_sample_config():
        success = False

    print("=" * 40)
    if success:
        print("🎉 統合テスト完全成功！")
        print("\n次のステップ:")
        print("1. python quick_run.py でシステム起動")
        print("2. Jupyter NotebookでAI設計")
        print("3. GUIで対戦テスト")
    else:
        print("❌ 統合テスト失敗")
        print("不足パッケージをインストールしてください")

    return success


if __name__ == "__main__":
    run_integration_test()

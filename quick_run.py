#!/usr/bin/env python3
"""
🚀 Qugeister競技システム - ワンクリック実行
"""

import os
import sys
import subprocess


def run_gui():
    """修正版GUI起動"""
    print("🖥️  修正版GUI起動中...")
    os.chdir("gui")
    try:
        subprocess.run([sys.executable, "fixed_game_viewer.py"])
    except KeyboardInterrupt:
        print("\n👋 GUI終了")
    finally:
        os.chdir("..")


def run_notebook():
    """Jupyter Notebook起動"""
    print("📓 Jupyter Notebook起動中...")
    try:
        subprocess.run(["jupyter", "notebook", "ai_design_notebook.ipynb"])
    except KeyboardInterrupt:
        print("\n👋 Notebook終了")
    except FileNotFoundError:
        print("❌ Jupyter がインストールされていません")
        print("インストール: pip install jupyter")


def run_integration_test():
    """統合テスト実行"""
    print("🧪 統合テスト実行中...")
    try:
        subprocess.run([sys.executable, "notebook_integration.py"])
    except Exception as e:
        print(f"❌ テスト失敗: {e}")


def run_all():
    """すべて順番に実行"""
    print("🚀 全システム起動!")
    print("1. 統合テスト")
    run_integration_test()

    print("\n2. GUI起動（修正版）")
    choice = input("GUIを起動しますか？ (y/N): ")
    if choice.lower() == "y":
        run_gui()

    print("\n3. Notebook起動")
    choice = input("Jupyter Notebookを起動しますか？ (y/N): ")
    if choice.lower() == "y":
        run_notebook()


def main():
    """メインメニュー"""
    print("=" * 50)
    print("🎮 Qugeister競技システム")
    print("=" * 50)
    print("1. 修正版GUI（トーナメント+表示）")
    print("2. Jupyter Notebook（AI設計）")
    print("3. 統合テスト")
    print("4. すべて実行")
    print("0. 終了")
    print("-" * 50)

    while True:
        choice = input("選択してください (0-4): ")

        if choice == "1":
            run_gui()
            break
        elif choice == "2":
            run_notebook()
            break
        elif choice == "3":
            run_integration_test()
            break
        elif choice == "4":
            run_all()
            break
        elif choice == "0":
            print("👋 終了します")
            break
        else:
            print("❌ 無効な選択です")


if __name__ == "__main__":
    main()


# Mini2ガイスター追加オプション
def run_mini2_geister():
    """Mini2ガイスター起動"""
    print("🎮 Mini2ガイスター起動中...")
    os.chdir("mini2_geister")
    try:
        subprocess.run([sys.executable, "run_mini2.py"])
    except KeyboardInterrupt:
        print("\n👋 Mini2ガイスター終了")
    finally:
        os.chdir("..")


# メインメニューに追加
# 既存のメイン関数に以下を追加:
# print("5. Mini2ガイスター（量子AI）")
# elif choice == "5":
#     run_mini2_geister()

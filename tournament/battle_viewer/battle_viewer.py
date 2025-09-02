#!/usr/bin/env python3
"""
段階4: 個別対戦観戦システム
Battle Viewer - Individual Battle Viewer
"""

import sys
import os
import json
from pathlib import Path

# パス設定
sys.path.append(str(Path(__file__).parent.parent.parent))

def find_battle_results():
    """利用可能な対戦結果を探す"""
    result_dirs = [
        'tournament_system/results/',
        'qugeister_competitive/results/',
        'gui/results/'
    ]
    
    all_results = []
    for result_dir in result_dirs:
        path = Path(result_dir)
        if path.exists():
            json_files = list(path.glob('*.json'))
            for json_file in json_files:
                if 'tournament' in json_file.name:
                    all_results.append(json_file)
    
    return all_results

def load_tournament_results(result_file):
    """トーナメント結果を読み込み"""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 結果ファイル読み込みエラー: {e}")
        return None

def display_battle_summary(results):
    """対戦サマリー表示"""
    if 'battles' in results:
        battles = results['battles']
        print(f"📊 総対戦数: {len(battles)}")
        print("=" * 50)
        
        for i, battle in enumerate(battles[:10]):  # 最初の10戦のみ表示
            ai1 = battle.get('ai1', 'AI1')
            ai2 = battle.get('ai2', 'AI2') 
            winner = battle.get('winner', 'Unknown')
            games = battle.get('games', [])
            
            print(f"⚔️  対戦 {i+1}: {ai1} vs {ai2}")
            print(f"   🏆 勝者: {winner}")
            print(f"   🎮 ゲーム数: {len(games)}")
            print()
            
        if len(battles) > 10:
            print(f"... 他 {len(battles) - 10} 対戦")

def display_rankings(results):
    """ランキング表示"""
    if 'rankings' in results:
        rankings = results['rankings']
        print("🏆 最終ランキング")
        print("=" * 50)
        
        for i, rank_info in enumerate(rankings, 1):
            ai_name = rank_info.get('ai', 'Unknown')
            wins = rank_info.get('wins', 0)
            total = rank_info.get('total_games', 0)
            win_rate = rank_info.get('win_rate', 0.0)
            
            print(f"{i:2d}位: {ai_name}")
            print(f"      勝率: {win_rate:.1%} ({wins}/{total})")
            print()

def interactive_battle_viewer():
    """インタラクティブ対戦観戦"""
    print("👀 個別対戦観戦システム")
    print("=" * 50)
    
    # 結果ファイル検索
    result_files = find_battle_results()
    if not result_files:
        print("❌ 対戦結果が見つかりません")
        print("💡 先にトーナメントを実行してください:")
        print("   python tournament/tournament_runner.py")
        return
    
    # 結果ファイル選択
    print("利用可能な対戦結果:")
    for i, result_file in enumerate(result_files, 1):
        print(f"  {i}. {result_file.name}")
    
    try:
        choice = int(input("\n観戦する結果を選択してください (番号): ")) - 1
        if 0 <= choice < len(result_files):
            selected_file = result_files[choice]
            print(f"\n選択: {selected_file}")
            
            # 結果読み込み・表示
            results = load_tournament_results(selected_file)
            if results:
                print("\n" + "=" * 50)
                display_rankings(results)
                print("=" * 50)
                display_battle_summary(results)
        else:
            print("❌ 無効な選択です")
    except ValueError:
        print("❌ 数字を入力してください")

def quick_view_latest():
    """最新結果の簡単表示"""
    result_files = find_battle_results()
    if not result_files:
        print("❌ 対戦結果が見つかりません")
        return
    
    # 最新ファイルを取得
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    print(f"📊 最新の対戦結果: {latest_file.name}")
    print("=" * 50)
    
    results = load_tournament_results(latest_file)
    if results:
        display_rankings(results)

def main():
    """メイン実行"""
    print("👀 Battle Viewer - 段階4: 個別対戦観戦")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            quick_view_latest()
        elif sys.argv[1] == '--help':
            print_help()
        else:
            print("❌ 無効なオプション")
            print_help()
    else:
        interactive_battle_viewer()

def print_help():
    """ヘルプ表示"""
    print("""
👀 個別対戦観戦システム使用方法:

🎯 インタラクティブ観戦:
   python tournament/battle_viewer/battle_viewer.py

⚡ 最新結果の簡単表示:
   python tournament/battle_viewer/battle_viewer.py --quick

💡 観戦前の準備:
   1. python learning/recipe_trainer.py --batch
   2. python tournament/tournament_runner.py
   3. python tournament/battle_viewer/battle_viewer.py
""")

if __name__ == "__main__":
    main()
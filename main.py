#!/usr/bin/env python3
"""
Qugeister AI Competition - メイン実行スクリプト
"""

import os
import sys

# パッケージのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🎯 Qugeister AI Competition")
    print("=" * 40)
    print("1. トーナメント実行 + GUI表示")
    print("2. 既存結果をGUIで表示")
    print("3. コマンドラインでトーナメント実行")
    
    choice = input("\n選択してください (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 トーナメント + GUI実行...")
        from gui.game_viewer import run_tournament_and_gui
        run_tournament_and_gui()
        
    elif choice == "2":
        print("\n🎮 GUI表示...")
        results_file = "results/tournament_results.json"
        
        if not os.path.exists(results_file):
            print(f"❌ 結果ファイルが見つかりません: {results_file}")
            print("   選択肢1でトーナメントを実行してください")
            return
        
        from gui.game_viewer import GameGUI
        try:
            gui = GameGUI()
            gui.load_game_logs(results_file)
            gui.run()
        except ImportError:
            print("❌ pygame が必要です: pip install pygame")
        
    elif choice == "3":
        print("\n🏆 コマンドライントーナメント...")
        from qugeister_competitive.ai_base import RandomAI, SimpleAI, AggressiveAI
        from qugeister_competitive.tournament import TournamentManager
        
        # AI作成
        ais = [
            RandomAI("A"), SimpleAI("A"), AggressiveAI("A"),
            RandomAI("B"), SimpleAI("B")
        ]
        
        # トーナメント実行
        tournament = TournamentManager()
        for ai in ais:
            tournament.add_participant(ai)
        
        results = tournament.run_round_robin(games_per_pair=5)
        
        # 順位表示
        print("\n📊 最終順位:")
        print("-" * 50)
        leaderboard = tournament.get_leaderboard()
        for i, entry in enumerate(leaderboard):
            print(f"{i+1:2d}. {entry['name']:12s} | Rating: {entry['rating']:4d} | "
                  f"Win Rate: {entry['win_rate']:5.1f}% ({entry['wins']}/{entry['games']})")
        
        # 結果保存
        results_file = "results/tournament_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        tournament.save_results(results_file)
        print(f"\n💾 結果保存: {results_file}")
    
    else:
        print("❌ 無効な選択です")

if __name__ == "__main__":
    main()

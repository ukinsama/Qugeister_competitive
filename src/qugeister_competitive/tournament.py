"""
トーナメント管理とレーティングシステム
"""

import time
import json
from typing import List, Dict, Any
from .game_engine import GeisterGame
from .ai_base import BaseAI


class TournamentManager:
    """トーナメント管理システム"""

    def __init__(self):
        self.participants = []
        self.results = []
        self.game_logs = []

    def add_participant(self, ai: BaseAI):
        """参加者追加"""
        self.participants.append(ai)
        print(f"✅ 参加者追加: {ai.name} (Player {ai.player_id})")

    def run_round_robin(self, games_per_pair: int = 5) -> Dict[str, Any]:
        """総当たりトーナメント実行"""
        print(f"🏟️ 総当たりトーナメント開始 ({len(self.participants)}体参加)")
        print(f"   各ペア {games_per_pair}ゲーム")

        total_games = 0
        start_time = time.time()

        # 全ペアで対戦
        for i, ai1 in enumerate(self.participants):
            for j, ai2 in enumerate(self.participants):
                if i < j:  # 重複を避ける
                    print(f"\n--- {ai1.name} vs {ai2.name} ---")

                    wins_ai1 = 0
                    wins_ai2 = 0

                    for game_num in range(games_per_pair):
                        # 先手後手を交互に
                        if game_num % 2 == 0:
                            winner, game_log = self._play_single_game(ai1, ai2)
                            if winner == ai1.player_id:
                                wins_ai1 += 1
                            elif winner == ai2.player_id:
                                wins_ai2 += 1
                        else:
                            winner, game_log = self._play_single_game(ai2, ai1)
                            if winner == ai1.player_id:
                                wins_ai1 += 1
                            elif winner == ai2.player_id:
                                wins_ai2 += 1

                        total_games += 1
                        self.game_logs.append(game_log)

                    print(f"   結果: {ai1.name} {wins_ai1}-{wins_ai2} {ai2.name}")

                    # レーティング更新
                    self._update_ratings(ai1, ai2, wins_ai1, wins_ai2)

        duration = time.time() - start_time

        # 結果集計
        tournament_result = {
            "total_games": total_games,
            "duration": duration,
            "participants": len(self.participants),
            "final_ratings": {ai.name: ai.rating for ai in self.participants},
            "win_rates": {ai.name: ai.win_rate for ai in self.participants},
        }

        print("\n🎉 トーナメント完了!")
        print(f"   総ゲーム数: {total_games}")
        print(f"   実行時間: {duration:.1f}秒")

        return tournament_result

    def _play_single_game(self, player_a: BaseAI, player_b: BaseAI) -> tuple:
        """単一ゲーム実行"""
        game = GeisterGame()
        game_log = {"player_a": player_a.name, "player_b": player_b.name, "moves": [], "winner": None, "turns": 0}

        max_turns = 100

        for turn in range(max_turns):
            current_ai = player_a if game.current_player == "A" else player_b

            # 合法手取得
            legal_moves = game.get_legal_moves(game.current_player)
            if not legal_moves:
                break

            # AI思考
            game_state = game.get_game_state(game.current_player)
            move = current_ai.get_move(game_state, legal_moves)

            if move is None:
                break

            # 手実行
            success = game.make_move(move[0], move[1])
            if not success:
                break

            # ログ記録
            game_log["moves"].append(
                {"turn": turn, "player": game.current_player, "move": move, "board_state": game.board.tolist()}
            )

            # 勝利判定
            if game.game_over:
                break

        game_log["winner"] = game.winner
        game_log["turns"] = len(game_log["moves"])

        # 結果記録
        if game.winner == "A":
            player_a.record_result(True)
            player_b.record_result(False)
        elif game.winner == "B":
            player_a.record_result(False)
            player_b.record_result(True)
        else:
            player_a.record_result(False)
            player_b.record_result(False)

        return game.winner, game_log

    def _update_ratings(self, ai1: BaseAI, ai2: BaseAI, wins1: int, wins2: int):
        """Eloレーティング更新"""
        total_games = wins1 + wins2
        if total_games == 0:
            return

        # 期待スコア計算
        expected1 = 1 / (1 + 10 ** ((ai2.rating - ai1.rating) / 400))
        expected2 = 1 / (1 + 10 ** ((ai1.rating - ai2.rating) / 400))

        # 実際のスコア
        actual1 = wins1 / total_games
        actual2 = wins2 / total_games

        # レーティング更新 (K=32)
        K = 32
        ai1.rating += K * (actual1 - expected1)
        ai2.rating += K * (actual2 - expected2)

        ai1.rating = max(100, ai1.rating)  # 最低レーティング
        ai2.rating = max(100, ai2.rating)

    def get_leaderboard(self) -> List[Dict]:
        """リーダーボード取得"""
        leaderboard = []
        for ai in sorted(self.participants, key=lambda x: x.rating, reverse=True):
            leaderboard.append(
                {
                    "name": ai.name,
                    "rating": round(ai.rating),
                    "games": ai.games_played,
                    "wins": ai.wins,
                    "win_rate": round(ai.win_rate * 100, 1),
                }
            )
        return leaderboard

    def save_results(self, filepath: str):
        """結果保存"""
        results = {
            "tournament_info": {"participants": len(self.participants), "total_games": len(self.game_logs)},
            "leaderboard": self.get_leaderboard(),
            "game_logs": self.game_logs,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 結果保存: {filepath}")

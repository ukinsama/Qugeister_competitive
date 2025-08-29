#!/usr/bin/env python3
"""
CQCNNデータ収集システム
対戦データを自動収集して学習用データセットを作成
"""

import sys
import os
import numpy as np
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src", "qugeister_competitive")
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

print("📂 モジュール読み込み中...")

# ゲームエンジンを読み込み
try:
    with open(os.path.join(src_path, "game_engine.py"), "r") as f:
        exec(f.read())
    with open(os.path.join(src_path, "ai_base.py"), "r") as f:
        ai_base_code = f.read().replace("from .game_engine", "# from .game_engine")
        exec(ai_base_code)
    print("✅ ゲームエンジン読み込み完了")
except Exception as e:
    print(f"⚠️ エラー: {e}")

print("✅ データ収集システム準備完了\n")


# ================================================================================
# データ構造定義
# ================================================================================


@dataclass
class BoardSnapshot:
    """盤面スナップショット"""

    turn: int
    board: np.ndarray
    player_a_pieces: Dict[Tuple[int, int], str]
    player_b_pieces: Dict[Tuple[int, int], str]
    current_player: str
    move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]

    def to_dict(self):
        """辞書形式に変換"""
        return {
            "turn": self.turn,
            "board": self.board.tolist(),
            "player_a_pieces": {str(k): v for k, v in self.player_a_pieces.items()},
            "player_b_pieces": {str(k): v for k, v in self.player_b_pieces.items()},
            "current_player": self.current_player,
            "move": self.move,
        }


@dataclass
class GameRecord:
    """ゲーム記録"""

    game_id: str
    agent1_name: str
    agent2_name: str
    winner: str
    total_moves: int
    snapshots: List[BoardSnapshot]
    start_time: str
    end_time: str

    def to_dict(self):
        """辞書形式に変換"""
        return {
            "game_id": self.game_id,
            "agent1_name": self.agent1_name,
            "agent2_name": self.agent2_name,
            "winner": self.winner,
            "total_moves": self.total_moves,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class TrainingDataPoint:
    """学習用データポイント"""

    board_state: np.ndarray  # 盤面状態
    enemy_position: Tuple[int, int]  # 敵駒位置
    piece_type: str  # 正解ラベル（good/bad）
    game_phase: str  # ゲームフェーズ（early/mid/late）
    player_view: str  # 視点（A/B）
    confidence_hint: float  # ヒント情報（オプション）


# ================================================================================
# データ収集器
# ================================================================================


class CQCNNDataCollector:
    """CQCNNデータ収集器"""

    def __init__(self, save_dir: str = "cqcnn_training_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # データ保存場所
        self.game_records_dir = os.path.join(save_dir, "game_records")
        self.training_data_dir = os.path.join(save_dir, "training_data")
        self.statistics_dir = os.path.join(save_dir, "statistics")

        for dir_path in [self.game_records_dir, self.training_data_dir, self.statistics_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 収集統計
        self.stats = {
            "total_games": 0,
            "total_snapshots": 0,
            "total_training_points": 0,
            "collection_start": datetime.now().isoformat(),
        }

        # 現在のゲーム記録
        self.current_game_record = None
        self.current_snapshots = []

    def start_game_recording(self, agent1_name: str, agent2_name: str) -> str:
        """ゲーム記録を開始"""
        game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

        self.current_game_record = {
            "game_id": game_id,
            "agent1_name": agent1_name,
            "agent2_name": agent2_name,
            "start_time": datetime.now().isoformat(),
            "snapshots": [],
        }

        self.current_snapshots = []

        print(f"📹 記録開始: {game_id}")
        return game_id

    def record_snapshot(self, game: Any, move: Optional[Tuple] = None):
        """盤面スナップショットを記録"""
        if self.current_game_record is None:
            return

        snapshot = BoardSnapshot(
            turn=game.turn,
            board=game.board.copy(),
            player_a_pieces=game.player_a_pieces.copy(),
            player_b_pieces=game.player_b_pieces.copy(),
            current_player=game.current_player,
            move=move,
        )

        self.current_snapshots.append(snapshot)
        self.stats["total_snapshots"] += 1

    def end_game_recording(self, winner: str) -> GameRecord:
        """ゲーム記録を終了"""
        if self.current_game_record is None:
            return None

        # ゲーム記録を完成
        game_record = GameRecord(
            game_id=self.current_game_record["game_id"],
            agent1_name=self.current_game_record["agent1_name"],
            agent2_name=self.current_game_record["agent2_name"],
            winner=winner,
            total_moves=len(self.current_snapshots),
            snapshots=self.current_snapshots,
            start_time=self.current_game_record["start_time"],
            end_time=datetime.now().isoformat(),
        )

        # ファイルに保存
        self._save_game_record(game_record)

        # 学習データを抽出
        training_data = self._extract_training_data(game_record)
        self._save_training_data(training_data, game_record.game_id)

        self.stats["total_games"] += 1
        self.stats["total_training_points"] += len(training_data)

        print(f"📹 記録終了: {game_record.game_id} (勝者: {winner}, {len(training_data)}データポイント)")

        # リセット
        self.current_game_record = None
        self.current_snapshots = []

        return game_record

    def _extract_training_data(self, game_record: GameRecord) -> List[TrainingDataPoint]:
        """ゲーム記録から学習データを抽出"""
        training_data = []

        for snapshot in game_record.snapshots:
            # ゲームフェーズを判定
            if snapshot.turn < 10:
                phase = "early"
            elif snapshot.turn < 30:
                phase = "mid"
            else:
                phase = "late"

            # プレイヤーAの視点
            for pos, piece_type in snapshot.player_b_pieces.items():
                # 盤面上に存在する駒のみ
                if snapshot.board[pos[1], pos[0]] == -1:
                    data_point = TrainingDataPoint(
                        board_state=snapshot.board.copy(),
                        enemy_position=pos,
                        piece_type=piece_type,
                        game_phase=phase,
                        player_view="A",
                        confidence_hint=self._calculate_confidence_hint(snapshot, pos, "A"),
                    )
                    training_data.append(data_point)

            # プレイヤーBの視点
            for pos, piece_type in snapshot.player_a_pieces.items():
                # 盤面上に存在する駒のみ
                if snapshot.board[pos[1], pos[0]] == 1:
                    data_point = TrainingDataPoint(
                        board_state=snapshot.board.copy(),
                        enemy_position=pos,
                        piece_type=piece_type,
                        game_phase=phase,
                        player_view="B",
                        confidence_hint=self._calculate_confidence_hint(snapshot, pos, "B"),
                    )
                    training_data.append(data_point)

        return training_data

    def _calculate_confidence_hint(self, snapshot: BoardSnapshot, pos: Tuple[int, int], player_view: str) -> float:
        """確信度のヒントを計算（位置や状況から）"""
        x, y = pos

        # 前線にいる駒は悪玉の可能性が高い
        if player_view == "A":
            forward_score = y / 5.0  # Bの駒が下にいるほど高スコア
        else:
            forward_score = (5 - y) / 5.0  # Aの駒が上にいるほど高スコア

        # 中央にいる駒は重要
        center_score = 1.0 - (abs(x - 2.5) / 2.5)

        # 総合的な確信度ヒント
        confidence = (forward_score + center_score) / 2.0
        return min(max(confidence, 0.3), 0.9)  # 0.3～0.9の範囲

    def _save_game_record(self, game_record: GameRecord):
        """ゲーム記録を保存"""
        filename = os.path.join(self.game_records_dir, f"{game_record.game_id}.json")
        with open(filename, "w") as f:
            json.dump(game_record.to_dict(), f, indent=2)

    def _save_training_data(self, training_data: List[TrainingDataPoint], game_id: str):
        """学習データを保存"""
        filename = os.path.join(self.training_data_dir, f"{game_id}_training.pkl")
        with open(filename, "wb") as f:
            pickle.dump(training_data, f)

    def save_statistics(self):
        """統計情報を保存"""
        self.stats["collection_end"] = datetime.now().isoformat()
        filename = os.path.join(self.statistics_dir, "collection_stats.json")
        with open(filename, "w") as f:
            json.dump(self.stats, f, indent=2)

        print("\n📊 収集統計:")
        print(f"  総ゲーム数: {self.stats['total_games']}")
        print(f"  総スナップショット: {self.stats['total_snapshots']}")
        print(f"  総学習データポイント: {self.stats['total_training_points']}")
        print(f"  保存先: {self.save_dir}")

    def load_all_training_data(self) -> List[TrainingDataPoint]:
        """全ての学習データを読み込み"""
        all_data = []

        for filename in os.listdir(self.training_data_dir):
            if filename.endswith("_training.pkl"):
                filepath = os.path.join(self.training_data_dir, filename)
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    all_data.extend(data)

        print(f"📚 {len(all_data)}個の学習データを読み込みました")
        return all_data


# ================================================================================
# データ収集機能付きゲーム実行
# ================================================================================


class DataCollectionGame:
    """データ収集機能付きゲーム"""

    def __init__(self, collector: CQCNNDataCollector):
        self.collector = collector
        self.game = GeisterGame()

    def run_game_with_collection(self, agent1: Any, agent2: Any, verbose: bool = False) -> Dict:
        """データ収集しながらゲーム実行"""

        # 記録開始
        game_id = self.collector.start_game_recording(agent1.name, agent2.name)

        # ゲームリセット
        self.game.reset_game()

        if verbose:
            print(f"\n🎮 {agent1.name} vs {agent2.name}")

        move_count = 0
        max_moves = 100

        # 初期状態を記録
        self.collector.record_snapshot(self.game)

        while not self.game.game_over and move_count < max_moves:
            # 現在のプレイヤー
            current_agent = agent1 if self.game.current_player == "A" else agent2

            # 合法手取得
            legal_moves = self.game.get_legal_moves(self.game.current_player)
            if not legal_moves:
                break

            # 手を選択
            game_state = self.game.get_game_state(self.game.current_player)
            move = current_agent.get_move(game_state, legal_moves)

            if not move:
                break

            # 手を実行
            success = self.game.make_move(move[0], move[1])
            if not success:
                break

            # スナップショットを記録
            self.collector.record_snapshot(self.game, move)

            move_count += 1

            if verbose and move_count <= 3:
                print(f"  Move {move_count}: {move[0]} → {move[1]}")

        # 記録終了
        game_record = self.collector.end_game_recording(self.game.winner)

        if verbose:
            if self.game.winner in ["A", "B"]:
                winner_name = agent1.name if self.game.winner == "A" else agent2.name
                print(f"  🏆 勝者: {winner_name}")
            else:
                print("  🤝 引き分け")

        return {"game_id": game_id, "winner": self.game.winner, "total_moves": move_count, "game_record": game_record}


# ================================================================================
# データ収集キャンペーン
# ================================================================================


def run_data_collection_campaign(num_games: int = 10):
    """データ収集キャンペーンを実行"""
    print("🚀 データ収集キャンペーン開始")
    print("=" * 70)

    # データ収集器を初期化
    collector = CQCNNDataCollector(save_dir=f"cqcnn_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # ゲーム実行器
    game_runner = DataCollectionGame(collector)

    # サンプルエージェント（実際のエージェントに置き換え可能）
    agents = [RandomAI("A"), SimpleAI("A"), AggressiveAI("A")]

    print(f"\n📊 {num_games}ゲームのデータを収集します")
    print(f"参加エージェント: {[a.name for a in agents]}")
    print("-" * 50)

    # ゲーム実行
    for game_num in range(num_games):
        # ランダムに2つのエージェントを選択
        agent1, agent2 = random.sample(agents, 2)

        # プレイヤーIDを設定
        agent1.player_id = "A"
        agent2.player_id = "B"

        print(f"\nGame {game_num + 1}/{num_games}: {agent1.name} vs {agent2.name}")

        # データ収集しながらゲーム実行
        game_runner.run_game_with_collection(agent1, agent2, verbose=True)

    # 統計保存
    collector.save_statistics()

    print("\n" + "=" * 70)
    print("✅ データ収集完了！")

    return collector


# ================================================================================
# データ分析とプレビュー
# ================================================================================


def analyze_collected_data(collector: CQCNNDataCollector):
    """収集したデータを分析"""
    print("\n📊 収集データ分析")
    print("=" * 70)

    # 全学習データを読み込み
    all_data = collector.load_all_training_data()

    if not all_data:
        print("⚠️ データがありません")
        return

    # 統計分析
    good_count = sum(1 for d in all_data if d.piece_type == "good")
    bad_count = sum(1 for d in all_data if d.piece_type == "bad")

    phase_counts = {}
    for d in all_data:
        phase_counts[d.game_phase] = phase_counts.get(d.game_phase, 0) + 1

    player_counts = {}
    for d in all_data:
        player_counts[d.player_view] = player_counts.get(d.player_view, 0) + 1

    print("\n【データポイント統計】")
    print(f"  総数: {len(all_data)}")
    print(f"  善玉: {good_count} ({good_count / len(all_data) * 100:.1f}%)")
    print(f"  悪玉: {bad_count} ({bad_count / len(all_data) * 100:.1f}%)")

    print("\n【ゲームフェーズ分布】")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} ({count / len(all_data) * 100:.1f}%)")

    print("\n【視点分布】")
    for player, count in player_counts.items():
        print(f"  Player {player}: {count} ({count / len(all_data) * 100:.1f}%)")

    # サンプルデータ表示
    print("\n【サンプルデータ（最初の3個）】")
    for i, data_point in enumerate(all_data[:3]):
        print(f"\n  データ {i + 1}:")
        print(f"    位置: {data_point.enemy_position}")
        print(f"    タイプ: {data_point.piece_type}")
        print(f"    フェーズ: {data_point.game_phase}")
        print(f"    視点: Player {data_point.player_view}")
        print(f"    確信度ヒント: {data_point.confidence_hint:.2f}")


# ================================================================================
# メイン実行
# ================================================================================


def main():
    """メイン実行"""
    print("=" * 70)
    print("📚 CQCNNデータ収集システム")
    print("=" * 70)

    print("\n【メニュー】")
    print("1. クイック収集（10ゲーム）")
    print("2. 標準収集（50ゲーム）")
    print("3. 大規模収集（100ゲーム）")
    print("4. カスタム収集")

    choice = input("\n選択 (1-4): ").strip()

    if choice == "1":
        num_games = 10
    elif choice == "2":
        num_games = 50
    elif choice == "3":
        num_games = 100
    elif choice == "4":
        num_games = int(input("ゲーム数を入力: "))
    else:
        print("無効な選択")
        return

    # データ収集実行
    start_time = time.time()
    collector = run_data_collection_campaign(num_games)
    elapsed = time.time() - start_time

    # データ分析
    analyze_collected_data(collector)

    print(f"\n⏱️ 実行時間: {elapsed:.1f}秒")
    print(f"📁 データ保存先: {collector.save_dir}")

    print("\n💡 次のステップ:")
    print("  1. 収集したデータでCQCNNを学習")
    print("  2. 学習済みモデルを競技システムに統合")
    print("  3. 学習済み vs 未学習で性能比較")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 中断されました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback

        traceback.print_exc()

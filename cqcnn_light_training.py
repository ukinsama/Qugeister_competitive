#!/usr/bin/env python3
"""
CQCNN軽量学習版 - 学習による勝率向上テスト
最小限の学習で効果を確認
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import random
import time
import json

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'qugeister_competitive')
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

print("📂 モジュール読み込み中...")

# 必要なファイルを直接読み込み
with open(os.path.join(src_path, 'game_engine.py'), 'r') as f:
    exec(f.read())

with open(os.path.join(src_path, 'ai_base.py'), 'r') as f:
    code = f.read().replace('from .game_engine', '# from .game_engine')
    exec(code)

from separated_cqcnn_qmap import (
    CQCNNPieceEstimator,
    QValueMapGenerator,
    IntegratedCQCNNSystem
)

print("✅ モジュール読み込み完了\n")


# ================================================================================
# 軽量学習システム
# ================================================================================

class LightweightTrainer:
    """軽量学習システム"""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        # 軽量CQCNNシステム
        self.system = IntegratedCQCNNSystem(n_qubits=n_qubits, n_layers=n_layers)
        self.optimizer = optim.Adam(self.system.estimator.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()  # Binary Cross Entropy
        
        self.training_stats = {
            'before_accuracy': 0,
            'after_accuracy': 0,
            'loss_history': []
        }
    
    def generate_simple_training_data(self, num_games: int = 10):
        """簡易学習データ生成"""
        print(f"📊 学習データ生成中 ({num_games}ゲーム)...")
        data = []
        
        for game_idx in range(num_games):
            game = GeisterGame()
            
            # ランダムに20手進める
            for _ in range(20):
                legal_moves = game.get_legal_moves(game.current_player)
                if not legal_moves or game.game_over:
                    break
                move = random.choice(legal_moves)
                game.make_move(move[0], move[1])
            
            # 現在の盤面から学習データを作成
            if not game.game_over:
                # プレイヤーAの視点
                board_tensor = self._create_board_tensor(game.board, "A")
                
                # 敵駒（プレイヤーB）の情報
                for pos, piece_type in game.player_b_pieces.items():
                    data.append({
                        'board_tensor': board_tensor,
                        'position': pos,
                        'is_good': piece_type == "good",
                        'player': "A"
                    })
            
            print(f"  ゲーム {game_idx + 1}/{num_games} 完了")
        
        print(f"✅ {len(data)}個のデータポイント生成\n")
        return data
    
    def _create_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        """ボード状態をテンソルに変換"""
        tensor = torch.zeros(1, 3, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val
        
        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))
        
        return tensor
    
    def train_quick(self, training_data: List[Dict], epochs: int = 10):
        """軽量学習実行"""
        print(f"🎓 軽量学習開始 (エポック数: {epochs})")
        print("-" * 40)
        
        if not training_data:
            print("⚠️ 学習データが空です")
            return
        
        # 学習前の精度測定
        before_correct = 0
        for data_point in training_data[:10]:  # 最初の10個でテスト
            pred = self._predict_piece_type(data_point['board_tensor'], data_point['position'])
            if (pred > 0.5) == data_point['is_good']:
                before_correct += 1
        self.training_stats['before_accuracy'] = before_correct / min(10, len(training_data))
        
        # 学習ループ
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # データをシャッフル
            random.shuffle(training_data)
            
            for data_point in training_data:
                # 予測
                self.optimizer.zero_grad()
                
                # CQCNNで推定
                estimations = self.system.estimator(
                    data_point['board_tensor'],
                    [data_point['position']]
                )
                
                if estimations:
                    est = estimations[0]
                    pred_good = torch.tensor([est.good_probability], requires_grad=True)
                    
                    # 真のラベル
                    target = torch.tensor([1.0 if data_point['is_good'] else 0.0])
                    
                    # 損失計算
                    loss = self.criterion(pred_good, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # 精度計算
                    if (pred_good.item() > 0.5) == data_point['is_good']:
                        correct += 1
                    total += 1
            
            # エポック統計
            if total > 0:
                accuracy = correct / total
                avg_loss = epoch_loss / total
                self.training_stats['loss_history'].append(avg_loss)
                
                if (epoch + 1) % 2 == 0:
                    print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        
        # 学習後の精度測定
        after_correct = 0
        for data_point in training_data[:10]:
            pred = self._predict_piece_type(data_point['board_tensor'], data_point['position'])
            if (pred > 0.5) == data_point['is_good']:
                after_correct += 1
        self.training_stats['after_accuracy'] = after_correct / min(10, len(training_data))
        
        print("\n✅ 学習完了！")
        print(f"  学習前精度: {self.training_stats['before_accuracy']:.1%}")
        print(f"  学習後精度: {self.training_stats['after_accuracy']:.1%}")
        improvement = self.training_stats['after_accuracy'] - self.training_stats['before_accuracy']
        print(f"  改善幅: {improvement:+.1%}\n")
    
    def _predict_piece_type(self, board_tensor: torch.Tensor, position: Tuple) -> float:
        """駒タイプを予測"""
        with torch.no_grad():
            estimations = self.system.estimator(board_tensor, [position])
            if estimations:
                return estimations[0].good_probability
        return 0.5


# ================================================================================
# 学習済みCQCNN AI
# ================================================================================

class TrainedCQCNNAI(BaseAI):
    """学習済みCQCNN AI"""
    
    def __init__(self, player_id: str, system: IntegratedCQCNNSystem, name: str = "TrainedCQCNN"):
        super().__init__(name, player_id)
        self.system = system
        self.exploration_rate = 0.1  # 学習後は探索率を下げる
    
    def get_move(self, game_state: GameState, legal_moves: List) -> Optional[Tuple]:
        """学習済みモデルで手を選択"""
        if not legal_moves:
            return None
        
        # 10%の確率でランダム
        if random.random() < self.exploration_rate:
            return random.choice(legal_moves)
        
        # Q値マップを使った評価
        try:
            # ゲーム状態を辞書形式に変換
            game_dict = {
                'board': game_state.board.tolist(),
                'current_player': self.player_id,
                'turn': game_state.turn
            }
            
            # 自分の駒情報
            my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
            
            # 駒推定とQ値マップ生成
            estimation_data, q_map = self.system.process_game_state(game_dict, my_pieces)
            
            # 最適手選択
            best_move = None
            best_score = -float('inf')
            
            for move in legal_moves:
                from_pos, to_pos = move
                
                # 基本スコア
                score = self._evaluate_move(move, game_state)
                
                # Q値も考慮（可能なら）
                try:
                    dx = to_pos[0] - from_pos[0]
                    dy = to_pos[1] - from_pos[1]
                    
                    if dy == -1 and dx == 0:    dir_idx = 0
                    elif dx == 1 and dy == 0:   dir_idx = 1
                    elif dy == 1 and dx == 0:   dir_idx = 2
                    elif dx == -1 and dy == 0:  dir_idx = 3
                    else: dir_idx = -1
                    
                    if dir_idx >= 0:
                        q_value = q_map[from_pos[1], from_pos[0], dir_idx]
                        score += q_value * 0.1  # Q値の影響を調整
                except:
                    pass
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_move if best_move else random.choice(legal_moves)
            
        except Exception as e:
            # エラー時は簡易評価
            return self._get_simple_move(game_state, legal_moves)
    
    def _evaluate_move(self, move: Tuple, game_state: GameState) -> float:
        """手の評価"""
        from_pos, to_pos = move
        score = 0.0
        
        # 前進ボーナス
        if self.player_id == "A":
            score += (to_pos[1] - from_pos[1]) * 3.0
        else:
            score += (from_pos[1] - to_pos[1]) * 3.0
        
        # 中央制御
        center_dist = abs(to_pos[0] - 2.5)
        score += (2.5 - center_dist) * 1.0
        
        # 駒取り
        opponent_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces
        if to_pos in opponent_pieces:
            score += 8.0
        
        return score
    
    def _get_simple_move(self, game_state: GameState, legal_moves: List) -> Tuple:
        """簡易手選択"""
        scores = []
        for move in legal_moves:
            score = self._evaluate_move(move, game_state)
            scores.append((move, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else random.choice(legal_moves)


# ================================================================================
# 比較実験システム
# ================================================================================

def run_comparison_tournament(ai_list: List[BaseAI], games_per_pair: int = 5):
    """AI同士の総当たり戦"""
    print("\n🏆 比較トーナメント")
    print("=" * 60)
    
    results = {}
    
    for ai in ai_list:
        results[ai.name] = {'wins': 0, 'games': 0}
    
    # 総当たり戦
    for i, ai1 in enumerate(ai_list):
        for j, ai2 in enumerate(ai_list):
            if i >= j:  # 自分自身との対戦と重複を避ける
                continue
            
            print(f"\n{ai1.name} vs {ai2.name} ({games_per_pair}ゲーム)")
            
            for game_num in range(games_per_pair):
                game = GeisterGame()
                
                # 先手後手を交代
                if game_num % 2 == 0:
                    player_a, player_b = ai1, ai2
                    ai1.player_id, ai2.player_id = "A", "B"
                else:
                    player_a, player_b = ai2, ai1
                    ai1.player_id, ai2.player_id = "B", "A"
                
                # ゲーム実行
                max_turns = 100
                for turn in range(max_turns):
                    current_ai = player_a if game.current_player == "A" else player_b
                    legal_moves = game.get_legal_moves(game.current_player)
                    
                    if not legal_moves or game.game_over:
                        break
                    
                    game_state = game.get_game_state(game.current_player)
                    move = current_ai.get_move(game_state, legal_moves)
                    
                    if move:
                        game.make_move(move[0], move[1])
                
                # 結果記録
                if game.winner == "A":
                    winner = player_a
                elif game.winner == "B":
                    winner = player_b
                else:
                    winner = None
                
                if winner:
                    results[winner.name]['wins'] += 1
                
                results[ai1.name]['games'] += 1
                results[ai2.name]['games'] += 1
                
                # 結果表示
                result_str = winner.name if winner else "Draw"
                print(f"  Game {game_num + 1}: {result_str}")
    
    # 最終結果表示
    print("\n" + "=" * 60)
    print("📊 最終結果")
    print("-" * 40)
    
    # 勝率でソート
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['wins'] / max(x[1]['games'], 1), 
                          reverse=True)
    
    for name, stats in sorted_results:
        win_rate = stats['wins'] / max(stats['games'], 1) * 100
        print(f"{name:15s}: {stats['wins']:2d}/{stats['games']:2d} 勝 (勝率: {win_rate:.0f}%)")
    
    return results


# ================================================================================
# メイン実行
# ================================================================================

def main():
    """メイン実行関数"""
    print("🚀 CQCNN軽量学習版 - 勝率向上テスト")
    print("=" * 60)
    
    # Phase 1: 学習前のベースライン測定
    print("\n[Phase 1] 学習前ベースライン測定")
    print("-" * 40)
    
    # 未学習CQCNN
    untrained_system = IntegratedCQCNNSystem(n_qubits=4, n_layers=2)
    untrained_ai = TrainedCQCNNAI("A", untrained_system, "CQCNN_未学習")
    
    # 対戦相手
    random_ai = RandomAI("B")
    simple_ai = SimpleAI("B")
    
    print("📊 学習前の性能測定（各5ゲーム）")
    
    # 学習前のテスト
    untrained_wins = 0
    for opponent in [random_ai, simple_ai]:
        wins = 0
        for i in range(5):
            game = GeisterGame()
            
            # プレイヤー設定
            if i % 2 == 0:
                untrained_ai.player_id = "A"
                opponent.player_id = "B"
                player_a, player_b = untrained_ai, opponent
            else:
                untrained_ai.player_id = "B"
                opponent.player_id = "A"
                player_a, player_b = opponent, untrained_ai
            
            # ゲーム実行（簡略版）
            for _ in range(100):
                if game.game_over:
                    break
                current = player_a if game.current_player == "A" else player_b
                moves = game.get_legal_moves(game.current_player)
                if moves:
                    move = current.get_move(game.get_game_state(game.current_player), moves)
                    if move:
                        game.make_move(move[0], move[1])
            
            if (i % 2 == 0 and game.winner == "A") or (i % 2 == 1 and game.winner == "B"):
                wins += 1
                untrained_wins += 1
        
        print(f"  vs {opponent.name}: {wins}/5 勝")
    
    baseline_winrate = untrained_wins / 10 * 100
    print(f"\n📈 学習前勝率: {baseline_winrate:.0f}%")
    
    # Phase 2: 軽量学習
    print("\n[Phase 2] 軽量学習実行")
    print("-" * 40)
    
    trainer = LightweightTrainer(n_qubits=4, n_layers=2)
    
    # 学習データ生成
    training_data = trainer.generate_simple_training_data(num_games=15)
    
    # 学習実行
    trainer.train_quick(training_data, epochs=10)
    
    # Phase 3: 学習後テスト
    print("\n[Phase 3] 学習後性能測定")
    print("-" * 40)
    
    # 学習済みAI
    trained_ai = TrainedCQCNNAI("A", trainer.system, "CQCNN_学習済")
    
    print("📊 学習後の性能測定（各5ゲーム）")
    
    # 学習後のテスト
    trained_wins = 0
    for opponent in [random_ai, simple_ai]:
        wins = 0
        for i in range(5):
            game = GeisterGame()
            
            # プレイヤー設定
            if i % 2 == 0:
                trained_ai.player_id = "A"
                opponent.player_id = "B"
                player_a, player_b = trained_ai, opponent
            else:
                trained_ai.player_id = "B"
                opponent.player_id = "A"
                player_a, player_b = opponent, trained_ai
            
            # ゲーム実行
            for _ in range(100):
                if game.game_over:
                    break
                current = player_a if game.current_player == "A" else player_b
                moves = game.get_legal_moves(game.current_player)
                if moves:
                    move = current.get_move(game.get_game_state(game.current_player), moves)
                    if move:
                        game.make_move(move[0], move[1])
            
            if (i % 2 == 0 and game.winner == "A") or (i % 2 == 1 and game.winner == "B"):
                wins += 1
                trained_wins += 1
        
        print(f"  vs {opponent.name}: {wins}/5 勝")
    
    trained_winrate = trained_wins / 10 * 100
    print(f"\n📈 学習後勝率: {trained_winrate:.0f}%")
    
    # Phase 4: 総合比較
    print("\n[Phase 4] 総合比較トーナメント")
    print("-" * 40)
    
    # 全AI参加のトーナメント
    all_ais = [
        untrained_ai,
        trained_ai,
        random_ai,
        simple_ai,
        AggressiveAI("A")
    ]
    
    tournament_results = run_comparison_tournament(all_ais, games_per_pair=3)
    
    # Phase 5: 結果分析
    print("\n[Phase 5] 結果分析")
    print("=" * 60)
    
    print("\n📊 学習効果分析:")
    print(f"  学習前勝率: {baseline_winrate:.0f}%")
    print(f"  学習後勝率: {trained_winrate:.0f}%")
    improvement = trained_winrate - baseline_winrate
    print(f"  改善幅: {improvement:+.0f}%")
    
    if improvement > 0:
        print(f"\n✨ 学習成功！")
        print(f"  {improvement:.0f}%の性能向上を達成")
    elif improvement == 0:
        print(f"\n📈 変化なし")
        print(f"  より多くの学習データが必要かもしれません")
    else:
        print(f"\n🔄 パラメータ調整が必要")
        print(f"  学習方法の見直しが必要です")
    
    print("\n💡 改善のヒント:")
    if trainer.training_stats['after_accuracy'] > 0.6:
        print("  ✅ 駒推定精度は良好")
    else:
        print("  📈 駒推定精度の向上が必要")
    
    print("  - より多くの学習データ")
    print("  - エポック数の増加")
    print("  - 学習率の調整")
    
    print("\n✅ 実験完了！")


if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        elapsed = time.time() - start_time
        print(f"\n⏱️ 実行時間: {elapsed:.1f}秒")
    except KeyboardInterrupt:
        print("\n⚠️ 実行中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()

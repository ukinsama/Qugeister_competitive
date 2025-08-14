#!/usr/bin/env python3
"""
分離型CQCNN学習・デモ対戦システム
駒推定の学習から実戦対戦まで完全実装
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional, Any
import random
import time
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt

# プロジェクトパス追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'qugeister_competitive')
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

# 必要なファイルを直接読み込んで実行
print("📂 モジュール読み込み中...")

# game_engine.pyを読み込み
with open(os.path.join(src_path, 'game_engine.py'), 'r') as f:
    game_engine_code = f.read()
exec(game_engine_code)

# ai_base.pyを読み込み（相対インポート部分を削除）
with open(os.path.join(src_path, 'ai_base.py'), 'r') as f:
    ai_base_code = f.read()
# 相対インポート行を削除
ai_base_code = ai_base_code.replace('from .game_engine import GeisterGame, GameState', 
                                    '# from .game_engine import GeisterGame, GameState')
exec(ai_base_code)

# tournament.pyを読み込み（相対インポート部分を削除）
with open(os.path.join(src_path, 'tournament.py'), 'r') as f:
    tournament_code = f.read()
tournament_code = tournament_code.replace('from .game_engine import GeisterGame', 
                                        '# from .game_engine import GeisterGame')
tournament_code = tournament_code.replace('from .ai_base import BaseAI', 
                                        '# from .ai_base import BaseAI')
exec(tournament_code)

# separated_cqcnn_qmap.pyを読み込み
from separated_cqcnn_qmap import (
    CQCNNPieceEstimator, 
    PieceEstimationDataExporter,
    QValueMapGenerator,
    IntegratedCQCNNSystem
)

print("✅ モジュール読み込み完了")


# ================================================================================
# Part 1: CQCNN学習システム
# ================================================================================

class CQCNNTrainer:
    """CQCNN駒推定モデルの学習システム"""
    
    def __init__(self, estimator: CQCNNPieceEstimator, learning_rate: float = 0.001):
        self.estimator = estimator
        self.optimizer = optim.Adam(estimator.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'epochs': []
        }
    
    def generate_training_data(self, num_games: int = 100) -> List[Dict]:
        """ゲームプレイから学習データを生成"""
        print(f"📊 学習データ生成中 ({num_games}ゲーム)...")
        training_data = []
        
        for game_idx in range(num_games):
            game = GeisterGame()
            player_a = RandomAI("A")
            player_b = RandomAI("B")
            
            # ゲームを進行
            turn_count = 0
            max_turns = 50
            
            while not game.game_over and turn_count < max_turns:
                current_player = player_a if game.current_player == "A" else player_b
                legal_moves = game.get_legal_moves(game.current_player)
                
                if not legal_moves:
                    break
                
                # 現在の状態を記録（相手駒の真の種類を含む）
                if turn_count > 5:  # 序盤は除外
                    board_state = game.board.copy()
                    
                    # プレイヤーAの視点
                    if random.random() < 0.5:
                        enemy_pieces = game.player_b_pieces.copy()
                        data_point = {
                            'board': board_state,
                            'player': 'A',
                            'enemy_pieces': enemy_pieces,
                            'turn': turn_count
                        }
                        training_data.append(data_point)
                    
                    # プレイヤーBの視点
                    else:
                        enemy_pieces = game.player_a_pieces.copy()
                        data_point = {
                            'board': board_state,
                            'player': 'B',
                            'enemy_pieces': enemy_pieces,
                            'turn': turn_count
                        }
                        training_data.append(data_point)
                
                # 手を実行
                move = current_player.get_move(game.get_game_state(game.current_player), legal_moves)
                if move:
                    game.make_move(move[0], move[1])
                
                turn_count += 1
            
            if (game_idx + 1) % 20 == 0:
                print(f"  {game_idx + 1}/{num_games} ゲーム完了")
        
        print(f"✅ {len(training_data)}個の学習データを生成")
        return training_data
    
    def train(self, training_data: List[Dict], epochs: int = 50, batch_size: int = 32):
        """モデルを学習"""
        print(f"\n🎓 CQCNN学習開始 (エポック数: {epochs})")
        print("=" * 60)
        
        for epoch in range(epochs):
            epoch_losses = []
            correct_predictions = 0
            total_predictions = 0
            
            # データをシャッフル
            random.shuffle(training_data)
            
            # バッチ処理
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                batch_loss = 0
                batch_correct = 0
                batch_total = 0
                
                for data in batch:
                    # ボード状態をテンソルに変換
                    board_tensor = self._prepare_board_tensor(
                        data['board'], 
                        data['player']
                    )
                    
                    # 敵駒の位置と真のラベル
                    enemy_positions = []
                    true_labels = []
                    
                    enemy_val = -1 if data['player'] == 'A' else 1
                    for pos, piece_type in data['enemy_pieces'].items():
                        if data['board'][pos[1], pos[0]] == enemy_val:
                            enemy_positions.append(pos)
                            # 0: good, 1: bad
                            true_labels.append(0 if piece_type == "good" else 1)
                    
                    if not enemy_positions:
                        continue
                    
                    # 推定実行
                    estimations = self.estimator(board_tensor, enemy_positions)
                    
                    # 各駒に対する損失計算
                    for est, true_label in zip(estimations, true_labels):
                        # 推定確率を取得
                        pred_probs = torch.tensor(
                            [est.good_probability, est.bad_probability],
                            requires_grad=True
                        )
                        
                        # 損失計算
                        target = torch.tensor([true_label], dtype=torch.long)
                        loss = self.criterion(pred_probs.unsqueeze(0), target)
                        batch_loss += loss
                        
                        # 精度計算
                        pred_label = 0 if est.good_probability > est.bad_probability else 1
                        if pred_label == true_label:
                            batch_correct += 1
                        batch_total += 1
                
                # バッチ更新
                if batch_total > 0:
                    avg_loss = batch_loss / batch_total
                    self.optimizer.zero_grad()
                    avg_loss.backward()
                    self.optimizer.step()
                    
                    epoch_losses.append(avg_loss.item())
                    correct_predictions += batch_correct
                    total_predictions += batch_total
            
            # エポック統計
            if total_predictions > 0:
                epoch_accuracy = correct_predictions / total_predictions
                avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
                
                self.training_history['epochs'].append(epoch + 1)
                self.training_history['losses'].append(avg_epoch_loss)
                self.training_history['accuracies'].append(epoch_accuracy)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Loss: {avg_epoch_loss:.4f} | "
                          f"Accuracy: {epoch_accuracy:.3f}")
        
        print("\n✅ 学習完了！")
        self._plot_training_history()
    
    def _prepare_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        """ボード状態をテンソルに変換"""
        tensor = torch.zeros(1, 3, 6, 6)
        player_val = 1 if player == 'A' else -1
        enemy_val = -player_val
        
        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))
        
        return tensor
    
    def _plot_training_history(self):
        """学習履歴をプロット"""
        if not self.training_history['epochs']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失推移
        ax1.plot(self.training_history['epochs'], 
                self.training_history['losses'], 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # 精度推移
        ax2.plot(self.training_history['epochs'], 
                self.training_history['accuracies'], 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Piece Type Prediction Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('cqcnn_training_history.png')
        print("📊 学習履歴を保存: cqcnn_training_history.png")
    
    def save_model(self, filepath: str):
        """モデルを保存"""
        torch.save({
            'model_state': self.estimator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        print(f"💾 モデルを保存: {filepath}")
    
    def load_model(self, filepath: str):
        """モデルを読み込み"""
        checkpoint = torch.load(filepath)
        self.estimator.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_history = checkpoint.get('training_history', {})
        print(f"📂 モデルを読み込み: {filepath}")


# ================================================================================
# Part 2: CQCNN AIエージェント
# ================================================================================

class CQCNNAI(BaseAI):
    """分離型CQCNNを使用するAIエージェント"""
    
    def __init__(self, player_id: str, system: IntegratedCQCNNSystem, name: str = "CQCNN_AI"):
        super().__init__(name, player_id)
        self.system = system
        self.last_q_map = None
        self.exploration_rate = 0.1  # 探索率
    
    def get_move(self, game_state: GameState, legal_moves: List) -> Optional[Tuple]:
        """CQCNNとQ値マップを使って手を選択"""
        if not legal_moves:
            return None
        
        # 探索（ランダム選択）
        if random.random() < self.exploration_rate:
            return random.choice(legal_moves)
        
        # ゲーム状態を辞書形式に変換
        game_dict = {
            'board': game_state.board.tolist(),
            'current_player': self.player_id,
            'turn': game_state.turn
        }
        
        # 自分の駒情報
        my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
        
        # 駒推定とQ値マップ生成
        try:
            estimation_data, q_map = self.system.process_game_state(game_dict, my_pieces)
            self.last_q_map = q_map
            
            # 最適な合法手を選択
            best_move = None
            best_q = -float('inf')
            
            for move in legal_moves:
                from_pos, to_pos = move
                
                # 移動方向を判定
                dx = to_pos[0] - from_pos[0]
                dy = to_pos[1] - from_pos[1]
                
                # 方向インデックス
                if dy == -1 and dx == 0:    dir_idx = 0  # 上
                elif dx == 1 and dy == 0:   dir_idx = 1  # 右
                elif dy == 1 and dx == 0:   dir_idx = 2  # 下
                elif dx == -1 and dy == 0:  dir_idx = 3  # 左
                else: continue  # 斜め移動は無視
                
                # Q値を取得
                q_value = q_map[from_pos[1], from_pos[0], dir_idx]
                
                if q_value > best_q:
                    best_q = q_value
                    best_move = move
            
            return best_move if best_move else random.choice(legal_moves)
            
        except Exception as e:
            print(f"⚠️ CQCNN推定エラー: {e}")
            return random.choice(legal_moves)


# ================================================================================
# Part 3: デモ対戦システム
# ================================================================================

class CQCNNDemoBattle:
    """CQCNN AIのデモ対戦システム"""
    
    def __init__(self):
        self.results = {
            'games': [],
            'win_rates': {},
            'statistics': {}
        }
    
    def run_single_game(self, player_a: BaseAI, player_b: BaseAI, 
                       verbose: bool = False) -> Dict:
        """単一ゲーム実行"""
        game = GeisterGame()
        move_history = []
        
        if verbose:
            print(f"\n🎮 {player_a.name} vs {player_b.name}")
            print("-" * 40)
        
        max_turns = 100
        for turn in range(max_turns):
            current_player = player_a if game.current_player == "A" else player_b
            legal_moves = game.get_legal_moves(game.current_player)
            
            if not legal_moves:
                break
            
            # 手を選択
            game_state = game.get_game_state(game.current_player)
            move = current_player.get_move(game_state, legal_moves)
            
            if not move:
                break
            
            # 手を実行
            success = game.make_move(move[0], move[1])
            if not success:
                break
            
            move_history.append({
                'turn': turn,
                'player': game.current_player,
                'move': move
            })
            
            if verbose and turn < 10:  # 最初の10手のみ表示
                print(f"  Turn {turn+1}: {game.current_player} {move[0]} → {move[1]}")
            
            if game.game_over:
                break
        
        # 結果記録
        result = {
            'player_a': player_a.name,
            'player_b': player_b.name,
            'winner': game.winner,
            'turns': len(move_history),
            'move_history': move_history
        }
        
        if verbose:
            if game.winner in ["A", "B"]:
                winner_name = player_a.name if game.winner == "A" else player_b.name
                print(f"🏆 勝者: {winner_name}")
            else:
                print("🤝 引き分け")
            print(f"📊 総手数: {len(move_history)}")
        
        return result
    
    def run_tournament(self, cqcnn_ai: CQCNNAI, opponents: List[BaseAI], 
                      games_per_opponent: int = 10) -> Dict:
        """トーナメント実行"""
        print("\n🏆 CQCNNトーナメント開始")
        print("=" * 60)
        
        total_games = 0
        total_wins = 0
        
        for opponent in opponents:
            wins = 0
            draws = 0
            losses = 0
            
            print(f"\n📊 vs {opponent.name} ({games_per_opponent}ゲーム)")
            
            for game_num in range(games_per_opponent):
                # 先手後手を交代
                if game_num % 2 == 0:
                    cqcnn_ai.player_id = "A"
                    opponent.player_id = "B"
                    result = self.run_single_game(cqcnn_ai, opponent)
                    
                    if result['winner'] == "A":
                        wins += 1
                    elif result['winner'] == "B":
                        losses += 1
                    else:
                        draws += 1
                else:
                    cqcnn_ai.player_id = "B"
                    opponent.player_id = "A"
                    result = self.run_single_game(opponent, cqcnn_ai)
                    
                    if result['winner'] == "B":
                        wins += 1
                    elif result['winner'] == "A":
                        losses += 1
                    else:
                        draws += 1
                
                self.results['games'].append(result)
                total_games += 1
                
                if result['winner'] == cqcnn_ai.player_id:
                    total_wins += 1
            
            # 対戦相手別の結果
            win_rate = wins / games_per_opponent
            self.results['win_rates'][opponent.name] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': win_rate
            }
            
            print(f"  結果: {wins}勝 {losses}敗 {draws}分 (勝率: {win_rate:.1%})")
        
        # 全体統計
        overall_win_rate = total_wins / total_games if total_games > 0 else 0
        self.results['statistics'] = {
            'total_games': total_games,
            'total_wins': total_wins,
            'overall_win_rate': overall_win_rate
        }
        
        print("\n" + "=" * 60)
        print(f"📈 総合成績: {total_wins}/{total_games} (勝率: {overall_win_rate:.1%})")
        
        return self.results
    
    def analyze_performance(self):
        """パフォーマンス分析"""
        print("\n📊 パフォーマンス分析")
        print("-" * 40)
        
        # 対戦相手別分析
        for opponent_name, stats in self.results['win_rates'].items():
            print(f"\nvs {opponent_name}:")
            print(f"  勝率: {stats['win_rate']:.1%}")
            print(f"  詳細: {stats['wins']}勝 {stats['losses']}敗 {stats['draws']}分")
        
        # ゲーム長分析
        game_lengths = [g['turns'] for g in self.results['games']]
        if game_lengths:
            avg_length = np.mean(game_lengths)
            print(f"\n平均ゲーム長: {avg_length:.1f}手")
            print(f"最短ゲーム: {min(game_lengths)}手")
            print(f"最長ゲーム: {max(game_lengths)}手")


# ================================================================================
# Part 4: メイン実行
# ================================================================================

def main():
    """メイン実行関数"""
    print("🚀 分離型CQCNN学習・デモ対戦システム")
    print("=" * 70)
    
    # 1. システム初期化
    print("\n[Phase 1] システム初期化")
    print("-" * 40)
    
    # CQCNNシステム作成
    system = IntegratedCQCNNSystem(n_qubits=8, n_layers=3)
    print("✅ CQCNNシステム初期化完了")
    
    # 学習器作成
    trainer = CQCNNTrainer(system.estimator)
    print("✅ 学習システム初期化完了")
    
    # 2. 学習データ生成と学習
    print("\n[Phase 2] 学習フェーズ")
    print("-" * 40)
    
    # 学習データ生成
    training_data = trainer.generate_training_data(num_games=50)
    
    # モデル学習
    trainer.train(training_data, epochs=30, batch_size=16)
    
    # モデル保存
    trainer.save_model("cqcnn_model.pth")
    
    # 3. CQCNN AIエージェント作成
    print("\n[Phase 3] AIエージェント作成")
    print("-" * 40)
    
    cqcnn_ai = CQCNNAI("A", system, "CQCNN_AI_v1")
    print(f"✅ {cqcnn_ai.name}を作成")
    
    # 対戦相手作成
    opponents = [
        RandomAI("B"),
        SimpleAI("B"),
        AggressiveAI("B")
    ]
    
    print("✅ 対戦相手AI準備完了:")
    for opp in opponents:
        print(f"  - {opp.name}")
    
    # 4. デモ対戦
    print("\n[Phase 4] デモ対戦")
    print("-" * 40)
    
    # 単一ゲームデモ
    print("\n💡 デモゲーム: CQCNN_AI vs RandomAI")
    demo_battle = CQCNNDemoBattle()
    demo_result = demo_battle.run_single_game(cqcnn_ai, opponents[0], verbose=True)
    
    # 5. トーナメント実行
    print("\n[Phase 5] トーナメント")
    print("-" * 40)
    
    tournament_results = demo_battle.run_tournament(
        cqcnn_ai, 
        opponents, 
        games_per_opponent=20
    )
    
    # 6. 結果分析
    print("\n[Phase 6] 結果分析")
    print("-" * 40)
    
    demo_battle.analyze_performance()
    
    # 7. 結果保存
    print("\n[Phase 7] 結果保存")
    print("-" * 40)
    
    # 結果をJSONで保存
    with open("cqcnn_demo_results.json", "w") as f:
        json.dump({
            'training_history': trainer.training_history,
            'tournament_results': tournament_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print("✅ 結果を保存: cqcnn_demo_results.json")
    
    # 8. 実験サマリー
    print("\n" + "=" * 70)
    print("🎯 実験完了サマリー")
    print("=" * 70)
    
    print(f"学習:")
    print(f"  - 学習ゲーム数: 50")
    print(f"  - エポック数: 30")
    if trainer.training_history['accuracies']:
        final_acc = trainer.training_history['accuracies'][-1]
        print(f"  - 最終精度: {final_acc:.3f}")
    
    print(f"\nトーナメント:")
    print(f"  - 総ゲーム数: {tournament_results['statistics']['total_games']}")
    print(f"  - 総合勝率: {tournament_results['statistics']['overall_win_rate']:.1%}")
    
    print("\n🎉 実験完了！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 実行中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()

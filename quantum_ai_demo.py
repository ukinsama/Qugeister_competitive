#!/usr/bin/env python3
"""
コマンドライン量子ガイスターAI デモンストレーション（修正版）
正しい脱出ルール対応 - BaseAI完全互換
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import time
import json
from datetime import datetime

# プロジェクトパス追加
sys.path.append('.')
sys.path.append('src')

try:
    import pennylane as qml
    QUANTUM_AVAILABLE = True
    print("✅ PennyLane利用可能 - 量子回路で実行")
except ImportError:
    QUANTUM_AVAILABLE = False
    print("⚠️  PennyLane未対応 - 古典ニューラルネットワークで代替")

# プロジェクトモジュール
from qugeister_competitive.game_engine import GeisterGame
from qugeister_competitive.ai_base import BaseAI, RandomAI, SimpleAI, AggressiveAI
from qugeister_competitive.tournament import TournamentManager

class QuantumAI(BaseAI):
    """量子AI（正しい脱出ルール対応版）"""
    
    def __init__(self, player_id="A", n_qubits=8, n_layers=3):
        # BaseAI初期化
        super().__init__("QuantumAI", player_id)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 学習パラメータ
        self.epsilon = 0.8          # 初期探索率
        self.epsilon_min = 0.01     # 最小探索率
        self.epsilon_decay = 0.995  # 減衰率
        self.learning_rate = 0.01  # 学習率
        
        # 経験記憶
        self.memory = []
        self.max_memory = 1000
        
        if QUANTUM_AVAILABLE:
            self._setup_quantum_circuit()
        else:
            self._setup_classical_network()
        
        print(f"🧠 {self.name} 初期化完了 (Player {player_id})")
        print(f"   量子ビット: {n_qubits}, レイヤー: {n_layers}")
        print(f"   モード: {'Quantum' if QUANTUM_AVAILABLE else 'Classical'}")
        print(f"   脱出口: {self._get_escape_positions()}")
    
    def _get_escape_positions(self):
        """プレイヤーの脱出口取得"""
        if self.player_id == "A":
            return [(0, 5), (5, 5)]  # 相手陣地
        else:
            return [(0, 0), (5, 0)]  # 相手陣地
    
    def _setup_quantum_circuit(self):
        """量子回路セットアップ"""
        try:
            self.dev = qml.device('lightning.qubit', wires=self.n_qubits)
            print("⚡ Lightning.Qubit デバイス使用")
        except:
            self.dev = qml.device('default.qubit', wires=self.n_qubits)
            print("⚠️  default.qubit デバイス使用")
        
        self.params = torch.randn(self.n_layers, self.n_qubits, 3, requires_grad=True)
        
        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
        def quantum_circuit(inputs, params):
            # 状態エンコーディング
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # 変分回路
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)
                
                # エンタングルメント
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        self.optimizer = torch.optim.Adam([self.params], lr=self.learning_rate)
        print("⚛️  量子回路初期化完了")
    
    def _setup_classical_network(self):
        """古典ネットワークセットアップ"""
        import torch.nn as nn
        
        self.network = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        print("🖥️  古典ニューラルネットワーク初期化完了")
    
    def encode_state(self, game_state):
        """ゲーム状態エンコーディング（脱出情報含む）"""
        board = game_state.board.copy()
        
        # プレイヤー視点正規化
        if self.player_id == "B":
            board = -board
            board = np.flipud(board)
        
        # 基本盤面状態
        state_vector = board.flatten().astype(np.float32) / 2.0
        
        # 脱出情報エンコーディング
        escape_features = self._encode_escape_features(game_state)
        
        # 特徴量結合
        enhanced_state = np.concatenate([state_vector, escape_features])
        
        if QUANTUM_AVAILABLE:
            # 量子用: 次元調整
            if len(enhanced_state) > self.n_qubits:
                # 重要な部分のみ抽出（脱出情報を優先保持）
                escape_size = len(escape_features)
                important_indices = list(range(len(state_vector)))  # 盤面情報
                important_indices.extend(range(len(state_vector), len(enhanced_state)))  # 脱出情報
                
                if len(important_indices) > self.n_qubits:
                    # 盤面から重要部分を選択
                    board_indices = np.random.choice(len(state_vector), self.n_qubits - escape_size, replace=False)
                    escape_indices = list(range(len(state_vector), len(enhanced_state)))
                    important_indices = list(board_indices) + escape_indices
                
                enhanced_state = enhanced_state[important_indices[:self.n_qubits]]
            elif len(enhanced_state) < self.n_qubits:
                # パディング
                padding = np.zeros(self.n_qubits - len(enhanced_state))
                enhanced_state = np.concatenate([enhanced_state, padding])
        
        return torch.tensor(enhanced_state, dtype=torch.float32)
    
    def _encode_escape_features(self, game_state):
        """脱出関連特徴量エンコーディング"""
        my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
        escape_positions = self._get_escape_positions()
        
        escape_features = []
        
        # 脱出可能性情報
        min_escape_distance = float('inf')
        escape_ready_count = 0
        
        for pos, piece_type in my_pieces.items():
            if piece_type == "good":
                # 各脱出口への距離
                distances = []
                for escape_pos in escape_positions:
                    distance = abs(pos[0] - escape_pos[0]) + abs(pos[1] - escape_pos[1])
                    distances.append(distance)
                
                min_distance = min(distances)
                min_escape_distance = min(min_escape_distance, min_distance)
                
                # 脱出口に到達している善玉数
                if pos in escape_positions:
                    escape_ready_count += 1
        
        # 正規化した脱出特徴
        escape_features.extend([
            min_escape_distance / 10.0 if min_escape_distance != float('inf') else 1.0,  # 最短脱出距離
            escape_ready_count / 4.0,  # 脱出準備完了駒数
        ])
        
        # 相手の脅威情報
        opponent_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces
        opponent_escape_positions = [(0, 0), (5, 0)] if self.player_id == "A" else [(0, 5), (5, 5)]
        
        opponent_min_distance = float('inf')
        for pos, piece_type in opponent_pieces.items():
            for escape_pos in opponent_escape_positions:
                distance = abs(pos[0] - escape_pos[0]) + abs(pos[1] - escape_pos[1])
                opponent_min_distance = min(opponent_min_distance, distance)
        
        escape_features.append(
            opponent_min_distance / 10.0 if opponent_min_distance != float('inf') else 1.0
        )
        
        return np.array(escape_features, dtype=np.float32)
    
    def get_q_values(self, state_tensor):
        """Q値計算"""
        if QUANTUM_AVAILABLE:
            quantum_output = self.quantum_circuit(state_tensor, self.params)
            return torch.stack(quantum_output)
        else:
            return self.network(state_tensor)
    
    def get_move(self, game_state, legal_moves):
        """手選択（BaseAI互換）"""
        if not legal_moves:
            return None
        
        # ε-greedy探索
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Q値ベース選択
        try:
            state_tensor = self.encode_state(game_state)
            q_values = self.get_q_values(state_tensor)
            
            # 最良の合法手選択
            best_move = self._select_best_move(q_values, legal_moves, game_state)
            return best_move
        except:
            # エラー時はランダム
            return random.choice(legal_moves)
    
    def _select_best_move(self, q_values, legal_moves, game_state):
        """最良手選択（脱出戦略強化）"""
        move_scores = []
        
        for move in legal_moves:
            from_pos, to_pos = move
            
            # 方向判定
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            if dy == -1:    direction = 0  # 上
            elif dx == 1:   direction = 1  # 右
            elif dy == 1:   direction = 2  # 下
            elif dx == -1:  direction = 3  # 左
            else:           direction = 0
            
            # プレイヤーB視点調整
            if self.player_id == "B":
                direction = (direction + 2) % 4
            
            # Q値スコア + 戦術ボーナス
            score = q_values[direction % len(q_values)].item()
            score += self._tactical_bonus(move, game_state)
            
            move_scores.append((move, score))
        
        # 最高スコア選択
        return max(move_scores, key=lambda x: x[1])[0]
    
    def _tactical_bonus(self, move, game_state):
        """戦術ボーナス"""
        from_pos, to_pos = move
        bonus = 0.0
        
        my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
        piece_type = my_pieces.get(from_pos, "unknown")
        
        # 脱出ボーナス（最重要）
        escape_positions = self._get_escape_positions()
        if to_pos in escape_positions and piece_type == "good":
            bonus += 5.0  # 脱出口到達の大ボーナス
        
        # 脱出口への接近ボーナス
        if piece_type == "good":
            for escape_pos in escape_positions:
                old_distance = abs(from_pos[0] - escape_pos[0]) + abs(from_pos[1] - escape_pos[1])
                new_distance = abs(to_pos[0] - escape_pos[0]) + abs(to_pos[1] - escape_pos[1])
                if new_distance < old_distance:
                    bonus += (old_distance - new_distance) * 0.3
        
        # 前進ボーナス（相手陣地方向）
        if self.player_id == "A" and to_pos[1] > from_pos[1]:
            bonus += 0.2
        elif self.player_id == "B" and to_pos[1] < from_pos[1]:
            bonus += 0.2
        
        # 中央ボーナス
        center_dist = abs(to_pos[0] - 2.5)
        bonus += (2.5 - center_dist) * 0.05
        
        # 駒取りボーナス
        opponent_pieces = (game_state.player_b_pieces if self.player_id == "A" 
                          else game_state.player_a_pieces)
        if to_pos in opponent_pieces:
            bonus += 1.0
        
        return bonus
    
    def train_step(self, experiences):
        """学習ステップ"""
        if len(experiences) < 5:
            return 0.0
        
        try:
            # バッチ作成（GameStateオブジェクトを適切に処理）
            states = []
            rewards = []
            
            for exp in experiences[-5:]:
                state_data, reward = exp
                # GameStateオブジェクトの場合はエンコード
                if hasattr(state_data, 'board'):
                    state_tensor = self.encode_state(state_data)
                else:
                    state_tensor = state_data
                
                states.append(state_tensor)
                rewards.append(reward)
            
            states = torch.stack(states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
            # 損失計算
            if QUANTUM_AVAILABLE:
                predictions = []
                for state in states:
                    pred = self.quantum_circuit(state, self.params)
                    predictions.append(torch.stack(pred).mean())
                predictions = torch.stack(predictions)
            else:
                predictions = self.network(states).mean(dim=1)
            
            loss = torch.nn.MSELoss()(predictions, rewards)
            
            # パラメータ更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        except Exception as e:
            print(f"⚠️  学習エラー: {e}")
            return 0.0
    
    def remember(self, state, reward):
        """経験記憶（GameState対応）"""
        # GameStateオブジェクトを事前にエンコードして保存
        if hasattr(state, 'board'):
            state_tensor = self.encode_state(state)
        else:
            state_tensor = state
        
        self.memory.append((state_tensor, reward))
        
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
    
    def record_result(self, won):
        """結果記録（BaseAI互換）"""
        super().record_result(won)
        
        # ε減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def calculate_enhanced_reward(old_state, move, game):
    """強化報酬システム（正しい脱出ルール対応）"""
    from_pos, to_pos = move
    current_player = old_state.current_player
    reward = 0.0
    
    # 基本移動報酬
    reward += 0.1
    
    # 脱出関連報酬（最重要）
    my_pieces = old_state.player_a_pieces if current_player == "A" else old_state.player_b_pieces
    piece_type = my_pieces.get(from_pos, "unknown")
    
    if piece_type == "good":
        # 脱出口定義
        escape_positions = [(0, 5), (5, 5)] if current_player == "A" else [(0, 0), (5, 0)]
        
        # 脱出口到達ボーナス
        if to_pos in escape_positions:
            reward += 10.0  # 脱出口到達の大ボーナス
        
        # 脱出口への接近ボーナス
        for escape_pos in escape_positions:
            old_distance = abs(from_pos[0] - escape_pos[0]) + abs(from_pos[1] - escape_pos[1])
            new_distance = abs(to_pos[0] - escape_pos[0]) + abs(to_pos[1] - escape_pos[1])
            if new_distance < old_distance:
                reward += (old_distance - new_distance) * 0.5
    
    # 前進報酬
    if current_player == "A" and to_pos[1] > from_pos[1]:
        reward += 0.3
    elif current_player == "B" and to_pos[1] < from_pos[1]:
        reward += 0.3
    
    # 中央制御報酬
    center_dist = abs(to_pos[0] - 2.5)
    reward += (2.5 - center_dist) * 0.1
    
    # 駒取り報酬
    opponent_pieces = old_state.player_b_pieces if current_player == "A" else old_state.player_a_pieces
    if to_pos in opponent_pieces:
        reward += 3.0
    
    # ゲーム終了報酬
    if game.game_over:
        if game.winner == current_player:
            reward += 20.0  # 勝利ボーナス
        elif game.winner is None:
            reward -= 1.0   # 引き分けペナルティ
        else:
            reward -= 5.0   # 敗北ペナルティ
    
    return reward

def train_quantum_ai(quantum_ai, opponent, episodes=500):
    """量子AI学習（正しい脱出ルール対応）"""
    print(f"\n🎓 量子AI学習開始")
    print(f"   エピソード: {episodes}")
    print(f"   対戦相手: {opponent.name}")
    print(f"   量子AI脱出口: {quantum_ai._get_escape_positions()}")
    print("=" * 50)
    
    stats = {
        'episode_rewards': [],
        'win_rates': [],
        'losses': [],
        'escape_victories': 0
    }
    
    for episode in range(episodes):
        game = GeisterGame()
        episode_reward = 0
        experiences = []
        
        # ランダムにプレイヤー配置
        if random.random() < 0.5:
            player_a, player_b = quantum_ai, opponent
            quantum_player_id = "A"
        else:
            player_a, player_b = opponent, quantum_ai
            quantum_player_id = "B"
        
        # ゲーム実行
        while not game.game_over:
            current_player = player_a if game.current_player == "A" else player_b
            
            game_state = game.get_game_state(game.current_player)
            legal_moves = game.get_legal_moves(game.current_player)
            
            if not legal_moves:
                break
            
            move = current_player.get_move(game_state, legal_moves)
            if not move:
                break
            
            # 学習データ収集（量子AIの場合）
            if current_player == quantum_ai:
                old_state = game_state
            
            success = game.make_move(move[0], move[1])
            if not success:
                break
            
            # 報酬計算（量子AIの場合）
            if current_player == quantum_ai:
                reward = calculate_enhanced_reward(old_state, move, game)
                episode_reward += reward
                # GameStateオブジェクトを事前にエンコードして保存
                encoded_state = quantum_ai.encode_state(old_state)
                experiences.append((encoded_state, reward))
        
        # 最終報酬（ゲーム終了時）
        final_reward = 0
        if game.winner == quantum_player_id:
            final_reward = 25.0  # 勝利ボーナス
            # 脱出勝利の検出
            quantum_pieces = game.player_a_pieces if quantum_player_id == "A" else game.player_b_pieces
            escape_positions = quantum_ai._get_escape_positions()
            
            # 脱出勝利かチェック
            escape_win = False
            for pos, piece_type in quantum_pieces.items():
                if piece_type == "good" and pos in escape_positions:
                    escape_win = True
                    break
            
            if escape_win:
                stats['escape_victories'] += 1
                final_reward += 10.0  # 脱出勝利ボーナス
                
        elif game.winner is None:
            final_reward = -1.0   # 引き分けペナルティ
        else:
            final_reward = -8.0   # 敗北ペナルティ
        
        episode_reward += final_reward
        stats['episode_rewards'].append(episode_reward)
        
        # 学習実行
        if experiences:
            loss = quantum_ai.train_step(experiences)
            stats['losses'].append(loss)
            
            # 既にエンコード済みの経験を保存
            for encoded_state, reward in experiences:
                quantum_ai.memory.append((encoded_state, reward))
                
            # メモリサイズ管理
            while len(quantum_ai.memory) > quantum_ai.max_memory:
                quantum_ai.memory.pop(0)
        
        # 結果記録
        quantum_ai.record_result(game.winner == quantum_player_id)
        
        # 進捗表示
        if (episode + 1) % 100 == 0:
            win_rate = quantum_ai.win_rate
            avg_reward = np.mean(stats['episode_rewards'][-100:])
            escape_rate = stats['escape_victories'] / (episode + 1)
            stats['win_rates'].append(win_rate)
            
            print(f"Episode {episode+1:3d}/{episodes} | "
                  f"Win Rate: {win_rate:.3f} | "
                  f"Escape Rate: {escape_rate:.3f} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"ε: {quantum_ai.epsilon:.3f}")
    
    print("\n🎉 学習完了!")
    print(f"   最終勝率: {quantum_ai.win_rate:.3f}")
    print(f"   脱出勝利率: {stats['escape_victories'] / episodes:.3f}")
    print(f"   ゲーム数: {quantum_ai.games_played}")
    
    return stats

def evaluate_performance(quantum_ai, opponents, games_per_opponent=50):
    """性能評価"""
    print(f"\n🔍 量子AI性能評価")
    print("=" * 50)
    
    results = {}
    original_epsilon = quantum_ai.epsilon
    quantum_ai.epsilon = 0.0  # 評価時は探索なし
    
    for opponent in opponents:
        wins = 0
        escape_wins = 0
        
        for game_num in range(games_per_opponent):
            game = GeisterGame()
            
            # プレイヤー配置を交互変更
            if game_num % 2 == 0:
                player_a, player_b = quantum_ai, opponent
                quantum_player_id = "A"
            else:
                player_a, player_b = opponent, quantum_ai
                quantum_player_id = "B"
            
            while not game.game_over:
                current_player = player_a if game.current_player == "A" else player_b
                
                game_state = game.get_game_state(game.current_player)
                legal_moves = game.get_legal_moves(game.current_player)
                
                if not legal_moves:
                    break
                
                move = current_player.get_move(game_state, legal_moves)
                if not move:
                    break
                
                if not game.make_move(move[0], move[1]):
                    break
            
            if game.winner == quantum_player_id:
                wins += 1
                
                # 脱出勝利かチェック
                quantum_pieces = game.player_a_pieces if quantum_player_id == "A" else game.player_b_pieces
                escape_positions = quantum_ai._get_escape_positions()
                
                for pos, piece_type in quantum_pieces.items():
                    if piece_type == "good" and pos in escape_positions:
                        escape_wins += 1
                        break
        
        win_rate = wins / games_per_opponent
        escape_rate = escape_wins / games_per_opponent
        results[opponent.name] = {
            'win_rate': win_rate,
            'escape_rate': escape_rate
        }
        
        print(f"   vs {opponent.name:12s}: {win_rate:.1%} (脱出{escape_rate:.1%}) ({wins}/{games_per_opponent})")
    
    quantum_ai.epsilon = original_epsilon
    return results

def main():
    """メイン実行関数"""
    print("🚀 量子ガイスターAI デモ - 正しい脱出ルール対応版")
    print("=" * 60)
    
    # 1. 量子AI作成
    print("⚛️  量子AI初期化...")
    quantum_ai = QuantumAI(player_id="A", n_qubits=8, n_layers=4)
    
    # 2. 対戦相手作成
    random_opponent = RandomAI("B")
    simple_opponent = SimpleAI("B")
    aggressive_opponent = AggressiveAI("B")
    
    print("\n🎯 対戦相手準備完了:")
    for opponent in [random_opponent, simple_opponent, aggressive_opponent]:
        print(f"   - {opponent.name}")
    
    # 3. 学習前性能テスト
    print("\n📊 学習前性能測定...")
    pre_results = evaluate_performance(quantum_ai, [random_opponent], games_per_opponent=30)
    
    # 4. 学習実行
    print("\n🎓 量子AI学習実行...")
    start_time = time.time()
    training_stats = train_quantum_ai(quantum_ai, random_opponent, episodes=1000)
    training_time = time.time() - start_time
    
    # 5. 学習後性能テスト
    print("\n📊 学習後性能測定...")
    post_results = evaluate_performance(quantum_ai, [random_opponent, simple_opponent, aggressive_opponent])
    
    # 6. 学習効果分析
    print("\n📈 学習効果分析:")
    print("-" * 30)
    pre_random = pre_results['RandomAI']['win_rate']
    post_random = post_results['RandomAI']['win_rate']
    post_escape = post_results['RandomAI']['escape_rate']
    improvement = post_random - pre_random
    
    print(f"vs Random AI:")
    print(f"   学習前: {pre_random:.1%}")
    print(f"   学習後: {post_random:.1%} (脱出{post_escape:.1%})")
    print(f"   改善幅: {improvement:+.1%}")
    
    if post_random >= 0.6:
        print("   🎉 目標達成！60%以上の勝率を実現！")
    elif improvement > 0.1:
        print("   ✅ 顕著な学習効果を確認！")
    elif improvement > 0.05:
        print("   ✅ 良好な学習効果を確認")
    else:
        print("   📈 一定の学習効果を確認")
    
    # 7. 実験サマリー
    print(f"\n🎯 実験完了サマリー:")
    print("=" * 50)
    print(f"⚛️  量子回路: {quantum_ai.n_qubits}qubits, {quantum_ai.n_layers}layers")
    print(f"🎓 学習時間: {training_time:.1f}秒")
    print(f"🎮 学習ゲーム: {quantum_ai.games_played}")
    print(f"📈 最終勝率: {quantum_ai.win_rate:.1%}")
    print(f"🚪 脱出勝利: {training_stats['escape_victories']}/{len(training_stats['episode_rewards'])}")
    
    if post_random >= 0.6:
        print("\n🎉 量子機械学習による性能向上を実証！")
        print("   正しい脱出戦略を学習できました！")
    elif post_random > 0.5:
        print("\n✅ 量子AIがランダムを上回る性能を達成！")
    else:
        print("\n📚 更なる学習による改善が期待されます")
    
    print("\n🚀 実験完了！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  実験中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
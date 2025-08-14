#!/usr/bin/env python3
"""
Mini2ガイスター統合システム修正版
Part 1, 2, 3を統合した完全版
"""

import numpy as np
import random
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
import json
import os
import time
import threading
from datetime import datetime
import argparse
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

# Part 1からのコード統合
class Mini2GeisterGame:
    """2駒版Miniガイスターゲームエンジン"""
    
    def __init__(self):
        self.board_size = 4
        self.pieces_per_player = 2  # 善玉1 + 悪玉1
        
        # 脱出口位置（6x6との相対位置維持）
        self.escape_positions = {
            'A': [(3, 0), (3, 3)],  # プレイヤーAの脱出口
            'B': [(0, 0), (0, 3)]   # プレイヤーBの脱出口
        }
        
        # 初期配置位置
        self.initial_positions = {
            'A': [(3, 1), (3, 2)],  # 下側1行の中央2列
            'B': [(0, 1), (0, 2)]   # 上側1行の中央2列
        }
        
        self.reset_game()
    
    def reset_game(self):
        """ゲームリセット"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 'A'
        self.game_over = False
        self.winner = None
        self.win_reason = None
        self.move_count = 0
        
        # 駒の情報
        self.pieces = {
            'A': {'positions': [], 'types': []},  # types: 0=悪玉, 1=善玉
            'B': {'positions': [], 'types': []}
        }
        
        # ランダムに善悪を決めて配置
        self._setup_random_pieces()
    
    def _setup_random_pieces(self):
        """各プレイヤーの2駒にランダムで善悪割り当て"""
        for player in ['A', 'B']:
            positions = self.initial_positions[player].copy()
            types = [0, 1]  # 0=悪玉, 1=善玉
            random.shuffle(types)
            
            for i, pos in enumerate(positions):
                r, c = pos
                piece_id = len(self.pieces[player]['positions']) + 1
                if player == 'A':
                    self.board[r, c] = piece_id  # A駒: 1,2
                else:
                    self.board[r, c] = piece_id + 10  # B駒: 11,12
                
                self.pieces[player]['positions'].append(pos)
                self.pieces[player]['types'].append(types[i])
    
    def get_state_tensor(self, player_perspective='A'):
        """ゲーム状態をテンソル形式で取得"""
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        
        my_pieces = self.pieces[player_perspective]['positions']
        opponent = 'B' if player_perspective == 'A' else 'A'
        opponent_pieces = self.pieces[opponent]['positions']
        
        # 自分の駒
        for pos in my_pieces:
            if pos:  # None でない場合
                r, c = pos
                state[0, r, c] = 1.0
        
        # 相手の駒
        for pos in opponent_pieces:
            if pos:
                r, c = pos
                state[1, r, c] = 1.0
        
        # 脱出口
        for r, c in self.escape_positions[player_perspective]:
            state[2, r, c] = 1.0
        
        # 空きマス
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0:
                    state[3, r, c] = 1.0
        
        return torch.tensor(state, dtype=torch.float32)
    
    def get_possible_actions(self, player=None):
        """可能な行動リストを取得"""
        if player is None:
            player = self.current_player
        
        actions = []
        piece_positions = self.pieces[player]['positions']
        
        for i, pos in enumerate(piece_positions):
            if pos is None:  # 取られた駒はスキップ
                continue
                
            r, c = pos
            
            # 4方向への移動をチェック
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_r, new_c = r + dr, c + dc
                
                # 盤内移動
                if 0 <= new_r < self.board_size and 0 <= new_c < self.board_size:
                    if self._is_valid_move(player, (r, c), (new_r, new_c)):
                        actions.append(((r, c), (new_r, new_c)))
                
                # 脱出移動（善玉のみ）
                elif self._can_escape(player, i, (r, c), (new_r, new_c)):
                    actions.append(((r, c), (new_r, new_c)))
        
        return actions
    
    def _is_valid_move(self, player, from_pos, to_pos):
        """盤内移動の有効性チェック"""
        r, c = to_pos
        target_piece = self.board[r, c]
        
        if target_piece == 0:  # 空きマス
            return True
        
        # 相手の駒なら取れる
        if player == 'A' and target_piece > 10:
            return True
        elif player == 'B' and 1 <= target_piece <= 10:
            return True
        
        return False
    
    def _can_escape(self, player, piece_index, from_pos, to_pos):
        """脱出可能性チェック"""
        # 善玉のみ脱出可能
        if self.pieces[player]['types'][piece_index] != 1:
            return False
        
        # 脱出口からの移動のみ
        if from_pos not in self.escape_positions[player]:
            return False
        
        return True
    
    def make_move(self, from_pos, to_pos):
        """手を実行"""
        if self.game_over:
            return False
        
        player = self.current_player
        r1, c1 = from_pos
        r2, c2 = to_pos
        
        # 盤外移動（脱出）の場合
        if not (0 <= r2 < self.board_size and 0 <= c2 < self.board_size):
            return self._execute_escape(player, from_pos)
        
        # 通常移動
        return self._execute_normal_move(player, from_pos, to_pos)
    
    def _execute_escape(self, player, from_pos):
        """脱出実行"""
        # 駒のインデックスを見つける
        piece_index = None
        for i, pos in enumerate(self.pieces[player]['positions']):
            if pos == from_pos:
                piece_index = i
                break
        
        if piece_index is None:
            return False
        
        # 善玉脱出勝利
        if self.pieces[player]['types'][piece_index] == 1:
            self.board[from_pos[0], from_pos[1]] = 0
            self.pieces[player]['positions'][piece_index] = None
            self.winner = player
            self.win_reason = "escape"
            self.game_over = True
            return True
        
        return False
    
    def _execute_normal_move(self, player, from_pos, to_pos):
        """通常移動実行"""
        r1, c1 = from_pos
        r2, c2 = to_pos
        
        moving_piece = self.board[r1, c1]
        target_piece = self.board[r2, c2]
        
        # 駒のインデックスを見つける
        piece_index = None
        for i, pos in enumerate(self.pieces[player]['positions']):
            if pos == from_pos:
                piece_index = i
                break
        
        if piece_index is None:
            return False
        
        # 相手駒を取る場合
        if target_piece != 0:
            opponent = 'B' if player == 'A' else 'A'
            captured_index = None
            
            for i, pos in enumerate(self.pieces[opponent]['positions']):
                if pos == to_pos:
                    captured_index = i
                    break
            
            if captured_index is not None:
                captured_type = self.pieces[opponent]['types'][captured_index]
                self.pieces[opponent]['positions'][captured_index] = None
                
                # 勝利条件チェック
                if captured_type == 1:  # 善玉を取った
                    self.winner = player
                    self.win_reason = "captured_good"
                    self.game_over = True
                elif captured_type == 0:  # 悪玉を取った
                    self.winner = opponent
                    self.win_reason = "captured_bad"
                    self.game_over = True
        
        # 移動実行
        self.board[r1, c1] = 0
        self.board[r2, c2] = moving_piece
        self.pieces[player]['positions'][piece_index] = to_pos
        
        self.move_count += 1
        self.current_player = 'B' if player == 'A' else 'A'
        
        return True
    
    def is_terminal(self):
        """ゲーム終了判定"""
        return self.game_over
    
    def get_reward(self, player):
        """報酬計算"""
        if not self.game_over:
            return 0.0
        
        if self.winner == player:
            if self.win_reason == "escape":
                return 100.0  # 脱出勝利
            elif self.win_reason == "captured_good":
                return 80.0   # 善玉取り勝利
            else:
                return 90.0   # その他勝利
        else:
            if self.win_reason == "captured_bad":
                return -100.0  # 悪玉取られ負け
            else:
                return -80.0   # その他負け
    
    def display_board(self):
        """ボード表示（デバッグ用）"""
        print(f"Move {self.move_count}, Player {self.current_player}'s turn")
        print("  0 1 2 3")
        for r in range(self.board_size):
            row = f"{r} "
            for c in range(self.board_size):
                if (r, c) in self.escape_positions['A']:
                    row += "A "
                elif (r, c) in self.escape_positions['B']:
                    row += "B "
                elif self.board[r, c] == 0:
                    row += ". "
                else:
                    row += f"{self.board[r, c]} "
            print(row)
        print()


class QuantumQNetwork(nn.Module):
    """軽量量子Q学習ネットワーク"""
    
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        try:
            # 量子デバイス
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except:
            # lightning.qubitが使えない場合はdefault.qubitを使用
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 古典前処理層
        self.state_encoder = nn.Sequential(
            nn.Linear(64, 32),  # 4x4x4 = 64次元
            nn.ReLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh()  # [-1, 1]に正規化
        )
        
        # 量子回路パラメータ
        self.q_params = nn.Parameter(torch.randn(n_layers * n_qubits) * 0.1)
        
        # 古典後処理層
        self.q_value_head = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # 4x4盤面のQ値
            nn.Tanh()
        )
        
        # 量子回路定義
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def quantum_circuit(inputs, weights):
            # 状態エンコーディング
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
                
                # Rotation layer
                for i in range(self.n_qubits):
                    param_idx = layer * self.n_qubits + i
                    qml.RY(weights[param_idx], wires=i)
            
            # 測定
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
    
    def forward(self, state):
        """前向き伝播"""
        batch_size = state.shape[0]
        
        # 状態を平坦化
        state_flat = state.view(batch_size, -1)
        
        # 古典エンコーディング
        encoded = self.state_encoder(state_flat)
        
        # 量子回路実行（バッチ処理）
        quantum_outputs = []
        for i in range(batch_size):
            q_out = self.quantum_circuit(encoded[i], self.q_params)
            quantum_outputs.append(torch.stack(q_out))
        
        quantum_features = torch.stack(quantum_outputs)
        
        # Q値計算
        q_values = self.q_value_head(quantum_features)
        
        return q_values.view(batch_size, 4, 4)


class QuantumQAgent:
    """量子Q学習エージェント"""
    
    def __init__(self, player_id, learning_rate=0.001, epsilon=0.9, 
                 epsilon_decay=0.995, epsilon_min=0.1, memory_size=2000):
        self.player_id = player_id
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        # Q-Network
        self.q_network = QuantumQNetwork()
        self.target_network = QuantumQNetwork()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience Replay
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = 32
        
        # 学習統計
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'losses': [],
            'epsilon_values': []
        }
    
    def get_action(self, game_state, possible_actions, training=True):
        """行動選択"""
        if training and random.random() < self.epsilon:
            return random.choice(possible_actions) if possible_actions else None
        
        if not possible_actions:
            return None
        
        # Q値計算
        state_tensor = game_state.get_state_tensor(self.player_id).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)
        
        # 可能な行動のQ値のみ考慮
        best_action = None
        best_q_value = float('-inf')
        
        for action in possible_actions:
            from_pos, to_pos = action
            
            # 盤外移動（脱出）の場合は特別処理
            if not (0 <= to_pos[0] < 4 and 0 <= to_pos[1] < 4):
                # 脱出行動は高い価値を与える
                q_val = 10.0
            else:
                q_val = q_values[to_pos[0], to_pos[1]].item()
            
            if q_val > best_q_value:
                best_q_value = q_val
                best_action = action
        
        return best_action
    
    def remember(self, state, action, reward, next_state, done):
        """経験を記憶"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self):
        """経験再生学習"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp[0] for exp in batch])
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool)
        
        current_q_values = self.q_network(states)
        
        with torch.no_grad():
            next_states = torch.stack([exp[3] for exp in batch])
            next_q_values = self.target_network(next_states)
            target_q_values = rewards + 0.95 * torch.max(next_q_values.view(self.batch_size, -1), dim=1)[0] * ~dones
        
        # 損失計算（簡略化）
        loss = nn.MSELoss()(current_q_values.view(self.batch_size, -1).max(dim=1)[0], target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_stats['losses'].append(loss.item())
        
        # ε減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_stats['epsilon_values'].append(self.epsilon)
    
    def update_target_network(self):
        """ターゲットネットワーク更新"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """モデル保存"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        """モデル読み込み"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_stats = checkpoint['training_stats']
        self.epsilon = checkpoint['epsilon']


# 基本エージェント
class RandomAgent:
    """ランダムエージェント"""
    
    def __init__(self, player_id):
        self.player_id = player_id
    
    def get_action(self, game_state, possible_actions, training=False):
        return random.choice(possible_actions) if possible_actions else None


class SmartRandomAgent:
    """賢いランダムエージェント（明らかに悪い手を避ける）"""
    
    def __init__(self, player_id):
        self.player_id = player_id
    
    def get_action(self, game_state, possible_actions, training=False):
        if not possible_actions:
            return None
        
        # 脱出可能な手があれば優先
        escape_actions = []
        for action in possible_actions:
            from_pos, to_pos = action
            if not (0 <= to_pos[0] < 4 and 0 <= to_pos[1] < 4):
                escape_actions.append(action)
        
        if escape_actions:
            return random.choice(escape_actions)
        
        return random.choice(possible_actions)


# 簡易対戦システム
class SimpleTournament:
    """簡易対戦システム"""
    
    def __init__(self):
        self.agents = {}
        self.results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
    
    def register_agent(self, name, agent):
        """エージェント登録"""
        self.agents[name] = agent
        print(f"✅ エージェント '{name}' 登録")
    
    def play_match(self, agent_a_name, agent_b_name, verbose=False):
        """単一試合"""
        game = Mini2GeisterGame()
        agent_a = self.agents[agent_a_name]
        agent_b = self.agents[agent_b_name]
        
        max_moves = 50
        move_count = 0
        
        while not game.is_terminal() and move_count < max_moves:
            current_agent_name = agent_a_name if game.current_player == 'A' else agent_b_name
            current_agent = agent_a if game.current_player == 'A' else agent_b
            
            possible_actions = game.get_possible_actions()
            if not possible_actions:
                break
            
            action = current_agent.get_action(game, possible_actions, training=False)
            if action is None:
                break
            
            if not game.make_move(action[0], action[1]):
                break
                
            move_count += 1
            
            if verbose and move_count % 10 == 0:
                game.display_board()
        
        # 結果記録
        if game.is_terminal():
            winner_name = agent_a_name if game.winner == 'A' else agent_b_name
            loser_name = agent_b_name if game.winner == 'A' else agent_a_name
            
            self.results[winner_name]['wins'] += 1
            self.results[loser_name]['losses'] += 1
            
            return winner_name, game.win_reason, move_count
        else:
            self.results[agent_a_name]['draws'] += 1
            self.results[agent_b_name]['draws'] += 1
            return "Draw", "timeout", move_count
    
    def run_tournament(self, episodes_per_pair=50):
        """トーナメント実行"""
        agent_names = list(self.agents.keys())
        print(f"🏆 トーナメント開始: {len(agent_names)}エージェント")
        
        for i, agent_a in enumerate(agent_names):
            for j, agent_b in enumerate(agent_names):
                if i >= j:  # 重複を避ける
                    continue
                
                print(f"⚔️ {agent_a} vs {agent_b}")
                wins_a = 0
                
                for episode in range(episodes_per_pair):
                    winner, reason, moves = self.play_match(agent_a, agent_b)
                    if winner == agent_a:
                        wins_a += 1
                
                winrate_a = wins_a / episodes_per_pair
                print(f"   {agent_a}: {winrate_a:.1%} ({wins_a}/{episodes_per_pair})")
        
        self.print_rankings()
    
    def print_rankings(self):
        """ランキング表示"""
        print("\n🏆 ランキング:")
        print("="*50)
        
        for agent_name, stats in self.results.items():
            total = stats['wins'] + stats['losses'] + stats['draws']
            winrate = stats['wins'] / total if total > 0 else 0
            print(f"{agent_name:15s} | 勝率: {winrate:6.1%} | "
                  f"{stats['wins']:3d}勝 {stats['losses']:3d}敗 {stats['draws']:3d}分")


def load_config():
    """設定読み込み"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    default_config = {
        "game_settings": {"board_size": 4, "pieces_per_player": 2, "max_moves": 50},
        "quantum_settings": {"n_qubits": 4, "n_layers": 2, "learning_rate": 0.001},
        "training_settings": {"episodes": 500, "batch_size": 32},
        "evaluation_settings": {"tournament_episodes": 50}
    }
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default_config


def train_quantum_agent(episodes=500):
    """量子エージェント訓練"""
    print(f"🚀 量子エージェント訓練開始: {episodes}エピソード")
    
    quantum_agent = QuantumQAgent('A', epsilon=0.9)
    opponent = SmartRandomAgent('B')
    
    wins = 0
    
    for episode in range(episodes):
        game = Mini2GeisterGame()
        states = []
        actions = []
        
        while not game.is_terminal():
            if game.current_player == 'A':  # 量子エージェント
                state = game.get_state_tensor('A')
                possible_actions = game.get_possible_actions()
                
                if not possible_actions:
                    break
                
                action = quantum_agent.get_action(game, possible_actions, training=True)
                states.append(state)
                actions.append(action)
            else:  # 相手エージェント
                possible_actions = game.get_possible_actions()
                if possible_actions:
                    action = opponent.get_action(game, possible_actions)
                else:
                    break
            
            if action:
                game.make_move(action[0], action[1])
        
        # 学習更新
        final_reward = game.get_reward('A')
        for i, (state, action) in enumerate(zip(states, actions)):
            reward = final_reward if i == len(states) - 1 else 0
            next_state = states[i + 1] if i + 1 < len(states) else state
            done = (i == len(states) - 1)
            quantum_agent.remember(state, action, reward, next_state, done)
        
        if len(quantum_agent.memory) > quantum_agent.batch_size:
            quantum_agent.replay()
        
        if game.winner == 'A':
            wins += 1
        
        if episode % 100 == 0:
            winrate = wins / (episode + 1)
            print(f"Episode {episode:4d}/{episodes} | 勝率: {winrate:.1%} | ε: {quantum_agent.epsilon:.3f}")
        
        if episode % 100 == 0:
            quantum_agent.update_target_network()
    
    final_winrate = wins / episodes
    print(f"✅ 訓練完了! 最終勝率: {final_winrate:.1%}")
    
    return quantum_agent


def quick_start():
    """クイックスタート"""
    print("🚀 Mini2ガイスター クイックスタート")
    
    # エージェント作成
    tournament = SimpleTournament()
    tournament.register_agent("Quantum", QuantumQAgent('A', epsilon=0.1))
    tournament.register_agent("Random", RandomAgent('A'))
    tournament.register_agent("SmartRandom", SmartRandomAgent('A'))
    
    # クイック評価（量子 vs ランダム）
    print("\n🎯 クイック評価: 量子 vs ランダム (20試合)")
    wins = 0
    for i in range(20):
        winner, reason, moves = tournament.play_match("Quantum", "Random")
        if winner == "Quantum":
            wins += 1
    
    winrate = wins / 20
    print(f"📊 量子エージェント勝率: {winrate:.1%} ({wins}/20)")
    
    if winrate > 0.6:
        print("🎉 量子エージェントは良好な性能を示しています！")
    else:
        print("⚠️ 量子エージェントの訓練が必要かもしれません")


def interactive_mode():
    """対話モード"""
    config = load_config()
    
    print("🎮 Mini2ガイスター対話モード")
    print("="*40)
    
    tournament = SimpleTournament()
    
    while True:
        print("\n選択してください:")
        print("1. クイック評価")
        print("2. 量子エージェント訓練")
        print("3. トーナメント実行")
        print("4. 1対1観戦")
        print("0. 終了")
        
        choice = input(">>> ").strip()
        
        if choice == "1":
            quick_start()
        elif choice == "2":
            episodes = int(input("訓練エピソード数 (デフォルト500): ") or "500")
            trained_agent = train_quantum_agent(episodes)
            tournament.register_agent("TrainedQuantum", trained_agent)
            print("✅ 訓練済みエージェントを登録しました")
        elif choice == "3":
            # 基本エージェントを登録
            if "Random" not in tournament.agents:
                tournament.register_agent("Random", RandomAgent('A'))
                tournament.register_agent("SmartRandom", SmartRandomAgent('A'))
                tournament.register_agent("Quantum", QuantumQAgent('A', epsilon=0.1))
            
            episodes = int(input("対戦数/ペア (デフォルト30): ") or "30")
            tournament.run_tournament(episodes)
        elif choice == "4":
            if len(tournament.agents) < 2:
                tournament.register_agent("Random", RandomAgent('A'))
                tournament.register_agent("SmartRandom", SmartRandomAgent('A'))
            
            agents = list(tournament.agents.keys())
            print(f"利用可能エージェント: {', '.join(agents)}")
            
            agent_a = input("エージェントA: ").strip()
            agent_b = input("エージェントB: ").strip()
            
            if agent_a in agents and agent_b in agents:
                print(f"\n⚔️ {agent_a} vs {agent_b}")
                winner, reason, moves = tournament.play_match(agent_a, agent_b, verbose=True)
                print(f"🏆 勝者: {winner} ({reason}) - {moves}手")
            else:
                print("❌ 無効なエージェント名")
        elif choice == "0":
            print("👋 終了")
            break
        else:
            print("❌ 無効な選択")


def main():
    """メイン関数"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_start()
        elif sys.argv[1] == "train":
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 500
            train_quantum_agent(episodes)
        elif sys.argv[1] == "interactive":
            interactive_mode()
        else:
            print("使用法: python mini2_integrated_system.py [quick|train|interactive]")
    else:
        # デフォルトは対話モード
        interactive_mode()


if __name__ == "__main__":
    print("🎮 Mini2ガイスター競技システム")
    print("="*40)
    main()
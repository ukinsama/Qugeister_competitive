"""
量子強化学習エージェント実装
現在のai_base.pyのBaseAIクラスと完全互換
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple, Optional
import os

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("⚠️  PennyLane not available. Install with: pip install pennylane")

from qugeister_competitive.ai_base import BaseAI
from qugeister_competitive.game_engine import GameState

class QuantumAgent(BaseAI):
    """量子強化学習エージェント"""
    
    def __init__(self, player_id: str, n_qubits=6, n_layers=3, learning_rate=0.01):
        super().__init__("QuantumAI", player_id)
        
        # 量子回路パラメータ
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
        # 量子デバイス設定
        if PENNYLANE_AVAILABLE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._setup_quantum_circuit()
        else:
            # フォールバック: 古典ニューラルネットワーク
            print("🔄 PennyLane未対応：古典ニューラルネットワークで代替")
            self._setup_classical_fallback()
        
        # 強化学習パラメータ
        self.epsilon = 0.9  # 探索率（高めからスタート）
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # 割引率
        self.batch_size = 32
        
        # 経験リプレイ
        self.memory = deque(maxlen=2000)
        self.training_step = 0
        
        # パフォーマンス記録
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon_values': []
        }
    
    def _setup_quantum_circuit(self):
        """量子回路セットアップ"""
        # 変分パラメータ
        self.params = torch.randn(self.n_layers, self.n_qubits, 3, requires_grad=True)
        
        # 古典前処理
        self.input_encoder = nn.Linear(36, self.n_qubits)  # 6x6=36
        self.output_decoder = nn.Linear(self.n_qubits, 4)  # 4方向の価値
        
        # 量子回路定義
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def quantum_circuit(inputs, params):
            # 状態エンコーディング
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # 変分レイヤー
            for layer in range(self.n_layers):
                # 回転ゲート
                for i in range(self.n_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)
                
                # エンタングルメント
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])  # 循環結合
            
            # 測定
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # オプティマイザー
        self.optimizer = optim.Adam([self.params] + 
                                   list(self.input_encoder.parameters()) + 
                                   list(self.output_decoder.parameters()), 
                                   lr=self.learning_rate)
    
    def _setup_classical_fallback(self):
        """古典ニューラルネットワークフォールバック"""
        self.network = nn.Sequential(
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.quantum_circuit = None
    
    def encode_game_state(self, game_state: GameState) -> torch.Tensor:
        """ゲーム状態をテンソルにエンコード"""
        # ボード状態を数値化
        board = game_state.board.copy()
        
        # プレイヤー視点で正規化
        if self.player_id == "B":
            board = -board  # B視点では符号反転
            board = np.flipud(board)  # 上下反転
        
        # フラット化して正規化
        state_vector = board.flatten().astype(np.float32)
        state_vector = state_vector / 2.0  # [-1, 1]に正規化
        
        return torch.tensor(state_vector, dtype=torch.float32)
    
    def get_q_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Q値を計算"""
        if self.quantum_circuit is not None:
            # 量子回路使用
            encoded_state = torch.tanh(self.input_encoder(state_tensor))
            quantum_output = self.quantum_circuit(encoded_state, self.params)
            q_values = self.output_decoder(torch.stack(quantum_output))
        else:
            # 古典ネットワーク使用
            q_values = self.network(state_tensor)
        
        return q_values
    
    def get_move(self, game_state: GameState, legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """手を選択（BaseAIインターフェース準拠）"""
        if not legal_moves:
            return None
        
        # ε-greedy探索
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Q値に基づく行動選択
        state_tensor = self.encode_game_state(game_state)
        q_values = self.get_q_values(state_tensor)
        
        # 合法手に対応する行動を評価
        best_move = self._select_best_legal_move(q_values, legal_moves, game_state)
        return best_move
    
    def _select_best_legal_move(self, q_values: torch.Tensor, legal_moves: List, game_state: GameState) -> Tuple:
        """Q値と合法手から最適手を選択"""
        move_scores = []
        
        for move in legal_moves:
            from_pos, to_pos = move
            
            # 移動方向を判定
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            # 方向インデックス (上右下左)
            if dy == -1:    direction = 0  # 上
            elif dx == 1:   direction = 1  # 右
            elif dy == 1:   direction = 2  # 下
            elif dx == -1:  direction = 3  # 左
            else:           direction = 0  # デフォルト
            
            # プレイヤーB視点では方向調整
            if self.player_id == "B":
                direction = (direction + 2) % 4  # 上下反転
            
            score = q_values[direction].item()
            
            # 戦術的ボーナス
            score += self._calculate_tactical_bonus(move, game_state)
            
            move_scores.append((move, score))
        
        # 最高スコアの手を選択
        best_move = max(move_scores, key=lambda x: x[1])[0]
        return best_move
    
    def _calculate_tactical_bonus(self, move: Tuple, game_state: GameState) -> float:
        """戦術的ボーナス計算"""
        from_pos, to_pos = move
        bonus = 0.0
        
        # 前進ボーナス
        if self.player_id == "A" and to_pos[1] > from_pos[1]:
            bonus += 0.1
        elif self.player_id == "B" and to_pos[1] < from_pos[1]:
            bonus += 0.1
        
        # 中央寄りボーナス
        center_distance = abs(to_pos[0] - 2.5)
        bonus += (2.5 - center_distance) * 0.02
        
        # 相手駒取りボーナス
        opponent_pieces = (game_state.player_b_pieces if self.player_id == "A" 
                          else game_state.player_a_pieces)
        if to_pos in opponent_pieces:
            bonus += 0.5
        
        return bonus
    
    def remember(self, state, action, reward, next_state, done):
        """経験を記憶"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay_training(self):
        """経験リプレイ学習"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # バッチサンプリング
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch])
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool)
        
        # 現在のQ値
        current_q_values = []
        for i, state in enumerate(states):
            q_vals = self.get_q_values(state)
            current_q_values.append(q_vals[actions[i]])
        current_q_values = torch.stack(current_q_values)
        
        # ターゲットQ値
        target_q_values = []
        for i, (reward, next_state, done) in enumerate(zip(rewards, next_states, dones)):
            if done:
                target = reward
            else:
                next_q_vals = self.get_q_values(next_state)
                target = reward + self.gamma * torch.max(next_q_vals)
            target_q_values.append(target)
        target_q_values = torch.stack(target_q_values)
        
        # 損失計算
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        # パラメータ更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ε値減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        return loss.item()
    
    def save_model(self, filepath: str):
        """モデル保存"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_type': 'quantum' if self.quantum_circuit else 'classical',
            'player_id': self.player_id,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'training_history': self.training_history
        }
        
        if self.quantum_circuit:
            save_dict.update({
                'quantum_params': self.params,
                'input_encoder_state': self.input_encoder.state_dict(),
                'output_decoder_state': self.output_decoder.state_dict()
            })
        else:
            save_dict['network_state'] = self.network.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"💾 量子エージェントモデル保存: {filepath}")
    
    def load_model(self, filepath: str):
        """モデル読み込み"""
        if not os.path.exists(filepath):
            print(f"❌ モデルファイルが見つかりません: {filepath}")
            return False
        
        checkpoint = torch.load(filepath)
        
        self.training_step = checkpoint.get('training_step', 0)
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.training_history = checkpoint.get('training_history', {})
        
        if checkpoint['model_type'] == 'quantum' and self.quantum_circuit:
            self.params = checkpoint['quantum_params']
            self.input_encoder.load_state_dict(checkpoint['input_encoder_state'])
            self.output_decoder.load_state_dict(checkpoint['output_decoder_state'])
        elif checkpoint['model_type'] == 'classical' and not self.quantum_circuit:
            self.network.load_state_dict(checkpoint['network_state'])
        
        print(f"✅ 量子エージェントモデル読み込み: {filepath}")
        return True
    
    def get_model_info(self) -> dict:
        """モデル情報取得"""
        return {
            'name': self.name,
            'type': 'quantum' if self.quantum_circuit else 'classical',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': self.win_rate
        }
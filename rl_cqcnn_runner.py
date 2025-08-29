#!/usr/bin/env python3
"""
強化学習版CQCNN競技ランナー
自己対戦による学習機能を持つ完全版システム
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import deque
import random
import matplotlib.pyplot as plt


# ================================================================================
# Part 1: 基本定義と経験データ構造
# ================================================================================


class Experience(NamedTuple):
    """強化学習用の経験データ"""

    state: np.ndarray
    action: Tuple
    reward: float
    next_state: Optional[np.ndarray]
    done: bool

    # CQCNN特有の情報
    enemy_positions: List[Tuple[int, int]]
    estimations: Dict[Tuple[int, int], Dict[str, float]]
    q_map: np.ndarray


class RLConfig:
    """強化学習設定"""

    def __init__(self):
        # 学習パラメータ
        self.learning_rate = 0.001
        self.gamma = 0.95  # 割引率
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

        # バッファ設定
        self.buffer_size = 10000
        self.batch_size = 32
        self.update_target_every = 100

        # 報酬設定
        self.reward_correct_estimation = 1.0
        self.reward_wrong_estimation = -0.5
        self.reward_win = 10.0
        self.reward_lose = -10.0
        self.reward_draw = 0.0
        self.reward_capture = 2.0
        self.reward_move = -0.01


# ================================================================================
# Part 2: 量子回路層（強化学習対応版）
# ================================================================================


class RLQuantumCircuitLayer(nn.Module):
    """強化学習に最適化された量子回路層"""

    def __init__(self, n_qubits: int, n_layers: int, dropout_rate: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_rate)

        # 量子回路パラメータ（学習可能）
        self.rotation_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.entanglement_params = nn.Parameter(torch.randn(n_layers, n_qubits - 1) * 0.1)

        # バッチ正規化
        self.batch_norm = nn.BatchNorm1d(n_qubits)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        batch_size = x.shape[0]

        # 入力を量子ビット数に調整
        if x.shape[1] != self.n_qubits:
            fc = nn.Linear(x.shape[1], self.n_qubits).to(x.device)
            x = fc(x)

        # 量子状態を初期化
        quantum_state = torch.zeros(batch_size, self.n_qubits, 2, dtype=torch.complex64)
        quantum_state[:, :, 0] = 1.0

        # 入力エンコーディング（振幅エンコーディング）
        for i in range(self.n_qubits):
            angle = x[:, i].unsqueeze(1) * np.pi
            quantum_state[:, i, 0] = torch.cos(angle / 2).squeeze()
            quantum_state[:, i, 1] = torch.sin(angle / 2).squeeze()

        # パラメータ化量子回路
        for layer in range(self.n_layers):
            # 回転ゲート
            for i in range(self.n_qubits):
                rx = self.rotation_params[layer, i, 0]
                ry = self.rotation_params[layer, i, 1]
                rz = self.rotation_params[layer, i, 2]

                # Pauli回転を適用
                quantum_state[:, i] = self._apply_rotation(quantum_state[:, i], rx, ry, rz)

            # エンタングルメント
            for i in range(self.n_qubits - 1):
                strength = torch.sigmoid(self.entanglement_params[layer, i])
                quantum_state = self._apply_entanglement(quantum_state, i, i + 1, strength)

        # 測定（期待値）
        measurements = (quantum_state.abs() ** 2)[:, :, 1].real

        # バッチ正規化とドロップアウト
        if training:
            measurements = self.batch_norm(measurements)
            measurements = self.dropout(measurements)

        return measurements

    def _apply_rotation(self, state: torch.Tensor, rx: float, ry: float, rz: float) -> torch.Tensor:
        """Pauli回転を適用"""
        # RZ
        phase_z = torch.exp(1j * rz / 2)
        state[:, 0] *= phase_z
        state[:, 1] *= torch.conj(phase_z)

        # RY
        c_y = torch.cos(ry / 2)
        s_y = torch.sin(ry / 2)
        temp0 = c_y * state[:, 0] - s_y * state[:, 1]
        temp1 = s_y * state[:, 0] + c_y * state[:, 1]
        state[:, 0] = temp0
        state[:, 1] = temp1

        # RX
        c_x = torch.cos(rx / 2)
        s_x = torch.sin(rx / 2) * 1j
        temp0 = c_x * state[:, 0] - s_x * state[:, 1]
        temp1 = -s_x * state[:, 0] + c_x * state[:, 1]
        state[:, 0] = temp0
        state[:, 1] = temp1

        return state

    def _apply_entanglement(self, state: torch.Tensor, q1: int, q2: int, strength: float) -> torch.Tensor:
        """制御NOTゲート風のエンタングルメントを適用"""
        control = state[:, q1, 1].abs()
        state[:, q2] = (1 - strength) * state[:, q2] + strength * control.unsqueeze(1) * state[:, q2].roll(1, dims=1)
        return state


# ================================================================================
# Part 3: 強化学習版CQCNN推定器
# ================================================================================


class RLCQCNNEstimator(nn.Module):
    """強化学習用CQCNN敵駒推定器"""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()

        # パラメータを保存
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # CNN層（特徴抽出）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 量子回路層
        self.quantum_layer = RLQuantumCircuitLayer(n_qubits, n_layers)

        # 統合層
        self.fc1 = nn.Linear(64 * 2 * 2 + n_qubits, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)  # 6種類の駒

        # ドロップアウト
        self.dropout = nn.Dropout(0.2)

    def forward(self, board_state: torch.Tensor, position: Tuple[int, int], training: bool = True) -> torch.Tensor:
        """
        敵駒タイプを推定

        Args:
            board_state: ボード状態 (batch_size, 3, 6, 5)
            position: 推定対象の位置
            training: 学習モードかどうか

        Returns:
            駒タイプの確率分布 (batch_size, 6)
        """
        # 局所的な特徴を抽出
        local_features = self._extract_local_features(board_state, position)

        # CNN処理
        x = F.relu(self.conv1(local_features))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        cnn_features = x.flatten(start_dim=1)

        # 量子回路処理
        quantum_input = cnn_features[:, : self.n_qubits]
        quantum_features = self.quantum_layer(quantum_input, training)

        # 特徴統合
        combined = torch.cat([cnn_features, quantum_features], dim=1)

        # 最終予測
        x = F.relu(self.fc1(combined))
        if training:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if training:
            x = self.dropout(x)
        x = self.fc3(x)

        return F.softmax(x, dim=1)

    def _extract_local_features(self, board_state: torch.Tensor, position: Tuple[int, int]) -> torch.Tensor:
        """位置周辺の局所特徴を抽出"""
        x, y = position
        batch_size = board_state.shape[0]

        # 5x5の局所領域を切り出し
        local = torch.zeros(batch_size, 3, 5, 5)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 5 and 0 <= ny < 6:
                    local[:, :, dx + 2, dy + 2] = board_state[:, :, ny, nx]

        return local


# ================================================================================
# Part 4: DQN（Deep Q-Network）によるQ値学習
# ================================================================================


class DQNCQCNN(nn.Module):
    """DQNベースのQ値推定ネットワーク"""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()

        # パラメータを保存
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 特徴抽出部（CNN + 量子回路）
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)  # 4ch: board + estimations
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.quantum_layer = RLQuantumCircuitLayer(n_qubits, n_layers)

        # Q値計算部
        self.fc1 = nn.Linear(64 * 6 * 5 + n_qubits, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 240)  # 6x5x8 = 240 (全位置・全方向のQ値)

        self.dropout = nn.Dropout(0.2)

    def forward(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        状態からQ値マップを生成

        Args:
            state: 状態テンソル (batch_size, 4, 6, 5)
            training: 学習モードかどうか

        Returns:
            Q値マップ (batch_size, 240)
        """
        # CNN特徴抽出
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        cnn_features = x.flatten(start_dim=1)

        # 量子回路処理
        quantum_input = cnn_features[:, : self.n_qubits]
        quantum_features = self.quantum_layer(quantum_input, training)

        # 結合
        combined = torch.cat([cnn_features, quantum_features], dim=1)

        # Q値計算
        x = F.relu(self.fc1(combined))
        if training:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if training:
            x = self.dropout(x)
        q_values = self.fc3(x)

        return q_values


# ================================================================================
# Part 5: 経験リプレイバッファ
# ================================================================================


class ExperienceReplayBuffer:
    """優先度付き経験リプレイバッファ"""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # 優先度の重み
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def push(self, experience: Experience, priority: float = None):
        """経験を追加"""
        if priority is None:
            priority = max(self.priorities, default=1.0)

        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], torch.Tensor]:
        """優先度に基づいてサンプリング"""
        if len(self.buffer) < batch_size:
            return list(self.buffer), torch.ones(len(self.buffer))

        # 優先度を確率に変換
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        # サンプリング
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # 重要度サンプリング重み
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """優先度を更新"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# ================================================================================
# Part 6: 強化学習トレーナー
# ================================================================================


class RLCQCNNTrainer:
    """強化学習によるCQCNN学習システム"""

    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()

        # ネットワーク
        self.estimator = RLCQCNNEstimator()
        self.q_network = DQNCQCNN()
        self.target_q_network = DQNCQCNN()
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # オプティマイザ
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=self.config.learning_rate)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

        # リプレイバッファ
        self.replay_buffer = ExperienceReplayBuffer(self.config.buffer_size)

        # 学習状態
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self.episodes = 0

        # 統計情報
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "loss_history": [],
            "win_rate": deque(maxlen=100),
            "estimation_accuracy": deque(maxlen=100),
        }

    def select_action(self, state: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        """ε-greedy行動選択"""
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        # Q値に基づく選択
        with torch.no_grad():
            state_tensor = self._prepare_state_tensor(state)
            q_values = self.q_network(state_tensor, training=False)
            q_map = q_values.reshape(6, 5, 8)

            best_move = None
            best_q = -float("inf")

            for move in legal_moves:
                from_pos, to_pos = move[:2]
                dir_idx = self._get_direction_index(from_pos, to_pos)
                if dir_idx is not None:
                    q = q_map[from_pos[1], from_pos[0], dir_idx].item()
                    if q > best_q:
                        best_q = q
                        best_move = move

            return best_move if best_move else random.choice(legal_moves)

    def train_step(self, batch_size: int = None) -> float:
        """1ステップの学習"""
        if len(self.replay_buffer) < (batch_size or self.config.batch_size):
            return 0.0

        batch_size = batch_size or self.config.batch_size
        experiences, weights = self.replay_buffer.sample(batch_size)

        # バッチデータを準備
        states = torch.stack([self._prepare_state_tensor(e.state) for e in experiences])
        actions = torch.tensor([self._action_to_index(e.action) for e in experiences])
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.stack(
            [
                self._prepare_state_tensor(e.next_state) if e.next_state is not None else torch.zeros_like(states[0])
                for e in experiences
            ]
        )
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)

        # 現在のQ値
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 次の状態の最大Q値（ターゲットネットワーク使用）
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states, training=False).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

        # 損失計算（重要度サンプリング重みを適用）
        loss = (weights * F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction="none")).mean()

        # 最適化
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()

        # 優先度更新
        with torch.no_grad():
            td_errors = torch.abs(current_q_values.squeeze() - target_q_values)
            td_errors.cpu().numpy() + 1e-6
            # サンプリングしたインデックスを取得する必要がある
            # ここでは簡略化のため省略

        self.steps += 1

        # ターゲットネットワーク更新
        if self.steps % self.config.update_target_every == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train_estimator(self, experiences: List[Experience]) -> float:
        """推定器の学習"""
        if not experiences:
            return 0.0

        total_loss = 0.0

        for exp in experiences:
            if not exp.enemy_positions:
                continue

            # 真の駒タイプ（シミュレーションでは仮定）
            true_types = self._get_true_piece_types(exp)

            for pos in exp.enemy_positions:
                # 推定
                state_tensor = self._prepare_state_tensor(exp.state)
                estimation = self.estimator(state_tensor.unsqueeze(0), pos)

                # 損失計算
                target = torch.tensor(true_types[pos], dtype=torch.float32)
                loss = F.cross_entropy(estimation, target.unsqueeze(0))

                # 最適化
                self.estimator_optimizer.zero_grad()
                loss.backward()
                self.estimator_optimizer.step()

                total_loss += loss.item()

        return total_loss / max(len(experiences), 1)

    def episode_end(self, experiences: List[Experience], reward: float, won: bool):
        """エピソード終了処理"""
        # バッファに追加
        for exp in experiences:
            self.replay_buffer.push(exp)

        # 学習
        if len(self.replay_buffer) >= self.config.batch_size:
            for _ in range(min(10, len(experiences))):
                self.train_step()

        # 推定器学習
        self.train_estimator(experiences)

        # ε減衰
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        # 統計記録
        self.episodes += 1
        self.training_stats["episode_rewards"].append(reward)
        self.training_stats["episode_lengths"].append(len(experiences))
        self.training_stats["win_rate"].append(1.0 if won else 0.0)

    def _prepare_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """状態をテンソルに変換"""
        if isinstance(state, torch.Tensor):
            return state
        return torch.tensor(state, dtype=torch.float32)

    def _action_to_index(self, action: Tuple) -> int:
        """行動をインデックスに変換"""
        from_pos, to_pos = action[:2]
        dir_idx = self._get_direction_index(from_pos, to_pos)
        return from_pos[1] * 40 + from_pos[0] * 8 + (dir_idx or 0)

    def _get_direction_index(self, from_pos: Tuple, to_pos: Tuple) -> Optional[int]:
        """方向インデックスを取得"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        try:
            return directions.index((dx, dy))
        except ValueError:
            return None

    def _get_true_piece_types(self, experience: Experience) -> Dict:
        """真の駒タイプを取得（シミュレーション用）"""
        # 実際のゲームでは真のタイプが分かるが、
        # ここではランダムに生成
        types = {}
        piece_types = [0, 1, 2, 3, 4, 5]  # P, K, Q, R, B, N

        for pos in experience.enemy_positions:
            # 推定値に基づいて最も可能性の高いタイプを真値とする（簡略化）
            if pos in experience.estimations:
                probs = list(experience.estimations[pos].values())
                types[pos] = np.argmax(probs)
            else:
                types[pos] = random.choice(piece_types)

        return types

    def save_checkpoint(self, filepath: str):
        """チェックポイント保存"""
        torch.save(
            {
                "estimator_state": self.estimator.state_dict(),
                "q_network_state": self.q_network.state_dict(),
                "target_q_network_state": self.target_q_network.state_dict(),
                "estimator_optimizer": self.estimator_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "episodes": self.episodes,
                "training_stats": self.training_stats,
            },
            filepath,
        )
        print(f"💾 チェックポイント保存: {filepath}")

    def load_checkpoint(self, filepath: str):
        """チェックポイント読み込み"""
        checkpoint = torch.load(filepath)
        self.estimator.load_state_dict(checkpoint["estimator_state"])
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state"])
        self.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.training_stats = checkpoint["training_stats"]
        print(f"✅ チェックポイント読み込み: {filepath}")


# ================================================================================
# Part 7: 自己対戦システム
# ================================================================================


class SelfPlayAgent:
    """自己対戦用エージェント"""

    def __init__(self, trainer: RLCQCNNTrainer, player_id: str):
        self.trainer = trainer
        self.player_id = player_id
        self.experiences = []

    def get_move(self, state: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        """手を選択"""
        return self.trainer.select_action(state, legal_moves)

    def record_experience(
        self,
        state: np.ndarray,
        action: Tuple,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
        enemy_positions: List,
        estimations: Dict,
        q_map: np.ndarray,
    ):
        """経験を記録"""
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            enemy_positions=enemy_positions,
            estimations=estimations,
            q_map=q_map,
        )
        self.experiences.append(exp)


class SelfPlaySystem:
    """自己対戦による学習システム"""

    def __init__(self, trainer: RLCQCNNTrainer):
        self.trainer = trainer
        self.board_size = (6, 5)
        self.max_turns = 100

    def play_episode(self) -> Dict:
        """1エピソードの自己対戦"""
        # エージェント作成
        agent1 = SelfPlayAgent(self.trainer, "A")
        agent2 = SelfPlayAgent(self.trainer, "B")

        # ゲーム状態初期化
        board = self._initialize_board()
        turn = 0

        # ゲームループ
        while turn < self.max_turns:
            turn += 1

            # プレイヤー1の手番
            legal_moves1 = self._get_legal_moves(board, "A")
            if legal_moves1:
                state1 = self._board_to_state(board, "A")
                move1 = agent1.get_move(state1, legal_moves1)

                # 報酬計算
                reward1 = self._calculate_reward(board, move1, "A")

                # ボード更新
                board = self._apply_move(board, move1, "A")

                # 経験記録
                next_state1 = self._board_to_state(board, "A")
                agent1.record_experience(
                    state1,
                    move1,
                    reward1,
                    next_state1,
                    False,
                    self._get_enemy_positions(board, "A"),
                    {},
                    np.zeros((6, 5, 8)),
                )

            # 勝敗判定
            winner = self._check_winner(board)
            if winner:
                break

            # プレイヤー2の手番（同様の処理）
            legal_moves2 = self._get_legal_moves(board, "B")
            if legal_moves2:
                state2 = self._board_to_state(board, "B")
                move2 = agent2.get_move(state2, legal_moves2)
                reward2 = self._calculate_reward(board, move2, "B")
                board = self._apply_move(board, move2, "B")
                next_state2 = self._board_to_state(board, "B")
                agent2.record_experience(
                    state2,
                    move2,
                    reward2,
                    next_state2,
                    False,
                    self._get_enemy_positions(board, "B"),
                    {},
                    np.zeros((6, 5, 8)),
                )

            winner = self._check_winner(board)
            if winner:
                break

        # エピソード終了処理
        if winner == "A":
            final_reward1 = self.trainer.config.reward_win
            final_reward2 = self.trainer.config.reward_lose
        elif winner == "B":
            final_reward1 = self.trainer.config.reward_lose
            final_reward2 = self.trainer.config.reward_win
        else:
            final_reward1 = self.trainer.config.reward_draw
            final_reward2 = self.trainer.config.reward_draw

        # 最終報酬を記録
        if agent1.experiences:
            agent1.experiences[-1] = agent1.experiences[-1]._replace(
                reward=agent1.experiences[-1].reward + final_reward1, done=True
            )
        if agent2.experiences:
            agent2.experiences[-1] = agent2.experiences[-1]._replace(
                reward=agent2.experiences[-1].reward + final_reward2, done=True
            )

        # 学習
        all_experiences = agent1.experiences + agent2.experiences
        total_reward = sum(exp.reward for exp in all_experiences)

        self.trainer.episode_end(all_experiences, total_reward, winner == "A")

        return {"winner": winner, "turns": turn, "total_reward": total_reward, "experiences": len(all_experiences)}

    def _initialize_board(self) -> np.ndarray:
        """ボード初期化"""
        board = np.zeros(self.board_size)
        # 簡略化のため、ランダムに駒を配置
        for i in range(5):
            board[0, i] = 1  # プレイヤーA
            board[5, i] = -1  # プレイヤーB
        return board

    def _board_to_state(self, board: np.ndarray, player: str) -> np.ndarray:
        """ボードを状態テンソルに変換"""
        state = np.zeros((4, 6, 5))
        player_val = 1 if player == "A" else -1

        # チャンネル0: 自分の駒
        state[0] = (board == player_val).astype(float)
        # チャンネル1: 敵の駒
        state[1] = (board == -player_val).astype(float)
        # チャンネル2: 空きマス
        state[2] = (board == 0).astype(float)
        # チャンネル3: ボード全体
        state[3] = board / 2 + 0.5

        return state

    def _get_legal_moves(self, board: np.ndarray, player: str) -> List[Tuple]:
        """合法手を取得"""
        moves = []
        player_val = 1 if player == "A" else -1

        for y in range(6):
            for x in range(5):
                if board[y, x] == player_val:
                    # 8方向の移動を確認
                    for dx, dy in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 5 and 0 <= ny < 6:
                            if board[ny, nx] != player_val:
                                moves.append(((x, y), (nx, ny)))

        return moves

    def _apply_move(self, board: np.ndarray, move: Tuple, player: str) -> np.ndarray:
        """手を適用"""
        new_board = board.copy()
        from_pos, to_pos = move[:2]
        player_val = 1 if player == "A" else -1

        new_board[from_pos[1], from_pos[0]] = 0
        new_board[to_pos[1], to_pos[0]] = player_val

        return new_board

    def _calculate_reward(self, board: np.ndarray, move: Tuple, player: str) -> float:
        """報酬を計算"""
        reward = self.trainer.config.reward_move

        from_pos, to_pos = move[:2]
        player_val = 1 if player == "A" else -1

        # 敵駒を取った場合
        if board[to_pos[1], to_pos[0]] == -player_val:
            reward += self.trainer.config.reward_capture

        # 中央への移動を評価
        center_x, center_y = 2, 3
        dist_before = abs(from_pos[0] - center_x) + abs(from_pos[1] - center_y)
        dist_after = abs(to_pos[0] - center_x) + abs(to_pos[1] - center_y)
        reward += (dist_before - dist_after) * 0.1

        return reward

    def _check_winner(self, board: np.ndarray) -> Optional[str]:
        """勝者を判定"""
        # 簡略化: 片方の駒が全てなくなったら負け
        has_a = np.any(board == 1)
        has_b = np.any(board == -1)

        if not has_b:
            return "A"
        elif not has_a:
            return "B"
        else:
            return None

    def _get_enemy_positions(self, board: np.ndarray, player: str) -> List[Tuple]:
        """敵駒の位置を取得"""
        enemy_val = -1 if player == "A" else 1
        positions = []

        for y in range(6):
            for x in range(5):
                if board[y, x] == enemy_val:
                    positions.append((x, y))

        return positions


# ================================================================================
# Part 8: 学習実行システム
# ================================================================================


class RLCQCNNRunner:
    """強化学習版CQCNN実行システム"""

    def __init__(self):
        self.config = RLConfig()
        self.trainer = RLCQCNNTrainer(self.config)
        self.self_play = SelfPlaySystem(self.trainer)

    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """学習を実行"""
        print("🚀 強化学習開始")
        print("=" * 70)
        print(f"エピソード数: {num_episodes}")
        print(f"保存間隔: {save_interval}エピソード")
        print(f"初期ε: {self.config.epsilon_start}")
        print(f"最終ε: {self.config.epsilon_end}")
        print("=" * 70)

        for episode in range(num_episodes):
            # 自己対戦
            self.self_play.play_episode()

            # 進捗表示
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.trainer.training_stats["episode_rewards"][-10:])
                avg_length = np.mean(self.trainer.training_stats["episode_lengths"][-10:])
                win_rate = np.mean(list(self.trainer.training_stats["win_rate"]))

                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {avg_reward:.2f} | "
                    f"Length: {avg_length:.1f} | "
                    f"Win Rate: {win_rate:.2%} | "
                    f"ε: {self.trainer.epsilon:.3f}"
                )

            # 定期保存
            if (episode + 1) % save_interval == 0:
                self.trainer.save_checkpoint(f"rl_cqcnn_ep{episode + 1}.pth")

        print("\n✅ 学習完了！")

        # 最終保存
        self.trainer.save_checkpoint("rl_cqcnn_final.pth")

        # 学習曲線をプロット
        self.plot_training_curves()

    def plot_training_curves(self):
        """学習曲線をプロット"""
        stats = self.trainer.training_stats

        if not stats["episode_rewards"]:
            print("⚠️ プロット用のデータがありません")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 報酬推移
        axes[0, 0].plot(stats["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")

        # エピソード長
        axes[0, 1].plot(stats["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Number of Steps")

        # 勝率（移動平均）
        win_rates = list(stats["win_rate"])
        if win_rates:
            axes[1, 0].plot(win_rates)
            axes[1, 0].set_title("Win Rate (Moving Average)")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Win Rate")

        # 損失推移
        if stats["loss_history"]:
            axes[1, 1].plot(stats["loss_history"])
            axes[1, 1].set_title("Training Loss")
            axes[1, 1].set_xlabel("Training Step")
            axes[1, 1].set_ylabel("Loss")

        plt.tight_layout()
        plt.savefig("rl_training_curves.png")
        print("📊 学習曲線を保存: rl_training_curves.png")

    def evaluate(self, num_games: int = 100) -> Dict:
        """学習済みモデルを評価"""
        print(f"\n📊 モデル評価 ({num_games}ゲーム)")

        wins = 0
        total_rewards = []

        # 評価モードに設定（εを0に）
        original_epsilon = self.trainer.epsilon
        self.trainer.epsilon = 0.0

        for game in range(num_games):
            result = self.self_play.play_episode()

            if result["winner"] == "A":
                wins += 1
            total_rewards.append(result["total_reward"])

            if (game + 1) % 20 == 0:
                print(f"  評価進捗: {game + 1}/{num_games}")

        # εを元に戻す
        self.trainer.epsilon = original_epsilon

        results = {
            "win_rate": wins / num_games,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "max_reward": np.max(total_rewards),
            "min_reward": np.min(total_rewards),
        }

        print("\n評価結果:")
        print(f"  勝率: {results['win_rate']:.2%}")
        print(f"  平均報酬: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  最大報酬: {results['max_reward']:.2f}")
        print(f"  最小報酬: {results['min_reward']:.2f}")

        return results


# ================================================================================
# メイン実行
# ================================================================================


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("🤖 強化学習版CQCNN競技システム")
    print("=" * 70)

    runner = RLCQCNNRunner()

    while True:
        print("\n【メニュー】")
        print("1. 新規学習開始")
        print("2. 学習再開（チェックポイントから）")
        print("3. モデル評価")
        print("4. デモ対戦")
        print("0. 終了")

        choice = input("\n選択 (0-4): ").strip()

        if choice == "0":
            break

        elif choice == "1":
            episodes = int(input("学習エピソード数 (100-10000): ") or "1000")
            runner.train(num_episodes=episodes)

        elif choice == "2":
            checkpoint = input("チェックポイントファイル名: ").strip()
            try:
                runner.trainer.load_checkpoint(checkpoint)
                episodes = int(input("追加学習エピソード数: ") or "500")
                runner.train(num_episodes=episodes)
            except FileNotFoundError:
                print(f"❌ ファイルが見つかりません: {checkpoint}")

        elif choice == "3":
            games = int(input("評価ゲーム数 (10-1000): ") or "100")
            runner.evaluate(num_games=games)

        elif choice == "4":
            print("\n🎮 デモ対戦")
            result = runner.self_play.play_episode()
            print(f"勝者: {result['winner'] or '引き分け'}")
            print(f"ターン数: {result['turns']}")
            print(f"総報酬: {result['total_reward']:.2f}")

    print("\n👋 終了します")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 中断されました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback

        traceback.print_exc()

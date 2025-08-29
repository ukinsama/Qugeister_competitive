#!/usr/bin/env python3
"""
強化学習版CQCNN駒推定システム
自己対戦による学習と教師あり学習版との比較実験
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, List, Optional
import random
from collections import deque
import time
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src", "qugeister_competitive")
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

# 既存モジュールのインポート
from separated_cqcnn_qmap import CQCNNPieceEstimator, QValueMapGenerator, IntegratedCQCNNSystem


# ================================================================================
# Part 1: 強化学習用の報酬システム
# ================================================================================


class RLRewardSystem:
    """強化学習用の報酬計算システム"""

    def __init__(self):
        self.weights = {
            # 基本報酬
            "win_game": 10.0,
            "lose_game": -10.0,
            "draw_game": 0.0,
            # 駒推定関連
            "correct_good_estimation": 2.0,  # 善玉を正しく推定
            "correct_bad_estimation": 1.5,  # 悪玉を正しく推定
            "wrong_estimation": -1.0,  # 間違った推定
            # 行動結果
            "capture_good": 5.0,  # 善玉捕獲
            "capture_bad": -3.0,  # 悪玉捕獲
            "good_escape": 8.0,  # 善玉脱出
            "lost_good": -4.0,  # 善玉喪失
            # 戦略的要素
            "forward_progress": 0.2,  # 前進
            "control_center": 0.1,  # 中央制御
            "piece_protection": 0.3,  # 駒の保護
        }

        self.episode_rewards = []
        self.estimation_accuracy_history = []

    def calculate_step_reward(self, old_state: Dict, action: Tuple, new_state: Dict, estimation: Dict) -> float:
        """1ステップの報酬を計算"""
        reward = 0.0

        # 移動の基本評価
        from_pos, to_pos = action

        # 前進報酬
        if old_state["current_player"] == "A":
            if to_pos[1] > from_pos[1]:
                reward += self.weights["forward_progress"]
        else:
            if to_pos[1] < from_pos[1]:
                reward += self.weights["forward_progress"]

        # 駒取りの評価
        if to_pos in new_state.get("captured_pieces", {}):
            captured_type = new_state["captured_pieces"][to_pos]
            if captured_type == "good":
                reward += self.weights["capture_good"]
            else:
                reward += self.weights["capture_bad"]

        # 中央制御
        center_dist = abs(to_pos[0] - 2.5) + abs(to_pos[1] - 2.5)
        reward += self.weights["control_center"] * (5 - center_dist) / 5

        return reward

    def calculate_episode_reward(self, winner: str, player: str, final_estimations: List[Dict]) -> float:
        """エピソード終了時の報酬を計算"""
        reward = 0.0

        # 勝敗報酬
        if winner == player:
            reward += self.weights["win_game"]
        elif winner == "Draw":
            reward += self.weights["draw_game"]
        else:
            reward += self.weights["lose_game"]

        # 推定精度ボーナス
        if final_estimations:
            correct_estimations = sum(1 for e in final_estimations if e.get("correct", False))
            accuracy = correct_estimations / len(final_estimations)
            reward += accuracy * 3.0
            self.estimation_accuracy_history.append(accuracy)

        self.episode_rewards.append(reward)
        return reward

    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        return {
            "avg_episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "avg_estimation_accuracy": np.mean(self.estimation_accuracy_history[-100:])
            if self.estimation_accuracy_history
            else 0,
            "total_episodes": len(self.episode_rewards),
        }


# ================================================================================
# Part 2: 経験リプレイバッファ
# ================================================================================


@dataclass
class Experience:
    """経験データ"""

    state: np.ndarray
    enemy_positions: List[Tuple[int, int]]
    true_enemy_types: Dict[Tuple[int, int], str]
    estimations: Dict
    reward: float
    next_state: Optional[np.ndarray]
    done: bool


class ExperienceReplayBuffer:
    """経験リプレイバッファ"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """経験を追加"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """バッチサンプリング"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ================================================================================
# Part 3: 強化学習版CQCNNトレーナー
# ================================================================================


class RLCQCNNTrainer:
    """強化学習によるCQCNN学習システム"""

    def __init__(
        self,
        estimator: CQCNNPieceEstimator,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        # モデルとオプティマイザ
        self.estimator = estimator
        self.target_estimator = CQCNNPieceEstimator(estimator.n_qubits, estimator.n_layers)
        self.target_estimator.load_state_dict(estimator.state_dict())

        self.optimizer = optim.Adam(estimator.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # ハイパーパラメータ
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 学習用コンポーネント
        self.replay_buffer = ExperienceReplayBuffer()
        self.reward_system = RLRewardSystem()

        # 学習履歴
        self.training_history = {
            "episodes": [],
            "rewards": [],
            "losses": [],
            "win_rates": [],
            "estimation_accuracy": [],
        }

        self.episodes_trained = 0
        self.update_target_every = 100

    def select_action(self, estimations: Dict, legal_moves: List, q_map: np.ndarray) -> Tuple:
        """ε-greedy行動選択"""
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        # Q値に基づく選択
        best_move = None
        best_q = -float("inf")

        for move in legal_moves:
            from_pos, to_pos = move
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]

            # 方向インデックス
            if dy == -1 and dx == 0:
                dir_idx = 0
            elif dx == 1 and dy == 0:
                dir_idx = 1
            elif dy == 1 and dx == 0:
                dir_idx = 2
            elif dx == -1 and dy == 0:
                dir_idx = 3
            else:
                continue

            q_value = q_map[from_pos[1], from_pos[0], dir_idx]

            if q_value > best_q:
                best_q = q_value
                best_move = move

        return best_move if best_move else random.choice(legal_moves)

    def train_step(self, batch_size: int = 32) -> float:
        """1ステップの学習"""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # バッチサンプリング
        batch = self.replay_buffer.sample(batch_size)

        total_loss = 0.0
        loss_count = 0

        for experience in batch:
            # ボード状態をテンソルに変換
            board_tensor = self._prepare_board_tensor(
                experience.state,
                "A",  # 仮のプレイヤー
            )

            # 敵駒位置ごとに学習
            for pos in experience.enemy_positions:
                if pos not in experience.true_enemy_types:
                    continue

                # 真のタイプ
                true_type = experience.true_enemy_types[pos]

                # 現在のネットワークで推定（勾配追跡あり）
                # CQCNNPieceEstimatorのforward処理を直接実行
                features = self.estimator.feature_conv(board_tensor)
                features_flat = features.view(features.size(0), -1)

                # 量子回路シミュレーション用の入力を準備
                # 簡易的な量子回路のため、インプレース操作を避ける
                n_qubits = self.estimator.n_qubits
                board_flat = board_tensor.view(board_tensor.size(0), -1)
                quantum_input = board_flat[:, :n_qubits].clone()  # clone()で新しいテンソルを作成

                # 量子回路シミュレーション（インプレース操作を避ける）
                batch_size_q = quantum_input.shape[0]
                q_state = torch.zeros(batch_size_q, n_qubits)

                for layer in range(self.estimator.n_layers):
                    new_q_state = torch.zeros_like(q_state)
                    for q in range(n_qubits):
                        rotation = self.estimator.quantum_params[layer, q]
                        # インプレース操作を避ける
                        new_q_state[:, q] = torch.sin(
                            rotation[0] * quantum_input[:, q % quantum_input.shape[1]]
                        ) * torch.cos(rotation[1] * quantum_input[:, (q + 1) % quantum_input.shape[1]]) + torch.tanh(
                            rotation[2] * q_state[:, q]
                        )
                    q_state = new_q_state  # 新しいテンソルに置き換え

                quantum_out = q_state

                # 結合
                combined = torch.cat([features_flat, quantum_out], dim=1)

                # 駒タイプ予測
                type_logits = self.estimator.piece_type_head(combined)
                type_probs = torch.softmax(type_logits, dim=1)

                # 目標値の計算
                if true_type == "good":
                    target = torch.tensor([[1.0, 0.0]], requires_grad=False)
                else:
                    target = torch.tensor([[0.0, 1.0]], requires_grad=False)

                # 報酬を反映
                reward_factor = 1.0 + experience.reward * 0.1
                target = target * reward_factor
                target = torch.clamp(target, 0.0, 1.0)

                # 損失計算
                loss = self.criterion(type_probs, target)

                # バックプロパゲーション
                self.optimizer.zero_grad()
                loss.backward()

                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), 1.0)

                self.optimizer.step()

                total_loss += loss.item()
                loss_count += 1

        return total_loss / max(loss_count, 1)

    def update_target_network(self):
        """ターゲットネットワークを更新"""
        self.target_estimator.load_state_dict(self.estimator.state_dict())

    def train_episode(self, game_func, verbose: bool = False) -> Dict:
        """1エピソードの学習（自己対戦）"""
        # ゲーム実行
        game_result = game_func(self, verbose=verbose)

        # 経験をバッファに追加
        for exp in game_result.get("experiences", []):
            self.replay_buffer.push(exp)

        # 学習実行
        losses = []
        for _ in range(10):  # エピソードごとに10回学習
            loss = self.train_step()
            if loss > 0:
                losses.append(loss)

        # εの減衰
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # ターゲットネットワークの更新
        self.episodes_trained += 1
        if self.episodes_trained % self.update_target_every == 0:
            self.update_target_network()

        # 履歴記録
        avg_loss = np.mean(losses) if losses else 0
        self.training_history["episodes"].append(self.episodes_trained)
        self.training_history["losses"].append(avg_loss)
        self.training_history["rewards"].append(game_result.get("total_reward", 0))

        return {
            "episode": self.episodes_trained,
            "reward": game_result.get("total_reward", 0),
            "loss": avg_loss,
            "epsilon": self.epsilon,
        }

    def _prepare_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        """ボード状態をテンソルに変換"""
        tensor = torch.zeros(1, 3, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val

        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))

        return tensor

    def save_checkpoint(self, filepath: str):
        """チェックポイントを保存"""
        torch.save(
            {
                "estimator_state": self.estimator.state_dict(),
                "target_state": self.target_estimator.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episodes": self.episodes_trained,
                "training_history": self.training_history,
            },
            filepath,
        )
        print(f"💾 チェックポイント保存: {filepath}")

    def load_checkpoint(self, filepath: str):
        """チェックポイントを読み込み"""
        # PyTorch 2.6対応: weights_only=Falseを明示的に指定
        checkpoint = torch.load(filepath, weights_only=False)
        self.estimator.load_state_dict(checkpoint["estimator_state"])
        self.target_estimator.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.episodes_trained = checkpoint["episodes"]
        self.training_history = checkpoint["training_history"]
        print(f"📂 チェックポイント読み込み: {filepath}")


# ================================================================================
# Part 4: 自己対戦システム
# ================================================================================


class SelfPlaySystem:
    """自己対戦による学習システム"""

    def __init__(self, rl_trainer: RLCQCNNTrainer):
        self.trainer = rl_trainer
        self.qmap_generator = QValueMapGenerator()
        self.game_history = []

    def play_episode(self, verbose: bool = False) -> Dict:
        """1エピソードの自己対戦"""
        # ダミーのゲーム実行（実際のゲームエンジンと統合が必要）
        experiences = []
        total_reward = 0.0

        # 仮のゲームループ
        for turn in range(50):
            # 仮のボード状態
            board = np.random.randint(-1, 2, (6, 6))
            enemy_positions = [(i, j) for i in range(6) for j in range(6) if board[j, i] == -1][:4]

            if not enemy_positions:
                break

            # 推定実行
            estimations = {}
            true_types = {}

            for pos in enemy_positions:
                # 仮の真のタイプ
                true_types[pos] = random.choice(["good", "bad"])

                # 推定（学習中のモデルを使用）
                board_tensor = self.trainer._prepare_board_tensor(board, "A")
                with torch.no_grad():
                    # 直接特徴抽出して推定
                    features = self.trainer.estimator.feature_conv(board_tensor)
                    features_flat = features.view(features.size(0), -1)

                    # quantum_circuit_simulationに正しい形状の入力を渡す
                    # n_qubits次元のベクトルに変換
                    n_qubits = self.trainer.estimator.n_qubits
                    if features_flat.shape[1] > n_qubits:
                        # Linear層で次元削減
                        linear = nn.Linear(features_flat.shape[1], n_qubits)
                        quantum_input = torch.tanh(linear(features_flat))
                    else:
                        # パディングして次元を合わせる
                        padding = n_qubits - features_flat.shape[1]
                        quantum_input = F.pad(features_flat, (0, padding), "constant", 0)

                    quantum_out = self.trainer.estimator.quantum_circuit_simulation(quantum_input)
                    combined = torch.cat([features_flat, quantum_out], dim=1)
                    type_logits = self.trainer.estimator.piece_type_head(combined)
                    type_probs = torch.softmax(type_logits, dim=1)

                    estimations[pos] = {
                        "good_prob": type_probs[0, 0].item(),
                        "bad_prob": type_probs[0, 1].item(),
                        "confidence": max(type_probs[0].tolist()),
                    }

            # 報酬計算
            step_reward = 0.0
            for pos, est in estimations.items():
                true_type = true_types[pos]
                if true_type == "good" and est["good_prob"] > 0.5:
                    step_reward += 1.0
                elif true_type == "bad" and est["bad_prob"] > 0.5:
                    step_reward += 1.0
                else:
                    step_reward -= 0.5

            # 経験を記録
            exp = Experience(
                state=board,
                enemy_positions=enemy_positions,
                true_enemy_types=true_types,
                estimations=estimations,
                reward=step_reward,
                next_state=None,
                done=False,
            )
            experiences.append(exp)
            total_reward += step_reward

            if verbose and turn % 10 == 0:
                print(f"  Turn {turn}: Reward = {step_reward:.2f}")

        return {"experiences": experiences, "total_reward": total_reward, "turns": len(experiences)}


# ================================================================================
# Part 5: 比較実験システム
# ================================================================================


class ComparisonExperiment:
    """教師あり学習 vs 強化学習の比較実験"""

    def __init__(self):
        # 教師あり学習版
        self.sl_system = IntegratedCQCNNSystem(n_qubits=8, n_layers=3)

        # 強化学習版
        self.rl_estimator = CQCNNPieceEstimator(n_qubits=8, n_layers=3)
        self.rl_trainer = RLCQCNNTrainer(self.rl_estimator)
        self.self_play = SelfPlaySystem(self.rl_trainer)

        # 実験結果
        self.results = {"sl_performance": [], "rl_performance": [], "training_episodes": []}

    def train_rl_model(self, episodes: int = 100, verbose: bool = True):
        """強化学習モデルを学習"""
        print(f"\n🤖 強化学習モデル学習開始 ({episodes}エピソード)")
        print("=" * 60)

        for episode in range(episodes):
            # 自己対戦による学習
            result = self.self_play.play_episode(verbose=False)

            # 学習実行
            train_result = self.rl_trainer.train_episode(lambda trainer, verbose: result, verbose=False)

            if verbose and (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{episodes}: "
                    f"Reward = {train_result['reward']:.2f}, "
                    f"Loss = {train_result['loss']:.4f}, "
                    f"ε = {train_result['epsilon']:.3f}"
                )

        print("✅ 強化学習完了")

    def evaluate_models(self, test_cases: int = 100) -> Dict:
        """両モデルを評価"""
        print(f"\n📊 モデル評価 ({test_cases}テストケース)")
        print("-" * 40)

        sl_correct = 0
        rl_correct = 0

        for _ in range(test_cases):
            # テストケース生成
            board = np.random.randint(-1, 2, (6, 6))
            enemy_positions = [(i, j) for i in range(6) for j in range(6) if board[j, i] == -1][:4]

            if not enemy_positions:
                continue

            # 真のタイプ
            true_types = {pos: random.choice(["good", "bad"]) for pos in enemy_positions}

            # ボードテンソル準備（共通メソッドを使用）
            board_tensor_sl = self._prepare_board_tensor(board, "A")
            board_tensor_rl = self._prepare_board_tensor(board, "A")

            # 評価
            for pos in enemy_positions:
                true_type = true_types[pos]

                with torch.no_grad():
                    # SL版の推定
                    features_sl = self.sl_system.estimator.feature_conv(board_tensor_sl)
                    features_flat_sl = features_sl.view(features_sl.size(0), -1)

                    # 量子入力の準備（SL版）
                    n_qubits_sl = self.sl_system.estimator.n_qubits
                    board_flat_sl = board_tensor_sl.view(board_tensor_sl.size(0), -1)
                    quantum_input_sl = board_flat_sl[:, :n_qubits_sl].clone()

                    # 量子回路シミュレーション（SL版）
                    batch_size_sl = quantum_input_sl.shape[0]
                    q_state_sl = torch.zeros(batch_size_sl, n_qubits_sl)

                    for layer in range(self.sl_system.estimator.n_layers):
                        new_q_state_sl = torch.zeros_like(q_state_sl)
                        for q in range(n_qubits_sl):
                            rotation_sl = self.sl_system.estimator.quantum_params[layer, q]
                            new_q_state_sl[:, q] = torch.sin(
                                rotation_sl[0] * quantum_input_sl[:, q % quantum_input_sl.shape[1]]
                            ) * torch.cos(
                                rotation_sl[1] * quantum_input_sl[:, (q + 1) % quantum_input_sl.shape[1]]
                            ) + torch.tanh(rotation_sl[2] * q_state_sl[:, q])
                        q_state_sl = new_q_state_sl

                    quantum_out_sl = q_state_sl
                    combined_sl = torch.cat([features_flat_sl, quantum_out_sl], dim=1)
                    type_logits_sl = self.sl_system.estimator.piece_type_head(combined_sl)
                    type_probs_sl = torch.softmax(type_logits_sl, dim=1)

                    if true_type == "good" and type_probs_sl[0, 0].item() > 0.5:
                        sl_correct += 1
                    elif true_type == "bad" and type_probs_sl[0, 1].item() > 0.5:
                        sl_correct += 1

                    # RL版の推定
                    features_rl = self.rl_estimator.feature_conv(board_tensor_rl)
                    features_flat_rl = features_rl.view(features_rl.size(0), -1)

                    # 量子入力の準備（RL版）
                    n_qubits_rl = self.rl_estimator.n_qubits
                    board_flat_rl = board_tensor_rl.view(board_tensor_rl.size(0), -1)
                    quantum_input_rl = board_flat_rl[:, :n_qubits_rl].clone()

                    # 量子回路シミュレーション（RL版）
                    batch_size_rl = quantum_input_rl.shape[0]
                    q_state_rl = torch.zeros(batch_size_rl, n_qubits_rl)

                    for layer in range(self.rl_estimator.n_layers):
                        new_q_state_rl = torch.zeros_like(q_state_rl)
                        for q in range(n_qubits_rl):
                            rotation_rl = self.rl_estimator.quantum_params[layer, q]
                            new_q_state_rl[:, q] = torch.sin(
                                rotation_rl[0] * quantum_input_rl[:, q % quantum_input_rl.shape[1]]
                            ) * torch.cos(
                                rotation_rl[1] * quantum_input_rl[:, (q + 1) % quantum_input_rl.shape[1]]
                            ) + torch.tanh(rotation_rl[2] * q_state_rl[:, q])
                        q_state_rl = new_q_state_rl

                    quantum_out_rl = q_state_rl
                    combined_rl = torch.cat([features_flat_rl, quantum_out_rl], dim=1)
                    type_logits_rl = self.rl_estimator.piece_type_head(combined_rl)
                    type_probs_rl = torch.softmax(type_logits_rl, dim=1)

                    if true_type == "good" and type_probs_rl[0, 0].item() > 0.5:
                        rl_correct += 1
                    elif true_type == "bad" and type_probs_rl[0, 1].item() > 0.5:
                        rl_correct += 1

        total_predictions = len([1 for _ in range(test_cases) for _ in range(4)])

        sl_accuracy = sl_correct / max(total_predictions, 1)
        rl_accuracy = rl_correct / max(total_predictions, 1)

        print(f"教師あり学習: {sl_accuracy:.1%} ({sl_correct}/{total_predictions})")
        print(f"強化学習:     {rl_accuracy:.1%} ({rl_correct}/{total_predictions})")

        return {
            "sl_accuracy": sl_accuracy,
            "rl_accuracy": rl_accuracy,
            "sl_correct": sl_correct,
            "rl_correct": rl_correct,
            "total": total_predictions,
        }

    def _prepare_board_tensor(self, board: np.ndarray, player: str) -> torch.Tensor:
        """ボード状態をテンソルに変換（共通メソッド）"""
        tensor = torch.zeros(1, 3, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val

        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))

        return tensor

    def plot_comparison(self):
        """学習曲線を比較プロット"""
        if not self.rl_trainer.training_history["episodes"]:
            print("⚠️ 学習データがありません")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 報酬推移
        ax1.plot(
            self.rl_trainer.training_history["episodes"], self.rl_trainer.training_history["rewards"], "b-", alpha=0.7
        )
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("強化学習: 報酬推移")
        ax1.grid(True)

        # 損失推移
        ax2.plot(
            self.rl_trainer.training_history["episodes"], self.rl_trainer.training_history["losses"], "r-", alpha=0.7
        )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.set_title("強化学習: 損失推移")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("rl_vs_sl_comparison.png")
        print("📊 グラフを保存: rl_vs_sl_comparison.png")


# ================================================================================
# Part 6: メイン実行
# ================================================================================


def main():
    """メイン実行関数"""
    print("🚀 強化学習版CQCNN vs 教師あり学習版CQCNN")
    print("=" * 70)

    # 実験システム初期化
    experiment = ComparisonExperiment()

    print("\n【実験メニュー】")
    print("1. 強化学習モデルを学習")
    print("2. 両モデルを評価・比較")
    print("3. 完全な比較実験を実行")

    choice = input("\n選択 (1-3): ").strip()

    if choice == "1":
        episodes = int(input("学習エピソード数 (10-1000): ") or "100")
        experiment.train_rl_model(episodes=episodes)
        experiment.rl_trainer.save_checkpoint("rl_cqcnn_checkpoint.pth")

    elif choice == "2":
        # チェックポイントがあれば読み込み
        if os.path.exists("rl_cqcnn_checkpoint.pth"):
            experiment.rl_trainer.load_checkpoint("rl_cqcnn_checkpoint.pth")

        experiment.evaluate_models(test_cases=100)

    elif choice == "3":
        print("\n📝 完全な比較実験")
        print("-" * 40)

        # 強化学習モデルを学習
        experiment.train_rl_model(episodes=200)

        # 評価
        print("\n" + "=" * 70)
        results = experiment.evaluate_models(test_cases=200)

        # グラフ作成
        experiment.plot_comparison()

        # 結果サマリー
        print("\n" + "=" * 70)
        print("🎯 実験結果サマリー")
        print("=" * 70)

        print("\n最終性能比較:")
        print(f"  教師あり学習: {results['sl_accuracy']:.1%}")
        print(f"  強化学習:     {results['rl_accuracy']:.1%}")

        if results["rl_accuracy"] > results["sl_accuracy"]:
            improvement = (results["rl_accuracy"] - results["sl_accuracy"]) * 100
            print(f"\n✨ 強化学習が {improvement:.1f}% 優れています！")
        elif results["sl_accuracy"] > results["rl_accuracy"]:
            improvement = (results["sl_accuracy"] - results["rl_accuracy"]) * 100
            print(f"\n📚 教師あり学習が {improvement:.1f}% 優れています")
        else:
            print("\n🤝 両手法はほぼ同等の性能です")

        # 結果を保存
        with open("comparison_results.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "rl_history": experiment.rl_trainer.training_history,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )

        print("\n💾 結果を保存: comparison_results.json")
        print("📊 グラフを保存: rl_vs_sl_comparison.png")

    else:
        print("無効な選択です")

    print("\n✅ 実験完了！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 実行中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback

        traceback.print_exc()

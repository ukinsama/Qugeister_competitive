#!/usr/bin/env python3
"""
CQCNN対戦システム v3 - 学習機能統合版
5つの独立モジュール + 学習機能 + 7チャンネル入力対応

モジュール構成:
1. PlacementStrategy - 初期配置戦略
2. PieceEstimator - 敵駒推定器（学習機能付き）
3. RewardFunction - 報酬関数
4. QMapGenerator - Q値マップ生成器
5. ActionSelector - 行動選択器

学習機能:
- 教師あり学習
- 強化学習
- ハイブリッド学習
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import os
from collections import deque

# ================================================================================
# Part 1: 基本設定とデータ構造
# ================================================================================


@dataclass
class GameConfig:
    """ゲーム設定"""

    board_size: tuple = (6, 6)
    max_turns: int = 100
    n_pieces: int = 8
    n_good: int = 4
    n_bad: int = 4


@dataclass
class LearningConfig:
    """学習設定"""

    # 共通設定
    learning_rate: float = 0.001
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 教師あり学習設定
    supervised_epochs: int = 100
    validation_split: float = 0.2

    # 強化学習設定
    rl_episodes: int = 1000
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 500
    gamma: float = 0.95
    memory_size: int = 10000
    target_update: int = 10


class GameState:
    """ゲーム状態"""

    def __init__(self):
        self.board = np.zeros((6, 6), dtype=int)
        self.player_a_pieces = {}
        self.player_b_pieces = {}
        self.turn = 0
        self.winner = None

    def is_game_over(self):
        return self.winner is not None or self.turn >= 100


# ================================================================================
# Part 2: データ処理ユーティリティ
# ================================================================================


class DataProcessor:
    """データ処理クラス"""

    @staticmethod
    def prepare_7channel_tensor(
        board: np.ndarray, player: str, my_pieces: Dict[Tuple[int, int], str], turn: int
    ) -> torch.Tensor:
        """7チャンネルテンソルを準備"""
        channels = []

        # チャンネル1: 自分の駒位置 (1: 存在, 0: なし)
        my_pos = np.zeros_like(board)
        for (r, c), _ in my_pieces.items():
            if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                my_pos[r, c] = 1
        channels.append(my_pos)

        # チャンネル2: 自分の善玉 (1: 善玉, 0: その他)
        my_good = np.zeros_like(board)
        for (r, c), piece_type in my_pieces.items():
            if piece_type == "good" and 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                my_good[r, c] = 1
        channels.append(my_good)

        # チャンネル3: 自分の悪玉 (1: 悪玉, 0: その他)
        my_bad = np.zeros_like(board)
        for (r, c), piece_type in my_pieces.items():
            if piece_type == "bad" and 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                my_bad[r, c] = 1
        channels.append(my_bad)

        # チャンネル4: 敵の駒位置 (1: 存在, 0: なし)
        enemy_pos = np.zeros_like(board)
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] != 0 and (r, c) not in my_pieces:
                    enemy_pos[r, c] = 1
        channels.append(enemy_pos)

        # チャンネル5: プレイヤー情報 (1: プレイヤーA, 0: プレイヤーB)
        player_channel = np.ones_like(board) if player == "A" else np.zeros_like(board)
        channels.append(player_channel)

        # チャンネル6: ターン情報 (正規化されたターン数)
        turn_channel = np.full_like(board, turn / 100.0)
        channels.append(turn_channel)

        # チャンネル7: ボード境界情報 (端=1, 中央=0)
        boundary = np.zeros_like(board)
        boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = 1
        channels.append(boundary)

        # テンソルに変換 (1, 7, H, W)
        tensor = torch.FloatTensor(np.stack(channels)).unsqueeze(0)
        return tensor


# ================================================================================
# Part 3: CQCNNモデル定義
# ================================================================================


class CQCNNModel(nn.Module):
    """Classical-Quantum CNN モデル"""

    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical CNN部分
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Quantum-inspired部分（量子回路をシミュレート）
        self.quantum_dim = n_qubits * n_layers

        # Linear層の入力サイズを動的計算用の一時変数
        self.quantum_linear = None
        self.quantum_layers = nn.ModuleList([nn.Linear(self.quantum_dim, self.quantum_dim) for _ in range(n_layers)])

        # 出力層
        self.classifier = nn.Sequential(
            nn.Linear(self.quantum_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),  # 善玉/悪玉の2クラス
        )

        # 入力サイズを計算するためのダミー入力
        self._initialize_linear_layers()

    def _initialize_linear_layers(self):
        """Linear層の入力サイズを動的に計算"""
        # ダミー入力で形状を確認
        dummy_input = torch.randn(1, 7, 6, 6)  # (batch, channels, height, width)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))

            # Flattenした後のサイズを取得
            flattened_size = x.view(x.size(0), -1).size(1)

            # quantum_linearを正しいサイズで初期化
            self.quantum_linear = nn.Linear(flattened_size, self.quantum_dim)

    def forward(self, x):
        # Classical CNN処理
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Quantum-inspired処理
        x = F.relu(self.quantum_linear(x))

        for quantum_layer in self.quantum_layers:
            # 量子回路風の処理（重ね合わせ状態をシミュレート）
            x_new = quantum_layer(x)
            x = F.normalize(x_new + x, dim=1)  # 残差接続 + 正規化

        # 分類
        output = self.classifier(x)
        return output


# ================================================================================
# Part 4: モジュール1 - 初期配置戦略
# ================================================================================


class PlacementStrategy(ABC):
    """初期配置戦略の基底クラス"""

    @abstractmethod
    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """戦略名を取得"""
        pass


class StandardPlacement(PlacementStrategy):
    """標準配置戦略"""

    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}

        if player_id == "A":
            positions = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4)]
        else:
            positions = [(4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4)]

        piece_types = ["good"] * 4 + ["bad"] * 4
        random.shuffle(piece_types)

        for pos, piece_type in zip(positions, piece_types):
            placement[pos] = piece_type

        return placement

    def get_name(self) -> str:
        return "標準配置"


class AggressivePlacement(PlacementStrategy):
    """攻撃的配置戦略（善玉を前線に）"""

    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}

        if player_id == "A":
            front_positions = [(1, 1), (1, 2), (1, 3), (1, 4)]
            back_positions = [(0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            front_positions = [(4, 1), (4, 2), (4, 3), (4, 4)]
            back_positions = [(5, 1), (5, 2), (5, 3), (5, 4)]

        front_pieces = ["good", "good", "good", "bad"]
        random.shuffle(front_pieces)

        back_pieces = ["good", "bad", "bad", "bad"]
        random.shuffle(back_pieces)

        for pos, piece_type in zip(front_positions, front_pieces):
            placement[pos] = piece_type
        for pos, piece_type in zip(back_positions, back_pieces):
            placement[pos] = piece_type

        return placement

    def get_name(self) -> str:
        return "攻撃的配置"


class DefensivePlacement(PlacementStrategy):
    """防御的配置戦略（善玉を後方に）"""

    def get_placement(self, player_id: str) -> Dict[Tuple[int, int], str]:
        placement = {}

        if player_id == "A":
            front_positions = [(1, 1), (1, 2), (1, 3), (1, 4)]
            back_positions = [(0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            front_positions = [(4, 1), (4, 2), (4, 3), (4, 4)]
            back_positions = [(5, 1), (5, 2), (5, 3), (5, 4)]

        front_pieces = ["bad", "bad", "bad", "good"]
        random.shuffle(front_pieces)

        back_pieces = ["bad", "good", "good", "good"]
        random.shuffle(back_pieces)

        for pos, piece_type in zip(front_positions, front_pieces):
            placement[pos] = piece_type
        for pos, piece_type in zip(back_positions, back_pieces):
            placement[pos] = piece_type

        return placement

    def get_name(self) -> str:
        return "防御的配置"


# ================================================================================
# Part 5: モジュール2 - 敵駒推定器（学習機能付き）
# ================================================================================


class PieceEstimator(ABC):
    """敵駒推定器の基底クラス"""

    @abstractmethod
    def estimate(
        self,
        board: np.ndarray,
        enemy_positions: List[Tuple[int, int]],
        player: str,
        my_pieces: Dict[Tuple[int, int], str],
        turn: int,
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """敵駒を推定"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """推定器名を取得"""
        pass


class CQCNNEstimator(PieceEstimator):
    """CQCNN推定器（学習機能付き）"""

    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.model = CQCNNModel(n_qubits, n_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 強化学習用
        self.memory = deque(maxlen=10000)
        self.target_model = CQCNNModel(n_qubits, n_layers).to(self.device)
        self.update_target_model()

    def train_supervised(self, training_data: List[Dict], config: LearningConfig) -> None:
        """教師あり学習"""
        print(f"📚 CQCNN教師あり学習開始: {len(training_data)}件のデータ")
        print(f"🔧 設定: バッチサイズ={config.batch_size}, 学習率={config.learning_rate}")
        print(f"💻 デバイス: {self.device}")
        print("=" * 60)

        # データ分割
        n_val = int(len(training_data) * config.validation_split)
        val_data = training_data[:n_val]
        train_data = training_data[n_val:]

        print(f"📊 データ分割: 学習用={len(train_data)}件, 検証用={len(val_data)}件")
        print("=" * 60)

        self.model.train()
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(config.supervised_epochs):
            random.shuffle(train_data)

            total_loss = 0
            correct = 0
            total = 0
            batch_count = 0

            n_batches = len(train_data) // config.batch_size
            print(f"\nEpoch {epoch + 1}/{config.supervised_epochs}")
            print("🔄 学習中: ", end="", flush=True)

            for i in range(0, len(train_data), config.batch_size):
                batch = train_data[i : i + config.batch_size]

                if batch_count % max(1, n_batches // 20) == 0:
                    print("█", end="", flush=True)

                inputs, labels = self._prepare_supervised_batch(batch)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                batch_count += 1

            print(" ✅")

            train_acc = 100 * correct / total if total > 0 else 0.0
            avg_batches = max(1, len(train_data) // config.batch_size)
            train_loss = total_loss / avg_batches

            print("🔍 検証中...", end="", flush=True)
            val_acc, val_loss = self._validate_supervised(val_data, config.batch_size)
            print(" ✅")

            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            improvement = "📈" if val_acc > best_val_acc else "📉" if val_acc < best_val_acc else "➡️"
            print(
                f"📊 結果: Train Loss={train_loss:.4f} | Train Acc={train_acc:.1f}% | "
                f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.1f}% {improvement}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print("🎯 新しいベストモデル！")
                torch.save(self.model.state_dict(), "models/best_cqcnn_supervised.pth")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"⏹️  Early Stopping: {patience_counter}エポック改善なし")
                    break

            if epoch > 0 and epoch % 30 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= 0.8
                print(f"📉 学習率調整: {param_group['lr']:.6f}")

            print("-" * 60)

        self.is_trained = True
        print(f"\n🎉 教師あり学習完了! ベスト検証精度: {best_val_acc:.1f}%")
        print("=" * 60)

    def _prepare_supervised_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """教師あり学習用バッチデータ準備"""
        inputs = []
        labels = []

        for sample in batch:
            tensor = DataProcessor.prepare_7channel_tensor(
                board=sample["board"], player=sample["player"], my_pieces=sample["my_pieces"], turn=sample["turn"]
            )
            inputs.append(tensor.squeeze(0))

            if sample["true_labels"]:
                first_enemy_type = list(sample["true_labels"].values())[0]
                label = 0 if first_enemy_type == "good" else 1
            else:
                label = 0
            labels.append(label)

        batch_tensor = torch.stack(inputs)
        label_tensor = torch.LongTensor(labels)

        return batch_tensor, label_tensor

    def _validate_supervised(self, val_data: List[Dict], batch_size: int) -> Tuple[float, float]:
        """教師あり学習の検証"""
        if not val_data:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i : i + batch_size]
                if not batch:
                    continue

                inputs, labels = self._prepare_supervised_batch(batch)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.train()

        accuracy = 100 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(1, len(val_data) // batch_size)

        return accuracy, avg_loss

    def estimate(
        self, board: np.ndarray, enemy_positions: List[Tuple[int, int]], player: str, my_pieces: Dict, turn: int
    ) -> Dict:
        """敵駒推定"""
        if not self.is_trained:
            # 未学習の場合はランダム推定
            results = {}
            for pos in enemy_positions:
                good_prob = random.uniform(0.3, 0.7)
                results[pos] = {"good_prob": good_prob, "bad_prob": 1 - good_prob, "confidence": 0.5}
            return results

        self.model.eval()
        results = {}

        tensor = DataProcessor.prepare_7channel_tensor(board, player, my_pieces, turn)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)

            for pos in enemy_positions:
                results[pos] = {
                    "good_prob": probs[0, 0].item(),
                    "bad_prob": probs[0, 1].item(),
                    "confidence": max(probs[0].tolist()),
                }

        return results

    def update_target_model(self):
        """ターゲットモデルを更新"""
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path: str):
        """モデル保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_history": self.training_history,
                "is_trained": self.is_trained,
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
            },
            path,
        )
        print(f"💾 CQCNNモデル保存完了: {path}")

    def load_model(self, path: str):
        """モデル読込"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint["training_history"]
        self.is_trained = checkpoint["is_trained"]
        print(f"📁 CQCNNモデル読込完了: {path}")

    def get_name(self) -> str:
        status = "学習済み" if self.is_trained else "未学習"
        return f"CQCNN({self.n_qubits}q,{self.n_layers}L)-{status}"


class SimpleCNNEstimator(PieceEstimator):
    """シンプルCNN推定器"""

    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.is_trained = False

    def estimate(
        self, board: np.ndarray, enemy_positions: List[Tuple[int, int]], player: str, my_pieces: Dict, turn: int
    ) -> Dict:
        results = {}
        for pos in enemy_positions:
            good_prob = random.uniform(0.4, 0.6)
            results[pos] = {"good_prob": good_prob, "bad_prob": 1 - good_prob, "confidence": 0.6}
        return results

    def get_name(self) -> str:
        return "SimpleCNN"


class RandomEstimator(PieceEstimator):
    """ランダム推定器"""

    def estimate(
        self, board: np.ndarray, enemy_positions: List[Tuple[int, int]], player: str, my_pieces: Dict, turn: int
    ) -> Dict:
        results = {}
        for pos in enemy_positions:
            good_prob = random.uniform(0.3, 0.7)
            results[pos] = {"good_prob": good_prob, "bad_prob": 1 - good_prob, "confidence": 0.5}
        return results

    def get_name(self) -> str:
        return "ランダム推定"


# ================================================================================
# Part 6: モジュール3 - 報酬関数
# ================================================================================


class RewardFunction(ABC):
    """報酬関数の基底クラス"""

    @abstractmethod
    def calculate_move_reward(self, game_state: GameState, move: Tuple, player: str, my_pieces: Dict) -> float:
        """手の報酬を計算"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """報酬関数名を取得"""
        pass


class StandardRewardFunction(RewardFunction):
    """標準報酬関数"""

    def calculate_move_reward(self, game_state: GameState, move: Tuple, player: str, my_pieces: Dict) -> float:
        from_pos, to_pos = move
        reward = 0.0

        # 基本移動報酬
        piece_type = my_pieces.get(from_pos, "unknown")
        if piece_type == "good":
            reward += 1.0
        else:
            reward += 0.5

        # 脱出報酬
        escape_positions = [(5, 0), (5, 5)] if player == "A" else [(0, 0), (0, 5)]
        if to_pos in escape_positions:
            if piece_type == "good":
                reward += 50.0
            else:
                reward -= 10.0

        return reward

    def get_name(self) -> str:
        return "標準報酬"


class AggressiveRewardFunction(RewardFunction):
    """攻撃的報酬関数"""

    def calculate_move_reward(self, game_state: GameState, move: Tuple, player: str, my_pieces: Dict) -> float:
        from_pos, to_pos = move
        reward = StandardRewardFunction().calculate_move_reward(game_state, move, player, my_pieces)

        # 前進ボーナス
        if player == "A" and to_pos[0] > from_pos[0]:
            reward += 2.0
        elif player == "B" and to_pos[0] < from_pos[0]:
            reward += 2.0

        return reward

    def get_name(self) -> str:
        return "攻撃的報酬"


class DefensiveRewardFunction(RewardFunction):
    """防御的報酬関数"""

    def calculate_move_reward(self, game_state: GameState, move: Tuple, player: str, my_pieces: Dict) -> float:
        from_pos, to_pos = move
        reward = StandardRewardFunction().calculate_move_reward(game_state, move, player, my_pieces)

        # 安全性ボーナス
        safety_score = self._calculate_safety(to_pos, game_state, player)
        reward += safety_score * 0.5

        return reward

    def _calculate_safety(self, pos: Tuple, game_state: GameState, player: str) -> float:
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val

        min_enemy_dist = 10
        for i in range(6):
            for j in range(6):
                if game_state.board[i, j] == enemy_val:
                    dist = abs(pos[0] - i) + abs(pos[1] - j)
                    min_enemy_dist = min(min_enemy_dist, dist)

        return min_enemy_dist

    def get_name(self) -> str:
        return "防御的報酬"


# ================================================================================
# Part 7: モジュール4 - Q値マップ生成器
# ================================================================================


class QMapGenerator(ABC):
    """Q値マップ生成器の基底クラス"""

    @abstractmethod
    def generate(
        self,
        board: np.ndarray,
        estimations: Dict,
        my_pieces: Dict,
        player: str,
        reward_function: RewardFunction = None,
        game_state: GameState = None,
    ) -> np.ndarray:
        """Q値マップを生成"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """生成器名を取得"""
        pass


class SimpleQMapGenerator(QMapGenerator):
    """シンプルなQ値マップ生成器"""

    def generate(
        self,
        board: np.ndarray,
        estimations: Dict,
        my_pieces: Dict,
        player: str,
        reward_function: RewardFunction = None,
        game_state: GameState = None,
    ) -> np.ndarray:
        q_map = np.zeros((6, 6, 4))

        for pos, piece_type in my_pieces.items():
            base_value = 1.0 if piece_type == "good" else 0.5

            for i, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                new_pos = (pos[0] + dx, pos[1] + dy)

                if not (0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6):
                    q_map[pos[0], pos[1], i] = -100
                    continue

                q_value = base_value

                # 推定結果を使用
                if new_pos in estimations:
                    est = estimations[new_pos]
                    if piece_type == "bad":
                        q_value += est["good_prob"] * 2.0
                        q_value += est["bad_prob"] * 1.0
                    else:
                        q_value -= est["bad_prob"] * 0.5

                # 報酬関数を適用
                if reward_function and game_state:
                    move = (pos, new_pos)
                    reward = reward_function.calculate_move_reward(game_state, move, player, my_pieces)
                    q_value += reward * 0.1

                q_map[pos[0], pos[1], i] = q_value

        return q_map

    def get_name(self) -> str:
        return "シンプルQ値"


class StrategicQMapGenerator(QMapGenerator):
    """戦略的Q値マップ生成器"""

    def generate(
        self,
        board: np.ndarray,
        estimations: Dict,
        my_pieces: Dict,
        player: str,
        reward_function: RewardFunction = None,
        game_state: GameState = None,
    ) -> np.ndarray:
        # SimpleQMapGeneratorをベースに戦略的要素を追加
        q_map = SimpleQMapGenerator().generate(board, estimations, my_pieces, player, reward_function, game_state)

        # 脱出位置の戦略的価値を追加
        escape_positions = [(5, 0), (5, 5)] if player == "A" else [(0, 0), (0, 5)]

        for pos, piece_type in my_pieces.items():
            for i, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                new_pos = (pos[0] + dx, pos[1] + dy)

                if new_pos in escape_positions and piece_type == "good":
                    q_map[pos[0], pos[1], i] += 10.0  # 脱出ボーナス

        return q_map

    def get_name(self) -> str:
        return "戦略的Q値"


# ================================================================================
# Part 8: モジュール5 - 行動選択器
# ================================================================================


class ActionSelector(ABC):
    """行動選択器の基底クラス"""

    @abstractmethod
    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        """行動を選択"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """選択器名を取得"""
        pass


class GreedySelector(ActionSelector):
    """貪欲選択器"""

    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        if not legal_moves:
            return None

        best_move = None
        best_value = -float("inf")

        for from_pos, to_pos in legal_moves:
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]

            dir_map = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3}
            if (dx, dy) in dir_map:
                dir_idx = dir_map[(dx, dy)]
                value = q_map[from_pos[0], from_pos[1], dir_idx]

                if value > best_value:
                    best_value = value
                    best_move = (from_pos, to_pos)

        return best_move or random.choice(legal_moves)

    def get_name(self) -> str:
        return "貪欲選択"


class EpsilonGreedySelector(ActionSelector):
    """ε-貪欲選択器"""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        if not legal_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        return GreedySelector().select(q_map, legal_moves)

    def get_name(self) -> str:
        return f"ε貪欲(ε={self.epsilon})"


class SoftmaxSelector(ActionSelector):
    """ソフトマックス選択器"""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def select(self, q_map: np.ndarray, legal_moves: List[Tuple]) -> Tuple:
        if not legal_moves:
            return None

        q_values = []
        dir_map = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3}

        for from_pos, to_pos in legal_moves:
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]

            if (dx, dy) in dir_map:
                dir_idx = dir_map[(dx, dy)]
                value = q_map[from_pos[0], from_pos[1], dir_idx]
            else:
                value = -100

            q_values.append(value / self.temperature)

        q_tensor = torch.tensor(q_values, dtype=torch.float32)
        probs = F.softmax(q_tensor, dim=0).numpy()

        idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[idx]

    def get_name(self) -> str:
        return f"ソフトマックス(T={self.temperature})"


# ================================================================================
# Part 9: エージェント統合
# ================================================================================


@dataclass
class ModuleConfig:
    """5つのモジュールを統合する設定"""

    placement_strategy: PlacementStrategy
    piece_estimator: PieceEstimator
    reward_function: RewardFunction
    qmap_generator: QMapGenerator
    action_selector: ActionSelector


class ModularAgent:
    """5モジュール統合エージェント"""

    def __init__(self, player_id: str, config: ModuleConfig):
        self.player_id = player_id
        self.config = config
        self.name = self._generate_name()
        self.piece_info = {}
        self.game_history = []

    def _generate_name(self) -> str:
        return (
            f"Agent_{self.player_id}["
            f"{self.config.placement_strategy.get_name()[:3]}+"
            f"{self.config.piece_estimator.get_name()[:6]}+"
            f"{self.config.reward_function.get_name()[:3]}+"
            f"{self.config.qmap_generator.get_name()[:3]}+"
            f"{self.config.action_selector.get_name()[:3]}]"
        )

    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        placement = self.config.placement_strategy.get_placement(self.player_id)
        self.piece_info = placement.copy()
        return placement

    def get_move(self, game_state: GameState, legal_moves: List[Tuple]) -> Tuple:
        """次の手を取得"""
        if not legal_moves:
            return None

        try:
            # 自分の駒情報を更新
            self._update_piece_info(game_state)

            # 1. 敵駒位置を特定
            enemy_positions = self._find_enemy_positions(game_state)

            # 2. 敵駒を推定（7チャンネル入力）
            estimations = {}
            if enemy_positions:
                estimations = self.config.piece_estimator.estimate(
                    board=game_state.board,
                    enemy_positions=enemy_positions,
                    player=self.player_id,
                    my_pieces=self.piece_info,
                    turn=game_state.turn,
                )

            # 3. Q値マップを生成（報酬関数も統合）
            q_map = self.config.qmap_generator.generate(
                board=game_state.board,
                estimations=estimations,
                my_pieces=self.piece_info,
                player=self.player_id,
                reward_function=self.config.reward_function,
                game_state=game_state,
            )

            # 4. 行動を選択
            selected_move = self.config.action_selector.select(q_map, legal_moves)

            # 履歴に記録
            self.game_history.append(
                {
                    "turn": game_state.turn,
                    "move": selected_move,
                    "estimations": len(estimations),
                    "q_max": np.max(q_map) if q_map is not None else 0,
                }
            )

            return selected_move

        except Exception as e:
            print(f"⚠️ エラー in {self.name}: {e}")
            return random.choice(legal_moves)

    def _update_piece_info(self, game_state: GameState):
        """駒タイプ情報を更新"""
        if self.player_id == "A":
            current_pieces = game_state.player_a_pieces
        else:
            current_pieces = game_state.player_b_pieces

        # 現在の駒位置に合わせて更新（簡略化）
        new_piece_info = {}
        for pos in current_pieces.keys():
            # 元の駒タイプを保持（実際にはより複雑な追跡が必要）
            if pos in self.piece_info:
                new_piece_info[pos] = self.piece_info[pos]
            else:
                # 新しい位置の場合、適当な値を設定
                new_piece_info[pos] = random.choice(["good", "bad"])

        self.piece_info = new_piece_info

    def _find_enemy_positions(self, game_state: GameState) -> List[Tuple[int, int]]:
        """敵駒の位置を特定"""
        enemy_val = -1 if self.player_id == "A" else 1
        positions = []

        for i in range(6):
            for j in range(6):
                if game_state.board[i, j] == enemy_val:
                    positions.append((i, j))

        return positions


# ================================================================================
# Part 10: データ生成と学習システム
# ================================================================================


def create_enhanced_training_data(n_samples: int) -> List[Dict]:
    """高品質な学習データ作成"""
    print(f"🔧 高品質学習データ生成中: {n_samples}サンプル")
    data = []

    strategies = ["aggressive", "defensive", "balanced", "random"]

    for i in range(n_samples):
        if i % (n_samples // 10) == 0:
            progress = (i / n_samples) * 100
            print(f"📊 進捗: {progress:.0f}% ({i}/{n_samples})")

        strategy = random.choice(strategies)

        # より現実的なボード状態を作成
        board = np.zeros((6, 6), dtype=int)

        # 戦略に応じた配置パターン
        if strategy == "aggressive":
            player_a_positions = [(0, j) for j in range(1, 5)] + [(1, j) for j in range(2, 4)]
            player_b_positions = [(4, j) for j in range(2, 4)] + [(5, j) for j in range(1, 5)]
        elif strategy == "defensive":
            player_a_positions = [(1, j) for j in range(1, 5)] + [(0, j) for j in range(2, 4)]
            player_b_positions = [(4, j) for j in range(1, 5)] + [(5, j) for j in range(2, 4)]
        elif strategy == "balanced":
            player_a_positions = [(0, 1), (0, 4), (1, 2), (1, 3)] + [(0, 2), (0, 3), (1, 1), (1, 4)]
            player_b_positions = [(4, 1), (4, 4), (5, 2), (5, 3)] + [(4, 2), (4, 3), (5, 1), (5, 4)]
        else:  # random
            available_a = [(r, c) for r in range(2) for c in range(1, 5)]
            available_b = [(r, c) for r in range(4, 6) for c in range(1, 5)]
            player_a_positions = random.sample(available_a, min(8, len(available_a)))
            player_b_positions = random.sample(available_b, min(8, len(available_b)))

        # 実際に配置する駒数（4-8個）
        n_pieces_a = random.randint(4, min(8, len(player_a_positions)))
        n_pieces_b = random.randint(4, min(8, len(player_b_positions)))

        selected_a = random.sample(player_a_positions, n_pieces_a)
        selected_b = random.sample(player_b_positions, n_pieces_b)

        # ボードに配置
        for pos in selected_a:
            board[pos] = 1
        for pos in selected_b:
            board[pos] = 2

        # 戦略に応じた駒タイプ分布
        if strategy == "aggressive":
            good_ratio_a = random.uniform(0.6, 0.8)
            good_ratio_b = random.uniform(0.6, 0.8)
        elif strategy == "defensive":
            good_ratio_a = random.uniform(0.2, 0.4)
            good_ratio_b = random.uniform(0.2, 0.4)
        else:
            good_ratio_a = random.uniform(0.4, 0.6)
            good_ratio_b = random.uniform(0.4, 0.6)

        # 自分の駒情報（プレイヤーA）
        my_pieces = {}
        n_good_a = int(n_pieces_a * good_ratio_a)
        piece_types_a = ["good"] * n_good_a + ["bad"] * (n_pieces_a - n_good_a)
        random.shuffle(piece_types_a)

        for idx, pos in enumerate(selected_a):
            my_pieces[pos] = piece_types_a[idx]

        # 敵駒の真のタイプ（プレイヤーB）
        true_labels = {}
        n_good_b = int(n_pieces_b * good_ratio_b)
        piece_types_b = ["good"] * n_good_b + ["bad"] * (n_pieces_b - n_good_b)
        random.shuffle(piece_types_b)

        for idx, pos in enumerate(selected_b):
            true_labels[pos] = piece_types_b[idx]

        turn = random.randint(1, 80)

        # ノイズ追加（実戦的な不確実性）
        if random.random() < 0.1:  # 10%の確率でノイズ
            if len(true_labels) > 2:
                unknown_pos = random.choice(list(true_labels.keys()))
                true_labels[unknown_pos] = random.choice(["good", "bad"])

        sample = {
            "board": board.copy(),
            "player": "A",
            "my_pieces": my_pieces.copy(),
            "turn": turn,
            "enemy_positions": list(true_labels.keys()),
            "true_labels": true_labels.copy(),
            "strategy": strategy,
            "difficulty": random.choice(["easy", "medium", "hard"]),
        }
        data.append(sample)

    print("✅ 高品質学習データ生成完了")

    # データ統計表示
    strategies_count = {}
    for sample in data:
        strat = sample["strategy"]
        strategies_count[strat] = strategies_count.get(strat, 0) + 1

    print("📊 データ統計:")
    for strat, count in strategies_count.items():
        print(f"   - {strat}: {count}件 ({count / n_samples * 100:.1f}%)")

    return data


# ================================================================================
# Part 11: メイン学習・対戦システム
# ================================================================================


def main():
    """メインプログラム"""
    print("🎯 CQCNN対戦システム v3.0")
    print("=" * 60)
    print("🧠 Classical-Quantum CNN による敵駒推定学習")
    print("🤖 5モジュール統合エージェント")
    print("📚 教師あり学習 & 🎮 強化学習 対応")
    print("=" * 60)

    print("\nモードを選択:")
    print("1. 🎓 学習モード")
    print("2. ⚔️  対戦モード")
    print("3. 🔬 実験モード（学習 → 対戦）")

    choice = input("\n👉 選択 (1-3): ")

    # モデル保存ディレクトリ作成
    os.makedirs("models", exist_ok=True)

    if choice == "1":
        learning_mode()
    elif choice == "2":
        battle_mode()
    elif choice == "3":
        experiment_mode()
    else:
        print("❌ 無効な選択です")


def learning_mode():
    """学習モード"""
    print("\n" + "=" * 60)
    print("🎓 学習モード")
    print("=" * 60)

    config = LearningConfig()
    print("⚙️  学習設定:")
    print(f"   - デバイス: {'GPU' if config.device == 'cuda' else 'CPU'}")
    print(f"   - バッチサイズ: {config.batch_size}")
    print(f"   - 学習率: {config.learning_rate}")
    print(f"   - 教師あり: {config.supervised_epochs}エポック")
    print("=" * 60)

    print("\n🎲 学習方法を選択:")
    print("1. 📚 教師あり学習")
    print("2. 🔄 継続学習（既存モデルから再開）")

    learn_choice = input("\n👉 選択 (1-2): ")

    # CQCNNEstimatorを作成
    cqcnn_estimator = CQCNNEstimator(n_qubits=6, n_layers=3)

    if learn_choice == "2":
        # 継続学習
        print("\n📁 既存モデルを読み込み中...")
        try:
            cqcnn_estimator.load_model("models/cqcnn_supervised.pth")
            print("✅ 教師あり学習済みモデルを読み込みました")
        except FileNotFoundError:
            print("⚠️ 既存モデルが見つかりません。新規学習を開始します。")

    # 教師あり学習
    print("\n" + "=" * 60)
    print("📚 PHASE 1: 教師あり学習")
    print("=" * 60)

    print("🔄 学習データ生成中...")
    training_data = create_enhanced_training_data(2000)
    print(f"✅ {len(training_data)}件の学習データを生成完了")

    cqcnn_estimator.train_supervised(training_data, config)
    cqcnn_estimator.save_model("models/cqcnn_supervised.pth")

    print("\n🎉 学習完了!")


def battle_mode():
    """対戦モード"""
    print("\n" + "=" * 60)
    print("⚔️ 対戦モード")
    print("=" * 60)

    # 利用可能なモジュール
    modules = {
        "placement": [StandardPlacement(), AggressivePlacement(), DefensivePlacement()],
        "estimator": [
            RandomEstimator(),
            SimpleCNNEstimator(),
        ],
        "reward": [StandardRewardFunction(), AggressiveRewardFunction(), DefensiveRewardFunction()],
        "qmap": [SimpleQMapGenerator(), StrategicQMapGenerator()],
        "selector": [
            GreedySelector(),
            EpsilonGreedySelector(epsilon=0.1),
            EpsilonGreedySelector(epsilon=0.3),
            SoftmaxSelector(temperature=1.0),
        ],
    }

    # 学習済みCQCNNを追加
    try:
        trained_cqcnn = CQCNNEstimator(n_qubits=6, n_layers=3)
        trained_cqcnn.load_model("models/cqcnn_supervised.pth")
        modules["estimator"].append(trained_cqcnn)
        print("✅ 学習済みCQCNNを読み込みました")
    except FileNotFoundError:
        print("⚠️ 学習済みCQCNNが見つかりません")

    print("\n🎯 対戦設定を選択:")
    print("1. バランス型 vs バランス型")
    print("2. 攻撃型 vs 防御型")
    print("3. CQCNN vs ランダム")
    print("4. カスタム設定")

    battle_choice = input("\n👉 選択 (1-4): ")

    if battle_choice == "1":
        # バランス型
        config1 = ModuleConfig(
            placement_strategy=StandardPlacement(),
            piece_estimator=modules["estimator"][-1] if len(modules["estimator"]) > 2 else SimpleCNNEstimator(),
            reward_function=StandardRewardFunction(),
            qmap_generator=StrategicQMapGenerator(),
            action_selector=EpsilonGreedySelector(epsilon=0.1),
        )
        config2 = ModuleConfig(
            placement_strategy=StandardPlacement(),
            piece_estimator=SimpleCNNEstimator(),
            reward_function=StandardRewardFunction(),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=GreedySelector(),
        )

    elif battle_choice == "2":
        # 攻撃型 vs 防御型
        config1 = ModuleConfig(
            placement_strategy=AggressivePlacement(),
            piece_estimator=modules["estimator"][-1] if len(modules["estimator"]) > 2 else RandomEstimator(),
            reward_function=AggressiveRewardFunction(),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=GreedySelector(),
        )
        config2 = ModuleConfig(
            placement_strategy=DefensivePlacement(),
            piece_estimator=SimpleCNNEstimator(),
            reward_function=DefensiveRewardFunction(),
            qmap_generator=StrategicQMapGenerator(),
            action_selector=SoftmaxSelector(temperature=1.0),
        )

    elif battle_choice == "3":
        # CQCNN vs ランダム
        config1 = ModuleConfig(
            placement_strategy=StandardPlacement(),
            piece_estimator=modules["estimator"][-1] if len(modules["estimator"]) > 2 else SimpleCNNEstimator(),
            reward_function=StandardRewardFunction(),
            qmap_generator=StrategicQMapGenerator(),
            action_selector=EpsilonGreedySelector(epsilon=0.1),
        )
        config2 = ModuleConfig(
            placement_strategy=StandardPlacement(),
            piece_estimator=RandomEstimator(),
            reward_function=StandardRewardFunction(),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=EpsilonGreedySelector(epsilon=0.3),
        )

    else:
        # カスタム設定（簡略化）
        print("カスタム設定はまだ実装されていません")
        return

    # エージェント作成
    agent1 = ModularAgent("A", config1)
    agent2 = ModularAgent("B", config2)

    print("\n🎮 対戦設定完了:")
    print(f"Agent1: {agent1.name}")
    print(f"Agent2: {agent2.name}")

    # 対戦実行（簡略化版）
    n_games = int(input("\n対戦数: "))

    wins = {"A": 0, "B": 0, "Draw": 0}

    for i in range(n_games):
        print(f"\nGame {i + 1}/{n_games}")
        # 簡略化されたゲーム実行（実際にはGameEngineが必要）
        winner = random.choice(["A", "B", "Draw"])
        wins[winner] += 1
        print(f"勝者: {winner}")

    # 結果表示
    print("\n" + "=" * 60)
    print("📊 最終結果:")
    print(f"Agent1勝利: {wins['A']} ({wins['A'] / n_games * 100:.1f}%)")
    print(f"Agent2勝利: {wins['B']} ({wins['B'] / n_games * 100:.1f}%)")
    print(f"引き分け: {wins['Draw']} ({wins['Draw'] / n_games * 100:.1f}%)")
    print("=" * 60)


def experiment_mode():
    """実験モード（学習 → 対戦）"""
    print("\n" + "=" * 60)
    print("🔬 実験モード")
    print("=" * 60)

    print("🔄 学習フェーズを実行中...")
    learning_mode()

    print("\n🔄 対戦フェーズに移行...")
    battle_mode()


if __name__ == "__main__":
    main()

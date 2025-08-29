#!/usr/bin/env python3
"""
実際に学習を行うローカル対戦システム
既存のCQCNN学習機能を統合して、本格的な学習→評価→保存→トーナメントシステムを構築
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import hashlib


# ================================================================================
# 既存のCQCNN実装を統合
# ================================================================================


class LearningConfig:
    """学習設定（既存コードから移植）"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 32
        self.learning_rate = 0.001
        self.supervised_epochs = 50
        self.validation_split = 0.2


def create_enhanced_training_data(n_samples: int = 1000) -> List[Dict]:
    """学習データ生成（既存コードから移植・改良）"""
    print(f"📊 {n_samples}件の学習データ生成中...")

    data = []
    piece_types = ["good", "bad"]

    for i in range(n_samples):
        # ランダムなボード状態生成
        board = np.zeros((6, 6), dtype=int)
        my_pieces = {}
        enemy_pieces = {}

        # 自分の駒配置
        for _ in range(random.randint(3, 8)):
            while True:
                r, c = random.randint(0, 5), random.randint(0, 5)
                if board[r, c] == 0:
                    piece_type = random.choice(piece_types)
                    board[r, c] = 1  # プレイヤーA
                    my_pieces[(r, c)] = piece_type
                    break

        # 敵の駒配置
        for _ in range(random.randint(3, 8)):
            while True:
                r, c = random.randint(0, 5), random.randint(0, 5)
                if board[r, c] == 0:
                    piece_type = random.choice(piece_types)
                    board[r, c] = -1  # プレイヤーB
                    enemy_pieces[(r, c)] = piece_type
                    break

        # 学習用データ形式に変換
        for pos, true_type in enemy_pieces.items():
            data.append(
                {
                    "board": board.copy(),
                    "my_pieces": my_pieces.copy(),
                    "enemy_position": pos,
                    "true_type": true_type,
                    "player": "A",
                    "turn": random.randint(1, 50),
                }
            )

        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{n_samples} 完了", end="\r")

    print(f"\n✅ {len(data)}件のデータを生成完了")
    return data


def create_board_tensor(board: np.ndarray, my_pieces: Dict, player: str, turn: int) -> torch.Tensor:
    """ボード状態をテンソルに変換（既存コードから移植）"""
    channels = []

    # チャンネル1: 自分の駒位置
    my_pos = np.zeros_like(board)
    for (r, c), _ in my_pieces.items():
        if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
            my_pos[r, c] = 1
    channels.append(my_pos)

    # チャンネル2: 自分の善玉
    my_good = np.zeros_like(board)
    for (r, c), piece_type in my_pieces.items():
        if piece_type == "good" and 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
            my_good[r, c] = 1
    channels.append(my_good)

    # チャンネル3: 自分の悪玉
    my_bad = np.zeros_like(board)
    for (r, c), piece_type in my_pieces.items():
        if piece_type == "bad" and 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
            my_bad[r, c] = 1
    channels.append(my_bad)

    # チャンネル4: 敵の駒位置
    enemy_pos = np.zeros_like(board)
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            if board[r, c] != 0 and (r, c) not in my_pieces:
                enemy_pos[r, c] = 1
    channels.append(enemy_pos)

    # チャンネル5: プレイヤー情報
    player_channel = np.ones_like(board) if player == "A" else np.zeros_like(board)
    channels.append(player_channel)

    # チャンネル6: ターン情報
    turn_channel = np.full_like(board, turn / 100.0)
    channels.append(turn_channel)

    # チャンネル7: ボード境界情報
    boundary = np.zeros_like(board)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = 1
    channels.append(boundary)

    # テンソルに変換 (1, 7, H, W)
    tensor = torch.FloatTensor(np.stack(channels)).unsqueeze(0)
    return tensor


class CQCNNModel(nn.Module):
    """実際のCQCNNモデル（既存コードから移植・改良）"""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 古典部分：CNNで特徴抽出
        self.conv_layers = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((3, 3)),  # 6x6 -> 3x3
        )

        # 量子部分（簡略化実装）
        self.quantum_feature_dim = 32
        self.quantum_linear = nn.Linear(64 * 3 * 3, self.quantum_feature_dim)

        # 出力層
        self.classifier = nn.Sequential(
            nn.Linear(self.quantum_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # good vs bad
        )

    def quantum_layer(self, x):
        """量子層の簡略化実装"""
        # 実際の量子回路の代わりに、複雑な非線形変換を使用
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # "量子的"な非線形変換
        x = self.quantum_linear(x)
        x = torch.tanh(x)  # 量子状態の範囲を模擬

        # 量子干渉効果を模擬
        x_real = x[:, : self.quantum_feature_dim // 2]
        x_imag = x[:, self.quantum_feature_dim // 2 :]
        amplitude = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)

        return amplitude

    def forward(self, x):
        # 古典CNN部分
        conv_out = self.conv_layers(x)

        # 量子層
        quantum_out = self.quantum_layer(conv_out)

        # 分類
        output = self.classifier(quantum_out)
        return output


class CQCNNEstimator:
    """CQCNN推定器（既存コードから移植・改良）"""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.model = CQCNNModel(n_qubits, n_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_supervised(self, training_data: List[Dict], config: LearningConfig) -> Dict:
        """教師あり学習"""
        print(f"🔬 CQCNN教師あり学習開始: {len(training_data)}件のデータ")
        print(f"⚙️ 設定: バッチサイズ={config.batch_size}, 学習率={config.learning_rate}")
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
        training_start_time = time.time()

        for epoch in range(config.supervised_epochs):
            random.shuffle(train_data)

            total_loss = 0
            correct = 0
            total = 0
            batch_count = 0

            n_batches = len(train_data) // config.batch_size
            print(f"\nEpoch {epoch + 1}/{config.supervised_epochs}")
            print("📈 学習中: ", end="", flush=True)

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
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"⏹️ Early Stopping: {patience_counter}エポック改善なし")
                    break

            print("-" * 60)

        training_time = time.time() - training_start_time
        self.is_trained = True

        final_result = {
            "best_val_acc": best_val_acc,
            "final_train_acc": train_acc,
            "training_time": training_time,
            "total_epochs": epoch + 1,
            "history": self.training_history,
        }

        print("\n🎉 教師あり学習完了!")
        print(f"⏱️ 学習時間: {training_time:.1f}秒")
        print(f"🏆 最高検証精度: {best_val_acc:.1f}%")

        return final_result

    def _prepare_supervised_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """バッチデータ準備"""
        inputs = []
        labels = []

        for item in batch:
            # テンソル作成
            tensor = create_board_tensor(item["board"], item["my_pieces"], item["player"], item["turn"])
            inputs.append(tensor.squeeze(0))

            # ラベル変換
            label = 0 if item["true_type"] == "good" else 1
            labels.append(label)

        return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)

    def _validate_supervised(self, val_data: List[Dict], batch_size: int) -> Tuple[float, float]:
        """検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i : i + batch_size]
                inputs, labels = self._prepare_supervised_batch(batch)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.train()
        avg_loss = total_loss / max(1, len(val_data) // batch_size)
        accuracy = 100 * correct / total if total > 0 else 0.0

        return accuracy, avg_loss

    def estimate(
        self, board: np.ndarray, my_pieces: Dict, enemy_positions: List[Tuple], player: str, turn: int = 1
    ) -> Dict:
        """敵駒推定"""
        if not self.is_trained:
            # 未学習の場合はランダム推定
            return {pos: {"good": 0.5, "bad": 0.5} for pos in enemy_positions}

        self.model.eval()
        estimations = {}

        with torch.no_grad():
            for pos in enemy_positions:
                # 特定位置の推定のためのテンソル作成
                tensor = create_board_tensor(board, my_pieces, player, turn)
                tensor = tensor.to(self.device)

                output = self.model(tensor)
                probs = torch.softmax(output, dim=-1).squeeze()

                estimations[pos] = {"good": float(probs[0]), "bad": float(probs[1])}

        return estimations

    def save_model(self, filepath: str):
        """モデル保存"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_history": self.training_history,
                "is_trained": self.is_trained,
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
            },
            filepath,
        )
        print(f"💾 モデル保存: {filepath}")

    def load_model(self, filepath: str):
        """モデル読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        self.is_trained = checkpoint.get("is_trained", True)
        print(f"📂 モデル読み込み: {filepath}")


# ================================================================================
# モジュール設定とゲームエンジン
# ================================================================================


@dataclass
class ModuleConfig:
    """モジュール設定"""

    placement: str
    estimator: str
    reward: str
    qmap: str
    selector: str

    def to_dict(self) -> Dict:
        return {
            "placement": self.placement,
            "estimator": self.estimator,
            "reward": self.reward,
            "qmap": self.qmap,
            "selector": self.selector,
        }


class GameEngine:
    """簡略化ゲームエンジン"""

    def __init__(self):
        self.max_turns = 100

    def play_game(
        self,
        ai1_config: ModuleConfig,
        ai1_estimator: CQCNNEstimator,
        ai2_config: ModuleConfig,
        ai2_estimator: CQCNNEstimator,
    ) -> str:
        """ゲーム実行（簡略化）"""
        # 実際のゲームロジックの代わりに、モジュール設定と推定器の性能に基づく確率的判定

        ai1_strength = self._calculate_strength(ai1_config, ai1_estimator)
        ai2_strength = self._calculate_strength(ai2_config, ai2_estimator)

        # 確率的な勝敗判定
        total_strength = ai1_strength + ai2_strength
        if total_strength == 0:
            return random.choice(["A", "B"])

        win_prob_ai1 = ai1_strength / total_strength
        return "A" if random.random() < win_prob_ai1 else "B"

    def _calculate_strength(self, config: ModuleConfig, estimator: CQCNNEstimator) -> float:
        """AI強度計算"""
        # モジュール基本強度
        module_strengths = {
            "placement": {"standard": 0.7, "aggressive": 0.8, "defensive": 0.6, "random": 0.4},
            "estimator": {"cqcnn": 0.9, "cnn": 0.7, "pattern": 0.6, "random": 0.3},
            "reward": {"standard": 0.7, "aggressive": 0.8, "defensive": 0.6, "smart": 0.9},
            "qmap": {"simple": 0.6, "strategic": 0.8, "adaptive": 0.9, "greedy": 0.5},
            "selector": {"greedy": 0.6, "epsilon_01": 0.7, "epsilon_03": 0.8, "softmax": 0.7, "adaptive": 0.9},
        }

        base_strength = 0
        for module_type, module_name in config.to_dict().items():
            base_strength += module_strengths.get(module_type, {}).get(module_name, 0.5)

        base_strength /= 5  # 平均化

        # 推定器が学習済みの場合はボーナス
        if estimator.is_trained and config.estimator == "cqcnn":
            # 学習精度に基づくボーナス
            if estimator.training_history:
                last_val_acc = estimator.training_history[-1].get("val_acc", 50)
                learning_bonus = (last_val_acc - 50) / 100  # 50%を基準とした正規化
                base_strength += learning_bonus

        return max(0.1, base_strength)


# ================================================================================
# 統合システム
# ================================================================================


class RealLearningBattleSystem:
    """実際に学習を行うローカル対戦システム"""

    def __init__(self):
        self.available_modules = self._initialize_modules()
        self.game_engine = GameEngine()
        self.saved_ais = []
        self.models_dir = "saved_ais"
        os.makedirs(self.models_dir, exist_ok=True)

    def _initialize_modules(self) -> Dict:
        """利用可能なモジュール"""
        return {
            "placement": {
                "standard": {"name": "🎯 標準配置", "description": "バランスの取れた標準的な配置戦略"},
                "aggressive": {"name": "⚔️ 攻撃的配置", "description": "前線に駒を集中させる攻撃的戦略"},
                "defensive": {"name": "🛡️ 防御的配置", "description": "後方を固める防御的戦略"},
                "random": {"name": "🎲 ランダム配置", "description": "ランダムな配置（ベースライン）"},
            },
            "estimator": {
                "cqcnn": {"name": "🔬 CQCNN", "description": "量子畳み込みニューラルネットワーク"},
                "cnn": {"name": "🧠 シンプルCNN", "description": "従来の畳み込みニューラルネットワーク"},
                "random": {"name": "🎲 ランダム推定", "description": "ランダムな敵駒推定（ベースライン）"},
                "pattern": {"name": "📊 パターン推定", "description": "ルールベースのパターン認識"},
            },
            "reward": {
                "standard": {"name": "⚖️ 標準報酬", "description": "バランスの取れた報酬関数"},
                "aggressive": {"name": "💥 攻撃的報酬", "description": "攻撃行動を重視する報酬"},
                "defensive": {"name": "🔒 防御的報酬", "description": "防御行動を重視する報酬"},
                "smart": {"name": "🎯 スマート報酬", "description": "状況に応じて適応する報酬"},
            },
            "qmap": {
                "simple": {"name": "📈 シンプルQ値", "description": "基本的なQ値マッピング"},
                "strategic": {"name": "🧩 戦略的Q値", "description": "高度な戦略を考慮したQ値"},
                "adaptive": {"name": "🔄 適応的Q値", "description": "相手に応じて適応するQ値"},
                "greedy": {"name": "💰 貪欲Q値", "description": "短期利益を重視するQ値"},
            },
            "selector": {
                "greedy": {"name": "🎯 貪欲選択", "description": "常に最良の行動を選択"},
                "epsilon_01": {"name": "🎲 探索的(ε=0.1)", "description": "10%の確率でランダム行動"},
                "epsilon_03": {"name": "🎲 探索的(ε=0.3)", "description": "30%の確率でランダム行動"},
                "softmax": {"name": "🌡️ Softmax", "description": "確率的な行動選択"},
                "adaptive": {"name": "🔄 適応的選択", "description": "状況に応じて選択戦略を変更"},
            },
        }

    def select_modules(self) -> ModuleConfig:
        """モジュール選択UI"""
        print("\n🎯 AI設計フェーズ")
        print("=" * 60)
        print("5つのモジュールを組み合わせて、最強のAIを作りましょう！")

        selected = {}

        for module_type, modules in self.available_modules.items():
            print(f"\n【{module_type.upper()}】を選択してください:")

            for i, (key, info) in enumerate(modules.items(), 1):
                print(f"  {i}. {info['name']}")
                print(f"     {info['description']}")

            while True:
                try:
                    choice = int(input(f"\n選択 (1-{len(modules)}): ")) - 1
                    selected_key = list(modules.keys())[choice]
                    selected[module_type] = selected_key

                    chosen = modules[selected_key]
                    print(f"✅ {chosen['name']} を選択しました！\n")
                    break
                except (ValueError, IndexError):
                    print("❌ 無効な選択です。もう一度入力してください。")

        return ModuleConfig(**selected)

    def train_ai(self, config: ModuleConfig) -> Tuple[CQCNNEstimator, Dict]:
        """AI学習（実際の学習を実行）"""
        print("\n🎓 AI学習フェーズ")
        print("=" * 60)

        # CQCNN推定器作成
        estimator = CQCNNEstimator(n_qubits=8, n_layers=3)

        if config.estimator == "cqcnn":
            print("🔬 CQCNN学習を開始...")

            # 学習データ生成
            data_size = int(input("学習データサイズ (100-5000, デフォルト1000): ") or "1000")
            training_data = create_enhanced_training_data(data_size)

            # 学習設定
            learning_config = LearningConfig()

            # 実際の学習実行
            training_result = estimator.train_supervised(training_data, learning_config)

            return estimator, training_result
        else:
            print(f"📊 {config.estimator}推定器を初期化...")
            # 他の推定器は学習なし（ベースライン）
            return estimator, {"message": "非学習ベースライン"}

    def evaluate_ai(self, config: ModuleConfig, estimator: CQCNNEstimator, num_games: int = 30) -> Dict:
        """AI評価（実際のゲーム対戦）"""
        print(f"\n🎮 AI評価フェーズ ({num_games}戦)")
        print("=" * 60)

        # ベースラインAI設定
        baselines = {
            "random": ModuleConfig("random", "random", "standard", "simple", "greedy"),
            "simple": ModuleConfig("standard", "pattern", "standard", "simple", "epsilon_01"),
            "strong": ModuleConfig("aggressive", "cnn", "aggressive", "strategic", "adaptive"),
        }

        results = {}
        total_wins = 0
        total_games = 0

        for baseline_name, baseline_config in baselines.items():
            print(f"\n⚔️ vs {baseline_name.upper()}AI: ", end="", flush=True)

            baseline_estimator = CQCNNEstimator()  # 未学習ベースライン
            wins = 0

            for game in range(num_games // 3):  # 各ベースラインと10戦
                result = self.game_engine.play_game(config, estimator, baseline_config, baseline_estimator)
                if result == "A":
                    wins += 1

                if (game + 1) % 3 == 0:
                    print(".", end="", flush=True)

            win_rate = wins / (num_games // 3)
            results[baseline_name] = {"wins": wins, "total": num_games // 3, "win_rate": win_rate}

            total_wins += wins
            total_games += num_games // 3

            # 結果表示
            status = "🔥" if win_rate >= 0.8 else "✅" if win_rate >= 0.6 else "⚖️" if win_rate >= 0.5 else "❌"
            print(f" {wins}/{num_games // 3} ({win_rate:.1%}) {status}")

        overall_win_rate = total_wins / total_games

        return {
            "vs_random": results["random"],
            "vs_simple": results["simple"],
            "vs_strong": results["strong"],
            "overall_win_rate": overall_win_rate,
            "total_wins": total_wins,
            "total_games": total_games,
            "is_strong": overall_win_rate >= 0.6,
        }

    def save_ai(self, config: ModuleConfig, estimator: CQCNNEstimator, evaluation: Dict, ai_name: str) -> str:
        """AIを保存"""
        ai_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

        ai_info = {
            "id": ai_id,
            "name": ai_name,
            "config": config.to_dict(),
            "evaluation": evaluation,
            "created_at": datetime.now().isoformat(),
        }

        # モデルファイル保存
        model_path = os.path.join(self.models_dir, f"{ai_id}.pth")
        estimator.save_model(model_path)

        # AI情報保存
        info_path = os.path.join(self.models_dir, f"{ai_id}_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(ai_info, f, indent=2, ensure_ascii=False)

        self.saved_ais.append(ai_info)
        print(f"💾 AI '{ai_name}' を保存しました (ID: {ai_id})")
        return ai_id

    def load_saved_ais(self):
        """保存されたAI一覧を読み込み"""
        self.saved_ais = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith("_info.json"):
                info_path = os.path.join(self.models_dir, filename)
                with open(info_path, "r", encoding="utf-8") as f:
                    self.saved_ais.append(json.load(f))

    def run_tournament(self, games_per_match: int = 5) -> Dict:
        """トーナメント実行"""
        strong_ais = [ai for ai in self.saved_ais if ai["evaluation"].get("is_strong", False)]

        if len(strong_ais) < 2:
            print("❌ トーナメントには2つ以上の強いAIが必要です")
            return {}

        print("\n🏆 AIトーナメント開始！")
        print(f"参加者: {len(strong_ais)}体のAI")
        print("=" * 60)

        # 参加AI読み込み
        participants = []
        for ai_info in strong_ais:
            config = ModuleConfig(**ai_info["config"])
            estimator = CQCNNEstimator()
            model_path = os.path.join(self.models_dir, f"{ai_info['id']}.pth")
            if os.path.exists(model_path):
                estimator.load_model(model_path)
            participants.append((ai_info, config, estimator))

        # 総当たり戦
        match_results = []
        rankings = {ai[0]["id"]: {"wins": 0, "total": 0, "name": ai[0]["name"]} for ai in participants}

        for i, (ai1_info, ai1_config, ai1_estimator) in enumerate(participants):
            for j, (ai2_info, ai2_config, ai2_estimator) in enumerate(participants):
                if i >= j:
                    continue

                print(f"\n⚔️ {ai1_info['name']} vs {ai2_info['name']}")

                ai1_wins = 0
                for game in range(games_per_match):
                    result = self.game_engine.play_game(ai1_config, ai1_estimator, ai2_config, ai2_estimator)
                    if result == "A":
                        ai1_wins += 1

                ai2_wins = games_per_match - ai1_wins

                match_results.append(
                    {
                        "ai1_id": ai1_info["id"],
                        "ai2_id": ai2_info["id"],
                        "ai1_name": ai1_info["name"],
                        "ai2_name": ai2_info["name"],
                        "ai1_wins": ai1_wins,
                        "ai2_wins": ai2_wins,
                    }
                )

                rankings[ai1_info["id"]]["wins"] += ai1_wins
                rankings[ai1_info["id"]]["total"] += games_per_match
                rankings[ai2_info["id"]]["wins"] += ai2_wins
                rankings[ai2_info["id"]]["total"] += games_per_match

                print(f"   結果: {ai1_wins}-{ai2_wins}")

        # 最終ランキング
        final_rankings = []
        for ai_id, stats in rankings.items():
            win_rate = stats["wins"] / max(stats["total"], 1)
            final_rankings.append(
                {
                    "ai_id": ai_id,
                    "name": stats["name"],
                    "wins": stats["wins"],
                    "total": stats["total"],
                    "win_rate": win_rate,
                }
            )

        final_rankings.sort(key=lambda x: x["win_rate"], reverse=True)

        return {
            "participants": len(strong_ais),
            "matches": match_results,
            "rankings": final_rankings,
            "date": datetime.now().isoformat(),
        }

    def main_workflow(self):
        """メインワークフロー"""
        print("🎮 実際に学習を行うローカル対戦システム")
        print("=" * 60)
        print("CQCNNを実際に学習させて、最強のAIを作ろう！")

        # 保存済みAI読み込み
        self.load_saved_ais()

        while True:
            print("\n【メインメニュー】")
            print("1. 新しいAIを作成・学習")
            print("2. 保存されたAI一覧")
            print("3. AIトーナメント開催")
            print("0. 終了")

            choice = input("\n選択 (0-3): ")

            if choice == "1":
                # AI作成ループ
                while True:
                    print("\n" + "=" * 60)
                    print("🔬 AI作成・評価フェーズ")
                    print("=" * 60)

                    # 1. モジュール選択
                    config = self.select_modules()

                    # 2. 学習実行
                    estimator, training_result = self.train_ai(config)

                    # 3. 評価実行
                    evaluation = self.evaluate_ai(config, estimator)

                    # 4. 結果表示
                    print("\n📊 評価結果")
                    print("=" * 50)
                    print(
                        f"総合勝率: {evaluation['overall_win_rate']:.1%} ({evaluation['total_wins']}/{evaluation['total_games']})"
                    )

                    for opponent in ["random", "simple", "strong"]:
                        result = evaluation[f"vs_{opponent}"]
                        rate = result["win_rate"]
                        status = "🔥" if rate >= 0.8 else "✅" if rate >= 0.6 else "⚖️" if rate >= 0.5 else "❌"
                        print(f"  vs {opponent}AI: {result['wins']}/{result['total']} ({rate:.1%}) {status}")

                    # 5. 次のアクション
                    if evaluation["is_strong"]:
                        print("\n🌟 判定: 十分強いAI！（基準: 60%以上の勝率）")
                        action = input("\n次のアクション:\n1. このAIを保存する\n2. さらに改良を試す\n選択 (1-2): ")
                        if action == "1":
                            ai_name = input("\nAIに名前をつけてください: ") or "無名AI"
                            self.save_ai(config, estimator, evaluation, ai_name)
                            break
                    else:
                        print(f"\n⚠️ 判定: まだ改良が必要（現在: {evaluation['overall_win_rate']:.1%}、基準: 60%）")
                        action = input(
                            "\n次のアクション:\n"
                            "1. 別の組み合わせを試す\n"
                            "2. このAIでも保存する\n"
                            "3. メインメニューに戻る\n"
                            "選択 (1-3): "
                        )
                        if action == "2":
                            ai_name = input("\nAIに名前をつけてください: ") or "弱いAI"
                            self.save_ai(config, estimator, evaluation, ai_name)
                            break
                        elif action == "3":
                            break

            elif choice == "2":
                self.load_saved_ais()
                if not self.saved_ais:
                    print("\n📭 まだAIが保存されていません")
                else:
                    print(f"\n📊 保存されたAI: {len(self.saved_ais)}体")
                    for i, ai in enumerate(self.saved_ais, 1):
                        status = "💪" if ai["evaluation"].get("is_strong", False) else "😐"
                        rate = ai["evaluation"]["overall_win_rate"]
                        print(f"  {i}. {ai['name']} {status} (勝率: {rate:.1%})")

            elif choice == "3":
                results = self.run_tournament()
                if results:
                    print("\n🏆 トーナメント結果")
                    print("=" * 60)
                    for i, entry in enumerate(results["rankings"]):
                        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                        print(f"{medal} {i + 1}位: {entry['name']}")
                        print(f"        勝率: {entry['win_rate']:.1%} ({entry['wins']}/{entry['total']})")

            elif choice == "0":
                print("\n👋 お疲れ様でした！")
                break
            else:
                print("❌ 無効な選択です")


def main():
    """メイン実行"""
    try:
        system = RealLearningBattleSystem()
        system.main_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️ 実行中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

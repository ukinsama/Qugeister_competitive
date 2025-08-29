#!/usr/bin/env python3
"""
分離型システム:
1. CQCNNによる敵駒タイプ推定
2. 推定結果からQ値マップ生成
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
import json


# ================================================================================
# Part 1: CQCNNによる敵駒推定モジュール
# ================================================================================


@dataclass
class PieceEstimation:
    """駒推定結果を格納するデータクラス"""

    position: Tuple[int, int]
    good_probability: float  # 善玉である確率
    bad_probability: float  # 悪玉である確率
    confidence: float  # 推定の確信度

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "position": self.position,
            "good_prob": self.good_probability,
            "bad_prob": self.bad_probability,
            "confidence": self.confidence,
        }


class CQCNNPieceEstimator(nn.Module):
    """敵駒タイプ推定専用CQCNN"""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 特徴抽出層
        self.feature_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3ch: 自駒/敵駒/空
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6x6 -> 3x3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 量子回路パラメータ（シミュレーション用）
        self.quantum_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        # 駒タイプ推定ヘッド
        self.piece_type_head = nn.Sequential(
            nn.Linear(64 * 3 * 3 + n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [good_prob, bad_prob]
        )

        # 確信度推定ヘッド
        self.confidence_head = nn.Sequential(
            nn.Linear(64 * 3 * 3 + n_qubits, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def quantum_circuit_simulation(self, x: torch.Tensor) -> torch.Tensor:
        """量子回路のシミュレーション"""
        batch_size = x.shape[0]

        # 簡易的な量子計算シミュレーション
        q_state = torch.zeros(batch_size, self.n_qubits)

        for layer in range(self.n_layers):
            # 回転ゲート
            for q in range(self.n_qubits):
                rotation = self.quantum_params[layer, q]
                q_state[:, q] = torch.sin(rotation[0] * x[:, q % x.shape[1]]) * torch.cos(
                    rotation[1] * x[:, (q + 1) % x.shape[1]]
                ) + torch.tanh(rotation[2] * q_state[:, q])

        return q_state

    def forward(self, board_state: torch.Tensor, target_positions: List[Tuple[int, int]]) -> List[PieceEstimation]:
        """
        敵駒タイプを推定

        Args:
            board_state: ボード状態 (batch_size, 3, 6, 6)
            target_positions: 推定対象の敵駒位置リスト

        Returns:
            各敵駒の推定結果リスト
        """
        batch_size = board_state.shape[0]

        # CNN特徴抽出
        conv_features = self.feature_conv(board_state)
        conv_flat = conv_features.view(batch_size, -1)

        # 量子特徴抽出
        board_flat = board_state.view(batch_size, -1)
        quantum_features = self.quantum_circuit_simulation(board_flat[:, : self.n_qubits])

        # 特徴結合
        combined_features = torch.cat([conv_flat, quantum_features], dim=1)

        # 各駒位置に対する推定
        estimations = []
        for pos in target_positions:
            # 位置特有の特徴を追加（オプション）
            pos_encoding = torch.tensor([pos[0] / 5.0, pos[1] / 5.0], dtype=torch.float32)
            pos_encoding = pos_encoding.unsqueeze(0).repeat(batch_size, 1)

            # タイプ推定
            type_logits = self.piece_type_head(combined_features)
            type_probs = F.softmax(type_logits, dim=1)

            # 確信度推定
            confidence = self.confidence_head(combined_features)

            # バッチ平均を取得（単一推定の場合）
            if batch_size == 1:
                estimation = PieceEstimation(
                    position=pos,
                    good_probability=type_probs[0, 0].item(),
                    bad_probability=type_probs[0, 1].item(),
                    confidence=confidence[0, 0].item(),
                )
            else:
                # バッチ全体の平均
                estimation = PieceEstimation(
                    position=pos,
                    good_probability=type_probs[:, 0].mean().item(),
                    bad_probability=type_probs[:, 1].mean().item(),
                    confidence=confidence.mean().item(),
                )

            estimations.append(estimation)

        return estimations


class PieceEstimationDataExporter:
    """駒推定データのエクスポート管理"""

    def __init__(self, estimator: CQCNNPieceEstimator):
        self.estimator = estimator
        self.estimation_history = []

    def estimate_and_export(self, game_state: dict) -> Dict[str, Any]:
        """
        ゲーム状態から駒推定を実行し、結果をエクスポート

        Args:
            game_state: ゲーム状態の辞書
                - board: 6x6のボード配列
                - current_player: 現在のプレイヤー
                - turn: ターン数

        Returns:
            エクスポート用データ辞書
        """
        board = np.array(game_state["board"])
        current_player = game_state["current_player"]
        turn = game_state.get("turn", 0)

        # ボード状態をテンソルに変換
        board_tensor = self._prepare_board_tensor(board, current_player)

        # 敵駒位置を特定
        enemy_positions = self._find_enemy_positions(board, current_player)

        if not enemy_positions:
            return {"turn": turn, "player": current_player, "estimations": [], "summary": {"total_enemies": 0}}

        # 推定実行
        with torch.no_grad():
            estimations = self.estimator(board_tensor, enemy_positions)

        # エクスポート用データ作成
        export_data = {
            "turn": turn,
            "player": current_player,
            "estimations": [est.to_dict() for est in estimations],
            "summary": self._create_summary(estimations),
            "board_state": board.tolist(),
        }

        # 履歴に追加
        self.estimation_history.append(export_data)

        return export_data

    def _prepare_board_tensor(self, board: np.ndarray, current_player: str) -> torch.Tensor:
        """ボード状態を3チャンネルテンソルに変換"""
        tensor = torch.zeros(1, 3, 6, 6)  # batch_size=1

        player_val = 1 if current_player == "A" else -1
        enemy_val = -player_val

        # チャンネル0: 自駒
        tensor[0, 0] = torch.from_numpy((board == player_val).astype(np.float32))
        # チャンネル1: 敵駒
        tensor[0, 1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        # チャンネル2: 空マス
        tensor[0, 2] = torch.from_numpy((board == 0).astype(np.float32))

        return tensor

    def _find_enemy_positions(self, board: np.ndarray, current_player: str) -> List[Tuple[int, int]]:
        """敵駒の位置を特定"""
        enemy_val = -1 if current_player == "A" else 1
        positions = []

        for y in range(6):
            for x in range(6):
                if board[y, x] == enemy_val:
                    positions.append((x, y))

        return positions

    def _create_summary(self, estimations: List[PieceEstimation]) -> dict:
        """推定結果のサマリー作成"""
        if not estimations:
            return {"total_enemies": 0}

        good_count = sum(1 for e in estimations if e.good_probability > 0.5)
        bad_count = sum(1 for e in estimations if e.bad_probability > 0.5)
        avg_confidence = np.mean([e.confidence for e in estimations])

        return {
            "total_enemies": len(estimations),
            "estimated_good": good_count,
            "estimated_bad": bad_count,
            "average_confidence": float(avg_confidence),
            "high_confidence_count": sum(1 for e in estimations if e.confidence > 0.7),
        }

    def save_to_file(self, filepath: str):
        """推定履歴をファイルに保存"""
        with open(filepath, "w") as f:
            json.dump(self.estimation_history, f, indent=2)
        print(f"📁 推定データを保存: {filepath}")


# ================================================================================
# Part 2: Q値マップ生成モジュール
# ================================================================================


class QValueMapGenerator:
    """駒推定結果からQ値マップを生成"""

    def __init__(self, board_size: int = 6):
        self.board_size = board_size

        # Q値計算用の重み
        self.weights = {
            "capture_good": 10.0,  # 善玉捕獲の価値
            "capture_bad": -5.0,  # 悪玉捕獲のペナルティ
            "escape_progress": 8.0,  # 脱出への前進
            "center_control": 2.0,  # 中央制御
            "safety": 3.0,  # 安全性
            "threat": 4.0,  # 脅威度
            "uncertainty_penalty": -1.0,  # 不確実性ペナルティ
        }

    def generate_q_map(
        self,
        board_state: np.ndarray,
        piece_estimations: List[Dict],
        current_player: str,
        my_pieces: Dict[Tuple[int, int], str],
    ) -> np.ndarray:
        """
        駒推定結果からQ値マップを生成

        Args:
            board_state: 現在のボード状態
            piece_estimations: 駒推定結果のリスト
            current_player: 現在のプレイヤー
            my_pieces: 自分の駒情報 {位置: タイプ}

        Returns:
            Q値マップ (6, 6, 4) - 各位置の4方向への移動価値
        """
        q_map = np.zeros((self.board_size, self.board_size, 4))  # 4方向: 上右下左

        # 推定結果を位置ベースの辞書に変換
        estimations_dict = {}
        for est in piece_estimations:
            pos = tuple(est["position"])
            estimations_dict[pos] = est

        # 各自駒に対してQ値を計算
        for piece_pos, piece_type in my_pieces.items():
            x, y = piece_pos

            # 4方向の移動を評価
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左

            for dir_idx, (dx, dy) in enumerate(directions):
                new_x, new_y = x + dx, y + dy

                # 境界チェック
                if not (0 <= new_x < self.board_size and 0 <= new_y < self.board_size):
                    q_map[y, x, dir_idx] = -100  # 移動不可
                    continue

                # Q値計算
                q_value = self._calculate_position_value(
                    from_pos=(x, y),
                    to_pos=(new_x, new_y),
                    piece_type=piece_type,
                    board_state=board_state,
                    estimations_dict=estimations_dict,
                    current_player=current_player,
                )

                q_map[y, x, dir_idx] = q_value

        return q_map

    def _calculate_position_value(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        piece_type: str,
        board_state: np.ndarray,
        estimations_dict: Dict,
        current_player: str,
    ) -> float:
        """特定の移動に対するQ値を計算"""
        q_value = 0.0

        # 1. 駒捕獲の評価
        if to_pos in estimations_dict:
            est = estimations_dict[to_pos]
            good_prob = est["good_prob"]
            bad_prob = est["bad_prob"]
            confidence = est["confidence"]

            # 期待値計算
            capture_value = good_prob * self.weights["capture_good"] + bad_prob * self.weights["capture_bad"]

            # 確信度で重み付け
            q_value += capture_value * confidence

            # 不確実性ペナルティ
            if confidence < 0.5:
                q_value += self.weights["uncertainty_penalty"]

        # 2. 脱出への前進評価（善玉の場合）
        if piece_type == "good":
            escape_value = self._evaluate_escape_progress(from_pos, to_pos, current_player)
            q_value += escape_value * self.weights["escape_progress"]

        # 3. 位置価値評価
        position_value = self._evaluate_position(to_pos, current_player)
        q_value += position_value

        # 4. 安全性評価
        safety_value = self._evaluate_safety(to_pos, board_state, estimations_dict)
        q_value += safety_value * self.weights["safety"]

        # 5. 脅威創出評価
        threat_value = self._evaluate_threat_creation(to_pos, estimations_dict)
        q_value += threat_value * self.weights["threat"]

        return q_value

    def _evaluate_escape_progress(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], player: str) -> float:
        """脱出への前進度を評価"""
        if player == "A":
            # プレイヤーAは上方向（Y増加）が前進
            progress = (to_pos[1] - from_pos[1]) / 5.0
            # 脱出口への接近ボーナス
            if to_pos[1] == 5 and (to_pos[0] == 0 or to_pos[0] == 5):
                progress += 1.0
        else:
            # プレイヤーBは下方向（Y減少）が前進
            progress = (from_pos[1] - to_pos[1]) / 5.0
            # 脱出口への接近ボーナス
            if to_pos[1] == 0 and (to_pos[0] == 0 or to_pos[0] == 5):
                progress += 1.0

        return max(0, progress)

    def _evaluate_position(self, pos: Tuple[int, int], player: str) -> float:
        """位置の戦略的価値を評価"""
        x, y = pos
        value = 0.0

        # 中央制御ボーナス
        center_distance = abs(x - 2.5) + abs(y - 2.5)
        value += self.weights["center_control"] * (1.0 - center_distance / 5.0)

        # 前線ボーナス
        if player == "A":
            value += y / 5.0  # 上方向への進出
        else:
            value += (5 - y) / 5.0  # 下方向への進出

        return value

    def _evaluate_safety(self, pos: Tuple[int, int], board_state: np.ndarray, estimations_dict: Dict) -> float:
        """位置の安全性を評価"""
        x, y = pos
        danger = 0.0

        # 周囲の敵駒による脅威を評価
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                check_x, check_y = x + dx, y + dy
                if (check_x, check_y) in estimations_dict:
                    est = estimations_dict[(check_x, check_y)]
                    # 善玉の可能性が高い敵駒は脅威度が高い
                    threat_level = est["good_prob"] * est["confidence"]
                    distance = max(abs(dx), abs(dy))
                    danger += threat_level / distance

        # 安全性は脅威度の逆数
        safety = 1.0 / (1.0 + danger)
        return safety

    def _evaluate_threat_creation(self, pos: Tuple[int, int], estimations_dict: Dict) -> float:
        """脅威創出能力を評価"""
        x, y = pos
        threat = 0.0

        # 移動先から攻撃可能な敵駒を評価
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if abs(dx) + abs(dy) != 1:  # 4方向のみ
                    continue

                target_x, target_y = x + dx, y + dy
                if (target_x, target_y) in estimations_dict:
                    est = estimations_dict[(target_x, target_y)]
                    # 善玉の可能性が高い敵駒への脅威創出価値
                    threat += est["good_prob"] * est["confidence"]

        return threat

    def export_q_map(self, q_map: np.ndarray, filepath: str):
        """Q値マップをファイルにエクスポート"""
        export_data = {
            "shape": q_map.shape,
            "q_values": q_map.tolist(),
            "statistics": {
                "max_q": float(np.max(q_map)),
                "min_q": float(np.min(q_map)),
                "mean_q": float(np.mean(q_map)),
                "std_q": float(np.std(q_map)),
            },
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"📊 Q値マップを保存: {filepath}")


# ================================================================================
# Part 3: 統合実行システム
# ================================================================================


class IntegratedCQCNNSystem:
    """統合システム: 駒推定→Q値マップ生成"""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        # 駒推定器
        self.estimator = CQCNNPieceEstimator(n_qubits, n_layers)
        self.exporter = PieceEstimationDataExporter(self.estimator)

        # Q値マップ生成器
        self.q_generator = QValueMapGenerator()

        # 実行履歴
        self.execution_history = []

    def process_game_state(self, game_state: dict, my_pieces: dict) -> Tuple[Dict, np.ndarray]:
        """
        ゲーム状態を処理して駒推定とQ値マップを生成

        Args:
            game_state: ゲーム状態
            my_pieces: 自分の駒情報

        Returns:
            (駒推定結果, Q値マップ)
        """
        # Step 1: 駒推定実行
        estimation_data = self.exporter.estimate_and_export(game_state)

        # Step 2: Q値マップ生成
        q_map = self.q_generator.generate_q_map(
            board_state=np.array(game_state["board"]),
            piece_estimations=estimation_data["estimations"],
            current_player=game_state["current_player"],
            my_pieces=my_pieces,
        )

        # 履歴記録
        self.execution_history.append(
            {
                "turn": game_state.get("turn", 0),
                "estimation": estimation_data,
                "q_map_stats": {
                    "max_q": float(np.max(q_map)),
                    "min_q": float(np.min(q_map)),
                    "mean_q": float(np.mean(q_map)),
                },
            }
        )

        return estimation_data, q_map

    def save_all_data(self, base_path: str):
        """全データを保存"""
        import os

        os.makedirs(base_path, exist_ok=True)

        # 駒推定データ保存
        self.exporter.save_to_file(f"{base_path}/piece_estimations.json")

        # 実行履歴保存
        with open(f"{base_path}/execution_history.json", "w") as f:
            json.dump(self.execution_history, f, indent=2)

        print(f"✅ 全データを {base_path} に保存完了")


# ================================================================================
# デモ実行
# ================================================================================


def demo():
    """デモ実行"""
    print("🚀 分離型CQCNN駒推定＆Q値マップ生成デモ")
    print("=" * 60)

    # システム初期化
    system = IntegratedCQCNNSystem(n_qubits=8, n_layers=3)

    # サンプルゲーム状態
    game_state = {
        "board": [
            [1, 1, 0, 0, -1, -1],
            [1, 1, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, -1, -1],
        ],
        "current_player": "A",
        "turn": 10,
    }

    # 自分の駒情報
    my_pieces = {(0, 0): "good", (1, 0): "good", (0, 1): "bad", (1, 1): "bad", (0, 4): "good"}

    # 処理実行
    print("\n📊 駒推定実行中...")
    estimation_data, q_map = system.process_game_state(game_state, my_pieces)

    # 結果表示
    print("\n🎯 駒推定結果:")
    for est in estimation_data["estimations"]:
        print(
            f"  位置{est['position']}: 善玉確率={est['good_prob']:.2f}, "
            f"悪玉確率={est['bad_prob']:.2f}, 確信度={est['confidence']:.2f}"
        )

    print("\n📈 推定サマリー:")
    summary = estimation_data["summary"]
    print(f"  敵駒総数: {summary['total_enemies']}")
    print(f"  推定善玉: {summary['estimated_good']}")
    print(f"  推定悪玉: {summary['estimated_bad']}")
    print(f"  平均確信度: {summary['average_confidence']:.3f}")

    print("\n🗺️ Q値マップ統計:")
    print(f"  最大Q値: {np.max(q_map):.2f}")
    print(f"  最小Q値: {np.min(q_map):.2f}")
    print(f"  平均Q値: {np.mean(q_map):.2f}")

    # 最適手の提案
    best_q = -float("inf")
    best_move = None
    for piece_pos in my_pieces:
        x, y = piece_pos
        for dir_idx, (dx, dy) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            if q_map[y, x, dir_idx] > best_q:
                best_q = q_map[y, x, dir_idx]
                best_move = (piece_pos, (x + dx, y + dy))

    if best_move:
        print(f"\n💡 推奨手: {best_move[0]} → {best_move[1]} (Q値: {best_q:.2f})")

    # データ保存
    system.save_all_data("cqcnn_output")
    print("\n✅ デモ完了！")


if __name__ == "__main__":
    demo()

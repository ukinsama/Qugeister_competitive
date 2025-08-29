#!/usr/bin/env python3
"""
CQCNN GUI対戦システム - ビジュアル競技版
保存されたAIモデルを読み込んで、GUIで対戦を可視化

必要なパッケージ:
pip install pygame torch numpy
"""

import pygame
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import random
import os
import pickle
from datetime import datetime
import queue

# ================================================================================
# Part 1: 基本設定とカラー定義
# ================================================================================


# Pygameカラー定義
class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (50, 50, 50)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    CYAN = (0, 255, 255)

    # ゲーム特有の色
    GOOD_PIECE = (100, 200, 100)  # 善玉: 緑系
    BAD_PIECE = (200, 100, 100)  # 悪玉: 赤系
    UNKNOWN_PIECE = (150, 150, 150)  # 不明: グレー
    PLAYER_A = (100, 150, 255)  # プレイヤーA: 青系
    PLAYER_B = (255, 150, 100)  # プレイヤーB: オレンジ系
    ESCAPE_ZONE = (255, 255, 200)  # 脱出口: 薄黄色
    HIGHLIGHT = (255, 255, 0, 128)  # ハイライト: 半透明黄色


# ================================================================================
# Part 1.5: AIエージェントインターフェース
# ================================================================================


class BaseAgent(ABC):
    """エージェントの基底クラス"""

    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        self.last_estimations = {}

    @abstractmethod
    def get_move(self, game_state, legal_moves: List[Tuple]) -> Optional[Tuple]:
        """次の手を取得"""
        pass

    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        # デフォルトはランダム配置
        if self.player_id == "A":
            positions = [(1, 1), (1, 2), (1, 3), (1, 4), (0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            positions = [(4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4)]

        piece_types = ["good"] * 4 + ["bad"] * 4
        random.shuffle(piece_types)

        return dict(zip(positions, piece_types))


class RandomAgent(BaseAgent):
    """ランダムに手を選択するエージェント"""

    def __init__(self, player_id: str):
        super().__init__(player_id, f"RandomAgent_{player_id}")

    def get_move(self, game_state, legal_moves: List[Tuple]) -> Optional[Tuple]:
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class SimpleAgent(BaseAgent):
    """簡単な評価関数を持つエージェント"""

    def __init__(self, player_id: str):
        super().__init__(player_id, f"SimpleAgent_{player_id}")

    def get_move(self, game_state, legal_moves: List[Tuple]) -> Optional[Tuple]:
        if not legal_moves:
            return None

        best_move = None
        best_value = -float("inf")

        for move in legal_moves:
            value = self._evaluate_move(move, game_state)
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def _evaluate_move(self, move: Tuple, game_state) -> float:
        from_pos, to_pos = move
        value = 0.0

        # 前進を評価
        if self.player_id == "A":
            value += (to_pos[0] - from_pos[0]) * 0.5
        else:
            value += (from_pos[0] - to_pos[0]) * 0.5

        # 敵駒を取る場合
        player_val = 1 if self.player_id == "A" else -1
        enemy_val = -player_val
        if game_state.board[to_pos] == enemy_val:
            value += 2.0

        # 脱出口への距離
        if self.player_id == "A":
            escape_positions = [(5, 0), (5, 5)]
        else:
            escape_positions = [(0, 0), (0, 5)]

        min_dist = min(abs(to_pos[0] - ep[0]) + abs(to_pos[1] - ep[1]) for ep in escape_positions)
        value -= min_dist * 0.1

        return value + random.random() * 0.1


class CQCNNAgent(BaseAgent):
    """CQCNN（量子回路）を使用するエージェント"""

    def __init__(self, player_id: str, model_path: Optional[str] = None):
        super().__init__(player_id, f"CQCNNAgent_{player_id}")
        self.model = self._load_or_create_model(model_path)
        self.pieces_info = {}  # 自分の駒情報を保持

    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得（駒情報を記録）"""
        placement = super().get_initial_placement()
        self.pieces_info = placement.copy()  # 駒タイプを記録
        return placement

    def _load_or_create_model(self, model_path: Optional[str]):
        """モデルを読み込むか新規作成"""

        class DummyCQCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(36, 2)

            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))

        model = DummyCQCNN()

        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path))
                print(f"モデルを読み込みました: {model_path}")
            except Exception as e:
                print(f"モデルの読み込みに失敗: {model_path}, Error: {e}")

        return model

    def _prepare_board_tensor_with_pieces(self, board: np.ndarray, my_pieces: Dict, player: str) -> torch.Tensor:
        """駒タイプを含むボード状態をテンソルに変換（7チャンネル版）"""
        tensor = torch.zeros(1, 7, 6, 6)
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val

        # チャンネル0: 自分の善玉の位置
        for pos, piece_type in my_pieces.items():
            if piece_type == "good" and board[pos] == player_val:
                tensor[0, 0, pos[0], pos[1]] = 1.0

        # チャンネル1: 自分の悪玉の位置
        for pos, piece_type in my_pieces.items():
            if piece_type == "bad" and board[pos] == player_val:
                tensor[0, 1, pos[0], pos[1]] = 1.0

        # チャンネル2: 相手の駒の位置（種類不明）
        tensor[0, 2] = torch.from_numpy((board == enemy_val).astype(np.float32))

        # チャンネル3: 空きマス
        tensor[0, 3] = torch.from_numpy((board == 0).astype(np.float32))

        # チャンネル4: 自分の脱出口
        if player == "A":
            tensor[0, 4, 5, 0] = 1.0
            tensor[0, 4, 5, 5] = 1.0
        else:
            tensor[0, 4, 0, 0] = 1.0
            tensor[0, 4, 0, 5] = 1.0

        # チャンネル5: 相手の脱出口
        if player == "A":
            tensor[0, 5, 0, 0] = 1.0
            tensor[0, 5, 0, 5] = 1.0
        else:
            tensor[0, 5, 5, 0] = 1.0
            tensor[0, 5, 5, 5] = 1.0

        # チャンネル6: ターン進行度
        if hasattr(self, "current_turn"):
            tensor[0, 6, :, :] = self.current_turn / 100.0

        return tensor

    def get_move(self, game_state, legal_moves: List[Tuple]) -> Optional[Tuple]:
        if not legal_moves:
            return None

        # 自分の駒情報を更新（移動により変化）
        if self.player_id == "A":
            current_pieces = game_state.player_a_pieces
        else:
            current_pieces = game_state.player_b_pieces

        # 駒タイプ情報を保持しながら更新
        for pos in list(self.pieces_info.keys()):
            if pos not in current_pieces:
                del self.pieces_info[pos]

        for pos in current_pieces:
            if pos not in self.pieces_info:
                # 新しい位置の駒（移動してきた駒）
                # 元の駒タイプを推測（簡略化）
                self.pieces_info[pos] = current_pieces[pos]

        # 敵駒の推定
        enemy_val = -1 if self.player_id == "A" else 1
        self.last_estimations = {}

        for i in range(6):
            for j in range(6):
                if game_state.board[i, j] == enemy_val:
                    # CQCNNモデルで推定（現在はダミー値）
                    self.last_estimations[(i, j)] = {
                        "good_prob": random.random(),
                        "bad_prob": random.random(),
                        "confidence": random.random(),
                    }

        # 評価に基づいて手を選択
        best_move = None
        best_value = -float("inf")

        for move in legal_moves:
            value = self._evaluate_move_with_estimation(move, game_state)
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def _evaluate_move_with_estimation(self, move: Tuple, game_state) -> float:
        from_pos, to_pos = move
        value = 0.0

        # 自分の駒タイプを確認
        piece_type = self.pieces_info.get(from_pos, "unknown")

        if piece_type == "good":
            # 善玉：脱出を優先
            if self.player_id == "A":
                escape_positions = [(5, 0), (5, 5)]
            else:
                escape_positions = [(0, 0), (0, 5)]

            # 脱出口への距離
            min_dist_after = min(abs(to_pos[0] - ep[0]) + abs(to_pos[1] - ep[1]) for ep in escape_positions)
            value -= min_dist_after * 0.5  # 近いほど高評価

            # 脱出口到達で最高評価
            if to_pos in escape_positions:
                value += 10.0

            # 相手の悪玉を避ける
            if to_pos in self.last_estimations:
                est = self.last_estimations[to_pos]
                value -= est["bad_prob"] * 3.0

        elif piece_type == "bad":
            # 悪玉：敵を取る
            if to_pos in self.last_estimations:
                est = self.last_estimations[to_pos]
                # 相手の善玉を取ると高得点
                value += est["good_prob"] * 3.0
                # 相手の悪玉も取る価値あり
                value += est["bad_prob"] * 1.0

        # 基本的な前進評価
        if self.player_id == "A":
            value += (to_pos[0] - from_pos[0]) * 0.1
        else:
            value += (from_pos[0] - to_pos[0]) * 0.1

        return value + random.random() * 0.1


# ================================================================================
# Part 1.6: Pygame内蔵AI選択メニュー
# ================================================================================


class AISelectionMenu:
    """Pygame内蔵のAI選択メニュー"""

    def __init__(self, screen):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()

        # フォント
        pygame.font.init()
        self.font_title = pygame.font.Font(None, 48)
        self.font_large = pygame.font.Font(None, 36)
        self.font_normal = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # 利用可能なAI
        self.ai_options = [
            ("ランダムAI", "random"),
            ("シンプルAI", "simple"),
            ("CQCNN AI", "cqcnn"),
        ]

        # 保存済みモデルをスキャン
        self._scan_saved_models()

        # 選択状態
        self.selected_a = 0
        self.selected_b = 1
        self.selection_confirmed = False
        self.selected_agents = {"A": None, "B": None}

        # UIレイアウト
        self.title_y = 50
        self.player_a_y = 150
        self.player_b_y = 350
        self.button_y = 550

        # ボタン
        self.start_button = pygame.Rect(self.width // 2 - 150, self.button_y, 120, 50)
        self.cancel_button = pygame.Rect(self.width // 2 + 30, self.button_y, 120, 50)

    def _scan_saved_models(self):
        """保存済みモデルをスキャン"""
        if os.path.exists("saved_models"):
            for file in os.listdir("saved_models"):
                if file.endswith(".pth") or file.endswith(".pkl"):
                    self.ai_options.append((f"保存: {file[:20]}", f"saved:{file}"))

    def show(self) -> Dict[str, Any]:
        """選択メニューを表示"""
        clock = pygame.time.Clock()
        running = True

        while running and not self.selection_confirmed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_RETURN:
                        self._confirm_selection()
                        running = False
                    # Player A選択
                    elif event.key == pygame.K_a:
                        self.selected_a = (self.selected_a + 1) % len(self.ai_options)
                    elif event.key == pygame.K_q:
                        self.selected_a = (self.selected_a - 1) % len(self.ai_options)
                    # Player B選択
                    elif event.key == pygame.K_s:
                        self.selected_b = (self.selected_b + 1) % len(self.ai_options)
                    elif event.key == pygame.K_w:
                        self.selected_b = (self.selected_b - 1) % len(self.ai_options)
                    # プリセット
                    elif event.key == pygame.K_1:
                        self.selected_a = 0  # ランダム
                        self.selected_b = 0  # ランダム
                    elif event.key == pygame.K_2:
                        self.selected_a = 1  # シンプル
                        self.selected_b = 0  # ランダム
                    elif event.key == pygame.K_3:
                        self.selected_a = 2  # CQCNN
                        self.selected_b = 1  # シンプル

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    # ボタンクリック
                    if self.start_button.collidepoint(mouse_pos):
                        self._confirm_selection()
                        running = False
                    elif self.cancel_button.collidepoint(mouse_pos):
                        return None

                    # AI選択クリック
                    for i, (name, _) in enumerate(self.ai_options):
                        # Player A
                        rect_a = pygame.Rect(100, self.player_a_y + 50 + i * 30, 400, 25)
                        if rect_a.collidepoint(mouse_pos):
                            self.selected_a = i

                        # Player B
                        rect_b = pygame.Rect(self.width // 2 + 100, self.player_b_y + 50 + i * 30, 400, 25)
                        if rect_b.collidepoint(mouse_pos):
                            self.selected_b = i

            self._draw()
            clock.tick(30)

        if self.selection_confirmed:
            return self.selected_agents
        return None

    def _draw(self):
        """メニューを描画"""
        # 背景
        self.screen.fill(Colors.DARK_GRAY)

        # タイトル
        title = self.font_title.render("AI Selection", True, Colors.WHITE)
        title_rect = title.get_rect(center=(self.width // 2, self.title_y))
        self.screen.blit(title, title_rect)

        # Player A選択
        self._draw_player_selection("Player A (Blue)", self.player_a_y, self.selected_a, Colors.PLAYER_A, 100)

        # Player B選択
        self._draw_player_selection(
            "Player B (Orange)", self.player_b_y, self.selected_b, Colors.PLAYER_B, self.width // 2 + 100
        )

        # プリセット説明
        preset_y = 480
        preset_text = [
            "Presets: [1] Random vs Random  [2] Simple vs Random  [3] CQCNN vs Simple",
            "Controls: [Q/A] Select Player A  [W/S] Select Player B  [Enter] Start",
        ]
        for i, text in enumerate(preset_text):
            rendered = self.font_small.render(text, True, Colors.LIGHT_GRAY)
            rect = rendered.get_rect(center=(self.width // 2, preset_y + i * 25))
            self.screen.blit(rendered, rect)

        # ボタン
        pygame.draw.rect(self.screen, Colors.GREEN, self.start_button)
        pygame.draw.rect(self.screen, Colors.BLACK, self.start_button, 2)
        start_text = self.font_normal.render("Start", True, Colors.WHITE)
        start_rect = start_text.get_rect(center=self.start_button.center)
        self.screen.blit(start_text, start_rect)

        pygame.draw.rect(self.screen, Colors.RED, self.cancel_button)
        pygame.draw.rect(self.screen, Colors.BLACK, self.cancel_button, 2)
        cancel_text = self.font_normal.render("Cancel", True, Colors.WHITE)
        cancel_rect = cancel_text.get_rect(center=self.cancel_button.center)
        self.screen.blit(cancel_text, cancel_rect)

        pygame.display.flip()

    def _draw_player_selection(self, title: str, y: int, selected: int, color: tuple, x: int):
        """プレイヤー選択エリアを描画"""
        # タイトル
        title_text = self.font_large.render(title, True, color)
        self.screen.blit(title_text, (x, y))

        # 選択肢
        for i, (name, _) in enumerate(self.ai_options):
            y_pos = y + 50 + i * 30

            # 選択されている場合はハイライト
            if i == selected:
                pygame.draw.rect(self.screen, color, (x - 10, y_pos - 2, 420, 28), 2)
                text_color = color
            else:
                text_color = Colors.WHITE

            text = self.font_normal.render(name, True, text_color)
            self.screen.blit(text, (x, y_pos))

    def _confirm_selection(self):
        """選択を確定"""
        # Player A作成
        ai_type_a = self.ai_options[self.selected_a][1]
        if ai_type_a == "random":
            self.selected_agents["A"] = RandomAgent("A")
        elif ai_type_a == "simple":
            self.selected_agents["A"] = SimpleAgent("A")
        elif ai_type_a == "cqcnn":
            self.selected_agents["A"] = CQCNNAgent("A")
        elif ai_type_a.startswith("saved:"):
            model_file = ai_type_a.split(":", 1)[1]
            model_path = os.path.join("saved_models", model_file)
            self.selected_agents["A"] = CQCNNAgent("A", model_path)
        else:
            self.selected_agents["A"] = RandomAgent("A")

        # Player B作成
        ai_type_b = self.ai_options[self.selected_b][1]
        if ai_type_b == "random":
            self.selected_agents["B"] = RandomAgent("B")
        elif ai_type_b == "simple":
            self.selected_agents["B"] = SimpleAgent("B")
        elif ai_type_b == "cqcnn":
            self.selected_agents["B"] = CQCNNAgent("B")
        elif ai_type_b.startswith("saved:"):
            model_file = ai_type_b.split(":", 1)[1]
            model_path = os.path.join("saved_models", model_file)
            self.selected_agents["B"] = CQCNNAgent("B", model_path)
        else:
            self.selected_agents["B"] = RandomAgent("B")

        self.selection_confirmed = True


# ================================================================================
# Part 2: モデル保存・読み込みシステム
# (Part 1 duplicate removed - Colors class is already defined at line 28)
# ================================================================================


class ModelManager:
    """モデルの保存と読み込みを管理"""

    def __init__(self, base_dir: str = "saved_models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_agent(self, agent: Any, name: str) -> str:
        """エージェントを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        filepath = os.path.join(self.base_dir, filename)

        # エージェントの設定と学習済みパラメータを保存
        save_data = {
            "name": agent.name,
            "player_id": agent.player_id,
            "config": self._serialize_config(agent.config),
            "game_history": agent.game_history,
            "timestamp": timestamp,
        }

        # モデルのパラメータも保存（PyTorchモデルの場合）
        if hasattr(agent.config.estimator, "model"):
            model_path = os.path.join(self.base_dir, f"{name}_model_{timestamp}.pth")
            torch.save(agent.config.estimator.model.state_dict(), model_path)
            save_data["model_path"] = model_path

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        print(f"✅ エージェントを保存: {filepath}")
        return filepath

    def load_agent(self, filepath: str) -> Any:
        """エージェントを読み込み"""
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        # 設定を復元
        config = self._deserialize_config(save_data["config"])

        # モデルパラメータを復元
        if "model_path" in save_data and os.path.exists(save_data["model_path"]):
            if hasattr(config.estimator, "model"):
                config.estimator.model.load_state_dict(torch.load(save_data["model_path"]))
                print(f"✅ モデルパラメータを読み込み: {save_data['model_path']}")

        # エージェントを再構築
        from cqcnn_battle_system import ModularAgent  # 元のシステムから

        agent = ModularAgent(save_data["player_id"], config)
        agent.game_history = save_data["game_history"]

        print(f"✅ エージェントを読み込み: {save_data['name']}")
        return agent

    def list_saved_models(self) -> List[str]:
        """保存されたモデルのリストを取得"""
        models = []
        for file in os.listdir(self.base_dir):
            if file.endswith(".pkl"):
                models.append(os.path.join(self.base_dir, file))
        return sorted(models)

    def _serialize_config(self, config):
        """設定をシリアライズ可能な形式に変換"""
        return {
            "placement": config.placement.__class__.__name__,
            "estimator": config.estimator.__class__.__name__,
            "qmap_generator": config.qmap_generator.__class__.__name__,
            "action_selector": config.action_selector.__class__.__name__,
        }

    def _deserialize_config(self, data):
        """シリアライズされた設定を復元"""
        # 簡略化のため、クラス名から直接インスタンスを作成
        # 実際の実装では、クラスのレジストリを使用
        from cqcnn_battle_system import (
            AgentConfig,
            StandardPlacement,
            CQCNNEstimator,
            SimpleQMapGenerator,
            GreedySelector,
        )

        return AgentConfig(
            placement=StandardPlacement(),
            estimator=CQCNNEstimator(n_qubits=4, n_layers=2),
            qmap_generator=SimpleQMapGenerator(),
            action_selector=GreedySelector(),
        )


# ================================================================================
# Part 3: GUIゲームボード
# ================================================================================


class GameBoard:
    """ゲームボードのGUI表示"""

    def __init__(self, screen, x: int, y: int, size: int = 600):
        self.screen = screen
        self.x = x
        self.y = y
        self.size = size
        self.cell_size = size // 6
        self.board_state = np.zeros((6, 6), dtype=int)
        self.pieces = {"A": {}, "B": {}}
        self.selected_cell = None
        self.legal_moves = []
        self.last_move = None
        self.estimations = {}

        # フォント設定
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 30)
        self.font_large = pygame.font.Font(None, 40)

    def update_state(self, board_state: np.ndarray, player_a_pieces: Dict, player_b_pieces: Dict):
        """ボード状態を更新"""
        self.board_state = board_state.copy()
        self.pieces["A"] = player_a_pieces.copy()
        self.pieces["B"] = player_b_pieces.copy()

    def draw(self):
        """ボードを描画"""
        # 背景
        pygame.draw.rect(self.screen, Colors.WHITE, (self.x, self.y, self.size, self.size))

        # グリッド線
        for i in range(7):
            # 横線
            pygame.draw.line(
                self.screen,
                Colors.BLACK,
                (self.x, self.y + i * self.cell_size),
                (self.x + self.size, self.y + i * self.cell_size),
                2,
            )
            # 縦線
            pygame.draw.line(
                self.screen,
                Colors.BLACK,
                (self.x + i * self.cell_size, self.y),
                (self.x + i * self.cell_size, self.y + self.size),
                2,
            )

        # 脱出口をハイライト
        self._draw_escape_zones()

        # 最後の移動をハイライト
        if self.last_move:
            self._highlight_move(self.last_move)

        # 駒を描画
        self._draw_pieces()

        # 推定結果を表示
        if self.estimations:
            self._draw_estimations()

        # 座標ラベル
        self._draw_coordinates()

    def _draw_escape_zones(self):
        """脱出口を描画（ガイスターの正しい位置）"""
        # プレイヤーAの脱出口（相手陣地の左右上角）
        escape_zones_a = [
            (5, 0),  # 左上角（Aから見て）
            (5, 5),  # 右上角（Aから見て）
        ]

        # プレイヤーBの脱出口（相手陣地の左右下角）
        escape_zones_b = [
            (0, 0),  # 左下角（Bから見て）
            (0, 5),  # 右下角（Bから見て）
        ]

        # プレイヤーAの脱出口を青系で表示
        for row, col in escape_zones_a:
            x = self.x + col * self.cell_size
            y = self.y + row * self.cell_size

            # 薄い青で塗りつぶし
            s = pygame.Surface((self.cell_size, self.cell_size))
            s.set_alpha(100)
            s.fill((200, 200, 255))  # 薄い青
            self.screen.blit(s, (x, y))

            # "EXIT A"テキスト
            text = self.font_small.render("EXIT A", True, Colors.PLAYER_A)
            text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
            self.screen.blit(text, text_rect)

        # プレイヤーBの脱出口をオレンジ系で表示
        for row, col in escape_zones_b:
            x = self.x + col * self.cell_size
            y = self.y + row * self.cell_size

            # 薄いオレンジで塗りつぶし
            s = pygame.Surface((self.cell_size, self.cell_size))
            s.set_alpha(100)
            s.fill((255, 200, 200))  # 薄いオレンジ
            self.screen.blit(s, (x, y))

            # "EXIT B"テキスト
            text = self.font_small.render("EXIT B", True, Colors.PLAYER_B)
            text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
            self.screen.blit(text, text_rect)

    def _draw_pieces(self):
        """駒を描画"""
        for row in range(6):
            for col in range(6):
                if self.board_state[row, col] != 0:
                    self._draw_piece(row, col)

    def _draw_piece(self, row: int, col: int):
        """1つの駒を描画"""
        x = self.x + col * self.cell_size + self.cell_size // 2
        y = self.y + row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3

        # プレイヤーを判定
        if self.board_state[row, col] == 1:
            player = "A"
            color = Colors.PLAYER_A
        else:
            player = "B"
            color = Colors.PLAYER_B

        # 駒の種類を取得
        piece_type = None
        if (row, col) in self.pieces[player]:
            piece_type = self.pieces[player][(row, col)]

        # 外円（プレイヤー色）
        pygame.draw.circle(self.screen, color, (x, y), radius, 3)

        # 内円（駒タイプ色）
        if piece_type == "good":
            inner_color = Colors.GOOD_PIECE
            symbol = "G"
        elif piece_type == "bad":
            inner_color = Colors.BAD_PIECE
            symbol = "B"
        else:
            inner_color = Colors.UNKNOWN_PIECE
            symbol = "?"

        pygame.draw.circle(self.screen, inner_color, (x, y), radius - 5)

        # シンボルを描画
        text = self.font_medium.render(symbol, True, Colors.WHITE)
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)

    def _draw_estimations(self):
        """推定結果を表示"""
        for pos, estimation in self.estimations.items():
            row, col = pos
            x = self.x + col * self.cell_size
            y = self.y + row * self.cell_size

            # 推定確率を表示
            good_prob = estimation.get("good_prob", 0)
            bad_prob = estimation.get("bad_prob", 0)

            # 背景（半透明）
            s = pygame.Surface((self.cell_size, 20))
            s.set_alpha(180)
            s.fill(Colors.BLACK)
            self.screen.blit(s, (x, y + self.cell_size - 20))

            # テキスト
            text = f"G:{good_prob:.1%} B:{bad_prob:.1%}"
            rendered = self.font_small.render(text, True, Colors.WHITE)
            self.screen.blit(rendered, (x + 2, y + self.cell_size - 18))

    def _highlight_move(self, move):
        """移動をハイライト"""
        from_pos, to_pos = move

        # From位置（緑）
        from_x = self.x + from_pos[1] * self.cell_size
        from_y = self.y + from_pos[0] * self.cell_size
        pygame.draw.rect(self.screen, Colors.GREEN, (from_x, from_y, self.cell_size, self.cell_size), 4)

        # To位置（赤）
        to_x = self.x + to_pos[1] * self.cell_size
        to_y = self.y + to_pos[0] * self.cell_size
        pygame.draw.rect(self.screen, Colors.RED, (to_x, to_y, self.cell_size, self.cell_size), 4)

        # 矢印を描画
        self._draw_arrow(
            from_x + self.cell_size // 2,
            from_y + self.cell_size // 2,
            to_x + self.cell_size // 2,
            to_y + self.cell_size // 2,
        )

    def _draw_arrow(self, x1, y1, x2, y2):
        """矢印を描画"""
        pygame.draw.line(self.screen, Colors.YELLOW, (x1, y1), (x2, y2), 3)

        # 矢印の先端
        angle = np.arctan2(y2 - y1, x2 - x1)
        arrow_length = 15
        arrow_angle = np.pi / 6

        x3 = x2 - arrow_length * np.cos(angle - arrow_angle)
        y3 = y2 - arrow_length * np.sin(angle - arrow_angle)
        x4 = x2 - arrow_length * np.cos(angle + arrow_angle)
        y4 = y2 - arrow_length * np.sin(angle + arrow_angle)

        pygame.draw.polygon(self.screen, Colors.YELLOW, [(x2, y2), (x3, y3), (x4, y4)])

    def _draw_coordinates(self):
        """座標ラベルを描画"""
        for i in range(6):
            # 行番号
            text = self.font_small.render(str(i), True, Colors.BLACK)
            self.screen.blit(text, (self.x - 20, self.y + i * self.cell_size + self.cell_size // 2 - 10))

            # 列番号
            text = self.font_small.render(str(i), True, Colors.BLACK)
            self.screen.blit(text, (self.x + i * self.cell_size + self.cell_size // 2 - 5, self.y - 20))


# ================================================================================
# Part 4: 情報パネル
# ================================================================================


class InfoPanel:
    """ゲーム情報を表示するパネル"""

    def __init__(self, screen, x: int, y: int, width: int, height: int):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        pygame.font.init()
        self.font_title = pygame.font.Font(None, 36)
        self.font_normal = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)

        self.game_info = {
            "turn": 0,
            "current_player": "A",
            "agent1_name": "Agent 1",
            "agent2_name": "Agent 2",
            "status": "Ready",
            "winner": None,
        }

        self.agent1_stats = {"wins": 0, "losses": 0, "draws": 0}
        self.agent2_stats = {"wins": 0, "losses": 0, "draws": 0}

        self.log_messages = []
        self.max_log_messages = 10

    def update_info(self, **kwargs):
        """情報を更新"""
        self.game_info.update(kwargs)

    def add_log(self, message: str):
        """ログメッセージを追加"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages.pop(0)

    def draw(self):
        """パネルを描画"""
        # 背景
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(self.screen, Colors.BLACK, (self.x, self.y, self.width, self.height), 2)

        y_offset = self.y + 10

        # タイトル
        title = self.font_title.render("CQCNN Battle System", True, Colors.BLACK)
        self.screen.blit(title, (self.x + 10, y_offset))
        y_offset += 50

        # ゲーム情報
        self._draw_game_info(y_offset)
        y_offset += 120

        # エージェント情報
        self._draw_agent_info(y_offset)
        y_offset += 150

        # ログ
        self._draw_log(y_offset)

    def _draw_game_info(self, y_offset):
        """ゲーム情報を描画"""
        # ターン数
        text = self.font_normal.render(f"Turn: {self.game_info['turn']}", True, Colors.BLACK)
        self.screen.blit(text, (self.x + 10, y_offset))

        # 現在のプレイヤー
        current_color = Colors.PLAYER_A if self.game_info["current_player"] == "A" else Colors.PLAYER_B
        text = self.font_normal.render(f"Current: Player {self.game_info['current_player']}", True, current_color)
        self.screen.blit(text, (self.x + 10, y_offset + 30))

        # ステータス
        status_color = Colors.GREEN if self.game_info["status"] == "Playing" else Colors.ORANGE
        text = self.font_normal.render(f"Status: {self.game_info['status']}", True, status_color)
        self.screen.blit(text, (self.x + 10, y_offset + 60))

        # 勝者
        if self.game_info["winner"]:
            winner_color = Colors.PLAYER_A if self.game_info["winner"] == "A" else Colors.PLAYER_B
            text = self.font_normal.render(f"Winner: Player {self.game_info['winner']}", True, winner_color)
            self.screen.blit(text, (self.x + 10, y_offset + 90))

    def _draw_agent_info(self, y_offset):
        """エージェント情報を描画"""
        # Agent 1
        pygame.draw.rect(self.screen, Colors.PLAYER_A, (self.x + 10, y_offset, self.width - 20, 60), 2)

        name1 = (
            self.game_info["agent1_name"][:30] + "..."
            if len(self.game_info["agent1_name"]) > 30
            else self.game_info["agent1_name"]
        )
        text = self.font_small.render(f"A: {name1}", True, Colors.PLAYER_A)
        self.screen.blit(text, (self.x + 15, y_offset + 5))

        stats1 = f"W:{self.agent1_stats['wins']} L:{self.agent1_stats['losses']} D:{self.agent1_stats['draws']}"
        text = self.font_small.render(stats1, True, Colors.BLACK)
        self.screen.blit(text, (self.x + 15, y_offset + 30))

        # Agent 2
        pygame.draw.rect(self.screen, Colors.PLAYER_B, (self.x + 10, y_offset + 70, self.width - 20, 60), 2)

        name2 = (
            self.game_info["agent2_name"][:30] + "..."
            if len(self.game_info["agent2_name"]) > 30
            else self.game_info["agent2_name"]
        )
        text = self.font_small.render(f"B: {name2}", True, Colors.PLAYER_B)
        self.screen.blit(text, (self.x + 15, y_offset + 75))

        stats2 = f"W:{self.agent2_stats['wins']} L:{self.agent2_stats['losses']} D:{self.agent2_stats['draws']}"
        text = self.font_small.render(stats2, True, Colors.BLACK)
        self.screen.blit(text, (self.x + 15, y_offset + 100))

    def _draw_log(self, y_offset):
        """ログを描画"""
        # ログエリアの背景
        pygame.draw.rect(self.screen, Colors.WHITE, (self.x + 10, y_offset, self.width - 20, 200))
        pygame.draw.rect(self.screen, Colors.BLACK, (self.x + 10, y_offset, self.width - 20, 200), 1)

        # ログメッセージ
        for i, message in enumerate(self.log_messages):
            text = self.font_small.render(message, True, Colors.BLACK)
            self.screen.blit(text, (self.x + 15, y_offset + 5 + i * 20))


# ================================================================================
# Part 5: コントロールパネル
# ================================================================================


class ControlPanel:
    """ゲームコントロールパネル"""

    def __init__(self, screen, x: int, y: int, width: int, height: int):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.buttons = {
            "start": Button(x + 10, y + 10, 100, 40, "Start", Colors.GREEN),
            "pause": Button(x + 120, y + 10, 100, 40, "Pause", Colors.YELLOW),
            "reset": Button(x + 230, y + 10, 100, 40, "Reset", Colors.RED),
            "step": Button(x + 10, y + 60, 100, 40, "Step", Colors.CYAN),
            "fast": Button(x + 120, y + 60, 100, 40, "Fast", Colors.ORANGE),
            "slow": Button(x + 230, y + 60, 100, 40, "Slow", Colors.PURPLE),
        }

        self.speed = 1.0  # ゲーム速度
        self.is_paused = True
        self.step_mode = False

    def draw(self):
        """パネルを描画"""
        # 背景
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(self.screen, Colors.BLACK, (self.x, self.y, self.width, self.height), 2)

        # ボタンを描画
        for button in self.buttons.values():
            button.draw(self.screen)

        # 速度表示
        font = pygame.font.Font(None, 24)
        text = font.render(f"Speed: {self.speed:.1f}x", True, Colors.BLACK)
        self.screen.blit(text, (self.x + 10, self.y + 110))

    def handle_click(self, pos) -> Optional[str]:
        """クリック処理"""
        for name, button in self.buttons.items():
            if button.is_clicked(pos):
                return name
        return None


class Button:
    """シンプルなボタンクラス"""

    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: tuple):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.font = pygame.font.Font(None, 24)

    def draw(self, screen):
        """ボタンを描画"""
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, Colors.BLACK, self.rect, 2)

        text_surface = self.font.render(self.text, True, Colors.WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos) -> bool:
        """クリック判定"""
        return self.rect.collidepoint(pos)


# ================================================================================
# Part 6: メインGUIアプリケーション
# ================================================================================


class CQCNNBattleGUI:
    """メインGUIアプリケーション"""

    def __init__(self):
        pygame.init()

        # ウィンドウ設定
        self.width = 1400
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CQCNN Battle System - Visual Competition")

        # コンポーネント
        self.board = GameBoard(self.screen, 50, 50, 600)
        self.info_panel = InfoPanel(self.screen, 700, 50, 650, 500)
        self.control_panel = ControlPanel(self.screen, 700, 570, 650, 150)

        # ゲーム管理
        self.model_manager = ModelManager()
        self.game_engine = None
        self.agent1 = None
        self.agent2 = None
        self.current_player = "A"

        # タイミング制御
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.move_delay = 1000  # ミリ秒
        self.last_move_time = 0

        # 状態管理
        self.running = True
        self.game_running = False
        self.game_paused = True

        # スレッド通信用
        self.move_queue = queue.Queue()

        # AI選択ダイアログを表示
        self._select_agents()

    def run(self):
        """メインループ"""
        # エージェントが選択されていない場合は終了
        if not self.agent1 or not self.agent2:
            print("エージェントが選択されていません。終了します。")
            return

        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(self.fps)

        pygame.quit()

    def _select_agents(self):
        """AI選択メニューを表示"""
        menu = AISelectionMenu(self.screen)
        agents = menu.show()

        if agents:
            self.agent1 = agents["A"]
            self.agent2 = agents["B"]

            self.info_panel.update_info(agent1_name=self.agent1.name, agent2_name=self.agent2.name, status="Ready")

            self.info_panel.add_log(f"Player A: {self.agent1.name}を選択")
            self.info_panel.add_log(f"Player B: {self.agent2.name}を選択")
        else:
            # キャンセルされた場合はデフォルトエージェントを使用
            self._setup_demo_agents()

    def _setup_demo_agents(self):
        """デモ用のエージェントをセットアップ"""
        self.agent1 = RandomAgent("A")
        self.agent2 = SimpleAgent("B")

        self.info_panel.update_info(agent1_name=self.agent1.name, agent2_name=self.agent2.name, status="Ready")

        self.info_panel.add_log("デフォルトエージェントを使用")

    def _handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # コントロールパネルのクリック処理
                action = self.control_panel.handle_click(event.pos)
                if action:
                    self._handle_control_action(action)

            elif event.type == pygame.KEYDOWN:
                # キーボードショートカット
                if event.key == pygame.K_SPACE:
                    self._handle_control_action("pause" if not self.game_paused else "start")
                elif event.key == pygame.K_r:
                    self._handle_control_action("reset")
                elif event.key == pygame.K_s:
                    self._handle_control_action("step")
                elif event.key == pygame.K_c:
                    # Cキーで新しいAIを選択
                    self._change_agents()

    def _handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # コントロールパネルのクリック処理
                action = self.control_panel.handle_click(event.pos)
                if action:
                    self._handle_control_action(action)

            elif event.type == pygame.KEYDOWN:
                # キーボードショートカット
                if event.key == pygame.K_SPACE:
                    self._handle_control_action("pause" if not self.game_paused else "start")
                elif event.key == pygame.K_r:
                    self._handle_control_action("reset")
                elif event.key == pygame.K_s:
                    self._handle_control_action("step")

    def _handle_control_action(self, action: str):
        """コントロールアクション処理"""
        if action == "start":
            if not self.game_running:
                self._start_game()
            self.game_paused = False
            self.info_panel.add_log("ゲーム開始")

        elif action == "pause":
            self.game_paused = True
            self.info_panel.add_log("ゲーム一時停止")

        elif action == "reset":
            self._reset_game()
            self.info_panel.add_log("ゲームリセット")

        elif action == "step":
            if self.game_running:
                self._execute_one_move()
            self.info_panel.add_log("1手実行")

        elif action == "fast":
            self.control_panel.speed = min(5.0, self.control_panel.speed + 0.5)
            self.move_delay = int(1000 / self.control_panel.speed)
            self.info_panel.add_log(f"速度: {self.control_panel.speed}x")

        elif action == "slow":
            self.control_panel.speed = max(0.5, self.control_panel.speed - 0.5)
            self.move_delay = int(1000 / self.control_panel.speed)
            self.info_panel.add_log(f"速度: {self.control_panel.speed}x")

    def _start_game(self):
        """ゲームを開始"""
        self.game_engine = SimpleGameEngine()

        # エージェントの初期配置を取得
        placement_a = self.agent1.get_initial_placement()
        placement_b = self.agent2.get_initial_placement()

        # ゲームエンジンに配置を設定
        self.game_engine.initialize_game_with_placement(placement_a, placement_b)

        # ボード状態を更新
        self.board.update_state(
            self.game_engine.board, self.game_engine.player_a_pieces, self.game_engine.player_b_pieces
        )

        self.game_running = True
        self.current_player = "A"

        self.info_panel.update_info(turn=0, current_player="A", status="Playing")

    def _reset_game(self):
        """ゲームをリセット"""
        self.game_running = False
        self.game_paused = True

        if self.game_engine:
            self.game_engine.reset()

        self.board.board_state = np.zeros((6, 6))
        self.board.pieces = {"A": {}, "B": {}}
        self.board.last_move = None
        self.board.estimations = {}

        self.info_panel.update_info(turn=0, current_player="A", status="Ready", winner=None)

    def _execute_one_move(self):
        """1手実行"""
        if not self.game_engine or not self.game_running:
            return

        # 現在のプレイヤーのエージェントを取得
        current_agent = self.agent1 if self.current_player == "A" else self.agent2

        # 合法手を取得
        legal_moves = self.game_engine.get_legal_moves(self.current_player)

        if not legal_moves:
            self.info_panel.add_log(f"Player {self.current_player} has no legal moves")
            self._check_game_over()
            return

        # エージェントに手を選択させる
        game_state = self.game_engine.get_game_state()
        move = current_agent.get_move(game_state, legal_moves)

        if move:
            # 手を実行
            self.game_engine.make_move(move[0], move[1], self.current_player)

            # ボード更新
            self.board.update_state(
                self.game_engine.board, self.game_engine.player_a_pieces, self.game_engine.player_b_pieces
            )
            self.board.last_move = move

            # 推定結果を取得（相手の駒）
            if hasattr(current_agent, "last_estimations"):
                self.board.estimations = current_agent.last_estimations

            # 情報更新
            self.info_panel.update_info(turn=self.game_engine.turn, current_player=self.current_player)

            self.info_panel.add_log(f"P{self.current_player}: {move[0]}→{move[1]}")

            # 勝利判定
            winner = self.game_engine.check_winner()
            if winner:
                self._game_over(winner)
            else:
                # プレイヤー交代
                self.current_player = "B" if self.current_player == "A" else "A"
                self.info_panel.update_info(current_player=self.current_player)

    def _check_game_over(self):
        """ゲーム終了判定"""
        winner = self.game_engine.check_winner()
        if winner:
            self._game_over(winner)

    def _game_over(self, winner: str):
        """ゲーム終了処理"""
        self.game_running = False
        self.game_paused = True

        self.info_panel.update_info(status="Game Over", winner=winner)

        self.info_panel.add_log(f"🏆 Player {winner} wins!")

        # 統計更新
        if winner == "A":
            self.info_panel.agent1_stats["wins"] += 1
            self.info_panel.agent2_stats["losses"] += 1
        elif winner == "B":
            self.info_panel.agent1_stats["losses"] += 1
            self.info_panel.agent2_stats["wins"] += 1
        else:
            self.info_panel.agent1_stats["draws"] += 1
            self.info_panel.agent2_stats["draws"] += 1

    def _update(self):
        """ゲーム状態を更新"""
        if self.game_running and not self.game_paused:
            current_time = pygame.time.get_ticks()

            if current_time - self.last_move_time > self.move_delay:
                self._execute_one_move()
                self.last_move_time = current_time

    def _draw(self):
        """画面を描画"""
        # 背景
        self.screen.fill(Colors.DARK_GRAY)

        # 各コンポーネントを描画
        self.board.draw()
        self.info_panel.draw()
        self.control_panel.draw()

        # 画面更新
        pygame.display.flip()


# ================================================================================
# Part 7: 簡略化されたゲームシステム（デモ用）
# ================================================================================


# cqcnn_battle_system_simplified.py
class SimpleGameEngine:
    """簡略化されたゲームエンジン（GUI用）"""

    def __init__(self):
        self.board = np.zeros((6, 6), dtype=int)
        self.player_a_pieces = {}
        self.player_b_pieces = {}
        self.turn = 0
        self.winner = None

    def initialize_game(self, agent1, agent2):
        """ゲームを初期化"""
        # ガイスターの正しい初期配置
        # Player A (下側) - 中央4×2エリア
        positions_a = [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),  # 2行目
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
        ]  # 1行目（最下段）
        piece_types_a = ["good", "good", "good", "good", "bad", "bad", "bad", "bad"]
        random.shuffle(piece_types_a)  # ランダムに配置

        for pos, piece_type in zip(positions_a, piece_types_a):
            self.board[pos] = 1
            self.player_a_pieces[pos] = piece_type

        # Player B (上側) - 中央4×2エリア
        positions_b = [
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),  # 5行目
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
        ]  # 6行目（最上段）
        piece_types_b = ["good", "good", "good", "good", "bad", "bad", "bad", "bad"]
        random.shuffle(piece_types_b)  # ランダムに配置

        for pos, piece_type in zip(positions_b, piece_types_b):
            self.board[pos] = -1
            self.player_b_pieces[pos] = piece_type

    def get_legal_moves(self, player: str) -> List[Tuple]:
        """合法手を取得"""
        moves = []
        pieces = self.player_a_pieces if player == "A" else self.player_b_pieces

        for pos in pieces.keys():
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if self._is_valid_move(pos, new_pos, player):
                    moves.append((pos, new_pos))

        return moves

    def _is_valid_move(self, from_pos, to_pos, player):
        """移動の妥当性確認"""
        if not (0 <= to_pos[0] < 6 and 0 <= to_pos[1] < 6):
            return False

        player_val = 1 if player == "A" else -1
        if self.board[to_pos] == player_val:
            return False

        return True

    def make_move(self, from_pos, to_pos, player):
        """移動を実行"""
        player_val = 1 if player == "A" else -1
        enemy_val = -player_val

        # 敵駒を取る
        if self.board[to_pos] == enemy_val:
            enemy_pieces = self.player_b_pieces if player == "A" else self.player_a_pieces
            if to_pos in enemy_pieces:
                del enemy_pieces[to_pos]

        # 移動
        self.board[from_pos] = 0
        self.board[to_pos] = player_val

        pieces = self.player_a_pieces if player == "A" else self.player_b_pieces
        piece_type = pieces.pop(from_pos)
        pieces[to_pos] = piece_type

        # 脱出判定（ガイスターの正しいルール）
        # プレイヤーAの善玉は相手陣地の角（上側）から脱出
        # プレイヤーBの善玉は相手陣地の角（下側）から脱出
        if player == "A":
            escape_zones = [(5, 0), (5, 5)]  # Aの脱出口（相手陣地の上角）
        else:
            escape_zones = [(0, 0), (0, 5)]  # Bの脱出口（相手陣地の下角）

        if to_pos in escape_zones and piece_type == "good":
            self.winner = player

        self.turn += 1

    def check_winner(self):
        """勝者を判定"""
        if self.winner:
            return self.winner

        # 全駒がなくなった場合
        if not self.player_a_pieces:
            return "B"
        if not self.player_b_pieces:
            return "A"

        # 100ターン経過
        if self.turn >= 100:
            a_count = len(self.player_a_pieces)
            b_count = len(self.player_b_pieces)
            if a_count > b_count:
                return "A"
            elif b_count > a_count:
                return "B"
            else:
                return "Draw"

        return None

    def get_game_state(self):
        """ゲーム状態を取得"""

        class GameState:
            pass

        state = GameState()
        state.board = self.board
        state.player_a_pieces = self.player_a_pieces
        state.player_b_pieces = self.player_b_pieces
        state.turn = self.turn

        return state

    def reset(self):
        """ゲームをリセット"""
        self.board = np.zeros((6, 6), dtype=int)
        self.player_a_pieces = {}
        self.player_b_pieces = {}
        self.turn = 0
        self.winner = None


def create_demo_agent(player_id: str, name: str):
    """デモ用の簡単なエージェントを作成（未使用だが互換性のため残す）"""
    return RandomAgent(player_id)


# ================================================================================
# メイン実行
# ================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎮 CQCNN GUI Battle System")
    print("=" * 70)
    print("\n初回起動時にAI選択ダイアログが表示されます。")
    print("\n操作方法:")
    print("  Start: ゲーム開始")
    print("  Pause: 一時停止")
    print("  Reset: リセット")
    print("  Step: 1手進める")
    print("  Fast/Slow: 速度調整")
    print("\nキーボード:")
    print("  Space: 開始/一時停止")
    print("  R: リセット")
    print("  S: ステップ実行")
    print("  C: AIを変更")
    print("\n利用可能なAI:")
    print("  - ランダムAI: ランダムに手を選択")
    print("  - シンプルAI: 基本的な評価関数")
    print("  - CQCNN AI: 量子回路を使用（要モデルファイル）")
    print("  - カスタムAI: 独自のモデルファイルを読み込み")

    # saved_modelsディレクトリを作成
    os.makedirs("saved_models", exist_ok=True)

    print("\n起動中...")

    app = CQCNNBattleGUI()
    app.run()

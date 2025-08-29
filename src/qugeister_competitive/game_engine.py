"""
Qugeisterゲームエンジン
シンプルな6x6ガイスター実装（正しい脱出口判定版）
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class GameState:
    """ゲーム状態"""

    board: np.ndarray
    turn: int
    current_player: str
    player_a_pieces: Dict[Tuple[int, int], str]  # position -> piece_type
    player_b_pieces: Dict[Tuple[int, int], str]
    move_history: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    game_over: bool = False
    winner: Optional[str] = None


class GeisterGame:
    """6x6ガイスターゲーム実装（正しい脱出判定版）"""

    def __init__(self):
        self.board_size = 6
        self.reset_game()

    def reset_game(self):
        """ゲーム状態をリセット"""
        self.board = np.zeros((6, 6), dtype=int)  # 0:空, 1:A, -1:B
        self.turn = 0
        self.current_player = "A"

        # 初期配置（デフォルト）
        self.player_a_pieces = {
            (1, 0): "good",
            (2, 0): "good",
            (3, 0): "good",
            (4, 0): "good",
            (1, 1): "bad",
            (2, 1): "bad",
            (3, 1): "bad",
            (4, 1): "bad",
        }

        self.player_b_pieces = {
            (1, 5): "good",
            (2, 5): "good",
            (3, 5): "good",
            (4, 5): "good",
            (1, 4): "bad",
            (2, 4): "bad",
            (3, 4): "bad",
            (4, 4): "bad",
        }

        # ボードに駒を配置
        for pos in self.player_a_pieces:
            self.board[pos[1], pos[0]] = 1
        for pos in self.player_b_pieces:
            self.board[pos[1], pos[0]] = -1

        self.move_history = []
        self.game_over = False
        self.winner = None

    def get_legal_moves(self, player: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """合法手を取得"""
        pieces = self.player_a_pieces if player == "A" else self.player_b_pieces
        legal_moves = []

        for pos in pieces.keys():
            x, y = pos
            # 4方向に移動可能
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 6 and 0 <= new_y < 6:
                    # 自分の駒がない場所
                    if (new_x, new_y) not in pieces:
                        legal_moves.append(((x, y), (new_x, new_y)))

        return legal_moves

    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """手を実行"""
        if self.game_over:
            return False

        current_pieces = self.player_a_pieces if self.current_player == "A" else self.player_b_pieces
        opponent_pieces = self.player_b_pieces if self.current_player == "A" else self.player_a_pieces

        # 合法性チェック
        if from_pos not in current_pieces:
            return False

        legal_moves = self.get_legal_moves(self.current_player)
        if (from_pos, to_pos) not in legal_moves:
            return False

        # 駒移動
        piece_type = current_pieces[from_pos]
        del current_pieces[from_pos]

        # 相手駒を取る場合
        if to_pos in opponent_pieces:
            opponent_pieces[to_pos]
            del opponent_pieces[to_pos]

        current_pieces[to_pos] = piece_type

        # ボード更新
        self.board[from_pos[1], from_pos[0]] = 0
        self.board[to_pos[1], to_pos[0]] = 1 if self.current_player == "A" else -1

        # 履歴記録
        self.move_history.append((from_pos, to_pos))
        self.turn += 1

        # 勝利判定
        self._check_win_condition()

        # プレイヤー交代
        self.current_player = "B" if self.current_player == "A" else "A"

        return True

    def _check_win_condition(self):
        """勝利条件をチェック（正しい脱出判定）"""
        # 善玉全取り勝ち
        a_good_count = sum(1 for piece in self.player_a_pieces.values() if piece == "good")
        b_good_count = sum(1 for piece in self.player_b_pieces.values() if piece == "good")

        if a_good_count == 0:
            self.game_over = True
            self.winner = "B"
            return
        if b_good_count == 0:
            self.game_over = True
            self.winner = "A"
            return

        # 脱出勝ち（正しい脱出口判定）
        # プレイヤーAの脱出口: 相手陣地の (0,5) と (5,5)
        for pos, piece_type in self.player_a_pieces.items():
            if piece_type == "good" and (pos == (0, 5) or pos == (5, 5)):
                self.game_over = True
                self.winner = "A"
                return

        # プレイヤーBの脱出口: 相手陣地の (0,0) と (5,0)
        for pos, piece_type in self.player_b_pieces.items():
            if piece_type == "good" and (pos == (0, 0) or pos == (5, 0)):
                self.game_over = True
                self.winner = "B"
                return

        # 悪玉全取らせ勝ち
        a_bad_count = sum(1 for piece in self.player_a_pieces.values() if piece == "bad")
        b_bad_count = sum(1 for piece in self.player_b_pieces.values() if piece == "bad")

        if a_bad_count == 0:
            self.game_over = True
            self.winner = "A"  # Aの悪玉が全て取られた → Aの勝ち
            return
        if b_bad_count == 0:
            self.game_over = True
            self.winner = "B"  # Bの悪玉が全て取られた → Bの勝ち
            return

        # ターン制限
        if self.turn >= 100:
            self.game_over = True
            self.winner = "Draw"

    def get_game_state(self, player: str) -> GameState:
        """プレイヤー視点のゲーム状態を取得"""
        return GameState(
            board=self.board.copy(),
            turn=self.turn,
            current_player=self.current_player,
            player_a_pieces=self.player_a_pieces.copy(),
            player_b_pieces=self.player_b_pieces.copy(),
            move_history=self.move_history.copy(),
            game_over=self.game_over,
            winner=self.winner,
        )

    def display_board(self):
        """ボード表示（デバッグ用）"""
        print("  0 1 2 3 4 5")
        for y in range(6):
            row = f"{y} "
            for x in range(6):
                if self.board[y, x] == 1:
                    row += "A "
                elif self.board[y, x] == -1:
                    row += "B "
                else:
                    row += ". "
            print(row)
        print()

        # 正しい脱出口表示
        print("脱出口:")
        print("  プレイヤーA: (0,5), (5,5) - 相手陣地")
        print("  プレイヤーB: (0,0), (5,0) - 相手陣地")
        print()

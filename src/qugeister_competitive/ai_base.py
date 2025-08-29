"""
AI基底クラスとサンプルAI実装
"""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from .game_engine import GameState


class BaseAI(ABC):
    """AI基底クラス"""

    def __init__(self, name: str, player_id: str):
        self.name = name
        self.player_id = player_id
        self.games_played = 0
        self.wins = 0
        self.rating = 1500  # 初期レーティング

    @abstractmethod
    def get_move(
        self, game_state: GameState, legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """手を選択"""
        pass

    def record_result(self, won: bool):
        """結果を記録"""
        self.games_played += 1
        if won:
            self.wins += 1

    @property
    def win_rate(self) -> float:
        """勝率"""
        return self.wins / max(self.games_played, 1)


class RandomAI(BaseAI):
    """ランダムAI"""

    def __init__(self, player_id: str):
        super().__init__("RandomAI", player_id)

    def get_move(self, game_state: GameState, legal_moves: List) -> Optional[Tuple]:
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class SimpleAI(BaseAI):
    """シンプルなヒューリスティックAI"""

    def __init__(self, player_id: str):
        super().__init__("SimpleAI", player_id)

    def get_move(self, game_state: GameState, legal_moves: List) -> Optional[Tuple]:
        if not legal_moves:
            return None

        # 前進を優先
        best_moves = []
        best_score = -999

        for move in legal_moves:
            from_pos, to_pos = move
            score = 0

            # 前進ボーナス
            if self.player_id == "A":
                score += (to_pos[1] - from_pos[1]) * 2  # 上方向が前進
            else:
                score += (from_pos[1] - to_pos[1]) * 2  # 下方向が前進

            # 中央寄りボーナス
            score += 1 - abs(to_pos[0] - 2.5) / 2.5

            # 相手駒を取るボーナス
            opponent_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces
            if to_pos in opponent_pieces:
                score += 5

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else random.choice(legal_moves)


class AggressiveAI(BaseAI):
    """攻撃的AI"""

    def __init__(self, player_id: str):
        super().__init__("AggressiveAI", player_id)

    def get_move(self, game_state: GameState, legal_moves: List) -> Optional[Tuple]:
        if not legal_moves:
            return None

        # 攻撃を最優先
        opponent_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces

        # 相手駒を取る手があれば優先
        attack_moves = []
        for move in legal_moves:
            from_pos, to_pos = move
            if to_pos in opponent_pieces:
                attack_moves.append(move)

        if attack_moves:
            return random.choice(attack_moves)

        # 攻撃手がなければ前進
        advance_moves = []
        for move in legal_moves:
            from_pos, to_pos = move
            if self.player_id == "A" and to_pos[1] > from_pos[1]:
                advance_moves.append(move)
            elif self.player_id == "B" and to_pos[1] < from_pos[1]:
                advance_moves.append(move)

        return random.choice(advance_moves) if advance_moves else random.choice(legal_moves)

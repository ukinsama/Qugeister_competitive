#!/usr/bin/env python3
"""
高度な報酬関数システム
ガイスターの戦略的要素を全て考慮
"""


class IntelligentRewardSystem:
    """
    ガイスター専用高度報酬システム
    """

    def __init__(self):
        # 報酬重み設定
        self.weights = {
            # 基本行動コスト
            "move_cost": -0.02,
            # 位置的報酬
            "forward_progress": 0.3,
            "escape_zone_bonus": 2.0,
            "escape_achievement": 15.0,
            "center_control": 0.1,
            "safe_position": 0.05,
            # 戦術的報酬
            "piece_capture": 5.0,
            "good_piece_capture": 8.0,
            "bad_piece_sacrifice": 3.0,
            "piece_protection": 0.2,
            "threat_creation": 0.5,
            # 戦略的報酬
            "formation_quality": 0.3,
            "mobility_preservation": 0.1,
            "opponent_mobility_reduction": 0.2,
            "endgame_positioning": 0.4,
            # ペナルティ
            "backward_move": -0.2,
            "edge_exposure": -0.1,
            "good_piece_risk": -0.3,
            "stagnation": -0.1,
        }

    def calculate_reward(self, old_state, move, new_state, game_over=False, winner=None) -> float:
        """
        包括的報酬計算

        Args:
            old_state: 行動前のゲーム状態
            move: 実行された手 (from_pos, to_pos)
            new_state: 行動後のゲーム状態
            game_over: ゲーム終了フラグ
            winner: 勝者 (ゲーム終了時)
        """
        total_reward = 0.0
        current_player = old_state.current_player
        from_pos, to_pos = move

        # 基本移動コスト
        total_reward += self.weights["move_cost"]

        # ゲーム終了時の最終報酬
        if game_over:
            final_reward = self._calculate_final_reward(current_player, winner)
            return total_reward + final_reward

        # 1. 位置的報酬
        total_reward += self._calculate_positional_reward(old_state, move, new_state)

        # 2. 戦術的報酬
        total_reward += self._calculate_tactical_reward(old_state, move, new_state)

        # 3. 戦略的報酬
        total_reward += self._calculate_strategic_reward(old_state, move, new_state)

        # 4. ペナルティ
        total_reward += self._calculate_penalties(old_state, move, new_state)

        return total_reward

    def _calculate_final_reward(self, player, winner) -> float:
        """ゲーム終了時の最終報酬"""
        if winner == player:
            return 20.0  # 勝利大ボーナス
        elif winner is None:
            return -2.0  # 引き分けペナルティ
        else:
            return -10.0  # 敗北ペナルティ

    def _calculate_positional_reward(self, old_state, move, new_state) -> float:
        """位置的報酬計算"""
        reward = 0.0
        current_player = old_state.current_player
        from_pos, to_pos = move

        # 前進報酬
        if current_player == "A":
            forward_progress = to_pos[1] - from_pos[1]  # A は上方向（Y+）が前進
            escape_zone = to_pos[1] >= 5  # 最上段が脱出ゾーン
            is_escaping = to_pos[1] == 5 and from_pos[1] == 4
        else:
            forward_progress = from_pos[1] - to_pos[1]  # B は下方向（Y-）が前進
            escape_zone = to_pos[1] <= 0  # 最下段が脱出ゾーン
            is_escaping = to_pos[1] == 0 and from_pos[1] == 1

        # 前進ボーナス
        if forward_progress > 0:
            reward += self.weights["forward_progress"] * forward_progress

        # 脱出ゾーン到達ボーナス
        if escape_zone:
            reward += self.weights["escape_zone_bonus"]

            # 実際に脱出した場合の特別ボーナス
            if is_escaping:
                # 移動した駒が善玉かどうか判定
                player_pieces = old_state.player_a_pieces if current_player == "A" else old_state.player_b_pieces

                if from_pos in player_pieces:
                    piece_type = player_pieces[from_pos]
                    if piece_type == "good":
                        reward += self.weights["escape_achievement"]

        # 中央制圧ボーナス
        center_distance = abs(to_pos[0] - 2.5)
        center_bonus = (2.5 - center_distance) / 2.5
        reward += self.weights["center_control"] * center_bonus

        # 安全位置ボーナス（角や壁際でない）
        if 1 <= to_pos[0] <= 4 and 1 <= to_pos[1] <= 4:
            reward += self.weights["safe_position"]

        return reward

    def _calculate_tactical_reward(self, old_state, move, new_state) -> float:
        """戦術的報酬計算"""
        reward = 0.0
        current_player = old_state.current_player
        from_pos, to_pos = move

        # 駒取り分析
        if current_player == "A":
            old_opponent_pieces = old_state.player_b_pieces
            new_opponent_pieces = new_state.player_b_pieces
        else:
            old_opponent_pieces = old_state.player_a_pieces
            new_opponent_pieces = new_state.player_a_pieces

        # 駒を取った場合
        if len(new_opponent_pieces) < len(old_opponent_pieces):
            # 取られた駒の種類を特定
            captured_pos = None
            for pos in old_opponent_pieces:
                if pos not in new_opponent_pieces:
                    captured_pos = pos
                    break

            if captured_pos and captured_pos in old_opponent_pieces:
                captured_piece_type = old_opponent_pieces[captured_pos]

                if captured_piece_type == "good":
                    reward += self.weights["good_piece_capture"]
                else:  # captured_piece_type == "bad"
                    # 相手の悪玉を取るのは相手に有利（ブラフ戦略）
                    reward += self.weights["piece_capture"] * 0.3
            else:
                # 駒種類不明の場合は基本報酬
                reward += self.weights["piece_capture"]

        # 自分の悪玉を犠牲にする戦略（相手に取らせる）
        current_pieces = old_state.player_a_pieces if current_player == "A" else old_state.player_b_pieces

        if from_pos in current_pieces:
            moved_piece_type = current_pieces[from_pos]

            # 悪玉を危険な位置に移動（犠牲戦略）
            if moved_piece_type == "bad":
                danger_level = self._calculate_danger_level(to_pos, old_state, current_player)
                if danger_level > 0.5:
                    reward += self.weights["bad_piece_sacrifice"] * danger_level

        # 駒の保護（味方駒の近くに移動）
        protection_bonus = self._calculate_protection_bonus(to_pos, new_state, current_player)
        reward += self.weights["piece_protection"] * protection_bonus

        # 脅威創出（相手駒への攻撃的配置）
        threat_bonus = self._calculate_threat_bonus(to_pos, new_state, current_player)
        reward += self.weights["threat_creation"] * threat_bonus

        return reward

    def _calculate_strategic_reward(self, old_state, move, new_state) -> float:
        """戦略的報酬計算"""
        reward = 0.0
        current_player = old_state.current_player

        # フォーメーション品質
        formation_score = self._evaluate_formation(new_state, current_player)
        reward += self.weights["formation_quality"] * formation_score

        # 機動性保持
        mobility_score = self._calculate_mobility(new_state, current_player)
        reward += self.weights["mobility_preservation"] * mobility_score

        # 相手機動性削減
        opponent_mobility = self._calculate_mobility(new_state, "B" if current_player == "A" else "A")
        mobility_reduction = 1.0 - (opponent_mobility / 10.0)  # 正規化
        reward += self.weights["opponent_mobility_reduction"] * mobility_reduction

        # エンドゲーム配置（駒数が少ない場合の特別戦略）
        total_pieces = len(new_state.player_a_pieces) + len(new_state.player_b_pieces)
        if total_pieces <= 8:  # エンドゲーム判定
            endgame_score = self._evaluate_endgame_position(new_state, current_player)
            reward += self.weights["endgame_positioning"] * endgame_score

        return reward

    def _calculate_penalties(self, old_state, move, new_state) -> float:
        """ペナルティ計算"""
        penalty = 0.0
        current_player = old_state.current_player
        from_pos, to_pos = move

        # 後退ペナルティ
        if current_player == "A" and to_pos[1] < from_pos[1]:
            penalty += self.weights["backward_move"]
        elif current_player == "B" and to_pos[1] > from_pos[1]:
            penalty += self.weights["backward_move"]

        # 端への露出ペナルティ
        if to_pos[0] == 0 or to_pos[0] == 5:
            penalty += self.weights["edge_exposure"]

        # 善玉リスクペナルティ
        current_pieces = old_state.player_a_pieces if current_player == "A" else old_state.player_b_pieces

        if from_pos in current_pieces:
            moved_piece_type = current_pieces[from_pos]
            if moved_piece_type == "good":
                danger_level = self._calculate_danger_level(to_pos, new_state, current_player)
                penalty += self.weights["good_piece_risk"] * danger_level

        return penalty

    def _calculate_danger_level(self, position, game_state, current_player) -> float:
        """位置の危険度計算"""
        danger = 0.0
        x, y = position

        # 相手駒との距離分析
        opponent_pieces = game_state.player_b_pieces if current_player == "A" else game_state.player_a_pieces

        for opp_pos in opponent_pieces:
            distance = abs(x - opp_pos[0]) + abs(y - opp_pos[1])  # マンハッタン距離
            if distance == 1:  # 隣接
                danger += 0.8
            elif distance == 2:  # 2手で到達可能
                danger += 0.3

        return min(danger, 1.0)  # 0-1に正規化

    def _calculate_protection_bonus(self, position, game_state, current_player) -> float:
        """保護ボーナス計算"""
        protection = 0.0
        x, y = position

        # 味方駒との距離分析
        friendly_pieces = game_state.player_a_pieces if current_player == "A" else game_state.player_b_pieces

        for friend_pos in friendly_pieces:
            if friend_pos != position:  # 自分以外
                distance = abs(x - friend_pos[0]) + abs(y - friend_pos[1])
                if distance == 1:  # 隣接
                    protection += 0.5
                elif distance == 2:  # 近距離
                    protection += 0.2

        return min(protection, 1.0)

    def _calculate_threat_bonus(self, position, game_state, current_player) -> float:
        """脅威ボーナス計算"""
        threat = 0.0
        x, y = position

        # 相手駒への脅威分析
        opponent_pieces = game_state.player_b_pieces if current_player == "A" else game_state.player_a_pieces

        for opp_pos in opponent_pieces:
            distance = abs(x - opp_pos[0]) + abs(y - opp_pos[1])
            if distance == 1:  # 直接攻撃可能
                threat += 0.6
            elif distance == 2:  # 2手で攻撃可能
                threat += 0.2

        return min(threat, 1.0)

    def _evaluate_formation(self, game_state, current_player) -> float:
        """フォーメーション評価"""
        pieces = game_state.player_a_pieces if current_player == "A" else game_state.player_b_pieces

        if len(pieces) <= 2:
            return 0.0

        formation_score = 0.0
        positions = list(pieces.keys())

        # 駒同士の連携度
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if distance <= 2:  # 近距離連携
                    formation_score += 0.1

        # 前線形成ボーナス
        if current_player == "A":
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            formation_score += (avg_y - 2) * 0.1  # 前進度
        else:
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            formation_score += (3 - avg_y) * 0.1  # 前進度

        return max(formation_score, 0.0)

    def _calculate_mobility(self, game_state, player) -> float:
        """機動性計算"""
        pieces = game_state.player_a_pieces if player == "A" else game_state.player_b_pieces

        total_moves = 0
        for pos in pieces:
            x, y = pos
            # 4方向の移動可能性チェック
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 6 and 0 <= new_y < 6 and (new_x, new_y) not in pieces:
                    total_moves += 1

        return total_moves

    def _evaluate_endgame_position(self, game_state, current_player) -> float:
        """エンドゲーム配置評価"""
        pieces = game_state.player_a_pieces if current_player == "A" else game_state.player_b_pieces

        endgame_score = 0.0

        # 善玉の脱出ルート確保
        for pos, piece_type in pieces.items():
            if piece_type == "good":
                if current_player == "A":
                    # Aは上方向に脱出
                    escape_distance = 5 - pos[1]
                    endgame_score += (6 - escape_distance) * 0.2
                else:
                    # Bは下方向に脱出
                    escape_distance = pos[1]
                    endgame_score += (6 - escape_distance) * 0.2

        return endgame_score


# 報酬関数をQuantumAIに統合するための関数
def calculate_intelligent_reward(old_state, move, game, game_over=False, winner=None):
    """
    高度報酬関数のラッパー関数
    QuantumAIから呼び出される
    """
    reward_system = IntelligentRewardSystem()

    # 新しいゲーム状態を取得
    new_state = game.get_game_state(old_state.current_player)

    return reward_system.calculate_reward(
        old_state=old_state, move=move, new_state=new_state, game_over=game_over, winner=winner
    )

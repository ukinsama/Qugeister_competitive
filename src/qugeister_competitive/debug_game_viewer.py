#!/usr/bin/env python3
"""
完全統合ガイスターデバッグGUI
2段階脱出システム対応 - 1ファイル完結版
"""

import pygame
import numpy as np
from typing import Tuple, List


class DebugGeisterGame:
    """デバッグ用ガイスターゲーム（2段階脱出対応）"""

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

        print("🎮 デバッグモード開始")
        print("プレイヤーA（青）の駒:")
        for pos, piece_type in self.player_a_pieces.items():
            print(f"  {pos}: {piece_type}")
        print("プレイヤーB（赤）の駒:")
        for pos, piece_type in self.player_b_pieces.items():
            print(f"  {pos}: {piece_type}")

    def get_legal_moves(self, player: str) -> List[Tuple]:
        """合法手を取得（正しい脱出口からの脱出を含む）"""
        pieces = self.player_a_pieces if player == "A" else self.player_b_pieces
        legal_moves = []

        # 正しい脱出口の定義
        if player == "A":
            escape_positions = [(0, 5), (5, 5)]  # Aは相手陣地（上側）から脱出
        else:
            escape_positions = [(0, 0), (5, 0)]  # Bは相手陣地（下側）から脱出

        for pos in pieces.keys():
            x, y = pos

            # 脱出口にいる善玉は脱出可能
            if pos in escape_positions and pieces[pos] == "good":
                # 脱出の特別な移動（ボード外への移動）
                legal_moves.append((pos, "ESCAPE"))
                print(f"🚪 {player}の善玉が{pos}から脱出可能！")

            # 通常の4方向移動
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 6 and 0 <= new_y < 6:
                    # 自分の駒がない場所
                    if (new_x, new_y) not in pieces:
                        legal_moves.append(((x, y), (new_x, new_y)))

        return legal_moves

    def make_move(self, from_pos: Tuple[int, int], to_pos) -> bool:
        """手を実行（脱出対応）"""
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

        # 脱出処理
        if to_pos == "ESCAPE":
            piece_type = current_pieces[from_pos]
            if piece_type == "good":
                # 脱出口から駒を削除
                del current_pieces[from_pos]
                self.board[from_pos[1], from_pos[0]] = 0

                # 履歴記録
                self.move_history.append((from_pos, "ESCAPE"))
                self.turn += 1

                print(f"🎊 {self.current_player}の善玉が{from_pos}から脱出！")

                # 脱出勝利
                self.game_over = True
                self.winner = self.current_player
                print(f"🏆 プレイヤー{self.current_player}脱出勝利！")
                return True
            else:
                print("❌ 悪玉は脱出できません")
                return False

        # 通常の移動処理
        piece_type = current_pieces[from_pos]
        del current_pieces[from_pos]

        # 相手駒を取る場合
        captured_type = None
        if to_pos in opponent_pieces:
            captured_type = opponent_pieces[to_pos]
            del opponent_pieces[to_pos]
            print(f"🎯 {self.current_player}が{to_pos}で相手の{captured_type}駒を取得！")

        current_pieces[to_pos] = piece_type

        # ボード更新
        self.board[from_pos[1], from_pos[0]] = 0
        self.board[to_pos[1], to_pos[0]] = 1 if self.current_player == "A" else -1

        # 履歴記録
        self.move_history.append((from_pos, to_pos))
        self.turn += 1

        print(f"📋 手#{self.turn}: {self.current_player} {from_pos} → {to_pos} ({piece_type})")

        # 脱出口到達の通知（正しい脱出口）
        if self.current_player == "A" and (to_pos == (0, 5) or to_pos == (5, 5)) and piece_type == "good":
            print(f"🚪 プレイヤーAの善玉が相手陣地の脱出口{to_pos}に到達！次のターンで脱出可能")
        elif self.current_player == "B" and (to_pos == (0, 0) or to_pos == (5, 0)) and piece_type == "good":
            print(f"🚪 プレイヤーBの善玉が相手陣地の脱出口{to_pos}に到達！次のターンで脱出可能")

        # その他の勝利判定
        if not self.game_over:
            self._check_win_condition()

        # プレイヤー交代
        if not self.game_over:
            self.current_player = "B" if self.current_player == "A" else "A"

        return True

    def _check_win_condition(self):
        """その他の勝利条件をチェック"""
        # 善玉全取り勝ち
        a_good_count = sum(1 for piece in self.player_a_pieces.values() if piece == "good")
        b_good_count = sum(1 for piece in self.player_b_pieces.values() if piece == "good")

        if a_good_count == 0:
            self.game_over = True
            self.winner = "B"
            print("🏆 プレイヤーB勝利！（Aの善玉を全て取得）")
            return
        if b_good_count == 0:
            self.game_over = True
            self.winner = "A"
            print("🏆 プレイヤーA勝利！（Bの善玉を全て取得）")
            return

        # 悪玉全取らせ勝ち
        a_bad_count = sum(1 for piece in self.player_a_pieces.values() if piece == "bad")
        b_bad_count = sum(1 for piece in self.player_b_pieces.values() if piece == "bad")

        if a_bad_count == 0:
            self.game_over = True
            self.winner = "A"
            print("🏆 プレイヤーA勝利！（悪玉を全て取らせた）")
            return
        if b_bad_count == 0:
            self.game_over = True
            self.winner = "B"
            print("🏆 プレイヤーB勝利！（悪玉を全て取らせた）")
            return

        # ターン制限
        if self.turn >= 100:
            self.game_over = True
            self.winner = "Draw"
            print("📊 引き分け（100ターン経過）")


class DebugGUI:
    """デバッグ用ガイスターGUI"""

    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ガイスター デバッグGUI - 2段階脱出対応")

        # 色定義
        self.colors = {
            "background": (240, 240, 240),
            "board": (139, 69, 19),
            "grid": (101, 67, 33),
            "player_a": (100, 150, 255),
            "player_b": (255, 100, 100),
            "good_piece": (0, 200, 0),  # 善玉は緑
            "bad_piece": (200, 0, 0),  # 悪玉は赤
            "highlight": (255, 255, 0),
            "text": (0, 0, 0),
            "panel": (220, 220, 220),
            "escape": (255, 215, 0),  # 脱出口は金色
            "legal_move": (150, 255, 150),  # 合法手は薄緑
            "escape_ready": (255, 100, 255),  # 脱出可能は紫
        }

        # フォント
        self.font_small = pygame.font.Font(None, 16)
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)

        # レイアウト
        self.board_size = 400
        self.cell_size = self.board_size // 6
        self.board_x = 50
        self.board_y = 100

        # ゲーム状態
        self.game = DebugGeisterGame()
        self.selected_piece = None
        self.legal_moves = []

        print("🎮 デバッグGUI起動")
        print("操作方法:")
        print("  - 左クリック: 駒選択/移動")
        print("  - 右クリック: 脱出実行")
        print("  - ESCキー: 脱出実行")
        print("  - R: ゲームリセット")
        print("  - Q: 終了")

    def get_cell_from_mouse(self, mouse_pos):
        """マウス位置からセル座標を取得"""
        mx, my = mouse_pos
        if (
            self.board_x <= mx <= self.board_x + self.board_size
            and self.board_y <= my <= self.board_y + self.board_size
        ):
            x = (mx - self.board_x) // self.cell_size
            y = (my - self.board_y) // self.cell_size
            if 0 <= x < 6 and 0 <= y < 6:
                return (x, y)
        return None

    def handle_click(self, mouse_pos):
        """マウスクリック処理（脱出対応）"""
        if self.game.game_over:
            return

        cell = self.get_cell_from_mouse(mouse_pos)

        current_pieces = self.game.player_a_pieces if self.game.current_player == "A" else self.game.player_b_pieces

        if self.selected_piece is None:
            # 駒選択
            if cell and cell in current_pieces:
                self.selected_piece = cell
                self.legal_moves = []

                # 通常の移動先を取得
                for move in self.game.get_legal_moves(self.game.current_player):
                    if move[0] == cell:
                        if move[1] == "ESCAPE":
                            # 脱出の場合は特別マーク
                            self.legal_moves.append("ESCAPE")
                        else:
                            self.legal_moves.append(move[1])

                piece_type = current_pieces[cell]
                print(f"📍 {self.game.current_player}の{piece_type}駒を選択: {cell}")

                # 脱出可能かチェック（正しい脱出口）
                escape_positions = (
                    [(0, 5), (5, 5)]
                    if self.game.current_player == "A"  # Aは相手陣地から脱出
                    else [(0, 0), (5, 0)]
                )  # Bは相手陣地から脱出
                if cell in escape_positions and piece_type == "good":
                    print("🚪 この駒は相手陣地の脱出口にいます！右クリックまたはESCキーで脱出")
        else:
            # 移動実行
            if cell and cell in self.legal_moves:
                success = self.game.make_move(self.selected_piece, cell)
                if success:
                    print(f"✅ 移動成功: {self.selected_piece} → {cell}")
                else:
                    print(f"❌ 移動失敗: {self.selected_piece} → {cell}")
            else:
                # 別の駒を選択
                if cell and cell in current_pieces:
                    self.selected_piece = cell
                    self.legal_moves = []

                    for move in self.game.get_legal_moves(self.game.current_player):
                        if move[0] == cell:
                            if move[1] == "ESCAPE":
                                self.legal_moves.append("ESCAPE")
                            else:
                                self.legal_moves.append(move[1])

                    piece_type = current_pieces[cell]
                    print(f"📍 {self.game.current_player}の{piece_type}駒を選択: {cell}")
                    return
                else:
                    if cell:
                        print(f"❌ 不正な移動: {self.selected_piece} → {cell}")

            # 選択解除
            self.selected_piece = None
            self.legal_moves = []

    def handle_escape(self):
        """脱出処理"""
        if self.selected_piece and "ESCAPE" in self.legal_moves:
            success = self.game.make_move(self.selected_piece, "ESCAPE")
            if success:
                print(f"🎊 脱出成功: {self.selected_piece} → ESCAPE")
            self.selected_piece = None
            self.legal_moves = []
        else:
            print("❌ 脱出できません（脱出口にいる善玉を選択してください）")

    def draw_board(self):
        """盤面描画"""
        # ボード背景
        board_rect = pygame.Rect(self.board_x, self.board_y, self.board_size, self.board_size)
        pygame.draw.rect(self.screen, self.colors["board"], board_rect)

        # 脱出口をハイライト（正しい脱出口）
        escape_positions_a = [(0, 5), (5, 5)]  # Aの脱出口（相手陣地）
        escape_positions_b = [(0, 0), (5, 0)]  # Bの脱出口（相手陣地）

        for x, y in escape_positions_a + escape_positions_b:
            cell_rect = pygame.Rect(
                self.board_x + x * self.cell_size, self.board_y + y * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, self.colors["escape"], cell_rect)

        # 脱出可能ハイライト
        if self.selected_piece and "ESCAPE" in self.legal_moves:
            x, y = self.selected_piece
            cell_rect = pygame.Rect(
                self.board_x + x * self.cell_size, self.board_y + y * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, self.colors["escape_ready"], cell_rect, 5)

        # 合法手をハイライト
        for move in self.legal_moves:
            if move != "ESCAPE" and isinstance(move, tuple):
                x, y = move
                cell_rect = pygame.Rect(
                    self.board_x + x * self.cell_size, self.board_y + y * self.cell_size, self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.screen, self.colors["legal_move"], cell_rect, 3)

        # グリッド線
        for i in range(7):
            # 縦線
            start_x = self.board_x + i * self.cell_size
            pygame.draw.line(
                self.screen, self.colors["grid"], (start_x, self.board_y), (start_x, self.board_y + self.board_size), 2
            )

            # 横線
            start_y = self.board_y + i * self.cell_size
            pygame.draw.line(
                self.screen, self.colors["grid"], (self.board_x, start_y), (self.board_x + self.board_size, start_y), 2
            )

        # 駒描画
        self.draw_pieces()

        # 座標表示
        for i in range(6):
            # X座標
            text = self.font_small.render(str(i), True, self.colors["text"])
            self.screen.blit(text, (self.board_x + i * self.cell_size + self.cell_size // 2 - 5, self.board_y - 20))
            # Y座標
            text = self.font_small.render(str(i), True, self.colors["text"])
            self.screen.blit(text, (self.board_x - 20, self.board_y + i * self.cell_size + self.cell_size // 2 - 5))

    def draw_pieces(self):
        """駒描画（種類表示付き）"""
        # プレイヤーAの駒
        for pos, piece_type in self.game.player_a_pieces.items():
            x, y = pos
            center_x = self.board_x + x * self.cell_size + self.cell_size // 2
            center_y = self.board_y + y * self.cell_size + self.cell_size // 2

            # 選択ハイライト
            if self.selected_piece == pos:
                pygame.draw.circle(self.screen, self.colors["highlight"], (center_x, center_y), self.cell_size // 2, 3)

            # 駒の色（善玉は緑、悪玉は赤）
            piece_color = self.colors["good_piece"] if piece_type == "good" else self.colors["bad_piece"]
            pygame.draw.circle(self.screen, piece_color, (center_x, center_y), self.cell_size // 3)

            # プレイヤー表示
            text = self.font.render("A", True, (255, 255, 255))
            text_rect = text.get_rect(center=(center_x, center_y - 5))
            self.screen.blit(text, text_rect)

            # 駒種類表示
            type_text = "G" if piece_type == "good" else "B"
            type_surf = self.font_small.render(type_text, True, (255, 255, 255))
            type_rect = type_surf.get_rect(center=(center_x, center_y + 8))
            self.screen.blit(type_surf, type_rect)

        # プレイヤーBの駒
        for pos, piece_type in self.game.player_b_pieces.items():
            x, y = pos
            center_x = self.board_x + x * self.cell_size + self.cell_size // 2
            center_y = self.board_y + y * self.cell_size + self.cell_size // 2

            # 選択ハイライト
            if self.selected_piece == pos:
                pygame.draw.circle(self.screen, self.colors["highlight"], (center_x, center_y), self.cell_size // 2, 3)

            # 駒の色（善玉は緑、悪玉は赤）
            piece_color = self.colors["good_piece"] if piece_type == "good" else self.colors["bad_piece"]
            pygame.draw.circle(self.screen, piece_color, (center_x, center_y), self.cell_size // 3)

            # プレイヤー表示
            text = self.font.render("B", True, (255, 255, 255))
            text_rect = text.get_rect(center=(center_x, center_y - 5))
            self.screen.blit(text, text_rect)

            # 駒種類表示
            type_text = "G" if piece_type == "good" else "B"
            type_surf = self.font_small.render(type_text, True, (255, 255, 255))
            type_rect = type_surf.get_rect(center=(center_x, center_y + 8))
            self.screen.blit(type_surf, type_rect)

    def draw_info_panel(self):
        """情報パネル描画"""
        panel_x = self.board_x + self.board_size + 20
        panel_y = self.board_y
        panel_width = self.width - panel_x - 20
        panel_height = self.board_size

        # パネル背景
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.colors["panel"], panel_rect)
        pygame.draw.rect(self.screen, self.colors["grid"], panel_rect, 2)

        y_offset = panel_y + 10

        # 現在のプレイヤー
        current_text = f"現在: プレイヤー{self.game.current_player}"
        current_surf = self.font.render(current_text, True, self.colors["text"])
        self.screen.blit(current_surf, (panel_x + 10, y_offset))
        y_offset += 30

        # ターン数
        turn_text = f"ターン: {self.game.turn}"
        turn_surf = self.font.render(turn_text, True, self.colors["text"])
        self.screen.blit(turn_surf, (panel_x + 10, y_offset))
        y_offset += 30

        # 駒数情報
        a_good = sum(1 for p in self.game.player_a_pieces.values() if p == "good")
        a_bad = sum(1 for p in self.game.player_a_pieces.values() if p == "bad")
        b_good = sum(1 for p in self.game.player_b_pieces.values() if p == "good")
        b_bad = sum(1 for p in self.game.player_b_pieces.values() if p == "bad")

        pieces_info = [f"プレイヤーA: 善{a_good} 悪{a_bad}", f"プレイヤーB: 善{b_good} 悪{b_bad}"]

        for info in pieces_info:
            info_surf = self.font_small.render(info, True, self.colors["text"])
            self.screen.blit(info_surf, (panel_x + 10, y_offset))
            y_offset += 20

        y_offset += 10

        # 脱出口情報
        escape_text = "脱出口:"
        escape_surf = self.font.render(escape_text, True, self.colors["text"])
        self.screen.blit(escape_surf, (panel_x + 10, y_offset))
        y_offset += 25

        escape_info = ["A: (0,5), (5,5) 相手陣地", "B: (0,0), (5,0) 相手陣地"]

        for info in escape_info:
            info_surf = self.font_small.render(info, True, self.colors["text"])
            self.screen.blit(info_surf, (panel_x + 10, y_offset))
            y_offset += 18

        y_offset += 10

        # 脱出説明
        escape_help = ["脱出方法:", "1.善玉を相手陣地の脱出口に移動", "2.右クリック/ESCで脱出"]

        for help_text in escape_help:
            help_surf = self.font_small.render(help_text, True, self.colors["text"])
            self.screen.blit(help_surf, (panel_x + 10, y_offset))
            y_offset += 16

        y_offset += 10

        # 操作説明
        controls = ["操作:", "左クリック: 駒選択/移動", "右クリック: 脱出", "ESCキー: 脱出", "R: リセット", "Q: 終了"]

        for control in controls:
            control_surf = self.font_small.render(control, True, self.colors["text"])
            self.screen.blit(control_surf, (panel_x + 10, y_offset))
            y_offset += 16

        # 選択状況
        if self.selected_piece:
            y_offset += 10
            current_pieces = self.game.player_a_pieces if self.game.current_player == "A" else self.game.player_b_pieces
            piece_type = current_pieces[self.selected_piece]
            select_text = f"選択中: {self.selected_piece} ({piece_type})"
            select_surf = self.font_small.render(select_text, True, (0, 0, 255))
            self.screen.blit(select_surf, (panel_x + 10, y_offset))
            y_offset += 16

            if "ESCAPE" in self.legal_moves:
                escape_text = "🚪 脱出可能！"
                escape_surf = self.font_small.render(escape_text, True, (255, 0, 255))
                self.screen.blit(escape_surf, (panel_x + 10, y_offset))
                y_offset += 16

        # 勝利メッセージ
        if self.game.game_over:
            y_offset += 20
            if self.game.winner != "Draw":
                win_text = f"🏆 プレイヤー{self.game.winner}勝利!"
            else:
                win_text = "📊 引き分け"

            win_surf = self.font.render(win_text, True, (255, 0, 0))
            self.screen.blit(win_surf, (panel_x + 10, y_offset))

    def draw_title(self):
        """タイトル描画"""
        title = "ガイスター デバッグGUI"
        title_surf = self.title_font.render(title, True, self.colors["text"])
        title_rect = title_surf.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_surf, title_rect)

        subtitle = "緑=善玉(GOOD), 赤=悪玉(BAD), 金=脱出口(相手陣地), 紫=脱出可能"
        subtitle_surf = self.font_small.render(subtitle, True, self.colors["text"])
        subtitle_rect = subtitle_surf.get_rect(center=(self.width // 2, 60))
        self.screen.blit(subtitle_surf, subtitle_rect)

    def run(self):
        """メインループ"""
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左クリック
                        self.handle_click(event.pos)
                    elif event.button == 3:  # 右クリック - 脱出実行
                        self.handle_escape()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        print("🔄 ゲームリセット")
                        self.game.reset_game()
                        self.selected_piece = None
                        self.legal_moves = []
                    elif event.key == pygame.K_ESCAPE:  # ESCキーで脱出
                        self.handle_escape()

            # 描画
            self.screen.fill(self.colors["background"])
            self.draw_title()
            self.draw_board()
            self.draw_info_panel()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    """メイン実行"""
    print("🎮 ガイスター デバッグGUI 起動 - 2段階脱出対応")
    print("=" * 50)

    try:
        gui = DebugGUI()
        gui.run()
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

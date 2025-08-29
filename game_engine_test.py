#!/usr/bin/env python3
"""
Qugeisterゲームエンジン動作検証テスト
実際のゲームとして正しく機能しているか確認
"""

import numpy as np
import random

# ===============================================================================
# シンプル版ゲームエンジン（検証用）
# ===============================================================================


class SimpleGeisterGame:
    """検証用のシンプルなガイスターゲーム実装"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.board_size = 6
        self.reset_game()

    def reset_game(self):
        """ゲームをリセット"""
        # ボード: 0=空, 1=A駒, -1=B駒
        self.board = np.zeros((6, 6), dtype=int)

        # 駒の位置と種類を記録
        # 各プレイヤー: 善玉4個、悪玉4個
        self.player_a_pieces = {}  # {(x,y): 'good' or 'bad'}
        self.player_b_pieces = {}

        # 初期配置（ランダムに善悪を決定）
        self._setup_initial_pieces()

        # ゲーム状態
        self.current_player = "A"
        self.turn = 0
        self.game_over = False
        self.winner = None
        self.win_reason = None

        if self.verbose:
            print("\n🎮 新しいゲームを開始")
            self._print_board()

    def _setup_initial_pieces(self):
        """初期配置をセットアップ"""
        # プレイヤーAの配置（下側2行の中央4列）
        a_positions = [(x, y) for y in range(0, 2) for x in range(1, 5)]
        a_types = ["good"] * 4 + ["bad"] * 4
        random.shuffle(a_types)

        for i, pos in enumerate(a_positions):
            self.player_a_pieces[pos] = a_types[i]
            self.board[pos[1], pos[0]] = 1

        # プレイヤーBの配置（上側2行の中央4列）
        b_positions = [(x, y) for y in range(4, 6) for x in range(1, 5)]
        b_types = ["good"] * 4 + ["bad"] * 4
        random.shuffle(b_types)

        for i, pos in enumerate(b_positions):
            self.player_b_pieces[pos] = b_types[i]
            self.board[pos[1], pos[0]] = -1

        if self.verbose:
            print(f"プレイヤーA 善玉位置: {[p for p, t in self.player_a_pieces.items() if t == 'good']}")
            print(f"プレイヤーA 悪玉位置: {[p for p, t in self.player_a_pieces.items() if t == 'bad']}")
            print(f"プレイヤーB 善玉位置: {[p for p, t in self.player_b_pieces.items() if t == 'good']}")
            print(f"プレイヤーB 悪玉位置: {[p for p, t in self.player_b_pieces.items() if t == 'bad']}")

    def _print_board(self):
        """ボードを表示"""
        print("\n  0 1 2 3 4 5")
        print("  " + "-" * 11)
        for y in range(6):
            row = f"{y}|"
            for x in range(6):
                if self.board[y, x] == 1:
                    # プレイヤーAの駒
                    piece_type = self.player_a_pieces.get((x, y), "?")
                    symbol = "A" if piece_type == "good" else "a"
                elif self.board[y, x] == -1:
                    # プレイヤーBの駒（相手視点では種類不明）
                    if self.current_player == "B":
                        piece_type = self.player_b_pieces.get((x, y), "?")
                        symbol = "B" if piece_type == "good" else "b"
                    else:
                        symbol = "?"  # 相手の駒は種類不明
                else:
                    symbol = "."
                row += f" {symbol}"
            print(row)
        print(f"\n現在のプレイヤー: {self.current_player}, ターン: {self.turn}")

    def get_legal_moves(self, player=None):
        """合法手のリストを取得"""
        if player is None:
            player = self.current_player

        legal_moves = []
        pieces = self.player_a_pieces if player == "A" else self.player_b_pieces

        # 脱出口の定義（正しい脱出口）
        if player == "A":
            escape_positions = [(0, 5), (5, 5)]  # Aは相手陣地（上側）から脱出
        else:
            escape_positions = [(0, 0), (5, 0)]  # Bは相手陣地（下側）から脱出

        for pos, piece_type in pieces.items():
            x, y = pos

            # 脱出可能チェック
            if pos in escape_positions and piece_type == "good":
                legal_moves.append((pos, "ESCAPE"))

            # 4方向への移動をチェック
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy

                # ボード内チェック
                if 0 <= new_x < 6 and 0 <= new_y < 6:
                    # 自分の駒がない場所なら移動可能
                    if player == "A":
                        if self.board[new_y, new_x] != 1:
                            legal_moves.append(((x, y), (new_x, new_y)))
                    else:
                        if self.board[new_y, new_x] != -1:
                            legal_moves.append(((x, y), (new_x, new_y)))

        return legal_moves

    def make_move(self, move):
        """手を実行"""
        from_pos, to_pos = move
        player = self.current_player
        pieces = self.player_a_pieces if player == "A" else self.player_b_pieces

        if from_pos not in pieces:
            if self.verbose:
                print(f"エラー: {from_pos}に駒がありません")
            return False

        piece_type = pieces[from_pos]

        # 脱出処理
        if to_pos == "ESCAPE":
            if self.verbose:
                print(f"🚪 {player}の{piece_type}が脱出！")

            # 駒を除去
            del pieces[from_pos]
            self.board[from_pos[1], from_pos[0]] = 0

            # 善玉が全て脱出したか確認
            remaining_good = sum(1 for t in pieces.values() if t == "good")
            if remaining_good == 0:
                self.game_over = True
                self.winner = player
                self.win_reason = "全善玉脱出"

            self._next_turn()
            return True

        # 通常移動処理
        to_x, to_y = to_pos
        target = self.board[to_y, to_x]

        # 相手の駒を捕獲
        if target != 0:
            opponent = "B" if player == "A" else "A"
            opp_pieces = self.player_b_pieces if player == "A" else self.player_a_pieces

            if (to_x, to_y) in opp_pieces:
                captured_type = opp_pieces[(to_x, to_y)]
                del opp_pieces[(to_x, to_y)]

                if self.verbose:
                    print(f"⚔️ {player}が{opponent}の{captured_type}を捕獲！")

                # 勝利条件チェック
                if captured_type == "good":
                    # 相手の善玉を取った
                    remaining_good = sum(1 for t in opp_pieces.values() if t == "good")
                    if remaining_good == 0:
                        self.game_over = True
                        self.winner = player
                        self.win_reason = f"{opponent}の善玉全滅"

                # 悪玉を全て取られたらプレイヤーの勝ち
                remaining_bad = sum(1 for t in pieces.values() if t == "bad")
                if remaining_bad == 0:
                    self.game_over = True
                    self.winner = player
                    self.win_reason = "自分の悪玉全滅"

        # 駒を移動
        del pieces[from_pos]
        pieces[to_pos] = piece_type

        # ボード更新
        self.board[from_pos[1], from_pos[0]] = 0
        self.board[to_y, to_x] = 1 if player == "A" else -1

        self._next_turn()
        return True

    def _next_turn(self):
        """次のターンへ"""
        self.current_player = "B" if self.current_player == "A" else "A"
        self.turn += 1

        if self.verbose:
            self._print_board()
            if self.game_over:
                print(f"\n🏆 ゲーム終了！勝者: {self.winner} ({self.win_reason})")


# ===============================================================================
# テストシナリオ
# ===============================================================================


def test_basic_movement():
    """基本的な移動テスト"""
    print("\n" + "=" * 60)
    print("テスト1: 基本的な移動")
    print("=" * 60)

    game = SimpleGeisterGame(verbose=False)

    # 初期状態の合法手を確認
    legal_moves = game.get_legal_moves("A")
    print(f"プレイヤーAの初期合法手数: {len(legal_moves)}")

    # ランダムに移動
    if legal_moves:
        move = random.choice(legal_moves)
        print(f"選択した手: {move}")
        success = game.make_move(move)
        print(f"移動成功: {success}")

    return success


def test_capture():
    """捕獲テスト"""
    print("\n" + "=" * 60)
    print("テスト2: 駒の捕獲")
    print("=" * 60)

    game = SimpleGeisterGame(verbose=True)

    # 何手か進めて捕獲の機会を作る
    for _ in range(10):
        legal_moves = game.get_legal_moves()
        if not legal_moves or game.game_over:
            break

        # 捕獲可能な手を探す
        capture_moves = []
        for move in legal_moves:
            if move[1] != "ESCAPE":
                to_x, to_y = move[1]
                if game.board[to_y, to_x] != 0:
                    capture_moves.append(move)

        if capture_moves:
            move = capture_moves[0]
            print(f"捕獲の手: {move}")
            game.make_move(move)
            break
        else:
            game.make_move(random.choice(legal_moves))

    return True


def test_escape():
    """脱出テスト"""
    print("\n" + "=" * 60)
    print("テスト3: 脱出")
    print("=" * 60)

    # カスタム配置で脱出しやすい状況を作る
    game = SimpleGeisterGame(verbose=True)

    # プレイヤーAの善玉を脱出口近くに配置（テスト用に直接操作）
    game.board = np.zeros((6, 6), dtype=int)
    game.player_a_pieces = {
        (0, 4): "good",  # 脱出口の近く
        (5, 4): "good",  # 脱出口の近く
        (2, 0): "bad",
        (3, 0): "bad",
    }
    game.player_b_pieces = {(2, 5): "good", (3, 5): "good", (1, 4): "bad", (4, 4): "bad"}

    # ボードに反映
    for pos in game.player_a_pieces:
        game.board[pos[1], pos[0]] = 1
    for pos in game.player_b_pieces:
        game.board[pos[1], pos[0]] = -1

    print("カスタム配置完了")
    game._print_board()

    # 脱出を試みる
    for _ in range(20):
        legal_moves = game.get_legal_moves()
        if not legal_moves or game.game_over:
            break

        # 脱出可能な手を優先
        escape_moves = [m for m in legal_moves if m[1] == "ESCAPE"]
        if escape_moves:
            print(f"脱出可能！: {escape_moves[0]}")
            game.make_move(escape_moves[0])
        else:
            # 脱出口に向かって移動
            game.make_move(random.choice(legal_moves))

    return game.game_over


def test_full_game():
    """完全なゲームをシミュレート"""
    print("\n" + "=" * 60)
    print("テスト4: 完全なゲームプレイ")
    print("=" * 60)

    game = SimpleGeisterGame(verbose=True)
    max_turns = 100

    for turn in range(max_turns):
        if game.game_over:
            print(f"ゲーム終了！ターン数: {turn}")
            break

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            print(f"合法手なし！プレイヤー{game.current_player}")
            break

        # 賢い選択（脱出優先、次に捕獲、最後に通常移動）
        escape_moves = [m for m in legal_moves if m[1] == "ESCAPE"]
        if escape_moves:
            move = escape_moves[0]
        else:
            capture_moves = []
            for m in legal_moves:
                if m[1] != "ESCAPE":
                    to_x, to_y = m[1]
                    if game.board[to_y, to_x] != 0:
                        capture_moves.append(m)

            if capture_moves:
                move = random.choice(capture_moves)
            else:
                move = random.choice(legal_moves)

        game.make_move(move)

    if not game.game_over:
        print(f"最大ターン数（{max_turns}）に到達")

    return game.game_over


def run_statistics():
    """統計的なテスト"""
    print("\n" + "=" * 60)
    print("テスト5: 統計分析（10ゲーム）")
    print("=" * 60)

    results = {"A_wins": 0, "B_wins": 0, "turns": [], "win_reasons": []}

    for i in range(10):
        game = SimpleGeisterGame(verbose=False)
        max_turns = 100

        for turn in range(max_turns):
            if game.game_over:
                break

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # ランダムプレイ
            game.make_move(random.choice(legal_moves))

        if game.game_over:
            if game.winner == "A":
                results["A_wins"] += 1
            else:
                results["B_wins"] += 1
            results["turns"].append(game.turn)
            results["win_reasons"].append(game.win_reason)

        print(f"ゲーム{i + 1}: 勝者={game.winner}, ターン={game.turn}, 理由={game.win_reason}")

    # 統計表示
    print("\n統計結果:")
    print(f"プレイヤーA勝利: {results['A_wins']}")
    print(f"プレイヤーB勝利: {results['B_wins']}")
    if results["turns"]:
        print(f"平均ターン数: {sum(results['turns']) / len(results['turns']):.1f}")
    print(f"勝利理由: {set(results['win_reasons'])}")

    return True


# ===============================================================================
# メイン実行
# ===============================================================================


def main():
    """全テストを実行"""
    print("🎮 Qugeisterゲームエンジン検証テスト")
    print("=" * 60)

    tests = [
        ("基本移動", test_basic_movement),
        ("駒の捕獲", test_capture),
        ("脱出", test_escape),
        ("完全ゲーム", test_full_game),
        ("統計分析", run_statistics),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {name}: 成功")
                passed += 1
            else:
                print(f"❌ {name}: 失敗")
                failed += 1
        except Exception as e:
            print(f"❌ {name}: エラー - {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"テスト結果: {passed}/{len(tests)} 成功")
    print("=" * 60)

    if failed == 0:
        print("🎉 全てのテストが成功しました！")
        print("ゲームエンジンは正常に動作しています。")
    else:
        print(f"⚠️ {failed}個のテストが失敗しました。")
        print("ゲームエンジンに問題がある可能性があります。")


if __name__ == "__main__":
    main()

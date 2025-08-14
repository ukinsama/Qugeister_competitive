#!/usr/bin/env python3
"""
修正版GUI対戦システム
unhashable type: 'list' エラーの修正版
"""

import pygame
import json
import time
from typing import List, Dict, Any, Optional
import sys
import os

# パッケージのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qugeister_competitive.game_engine import GeisterGame
from qugeister_competitive.ai_base import RandomAI, SimpleAI, AggressiveAI
from qugeister_competitive.tournament import TournamentManager

class FixedGameGUI:
    """修正版ゲームGUIクラス"""
    
    def __init__(self, width: int = 900, height: int = 700):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Qugeister AI Battle - Fixed Version")
        
        # 色定義
        self.colors = {
            "background": (240, 240, 240),
            "board": (139, 69, 19),
            "grid": (101, 67, 33),
            "player_a": (100, 150, 255),
            "player_b": (255, 100, 100),
            "highlight": (255, 255, 0),
            "text": (0, 0, 0),
            "panel": (220, 220, 220),
            "button": (180, 180, 180),
            "button_hover": (160, 160, 160)
        }
        
        # フォント
        self.font_small = pygame.font.Font(None, 20)
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # レイアウト
        self.board_size = 400
        self.cell_size = self.board_size // 6
        self.board_x = 50
        self.board_y = 100
        
        # ゲーム状態
        self.game = GeisterGame()
        self.current_log = None
        self.current_move_index = 0
        self.playing = False
        self.auto_play_speed = 1000
        self.last_auto_move = 0
        
        # UI状態
        self.selected_game = 0
        self.game_logs = []
        self.leaderboard_data = []
        
        # ボタン領域
        self.buttons = {}
        self._setup_buttons()
    
    def _setup_buttons(self):
        """ボタン領域を設定"""
        button_y = self.height - 80
        button_width = 80
        button_height = 30
        button_spacing = 90
        
        start_x = 50
        
        buttons_info = [
            ("prev_game", "< Game", start_x),
            ("next_game", "Game >", start_x + button_spacing),
            ("reset", "Reset", start_x + button_spacing * 2),
            ("play_pause", "Play", start_x + button_spacing * 3),
            ("prev_move", "< Move", start_x + button_spacing * 4),
            ("next_move", "Move >", start_x + button_spacing * 5),
            ("quit", "Quit", start_x + button_spacing * 6)
        ]
        
        for btn_id, text, x in buttons_info:
            self.buttons[btn_id] = {
                "rect": pygame.Rect(x, button_y, button_width, button_height),
                "text": text,
                "hover": False
            }
    
    def load_game_logs(self, results_file: str):
        """ゲームログと順位表を読み込み（エラー修正版）"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # ゲームログ読み込み（エラー修正）
                raw_logs = data.get("game_logs", [])
                self.game_logs = []
                
                for log in raw_logs:
                    # リスト型のmoveを辞書キーとして使用していた問題を修正
                    fixed_log = {
                        "player_a": log.get("player_a", "Unknown"),
                        "player_b": log.get("player_b", "Unknown"), 
                        "winner": log.get("winner", "Draw"),
                        "turns": log.get("turns", 0),
                        "moves": []
                    }
                    
                    # 手の情報を安全に変換
                    for move_data in log.get("moves", []):
                        if isinstance(move_data, dict):
                            move = move_data.get("move", [])
                            if isinstance(move, list) and len(move) >= 2:
                                try:
                                    move_str = f"{tuple(move[0])} -> {tuple(move[1])}"
                                except:
                                    move_str = str(move)
                            else:
                                move_str = str(move)
                            
                            fixed_move = {
                                "turn": move_data.get("turn", 0),
                                "player": move_data.get("player", "A"),
                                "move_str": move_str,
                                "move": move,
                                "board_state": move_data.get("board_state", [])
                            }
                            fixed_log["moves"].append(fixed_move)
                    
                    self.game_logs.append(fixed_log)
                
                # 順位表読み込み
                self.leaderboard_data = data.get("leaderboard", [])
                
                print(f"✅ {len(self.game_logs)}ゲームのログを読み込み")
                print(f"✅ {len(self.leaderboard_data)}AIの順位データを読み込み")
                
        except Exception as e:
            print(f"❌ ログ読み込みエラー: {e}")
            self.game_logs = []
            self.leaderboard_data = []
    
    def draw_board(self):
        """盤面描画"""
        # ボード背景
        board_rect = pygame.Rect(self.board_x, self.board_y, self.board_size, self.board_size)
        pygame.draw.rect(self.screen, self.colors["board"], board_rect)
        
        # グリッド線
        for i in range(7):
            # 縦線
            start_x = self.board_x + i * self.cell_size
            pygame.draw.line(self.screen, self.colors["grid"], 
                           (start_x, self.board_y), 
                           (start_x, self.board_y + self.board_size), 2)
            
            # 横線
            start_y = self.board_y + i * self.cell_size
            pygame.draw.line(self.screen, self.colors["grid"],
                           (self.board_x, start_y),
                           (self.board_x + self.board_size, start_y), 2)
        
        # 駒描画
        if self.current_log and self.current_move_index <= len(self.current_log["moves"]):
            # 現在の手までの盤面を再構築
            self.game.reset_game()
            
            for i in range(min(self.current_move_index, len(self.current_log["moves"]))):
                move_data = self.current_log["moves"][i]
                move = move_data.get("move", [])
                if isinstance(move, list) and len(move) >= 2:
                    try:
                        from_pos = tuple(move[0])
                        to_pos = tuple(move[1])
                        self.game.make_move(from_pos, to_pos)
                    except:
                        pass
            
            # 駒描画
            board_state = self.game.board
            for y in range(6):
                for x in range(6):
                    if board_state[y, x] != 0:
                        center_x = self.board_x + x * self.cell_size + self.cell_size // 2
                        center_y = self.board_y + y * self.cell_size + self.cell_size // 2
                        
                        color = self.colors["player_a"] if board_state[y, x] == 1 else self.colors["player_b"]
                        pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 3)
                        
                        # プレイヤー表示
                        text = "A" if board_state[y, x] == 1 else "B"
                        text_surf = self.font.render(text, True, (255, 255, 255))
                        text_rect = text_surf.get_rect(center=(center_x, center_y))
                        self.screen.blit(text_surf, text_rect)
    
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
        
        # ゲーム選択
        if self.game_logs:
            text = f"Game: {self.selected_game + 1}/{len(self.game_logs)}"
            text_surf = self.font.render(text, True, self.colors["text"])
            self.screen.blit(text_surf, (panel_x + 10, y_offset))
            y_offset += 25
            
            if self.current_log:
                # 対戦カード
                text = f"{self.current_log['player_a']} vs {self.current_log['player_b']}"
                text_surf = self.font_small.render(text, True, self.colors["text"])
                self.screen.blit(text_surf, (panel_x + 10, y_offset))
                y_offset += 25
                
                # 手数
                max_moves = len(self.current_log['moves'])
                text = f"Move: {self.current_move_index}/{max_moves}"
                text_surf = self.font.render(text, True, self.colors["text"])
                self.screen.blit(text_surf, (panel_x + 10, y_offset))
                y_offset += 25
                
                # 勝者
                winner = self.current_log.get("winner", "Unknown")
                text = f"Winner: {winner}"
                text_surf = self.font.render(text, True, self.colors["text"])
                self.screen.blit(text_surf, (panel_x + 10, y_offset))
                y_offset += 30
        
        # 順位表
        if self.leaderboard_data:
            text = "🏆 Leaderboard:"
            text_surf = self.font.render(text, True, self.colors["text"])
            self.screen.blit(text_surf, (panel_x + 10, y_offset))
            y_offset += 25
            
            for i, entry in enumerate(self.leaderboard_data[:5]):
                name = entry.get('name', 'Unknown')[:12]
                rating = entry.get('rating', 0)
                win_rate = entry.get('win_rate', 0)
                
                text = f"{i+1}. {name}"
                text_surf = self.font_small.render(text, True, self.colors["text"])
                self.screen.blit(text_surf, (panel_x + 10, y_offset))
                
                text = f"R:{rating} W:{win_rate:.0f}%"
                text_surf = self.font_small.render(text, True, self.colors["text"])
                self.screen.blit(text_surf, (panel_x + 150, y_offset))
                
                y_offset += 18
    
    def draw_buttons(self):
        """ボタン描画"""
        mouse_pos = pygame.mouse.get_pos()
        
        for btn_id, btn_data in self.buttons.items():
            rect = btn_data["rect"]
            text = btn_data["text"]
            
            btn_data["hover"] = rect.collidepoint(mouse_pos)
            color = self.colors["button_hover"] if btn_data["hover"] else self.colors["button"]
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.colors["grid"], rect, 2)
            
            text_surf = self.font_small.render(text, True, self.colors["text"])
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
    
    def draw_title(self):
        """タイトル描画"""
        title = "Qugeister AI Battle Viewer (Fixed)"
        title_surf = self.title_font.render(title, True, self.colors["text"])
        title_rect = title_surf.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_surf, title_rect)
        
        status = "Playing" if self.playing else "Paused"
        status_surf = self.font.render(f"Status: {status}", True, self.colors["text"])
        self.screen.blit(status_surf, (50, 60))
    
    def load_game(self, game_index: int):
        """ゲームを読み込み"""
        if 0 <= game_index < len(self.game_logs):
            self.selected_game = game_index
            self.current_log = self.game_logs[game_index]
            self.current_move_index = 0
            self.playing = False
            self.game.reset_game()
    
    def step_forward(self):
        """1手進める"""
        if self.current_log and self.current_move_index < len(self.current_log["moves"]):
            self.current_move_index += 1
    
    def step_backward(self):
        """1手戻る"""
        if self.current_move_index > 0:
            self.current_move_index -= 1
    
    def reset_game(self):
        """ゲームをリセット"""
        self.current_move_index = 0
        self.playing = False
    
    def handle_button_click(self, pos):
        """ボタンクリック処理"""
        for btn_id, btn_data in self.buttons.items():
            if btn_data["rect"].collidepoint(pos):
                if btn_id == "prev_game":
                    if self.selected_game > 0:
                        self.load_game(self.selected_game - 1)
                elif btn_id == "next_game":
                    if self.selected_game < len(self.game_logs) - 1:
                        self.load_game(self.selected_game + 1)
                elif btn_id == "reset":
                    self.reset_game()
                elif btn_id == "play_pause":
                    self.playing = not self.playing
                    self.buttons[btn_id]["text"] = "Pause" if self.playing else "Play"
                elif btn_id == "prev_move":
                    self.step_backward()
                elif btn_id == "next_move":
                    self.step_forward()
                elif btn_id == "quit":
                    return False
                break
        return True
    
    def run(self):
        """メインループ"""
        clock = pygame.time.Clock()
        running = True
        
        if self.game_logs:
            self.load_game(0)
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if not self.handle_button_click(event.pos):
                            running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_LEFT:
                        self.step_backward()
                    elif event.key == pygame.K_RIGHT:
                        self.step_forward()
            
            if self.playing and current_time - self.last_auto_move > self.auto_play_speed:
                if self.current_log and self.current_move_index < len(self.current_log["moves"]):
                    self.step_forward()
                    self.last_auto_move = current_time
                else:
                    self.playing = False
                    self.buttons["play_pause"]["text"] = "Play"
            
            self.screen.fill(self.colors["background"])
            self.draw_title()
            self.draw_board()
            self.draw_info_panel()
            self.draw_buttons()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

def run_fixed_tournament_and_gui():
    """修正版トーナメント+GUI実行"""
    print("🚀 修正版 Qugeister AI Tournament & GUI")
    
    ais = [RandomAI("A"), SimpleAI("A"), AggressiveAI("A"), RandomAI("B"), SimpleAI("B")]
    
    for i, ai in enumerate(ais):
        ai.player_id = f"P{i}"
    
    tournament = TournamentManager()
    for ai in ais:
        tournament.add_participant(ai)
    
    print("🏆 トーナメント実行中...")
    results = tournament.run_round_robin(games_per_pair=2)
    
    print("📊 最終順位:")
    leaderboard = tournament.get_leaderboard()
    for i, entry in enumerate(leaderboard):
        print(f"{i+1:2d}. {entry['name']:12s} | Rating: {entry['rating']:4d} | Win Rate: {entry['win_rate']:5.1f}%")
    
    results_file = "results/fixed_tournament_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    tournament.save_results(results_file)
    
    print(f"💾 結果保存: {results_file}")
    print("🎮 修正版GUI起動中...")
    
    try:
        gui = FixedGameGUI()
        gui.load_game_logs(results_file)
        gui.run()
    except Exception as e:
        print(f"❌ GUI起動エラー: {e}")

if __name__ == "__main__":
    run_fixed_tournament_and_gui()

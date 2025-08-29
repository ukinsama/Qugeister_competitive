#!/usr/bin/env python3
"""
実際のガイスターゲーム統合システム
AIが本物のガイスターをプレイできる環境
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# 既存のゲームエンジンを使用
import sys
sys.path.append('./src/qugeister_competitive')
from game_engine import GeisterGame, GameState

class GeisterAIPlayer(ABC):
    """ガイスターAIプレイヤーの基底クラス"""
    
    def __init__(self, name: str, player_id: str):
        self.name = name
        self.player_id = player_id  # "A" or "B"
        
    @abstractmethod
    def choose_setup(self) -> Dict[Tuple[int, int], str]:
        """初期配置を選択"""
        pass
        
    @abstractmethod
    def choose_move(self, game_state: GameState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """次の手を選択"""
        pass

class CQCNNGeisterAI(GeisterAIPlayer):
    """CQCNN強化学習AIプレイヤー"""
    
    def __init__(self, name: str, player_id: str, model_path: Optional[str] = None):
        super().__init__(name, player_id)
        self.model = None
        self.epsilon = 0.1  # 探索率
        self.device = torch.device('cpu')
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"⚠️ {name}: モデルファイルが見つからないため、ランダムプレイヤーとして動作")
            
    def load_model(self, model_path: str):
        """保存されたモデルを読み込み"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # モデル構造を動的に作成
            if 'estimator_state' in checkpoint:
                from accurate_model_reconstructor import DynamicCQCNNEstimator
                estimator_dict = checkpoint['estimator_state']
                param_shapes = {name: list(param.shape) for name, param in estimator_dict.items()}
                
                self.model = DynamicCQCNNEstimator(param_shapes)
                self.model.load_state_dict(estimator_dict, strict=False)
                self.model.eval()
                
                # 学習された探索率を使用
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                    
                print(f"✅ {self.name}: モデル読み込み成功 (ε={self.epsilon:.3f})")
                
        except Exception as e:
            print(f"❌ {self.name}: モデル読み込み失敗: {e}")
            self.model = None
            
    def choose_setup(self) -> Dict[Tuple[int, int], str]:
        """初期配置を選択（GUI設計を参考に）"""
        positions = [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (3, 1), (4, 1)] if self.player_id == "A" else \
                   [(1, 5), (2, 5), (3, 5), (4, 5), (1, 4), (2, 4), (3, 4), (4, 4)]
                   
        # 戦略的配置（善玉を外側、悪玉を内側）
        if self.player_id == "A":
            setup = {
                (1, 0): "good", (4, 0): "good",  # 外側に善玉
                (2, 0): "bad", (3, 0): "bad",    # 中央に悪玉
                (1, 1): "good", (4, 1): "good",  # 外側に善玉  
                (2, 1): "bad", (3, 1): "bad"     # 中央に悪玉
            }
        else:
            setup = {
                (1, 5): "good", (4, 5): "good",  # 外側に善玉
                (2, 5): "bad", (3, 5): "bad",    # 中央に悪玉
                (1, 4): "good", (4, 4): "good",  # 外側に善玉
                (2, 4): "bad", (3, 4): "bad"     # 中央に悪玉
            }
            
        return setup
        
    def board_to_tensor(self, game_state: GameState) -> torch.Tensor:
        """ゲーム状態をテンソルに変換"""
        # 6x6を5x5に変換（AIが学習した形式）
        board_6x6 = game_state.board
        
        # 中央5x5を抽出
        board_5x5 = board_6x6[0:5, 0:5]
        
        # プレイヤー視点で正規化
        if self.player_id == "B":
            board_5x5 = -board_5x5  # Bの視点では符号反転
            
        # テンソルに変換
        return torch.tensor(board_5x5.flatten(), dtype=torch.float32).unsqueeze(0)
        
    def choose_move(self, game_state: GameState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """次の手を選択"""
        legal_moves = game_state.board if hasattr(game_state, 'board') else []
        
        # GeisterGameから合法手を取得
        game = GeisterGame()
        game.board = game_state.board
        game.player_a_pieces = game_state.player_a_pieces
        game.player_b_pieces = game_state.player_b_pieces
        game.current_player = self.player_id
        
        legal_moves = game.get_legal_moves(self.player_id)
        
        if not legal_moves:
            return None
            
        # ε-greedy戦略
        if self.model and random.random() > self.epsilon:
            # AIによる選択
            try:
                board_tensor = self.board_to_tensor(game_state)
                
                with torch.no_grad():
                    piece_logits, confidence = self.model(board_tensor)
                    
                # 各合法手を評価
                move_scores = []
                for move in legal_moves:
                    from_pos, to_pos = move
                    
                    # 移動先の価値を計算（簡易版）
                    score = self.evaluate_move(game_state, move, piece_logits, confidence)
                    move_scores.append((score, move))
                
                # 最高スコアの手を選択
                move_scores.sort(reverse=True)
                chosen_move = move_scores[0][1]
                
                print(f"🧠 {self.name}: AI選択 {chosen_move[0]}→{chosen_move[1]} (確信度: {confidence[0].item():.3f})")
                return chosen_move
                
            except Exception as e:
                print(f"⚠️ {self.name}: AI選択失敗、ランダム選択: {e}")
        
        # ランダム選択
        chosen_move = random.choice(legal_moves)
        print(f"🎲 {self.name}: ランダム選択 {chosen_move[0]}→{chosen_move[1]}")
        return chosen_move
        
    def evaluate_move(self, game_state: GameState, move: Tuple[Tuple[int, int], Tuple[int, int]], 
                     piece_logits: torch.Tensor, confidence: torch.Tensor) -> float:
        """手の価値を評価"""
        from_pos, to_pos = move
        
        # 基本スコア
        score = 0.0
        
        # 脱出を目指す
        if self.player_id == "A":
            if to_pos[1] == 5:  # 相手陣地に近づく
                score += 2.0
            if to_pos in [(0, 5), (5, 5)]:  # 脱出口
                score += 10.0
        else:
            if to_pos[1] == 0:  # 相手陣地に近づく
                score += 2.0
            if to_pos in [(0, 0), (5, 0)]:  # 脱出口
                score += 10.0
                
        # 相手駒を取る
        opponent_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces
        if to_pos in opponent_pieces:
            score += 3.0
            
        # 中央を避ける（安全性）
        if 1 <= to_pos[0] <= 4 and 2 <= to_pos[1] <= 3:
            score -= 0.5
            
        # 確信度をスコアに反映
        score += confidence[0].item() * 1.0
        
        return score

class RandomGeisterAI(GeisterAIPlayer):
    """ランダムAIプレイヤー（ベースライン）"""
    
    def choose_setup(self) -> Dict[Tuple[int, int], str]:
        """ランダム初期配置"""
        positions = [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (3, 1), (4, 1)] if self.player_id == "A" else \
                   [(1, 5), (2, 5), (3, 5), (4, 5), (1, 4), (2, 4), (3, 4), (4, 4)]
                   
        # ランダムに4個を善玉、4個を悪玉に
        pieces = ["good"] * 4 + ["bad"] * 4
        random.shuffle(pieces)
        
        return {pos: piece for pos, piece in zip(positions, pieces)}
        
    def choose_move(self, game_state: GameState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """ランダム手選択"""
        game = GeisterGame()
        game.board = game_state.board
        game.player_a_pieces = game_state.player_a_pieces
        game.player_b_pieces = game_state.player_b_pieces
        game.current_player = self.player_id
        
        legal_moves = game.get_legal_moves(self.player_id)
        
        if not legal_moves:
            return None
            
        return random.choice(legal_moves)

class RealGeisterMatch:
    """実際のガイスター対戦システム"""
    
    def __init__(self, player_a: GeisterAIPlayer, player_b: GeisterAIPlayer):
        self.player_a = player_a
        self.player_b = player_b
        self.game = GeisterGame()
        self.move_log = []
        
    def play_match(self, verbose: bool = True) -> Dict[str, Any]:
        """対戦を実行"""
        if verbose:
            print(f"🎮 対戦開始: {self.player_a.name} vs {self.player_b.name}")
            print("=" * 60)
            
        # 初期配置
        setup_a = self.player_a.choose_setup()
        setup_b = self.player_b.choose_setup()
        
        # ゲームに配置を反映
        self.game.player_a_pieces = setup_a
        self.game.player_b_pieces = setup_b
        
        # ボード更新
        self.game.board = np.zeros((6, 6), dtype=int)
        for pos in setup_a:
            self.game.board[pos[1], pos[0]] = 1
        for pos in setup_b:
            self.game.board[pos[1], pos[0]] = -1
            
        start_time = time.time()
        
        # メインゲームループ
        while not self.game.game_over and self.game.turn < 100:
            current_ai = self.player_a if self.game.current_player == "A" else self.player_b
            game_state = self.game.get_game_state(self.game.current_player)
            
            if verbose and self.game.turn % 10 == 0:
                print(f"Turn {self.game.turn}: {current_ai.name}のターン")
                
            # 手の選択
            move = current_ai.choose_move(game_state)
            
            if move is None:
                print(f"❌ {current_ai.name}: 合法手なし")
                break
                
            # 手の実行
            success = self.game.make_move(move[0], move[1])
            
            if not success:
                print(f"❌ {current_ai.name}: 不正な手 {move}")
                break
                
            self.move_log.append((self.game.current_player, move))
            
            # 盤面表示（10ターンごと）
            if verbose and self.game.turn % 20 == 0:
                self.game.display_board()
                
        end_time = time.time()
        
        # 結果
        result = {
            'winner': self.game.winner,
            'turns': self.game.turn,
            'duration': end_time - start_time,
            'move_log': self.move_log.copy(),
            'final_state': self.game.get_game_state("A")
        }
        
        if verbose:
            print(f"\n🏆 対戦結果:")
            print(f"  勝者: {result['winner']}")
            print(f"  ターン数: {result['turns']}")
            print(f"  対戦時間: {result['duration']:.2f}秒")
            print(f"  総手数: {len(result['move_log'])}")
            
        return result

class GeisterTournament:
    """ガイスタートーナメントシステム"""
    
    def __init__(self):
        self.players = []
        self.results = []
        
    def add_player(self, player: GeisterAIPlayer):
        """プレイヤー追加"""
        self.players.append(player)
        
    def run_tournament(self, games_per_pair: int = 10) -> Dict[str, Any]:
        """総当たりトーナメント実行"""
        print(f"🏆 ガイスタートーナメント開始")
        print(f"参加者: {len(self.players)} 人")
        print(f"対戦数: {games_per_pair} 回/ペア")
        print("=" * 60)
        
        tournament_results = {
            'participants': [p.name for p in self.players],
            'match_results': [],
            'rankings': [],
            'statistics': {}
        }
        
        total_matches = len(self.players) * (len(self.players) - 1) * games_per_pair
        match_count = 0
        
        # 総当たり戦
        for i, player1 in enumerate(self.players):
            for j, player2 in enumerate(self.players):
                if i != j:
                    wins_1 = 0
                    wins_2 = 0
                    draws = 0
                    
                    for game_num in range(games_per_pair):
                        match_count += 1
                        
                        # プレイヤーAとBを交互に
                        if game_num % 2 == 0:
                            p1_copy = CQCNNGeisterAI(player1.name, "A") if isinstance(player1, CQCNNGeisterAI) else RandomGeisterAI(player1.name, "A")
                            p2_copy = CQCNNGeisterAI(player2.name, "B") if isinstance(player2, CQCNNGeisterAI) else RandomGeisterAI(player2.name, "B")
                        else:
                            p1_copy = CQCNNGeisterAI(player1.name, "B") if isinstance(player1, CQCNNGeisterAI) else RandomGeisterAI(player1.name, "B")
                            p2_copy = CQCNNGeisterAI(player2.name, "A") if isinstance(player2, CQCNNGeisterAI) else RandomGeisterAI(player2.name, "A")
                            
                        match = RealGeisterMatch(p1_copy if p1_copy.player_id == "A" else p2_copy,
                                               p2_copy if p2_copy.player_id == "B" else p1_copy)
                        result = match.play_match(verbose=False)
                        
                        # 勝敗カウント
                        if result['winner'] == "A":
                            if p1_copy.player_id == "A":
                                wins_1 += 1
                            else:
                                wins_2 += 1
                        elif result['winner'] == "B":
                            if p1_copy.player_id == "B":
                                wins_1 += 1
                            else:
                                wins_2 += 1
                        else:
                            draws += 1
                            
                        print(f"進捗: {match_count}/{total_matches} ({match_count/total_matches*100:.1f}%)")
                    
                    match_result = {
                        'player1': player1.name,
                        'player2': player2.name,
                        'wins_1': wins_1,
                        'wins_2': wins_2,
                        'draws': draws,
                        'games': games_per_pair
                    }
                    
                    tournament_results['match_results'].append(match_result)
                    
                    print(f"📊 {player1.name} vs {player2.name}: {wins_1}-{wins_2}-{draws}")
        
        # ランキング計算
        rankings = self.calculate_rankings(tournament_results['match_results'])
        tournament_results['rankings'] = rankings
        
        print(f"\n🏆 最終ランキング:")
        for i, (name, wins, total, winrate) in enumerate(rankings):
            print(f"  {i+1}位: {name} ({wins}/{total}, 勝率{winrate:.1%})")
            
        return tournament_results
    
    def calculate_rankings(self, match_results: List[Dict]) -> List[Tuple[str, int, int, float]]:
        """ランキング計算"""
        player_stats = {}
        
        for result in match_results:
            p1, p2 = result['player1'], result['player2']
            
            if p1 not in player_stats:
                player_stats[p1] = {'wins': 0, 'total': 0}
            if p2 not in player_stats:
                player_stats[p2] = {'wins': 0, 'total': 0}
                
            player_stats[p1]['wins'] += result['wins_1']
            player_stats[p1]['total'] += result['games']
            player_stats[p2]['wins'] += result['wins_2']
            player_stats[p2]['total'] += result['games']
        
        # 勝率でソート
        rankings = []
        for name, stats in player_stats.items():
            winrate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            rankings.append((name, stats['wins'], stats['total'], winrate))
            
        rankings.sort(key=lambda x: x[3], reverse=True)
        return rankings

def main():
    """メイン実行"""
    print("🎮 実際のガイスターゲーム統合システム")
    print("=" * 70)
    
    # 利用可能なモデルを検索
    rl_model = None
    for file in os.listdir('.'):
        if file.startswith('rl_') and file.endswith('.pth'):
            rl_model = file
            break
    
    if rl_model:
        print(f"✅ 強化学習モデル発見: {rl_model}")
    else:
        print("⚠️ 強化学習モデルが見つからません")
    
    # プレイヤー作成
    players = [
        CQCNNGeisterAI("CQCNN_AI", "A", rl_model),
        RandomGeisterAI("Random_1", "B"),
        RandomGeisterAI("Random_2", "A"),
    ]
    
    if rl_model:
        players.append(CQCNNGeisterAI("CQCNN_AI_B", "B", rl_model))
    
    print(f"\n👥 参加プレイヤー: {len(players)} 人")
    for player in players:
        print(f"  - {player.name} ({player.__class__.__name__})")
    
    # 1. サンプル対戦
    print(f"\n🎯 サンプル対戦:")
    sample_match = RealGeisterMatch(players[0], players[1])
    sample_result = sample_match.play_match(verbose=True)
    
    # 2. ミニトーナメント
    print(f"\n🏆 ミニトーナメント実行:")
    tournament = GeisterTournament()
    for player in players:
        tournament.add_player(player)
    
    tournament_results = tournament.run_tournament(games_per_pair=5)
    
    print(f"\n🎉 統合完了!")
    print("✅ AIが実際のガイスターをプレイできる環境が整いました")
    print("✅ 強化学習AIと従来AIの性能比較が可能")
    print("✅ トーナメントシステムで継続的な評価が可能")

if __name__ == "__main__":
    main()
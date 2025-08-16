#!/usr/bin/env python3
"""
CQCNN競技実行システム
実際にゲームエンジンと統合して動作する完全版
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import random
import time
import json

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'qugeister_competitive')
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

print("📂 モジュール読み込み中...")

# ゲームエンジンを読み込み
try:
    with open(os.path.join(src_path, 'game_engine.py'), 'r') as f:
        game_engine_code = f.read()
    exec(game_engine_code)
    
    with open(os.path.join(src_path, 'ai_base.py'), 'r') as f:
        ai_base_code = f.read().replace('from .game_engine', '# from .game_engine')
    exec(ai_base_code)
    
    print("✅ ゲームエンジン読み込み完了")
except Exception as e:
    print(f"⚠️ ゲームエンジン読み込みエラー: {e}")
    print("デモモードで実行します")

# 前のアーティファクトからモジュールをインポート
from cqcnn_competition_framework import (
    InitialPlacementStrategy,
    StandardPlacement,
    DefensivePlacement,
    RandomPlacement,
    MixedPlacement,
    PieceEstimator,
    SimpleCQCNNEstimator,
    AdvancedCQCNNEstimator,
    RandomEstimator,
    QMapGenerator,
    SimpleQMapGenerator,
    StrategicQMapGenerator,
    ActionSelector,
    GreedySelector,
    EpsilonGreedySelector,
    SoftmaxSelector,
    ModuleConfig
)

from fixed_neural_qmap import FixedNeuralQMapGenerator as NeuralQMapGenerator

print("✅ 競技モジュール読み込み完了\n")


# ================================================================================
# ゲームエンジン統合版エージェント
# ================================================================================

class CQCNNGameAgent(BaseAI):
    """ゲームエンジンと統合されたCQCNNエージェント"""
    
    def __init__(self, player_id: str, config: ModuleConfig, name: str = None):
        # BaseAIを継承
        agent_name = name or self._generate_name(config)
        super().__init__(agent_name, player_id)
        
        self.config = config
        self.last_estimations = {}
        self.last_q_map = None
        self.move_history = []
    
    def _generate_name(self, config: ModuleConfig) -> str:
        """エージェント名を生成"""
        return f"CQCNN[{config.placement_strategy.get_strategy_name()[:4]}+" \
               f"{config.piece_estimator.get_estimator_name()[:6]}]"
    
    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置を取得"""
        return self.config.placement_strategy.get_placement(self.player_id)
    
    def get_move(self, game_state: GameState, legal_moves: List) -> Optional[Tuple]:
        """BaseAIインターフェースに準拠した手選択"""
        if not legal_moves:
            return None
        
        try:
            # 1. 敵駒位置を特定
            enemy_positions = self._find_enemy_positions(game_state)
            
            # 2. 敵駒タイプを推定
            if enemy_positions:
                self.last_estimations = self.config.piece_estimator.estimate(
                    game_state.board,
                    enemy_positions,
                    self.player_id
                )
            else:
                self.last_estimations = {}
            
            # 3. Q値マップを生成
            my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
            self.last_q_map = self.config.qmap_generator.generate(
                game_state.board,
                self.last_estimations,
                my_pieces,
                self.player_id
            )
            
            # 4. 行動を選択
            action = self.config.action_selector.select_action(self.last_q_map, legal_moves)
            
            # 履歴記録
            self.move_history.append({
                'turn': game_state.turn,
                'action': action,
                'estimations': len(self.last_estimations),
                'q_max': np.max(self.last_q_map) if self.last_q_map is not None else 0
            })
            
            return action
            
        except Exception as e:
            print(f"⚠️ エラー in {self.name}: {e}")
            return random.choice(legal_moves)
    
    def _find_enemy_positions(self, game_state: GameState) -> List[Tuple[int, int]]:
        """敵駒の位置を特定"""
        enemy_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces
        return list(enemy_pieces.keys())


# ================================================================================
# 実行可能な競技システム
# ================================================================================

class CQCNNCompetitionRunner:
    """実行可能なCQCNN競技システム"""
    
    def __init__(self):
        # 利用可能なモジュール
        self.modules = {
            'placement': [
                StandardPlacement(),
                DefensivePlacement(),
                RandomPlacement(),
                MixedPlacement()
            ],
            'estimator': [
                SimpleCQCNNEstimator(n_qubits=4, n_layers=2),
                AdvancedCQCNNEstimator(n_qubits=6, n_layers=3),
                RandomEstimator()
            ],
            'qmap': [
                SimpleQMapGenerator(),
                StrategicQMapGenerator(),
                NeuralQMapGenerator()
            ],
            'selector': [
                GreedySelector(),
                EpsilonGreedySelector(epsilon=0.1),
                EpsilonGreedySelector(epsilon=0.3),
                SoftmaxSelector(temperature=1.0)
            ]
        }
        
        # 対戦結果記録
        self.match_results = []
        self.agent_stats = {}
    
    def show_modules(self):
        """利用可能なモジュールを表示"""
        print("=" * 70)
        print("🎮 CQCNN競技システム - 利用可能なモジュール")
        print("=" * 70)
        
        print("\n【1. 初期配置戦略】")
        for i, module in enumerate(self.modules['placement']):
            print(f"  {i}: {module.get_strategy_name()}")
        
        print("\n【2. 敵駒推定器】")
        for i, module in enumerate(self.modules['estimator']):
            print(f"  {i}: {module.get_estimator_name()}")
        
        print("\n【3. Q値マップ生成器】")
        for i, module in enumerate(self.modules['qmap']):
            print(f"  {i}: {module.get_generator_name()}")
        
        print("\n【4. 行動選択器】")
        for i, module in enumerate(self.modules['selector']):
            print(f"  {i}: {module.get_selector_name()}")
    
    def create_agent(self, player_id: str, module_indices: Tuple[int, int, int, int], 
                    name: str = None) -> CQCNNGameAgent:
        """モジュール番号を指定してエージェントを作成"""
        placement_idx, estimator_idx, qmap_idx, selector_idx = module_indices
        
        config = ModuleConfig(
            placement_strategy=self.modules['placement'][placement_idx],
            piece_estimator=self.modules['estimator'][estimator_idx],
            qmap_generator=self.modules['qmap'][qmap_idx],
            action_selector=self.modules['selector'][selector_idx]
        )
        
        agent = CQCNNGameAgent(player_id, config, name)
        
        # 統計初期化
        if agent.name not in self.agent_stats:
            self.agent_stats[agent.name] = {
                'games': 0,
                'wins': 0,
                'modules': module_indices
            }
        
        return agent
    
    def run_game(self, agent1: CQCNNGameAgent, agent2: CQCNNGameAgent, 
                verbose: bool = True) -> Dict:
        """1ゲーム実行"""
        game = GeisterGame()
        
        # 初期配置を設定
        game.player_a_pieces = agent1.get_initial_placement()
        game.player_b_pieces = agent2.get_initial_placement()
        
        # ボードに配置
        game.board = np.zeros((6, 6), dtype=int)
        for pos in game.player_a_pieces:
            game.board[pos[1], pos[0]] = 1
        for pos in game.player_b_pieces:
            game.board[pos[1], pos[0]] = -1
        
        if verbose:
            print(f"\n🎮 {agent1.name} vs {agent2.name}")
            print("-" * 50)
        
        move_count = 0
        max_moves = 100
        
        while not game.game_over and move_count < max_moves:
            # 現在のプレイヤーを特定
            current_agent = agent1 if game.current_player == "A" else agent2
            
            # 合法手を取得
            legal_moves = game.get_legal_moves(game.current_player)
            if not legal_moves:
                break
            
            # 手を選択
            game_state = game.get_game_state(game.current_player)
            move = current_agent.get_move(game_state, legal_moves)
            
            if not move:
                break
            
            # 手を実行
            success = game.make_move(move[0], move[1])
            if not success:
                break
            
            move_count += 1
            
            if verbose and move_count <= 5:
                print(f"  Move {move_count}: {game.current_player} {move[0]} → {move[1]}")
        
        # 結果処理
        result = {
            'winner': game.winner,
            'moves': move_count,
            'agent1': agent1.name,
            'agent2': agent2.name
        }
        
        # 統計更新
        self.agent_stats[agent1.name]['games'] += 1
        self.agent_stats[agent2.name]['games'] += 1
        
        if game.winner == "A":
            self.agent_stats[agent1.name]['wins'] += 1
            agent1.record_result(True)
            agent2.record_result(False)
            if verbose:
                print(f"🏆 勝者: {agent1.name}")
        elif game.winner == "B":
            self.agent_stats[agent2.name]['wins'] += 1
            agent1.record_result(False)
            agent2.record_result(True)
            if verbose:
                print(f"🏆 勝者: {agent2.name}")
        else:
            if verbose:
                print("🤝 引き分け")
        
        if verbose:
            print(f"📊 総手数: {move_count}")
        
        self.match_results.append(result)
        return result
    
    def run_tournament(self, agents: List[CQCNNGameAgent], games_per_pair: int = 3):
        """総当たりトーナメント実行"""
        print("\n" + "=" * 70)
        print("🏆 トーナメント開始")
        print("=" * 70)
        
        total_games = 0
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i >= j:  # 重複を避ける
                    continue
                
                print(f"\n【{agent1.name} vs {agent2.name}】")
                
                for game_num in range(games_per_pair):
                    # 先手後手を交代
                    if game_num % 2 == 0:
                        agent1.player_id = "A"
                        agent2.player_id = "B"
                    else:
                        agent1.player_id = "B"
                        agent2.player_id = "A"
                    
                    result = self.run_game(agent1, agent2, verbose=False)
                    total_games += 1
                    
                    winner_name = "Draw"
                    if result['winner'] == "A":
                        winner_name = agent1.name if game_num % 2 == 0 else agent2.name
                    elif result['winner'] == "B":
                        winner_name = agent2.name if game_num % 2 == 0 else agent1.name
                    
                    print(f"  Game {game_num + 1}: {winner_name}")
        
        print(f"\n📊 総ゲーム数: {total_games}")
    
    def show_results(self):
        """結果を表示"""
        print("\n" + "=" * 70)
        print("📊 最終結果")
        print("=" * 70)
        
        # 勝率でソート
        sorted_agents = sorted(
            self.agent_stats.items(),
            key=lambda x: x[1]['wins'] / max(x[1]['games'], 1),
            reverse=True
        )
        
        print("\n【ランキング】")
        for rank, (name, stats) in enumerate(sorted_agents, 1):
            win_rate = stats['wins'] / max(stats['games'], 1) * 100
            modules = stats['modules']
            print(f"{rank}. {name}")
            print(f"   勝率: {win_rate:.1f}% ({stats['wins']}/{stats['games']})")
            print(f"   モジュール: 配置{modules[0]}, 推定{modules[1]}, Q値{modules[2]}, 選択{modules[3]}")


# ================================================================================
# インタラクティブ実行
# ================================================================================

def interactive_mode():
    """インタラクティブモード"""
    runner = CQCNNCompetitionRunner()
    
    print("🎮 CQCNN競技システム - インタラクティブモード")
    print("=" * 70)
    
    while True:
        print("\n【メニュー】")
        print("1. モジュール一覧を表示")
        print("2. カスタムエージェントを作成")
        print("3. クイック対戦（プリセット）")
        print("4. トーナメント実行")
        print("5. 結果表示")
        print("0. 終了")
        
        choice = input("\n選択 (0-5): ").strip()
        
        if choice == "0":
            break
        
        elif choice == "1":
            runner.show_modules()
        
        elif choice == "2":
            print("\n📝 カスタムエージェント作成")
            print("各モジュールの番号を入力してください")
            
            try:
                placement = int(input("初期配置 (0-3): "))
                estimator = int(input("推定器 (0-2): "))
                qmap = int(input("Q値生成 (0-2): "))
                selector = int(input("行動選択 (0-3): "))
                name = input("エージェント名 (省略可): ").strip() or None
                
                agent = runner.create_agent("A", (placement, estimator, qmap, selector), name)
                print(f"✅ エージェント作成: {agent.name}")
                
            except (ValueError, IndexError) as e:
                print(f"❌ エラー: {e}")
        
        elif choice == "3":
            print("\n⚡ クイック対戦")
            
            # プリセットエージェント
            agent1 = runner.create_agent("A", (0, 0, 0, 0), "標準型")
            agent2 = runner.create_agent("B", (1, 1, 1, 1), "高度型")
            
            runner.run_game(agent1, agent2)
        
        elif choice == "4":
            print("\n🏆 トーナメント設定")
            
            # プリセットエージェント群
            agents = [
                runner.create_agent("A", (0, 0, 0, 0), "標準Simple"),
                runner.create_agent("A", (1, 1, 1, 1), "守備Advanced"),
                runner.create_agent("A", (2, 2, 2, 2), "ランダム型"),
                runner.create_agent("A", (0, 1, 1, 0), "ハイブリッド")
            ]
            
            games = int(input("各ペアのゲーム数 (1-10): ") or "3")
            runner.run_tournament(agents, games)
        
        elif choice == "5":
            runner.show_results()
    
    print("\n👋 終了します")


def quick_demo():
    """クイックデモ実行"""
    print("🚀 CQCNN競技システム - クイックデモ")
    print("=" * 70)
    
    runner = CQCNNCompetitionRunner()
    
    # モジュール表示
    runner.show_modules()
    
    print("\n" + "=" * 70)
    print("📝 エージェント作成")
    print("=" * 70)
    
    # 4種類のエージェントを作成
    agents = [
        runner.create_agent("A", (0, 0, 0, 0), "標準型"),      # 全て基本
        runner.create_agent("A", (1, 1, 1, 1), "高度型"),      # 全て高度
        runner.create_agent("A", (2, 0, 1, 2), "混合型"),      # 混合
        runner.create_agent("A", (0, 1, 2, 0), "実験型")       # 実験的組み合わせ
    ]
    
    for agent in agents:
        modules = runner.agent_stats[agent.name]['modules']
        print(f"\n{agent.name}:")
        print(f"  配置: {runner.modules['placement'][modules[0]].get_strategy_name()}")
        print(f"  推定: {runner.modules['estimator'][modules[1]].get_estimator_name()}")
        print(f"  Q値: {runner.modules['qmap'][modules[2]].get_generator_name()}")
        print(f"  選択: {runner.modules['selector'][modules[3]].get_selector_name()}")
    
    # トーナメント実行
    runner.run_tournament(agents, games_per_pair=2)
    
    # 結果表示
    runner.show_results()
    
    print("\n✅ デモ完了！")


# ================================================================================
# メイン実行
# ================================================================================

def main():
    """メイン実行"""
    print("=" * 70)
    print("🎮 CQCNN競技実行システム")
    print("=" * 70)
    print("\n実行モードを選択してください:")
    print("1. クイックデモ（自動実行）")
    print("2. インタラクティブモード（対話型）")
    print("3. カスタム対戦（上級者向け）")
    
    mode = input("\n選択 (1-3): ").strip()
    
    if mode == "1":
        quick_demo()
    elif mode == "2":
        interactive_mode()
    elif mode == "3":
        print("\n📝 カスタム対戦モード")
        runner = CQCNNCompetitionRunner()
        runner.show_modules()
        
        print("\n2つのエージェントを作成して対戦させます")
        
        # エージェント1
        print("\n【エージェント1】")
        p1 = int(input("配置 (0-3): "))
        e1 = int(input("推定 (0-2): "))
        q1 = int(input("Q値 (0-2): "))
        s1 = int(input("選択 (0-3): "))
        
        # エージェント2  
        print("\n【エージェント2】")
        p2 = int(input("配置 (0-3): "))
        e2 = int(input("推定 (0-2): "))
        q2 = int(input("Q値 (0-2): "))
        s2 = int(input("選択 (0-3): "))
        
        agent1 = runner.create_agent("A", (p1, e1, q1, s1))
        agent2 = runner.create_agent("B", (p2, e2, q2, s2))
        
        games = int(input("\nゲーム数 (1-10): ") or "5")
        
        for i in range(games):
            print(f"\n--- Game {i+1}/{games} ---")
            runner.run_game(agent1, agent2)
        
        runner.show_results()
    else:
        print("無効な選択です")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 中断されました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()

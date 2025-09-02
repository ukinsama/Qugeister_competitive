#!/usr/bin/env python3
"""
段階3: トーナメント管理システム
学習済みモデル同士の対戦・分析・ランキング
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import itertools
import random

# 親ディレクトリから学習済みモデル読み込み
sys.path.append('..')

class TrainedModelLoader:
    """学習済みモデル読み込みクラス"""
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            # 複数のディレクトリを検索
            possible_dirs = [
                "../learning/trained_models",
                "../trained_models", 
                "./trained_models"
            ]
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    models_dir = dir_path
                    break
            if models_dir is None:
                models_dir = "../learning/trained_models"
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_metadata = {}
    
    def discover_models(self):
        """学習済みモデル発見"""
        if not os.path.exists(self.models_dir):
            print(f"⚠️ モデルディレクトリが見つかりません: {self.models_dir}")
            return []
        
        models = []
        for model_dir in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_dir)
            metadata_path = os.path.join(model_path, "metadata.json")
            
            if os.path.isdir(model_path) and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # モデルファイル存在確認
                    model_file = os.path.join(model_path, "model.pth")
                    if os.path.exists(model_file):
                        metadata['model_path'] = model_file
                        metadata['model_dir'] = model_path
                        models.append(metadata)
                        self.model_metadata[metadata['model_id']] = metadata
                        print(f"✅ 発見: {metadata['model_id']}")
                    
                except Exception as e:
                    print(f"⚠️ メタデータ読み込みエラー {model_dir}: {e}")
        
        print(f"📁 発見された学習済みモデル: {len(models)}個")
        return models
    
    def load_model(self, model_id):
        """モデル読み込み"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        if model_id not in self.model_metadata:
            raise ValueError(f"モデルが見つかりません: {model_id}")
        
        metadata = self.model_metadata[model_id]
        model_file = metadata['model_path']
        
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # モデル復元（簡略化）
            model_wrapper = TrainedModelWrapper(
                model_id=model_id,
                metadata=metadata,
                checkpoint=checkpoint
            )
            
            self.loaded_models[model_id] = model_wrapper
            print(f"🧠 モデル読み込み完了: {model_id}")
            
            return model_wrapper
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー {model_id}: {e}")
            return None

class TrainedModelWrapper:
    """学習済みモデルラッパー"""
    
    def __init__(self, model_id, metadata, checkpoint):
        self.model_id = model_id
        self.metadata = metadata
        self.checkpoint = checkpoint
        
        # 基本情報抽出
        self.ai_name = metadata.get('ai_name', model_id)
        self.creation_time = metadata.get('creation_time', 'unknown')
        self.total_parameters = metadata.get('model_architecture', {}).get('total_parameters', 0)
        
        # レシピ設定抽出
        self.recipe_config = metadata.get('recipe_config', {})
        self.learning_rate = self.recipe_config.get('learning_rate', 0.001)
        self.n_qubits = self.recipe_config.get('n_qubits', 4)
        
        # 戦闘統計
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_reward = 0
        
        # 戦略強度（メタデータから推定）
        self.strategy_strength = self._analyze_strategy_strength()
        
        print(f"🤖 {self.ai_name}: パラメータ数={self.total_parameters}, 戦略強度={self.strategy_strength:.3f}")
    
    def _analyze_strategy_strength(self):
        """メタデータから戦略強度を分析"""
        strength = 0.5  # ベース
        
        # 学習率から推定
        lr = self.learning_rate
        if lr > 0.01:
            strength += 0.15  # 高学習率は攻撃的
        elif lr < 0.0001:
            strength -= 0.1   # 低学習率は慎重
        
        # 量子ビット数から推定
        qubits = self.n_qubits
        if qubits > 6:
            strength += 0.1  # 複雑なモデル
        elif qubits < 3:
            strength -= 0.1  # シンプルなモデル
        
        # パラメーター変化量から推定
        training_results = self.metadata.get('training_results', {})
        param_change = training_results.get('parameter_change_total', 0)
        if param_change > 1.0:
            strength += 0.1  # よく学習されたモデル
        elif param_change < 0.1:
            strength -= 0.1  # 学習不足
        
        # 最終損失から推定
        final_loss = training_results.get('final_loss', 1.0)
        if final_loss < 0.1:
            strength += 0.1  # 低損失は良い性能
        elif final_loss > 1.0:
            strength -= 0.1  # 高損失は悪い性能
        
        return max(0.1, min(0.9, strength))
    
    def battle(self, opponent):
        """対戦実行"""
        # 戦略強度ベースの対戦（簡略化）
        my_roll = random.uniform(0.8, 1.2) * self.strategy_strength
        opp_roll = random.uniform(0.8, 1.2) * opponent.strategy_strength
        
        # 学習率差による補正
        lr_factor = np.log10(self.learning_rate) - np.log10(opponent.learning_rate)
        my_roll += lr_factor * 0.05
        
        # パラメータ数による補正
        param_ratio = self.total_parameters / max(1, opponent.total_parameters)
        if param_ratio > 1.2:
            my_roll += 0.02  # 大きいモデルの僅かな優位
        elif param_ratio < 0.8:
            my_roll -= 0.02  # 小さいモデルの僅かな劣勢
        
        if abs(my_roll - opp_roll) < 0.05:
            return 'draw'
        elif my_roll > opp_roll:
            return 'win'
        else:
            return 'loss'
    
    def record_result(self, result, reward):
        """結果記録"""
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1
        
        self.total_reward += reward
    
    def get_win_rate(self):
        """勝率計算"""
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0
    
    def get_stats(self):
        """統計情報取得"""
        total_games = self.wins + self.losses + self.draws
        return {
            'model_id': self.model_id,
            'ai_name': self.ai_name,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'games_played': total_games,
            'win_rate': self.get_win_rate(),
            'avg_reward': self.total_reward / max(1, total_games),
            'strategy_strength': self.strategy_strength,
            'learning_rate': self.learning_rate,
            'n_qubits': self.n_qubits,
            'total_parameters': self.total_parameters,
            'creation_time': self.creation_time
        }

class TournamentManager:
    """トーナメント管理クラス"""
    
    def __init__(self, games_per_match=30):
        self.games_per_match = games_per_match
        self.loader = TrainedModelLoader()
        self.participants = {}
        self.match_results = {}
        self.tournament_results = []
        
        print(f"🏆 トーナメント管理システム初期化")
        print(f"  1対戦あたりゲーム数: {games_per_match}")
    
    def discover_and_load_models(self):
        """モデル発見・読み込み"""
        models = self.loader.discover_models()
        
        loaded_count = 0
        for model_metadata in models:
            model_id = model_metadata['model_id']
            try:
                model_wrapper = self.loader.load_model(model_id)
                if model_wrapper:
                    self.participants[model_id] = model_wrapper
                    loaded_count += 1
            except Exception as e:
                print(f"❌ {model_id} 読み込み失敗: {e}")
        
        print(f"📚 読み込み済みモデル: {loaded_count}個")
        return loaded_count
    
    def run_match(self, model1_id, model2_id):
        """2モデル間の対戦"""
        model1 = self.participants[model1_id]
        model2 = self.participants[model2_id]
        
        print(f"⚔️ {model1.ai_name} vs {model2.ai_name}")
        
        match_stats = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}
        
        for game in range(self.games_per_match):
            result = model1.battle(model2)
            
            if result == 'win':
                match_stats['model1_wins'] += 1
                model1.record_result('win', 1)
                model2.record_result('loss', -1)
            elif result == 'loss':
                match_stats['model2_wins'] += 1
                model1.record_result('loss', -1)
                model2.record_result('win', 1)
            else:
                match_stats['draws'] += 1
                model1.record_result('draw', 0)
                model2.record_result('draw', 0)
        
        # 結果記録
        total_games = sum(match_stats.values())
        win_rate1 = match_stats['model1_wins'] / total_games
        win_rate2 = match_stats['model2_wins'] / total_games
        
        match_result = {
            'model1_id': model1_id,
            'model2_id': model2_id,
            'model1_name': model1.ai_name,
            'model2_name': model2.ai_name,
            'model1_wins': match_stats['model1_wins'],
            'model2_wins': match_stats['model2_wins'],
            'draws': match_stats['draws'],
            'total_games': total_games,
            'model1_win_rate': win_rate1,
            'model2_win_rate': win_rate2
        }
        
        self.match_results[(model1_id, model2_id)] = match_result
        
        print(f"  結果: {model1.ai_name} {win_rate1:.3f} - {win_rate2:.3f} {model2.ai_name}")
        
        return match_result
    
    def run_tournament(self):
        """総当たりトーナメント実行"""
        model_ids = list(self.participants.keys())
        total_matches = len(list(itertools.combinations(model_ids, 2)))
        
        if len(model_ids) < 2:
            print("❌ 対戦には最低2つのモデルが必要です")
            return
        
        print(f"\n🚀 総当たりトーナメント開始")
        print(f"参加モデル: {len(model_ids)}個")
        print(f"総対戦数: {total_matches}")
        print("=" * 60)
        
        match_count = 0
        for model1_id, model2_id in itertools.combinations(model_ids, 2):
            match_count += 1
            print(f"[{match_count}/{total_matches}]", end=" ")
            self.run_match(model1_id, model2_id)
        
        print(f"✅ トーナメント完了!")
    
    def generate_rankings(self):
        """ランキング生成"""
        rankings = []
        
        for model_wrapper in self.participants.values():
            stats = model_wrapper.get_stats()
            rankings.append(stats)
        
        # 勝率でソート
        rankings.sort(key=lambda x: x['win_rate'], reverse=True)
        
        # 順位付け
        for rank, stats in enumerate(rankings, 1):
            stats['rank'] = rank
        
        self.tournament_results = rankings
        return rankings
    
    def print_rankings(self):
        """ランキング表示"""
        rankings = self.generate_rankings()
        
        print(f"\n🏆 トーナメント結果ランキング")
        print("=" * 100)
        print(f"{'順位':<4} {'AI名':<25} {'勝率':<8} {'勝利':<6} {'敗北':<6} {'引分':<6} {'学習率':<10} {'量子':<6}")
        print("-" * 100)
        
        for stats in rankings:
            print(f"{stats['rank']:<4} {stats['ai_name']:<25} {stats['win_rate']:<8.3f} "
                  f"{stats['wins']:<6} {stats['losses']:<6} {stats['draws']:<6} "
                  f"{stats['learning_rate']:<10.4f} {stats['n_qubits']:<6}")
        
        return rankings
    
    def analyze_learning_effectiveness(self):
        """学習効果分析"""
        rankings = self.tournament_results if self.tournament_results else self.generate_rankings()
        
        print(f"\n🔬 学習効果分析")
        print("=" * 60)
        
        # 学習率と性能の相関
        lr_performance = [(stats['learning_rate'], stats['win_rate']) for stats in rankings]
        lr_performance.sort()
        
        print("📊 学習率別性能:")
        for lr, win_rate in lr_performance:
            print(f"  lr={lr:.4f} → 勝率={win_rate:.3f}")
        
        # 量子ビット数と性能の相関
        qubit_performance = {}
        for stats in rankings:
            qubits = stats['n_qubits']
            if qubits not in qubit_performance:
                qubit_performance[qubits] = []
            qubit_performance[qubits].append(stats['win_rate'])
        
        print(f"\n📊 量子ビット数別平均性能:")
        for qubits in sorted(qubit_performance.keys()):
            avg_performance = np.mean(qubit_performance[qubits])
            count = len(qubit_performance[qubits])
            print(f"  {qubits}qubit: 平均勝率={avg_performance:.3f} ({count}個)")
        
        # トップ性能モデル分析
        if rankings:
            top_model = rankings[0]
            print(f"\n🏆 最高性能モデル分析:")
            print(f"  モデル: {top_model['ai_name']}")
            print(f"  勝率: {top_model['win_rate']:.3f}")
            print(f"  学習率: {top_model['learning_rate']}")
            print(f"  量子ビット: {top_model['n_qubits']}")
            print(f"  パラメータ数: {top_model['total_parameters']}")
            print(f"  戦略強度: {top_model['strategy_strength']:.3f}")
    
    def save_results(self, results_dir="tournament_results"):
        """結果保存"""
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        rankings = self.tournament_results if self.tournament_results else self.generate_rankings()
        
        # CSV保存
        df = pd.DataFrame(rankings)
        csv_path = f"{results_dir}/tournament_rankings_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 詳細JSON保存
        detailed_results = {
            'timestamp': timestamp,
            'tournament_info': {
                'participants': len(self.participants),
                'games_per_match': self.games_per_match,
                'total_matches': len(self.match_results)
            },
            'rankings': rankings,
            'match_results': {f"{k[0]}_vs_{k[1]}": v for k, v in self.match_results.items()},
            'analysis': {
                'top_performer': rankings[0]['ai_name'] if rankings else 'None',
                'avg_win_rate': np.mean([r['win_rate'] for r in rankings]) if rankings else 0,
                'learning_rate_range': [
                    min(r['learning_rate'] for r in rankings) if rankings else 0,
                    max(r['learning_rate'] for r in rankings) if rankings else 0
                ]
            }
        }
        
        json_path = f"{results_dir}/tournament_detailed_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 結果保存完了:")
        print(f"  📊 CSV: {csv_path}")
        print(f"  📋 詳細: {json_path}")
        
        return {'csv_path': csv_path, 'json_path': json_path}

def main():
    """メイン実行"""
    print("🏆 段階3: トーナメント管理システム")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description="学習済みモデル対戦システム")
    parser.add_argument("--games", type=int, default=30, help="1対戦あたりゲーム数")
    parser.add_argument("--list-only", action="store_true", help="モデル一覧表示のみ")
    
    args = parser.parse_args()
    
    # トーナメント管理初期化
    tournament = TournamentManager(games_per_match=args.games)
    
    # モデル発見・読み込み
    model_count = tournament.discover_and_load_models()
    
    if model_count == 0:
        print("❌ 学習済みモデルが見つかりません")
        print("💡 先に段階2で学習を実行してください:")
        print("   cd ../learning && python recipe_trainer.py --batch")
        return
    
    if args.list_only:
        print(f"\n📚 読み込み済みモデル一覧:")
        for model_id, model in tournament.participants.items():
            stats = model.get_stats()
            print(f"🤖 {stats['ai_name']}")
            print(f"   ID: {model_id}")
            print(f"   作成: {stats['creation_time']}")
            print(f"   パラメータ: {stats['total_parameters']}")
            print(f"   学習率: {stats['learning_rate']}")
            print()
        return
    
    if model_count < 2:
        print("❌ 対戦には最低2つのモデルが必要です")
        return
    
    # トーナメント実行
    tournament.run_tournament()
    
    # 結果表示
    tournament.print_rankings()
    
    # 分析
    tournament.analyze_learning_effectiveness()
    
    # 結果保存
    tournament.save_results()
    
    print(f"\n🎉 トーナメント完了!")

if __name__ == "__main__":
    main()
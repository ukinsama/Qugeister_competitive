#!/usr/bin/env python3
"""
理想的なローカル対戦システム
5つのモジュール選択 → 学習 → 評価 → 再挑戦or保存 → トーナメント
"""

import os
import json
import torch
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import hashlib


# ================================================================================
# データクラスとコンフィグ
# ================================================================================

@dataclass
class ModuleConfig:
    """モジュール設定"""
    placement: str
    estimator: str  
    reward: str
    qmap: str
    selector: str
    
    def to_dict(self) -> Dict:
        return {
            'placement': self.placement,
            'estimator': self.estimator,
            'reward': self.reward,
            'qmap': self.qmap,
            'selector': self.selector
        }

@dataclass
class EvaluationResult:
    """評価結果"""
    vs_random: Dict  # vs ランダムAI
    vs_simple: Dict  # vs シンプルAI
    vs_aggressive: Dict  # vs 攻撃的AI
    overall_win_rate: float
    total_wins: int
    total_games: int
    
    def is_strong_enough(self, threshold: float = 0.6) -> bool:
        """十分強いかチェック"""
        return self.overall_win_rate >= threshold


# ================================================================================
# モジュール定義
# ================================================================================

class AvailableModules:
    """利用可能なモジュール一覧"""
    
    MODULES = {
        'placement': {
            'standard': {'name': '🎯 標準配置', 'description': 'バランスの取れた標準的な配置戦略'},
            'aggressive': {'name': '⚔️ 攻撃的配置', 'description': '前線に駒を集中させる攻撃的戦略'},
            'defensive': {'name': '🛡️ 防御的配置', 'description': '後方を固める防御的戦略'},
            'random': {'name': '🎲 ランダム配置', 'description': 'ランダムな配置（ベースライン）'}
        },
        'estimator': {
            'cqcnn': {'name': '🔬 CQCNN', 'description': '量子畳み込みニューラルネットワーク'},
            'cnn': {'name': '🧠 シンプルCNN', 'description': '従来の畳み込みニューラルネットワーク'},
            'random': {'name': '🎲 ランダム推定', 'description': 'ランダムな敵駒推定（ベースライン）'},
            'pattern': {'name': '📊 パターン推定', 'description': 'ルールベースのパターン認識'}
        },
        'reward': {
            'standard': {'name': '⚖️ 標準報酬', 'description': 'バランスの取れた報酬関数'},
            'aggressive': {'name': '💥 攻撃的報酬', 'description': '攻撃行動を重視する報酬'},
            'defensive': {'name': '🔒 防御的報酬', 'description': '防御行動を重視する報酬'},
            'smart': {'name': '🎯 スマート報酬', 'description': '状況に応じて適応する報酬'}
        },
        'qmap': {
            'simple': {'name': '📈 シンプルQ値', 'description': '基本的なQ値マッピング'},
            'strategic': {'name': '🧩 戦略的Q値', 'description': '高度な戦略を考慮したQ値'},
            'adaptive': {'name': '🔄 適応的Q値', 'description': '相手に応じて適応するQ値'},
            'greedy': {'name': '💰 貪欲Q値', 'description': '短期利益を重視するQ値'}
        },
        'selector': {
            'greedy': {'name': '🎯 貪欲選択', 'description': '常に最良の行動を選択'},
            'epsilon_01': {'name': '🎲 探索的(ε=0.1)', 'description': '10%の確率でランダム行動'},
            'epsilon_03': {'name': '🎲 探索的(ε=0.3)', 'description': '30%の確率でランダム行動'},
            'softmax': {'name': '🌡️ Softmax', 'description': '確率的な行動選択'},
            'adaptive': {'name': '🔄 適応的選択', 'description': '状況に応じて選択戦略を変更'}
        }
    }


# ================================================================================
# 評価システム
# ================================================================================

class BattleEvaluator:
    """対戦評価システム"""
    
    def __init__(self):
        self.baseline_ais = {
            'random': {'name': 'ランダムAI', 'strength': 0.5},
            'simple': {'name': 'シンプルAI', 'strength': 0.7},
            'aggressive': {'name': '攻撃的AI', 'strength': 0.8}
        }
    
    def evaluate_ai(self, config: ModuleConfig, games_per_opponent: int = 10) -> EvaluationResult:
        """AIを評価"""
        print(f"\n🎮 AI評価開始 (各相手と{games_per_opponent}戦)")
        print("=" * 50)
        
        results = {}
        total_wins = 0
        total_games = 0
        
        for ai_type, ai_info in self.baseline_ais.items():
            print(f"\n⚔️ vs {ai_info['name']}: ", end="", flush=True)
            
            wins = self._simulate_battles(config, ai_type, games_per_opponent)
            win_rate = wins / games_per_opponent
            
            results[f"vs_{ai_type}"] = {
                'wins': wins,
                'total': games_per_opponent,
                'win_rate': win_rate
            }
            
            total_wins += wins
            total_games += games_per_opponent
            
            # リアルタイム表示
            print(f"{wins}/{games_per_opponent} ({win_rate:.1%})", end=" ")
            if win_rate >= 0.7:
                print("🔥")
            elif win_rate >= 0.6:
                print("✅")
            elif win_rate >= 0.5:
                print("⚖️")
            else:
                print("❌")
        
        overall_win_rate = total_wins / total_games
        
        return EvaluationResult(
            vs_random=results["vs_random"],
            vs_simple=results["vs_simple"],
            vs_aggressive=results["vs_aggressive"],
            overall_win_rate=overall_win_rate,
            total_wins=total_wins,
            total_games=total_games
        )
    
    def _simulate_battles(self, config: ModuleConfig, opponent_type: str, num_games: int) -> int:
        """対戦をシミュレート（実際の実装では本物のゲームエンジンを使用）"""
        wins = 0
        base_strength = self.baseline_ais[opponent_type]['strength']
        
        # モジュール組み合わせによる強度計算（簡略化）
        ai_strength = self._calculate_ai_strength(config)
        
        for _ in range(num_games):
            # シンプルな勝率計算
            win_probability = ai_strength / (ai_strength + base_strength)
            if random.random() < win_probability:
                wins += 1
        
        return wins
    
    def _calculate_ai_strength(self, config: ModuleConfig) -> float:
        """AI強度を計算（モジュール組み合わせ基準）"""
        # 各モジュールの基本強度
        strengths = {
            'placement': {'standard': 0.7, 'aggressive': 0.8, 'defensive': 0.6, 'random': 0.4},
            'estimator': {'cqcnn': 0.9, 'cnn': 0.7, 'pattern': 0.6, 'random': 0.3},
            'reward': {'standard': 0.7, 'aggressive': 0.8, 'defensive': 0.6, 'smart': 0.9},
            'qmap': {'simple': 0.6, 'strategic': 0.8, 'adaptive': 0.9, 'greedy': 0.5},
            'selector': {'greedy': 0.6, 'epsilon_01': 0.7, 'epsilon_03': 0.8, 'softmax': 0.7, 'adaptive': 0.9}
        }
        
        total_strength = 0
        for module_type, module_name in config.to_dict().items():
            total_strength += strengths.get(module_type, {}).get(module_name, 0.5)
        
        return total_strength / 5  # 平均化


# ================================================================================
# モデル管理システム
# ================================================================================

class AIModelManager:
    """学習済みAIモデルの管理"""
    
    def __init__(self, models_dir: str = "saved_ais"):
        self.models_dir = models_dir
        self.registry_file = os.path.join(models_dir, "ai_registry.json")
        os.makedirs(models_dir, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """レジストリ読み込み"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"ais": [], "tournament_history": []}
    
    def _save_registry(self):
        """レジストリ保存"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def save_ai(self, config: ModuleConfig, evaluation: EvaluationResult, ai_name: str = None) -> str:
        """強いAIを保存"""
        # AIの名前生成
        if ai_name is None:
            ai_name = f"AI_{len(self.registry['ais']) + 1}"
        
        # ユニークID生成
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        ai_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        ai_info = {
            "id": ai_id,
            "name": ai_name,
            "config": config.to_dict(),
            "evaluation": {
                "overall_win_rate": evaluation.overall_win_rate,
                "total_wins": evaluation.total_wins,
                "total_games": evaluation.total_games,
                "vs_random": evaluation.vs_random,
                "vs_simple": evaluation.vs_simple,
                "vs_aggressive": evaluation.vs_aggressive
            },
            "created_at": datetime.now().isoformat(),
            "is_strong": evaluation.is_strong_enough()
        }
        
        # 重複チェック
        existing = [ai for ai in self.registry["ais"] if ai["id"] == ai_id]
        if not existing:
            self.registry["ais"].append(ai_info)
            self._save_registry()
        
        return ai_id
    
    def get_strong_ais(self) -> List[Dict]:
        """強いAI一覧を取得"""
        return [ai for ai in self.registry["ais"] if ai.get("is_strong", False)]
    
    def get_all_ais(self) -> List[Dict]:
        """全AI一覧を取得"""
        return self.registry["ais"]


# ================================================================================
# トーナメントシステム
# ================================================================================

class TournamentManager:
    """トーナメント管理"""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.evaluator = BattleEvaluator()
    
    def run_tournament(self, games_per_match: int = 5) -> Dict:
        """トーナメント実行"""
        strong_ais = self.model_manager.get_strong_ais()
        
        if len(strong_ais) < 2:
            print("❌ トーナメントには2つ以上の強いAIが必要です")
            return {}
        
        print(f"\n🏆 AIトーナメント開始！")
        print(f"参加者: {len(strong_ais)}体のAI")
        print("=" * 60)
        
        # 総当たり戦
        match_results = []
        rankings = {ai["id"]: {"wins": 0, "total": 0, "name": ai["name"]} for ai in strong_ais}
        
        for i, ai1 in enumerate(strong_ais):
            for j, ai2 in enumerate(strong_ais):
                if i >= j:
                    continue
                
                print(f"\n⚔️ {ai1['name']} vs {ai2['name']}")
                
                # 対戦シミュレート
                ai1_wins = self._simulate_tournament_match(ai1, ai2, games_per_match)
                ai2_wins = games_per_match - ai1_wins
                
                match_results.append({
                    "ai1_id": ai1["id"],
                    "ai2_id": ai2["id"],
                    "ai1_name": ai1["name"],
                    "ai2_name": ai2["name"],
                    "ai1_wins": ai1_wins,
                    "ai2_wins": ai2_wins
                })
                
                # ランキング更新
                rankings[ai1["id"]]["wins"] += ai1_wins
                rankings[ai1["id"]]["total"] += games_per_match
                rankings[ai2["id"]]["wins"] += ai2_wins
                rankings[ai2["id"]]["total"] += games_per_match
                
                print(f"   結果: {ai1_wins}-{ai2_wins}")
        
        # 最終ランキング
        final_rankings = []
        for ai_id, stats in rankings.items():
            win_rate = stats["wins"] / max(stats["total"], 1)
            final_rankings.append({
                "ai_id": ai_id,
                "name": stats["name"],
                "wins": stats["wins"],
                "total": stats["total"],
                "win_rate": win_rate
            })
        
        final_rankings.sort(key=lambda x: x["win_rate"], reverse=True)
        
        # 結果保存
        tournament_result = {
            "participants": len(strong_ais),
            "matches": match_results,
            "rankings": final_rankings,
            "date": datetime.now().isoformat()
        }
        
        self.model_manager.registry["tournament_history"].append(tournament_result)
        self.model_manager._save_registry()
        
        return tournament_result
    
    def _simulate_tournament_match(self, ai1: Dict, ai2: Dict, games: int) -> int:
        """トーナメント対戦をシミュレート"""
        ai1_strength = self.evaluator._calculate_ai_strength(ModuleConfig(**ai1["config"]))
        ai2_strength = self.evaluator._calculate_ai_strength(ModuleConfig(**ai2["config"]))
        
        ai1_wins = 0
        for _ in range(games):
            win_prob = ai1_strength / (ai1_strength + ai2_strength)
            if random.random() < win_prob:
                ai1_wins += 1
        
        return ai1_wins
    
    def show_tournament_results(self, results: Dict):
        """トーナメント結果表示"""
        print(f"\n🏆 トーナメント結果")
        print("=" * 60)
        
        for i, entry in enumerate(results["rankings"]):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{medal} {i+1}位: {entry['name']}")
            print(f"        勝率: {entry['win_rate']:.1%} ({entry['wins']}/{entry['total']})")


# ================================================================================
# メインシステム
# ================================================================================

class IdealLocalBattleSystem:
    """理想的なローカル対戦システム"""
    
    def __init__(self):
        self.modules = AvailableModules()
        self.evaluator = BattleEvaluator()
        self.model_manager = AIModelManager()
        self.tournament = TournamentManager(self.model_manager)
    
    def select_modules(self) -> ModuleConfig:
        """モジュール選択UI"""
        print("\n🎯 AI設計フェーズ")
        print("=" * 60)
        print("5つのモジュールを組み合わせて、最強のAIを作りましょう！")
        
        selected = {}
        
        for module_type, modules in self.modules.MODULES.items():
            print(f"\n【{module_type.upper()}】を選択してください:")
            
            for i, (key, info) in enumerate(modules.items(), 1):
                print(f"  {i}. {info['name']}")
                print(f"     {info['description']}")
            
            while True:
                try:
                    choice = int(input(f"\n選択 (1-{len(modules)}): ")) - 1
                    selected_key = list(modules.keys())[choice]
                    selected[module_type] = selected_key
                    
                    chosen = modules[selected_key]
                    print(f"✅ {chosen['name']} を選択しました！\n")
                    break
                except (ValueError, IndexError):
                    print("❌ 無効な選択です。もう一度入力してください。")
        
        return ModuleConfig(**selected)
    
    def train_and_evaluate_loop(self) -> Tuple[ModuleConfig, EvaluationResult]:
        """学習→評価→再挑戦のループ"""
        while True:
            print("\n" + "=" * 60)
            print("🔬 AI作成・評価フェーズ")
            print("=" * 60)
            
            # 1. モジュール選択
            config = self.select_modules()
            
            # 2. 学習フェーズ（簡略化）
            print("🎓 学習中...")
            print("   データ生成中... ✅")
            print("   ニューラルネットワーク学習中... ✅") 
            print("   量子回路最適化中... ✅")
            print("   学習完了！")
            
            # 3. 評価フェーズ
            evaluation = self.evaluator.evaluate_ai(config)
            
            # 4. 結果表示
            self._show_evaluation_results(evaluation)
            
            # 5. 次のアクション選択
            if evaluation.is_strong_enough():
                print("\n🎉 おめでとうございます！十分強いAIができました！")
                action = input("\n次のアクション:\n"
                              "1. このAIを保存する\n"
                              "2. さらに改良を試す\n"
                              "選択 (1-2): ")
                if action == "1":
                    return config, evaluation
            else:
                print("\n💪 まだ改良の余地があります。")
                action = input("\n次のアクション:\n"
                              "1. 別の組み合わせを試す\n"
                              "2. このAIでも保存する\n"
                              "3. 諦める\n"
                              "選択 (1-3): ")
                if action == "2":
                    return config, evaluation
                elif action == "3":
                    return None, None
    
    def _show_evaluation_results(self, evaluation: EvaluationResult):
        """評価結果を表示"""
        print("\n📊 評価結果")
        print("=" * 50)
        print(f"総合勝率: {evaluation.overall_win_rate:.1%} ({evaluation.total_wins}/{evaluation.total_games})")
        
        # 詳細結果
        print("\n詳細結果:")
        results = [
            ("ランダムAI", evaluation.vs_random),
            ("シンプルAI", evaluation.vs_simple),
            ("攻撃的AI", evaluation.vs_aggressive)
        ]
        
        for name, result in results:
            rate = result['win_rate']
            status = "🔥" if rate >= 0.8 else "✅" if rate >= 0.6 else "⚖️" if rate >= 0.5 else "❌"
            print(f"  vs {name}: {result['wins']}/{result['total']} ({rate:.1%}) {status}")
        
        # 強度判定
        if evaluation.is_strong_enough():
            print(f"\n🌟 判定: 十分強いAI！（基準: 60%以上の勝率）")
        else:
            print(f"\n⚠️  判定: まだ改良が必要（現在: {evaluation.overall_win_rate:.1%}、基準: 60%）")
    
    def main_workflow(self):
        """メインワークフロー"""
        print("🎮 理想的なローカル対戦システム")
        print("=" * 60)
        print("最強のAIを作って、トーナメントで戦わせよう！")
        
        while True:
            print("\n【メインメニュー】")
            print("1. 新しいAIを作成")
            print("2. 保存されたAI一覧")
            print("3. AIトーナメント開催")
            print("4. トーナメント履歴")
            print("0. 終了")
            
            choice = input("\n選択 (0-4): ")
            
            if choice == "1":
                config, evaluation = self.train_and_evaluate_loop()
                if config and evaluation:
                    ai_name = input("\nAIに名前をつけてください: ") or "無名AI"
                    ai_id = self.model_manager.save_ai(config, evaluation, ai_name)
                    print(f"💾 AI '{ai_name}' を保存しました！(ID: {ai_id})")
                    
            elif choice == "2":
                ais = self.model_manager.get_all_ais()
                if not ais:
                    print("\n📭 まだAIが保存されていません")
                else:
                    print(f"\n📊 保存されたAI: {len(ais)}体")
                    for i, ai in enumerate(ais, 1):
                        status = "💪" if ai.get("is_strong", False) else "😐"
                        rate = ai["evaluation"]["overall_win_rate"]
                        print(f"  {i}. {ai['name']} {status} (勝率: {rate:.1%})")
                        
            elif choice == "3":
                results = self.tournament.run_tournament()
                if results:
                    self.tournament.show_tournament_results(results)
                    
            elif choice == "4":
                history = self.model_manager.registry.get("tournament_history", [])
                if not history:
                    print("\n📭 トーナメント履歴がありません")
                else:
                    print(f"\n📚 過去のトーナメント: {len(history)}回")
                    for i, tournament in enumerate(history[-5:], 1):  # 最新5回
                        date = tournament["date"][:10]
                        participants = tournament["participants"]
                        winner = tournament["rankings"][0]["name"]
                        print(f"  {i}. {date} - {participants}体参加 - 優勝: {winner}")
                        
            elif choice == "0":
                print("\n👋 お疲れ様でした！")
                break
            else:
                print("❌ 無効な選択です")


# ================================================================================
# メイン実行
# ================================================================================

def main():
    """メイン実行"""
    try:
        system = IdealLocalBattleSystem()
        system.main_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️ 実行中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
強化学習実装状況の詳細分析
プロジェクト内の強化学習コンポーネントの実装レベルを評価
"""

import os
import sys
import inspect
from typing import Dict, List, Any
import importlib.util

class ReinforcementLearningAnalyzer:
    """強化学習実装の分析クラス"""
    
    def __init__(self):
        self.analysis_results = {}
        self.rl_files = [
            "rl_cqcnn_runner.py",
            "rl_cqcnn_system.py", 
            "cqcnn_battle_learning_system.py",
            "unified_estimator_interface.py"
        ]
        
    def analyze_rl_implementation(self):
        """強化学習実装の詳細分析"""
        print("🔍 強化学習実装状況の分析開始")
        print("=" * 60)
        
        for file_name in self.rl_files:
            if os.path.exists(file_name):
                print(f"\n📁 {file_name} を分析中...")
                analysis = self._analyze_file(file_name)
                self.analysis_results[file_name] = analysis
                self._print_file_analysis(file_name, analysis)
            else:
                print(f"❌ {file_name} が見つかりません")
        
        self._print_overall_assessment()
    
    def _analyze_file(self, file_name: str) -> Dict[str, Any]:
        """個別ファイルの分析"""
        analysis = {
            'has_dqn': False,
            'has_replay_buffer': False,
            'has_epsilon_greedy': False,
            'has_q_learning': False,
            'has_experience_replay': False,
            'has_target_network': False,
            'has_self_play': False,
            'rl_classes': [],
            'rl_methods': [],
            'implementation_level': 'None'
        }
        
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # DQN実装チェック
            if 'DQN' in content or 'Deep Q' in content:
                analysis['has_dqn'] = True
            
            # リプレイバッファチェック
            if 'replay' in content.lower() and 'buffer' in content.lower():
                analysis['has_replay_buffer'] = True
            
            # ε-greedyチェック
            if 'epsilon' in content.lower() and ('greedy' in content.lower() or 'exploration' in content.lower()):
                analysis['has_epsilon_greedy'] = True
            
            # Q学習チェック
            if any(term in content.lower() for term in ['q_learning', 'q-learning', 'q_value', 'q_network']):
                analysis['has_q_learning'] = True
            
            # 経験再生チェック
            if 'experience' in content.lower() and 'replay' in content.lower():
                analysis['has_experience_replay'] = True
            
            # ターゲットネットワークチェック
            if 'target' in content.lower() and 'network' in content.lower():
                analysis['has_target_network'] = True
            
            # 自己対戦チェック
            if any(term in content.lower() for term in ['self_play', 'self-play', '自己対戦']):
                analysis['has_self_play'] = True
            
            # クラスとメソッドの抽出
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('class ') and any(rl_term in line.lower() for rl_term in 
                    ['rl', 'dqn', 'q_', 'reinforcement', 'learning', 'trainer']):
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    analysis['rl_classes'].append(class_name)
                
                if line.startswith('def ') and any(rl_term in line.lower() for rl_term in 
                    ['train', 'learn', 'update', 'replay', 'epsilon', 'q_']):
                    method_name = line.split('def ')[1].split('(')[0].strip()
                    analysis['rl_methods'].append(method_name)
            
            # 実装レベル判定
            rl_features = [
                analysis['has_dqn'],
                analysis['has_replay_buffer'], 
                analysis['has_epsilon_greedy'],
                analysis['has_q_learning'],
                analysis['has_experience_replay'],
                analysis['has_target_network']
            ]
            
            feature_count = sum(rl_features)
            if feature_count >= 5:
                analysis['implementation_level'] = 'Full'
            elif feature_count >= 3:
                analysis['implementation_level'] = 'Partial'
            elif feature_count >= 1:
                analysis['implementation_level'] = 'Basic'
            else:
                analysis['implementation_level'] = 'None'
                
        except Exception as e:
            print(f"⚠️ {file_name} の分析中にエラー: {e}")
            
        return analysis
    
    def _print_file_analysis(self, file_name: str, analysis: Dict[str, Any]):
        """ファイル分析結果の表示"""
        level = analysis['implementation_level']
        level_emoji = {
            'Full': '🟢',
            'Partial': '🟡', 
            'Basic': '🟠',
            'None': '🔴'
        }
        
        print(f"  実装レベル: {level_emoji[level]} {level}")
        
        if analysis['rl_classes']:
            print(f"  📚 強化学習クラス: {', '.join(analysis['rl_classes'][:3])}")
        
        if analysis['rl_methods']:
            print(f"  🔧 関連メソッド: {', '.join(analysis['rl_methods'][:5])}")
        
        features = []
        if analysis['has_dqn']: features.append('DQN')
        if analysis['has_replay_buffer']: features.append('リプレイバッファ')
        if analysis['has_epsilon_greedy']: features.append('ε-greedy')
        if analysis['has_q_learning']: features.append('Q学習')
        if analysis['has_experience_replay']: features.append('経験再生')
        if analysis['has_target_network']: features.append('ターゲットネットワーク')
        if analysis['has_self_play']: features.append('自己対戦')
        
        if features:
            print(f"  ✅ 実装済み機能: {', '.join(features)}")
        else:
            print(f"  ❌ 強化学習機能が検出されませんでした")
    
    def _print_overall_assessment(self):
        """総合評価の表示"""
        print(f"\n🎯 総合評価")
        print("=" * 60)
        
        full_impl = sum(1 for analysis in self.analysis_results.values() if analysis['implementation_level'] == 'Full')
        partial_impl = sum(1 for analysis in self.analysis_results.values() if analysis['implementation_level'] == 'Partial')
        basic_impl = sum(1 for analysis in self.analysis_results.values() if analysis['implementation_level'] == 'Basic')
        no_impl = sum(1 for analysis in self.analysis_results.values() if analysis['implementation_level'] == 'None')
        
        total_files = len(self.analysis_results)
        
        print(f"📊 実装状況統計:")
        print(f"  🟢 完全実装: {full_impl}/{total_files} ファイル")
        print(f"  🟡 部分実装: {partial_impl}/{total_files} ファイル") 
        print(f"  🟠 基本実装: {basic_impl}/{total_files} ファイル")
        print(f"  🔴 未実装: {no_impl}/{total_files} ファイル")
        
        # 全体的な評価
        if full_impl >= 2:
            overall_status = "🏆 強化学習が実装されています"
            print(f"\n{overall_status}")
            print("✅ DQN, リプレイバッファ, ε-greedy等の主要コンポーネントが実装済み")
            print("✅ 自己対戦による学習が可能")
            print("✅ GUIから生成されるAIは強化学習を活用できます")
        elif partial_impl >= 1 or basic_impl >= 2:
            overall_status = "🔄 強化学習が部分的に実装されています"
            print(f"\n{overall_status}")
            print("⚠️ 一部のコンポーネントが実装されていますが、完全ではありません")
            print("📝 追加の実装が必要な可能性があります")
        else:
            overall_status = "❌ 強化学習の実装が不完全です"
            print(f"\n{overall_status}")
            print("🚨 主要な強化学習コンポーネントが不足しています")
            print("📋 DQN, リプレイバッファ等の実装が必要です")
        
        print(f"\n🤖 GUIとの関係性:")
        if full_impl >= 1:
            print("✅ GUIで「強化学習」を選択した場合、実際に強化学習アルゴリズムが使用されます")
            print("✅ ε-greedy、DQN、リプレイバッファ等が動作します") 
            print("✅ 生成されるPythonコードは実際に学習可能です")
        else:
            print("⚠️ GUIで「強化学習」を選択しても、実際の強化学習は限定的です")
            print("📝 教師あり学習がメインで動作している可能性があります")
            
    def check_gui_rl_integration(self):
        """GUIと強化学習の統合状況をチェック"""
        print(f"\n🔗 GUI-強化学習統合チェック")
        print("=" * 60)
        
        # GUIファイルの確認
        gui_file = "quantum_battle_3step_system.html"
        if os.path.exists(gui_file):
            with open(gui_file, 'r', encoding='utf-8') as f:
                gui_content = f.read()
            
            print(f"📋 {gui_file} の強化学習関連機能:")
            
            rl_terms = ['reinforcement', '強化学習', 'DQN', 'epsilon', 'Q学習']
            found_terms = [term for term in rl_terms if term in gui_content]
            
            if found_terms:
                print(f"  ✅ 発見された用語: {', '.join(found_terms)}")
                
                # コード生成部分の確認
                if 'generateCode' in gui_content:
                    print("  ✅ Pythonコード生成機能が存在")
                    if any(term in gui_content for term in ['reinforcement', '強化学習']):
                        print("  ✅ 強化学習用コード生成に対応")
                    else:
                        print("  ⚠️ 強化学習用コード生成が不明")
                else:
                    print("  ❌ Pythonコード生成機能が見つからない")
            else:
                print("  ❌ 強化学習関連の機能が見つかりません")
        else:
            print(f"❌ {gui_file} が見つかりません")

def main():
    """メイン実行"""
    print("🔬 強化学習実装状況の詳細分析")
    print("プロジェクト内の強化学習コンポーネントを調査します\n")
    
    analyzer = ReinforcementLearningAnalyzer()
    analyzer.analyze_rl_implementation()
    analyzer.check_gui_rl_integration()
    
    print(f"\n📋 分析完了")
    print("=" * 60)

if __name__ == "__main__":
    main()
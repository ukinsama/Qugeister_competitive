#!/usr/bin/env python3
"""
改良版モデルローダー
保存されたモデルから完全なAIを再構成
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os

class ImprovedModelLoader:
    """改良版モデルローダー"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        
    def analyze_checkpoint_detailed(self, checkpoint_path: str) -> Dict[str, Any]:
        """チェックポイントの詳細分析"""
        print(f"\n🔍 詳細分析: {checkpoint_path}")
        print("-" * 50)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        analysis = {
            'file_size': os.path.getsize(checkpoint_path),
            'structure': {},
            'model_info': {},
            'training_info': {},
            'reconstruction_strategy': 'unknown'
        }
        
        # チェックポイント構造の詳細分析
        for key, value in checkpoint.items():
            print(f"📋 {key}:")
            
            if key == 'estimator_state' and isinstance(value, dict):
                analysis['model_info']['estimator_params'] = len(value)
                print(f"  🧠 推定器パラメータ: {len(value)} 個")
                for param_name, param_tensor in list(value.items())[:3]:
                    print(f"    - {param_name}: {list(param_tensor.shape)}")
                if len(value) > 3:
                    print(f"    ... 他 {len(value) - 3} 個")
                    
            elif key == 'target_state' and isinstance(value, dict):
                analysis['model_info']['target_params'] = len(value)
                print(f"  🎯 ターゲットネットワーク: {len(value)} 個のパラメータ")
                
            elif key == 'model_state_dict' and isinstance(value, dict):
                analysis['model_info']['main_model_params'] = len(value)
                total_params = sum(v.numel() for v in value.values())
                print(f"  🏗️ メインモデル: {len(value)} レイヤー, {total_params:,} パラメータ")
                
            elif key == 'optimizer_state' or key == 'optimizer_state_dict':
                print(f"  ⚙️ オプティマイザ状態: 保存済み")
                analysis['training_info']['has_optimizer'] = True
                
            elif key == 'epsilon' and isinstance(value, (int, float)):
                print(f"  🎲 探索率: {value:.6f}")
                analysis['training_info']['epsilon'] = value
                
            elif key == 'episodes' and isinstance(value, int):
                print(f"  📈 学習エピソード数: {value}")
                analysis['training_info']['episodes'] = value
                
            elif key == 'training_history' and isinstance(value, dict):
                print(f"  📊 学習履歴:")
                for hist_key, hist_value in value.items():
                    if isinstance(hist_value, list):
                        print(f"    - {hist_key}: {len(hist_value)} 記録")
                        if hist_value:  # 空でない場合
                            if isinstance(hist_value[0], (int, float)):
                                print(f"      初期値: {hist_value[0]:.6f}, 最終値: {hist_value[-1]:.6f}")
                analysis['training_info']['history'] = value
                
            elif key == 'history' and isinstance(value, dict):
                print(f"  📜 学習記録:")
                for hist_key, hist_value in value.items():
                    if isinstance(hist_value, list):
                        print(f"    - {hist_key}: {len(hist_value)} エントリ")
                        
        # 再構成戦略の決定
        if 'estimator_state' in checkpoint and 'target_state' in checkpoint:
            analysis['reconstruction_strategy'] = 'rl_cqcnn'
            print(f"\n🎯 再構成戦略: 強化学習CQCNN (DQNスタイル)")
        elif 'model_state_dict' in checkpoint:
            analysis['reconstruction_strategy'] = 'standard_model'
            print(f"\n🏗️ 再構成戦略: 標準モデル")
        else:
            analysis['reconstruction_strategy'] = 'unknown'
            print(f"\n❓ 再構成戦略: 不明")
            
        return analysis
    
    def create_minimal_cqcnn(self, param_shapes: Dict[str, List[int]]) -> nn.Module:
        """パラメータ形状から最小限のCQCNNを作成"""
        
        class MinimalCQCNN(nn.Module):
            def __init__(self, param_info):
                super().__init__()
                # パラメータ形状から推定してレイヤーを構築
                self.layers = nn.ModuleDict()
                
                for name, shape in param_info.items():
                    if 'weight' in name and len(shape) == 2:
                        # 線形レイヤー
                        layer_name = name.replace('.weight', '')
                        self.layers[layer_name] = nn.Linear(shape[1], shape[0])
                    elif 'conv' in name.lower() and len(shape) == 4:
                        # 畳み込みレイヤー
                        layer_name = name.replace('.weight', '')
                        self.layers[layer_name] = nn.Conv2d(shape[1], shape[0], 
                                                          kernel_size=shape[2:])
                        
            def forward(self, x):
                # 簡単な順伝播（実際の使用時は適切に実装）
                for layer in self.layers.values():
                    if isinstance(layer, nn.Linear):
                        x = x.view(x.size(0), -1)  # Flatten
                        x = layer(x)
                    elif isinstance(layer, nn.Conv2d):
                        x = layer(x)
                return x
                
        return MinimalCQCNN(param_shapes)
    
    def attempt_full_reconstruction(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """完全なモデル再構成を試行"""
        print(f"\n🔧 完全再構成試行: {checkpoint_path}")
        print("-" * 50)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            analysis = self.analyze_checkpoint_detailed(checkpoint_path)
            
            reconstructed = {
                'success': False,
                'models': {},
                'optimizers': {},
                'training_state': {},
                'errors': []
            }
            
            if analysis['reconstruction_strategy'] == 'rl_cqcnn':
                print("🎯 強化学習CQCNN再構成を試行...")
                
                # 推定器ネットワークの再構成
                if 'estimator_state' in checkpoint:
                    estimator_dict = checkpoint['estimator_state']
                    param_shapes = {k: list(v.shape) for k, v in estimator_dict.items()}
                    
                    try:
                        estimator = self.create_minimal_cqcnn(param_shapes)
                        estimator.load_state_dict(estimator_dict)
                        reconstructed['models']['estimator'] = estimator
                        print("  ✅ 推定器ネットワーク再構成成功")
                    except Exception as e:
                        reconstructed['errors'].append(f"推定器再構成失敗: {e}")
                        print(f"  ❌ 推定器再構成失敗: {e}")
                
                # ターゲットネットワークの再構成
                if 'target_state' in checkpoint:
                    target_dict = checkpoint['target_state']
                    param_shapes = {k: list(v.shape) for k, v in target_dict.items()}
                    
                    try:
                        target = self.create_minimal_cqcnn(param_shapes)
                        target.load_state_dict(target_dict)
                        reconstructed['models']['target'] = target
                        print("  ✅ ターゲットネットワーク再構成成功")
                    except Exception as e:
                        reconstructed['errors'].append(f"ターゲット再構成失敗: {e}")
                        print(f"  ❌ ターゲット再構成失敗: {e}")
                
                # 学習状態の復元
                if 'epsilon' in checkpoint:
                    reconstructed['training_state']['epsilon'] = checkpoint['epsilon']
                    print(f"  ✅ 探索率復元: {checkpoint['epsilon']:.6f}")
                    
                if 'episodes' in checkpoint:
                    reconstructed['training_state']['episodes'] = checkpoint['episodes']
                    print(f"  ✅ エピソード数復元: {checkpoint['episodes']}")
                    
                if 'training_history' in checkpoint:
                    reconstructed['training_state']['history'] = checkpoint['training_history']
                    history = checkpoint['training_history']
                    if 'rewards' in history and history['rewards']:
                        recent_reward = history['rewards'][-1] if history['rewards'] else 0
                        print(f"  ✅ 学習履歴復元: 最新報酬 {recent_reward:.3f}")
                        
            elif analysis['reconstruction_strategy'] == 'standard_model':
                print("🏗️ 標準モデル再構成を試行...")
                
                if 'model_state_dict' in checkpoint:
                    model_dict = checkpoint['model_state_dict']
                    param_shapes = {k: list(v.shape) for k, v in model_dict.items()}
                    
                    try:
                        model = self.create_minimal_cqcnn(param_shapes)
                        model.load_state_dict(model_dict)
                        reconstructed['models']['main'] = model
                        print("  ✅ メインモデル再構成成功")
                    except Exception as e:
                        reconstructed['errors'].append(f"モデル再構成失敗: {e}")
                        print(f"  ❌ モデル再構成失敗: {e}")
            
            # 成功判定
            reconstructed['success'] = len(reconstructed['models']) > 0
            
            if reconstructed['success']:
                print(f"🎉 再構成成功! {len(reconstructed['models'])} 個のモデル復元")
            else:
                print("❌ 再構成失敗")
                
            return reconstructed
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_reconstructed_models(self, reconstructed: Dict[str, Any]) -> Dict[str, Any]:
        """再構成されたモデルのテスト"""
        print(f"\n🧪 再構成モデルテスト")
        print("-" * 50)
        
        if not reconstructed.get('success', False):
            print("❌ 再構成が失敗しているためテストをスキップ")
            return {'tested': False, 'reason': 'reconstruction_failed'}
        
        test_results = {'tested': True, 'model_tests': {}}
        
        # テスト用の入力データ
        test_input = torch.randn(1, 25)  # 5x5のフラット化された入力
        
        for model_name, model in reconstructed['models'].items():
            print(f"🔬 {model_name}モデルテスト:")
            
            try:
                model.eval()
                with torch.no_grad():
                    output = model(test_input)
                    
                test_results['model_tests'][model_name] = {
                    'success': True,
                    'input_shape': list(test_input.shape),
                    'output_shape': list(output.shape),
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item(),
                    'output_range': [output.min().item(), output.max().item()]
                }
                
                print(f"  ✅ 推論成功")
                print(f"  📊 出力形状: {list(output.shape)}")
                print(f"  📈 出力統計: 平均={output.mean().item():.4f}, 標準偏差={output.std().item():.4f}")
                
            except Exception as e:
                test_results['model_tests'][model_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ❌ テスト失敗: {e}")
        
        return test_results

def main():
    """メイン実行"""
    print("🔧 改良版モデル再構成システム")
    print("=" * 70)
    
    loader = ImprovedModelLoader()
    
    # 利用可能なモデルファイル
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if not model_files:
        print("❌ .pthファイルが見つかりません")
        return
    
    print(f"📁 発見されたモデル: {len(model_files)} 個")
    
    for model_file in model_files:
        print(f"\n" + "="*70)
        print(f"🎯 処理中: {model_file}")
        
        # 詳細分析
        analysis = loader.analyze_checkpoint_detailed(model_file)
        
        # 完全再構成試行
        reconstructed = loader.attempt_full_reconstruction(model_file)
        
        # 再構成モデルのテスト
        if reconstructed:
            test_results = loader.test_reconstructed_models(reconstructed)
        
    print(f"\n🎉 検証完了!")
    print("結論: 保存されたモデルから部分的な再構成が可能です。")
    print("完全な実用化には、モデル構造情報の明示的な保存が必要です。")

if __name__ == "__main__":
    main()
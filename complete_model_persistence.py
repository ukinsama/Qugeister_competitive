#!/usr/bin/env python3
"""
完全なモデル保存・読み込みシステム
学習済みAIの完全な永続化と復元
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import pickle
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class ModelMetadata:
    """モデルメタデータ"""
    model_name: str
    model_type: str
    creation_date: str
    training_episodes: int
    final_reward: float
    architecture_hash: str
    version: str = "1.0.0"
    description: str = ""

class UniversalModelSaver:
    """汎用モデル保存・復元システム"""
    
    def __init__(self, save_directory: str = "./saved_models"):
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)
        
    def calculate_architecture_hash(self, model: nn.Module) -> str:
        """モデル構造のハッシュ値計算"""
        model_str = str(model)
        return hashlib.md5(model_str.encode()).hexdigest()[:16]
    
    def save_complete_model(self, 
                          model: nn.Module,
                          model_name: str,
                          optimizer: optim.Optimizer = None,
                          training_history: Dict = None,
                          metadata: Dict = None,
                          additional_data: Dict = None) -> str:
        """完全なモデル保存"""
        
        print(f"💾 完全モデル保存開始: {model_name}")
        print("-" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_directory, f"{model_name}_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        
        # 1. モデル本体保存
        model_file = os.path.join(save_path, "model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_module': model.__class__.__module__,
        }, model_file)
        print(f"  ✅ モデル本体: {model_file}")
        
        # 2. オプティマイザ保存
        if optimizer:
            optimizer_file = os.path.join(save_path, "optimizer.pth")
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_class': optimizer.__class__.__name__,
                'optimizer_module': optimizer.__class__.__module__,
            }, optimizer_file)
            print(f"  ✅ オプティマイザ: {optimizer_file}")
        
        # 3. モデル構造保存
        architecture_file = os.path.join(save_path, "architecture.txt")
        with open(architecture_file, 'w') as f:
            f.write(str(model))
        print(f"  ✅ モデル構造: {architecture_file}")
        
        # 4. 設定パラメータ保存
        if hasattr(model, 'get_config'):
            config_file = os.path.join(save_path, "config.json")
            with open(config_file, 'w') as f:
                json.dump(model.get_config(), f, indent=2)
            print(f"  ✅ 設定パラメータ: {config_file}")
        
        # 5. 学習履歴保存
        if training_history:
            history_file = os.path.join(save_path, "training_history.json")
            # numpy配列をリストに変換
            history_clean = self._clean_for_json(training_history)
            with open(history_file, 'w') as f:
                json.dump(history_clean, f, indent=2)
            print(f"  ✅ 学習履歴: {history_file}")
        
        # 6. メタデータ作成・保存
        model_metadata = ModelMetadata(
            model_name=model_name,
            model_type=model.__class__.__name__,
            creation_date=datetime.now().isoformat(),
            training_episodes=training_history.get('episodes', 0) if training_history else 0,
            final_reward=training_history.get('final_reward', 0.0) if training_history else 0.0,
            architecture_hash=self.calculate_architecture_hash(model),
            description=metadata.get('description', '') if metadata else ''
        )
        
        metadata_file = os.path.join(save_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(asdict(model_metadata), f, indent=2)
        print(f"  ✅ メタデータ: {metadata_file}")
        
        # 7. 追加データ保存
        if additional_data:
            additional_file = os.path.join(save_path, "additional_data.pkl")
            with open(additional_file, 'wb') as f:
                pickle.dump(additional_data, f)
            print(f"  ✅ 追加データ: {additional_file}")
        
        # 8. インデックスファイル作成
        self._update_model_index(save_path, model_metadata)
        
        print(f"🎉 保存完了: {save_path}")
        return save_path
    
    def _clean_for_json(self, obj):
        """JSON保存用にデータをクリーンアップ"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _update_model_index(self, save_path: str, metadata: ModelMetadata):
        """モデルインデックス更新"""
        index_file = os.path.join(self.save_directory, "model_index.json")
        
        # 既存インデックス読み込み
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {"models": []}
        
        # 新しいモデル追加
        model_entry = asdict(metadata)
        model_entry["save_path"] = save_path
        index["models"].append(model_entry)
        
        # インデックス保存
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def list_saved_models(self) -> List[Dict]:
        """保存済みモデル一覧"""
        index_file = os.path.join(self.save_directory, "model_index.json")
        
        if not os.path.exists(index_file):
            return []
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        return index.get("models", [])
    
    def load_complete_model(self, model_path: str) -> Dict[str, Any]:
        """完全なモデル読み込み"""
        print(f"📂 完全モデル読み込み: {model_path}")
        print("-" * 50)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_path}")
        
        loaded_data = {}
        
        # 1. メタデータ読み込み
        metadata_file = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                loaded_data['metadata'] = json.load(f)
            print(f"  ✅ メタデータ読み込み完了")
        
        # 2. モデル本体読み込み
        model_file = os.path.join(model_path, "model.pth")
        if os.path.exists(model_file):
            loaded_data['model_checkpoint'] = torch.load(model_file, map_location='cpu')
            print(f"  ✅ モデル本体読み込み完了")
        
        # 3. オプティマイザ読み込み
        optimizer_file = os.path.join(model_path, "optimizer.pth")
        if os.path.exists(optimizer_file):
            loaded_data['optimizer_checkpoint'] = torch.load(optimizer_file, map_location='cpu')
            print(f"  ✅ オプティマイザ読み込み完了")
        
        # 4. 設定パラメータ読み込み
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded_data['config'] = json.load(f)
            print(f"  ✅ 設定パラメータ読み込み完了")
        
        # 5. 学習履歴読み込み
        history_file = os.path.join(model_path, "training_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                loaded_data['training_history'] = json.load(f)
            print(f"  ✅ 学習履歴読み込み完了")
        
        # 6. モデル構造読み込み
        architecture_file = os.path.join(model_path, "architecture.txt")
        if os.path.exists(architecture_file):
            with open(architecture_file, 'r') as f:
                loaded_data['architecture'] = f.read()
            print(f"  ✅ モデル構造読み込み完了")
        
        # 7. 追加データ読み込み
        additional_file = os.path.join(model_path, "additional_data.pkl")
        if os.path.exists(additional_file):
            with open(additional_file, 'rb') as f:
                loaded_data['additional_data'] = pickle.load(f)
            print(f"  ✅ 追加データ読み込み完了")
        
        print(f"🎉 読み込み完了: {len(loaded_data)} 個の要素")
        return loaded_data
    
    def create_model_from_save(self, model_path: str, device: str = 'cpu') -> Tuple[nn.Module, Optional[optim.Optimizer]]:
        """保存データからモデルとオプティマイザを復元"""
        print(f"🔧 モデル復元: {model_path}")
        print("-" * 50)
        
        loaded_data = self.load_complete_model(model_path)
        
        # モデルクラスの動的インポート
        if 'model_checkpoint' not in loaded_data:
            raise ValueError("モデルチェックポイントが見つかりません")
        
        model_checkpoint = loaded_data['model_checkpoint']
        
        # 設定パラメータからモデルを再構築
        if 'config' in loaded_data:
            config = loaded_data['config']
            # ここで実際のモデルクラスに応じて動的に復元
            print(f"  🔧 設定からモデル復元: {config}")
        
        # オプティマイザ復元
        optimizer = None
        if 'optimizer_checkpoint' in loaded_data:
            optimizer_checkpoint = loaded_data['optimizer_checkpoint']
            print(f"  ⚙️ オプティマイザ復元: {optimizer_checkpoint['optimizer_class']}")
        
        return None, None  # 実装は具体的なモデルクラスに応じて調整
    
    def export_model_summary(self, model_path: str) -> str:
        """モデルサマリーをエクスポート"""
        loaded_data = self.load_complete_model(model_path)
        
        summary = []
        summary.append("🤖 モデルサマリー")
        summary.append("=" * 50)
        
        if 'metadata' in loaded_data:
            metadata = loaded_data['metadata']
            summary.append(f"モデル名: {metadata['model_name']}")
            summary.append(f"モデル種別: {metadata['model_type']}")
            summary.append(f"作成日時: {metadata['creation_date']}")
            summary.append(f"学習エピソード: {metadata['training_episodes']}")
            summary.append(f"最終報酬: {metadata['final_reward']}")
            summary.append(f"説明: {metadata['description']}")
        
        if 'training_history' in loaded_data:
            history = loaded_data['training_history']
            summary.append(f"\n📈 学習履歴:")
            for key, values in history.items():
                if isinstance(values, list) and values:
                    summary.append(f"  {key}: {len(values)} 記録")
                    if all(isinstance(v, (int, float)) for v in values):
                        summary.append(f"    初期: {values[0]:.4f}, 最終: {values[-1]:.4f}")
        
        if 'architecture' in loaded_data:
            summary.append(f"\n🏗️ モデル構造:")
            arch_lines = loaded_data['architecture'].split('\n')[:10]
            for line in arch_lines:
                summary.append(f"  {line}")
            if len(loaded_data['architecture'].split('\n')) > 10:
                summary.append("  ...")
        
        return '\n'.join(summary)

class EnhancedCQCNNSaver(UniversalModelSaver):
    """CQCNN専用の拡張保存システム"""
    
    def save_cqcnn_model(self, 
                        estimator: nn.Module,
                        target: nn.Module,
                        optimizer: optim.Optimizer,
                        epsilon: float,
                        episodes: int,
                        training_history: Dict,
                        model_name: str) -> str:
        """CQCNN強化学習モデルの専用保存"""
        
        print(f"🧠 CQCNN強化学習モデル保存: {model_name}")
        print("-" * 50)
        
        # 追加データ（CQCNN特有）
        additional_data = {
            'estimator_state_dict': estimator.state_dict(),
            'target_state_dict': target.state_dict(),
            'epsilon': epsilon,
            'episodes': episodes,
            'model_type': 'CQCNN_RL',
            'quantum_params_shape': None,
            'conv_layers': None
        }
        
        # 量子パラメータ情報抽出
        if hasattr(estimator, 'quantum_params'):
            additional_data['quantum_params_shape'] = list(estimator.quantum_params.shape)
            print(f"  🌌 量子パラメータ形状: {additional_data['quantum_params_shape']}")
        
        # 畳み込み層情報抽出
        conv_info = []
        for name, module in estimator.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_info.append({
                    'name': name,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'padding': module.padding
                })
        additional_data['conv_layers'] = conv_info
        print(f"  🔍 畳み込み層: {len(conv_info)} 層")
        
        # メタデータ
        metadata = {
            'description': f'CQCNN強化学習モデル (ε={epsilon:.4f}, {episodes}エピソード)',
            'epsilon': epsilon,
            'episodes': episodes,
            'has_target_network': True,
            'quantum_enabled': hasattr(estimator, 'quantum_params')
        }
        
        # 学習履歴の拡張
        enhanced_history = training_history.copy()
        enhanced_history.update({
            'final_epsilon': epsilon,
            'total_episodes': episodes,
            'model_architecture': 'CQCNN_DQN'
        })
        
        # 保存実行
        save_path = self.save_complete_model(
            model=estimator,
            model_name=model_name,
            optimizer=optimizer,
            training_history=enhanced_history,
            metadata=metadata,
            additional_data=additional_data
        )
        
        print(f"🎉 CQCNN保存完了: {save_path}")
        return save_path
    
    def load_cqcnn_model(self, model_path: str) -> Tuple[Dict, str]:
        """CQCNN専用読み込み"""
        print(f"🧠 CQCNN専用読み込み: {model_path}")
        
        loaded_data = self.load_complete_model(model_path)
        
        if 'additional_data' not in loaded_data:
            raise ValueError("CQCNNデータが見つかりません")
        
        cqcnn_data = loaded_data['additional_data']
        
        # CQCNN特有データの検証
        required_keys = ['estimator_state_dict', 'target_state_dict', 'epsilon', 'episodes']
        missing_keys = [key for key in required_keys if key not in cqcnn_data]
        
        if missing_keys:
            raise ValueError(f"必要なCQCNNデータが不足: {missing_keys}")
        
        print(f"  ✅ CQCNN復元準備完了")
        print(f"    探索率: {cqcnn_data['epsilon']:.6f}")
        print(f"    エピソード: {cqcnn_data['episodes']}")
        
        if 'quantum_params_shape' in cqcnn_data:
            print(f"    量子パラメータ: {cqcnn_data['quantum_params_shape']}")
        
        if 'conv_layers' in cqcnn_data:
            print(f"    畳み込み層: {len(cqcnn_data['conv_layers'])} 層")
        
        return loaded_data, model_path

def demo_model_persistence():
    """モデル永続化のデモンストレーション"""
    print("🚀 完全モデル永続化システム デモ")
    print("=" * 70)
    
    saver = UniversalModelSaver()
    
    # 保存済みモデル一覧
    models = saver.list_saved_models()
    print(f"📂 保存済みモデル: {len(models)} 個")
    
    if models:
        print("保存済みモデル一覧:")
        for i, model in enumerate(models[-5:]):  # 最新5個
            print(f"  {i+1}. {model['model_name']} ({model['creation_date']})")
    
    # 強化学習モデルファイル検索
    rl_files = [f for f in os.listdir('.') if f.startswith('rl_') and f.endswith('.pth')]
    
    if rl_files:
        print(f"\n🎯 既存の強化学習ファイル: {len(rl_files)} 個")
        
        # CQCNN専用保存システム
        cqcnn_saver = EnhancedCQCNNSaver()
        
        print(f"\n🧠 CQCNN専用保存システム準備完了")
        print("次回の強化学習実行時に自動で完全保存されます")
        
        # 既存ファイルの分析
        for rl_file in rl_files:
            print(f"\n📊 分析: {rl_file}")
            try:
                checkpoint = torch.load(rl_file, map_location='cpu')
                
                analysis = {
                    'file_size': f"{os.path.getsize(rl_file) / 1024 / 1024:.2f} MB",
                    'keys': list(checkpoint.keys()),
                    'episodes': checkpoint.get('episodes', 'N/A'),
                    'epsilon': checkpoint.get('epsilon', 'N/A')
                }
                
                print(f"  サイズ: {analysis['file_size']}")
                print(f"  エピソード: {analysis['episodes']}")
                print(f"  探索率: {analysis['epsilon']}")
                print(f"  データ要素: {len(analysis['keys'])} 個")
                
                # 完全保存形式に変換可能かチェック
                convertible = 'estimator_state' in checkpoint and 'target_state' in checkpoint
                print(f"  完全保存変換: {'✅ 可能' if convertible else '❌ 要調整'}")
                
            except Exception as e:
                print(f"  ❌ 分析失敗: {e}")
    else:
        print("\n⚠️ 強化学習モデルファイルが見つかりません")
        print("強化学習を実行して .pth ファイルを生成してください:")
        print("  python rl_cqcnn_runner.py")
    
    print(f"\n🎉 モデル永続化システム準備完了！")
    print("特徴:")
    print("  ✅ 完全なモデル構造保存")
    print("  ✅ 学習履歴・メタデータ保存")
    print("  ✅ バージョン管理・インデックス化")
    print("  ✅ CQCNN専用最適化")
    print("  ✅ 復元時の検証機能")

if __name__ == "__main__":
    demo_model_persistence()
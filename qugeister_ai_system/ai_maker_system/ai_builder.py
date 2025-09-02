#!/usr/bin/env python3
"""
AIビルダーシステム - 全モジュールを統合してAIを生成
"""

import torch
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# コアモジュール
from .core.base_modules import CQCNNModel
from .core.game_state import GameConfig, LearningConfig
from .core.data_processor import DataProcessor

# 機能モジュール
from .modules.placement import PlacementFactory
from .modules.estimator import EstimatorFactory
from .modules.reward import RewardFactory
from .modules.qmap import QMapFactory
from .modules.action import ActionFactory

# 学習システム
from .learning.supervised import SupervisedLearning
from .learning.reinforcement import DQNReinforcementLearning


class AIBuilder:
    """AI制作システム - メインビルダークラス"""
    
    def __init__(self, output_dir: str = "generated_ais"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.built_ais = []
        
        print("🚀 AI Maker System 初期化完了")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
    
    def create_ai(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        設定に基づいてAIを作成
        
        Args:
            config: AI設定辞書
                - name: AI名
                - placement: 配置戦略設定
                - estimator: 推定器設定
                - reward: 報酬関数設定
                - qmap: Q値マップ設定
                - action: 行動選択設定
                - learning: 学習設定
        
        Returns:
            作成されたAI情報
        """
        print(f"🔨 AI作成開始: {config.get('name', 'UnknownAI')}")
        
        # 1. 基本設定
        ai_name = config.get('name', f"AI_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        game_config = GameConfig()
        learning_config = LearningConfig()
        
        # 学習設定を更新
        if 'learning' in config:
            learning_cfg = config['learning']
            learning_config.learning_rate = learning_cfg.get('learning_rate', 0.001)
            learning_config.batch_size = learning_cfg.get('batch_size', 32)
            learning_config.supervised_epochs = learning_cfg.get('epochs', 100)
            learning_config.rl_episodes = learning_cfg.get('episodes', 1000)
        
        # 2. モジュール作成
        modules = self._create_modules(config)
        
        # 3. メインモデル作成
        model_config = config.get('model', {})
        n_qubits = model_config.get('n_qubits', 6)
        n_layers = model_config.get('n_layers', 3)
        model = CQCNNModel(n_qubits, n_layers)
        
        print(f"🧠 CQCNNモデル作成: {n_qubits}量子ビット, {n_layers}層")
        print(f"📊 パラメータ数: {sum(p.numel() for p in model.parameters())}")
        
        # 4. 学習実行（オプション）
        trained_model = None
        training_results = None
        
        if config.get('auto_train', False):
            print("🎓 自動学習を開始...")
            trained_model, training_results = self._train_model(
                model, modules, learning_config, config
            )
        
        # 5. AI情報作成
        ai_info = {
            'name': ai_name,
            'creation_time': datetime.now().isoformat(),
            'config': config,
            'modules': {
                'placement': modules['placement'].get_name(),
                'estimator': modules['estimator'].get_name(),
                'reward': modules['reward'].get_name(),
                'qmap': modules['qmap'].get_name(),
                'action': modules['action'].get_name()
            },
            'model': {
                'class_name': model.__class__.__name__,
                'n_qubits': n_qubits,
                'n_layers': n_layers,
                'total_parameters': sum(p.numel() for p in model.parameters())
            },
            'training_results': training_results
        }
        
        # 6. 保存
        ai_path = self._save_ai(ai_info, model if trained_model is None else trained_model, modules)
        ai_info['path'] = str(ai_path)
        
        self.built_ais.append(ai_info)
        
        print(f"✅ AI作成完了: {ai_name}")
        print(f"💾 保存先: {ai_path}")
        
        return ai_info
    
    def _create_modules(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定からモジュールを作成"""
        modules = {}
        
        # 1. 配置戦略
        placement_cfg = config.get('placement', {'type': 'standard'})
        strategy_type = placement_cfg.pop('type', 'standard')
        modules['placement'] = PlacementFactory.create_placement(strategy_type, **placement_cfg)
        
        # 2. 推定器
        estimator_cfg = config.get('estimator', {'type': 'cqcnn'})
        if 'model' in config:
            estimator_cfg.update({
                'n_qubits': config['model'].get('n_qubits', 6),
                'n_layers': config['model'].get('n_layers', 3)
            })
        estimator_type = estimator_cfg.pop('type', 'cqcnn')
        modules['estimator'] = EstimatorFactory.create_estimator(estimator_type, **estimator_cfg)
        
        # 3. 報酬関数
        reward_cfg = config.get('reward', {'type': 'basic'})
        reward_type = reward_cfg.pop('type', 'basic')
        modules['reward'] = RewardFactory.create_reward(reward_type, **reward_cfg)
        
        # 4. Q値マップ生成器
        qmap_cfg = config.get('qmap', {'type': 'simple'})
        qmap_type = qmap_cfg.pop('type', 'simple')
        modules['qmap'] = QMapFactory.create_qmap_generator(qmap_type, **qmap_cfg)
        
        # 5. 行動選択器
        action_cfg = config.get('action', {'type': 'epsilon_greedy'})
        action_type = action_cfg.pop('type', 'epsilon_greedy')
        modules['action'] = ActionFactory.create_selector(action_type, **action_cfg)
        
        print(f"📦 モジュール作成完了:")
        for module_type, module in modules.items():
            print(f"  - {module_type}: {module.get_name()}")
        
        return modules
    
    def _train_model(
        self, 
        model: torch.nn.Module, 
        modules: Dict, 
        learning_config: LearningConfig,
        config: Dict[str, Any]
    ):
        """モデル学習"""
        learning_type = config.get('learning', {}).get('type', 'supervised')
        
        # ダミー学習データ生成
        training_data = self._generate_dummy_training_data(100)
        
        if learning_type == 'reinforcement':
            print("🎮 強化学習を実行...")
            learner = DQNReinforcementLearning()
        else:
            print("📚 教師あり学習を実行...")
            learner = SupervisedLearning()
        
        # 学習実行
        trained_model = learner.train(model, training_data, learning_config)
        
        # 評価
        test_data = self._generate_dummy_training_data(20)
        evaluation_results = learner.evaluate(trained_model, test_data)
        
        training_results = {
            'learning_type': learning_type,
            'training_history': learner.get_training_history()[-10:],  # 最後の10エポックのみ
            'evaluation': evaluation_results
        }
        
        return trained_model, training_results
    
    def _generate_dummy_training_data(self, num_samples: int) -> List[Dict]:
        """ダミー学習データ生成"""
        import random
        import numpy as np
        
        data = []
        for _ in range(num_samples):
            sample = {
                'board': np.random.randint(0, 5, (6, 6)),
                'player': random.choice(['A', 'B']),
                'my_pieces': {
                    (random.randint(0, 5), random.randint(0, 5)): random.choice(['good', 'bad'])
                    for _ in range(random.randint(1, 4))
                },
                'turn': random.randint(1, 50),
                'true_labels': {
                    (random.randint(0, 5), random.randint(0, 5)): random.choice(['good', 'bad'])
                    for _ in range(random.randint(1, 3))
                }
            }
            data.append(sample)
        
        return data
    
    def _save_ai(self, ai_info: Dict, model: torch.nn.Module, modules: Dict) -> Path:
        """AIを保存"""
        ai_dir = self.output_dir / ai_info['name']
        ai_dir.mkdir(exist_ok=True)
        
        # 1. AI情報保存
        info_path = ai_dir / "ai_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(ai_info, f, indent=2, ensure_ascii=False, default=str)
        
        # 2. モデル保存
        model_path = ai_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': ai_info['model']
        }, model_path)
        
        # 3. モジュール設定保存
        modules_path = ai_dir / "modules.json"
        module_configs = {
            name: module.get_config() 
            for name, module in modules.items()
        }
        with open(modules_path, 'w', encoding='utf-8') as f:
            json.dump(module_configs, f, indent=2, ensure_ascii=False, default=str)
        
        # 4. 実行可能Pythonスクリプト生成
        self._generate_executable_script(ai_dir, ai_info, modules)
        
        return ai_dir
    
    def _generate_executable_script(self, ai_dir: Path, ai_info: Dict, modules: Dict):
        """実行可能なPythonスクリプトを生成"""
        script_content = f'''#!/usr/bin/env python3
"""
自動生成AI: {ai_info['name']}
作成日時: {ai_info['creation_time']}
AI Maker System により自動生成
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# ===== CQCNNモデル定義 =====
class CQCNNModel(nn.Module):
    def __init__(self, n_qubits={ai_info['model']['n_qubits']}, n_layers={ai_info['model']['n_layers']}):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical CNN部分
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Quantum-inspired部分
        self.quantum_dim = n_qubits * n_layers
        self.quantum_linear = None
        self.quantum_layers = nn.ModuleList([
            nn.Linear(self.quantum_dim, self.quantum_dim) 
            for _ in range(n_layers)
        ])
        
        # 出力層
        self.classifier = nn.Sequential(
            nn.Linear(self.quantum_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )
        
        self._initialize_linear_layers()
    
    def _initialize_linear_layers(self):
        dummy_input = torch.randn(1, 7, 6, 6)
        with torch.no_grad():
            x = torch.relu(self.conv1(dummy_input))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            
            flattened_size = x.view(x.size(0), -1).size(1)
            self.quantum_linear = nn.Linear(flattened_size, self.quantum_dim)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.quantum_linear(x))
        
        for quantum_layer in self.quantum_layers:
            x_new = quantum_layer(x)
            x = torch.nn.functional.normalize(x_new + x, dim=1)
        
        output = self.classifier(x)
        return output

# ===== AI実行システム =====
class {ai_info['name'].replace(' ', '')}AI:
    def __init__(self):
        self.name = "{ai_info['name']}"
        self.model = CQCNNModel()
        self.load_model()
        
        print(f"🤖 {{self.name}} 初期化完了")
        print(f"📊 パラメータ数: {{sum(p.numel() for p in self.model.parameters())}}")
    
    def load_model(self):
        """学習済みモデルを読み込み"""
        try:
            checkpoint = torch.load("model.pth", map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("✅ 学習済みモデル読み込み成功")
        except Exception as e:
            print(f"⚠️ モデル読み込みエラー: {{e}}")
            print("ランダム初期化モデルを使用")
    
    def predict(self, board_state: np.ndarray, player: str, my_pieces: Dict, turn: int):
        """推論実行"""
        # 7チャンネルテンソル準備（簡略版）
        tensor = torch.randn(1, 7, 6, 6)  # ダミーテンソル
        
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
        
        return probabilities.numpy()[0]
    
    def get_action(self, game_state: Dict) -> Tuple:
        """行動決定"""
        # 戦略: {modules['action'].get_name()}
        predictions = self.predict(
            game_state.get('board', np.zeros((6, 6))),
            game_state.get('player', 'A'),
            game_state.get('my_pieces', {{}}),
            game_state.get('turn', 1)
        )
        
        # 簡略化された行動選択
        valid_actions = [(0, 0), (0, 1)]  # ダミー
        return valid_actions[0] if valid_actions else None

# ===== メイン実行 =====
if __name__ == "__main__":
    ai = {ai_info['name'].replace(' ', '')}AI()
    
    # テスト実行
    test_state = {{
        'board': np.random.randint(0, 5, (6, 6)),
        'player': 'A',
        'my_pieces': {{(0, 0): 'good', (0, 1): 'bad'}},
        'turn': 1
    }}
    
    action = ai.get_action(test_state)
    print(f"🎯 選択された行動: {{action}}")
    
    predictions = ai.predict(test_state['board'], test_state['player'], 
                           test_state['my_pieces'], test_state['turn'])
    print(f"🔮 予測結果: {{predictions}}")
    
    print(f"✅ {{ai.name}} テスト完了!")
'''
        
        script_path = ai_dir / f"{ai_info['name'].replace(' ', '_')}_ai.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"🐍 実行可能スクリプト生成: {script_path}")
    
    def list_available_modules(self) -> Dict[str, List[str]]:
        """利用可能なモジュール一覧"""
        return {
            'placement': PlacementFactory.get_available_strategies(),
            'estimator': EstimatorFactory.get_available_estimators(),
            'reward': RewardFactory.get_available_rewards(),
            'qmap': QMapFactory.get_available_generators(),
            'action': ActionFactory.get_available_selectors()
        }
    
    def get_built_ais(self) -> List[Dict]:
        """作成されたAI一覧"""
        return self.built_ais
    
    def create_ai_from_3step_config(self, step_config: Dict) -> Dict[str, Any]:
        """3stepシステムの設定からAI作成"""
        # 3stepの設定をai_builderの形式に変換
        ai_config = {
            'name': f"3Step_{step_config.get('reward', 'balanced')}_{datetime.now().strftime('%H%M%S')}",
            'placement': {'type': step_config.get('placement', 'standard')},
            'estimator': {'type': 'cqcnn'},
            'reward': {'type': step_config.get('reward', 'basic')},
            'qmap': {'type': 'strategic', 'strategy': step_config.get('reward', 'balanced')},
            'action': {'type': step_config.get('action', 'epsilon_greedy')},
            'model': {
                'n_qubits': step_config.get('qubits', 6),
                'n_layers': step_config.get('layers', 3)
            },
            'learning': {
                'type': step_config.get('learningMethod', 'supervised'),
                'learning_rate': step_config.get('learningRate', 0.001),
                'batch_size': step_config.get('batchSize', 32),
                'epochs': step_config.get('episodes', 100)
            },
            'auto_train': False  # 3stepでは手動学習
        }
        
        return self.create_ai(ai_config)


# ===== 使用例とテスト =====
def demo_ai_creation():
    """AI作成のデモ"""
    builder = AIBuilder()
    
    print("📋 利用可能なモジュール:")
    modules = builder.list_available_modules()
    for module_type, options in modules.items():
        print(f"  {module_type}: {', '.join(options)}")
    
    print("\n🔨 サンプルAI作成:")
    
    # サンプル設定
    configs = [
        {
            'name': 'AggressiveQuantumAI',
            'placement': {'type': 'aggressive'},
            'estimator': {'type': 'cqcnn', 'n_qubits': 8, 'n_layers': 4},
            'reward': {'type': 'aggressive'},
            'qmap': {'type': 'strategic', 'strategy': 'aggressive'},
            'action': {'type': 'epsilon_greedy', 'epsilon': 0.1},
            'model': {'n_qubits': 8, 'n_layers': 4},
            'learning': {'type': 'reinforcement', 'episodes': 500},
            'auto_train': True
        },
        {
            'name': 'DefensiveQuantumAI',
            'placement': {'type': 'defensive'},
            'estimator': {'type': 'cqcnn'},
            'reward': {'type': 'defensive'},
            'qmap': {'type': 'strategic', 'strategy': 'defensive'},
            'action': {'type': 'boltzmann', 'temperature': 0.8},
            'auto_train': False
        }
    ]
    
    for config in configs:
        ai_info = builder.create_ai(config)
        print(f"✅ 作成完了: {ai_info['name']}")
    
    print(f"\n📊 作成されたAI数: {len(builder.get_built_ais())}")


if __name__ == "__main__":
    demo_ai_creation()
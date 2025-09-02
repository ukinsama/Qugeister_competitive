# AI Maker System

**cqcnn_battle_learning_systemをベースにしたモジュール化されたAI制作システム**

## 🎯 概要

AI Maker Systemは、`cqcnn_battle_learning_system.py`の機能を5つの独立したモジュールに分離し、組み合わせ自由なAI制作システムとして再構築したものです。

## 📁 システム構造

```
ai_maker_system/
├── __init__.py                 # パッケージ初期化
├── core/                       # コアシステム
│   ├── __init__.py
│   ├── base_modules.py         # 抽象基底クラス・CQCNNモデル
│   ├── data_processor.py       # 7チャンネルデータ処理
│   └── game_state.py           # ゲーム状態・設定管理
├── modules/                    # 5つの機能モジュール
│   ├── __init__.py
│   ├── placement.py            # 初期配置戦略
│   ├── estimator.py            # 敵駒推定器（CQCNN学習付き）
│   ├── reward.py               # 報酬関数
│   ├── qmap.py                 # Q値マップ生成器
│   └── action.py               # 行動選択器
├── learning/                   # 学習システム
│   ├── __init__.py
│   ├── supervised.py           # 教師あり学習
│   └── reinforcement.py        # 強化学習（DQN）
├── ai_builder.py               # AIビルダー（メインシステム）
└── README.md                   # このファイル
```

## 🚀 使用方法

### 基本的な使い方

```python
from ai_maker_system import AIBuilder

# AIビルダー初期化
builder = AIBuilder(output_dir="my_ais")

# AI設定を定義
config = {
    'name': 'MyQuantumAI',
    'placement': {'type': 'aggressive'},
    'estimator': {'type': 'cqcnn', 'n_qubits': 6, 'n_layers': 3},
    'reward': {'type': 'aggressive'},
    'qmap': {'type': 'strategic', 'strategy': 'aggressive'},
    'action': {'type': 'epsilon_greedy', 'epsilon': 0.1},
    'model': {'n_qubits': 6, 'n_layers': 3},
    'learning': {'type': 'reinforcement', 'episodes': 500},
    'auto_train': True
}

# AI作成
ai_info = builder.create_ai(config)
print(f"AI作成完了: {ai_info['name']}")
```

### デモ実行

```bash
cd Qugeister_competitive
python ai_maker_demo.py
```

## 📦 5つの機能モジュール

### 1. 配置戦略モジュール (placement.py)
- `StandardPlacement`: 標準的な配置
- `AggressivePlacement`: 攻撃的配置（善玉を前線に）
- `DefensivePlacement`: 防御的配置（善玉を後方に）
- `CustomPlacement`: カスタム配置

### 2. 推定器モジュール (estimator.py)
- `CQCNNEstimator`: CQCNN学習機能付き推定器
- `SimpleEstimator`: シンプルなランダム推定器

### 3. 報酬関数モジュール (reward.py)
- `BasicReward`: 基本的な報酬関数
- `AggressiveReward`: 攻撃的報酬関数
- `DefensiveReward`: 防御的報酬関数
- `EscapeReward`: 脱出重視報酬関数

### 4. Q値マップ生成器モジュール (qmap.py)
- `SimpleQMapGenerator`: シンプルなQ値マップ
- `StrategicQMapGenerator`: 戦略的Q値マップ
- `LearnedQMapGenerator`: 学習済みモデルベース

### 5. 行動選択器モジュール (action.py)
- `GreedySelector`: 貪欲選択
- `EpsilonGreedySelector`: ε-貪欲選択
- `BoltzmannSelector`: ボルツマン選択
- `UCBSelector`: UCB選択
- `RandomSelector`: ランダム選択

## 🎓 学習システム

### 教師あり学習 (supervised.py)
- CQCNNモデルの教師あり学習
- バッチ学習・検証・Early Stopping
- 学習履歴の記録

### 強化学習 (reinforcement.py)  
- DQN（Deep Q-Network）実装
- 経験リプレイ・ターゲットネットワーク
- ε-greedy探索

## 🏗️ AIビルダーシステム

`AIBuilder`クラスが全体を統合し、以下の機能を提供:

- **AI作成**: 設定に基づいてAI生成
- **自動学習**: オプションで学習も実行
- **自動保存**: モデル・設定・実行スクリプトを保存
- **3step互換**: 既存の3stepシステムとの互換性

## 💾 出力形式

各AIは以下のファイルが生成されます:

```
generated_ais/MyQuantumAI/
├── ai_info.json              # AI詳細情報
├── model.pth                 # 学習済みモデル
├── modules.json              # モジュール設定
└── MyQuantumAI_ai.py         # 実行可能スクリプト
```

## 🔧 カスタマイズ

### 新しいモジュールの追加

1. 適切な抽象基底クラスを継承
2. `process()`, `get_name()`, `get_config()` を実装
3. ファクトリクラスに追加

例:
```python
class MyCustomReward(RewardFunction):
    def calculate_reward(self, state_before, action, state_after, player):
        # カスタム報酬ロジック
        return reward_value
    
    def get_name(self):
        return "My Custom Reward"
    
    def get_config(self):
        return {"type": "custom", "param": "value"}
```

## 🎯 3stepシステムとの統合

3stepシステムで生成された設定から直接AIを作成可能:

```python
# 3stepシステムの設定
step3_config = {
    'reward': 'aggressive',
    'qubits': 8,
    'layers': 3,
    'placement': 'aggressive',
    'learningMethod': 'reinforcement'
}

# AI作成
ai = builder.create_ai_from_3step_config(step3_config)
```

## 🔬 特徴

- **モジュール化**: 機能ごとに独立したモジュール
- **拡張性**: 新しいモジュールを簡単に追加可能
- **再利用性**: 既存のcqcnnコードを85%再利用
- **互換性**: 3stepシステムとの完全互換
- **自動化**: 学習からデプロイまで自動化

## 📊 利用可能なオプション

```python
# 全ての利用可能なオプションを表示
builder = AIBuilder()
modules = builder.list_available_modules()
print(modules)
# {
#   'placement': ['standard', 'aggressive', 'defensive', 'custom'],
#   'estimator': ['cqcnn', 'simple'], 
#   'reward': ['basic', 'aggressive', 'defensive', 'escape'],
#   'qmap': ['simple', 'strategic', 'learned'],
#   'action': ['greedy', 'epsilon_greedy', 'boltzmann', 'ucb', 'random']
# }
```

## 🎉 まとめ

AI Maker Systemにより、cqcnn_battle_learning_systemの複雑な機能を簡単に組み合わせてAIを作成できます。3stepシステムとの連携により、ビジュアル設定からコード生成、学習、実行まで一貫したワークフローを提供します。
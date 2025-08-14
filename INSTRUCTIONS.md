# 🎮 Qugeister競技システム - 使用手順

## 🚀 クイックスタート

### 1. 基本起動（推奨）
```bash
python quick_run.py
```
メニューから選択：
- **1**: 修正版GUI（エラー修正済み）
- **2**: Jupyter Notebook（AI設計）
- **3**: 統合テスト
- **4**: すべて実行

### 2. 個別起動
```bash
# GUI のみ
cd gui && python fixed_game_viewer.py

# Notebook のみ
jupyter notebook ai_design_notebook.ipynb

# 統合テスト のみ
python notebook_integration.py
```

## 📁 ファイル構成

```
qugeister_competitive/
├── gui/
│   └── fixed_game_viewer.py    # 修正版GUI（unhashable list エラー修正）
├── ai_design_notebook.ipynb    # AI設計用Jupyter Notebook
├── notebook_integration.py     # 統合システム
├── quick_run.py               # ワンクリック実行
├── INSTRUCTIONS.md            # この手順書
└── saved_configs/             # AI設定保存フォルダ（自動作成）
```

## 🛠️ セットアップ

### 必要なパッケージ
```bash
pip install pygame matplotlib jupyter pandas pillow scipy numpy
```

### オプション
```bash
pip install torch pennylane  # 量子機械学習（高度な機能用）
```

## 🎨 AI設計の流れ

### 1. Jupyter Notebookでカスタマイズ
```python
my_custom_config = {
    'strategy': 'aggressive',     # 戦略タイプ
    'risk_level': 0.8,           # リスク許容度
    'exploration_rate': 0.2,     # 探索率
    'memory_depth': 8,           # 記憶深度
    'bluff_probability': 0.3     # ブラフ頻度
}
```

### 2. 設定保存
```python
save_ai_config(my_custom_config, "my_super_ai.json")
```

### 3. GUIでテスト
- 修正版GUIを起動
- カスタムAI vs 他のAIで対戦
- 結果を分析

## 🏆 トーナメント機能

修正版GUIに搭載：
- **複数AI対戦**: 異なる設定のAI同士を対戦
- **勝率統計**: 各AIの勝率を記録
- **リアルタイム表示**: ゲーム進行をビジュアル表示
- **結果保存**: 対戦結果をファイル保存

## 🔧 トラブルシューティング

### よくあるエラー

#### "unhashable type: 'list'"
→ 修正版GUIを使用してください（`fixed_game_viewer.py`）

#### "pygame not found"
```bash
pip install pygame
```

#### "jupyter command not found"  
```bash
pip install jupyter
```

#### GUI が真っ黒
→ パッケージ再インストール：`pip install --upgrade pygame matplotlib`

### ログファイル
エラーが発生した場合：
- `error_log.txt` を確認
- `quick_run.py` を使用してデバッグ

## 🎯 カスタマイズ例

### 攻撃的AI
```python
aggressive_config = {
    'strategy': 'aggressive',
    'risk_level': 0.9,
    'exploration_rate': 0.1,
    'memory_depth': 5,
    'bluff_probability': 0.4
}
```

### 守備的AI  
```python
defensive_config = {
    'strategy': 'defensive',
    'risk_level': 0.2,
    'exploration_rate': 0.05,
    'memory_depth': 10,
    'bluff_probability': 0.1
}
```

### 学習型AI
```python
learning_config = {
    'strategy': 'balanced',
    'risk_level': 0.5,
    'exploration_rate': 0.3,  # 高い探索率で学習
    'memory_depth': 15,       # 長期記憶
    'bluff_probability': 0.25
}
```

## 📊 性能評価

### 統計機能
- 対戦勝率
- 平均ゲーム時間
- 戦略効果分析
- 相手適応度

### 結果出力
- CSVファイル出力
- グラフ生成（matplotlib）
- 統計レポート

## 🆘 サポート

問題が発生した場合：
1. `quick_run.py` のメニューから「3. 統合テスト」を実行
2. エラーログを確認
3. パッケージを再インストール
4. システムを再起動

---
*Qugeister競技システム v1.0 - 修正完成版*

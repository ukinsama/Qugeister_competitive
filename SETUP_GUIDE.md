# 🏆 Qugeister Competition Environment Setup

ミニマル大会実行のための環境構築ガイド

## 📋 システム要件

- Python 3.8以上 (推奨: Python 3.11)
- 4GB以上のRAM
- 1GB以上の空きディスク容量

## 🚀 クイックセットアップ

### 1. リポジトリクローン
```bash
git clone <あなたのリポジトリURL>
cd Qugeister_competitive
```

### 2. 仮想環境作成・有効化
```bash
# 仮想環境作成
python -m venv qugeister-env

# 有効化 (macOS/Linux)
source qugeister-env/bin/activate

# 有効化 (Windows)
qugeister-env\Scripts\activate
```

### 3. 依存関係インストール
```bash
# 基本的な依存関係
pip install torch numpy matplotlib

# または requirements.txtを使用
pip install -r requirements_minimal.txt
```

## 📦 ミニマル依存関係 (requirements_minimal.txt)

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## 🎮 大会実行方法

### 基本的なAI対戦
```bash
# 1. トーナメント実行
python tournament/tournament_manager.py

# 2. 個別バトル観戦
python battle_viewer/battle_viewer.py
```

### AI作成と学習
```bash
# 1. AIシステムでAI作成
python examples/integration_example.py

# 2. 学習実行
python quick_aggressive_recipe.py
python quick_defensive_recipe.py
python quick_escape_recipe.py

# 3. トーナメント実行
python tournament/tournament_manager.py
```

## 🧪 動作確認

### 環境テスト
```bash
python -c "import torch; import numpy; print('✅ PyTorch:', torch.__version__); print('✅ NumPy:', numpy.__version__)"
```

### システムテスト
```bash
# クリーンリポジトリのテスト（もしあれば）
python test_tournament_system.py
```

## 📁 重要なファイル・ディレクトリ

```
Qugeister_competitive/
├── tournament/                 # トーナメントシステム
├── ai_maker_system/            # AI作成システム
├── examples/                   # サンプルコード
├── quick_*_recipe.py          # 学習レシピ
├── trained_models/            # 学習済みモデル
└── tournament_results/        # 大会結果
```

## ⚡ ワンコマンドセットアップ

```bash
# 全部まとめて実行
git clone <リポジトリURL> && \
cd Qugeister_competitive && \
python -m venv qugeister-env && \
source qugeister-env/bin/activate && \
pip install torch numpy matplotlib && \
python examples/integration_example.py
```

## 🐛 トラブルシューティング

### PyTorchインストール問題
```bash
# CPU版のみインストール
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 権限エラー (Windows)
```bash
# PowerShellで実行ポリシー変更
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### メモリ不足
```bash
# スワップファイル使用を有効化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🏃‍♂️ 最小限の大会実行

最小限の操作で大会を開く場合:

```bash
# 1. 環境準備
python -m venv qugeister-env && source qugeister-env/bin/activate
pip install torch numpy

# 2. 既存モデルでトーナメント実行
python tournament/tournament_manager.py
```

## 📊 期待される出力例

```
🏆 段階3: トーナメント管理システム
============================================================
📁 発見された学習済みモデル: X個
🚀 総当たりトーナメント開始
参加モデル: X個
総対戦数: Y
[1/Y] ⚔️ ModelA vs ModelB
✅ トーナメント完了!
```

## 🤝 サポート

問題が発生した場合:
1. Python、PyTorchのバージョンを確認
2. 仮想環境が有効化されているか確認
3. エラーメッセージを確認

---
⚡ **Powered by Quantum-Inspired AI Technology**
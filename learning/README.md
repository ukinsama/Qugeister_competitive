# 🧠 Learning System - 段階2: AI学習システム

AIレシピから学習済みモデルを生成するシステム

## 📋 機能

- **バッチ学習**: 利用可能なすべてのAIレシピを自動学習
- **インタラクティブ学習**: レシピを選択して個別学習
- **モデル管理**: 学習済みモデルの自動保存・管理

## 🚀 使用方法

### バッチ学習（推奨）
```bash
cd learning
python recipe_trainer.py --batch
```

### インタラクティブ学習
```bash
cd learning
python recipe_trainer.py
```

## 📁 対応レシピファイル

- `quick_aggressive_recipe.py` - アグレッシブAI
- `quick_defensive_recipe.py` - ディフェンシブAI  
- `quick_escape_recipe.py` - 逃走AI
- `aggressiveai_integrated_recipe.py` - 統合アグレッシブAI
- `defensiveai_integrated_recipe.py` - 統合ディフェンシブAI
- `escapeai_integrated_recipe.py` - 統合逃走AI

## 📊 出力

学習完了後、以下が生成されます：
- `trained_models/[AI名]_[タイムスタンプ]/model.pth` - 学習済みモデル
- `trained_models/[AI名]_[タイムスタンプ]/ai_info.json` - AI情報

## 🔄 ワークフロー

1. **レシピ作成** (段階1) → AIレシピファイル作成
2. **学習実行** (段階2) → `python recipe_trainer.py --batch` 
3. **トーナメント** (段階3) → `python ../tournament_system/tournament_manager.py`
4. **個別観戦** (段階4) → 個別対戦観戦

## 💡 トラブルシューティング

### レシピが見つからない
```bash
# レシピファイルの存在確認
ls ../*recipe*.py
```

### 学習が失敗する
```bash
# 環境確認
python ../environment_check.py
```
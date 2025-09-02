# 🏆 Tournament System - 段階3: トーナメントシステム

学習済みAI同士の対戦トーナメントを実行

## 📋 機能

- **総当たりトーナメント**: 全AI同士の対戦実行
- **ランキング生成**: 勝率ベースのランキング作成  
- **詳細結果保存**: JSON・CSV形式での結果出力
- **リアルタイム進捗**: 対戦進捗のリアルタイム表示

## 🚀 使用方法

### 基本実行
```bash
python tournament/tournament_runner.py
```

### メインシステム直接実行  
```bash
python qugeister_ai_system/tournament_system/tournament_manager.py
```

### 簡単実行
```bash
python run_minimal_tournament.py
```

## 📁 必要なディレクトリ

トーナメント実行には以下のディレクトリに学習済みモデルが必要：
- `trained_models/` - メイン学習済みモデル
- `integrated_ais/` - 統合AI
- `quick_demo_ais/` - クイックデモAI  
- `tournament_system/ai_configs/` - トーナメント設定AI

## 📊 出力ファイル

実行後、以下のファイルが生成されます：
- `tournament_system/results/tournament_results_[日時].json` - 詳細対戦結果
- `tournament_system/results/rankings_[日時].csv` - ランキング表
- `tournament_system/results/win_rate_matrix_[日時].csv` - 勝率マトリックス

## 🔄 ワークフロー

1. **レシピ作成** (段階1) → AIレシピファイル作成
2. **学習実行** (段階2) → `python learning/recipe_trainer.py --batch`
3. **トーナメント** (段階3) → `python tournament/tournament_runner.py` ← **ここ**
4. **個別観戦** (段階4) → `python tournament/battle_viewer/battle_viewer.py`

## 💡 トラブルシューティング

### モデルが見つからない
```bash  
# 学習済みモデル確認
ls trained_models/
ls integrated_ais/

# モデルがない場合は学習実行
python learning/recipe_trainer.py --batch
```

### トーナメントエラー
```bash
# 環境確認
python environment_check.py

# メインシステム確認
ls qugeister_ai_system/tournament_system/
```
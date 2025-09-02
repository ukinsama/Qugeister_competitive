# 🎮 Qugeister AI System

量子インスパイアドAIを使用したGeisterゲームAI開発システム

## 🌟 特徴

- **3-Step AI Designer**: ビジュアルインターフェースでAI設計
- **AI Maker System**: モジュール型AI構築システム  
- **Tournament System**: AI対戦トーナメント管理
- **Quantum-Inspired Models**: CQCNN (Quantum Convolutional Neural Network)ベース

## 🚀 クイックスタート

### インストール
```bash
git clone https://github.com/yourusername/qugeister-ai-system.git
cd qugeister-ai-system
pip install -r requirements.txt
```

### AIの作成と学習
```bash
# 3つの戦略AIを自動生成
python examples/integration_example.py

# AIの学習
python examples/train_ai.py
```

### トーナメント実行
```bash
# AIトーナメント実行
python tournament_system/tournament_manager.py
```

### ビジュアルAI設計
```bash
# ブラウザで3-Step Designerを開く
open 3step_designer/index.html
```

## 📁 プロジェクト構造

```
qugeister_ai_system/
├── core/               # コアゲームエンジン
├── ai_maker_system/    # モジュール型AI作成
├── tournament_system/  # トーナメント管理
├── 3step_designer/     # ビジュアルデザイナー
├── examples/           # サンプルコード
├── docs/               # ドキュメント
└── tests/              # テスト
```

## 🤖 AI戦略

- **Aggressive**: 前進重視の攻撃的戦略
- **Defensive**: 後退重視の防御的戦略  
- **Escape**: 脱出重視の回避戦略
- **Balanced**: バランス型戦略

## 📊 システムアーキテクチャ

### AI Maker System
モジュール構成:
- Placement Module: 初期配置戦略
- Estimator Module: 状態評価（CQCNN）
- Reward Module: 報酬関数
- QMap Module: Q値マップ生成
- Action Module: 行動選択

### Tournament System
- Round-robin総当たり戦
- ELOレーティング計算
- 統計分析機能
- バトル記録・再生

## 🛠️ 技術スタック

- Python 3.8+
- PyTorch 2.0+
- NumPy
- HTML5/CSS3/JavaScript (3-Step Designer)

## 📝 ライセンス

MIT License

## 🤝 貢献

プルリクエスト歓迎！

## 📧 お問い合わせ

[あなたのメールアドレス]

---

⚡ Powered by Quantum-Inspired AI Technology

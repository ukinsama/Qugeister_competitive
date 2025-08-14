# 🎮 Mini2ガイスター競技システム

## 概要
2駒版ガイスター（善玉1個、悪玉1個）で量子Q学習AIの性能を評価するシステム

## 特徴
- **量子Q学習**: PennyLaneベースの量子機械学習
- **高速対戦**: 短いゲームで迅速な評価
- **リアルタイム分析**: ライブランキングボード
- **量子優位性検証**: 古典AIとの比較分析

## クイックスタート

### 1. 基本実行
```bash
cd mini2_geister
python run_mini2.py
```

### 2. クイック評価
```bash
python mini2_integrated_system.py quick
```

### 3. フルベンチマーク
```bash
python mini2_integrated_system.py benchmark
```

### 4. 対話モード
```bash
python mini2_integrated_system.py interactive
```

## システム構成

### ゲームエンジン
- **Mini2GeisterGame**: 4×4盤面、2駒版ガイスター
- **状態表現**: 4チャンネル × 4×4 テンソル
- **勝利条件**: 脱出、善玉取り、悪玉取られ

### AIエージェント
- **QuantumQAgent**: 量子Q学習エージェント
- **RandomAgent**: 完全ランダム
- **SmartRandomAgent**: 改良ランダム

### 競技システム
- **SpeedTournament**: 高速対戦システム
- **LiveRankingBoard**: リアルタイムランキング
- **QuantumAdvantageAnalyzer**: 量子優位性分析

## 結果例

```
🏆 MINI2ガイスター ランキングボード
====================================
🥇 TrainedQuantum   | 勝率:  85.2% | 戦績: 171勝  29敗 | 平均手数:  8.3
🥈 SmartRandom      | 勝率:  65.1% | 戦績: 130勝  70敗 | 平均手数: 12.1  
🥉 Random           | 勝率:  22.4% | 戦績:  45勝 155敗 | 平均手数: 15.7
====================================
⚛️ 量子優位性: 確認
```

## 設定カスタマイズ

`config.json`で各種パラメータを調整:
- 量子回路パラメータ
- 学習ハイパーパラメータ  
- 評価設定

## ファイル構成

```
mini2_geister/
├── mini2_integrated_system.py  # 統合システム
├── run_mini2.py               # 実行スクリプト
├── config.json               # 設定ファイル
├── models/                   # 訓練済みモデル
├── results/                  # 評価結果
└── logs/                    # ログファイル
```

## 今後の拡張

- [ ] より複雑な量子回路
- [ ] ハイブリッド古典-量子学習  
- [ ] 実量子デバイス対応
- [ ] マルチエージェント学習

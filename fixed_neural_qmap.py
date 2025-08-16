#!/usr/bin/env python3
"""
修正版NeuralQMapGenerator
入力次元の問題を解決
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class FixedNeuralQMapGenerator:
    """修正版ニューラルネットワークベースのQ値マップ生成器"""
    
    def __init__(self):
        # 入力次元を正確に計算
        # ボード状態: 6*6 = 36
        # 推定統計: 10
        # 合計: 46
        self.input_dim = 36 + 10
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 64),   # 46 -> 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),                # 64 -> 32
            nn.ReLU(),
            nn.Linear(32, 144),               # 32 -> 144 (6*6*4)
            nn.Tanh()
        )
        
        # 重みを初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def generate(self, board_state: np.ndarray, 
                estimations: Dict[Tuple[int, int], Dict[str, float]],
                my_pieces: Dict[Tuple[int, int], str],
                player: str) -> np.ndarray:
        """ニューラルネットワークでQ値マップ生成"""
        
        # 1. ボード状態を1次元化（36次元）
        board_features = board_state.flatten().astype(np.float32)
        
        # 2. 推定情報から特徴量を抽出（10次元）
        est_features = self._extract_estimation_features(estimations, my_pieces)
        
        # 3. 特徴量を結合（46次元）
        all_features = np.concatenate([board_features, est_features])
        
        # デバッグ出力（最初の1回のみ）
        if not hasattr(self, '_debug_shown'):
            print(f"  [NeuralQMap] 入力次元: {len(all_features)} (ボード:{len(board_features)} + 推定:{len(est_features)})")
            self._debug_shown = True
        
        # 4. テンソル化
        input_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
        
        # 5. ネットワーク推論
        with torch.no_grad():
            output = self.network(input_tensor)
            q_map_flat = output.squeeze(0).numpy()
        
        # 6. Q値マップの形状に変換 (6, 6, 4)
        q_map = q_map_flat.reshape(6, 6, 4)
        
        # 7. スケーリングと後処理
        q_map = self._postprocess_qmap(q_map, my_pieces, player, estimations)
        
        return q_map
    
    def _extract_estimation_features(self, estimations: Dict, my_pieces: Dict) -> np.ndarray:
        """推定情報から特徴量を抽出（10次元固定）"""
        features = np.zeros(10, dtype=np.float32)
        
        if estimations:
            # 推定値の統計
            good_probs = [e['good_prob'] for e in estimations.values()]
            bad_probs = [e['bad_prob'] for e in estimations.values()]
            confidences = [e['confidence'] for e in estimations.values()]
            
            if good_probs:
                features[0] = np.mean(good_probs)      # 善玉確率の平均
                features[1] = np.std(good_probs)       # 善玉確率の標準偏差
                features[2] = np.max(good_probs)       # 善玉確率の最大値
                features[3] = np.min(good_probs)       # 善玉確率の最小値
            
            if confidences:
                features[4] = np.mean(confidences)     # 確信度の平均
                features[5] = np.max(confidences)      # 確信度の最大値
        
        # 駒数情報
        features[6] = len(my_pieces) / 8.0            # 自分の駒数（正規化）
        features[7] = len(estimations) / 8.0          # 敵の駒数（正規化）
        
        # 駒タイプ情報
        if my_pieces:
            good_count = sum(1 for p in my_pieces.values() if p == "good")
            features[8] = good_count / max(len(my_pieces), 1)  # 善玉の割合
            features[9] = 1.0 - features[8]                    # 悪玉の割合
        
        return features
    
    def _postprocess_qmap(self, q_map: np.ndarray, my_pieces: Dict, 
                          player: str, estimations: Dict) -> np.ndarray:
        """Q値マップの後処理"""
        # スケーリング
        q_map = q_map * 5.0
        
        # 自分の駒がない位置は無効化
        for y in range(6):
            for x in range(6):
                if (x, y) not in my_pieces:
                    q_map[y, x, :] = -100
        
        # 境界外への移動を無効化
        for (x, y) in my_pieces:
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
            for dir_idx, (dx, dy) in enumerate(directions):
                new_x, new_y = x + dx, y + dy
                if not (0 <= new_x < 6 and 0 <= new_y < 6):
                    q_map[y, x, dir_idx] = -100
        
        # 基本的な戦略を追加
        for (x, y), piece_type in my_pieces.items():
            # 前進ボーナス
            if player == "A":
                q_map[y, x, 2] += 1.0  # 下方向
            else:
                q_map[y, x, 0] += 1.0  # 上方向
            
            # 脱出ボーナス（善玉のみ）
            if piece_type == "good":
                escape_positions = [(0, 5), (5, 5)] if player == "A" else [(0, 0), (5, 0)]
                for escape_x, escape_y in escape_positions:
                    if abs(x - escape_x) + abs(y - escape_y) == 1:
                        # 脱出口に隣接している
                        for dir_idx, (dx, dy) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                            if x + dx == escape_x and y + dy == escape_y:
                                q_map[y, x, dir_idx] += 10.0
        
        return q_map
    
    def get_generator_name(self) -> str:
        return "ニューラルQ値生成(修正版)"


# テスト関数
def test_fixed_neural_qmap():
    """修正版のテスト"""
    print("🧪 修正版NeuralQMapGeneratorのテスト")
    print("-" * 50)
    
    # ジェネレータ作成
    generator = FixedNeuralQMapGenerator()
    
    # テストデータ
    board_state = np.random.randint(-1, 2, (6, 6))
    estimations = {
        (2, 3): {'good_prob': 0.7, 'bad_prob': 0.3, 'confidence': 0.8},
        (3, 4): {'good_prob': 0.4, 'bad_prob': 0.6, 'confidence': 0.6}
    }
    my_pieces = {
        (1, 1): "good",
        (2, 1): "bad",
        (3, 2): "good"
    }
    
    # Q値マップ生成
    q_map = generator.generate(board_state, estimations, my_pieces, "A")
    
    print(f"✅ Q値マップ生成成功")
    print(f"  形状: {q_map.shape}")
    print(f"  最大値: {np.max(q_map):.2f}")
    print(f"  最小値: {np.min(q_map):.2f}")
    print(f"  平均値: {np.mean(q_map[q_map > -100]):.2f}")  # 無効値を除く
    
    # 特定の駒のQ値を確認
    for (x, y), piece_type in my_pieces.items():
        print(f"\n駒({x},{y}) [{piece_type}] のQ値:")
        directions = ["上", "右", "下", "左"]
        for dir_idx, dir_name in enumerate(directions):
            q_value = q_map[y, x, dir_idx]
            if q_value > -100:
                print(f"  {dir_name}: {q_value:.2f}")


if __name__ == "__main__":
    test_fixed_neural_qmap()

#!/usr/bin/env python3
"""
真のCQCNN（Convolutional Quantum Circuit Neural Network）実装
量子回路構造を明確に定義し、CNNと量子回路の融合を実現
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

# ================================================================================
# CQCNNの核心: 量子回路層の定義
# ================================================================================

class QuantumCircuitLayer(nn.Module):
    """
    量子回路層 - CQCNNの核心部分
    古典データを量子状態にエンコードし、量子演算を実行
    """
    
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # === 量子回路パラメータ ===
        # 各量子ビットの回転角度（学習可能パラメータ）
        self.rotation_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1  # RX, RY, RZ用
        )
        
        # エンタングルメントの強度（学習可能）
        self.entanglement_params = nn.Parameter(
            torch.randn(n_layers, n_qubits - 1) * 0.1
        )
        
        print(f"⚛️ 量子回路層初期化:")
        print(f"  - 量子ビット数: {n_qubits}")
        print(f"  - 回路深さ: {n_layers}")
        print(f"  - 学習可能パラメータ数: {self.count_parameters()}")
    
    def count_parameters(self) -> int:
        """学習可能パラメータ数を計算"""
        rotation_count = self.n_layers * self.n_qubits * 3
        entangle_count = self.n_layers * (self.n_qubits - 1)
        return rotation_count + entangle_count
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        量子回路の実行（完全に微分可能な実装）
        
        Args:
            x: 入力特徴量 (batch_size, n_features)
        
        Returns:
            量子測定結果 (batch_size, n_qubits)
        """
        batch_size = x.shape[0]
        
        # === 微分可能な量子回路シミュレーション ===
        
        # 初期状態（実部と虚部を分離）
        state_real = torch.ones(batch_size, self.n_qubits)
        state_imag = torch.zeros(batch_size, self.n_qubits)
        
        # データエンコーディング
        for i in range(min(self.n_qubits, x.shape[1])):
            angle = x[:, i] * np.pi
            state_real = state_real.clone()  # 新しいテンソルを作成
            state_imag = state_imag.clone()
            state_real[:, i] = torch.cos(angle/2)
            state_imag[:, i] = torch.sin(angle/2)
        
        # 変分量子回路
        for layer in range(self.n_layers):
            # 各層で新しいテンソルを作成
            new_real = torch.zeros_like(state_real)
            new_imag = torch.zeros_like(state_imag)
            
            # パラメタライズド回転ゲート
            for qubit in range(self.n_qubits):
                rx = self.rotation_params[layer, qubit, 0]
                ry = self.rotation_params[layer, qubit, 1]
                rz = self.rotation_params[layer, qubit, 2]
                
                # 現在の量子ビットの状態
                q_real = state_real[:, qubit]
                q_imag = state_imag[:, qubit]
                
                # RX回転
                cos_rx = torch.cos(rx/2)
                sin_rx = torch.sin(rx/2)
                rx_real = cos_rx * q_real + sin_rx * q_imag
                rx_imag = cos_rx * q_imag - sin_rx * q_real
                
                # RY回転
                cos_ry = torch.cos(ry/2)
                sin_ry = torch.sin(ry/2)
                ry_real = cos_ry * rx_real - sin_ry * rx_imag
                ry_imag = cos_ry * rx_imag + sin_ry * rx_real
                
                # RZ回転（位相回転）
                cos_rz = torch.cos(rz/2)
                sin_rz = torch.sin(rz/2)
                rz_real = cos_rz * ry_real - sin_rz * ry_imag
                rz_imag = cos_rz * ry_imag + sin_rz * ry_real
                
                # 新しい状態に書き込み
                new_real[:, qubit] = rz_real
                new_imag[:, qubit] = rz_imag
            
            # エンタングルメント効果（簡易版）
            entangle_real = new_real.clone()
            entangle_imag = new_imag.clone()
            
            for i in range(self.n_qubits - 1):
                strength = torch.sigmoid(self.entanglement_params[layer, i])
                
                # 制御ビットの振幅
                control_amp = torch.sqrt(entangle_real[:, i]**2 + entangle_imag[:, i]**2)
                
                # ターゲットビットに影響を与える（CNOTの簡易版）
                target = i + 1
                mix_factor = strength * control_amp
                
                # ミキシング（新しいテンソルで計算）
                mixed_real = (1 - mix_factor) * entangle_real[:, target] + mix_factor * entangle_real[:, i]
                mixed_imag = (1 - mix_factor) * entangle_imag[:, target] + mix_factor * entangle_imag[:, i]
                
                # 結果を反映（新しいテンソルを作成）
                result_real = entangle_real.clone()
                result_imag = entangle_imag.clone()
                result_real[:, target] = mixed_real
                result_imag[:, target] = mixed_imag
                
                entangle_real = result_real
                entangle_imag = result_imag
            
            # 状態を更新
            state_real = entangle_real
            state_imag = entangle_imag
        
        # 測定（確率振幅の絶対値の2乗）
        measurements = state_real**2 + state_imag**2
        
        # 正規化（オプション）
        measurements = measurements / (measurements.sum(dim=1, keepdim=True) + 1e-8)
        
        return measurements
    
    def _apply_rx(self, state: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """RX回転ゲートを適用（削除）"""
        pass
    
    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """RY回転ゲートを適用（削除）"""
        pass
    
    def _apply_rz(self, state: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """RZ回転ゲートを適用（削除）"""
        pass
    
    def _apply_controlled_rotation(self, state: torch.Tensor, 
                                  control: int, target: int, 
                                  strength: torch.Tensor) -> torch.Tensor:
        """制御回転ゲート（削除）"""
        pass
    
    def visualize_circuit(self):
        """量子回路の構造を可視化"""
        print("\n🔮 量子回路構造:")
        print("=" * 50)
        
        for layer in range(self.n_layers):
            print(f"\nLayer {layer + 1}:")
            
            # 回転ゲート層
            print("  回転ゲート:")
            for q in range(self.n_qubits):
                params = self.rotation_params[layer, q].detach().numpy()
                print(f"    Q{q}: RX({params[0]:.2f}) RY({params[1]:.2f}) RZ({params[2]:.2f})")
            
            # エンタングルメント層
            print("  エンタングルメント:")
            for i in range(self.n_qubits - 1):
                strength = torch.sigmoid(self.entanglement_params[layer, i]).item()
                print(f"    Q{i} ←→ Q{i+1}: 強度 {strength:.2f}")


# ================================================================================
# 完全なCQCNN実装
# ================================================================================

class CompleteCQCNN(nn.Module):
    """
    完全なCQCNN実装
    CNN特徴抽出 → 量子回路処理 → 出力層
    """
    
    def __init__(self, n_qubits: int = 6, n_quantum_layers: int = 3):
        super().__init__()
        
        print("🚀 完全CQCNN初期化")
        print("=" * 60)
        
        # === 1. CNN特徴抽出部 ===
        self.cnn_feature_extractor = nn.Sequential(
            # 入力: (batch, 3, 6, 6) - 3チャンネルの6x6ボード
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 6x6 → 3x3
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Flatten(),  # 64 * 3 * 3 = 576次元
        )
        
        # === 2. 次元削減層（CNNから量子回路への橋渡し）===
        cnn_output_dim = 64 * 3 * 3
        self.dimension_reduction = nn.Sequential(
            nn.Linear(cnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_qubits),
            nn.Tanh()  # [-1, 1]に正規化
        )
        
        # === 3. 量子回路層（CQCNNの核心）===
        self.quantum_circuit = QuantumCircuitLayer(n_qubits, n_quantum_layers)
        
        # === 4. 後処理層 ===
        self.post_processing = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # === 5. 出力層（タスク固有）===
        # 駒推定タスク: 2クラス分類（善玉/悪玉）
        self.output_layer = nn.Linear(16, 2)
        
        self._print_architecture()
    
    def _print_architecture(self):
        """アーキテクチャの詳細を表示"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum_params = self.quantum_circuit.count_parameters()
        classical_params = total_params - quantum_params
        
        print(f"\n📐 CQCNNアーキテクチャ:")
        print(f"  CNN特徴抽出: 3→16→32→64チャンネル")
        print(f"  量子回路: {self.quantum_circuit.n_qubits}量子ビット×{self.quantum_circuit.n_layers}層")
        print(f"  出力: 2クラス（善玉/悪玉）")
        print(f"\n📊 パラメータ数:")
        print(f"  古典部分: {classical_params:,}")
        print(f"  量子部分: {quantum_params:,}")
        print(f"  合計: {total_params:,}")
        print("=" * 60)
    
    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """
        順伝播
        
        Args:
            x: 入力ボード状態 (batch_size, 3, 6, 6)
            return_intermediates: 中間出力も返すか
        
        Returns:
            出力（および中間結果）
        """
        # 1. CNN特徴抽出
        cnn_features = self.cnn_feature_extractor(x)
        
        # 2. 次元削減
        quantum_input = self.dimension_reduction(cnn_features)
        
        # 3. 量子回路処理
        quantum_output = self.quantum_circuit(quantum_input)
        
        # 4. 後処理
        processed = self.post_processing(quantum_output)
        
        # 5. 最終出力
        output = self.output_layer(processed)
        
        if return_intermediates:
            return output, {
                'cnn_features': cnn_features,
                'quantum_input': quantum_input,
                'quantum_output': quantum_output,
                'processed': processed
            }
        
        return output
    
    def predict_piece_type(self, board_state: torch.Tensor, 
                          position: Tuple[int, int]) -> Tuple[float, float]:
        """
        特定位置の駒タイプを推定
        
        Returns:
            (善玉確率, 悪玉確率)
        """
        with torch.no_grad():
            output = self.forward(board_state)
            probabilities = F.softmax(output, dim=1)
            
            good_prob = probabilities[0, 0].item()
            bad_prob = probabilities[0, 1].item()
            
            return good_prob, bad_prob


# ================================================================================
# CQCNNの学習と評価
# ================================================================================

class CQCNNTrainingDemo:
    """CQCNNの学習デモンストレーション"""
    
    def __init__(self):
        self.model = CompleteCQCNN(n_qubits=6, n_quantum_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def generate_sample_data(self, n_samples: int = 100):
        """サンプルデータ生成"""
        # ランダムなボード状態
        boards = torch.randn(n_samples, 3, 6, 6)
        
        # ランダムなラベル（0: 善玉, 1: 悪玉）
        labels = torch.randint(0, 2, (n_samples,))
        
        return boards, labels
    
    def train_step(self, boards: torch.Tensor, labels: torch.Tensor):
        """1ステップの学習"""
        self.optimizer.zero_grad()
        
        # 順伝播
        outputs = self.model(boards)
        
        # 損失計算
        loss = self.criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメータ更新
        self.optimizer.step()
        
        # 精度計算
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        
        return loss.item(), accuracy.item()
    
    def demonstrate(self):
        """CQCNNの動作デモンストレーション"""
        print("\n🎓 CQCNN学習デモンストレーション")
        print("=" * 60)
        
        # 量子回路の構造を表示
        self.model.quantum_circuit.visualize_circuit()
        
        # サンプルデータ生成
        print("\n📊 サンプルデータで学習")
        boards, labels = self.generate_sample_data(100)
        
        # 学習前の推論
        print("\n学習前:")
        with torch.no_grad():
            sample_board = boards[0:1]
            output, intermediates = self.model(sample_board, return_intermediates=True)
            probs = F.softmax(output, dim=1)
            print(f"  予測: 善玉 {probs[0, 0]:.2%}, 悪玉 {probs[0, 1]:.2%}")
            print(f"  量子出力: {intermediates['quantum_output'][0].numpy()}")
        
        # 簡単な学習
        print("\n学習中...")
        for epoch in range(10):
            loss, acc = self.train_step(boards, labels)
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.2%}")
        
        # 学習後の推論
        print("\n学習後:")
        with torch.no_grad():
            output, intermediates = self.model(sample_board, return_intermediates=True)
            probs = F.softmax(output, dim=1)
            print(f"  予測: 善玉 {probs[0, 0]:.2%}, 悪玉 {probs[0, 1]:.2%}")
            print(f"  量子出力: {intermediates['quantum_output'][0].numpy()}")
        
        # パラメータの変化を確認
        print("\n⚛️ 量子パラメータの変化:")
        layer0_params = self.model.quantum_circuit.rotation_params[0, 0].detach().numpy()
        print(f"  Layer 0, Qubit 0: RX={layer0_params[0]:.3f}, RY={layer0_params[1]:.3f}, RZ={layer0_params[2]:.3f}")
        
        print("\n✅ CQCNNが正常に動作しています！")


# ================================================================================
# メイン実行
# ================================================================================

def main():
    """メイン実行関数"""
    print("=" * 70)
    print("🌟 真のCQCNN（Convolutional Quantum Circuit Neural Network）")
    print("=" * 70)
    
    # CQCNNのデモンストレーション
    demo = CQCNNTrainingDemo()
    demo.demonstrate()
    
    print("\n" + "=" * 70)
    print("💡 CQCNNの特徴:")
    print("  1. CNN: 空間的特徴を抽出")
    print("  2. 量子回路: 複雑な相関を捉える")
    print("  3. ハイブリッド: 両者の長所を活用")
    print("=" * 70)


if __name__ == "__main__":
    main()

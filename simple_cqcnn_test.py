#!/usr/bin/env python3
"""
シンプルCQCNN実装 - 動作確認版
量子回路なし、基本的なCNNで駒推定を学習
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random
import time

# ================================================================================
# Part 1: シンプルなCNNモデル
# ================================================================================

class SimpleCQCNN(nn.Module):
    """シンプルなCNN駒推定モデル（量子回路なし）"""
    
    def __init__(self):
        super().__init__()
        
        # CNN特徴抽出器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 6x6 -> 3x3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten()
        )
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 2クラス: good(0) or bad(1)
        )
    
    def forward(self, x):
        """順伝播"""
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


# ================================================================================
# Part 2: データ生成器
# ================================================================================

class SimpleDataGenerator:
    """シンプルな学習データ生成器"""
    
    @staticmethod
    def generate_random_board():
        """ランダムなボード状態を生成"""
        board = np.zeros((6, 6), dtype=np.float32)
        
        # プレイヤーAの駒を配置（1～4個）
        num_a = random.randint(1, 4)
        positions_a = random.sample([(i, j) for i in range(6) for j in range(6)], num_a)
        for pos in positions_a:
            board[pos[0], pos[1]] = 1
        
        # プレイヤーBの駒を配置（1～4個）
        available_positions = [(i, j) for i in range(6) for j in range(6) 
                              if board[i, j] == 0]
        num_b = min(random.randint(1, 4), len(available_positions))
        positions_b = random.sample(available_positions, num_b)
        for pos in positions_b:
            board[pos[0], pos[1]] = -1
        
        return board, positions_a, positions_b
    
    @staticmethod
    def board_to_tensor(board, player='A'):
        """ボードを3チャンネルテンソルに変換"""
        tensor = torch.zeros(3, 6, 6, dtype=torch.float32)
        
        player_val = 1 if player == 'A' else -1
        enemy_val = -player_val
        
        # チャンネル0: 自駒
        tensor[0] = torch.from_numpy((board == player_val).astype(np.float32))
        # チャンネル1: 敵駒
        tensor[1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        # チャンネル2: 空マス
        tensor[2] = torch.from_numpy((board == 0).astype(np.float32))
        
        return tensor
    
    @staticmethod
    def generate_batch(batch_size=32):
        """バッチデータを生成"""
        batch_inputs = []
        batch_labels = []
        
        for _ in range(batch_size):
            board, pos_a, pos_b = SimpleDataGenerator.generate_random_board()
            
            # プレイヤーAから見た場合
            if random.random() < 0.5:
                tensor = SimpleDataGenerator.board_to_tensor(board, 'A')
                # プレイヤーBの駒（敵駒）のラベルをランダムに設定
                label = random.randint(0, 1)  # 0: good, 1: bad
            else:
                # プレイヤーBから見た場合
                tensor = SimpleDataGenerator.board_to_tensor(board, 'B')
                # プレイヤーAの駒（敵駒）のラベルをランダムに設定
                label = random.randint(0, 1)
            
            batch_inputs.append(tensor)
            batch_labels.append(label)
        
        return torch.stack(batch_inputs), torch.tensor(batch_labels, dtype=torch.long)


# ================================================================================
# Part 3: トレーナー
# ================================================================================

class SimpleTrainer:
    """シンプルな学習管理クラス"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def train_epoch(self, num_batches=10, batch_size=32):
        """1エポックの学習"""
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        
        self.model.train()
        
        for _ in range(num_batches):
            # バッチデータ生成
            inputs, labels = SimpleDataGenerator.generate_batch(batch_size)
            
            # 順伝播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 逆伝播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 統計記録
            epoch_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
        
        avg_loss = np.mean(epoch_losses)
        accuracy = epoch_correct / epoch_total
        
        self.history['loss'].append(avg_loss)
        self.history['accuracy'].append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, epochs=50, num_batches=10, batch_size=32):
        """学習実行"""
        print("🎓 学習開始")
        print("=" * 60)
        
        for epoch in range(epochs):
            loss, accuracy = self.train_epoch(num_batches, batch_size)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {loss:.4f} | "
                      f"Accuracy: {accuracy:.3f}")
        
        print("\n✅ 学習完了！")
        self.plot_history()
    
    def plot_history(self):
        """学習履歴をプロット"""
        if not self.history['loss']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失
        ax1.plot(self.history['loss'], 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # 精度
        ax2.plot(self.history['accuracy'], 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('simple_cqcnn_training.png')
        print("📊 学習曲線を保存: simple_cqcnn_training.png")


# ================================================================================
# Part 4: テストと評価
# ================================================================================

class SimpleEvaluator:
    """モデル評価クラス"""
    
    @staticmethod
    def test_model(model, num_tests=100):
        """モデルをテスト"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _ in range(num_tests):
                inputs, labels = SimpleDataGenerator.generate_batch(1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    @staticmethod
    def inference_speed_test(model, num_iterations=100):
        """推論速度をテスト"""
        model.eval()
        
        # ウォームアップ
        dummy_input = torch.randn(1, 3, 6, 6)
        for _ in range(10):
            _ = model(dummy_input)
        
        # 速度測定
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                inputs, _ = SimpleDataGenerator.generate_batch(1)
                _ = model(inputs)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations * 1000  # ミリ秒
        
        return avg_time


# ================================================================================
# Part 5: メイン実行
# ================================================================================

def main():
    """メイン実行関数"""
    print("🚀 シンプルCQCNN動作確認")
    print("=" * 70)
    
    # 1. モデル作成
    print("\n[Step 1] モデル作成")
    print("-" * 40)
    model = SimpleCQCNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ モデル作成完了")
    print(f"   総パラメータ数: {total_params:,}")
    print(f"   学習可能パラメータ数: {trainable_params:,}")
    
    # 2. 学習
    print("\n[Step 2] 学習実行")
    print("-" * 40)
    trainer = SimpleTrainer(model, learning_rate=0.001)
    trainer.train(epochs=30, num_batches=20, batch_size=32)
    
    # 3. 評価
    print("\n[Step 3] モデル評価")
    print("-" * 40)
    
    # 精度テスト
    test_accuracy = SimpleEvaluator.test_model(model, num_tests=200)
    print(f"テスト精度: {test_accuracy:.1%}")
    
    # 速度テスト
    avg_inference_time = SimpleEvaluator.inference_speed_test(model, num_iterations=100)
    print(f"平均推論時間: {avg_inference_time:.2f}ms")
    
    # 4. モデル保存
    print("\n[Step 4] モデル保存")
    print("-" * 40)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': trainer.history
    }, 'simple_cqcnn_checkpoint.pth')
    print("💾 モデルを保存: simple_cqcnn_checkpoint.pth")
    
    # 5. サマリー
    print("\n" + "=" * 70)
    print("📊 実験サマリー")
    print("=" * 70)
    
    final_loss = trainer.history['loss'][-1] if trainer.history['loss'] else 0
    final_accuracy = trainer.history['accuracy'][-1] if trainer.history['accuracy'] else 0
    
    print(f"最終損失: {final_loss:.4f}")
    print(f"最終学習精度: {final_accuracy:.1%}")
    print(f"テスト精度: {test_accuracy:.1%}")
    print(f"推論速度: {avg_inference_time:.2f}ms/sample")
    
    # 期待値（ランダム）は50%なので、それを超えていれば学習成功
    if final_accuracy > 0.55:
        print("\n✅ 学習成功！モデルはランダム以上の性能を示しています。")
        print("   次のステップ: 実際のゲームデータで学習")
    else:
        print("\n⚠️  学習が不十分です。")
        print("   ハイパーパラメータの調整が必要かもしれません。")
    
    print("\n🎉 動作確認完了！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 実行中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()

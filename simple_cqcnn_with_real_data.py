#!/usr/bin/env python3
"""
シンプルCQCNN実装 - 実際のゲームデータ版
実際のゲームプレイから学習データを生成して学習
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
import time

# プロジェクトパス追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'qugeister_competitive')
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

# ゲームエンジンを読み込み
print("📂 ゲームエンジン読み込み中...")
try:
    with open(os.path.join(src_path, 'game_engine.py'), 'r') as f:
        game_engine_code = f.read()
    exec(game_engine_code)
    
    with open(os.path.join(src_path, 'ai_base.py'), 'r') as f:
        ai_base_code = f.read().replace('from .game_engine', '# from .game_engine')
    exec(ai_base_code)
    print("✅ ゲームエンジン読み込み完了")
except Exception as e:
    print(f"⚠️ ゲームエンジン読み込みエラー: {e}")
    print("ダミーモードで実行します")


# ================================================================================
# Part 1: シンプルなCNNモデル（前回と同じ）
# ================================================================================

class SimpleCQCNN(nn.Module):
    """シンプルなCNN駒推定モデル"""
    
    def __init__(self):
        super().__init__()
        
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
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


# ================================================================================
# Part 2: 実ゲームデータ生成器
# ================================================================================

class RealGameDataGenerator:
    """実際のゲームプレイからデータを生成"""
    
    @staticmethod
    def generate_game_data(num_games: int = 100) -> List[Dict]:
        """実際のゲームをプレイしてデータを収集"""
        print(f"🎮 {num_games}ゲームからデータ生成中...")
        all_data = []
        
        for game_idx in range(num_games):
            game = GeisterGame()
            player_a = RandomAI("A")
            player_b = RandomAI("B")
            
            # ゲーム進行
            turn_count = 0
            max_turns = 50
            
            while not game.game_over and turn_count < max_turns:
                current_player = player_a if game.current_player == "A" else player_b
                legal_moves = game.get_legal_moves(game.current_player)
                
                if not legal_moves:
                    break
                
                # ターン5以降のデータを収集（序盤は除外）
                if turn_count >= 5:
                    # 現在のボード状態を記録
                    board_state = game.board.copy()
                    
                    # プレイヤーAの視点からデータ作成
                    if random.random() < 0.5:
                        # プレイヤーBの駒（敵駒）の位置と種類を記録
                        for pos, piece_type in game.player_b_pieces.items():
                            if board_state[pos[1], pos[0]] == -1:  # 敵駒の確認
                                data = {
                                    'board': board_state.copy(),
                                    'player': 'A',
                                    'enemy_pos': pos,
                                    'enemy_type': piece_type,  # 'good' or 'bad'
                                    'turn': turn_count
                                }
                                all_data.append(data)
                    else:
                        # プレイヤーBの視点からデータ作成
                        for pos, piece_type in game.player_a_pieces.items():
                            if board_state[pos[1], pos[0]] == 1:  # 敵駒の確認
                                data = {
                                    'board': board_state.copy(),
                                    'player': 'B',
                                    'enemy_pos': pos,
                                    'enemy_type': piece_type,  # 'good' or 'bad'
                                    'turn': turn_count
                                }
                                all_data.append(data)
                
                # 手を実行
                move = current_player.get_move(
                    game.get_game_state(game.current_player), 
                    legal_moves
                )
                if move:
                    game.make_move(move[0], move[1])
                
                turn_count += 1
            
            if (game_idx + 1) % 20 == 0:
                print(f"  {game_idx + 1}/{num_games} ゲーム完了")
        
        print(f"✅ {len(all_data)}個のデータポイントを生成")
        return all_data
    
    @staticmethod
    def prepare_batch(data_list: List[Dict], batch_size: int = 32):
        """データリストからバッチを作成"""
        if len(data_list) < batch_size:
            batch_size = len(data_list)
        
        # ランダムサンプリング
        batch_data = random.sample(data_list, batch_size)
        
        batch_inputs = []
        batch_labels = []
        
        for data in batch_data:
            # ボードをテンソルに変換
            tensor = RealGameDataGenerator.board_to_tensor(
                data['board'], 
                data['player']
            )
            
            # ラベル（0: good, 1: bad）
            label = 0 if data['enemy_type'] == 'good' else 1
            
            batch_inputs.append(tensor)
            batch_labels.append(label)
        
        return torch.stack(batch_inputs), torch.tensor(batch_labels, dtype=torch.long)
    
    @staticmethod
    def board_to_tensor(board: np.ndarray, player: str) -> torch.Tensor:
        """ボードを3チャンネルテンソルに変換"""
        tensor = torch.zeros(3, 6, 6, dtype=torch.float32)
        
        player_val = 1 if player == 'A' else -1
        enemy_val = -player_val
        
        tensor[0] = torch.from_numpy((board == player_val).astype(np.float32))
        tensor[1] = torch.from_numpy((board == enemy_val).astype(np.float32))
        tensor[2] = torch.from_numpy((board == 0).astype(np.float32))
        
        return tensor


# ================================================================================
# Part 3: 改良版トレーナー
# ================================================================================

class ImprovedTrainer:
    """実データで学習するトレーナー"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
    
    def train_with_real_data(self, train_data: List[Dict], 
                            val_data: Optional[List[Dict]] = None,
                            epochs: int = 50, 
                            batch_size: int = 32):
        """実データで学習"""
        print(f"\n🎓 実データで学習開始")
        print(f"   学習データ数: {len(train_data)}")
        if val_data:
            print(f"   検証データ数: {len(val_data)}")
        print("=" * 60)
        
        for epoch in range(epochs):
            # 学習フェーズ
            self.model.train()
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            
            # データをシャッフル
            random.shuffle(train_data)
            
            # バッチごとに学習
            num_batches = len(train_data) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = train_data[start_idx:end_idx]
                
                # バッチ準備
                inputs, labels = RealGameDataGenerator.prepare_batch(batch_data, batch_size)
                
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
            
            # エポック統計
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            train_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(train_accuracy)
            
            # 検証フェーズ
            val_accuracy = 0
            if val_data:
                val_accuracy = self.evaluate(val_data, batch_size)
                self.history['val_accuracy'].append(val_accuracy)
            
            # 進捗表示
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.3f}", end="")
                if val_data:
                    print(f" | Val Acc: {val_accuracy:.3f}")
                else:
                    print()
        
        print("\n✅ 学習完了！")
        self.plot_history()
    
    def evaluate(self, data: List[Dict], batch_size: int = 32) -> float:
        """データセットで評価"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            num_batches = len(data) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                if not batch_data:
                    continue
                
                inputs, labels = RealGameDataGenerator.prepare_batch(batch_data, len(batch_data))
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0
    
    def plot_history(self):
        """学習履歴をプロット"""
        if not self.history['loss']:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失
        axes[0].plot(self.history['loss'], 'b-', label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)
        axes[0].legend()
        
        # 精度
        axes[1].plot(self.history['accuracy'], 'g-', label='Training Accuracy')
        if self.history['val_accuracy']:
            axes[1].plot(self.history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('real_data_training.png')
        print("📊 学習曲線を保存: real_data_training.png")


# ================================================================================
# Part 4: メイン実行
# ================================================================================

def main():
    """メイン実行関数"""
    print("🚀 シンプルCQCNN - 実ゲームデータ版")
    print("=" * 70)
    
    # 1. モデル作成
    print("\n[Step 1] モデル作成")
    print("-" * 40)
    model = SimpleCQCNN()
    print(f"✅ モデル作成完了")
    print(f"   パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. データ生成
    print("\n[Step 2] 実ゲームデータ生成")
    print("-" * 40)
    
    # ゲームプレイからデータ生成
    all_data = RealGameDataGenerator.generate_game_data(num_games=100)
    
    # 学習用と検証用に分割（8:2）
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"   学習データ: {len(train_data)}個")
    print(f"   検証データ: {len(val_data)}個")
    
    # 3. 学習実行
    print("\n[Step 3] 学習実行")
    print("-" * 40)
    trainer = ImprovedTrainer(model, learning_rate=0.001)
    trainer.train_with_real_data(
        train_data, 
        val_data,
        epochs=50,
        batch_size=32
    )
    
    # 4. 最終評価
    print("\n[Step 4] 最終評価")
    print("-" * 40)
    
    final_train_acc = trainer.evaluate(train_data, batch_size=32)
    final_val_acc = trainer.evaluate(val_data, batch_size=32)
    
    print(f"最終学習精度: {final_train_acc:.1%}")
    print(f"最終検証精度: {final_val_acc:.1%}")
    
    # 5. モデル保存
    print("\n[Step 5] モデル保存")
    print("-" * 40)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': trainer.history,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc
    }, 'real_data_model.pth')
    print("💾 モデルを保存: real_data_model.pth")
    
    # 6. サマリー
    print("\n" + "=" * 70)
    print("📊 実験サマリー")
    print("=" * 70)
    
    print(f"データ総数: {len(all_data)}個")
    print(f"最終学習精度: {final_train_acc:.1%}")
    print(f"最終検証精度: {final_val_acc:.1%}")
    
    # 成功判定
    if final_val_acc > 0.55:
        print("\n✅ 学習成功！")
        print("   モデルはランダム以上の性能を達成しました。")
        print("   次のステップ: 量子回路を追加してさらに性能向上")
    elif final_val_acc > 0.52:
        print("\n⚠️  部分的な学習成功")
        print("   わずかにランダムを上回っています。")
        print("   より多くのデータで学習することを推奨します。")
    else:
        print("\n❌ 学習が不十分")
        print("   モデルアーキテクチャやハイパーパラメータの見直しが必要です。")
    
    print("\n🎉 実験完了！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 実行中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()

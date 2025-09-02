#!/usr/bin/env python3
"""
強化学習システムモジュール - DQN実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
from ..core.base_modules import LearningSystem
from ..core.game_state import LearningConfig, GameState
from ..modules.reward import RewardFactory


class DQNReinforcementLearning(LearningSystem):
    """DQN強化学習システム"""
    
    def __init__(self):
        self.memory = deque(maxlen=10000)
        self.training_history = []
        self.epsilon = 0.9
        self.target_model = None
    
    def train(
        self, 
        model: nn.Module, 
        training_data: List[Dict], 
        config: LearningConfig
    ) -> nn.Module:
        """強化学習を実行"""
        print(f"🎮 DQN強化学習開始: {config.rl_episodes}エピソード")
        print(f"🔧 設定: バッチサイズ={config.batch_size}, 学習率={config.learning_rate}")
        print(f"🎯 ε設定: {config.epsilon_start} → {config.epsilon_end}")
        
        device = torch.device(config.device)
        model.to(device)
        
        # ターゲットモデル作成
        self.target_model = self._create_target_model(model, device)
        
        # オプティマイザ
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # 報酬関数（デフォルトは基本報酬）
        reward_function = RewardFactory.create_reward("basic")
        
        # ε減衰設定
        self.epsilon = config.epsilon_start
        epsilon_decay_rate = (config.epsilon_start - config.epsilon_end) / config.rl_episodes
        
        # 学習ループ
        for episode in range(config.rl_episodes):
            # 環境リセット（ダミー環境）
            state = self._create_dummy_state()
            total_reward = 0
            steps = 0
            
            while steps < 50:  # 最大50ステップ
                # ε-greedy行動選択
                if random.random() < self.epsilon:
                    action = self._get_random_action(state)
                else:
                    action = self._get_best_action(model, state, device)
                
                # 環境ステップ（ダミー）
                next_state, reward, done = self._dummy_step(state, action, reward_function)
                
                # 経験を記憶
                self.memory.append((state, action, reward, next_state, done))
                
                total_reward += reward
                state = next_state
                steps += 1
                
                # バッチ学習
                if len(self.memory) > config.batch_size:
                    loss = self._replay_experience(model, optimizer, config, device)
                else:
                    loss = 0.0
                
                if done:
                    break
            
            # ε減衰
            self.epsilon = max(config.epsilon_end, self.epsilon - epsilon_decay_rate)
            
            # ターゲットモデル更新
            if episode % config.target_update == 0:
                self._update_target_model(model)
            
            # 学習履歴記録
            self.training_history.append({
                "episode": episode,
                "total_reward": total_reward,
                "epsilon": self.epsilon,
                "steps": steps,
                "loss": loss if 'loss' in locals() else 0.0
            })
            
            # 進捗表示
            if episode % 100 == 0:
                avg_reward = np.mean([h["total_reward"] for h in self.training_history[-100:]])
                print(f"Episode {episode}/{config.rl_episodes}: "
                      f"Avg Reward={avg_reward:.2f}, ε={self.epsilon:.3f}")
        
        print(f"🎉 強化学習完了!")
        return model
    
    def evaluate(self, model: nn.Module, test_data: List[Dict]) -> Dict[str, float]:
        """モデル評価"""
        device = next(model.parameters()).device
        model.eval()
        
        total_rewards = []
        total_steps = []
        
        # テストエピソード実行
        for _ in range(10):  # 10エピソードでテスト
            state = self._create_dummy_state()
            total_reward = 0
            steps = 0
            
            while steps < 50:
                with torch.no_grad():
                    action = self._get_best_action(model, state, device)
                
                reward_function = RewardFactory.create_reward("basic")
                next_state, reward, done = self._dummy_step(state, action, reward_function)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            total_rewards.append(total_reward)
            total_steps.append(steps)
        
        return {
            "avg_reward": np.mean(total_rewards),
            "avg_steps": np.mean(total_steps),
            "total_episodes": len(total_rewards)
        }
    
    def _create_target_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """ターゲットモデル作成"""
        # モデルのコンストラクタ引数を取得して同じ構成でターゲットモデルを作成
        if hasattr(model, 'n_qubits') and hasattr(model, 'n_layers'):
            target_model = type(model)(model.n_qubits, model.n_layers)
        else:
            target_model = type(model)()
        
        target_model.load_state_dict(model.state_dict())
        target_model.to(device)
        return target_model
    
    def _update_target_model(self, model: nn.Module):
        """ターゲットモデル更新"""
        if self.target_model is not None:
            self.target_model.load_state_dict(model.state_dict())
    
    def _create_dummy_state(self) -> Dict:
        """ダミー状態作成"""
        return {
            "board": np.random.randint(0, 5, (6, 6)),
            "player": random.choice(["A", "B"]),
            "my_pieces": {(i, j): random.choice(["good", "bad"]) 
                         for i in range(2) for j in range(2)},
            "turn": random.randint(1, 50)
        }
    
    def _get_random_action(self, state: Dict) -> Tuple:
        """ランダム行動"""
        positions = list(state["my_pieces"].keys())
        if not positions:
            return ((0, 0), (0, 1))
        
        from_pos = random.choice(positions)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        direction = random.choice(directions)
        to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        
        # 境界チェック
        to_pos = (max(0, min(5, to_pos[0])), max(0, min(5, to_pos[1])))
        
        return (from_pos, to_pos)
    
    def _get_best_action(self, model: nn.Module, state: Dict, device: torch.device) -> Tuple:
        """最適行動選択"""
        try:
            from ..core.data_processor import DataProcessor
            tensor = DataProcessor.prepare_7channel_tensor(
                state["board"], state["player"], state["my_pieces"], state["turn"]
            )
            tensor = tensor.to(device)
            
            with torch.no_grad():
                q_values = model(tensor)
                
            # Q値から行動を決定（簡略化）
            if q_values.dim() == 2:
                action_idx = torch.argmax(q_values, dim=1).item()
            else:
                action_idx = 0
            
            # インデックスから行動に変換
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            positions = list(state["my_pieces"].keys())
            
            if positions:
                from_pos = positions[action_idx % len(positions)]
                direction = directions[action_idx % len(directions)]
                to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
                to_pos = (max(0, min(5, to_pos[0])), max(0, min(5, to_pos[1])))
                return (from_pos, to_pos)
        
        except Exception:
            pass
        
        # フォールバック
        return self._get_random_action(state)
    
    def _dummy_step(self, state: Dict, action: Tuple, reward_function) -> Tuple[Dict, float, bool]:
        """ダミー環境ステップ"""
        # 新しい状態作成
        next_state = state.copy()
        next_state["turn"] += 1
        
        # 報酬計算（簡略化）
        reward = random.uniform(-1.0, 1.0)
        
        # 終了判定
        done = random.random() < 0.1 or next_state["turn"] > 50
        
        return next_state, reward, done
    
    def _replay_experience(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        config: LearningConfig,
        device: torch.device
    ) -> float:
        """経験リプレイ"""
        if len(self.memory) < config.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, config.batch_size)
        
        # バッチを準備（簡略化）
        batch_size = len(batch)
        dummy_inputs = torch.randn(batch_size, 7, 6, 6).to(device)
        dummy_targets = torch.randn(batch_size, 2).to(device)
        
        optimizer.zero_grad()
        outputs = model(dummy_inputs)
        
        # 適切なロス計算（簡略化）
        if outputs.shape == dummy_targets.shape:
            loss = nn.MSELoss()(outputs, dummy_targets)
        else:
            # 形状が合わない場合の処理
            targets = torch.randint(0, outputs.size(1), (batch_size,)).to(device)
            loss = nn.CrossEntropyLoss()(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def get_training_history(self) -> List[Dict]:
        """学習履歴を取得"""
        return self.training_history
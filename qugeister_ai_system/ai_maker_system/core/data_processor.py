#!/usr/bin/env python3
"""
データ処理モジュール - 7チャンネルテンソル生成
"""

import numpy as np
import torch
from typing import Dict, Tuple


class DataProcessor:
    """7チャンネルデータ処理クラス"""
    
    @staticmethod
    def prepare_7channel_tensor(
        board: np.ndarray, 
        player: str, 
        my_pieces: Dict[Tuple[int, int], str], 
        turn: int
    ) -> torch.Tensor:
        """
        7チャンネルテンソルを準備
        
        Args:
            board: ゲームボード (6x6)
            player: プレイヤー ('A' or 'B')
            my_pieces: 自分の駒の位置と種類
            turn: ターン数
            
        Returns:
            torch.Tensor: (1, 7, 6, 6) のテンソル
        """
        channels = []
        
        # チャンネル1: 自分の駒位置 (1: 存在, 0: なし)
        my_pos = np.zeros_like(board)
        for (r, c), _ in my_pieces.items():
            if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                my_pos[r, c] = 1
        channels.append(my_pos)
        
        # チャンネル2: 自分の善玉 (1: 善玉, 0: その他)
        my_good = np.zeros_like(board)
        for (r, c), piece_type in my_pieces.items():
            if piece_type == "good" and 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                my_good[r, c] = 1
        channels.append(my_good)
        
        # チャンネル3: 自分の悪玉 (1: 悪玉, 0: その他)
        my_bad = np.zeros_like(board)
        for (r, c), piece_type in my_pieces.items():
            if piece_type == "bad" and 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                my_bad[r, c] = 1
        channels.append(my_bad)
        
        # チャンネル4: 敵の駒位置 (1: 存在, 0: なし)
        enemy_pos = np.zeros_like(board)
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] != 0 and (r, c) not in my_pieces:
                    enemy_pos[r, c] = 1
        channels.append(enemy_pos)
        
        # チャンネル5: プレイヤー情報 (1: プレイヤーA, 0: プレイヤーB)
        player_channel = np.ones_like(board) if player == "A" else np.zeros_like(board)
        channels.append(player_channel)
        
        # チャンネル6: ターン情報 (正規化されたターン数)
        turn_channel = np.full_like(board, turn / 100.0)
        channels.append(turn_channel)
        
        # チャンネル7: ボード境界情報 (端=1, 中央=0)
        boundary = np.zeros_like(board)
        boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = 1
        channels.append(boundary)
        
        # テンソルに変換 (1, 7, H, W)
        tensor = torch.FloatTensor(np.stack(channels)).unsqueeze(0)
        return tensor
    
    @staticmethod
    def extract_features(tensor: torch.Tensor) -> Dict[str, float]:
        """
        テンソルから特徴量を抽出
        
        Args:
            tensor: 7チャンネルテンソル
            
        Returns:
            Dict: 特徴量辞書
        """
        features = {}
        
        # バッチ次元を除去
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        channels = tensor.numpy()
        
        # 各チャンネルの統計量
        channel_names = [
            'my_pieces', 'my_good', 'my_bad', 
            'enemy_pieces', 'player_info', 'turn_info', 'boundary'
        ]
        
        for i, name in enumerate(channel_names):
            if i < len(channels):
                channel = channels[i]
                features[f'{name}_mean'] = float(np.mean(channel))
                features[f'{name}_sum'] = float(np.sum(channel))
                features[f'{name}_std'] = float(np.std(channel))
        
        return features
    
    @staticmethod
    def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        テンソルを正規化
        
        Args:
            tensor: 入力テンソル
            
        Returns:
            torch.Tensor: 正規化されたテンソル
        """
        # チャンネル毎に正規化
        normalized_channels = []
        
        for i in range(tensor.size(1)):  # チャンネル次元
            channel = tensor[:, i:i+1]
            channel_mean = channel.mean()
            channel_std = channel.std()
            
            if channel_std > 0:
                normalized_channel = (channel - channel_mean) / channel_std
            else:
                normalized_channel = channel
            
            normalized_channels.append(normalized_channel)
        
        return torch.cat(normalized_channels, dim=1)
    
    @staticmethod  
    def augment_data(tensor: torch.Tensor, augmentation_type: str = 'rotation') -> torch.Tensor:
        """
        データ拡張
        
        Args:
            tensor: 入力テンソル (1, 7, 6, 6)
            augmentation_type: 拡張タイプ ('rotation', 'flip')
            
        Returns:
            torch.Tensor: 拡張されたテンソル
        """
        if augmentation_type == 'rotation':
            # 90度回転
            return torch.rot90(tensor, k=1, dims=[2, 3])
        elif augmentation_type == 'flip':
            # 水平反転
            return torch.flip(tensor, dims=[3])
        else:
            return tensor
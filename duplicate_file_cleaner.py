#!/usr/bin/env python3
"""
重複ファイル整理ツール
類似機能を持つファイルを特定し、安全に削除
"""

import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple
import hashlib

class DuplicateFileCleaner:
    """重複ファイルの整理クラス"""
    
    def __init__(self):
        self.backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.duplicate_groups = {}
        
    def analyze_duplicates(self):
        """重複ファイルを分析"""
        print("🔍 重複ファイル分析開始")
        print("=" * 50)
        
        # 明らかに重複と思われるファイルグループ
        duplicate_groups = {
            'cqcnn_battle_learning_system_group': [
                'cqcnn_battle_learning_system.py',  # メインファイル（保持）
                'cqcnn_real_learning_battle_system.py',  # 削除対象
                'real_learning_battle_system.py',  # 削除対象
            ],
            'cqcnn_battle_system_group': [
                'cqcnn_battle_learning_system.py',  # メインファイル（保持）
                'cqcnn-battle-system.py',  # 削除対象
            ],
            'gui_test_group': [
                'gui_designed_ai_test_v2.py',  # 最新版（保持）
                'gui_designed_ai_test.py',  # 旧版（削除対象）
            ],
            'history_files': [
                # .historyディレクトリ内のファイルは基本的に削除対象
            ]
        }
        
        # 実際に存在するファイルのみをフィルター
        existing_groups = {}
        for group_name, files in duplicate_groups.items():
            existing_files = [f for f in files if os.path.exists(f)]
            if len(existing_files) > 1:  # 複数ファイルが存在する場合のみ
                existing_groups[group_name] = existing_files
        
        # .historyディレクトリの処理
        history_dir = '.history'
        if os.path.exists(history_dir):
            history_files = [f for f in os.listdir(history_dir) if f.endswith('.py')]
            if history_files:
                existing_groups['history_files'] = [os.path.join(history_dir, f) for f in history_files]
        
        self.duplicate_groups = existing_groups
        
        print(f"発見された重複グループ数: {len(existing_groups)}")
        for group_name, files in existing_groups.items():
            print(f"\n📁 {group_name}:")
            for file in files:
                size = os.path.getsize(file) if os.path.exists(file) else 0
                print(f"  - {file} ({size:,} bytes)")
        
        return existing_groups
    
    def analyze_file_similarity(self, file1: str, file2: str) -> float:
        """ファイル類似度を分析（簡易版）"""
        try:
            with open(file1, 'r', encoding='utf-8') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8') as f2:
                content2 = f2.read()
            
            # 簡易的な類似度計算（行数、文字数比較）
            lines1 = content1.split('\n')
            lines2 = content2.split('\n')
            
            common_lines = set(lines1) & set(lines2)
            total_lines = max(len(lines1), len(lines2))
            
            similarity = len(common_lines) / total_lines if total_lines > 0 else 0
            return similarity
            
        except Exception as e:
            print(f"⚠️ ファイル比較エラー ({file1} vs {file2}): {e}")
            return 0.0
    
    def create_backup(self, files_to_delete: List[str]):
        """削除前にバックアップを作成"""
        if not files_to_delete:
            return
        
        print(f"\n💾 バックアップ作成: {self.backup_dir}")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    # ディレクトリ構造を保持してバックアップ
                    backup_path = os.path.join(self.backup_dir, file_path)
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(file_path, backup_path)
                    print(f"  ✅ {file_path} → {backup_path}")
                except Exception as e:
                    print(f"  ❌ {file_path} のバックアップに失敗: {e}")
    
    def recommend_deletions(self) -> Dict[str, List[str]]:
        """削除推奨ファイルを提案"""
        print(f"\n🗑️ 削除推奨ファイル:")
        print("=" * 50)
        
        deletion_plan = {}
        
        # cqcnn_battle_learning_system関連
        if 'cqcnn_battle_learning_system_group' in self.duplicate_groups:
            files = self.duplicate_groups['cqcnn_battle_learning_system_group']
            keep_file = 'cqcnn_battle_learning_system.py'
            to_delete = [f for f in files if f != keep_file and os.path.exists(f)]
            if to_delete:
                deletion_plan['cqcnn_battle_system_duplicates'] = to_delete
                print(f"\n📋 CQCNN Battle System 重複:")
                print(f"  🟢 保持: {keep_file}")
                for f in to_delete:
                    print(f"  🔴 削除: {f}")
        
        # cqcnn-battle-system.py（ハイフン名）
        if 'cqcnn_battle_system_group' in self.duplicate_groups:
            files = self.duplicate_groups['cqcnn_battle_system_group']
            keep_file = 'cqcnn_battle_learning_system.py'
            to_delete = [f for f in files if f != keep_file and 'cqcnn-battle-system.py' in f]
            if to_delete:
                deletion_plan['hyphen_named_duplicates'] = to_delete
                print(f"\n📋 ハイフン命名重複:")
                print(f"  🟢 保持: {keep_file}")
                for f in to_delete:
                    print(f"  🔴 削除: {f}")
        
        # GUI テスト関連
        if 'gui_test_group' in self.duplicate_groups:
            files = self.duplicate_groups['gui_test_group']
            keep_file = 'gui_designed_ai_test_v2.py'
            to_delete = [f for f in files if f != keep_file and os.path.exists(f)]
            if to_delete:
                deletion_plan['gui_test_duplicates'] = to_delete
                print(f"\n📋 GUI テスト重複:")
                print(f"  🟢 保持: {keep_file}")
                for f in to_delete:
                    print(f"  🔴 削除: {f}")
        
        # .history ディレクトリ
        if 'history_files' in self.duplicate_groups:
            to_delete = self.duplicate_groups['history_files']
            deletion_plan['history_files'] = to_delete
            print(f"\n📋 履歴ファイル:")
            print(f"  🔴 削除対象: {len(to_delete)} 個のファイル")
            for f in to_delete[:5]:  # 最初の5個のみ表示
                print(f"    - {f}")
            if len(to_delete) > 5:
                print(f"    ... 他 {len(to_delete) - 5} 個")
        
        # その他の明らかな重複
        other_duplicates = []
        all_files = [f for f in os.listdir('.') if f.endswith('.py')]
        
        # test_ で始まる古いファイル（新しいバージョンがある場合）
        for file in all_files:
            if file.startswith('test_') and file != 'test_fixed_placement.py':
                # test_fixed_placement.py は重要なので保持
                if not any(file in group for group in deletion_plan.values()):
                    other_duplicates.append(file)
        
        if other_duplicates:
            deletion_plan['other_test_files'] = other_duplicates
            print(f"\n📋 その他のテストファイル:")
            for f in other_duplicates:
                print(f"  🟡 確認要: {f}")
        
        return deletion_plan
    
    def execute_deletion(self, deletion_plan: Dict[str, List[str]], confirm: bool = True):
        """削除を実行"""
        all_files_to_delete = []
        for category, files in deletion_plan.items():
            all_files_to_delete.extend(files)
        
        if not all_files_to_delete:
            print("削除対象ファイルがありません。")
            return
        
        print(f"\n削除予定ファイル総数: {len(all_files_to_delete)}")
        
        # バックアップ作成
        self.create_backup(all_files_to_delete)
        
        # 削除実行
        print(f"\n🗑️ ファイル削除実行:")
        deleted_count = 0
        for file_path in all_files_to_delete:
            if os.path.exists(file_path):
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    print(f"  ✅ 削除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ❌ 削除失敗: {file_path} - {e}")
            else:
                print(f"  ⚠️ ファイル見つからず: {file_path}")
        
        print(f"\n🎉 削除完了: {deleted_count} 個のファイル/ディレクトリを削除")
        print(f"💾 バックアップ場所: {self.backup_dir}")
        
        # 削除後の確認
        self.verify_deletion(deletion_plan)
    
    def verify_deletion(self, deletion_plan: Dict[str, List[str]]):
        """削除後の確認"""
        print(f"\n✅ 削除確認:")
        
        # 重要ファイルが残っているか確認
        important_files = [
            'cqcnn_battle_learning_system.py',
            'quantum_battle_3step_system.html',
            'rl_cqcnn_runner.py',
            'rl_cqcnn_system.py',
            'gui_designed_ai_test_v2.py'
        ]
        
        print("📋 重要ファイルの確認:")
        for file in important_files:
            if os.path.exists(file):
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} - 重要ファイルが見つかりません！")
        
        # Python ファイル数の確認
        py_files_after = len([f for f in os.listdir('.') if f.endswith('.py')])
        print(f"\n📊 削除後のPythonファイル数: {py_files_after}")

def main():
    """メイン実行"""
    print("🧹 重複ファイル整理ツール")
    print("Qugeister_competitive プロジェクトのクリーンアップ")
    print("=" * 60)
    
    cleaner = DuplicateFileCleaner()
    
    # 1. 重複分析
    duplicate_groups = cleaner.analyze_duplicates()
    
    if not duplicate_groups:
        print("重複ファイルは見つかりませんでした。")
        return
    
    # 2. 削除推奨
    deletion_plan = cleaner.recommend_deletions()
    
    if not deletion_plan:
        print("削除推奨ファイルはありません。")
        return
    
    # 3. 確認
    total_files = sum(len(files) for files in deletion_plan.values())
    print(f"\n⚠️ 確認:")
    print(f"削除対象: {total_files} 個のファイル/ディレクトリ")
    print("バックアップを作成してから削除します。")
    
    # 4. 削除実行
    cleaner.execute_deletion(deletion_plan, confirm=False)
    
    print(f"\n🎯 クリーンアップ完了!")
    print("プロジェクトの整理が完了しました。")

if __name__ == "__main__":
    main()
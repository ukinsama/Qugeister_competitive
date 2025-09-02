#!/usr/bin/env python3
"""
段階3: トーナメント実行システム
Tournament Runner - Legacy Tournament System
"""

import sys
import os
from pathlib import Path

# メインのトーナメントシステムに転送
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """メイン実行 - qugeister_ai_systemの方に転送"""
    print("🏆 Tournament Runner - Legacy System")
    print("=" * 50)
    print("🔄 メインのトーナメントシステムに転送します...")
    print("   実行: qugeister_ai_system/tournament_system/tournament_manager.py")
    print()
    
    # メインのトーナメントシステム実行
    tournament_path = Path(__file__).parent.parent / "qugeister_ai_system/tournament_system/tournament_manager.py"
    
    if tournament_path.exists():
        os.system(f"python {tournament_path}")
    else:
        print("❌ メインのトーナメントシステムが見つかりません")
        print("💡 代替方法:")
        print("   python run_minimal_tournament.py")

if __name__ == "__main__":
    main()
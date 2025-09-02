#!/usr/bin/env python3
"""
Qugeister Competition Environment Check
環境が正しくセットアップされているかを確認
"""

import sys
from pathlib import Path

def check_python_version():
    """Python バージョン確認"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_dependencies():
    """依存関係確認"""
    deps = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib (optional)'
    }
    
    results = {}
    for module, name in deps.items():
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            results[module] = True
        except ImportError:
            print(f"❌ {name}: Not installed")
            results[module] = False
    
    return results

def check_file_structure():
    """ファイル構造確認"""
    required_paths = [
        'ai_maker_system/',
        'trained_models/',
        'src/qugeister_competitive/',
        'qugeister_ai_system/tournament_system/tournament_manager.py',
    ]
    
    results = {}
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ {path_str}")
            results[path_str] = True
        else:
            print(f"❌ {path_str} not found")
            results[path_str] = False
    
    return results

def check_sample_execution():
    """サンプル実行テスト"""
    try:
        import torch
        import numpy as np
        
        # 簡単な計算テスト
        x = torch.randn(2, 3)
        y = np.array([1, 2, 3])
        
        print("✅ Basic computation test passed")
        return True
    except Exception as e:
        print(f"❌ Basic computation test failed: {e}")
        return False

def main():
    """メイン環境チェック"""
    print("🏆 Qugeister Competition Environment Check")
    print("=" * 50)
    
    # 各チェック実行
    python_ok = check_python_version()
    print()
    
    print("📦 Dependencies Check:")
    deps_results = check_dependencies()
    print()
    
    print("📁 File Structure Check:")
    files_results = check_file_structure()
    print()
    
    print("⚡ Functionality Check:")
    compute_ok = check_sample_execution()
    print()
    
    # 総合判定
    essential_deps = all([deps_results.get('torch', False), 
                         deps_results.get('numpy', False)])
    essential_files = files_results.get('qugeister_ai_system/tournament_system/tournament_manager.py', False)
    
    overall_ok = python_ok and essential_deps and essential_files and compute_ok
    
    print("=" * 50)
    if overall_ok:
        print("🎉 環境チェック完了! Qugeister Competition ready!")
        print("\n🚀 次のステップ:")
        print("   python run_minimal_tournament.py")
        print("   python qugeister_ai_system/examples/integration_example.py")
    else:
        print("⚠️  環境に問題があります。SETUP_GUIDE.mdを参照してください")
        print("\n🔧 修正が必要:")
        if not python_ok:
            print("   - Python 3.8以上をインストール")
        if not essential_deps:
            print("   - pip install torch numpy")
        if not essential_files:
            print("   - 正しいディレクトリにいるか確認")
    
    return overall_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
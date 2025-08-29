/**
 * Quantum Battle System - JavaScript
 * 分離されたJSファイル - 改善しやすい構造
 */

// =================
// グローバル変数
// =================
let currentStep = 1;
let selectedMethod = null;
let placedPieces = { good: 0, bad: 0 };
let currentPieceType = 'good';
let boardState = {};
let moduleConfigs = {
    placement: {},
    estimation: {},
    reward: {},
    qmap: {},
    action: {}
};

// =================
// 初期化
// =================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Quantum Battle System 初期化開始');
    
    initializeStepNavigation();
    initializePlacementBoard();
    initializeFormElements();
    initializeEventListeners();
    loadSavedConfiguration();
    
    console.log('✅ 初期化完了');
});

// =================
// ステップナビゲーション
// =================
function initializeStepNavigation() {
    const steps = document.querySelectorAll('.step-item');
    steps.forEach((step, index) => {
        step.addEventListener('click', () => goToStep(index + 1));
    });
    
    // 最初のステップをアクティブに
    showStep(currentStep);
}

function goToStep(stepNumber) {
    if (stepNumber < 1 || stepNumber > 5) return;
    
    // 現在のステップを保存
    saveCurrentStepData();
    
    // ステップを切り替え
    currentStep = stepNumber;
    showStep(stepNumber);
    
    console.log(`📍 ステップ ${stepNumber} に移動`);
}

function showStep(stepNumber) {
    // 全てのコンテンツを非表示
    document.querySelectorAll('.step-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // 選択されたコンテンツを表示
    const targetContent = document.getElementById(`step${stepNumber}`);
    if (targetContent) {
        targetContent.classList.add('active');
    }
    
    // ナビゲーションの状態更新
    updateStepNavigation(stepNumber);
}

function updateStepNavigation(activeStep) {
    const steps = document.querySelectorAll('.step-item');
    const connectors = document.querySelectorAll('.step-connector');
    
    steps.forEach((step, index) => {
        const stepNum = index + 1;
        step.classList.remove('active', 'completed');
        
        if (stepNum === activeStep) {
            step.classList.add('active');
        } else if (stepNum < activeStep) {
            step.classList.add('completed');
        }
    });
    
    connectors.forEach((connector, index) => {
        connector.classList.toggle('completed', index + 1 < activeStep);
    });
}

function nextStep() {
    if (currentStep < 5) {
        goToStep(currentStep + 1);
    }
}

function prevStep() {
    if (currentStep > 1) {
        goToStep(currentStep - 1);
    }
}

// =================
// 学習方法選択 (ステップ1)
// =================
function selectLearningMethod(method) {
    // 以前の選択をクリア
    document.querySelectorAll('.method-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // 新しい選択をマーク
    const selectedCard = document.querySelector(`[onclick="selectLearningMethod('${method}')"]`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
        selectedMethod = method;
        
        console.log(`🎯 学習方法選択: ${method}`);
        
        // 次のステップボタンを有効化
        enableNextStepButton();
    }
}

function enableNextStepButton() {
    const nextButton = document.getElementById('nextStepBtn');
    if (nextButton) {
        nextButton.disabled = false;
        nextButton.classList.remove('btn-disabled');
    }
}

// =================
// 配置ボード (ステップ2)
// =================
function initializePlacementBoard() {
    const board = document.getElementById('placementBoard');
    if (!board) return;
    
    board.innerHTML = '';
    
    for (let row = 0; row < 6; row++) {
        for (let col = 0; col < 6; col++) {
            const cell = document.createElement('div');
            cell.className = 'board-cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            
            // プレイヤーA配置エリア（下側2行、中央4列）
            if ((row === 0 || row === 1) && (col >= 1 && col <= 4)) {
                cell.classList.add('player-a-area');
                cell.title = 'プレイヤーA配置エリア';
                cell.addEventListener('click', () => placePiece(row, col, cell));
            }
            // プレイヤーB配置エリア（上側2行、中央4列）
            else if ((row === 4 || row === 5) && (col >= 1 && col <= 4)) {
                cell.classList.add('player-b-area');
                cell.title = 'プレイヤーB配置エリア';
            }
            
            board.appendChild(cell);
        }
    }
}

function placePiece(row, col, cell) {
    const maxPieces = 4;
    
    if (placedPieces.good >= maxPieces && currentPieceType === 'good') {
        showAlert('善玉は最大4個まで配置可能です', 'warning');
        return;
    }
    
    if (placedPieces.bad >= maxPieces && currentPieceType === 'bad') {
        showAlert('悪玉は最大4個まで配置可能です', 'warning');
        return;
    }
    
    const key = `${row}-${col}`;
    
    // 既存の駒を削除
    if (boardState[key]) {
        placedPieces[boardState[key]]--;
        delete boardState[key];
        cell.classList.remove('placed-good', 'placed-bad');
        cell.textContent = '';
    }
    
    // 新しい駒を配置
    if (currentPieceType === 'good') {
        boardState[key] = 'good';
        placedPieces.good++;
        cell.classList.add('placed-good');
        cell.textContent = 'G';
        currentPieceType = 'bad';
    } else {
        boardState[key] = 'bad';
        placedPieces.bad++;
        cell.classList.add('placed-bad');
        cell.textContent = 'B';
        currentPieceType = 'good';
    }
    
    updatePieceCount();
    checkPlacementCompletion();
}

function updatePieceCount() {
    const goodCount = document.getElementById('goodCount');
    const badCount = document.getElementById('badCount');
    
    if (goodCount) goodCount.textContent = `${placedPieces.good}/4`;
    if (badCount) badCount.textContent = `${placedPieces.bad}/4`;
}

function checkPlacementCompletion() {
    if (placedPieces.good === 4 && placedPieces.bad === 4) {
        showAlert('配置完了！次のステップに進めます', 'success');
        enableNextStepButton();
    }
}

function resetPlacement() {
    placedPieces = { good: 0, bad: 0 };
    boardState = {};
    currentPieceType = 'good';
    
    document.querySelectorAll('.board-cell').forEach(cell => {
        cell.classList.remove('placed-good', 'placed-bad');
        cell.textContent = '';
    });
    
    updatePieceCount();
    console.log('🔄 配置をリセット');
}

// =================
// フォーム要素
// =================
function initializeFormElements() {
    // レンジスライダーの値表示
    const ranges = document.querySelectorAll('input[type="range"]');
    ranges.forEach(range => {
        const valueDisplay = document.getElementById(range.id + 'Value');
        if (valueDisplay) {
            // 初期値設定
            valueDisplay.textContent = range.value;
            
            // 変更イベント
            range.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
                onParameterChange(this.id, this.value);
            });
        }
    });
    
    // セレクトボックスの変更
    const selects = document.querySelectorAll('select');
    selects.forEach(select => {
        select.addEventListener('change', function() {
            onParameterChange(this.id, this.value);
        });
    });
    
    // テキスト入力の変更
    const inputs = document.querySelectorAll('input[type="text"], input[type="number"], textarea');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            onParameterChange(this.id, this.value);
        });
    });
}

function onParameterChange(parameterId, value) {
    // パラメータ変更のログ
    console.log(`🔧 パラメータ変更: ${parameterId} = ${value}`);
    
    // リアルタイム計算・プレビュー更新
    updatePreview();
    
    // 自動保存
    autosave();
}

// =================
// イベントリスナー
// =================
function initializeEventListeners() {
    // キーボードショートカット
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 's':
                    e.preventDefault();
                    saveConfiguration();
                    break;
                case 'l':
                    e.preventDefault();
                    loadConfiguration();
                    break;
                case 'g':
                    e.preventDefault();
                    generateAICode();
                    break;
            }
        }
        
        // ステップ移動（Ctrl + 数字）
        if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '5') {
            e.preventDefault();
            goToStep(parseInt(e.key));
        }
    });
    
    // ウィンドウリサイズ
    window.addEventListener('resize', debounce(handleResize, 300));
    
    // フォーム送信防止
    document.addEventListener('submit', function(e) {
        e.preventDefault();
    });
}

function handleResize() {
    // レスポンシブ調整
    const isMobile = window.innerWidth < 768;
    document.body.classList.toggle('mobile-view', isMobile);
    
    console.log(`📱 画面リサイズ: ${window.innerWidth}px (モバイル: ${isMobile})`);
}

// =================
// AI コード生成
// =================
function generateAICode() {
    console.log('🤖 AIコード生成開始');
    
    // 設定収集
    const config = collectAllConfigurations();
    
    // コード生成
    const aiCode = buildAICode(config);
    
    // 表示
    displayGeneratedCode(aiCode);
    
    console.log('✅ AIコード生成完了');
}

function collectAllConfigurations() {
    return {
        learningMethod: selectedMethod,
        placement: {
            strategy: getElementValue('placementStrategy'),
            boardState: boardState,
            placedPieces: placedPieces
        },
        quantum: {
            qubits: parseInt(getElementValue('qubits')),
            layers: parseInt(getElementValue('layers')),
            learningRate: parseFloat(getElementValue('learningRate'))
        },
        rewards: {
            escape: parseInt(getElementValue('escapeReward')),
            capture: parseInt(getElementValue('captureReward')),
            advance: parseInt(getElementValue('advanceReward'))
        },
        action: {
            epsilonStart: parseFloat(getElementValue('epsilonStart')),
            epsilonEnd: parseFloat(getElementValue('epsilonEnd')),
            gamma: parseFloat(getElementValue('gamma'))
        }
    };
}

function buildAICode(config) {
    const timestamp = new Date().toISOString();
    
    return `#!/usr/bin/env python3
"""
Generated Quantum Geister AI
Created: ${timestamp}
Learning Method: ${config.learningMethod}
Auto-generated by Quantum Battle System Designer
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Dict, List, Tuple, Optional
import random

class QuantumGeisterAI:
    """量子強化学習ガイスターAI"""
    
    def __init__(self):
        # 量子デバイス初期化
        self.n_qubits = ${config.quantum.qubits}
        self.n_layers = ${config.quantum.layers}
        self.device = qml.device('default.qubit', wires=self.n_qubits)
        
        # 設定パラメータ
        self.config = {
            'learning_method': '${config.learningMethod}',
            'placement_strategy': '${config.placement.strategy}',
            'quantum_params': {
                'qubits': ${config.quantum.qubits},
                'layers': ${config.quantum.layers},
                'learning_rate': ${config.quantum.learningRate}
            },
            'reward_params': {
                'escape_reward': ${config.rewards.escape},
                'capture_reward': ${config.rewards.capture},
                'advance_reward': ${config.rewards.advance}
            },
            'rl_params': {
                'epsilon_start': ${config.action.epsilonStart},
                'epsilon_end': ${config.action.epsilonEnd},
                'gamma': ${config.action.gamma}
            }
        }
        
        # 強化学習パラメータ
        self.epsilon = self.config['rl_params']['epsilon_start']
        self.epsilon_decay = 0.995
        
        # 量子回路の重み初期化
        self.weights = np.random.random((self.n_layers, self.n_qubits, 2)) * 0.1
        
        print(f"🧠 QuantumGeisterAI 初期化完了")
        print(f"   量子ビット: {self.n_qubits}")
        print(f"   回路層数: {self.n_layers}")
        print(f"   学習方法: {config.learningMethod}")
    
    @property 
    def quantum_circuit(self):
        """量子回路定義"""
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            # エンコーディング層
            for i in range(self.n_qubits):
                qml.RY(inputs[i % len(inputs)], wires=i)
            
            # 変分層
            for layer in range(self.n_layers):
                # パラメータ化回転ゲート
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # エンタングルメント
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # 測定
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def get_initial_placement(self) -> Dict[Tuple[int, int], str]:
        """初期配置戦略"""
        placement = {}
        strategy = self.config['placement_strategy']
        
        # プレイヤーA配置可能位置
        positions = [(0,1), (0,2), (0,3), (0,4), (1,1), (1,2), (1,3), (1,4)]
        
        if strategy == 'aggressive':
            # 攻撃的: 善玉を前方に
            good_pos = positions[:4]
            bad_pos = positions[4:]
        elif strategy == 'defensive':
            # 守備的: 善玉を後方に
            good_pos = positions[4:]
            bad_pos = positions[:4]
        else:
            # バランス: 交互配置
            good_pos = [positions[0], positions[2], positions[5], positions[7]]
            bad_pos = [positions[1], positions[3], positions[4], positions[6]]
        
        # 配置設定
        for pos in good_pos:
            placement[pos] = 'good'
        for pos in bad_pos:
            placement[pos] = 'bad'
        
        return placement
    
    def estimate_enemy_pieces(self, board_state: np.ndarray) -> Dict:
        """敵駒推定（量子機械学習）"""
        # ボード状態を量子回路入力に変換
        board_flat = board_state.flatten()
        inputs = np.pad(board_flat, (0, max(0, self.n_qubits - len(board_flat))))[:self.n_qubits]
        
        # 量子回路で推論
        quantum_output = self.quantum_circuit(inputs, self.weights)
        
        # 確率分布計算
        probabilities = torch.softmax(torch.tensor(quantum_output), dim=0).numpy()
        
        return {
            'good_probability': float(probabilities[0]),
            'bad_probability': float(probabilities[1]),
            'confidence': float(np.max(probabilities)),
            'quantum_features': quantum_output
        }
    
    def calculate_reward(self, game_state: Dict, action: Tuple, result: Dict) -> float:
        """報酬計算"""
        reward = 0.0
        
        # 勝利条件報酬
        if result.get('is_escape'):
            reward += self.config['reward_params']['escape_reward']
        if result.get('captured_all_good'):
            reward += self.config['reward_params']['capture_reward']
        
        # 戦術行動報酬
        if result.get('piece_advanced'):
            reward += self.config['reward_params']['advance_reward']
        if result.get('enemy_piece_captured'):
            reward += 20  # 基本捕獲報酬
        
        # ペナルティ
        if result.get('good_piece_lost'):
            reward -= 50
        if result.get('dangerous_move'):
            reward -= 10
        
        return reward
    
    def select_action(self, game_state: Dict, legal_moves: List[Tuple]) -> Tuple:
        """行動選択（ε-greedy + 量子推論）"""
        if len(legal_moves) == 0:
            return None
        
        # ε-greedy戦略
        if random.random() < self.epsilon:
            # 探索: ランダム行動
            action = random.choice(legal_moves)
            print(f"🎲 探索行動: {action}")
        else:
            # 活用: 量子回路による最適行動選択
            action = self._select_best_action(game_state, legal_moves)
            print(f"🧠 活用行動: {action}")
        
        # 探索率減衰
        self.epsilon = max(
            self.config['rl_params']['epsilon_end'],
            self.epsilon * self.epsilon_decay
        )
        
        return action
    
    def _select_best_action(self, game_state: Dict, legal_moves: List[Tuple]) -> Tuple:
        """量子回路による最適行動選択"""
        best_action = None
        best_score = float('-inf')
        
        board_state = game_state.get('board', np.zeros((6, 6)))
        
        for move in legal_moves:
            # 行動評価
            score = self._evaluate_action(board_state, move)
            
            if score > best_score:
                best_score = score
                best_action = move
        
        return best_action or legal_moves[0]
    
    def _evaluate_action(self, board_state: np.ndarray, action: Tuple) -> float:
        """行動評価"""
        from_pos, to_pos = action
        
        # 基本スコア
        score = 0.0
        
        # 前進評価
        if to_pos[0] > from_pos[0]:  # プレイヤーAの場合、Y座標が増加
            score += 5.0
        
        # 脱出評価
        if to_pos in [(0, 5), (5, 5)]:  # プレイヤーAの脱出口
            score += 50.0
        
        # 安全性評価
        if 1 <= to_pos[0] <= 4 and 1 <= to_pos[1] <= 4:  # 中央は危険
            score -= 5.0
        
        # 量子回路による詳細評価
        try:
            board_input = board_state.flatten()
            inputs = np.pad(board_input, (0, max(0, self.n_qubits - len(board_input))))[:self.n_qubits]
            quantum_score = np.sum(self.quantum_circuit(inputs, self.weights))
            score += quantum_score * 0.1
        except Exception as e:
            print(f"⚠️ 量子評価エラー: {e}")
        
        return score
    
    def update_weights(self, experience: Dict):
        """重み更新（学習）"""
        # 簡易的な重み更新
        learning_rate = self.config['quantum_params']['learning_rate']
        
        # 報酬に基づく重み調整
        reward = experience.get('reward', 0)
        if reward > 0:
            self.weights += learning_rate * np.random.random(self.weights.shape) * 0.01
        else:
            self.weights -= learning_rate * np.random.random(self.weights.shape) * 0.01
        
        # 重みクリッピング
        self.weights = np.clip(self.weights, -np.pi, np.pi)
    
    def save_model(self, filepath: str):
        """モデル保存"""
        model_data = {
            'weights': self.weights.tolist(),
            'config': self.config,
            'epsilon': self.epsilon
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 モデル保存完了: {filepath}")
    
    def load_model(self, filepath: str):
        """モデル読み込み"""
        import json
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.weights = np.array(model_data['weights'])
        self.config.update(model_data['config'])
        self.epsilon = model_data['epsilon']
        
        print(f"📂 モデル読み込み完了: {filepath}")

# 使用例
if __name__ == "__main__":
    # AI初期化
    ai = QuantumGeisterAI()
    
    # 初期配置テスト
    placement = ai.get_initial_placement()
    print(f"初期配置: {placement}")
    
    # テストボード状態
    test_board = np.random.randint(-1, 2, (6, 6))
    
    # 敵駒推定テスト
    estimation = ai.estimate_enemy_pieces(test_board)
    print(f"敵駒推定: {estimation}")
    
    # 行動選択テスト
    legal_moves = [((0, 1), (0, 2)), ((1, 1), (1, 2))]
    game_state = {'board': test_board}
    action = ai.select_action(game_state, legal_moves)
    print(f"選択行動: {action}")
    
    print("🎉 量子ガイスターAI テスト完了！")
`;
}

function displayGeneratedCode(code) {
    const codeOutput = document.getElementById('codeOutput');
    if (codeOutput) {
        codeOutput.textContent = code;
        
        // コード表示エリアを表示
        const codeContainer = document.getElementById('generatedCode');
        if (codeContainer) {
            codeContainer.style.display = 'block';
        }
    }
    
    showAlert('AIコードが生成されました！', 'success');
}

// =================
// 設定保存・読み込み
// =================
function saveConfiguration() {
    const config = collectAllConfigurations();
    
    try {
        localStorage.setItem('quantum_battle_config', JSON.stringify(config));
        showAlert('設定を保存しました', 'success');
        console.log('💾 設定保存完了');
    } catch (error) {
        showAlert('設定の保存に失敗しました', 'error');
        console.error('❌ 設定保存エラー:', error);
    }
}

function loadSavedConfiguration() {
    try {
        const saved = localStorage.getItem('quantum_battle_config');
        if (saved) {
            const config = JSON.parse(saved);
            applyConfiguration(config);
            console.log('📂 保存設定を読み込み');
        }
    } catch (error) {
        console.error('⚠️ 設定読み込みエラー:', error);
    }
}

function loadConfiguration() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    
    input.onchange = function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const config = JSON.parse(e.target.result);
                    applyConfiguration(config);
                    showAlert('設定ファイルを読み込みました', 'success');
                } catch (error) {
                    showAlert('設定ファイルの読み込みに失敗しました', 'error');
                }
            };
            reader.readAsText(file);
        }
    };
    
    input.click();
}

function applyConfiguration(config) {
    // 学習方法
    if (config.learningMethod) {
        selectedMethod = config.learningMethod;
    }
    
    // 配置設定
    if (config.placement) {
        if (config.placement.boardState) {
            boardState = config.placement.boardState;
        }
        if (config.placement.placedPieces) {
            placedPieces = config.placement.placedPieces;
        }
    }
    
    // フォーム値の復元
    Object.keys(config).forEach(section => {
        if (typeof config[section] === 'object') {
            Object.keys(config[section]).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    element.value = config[section][key];
                    // 値表示の更新
                    const valueDisplay = document.getElementById(key + 'Value');
                    if (valueDisplay) {
                        valueDisplay.textContent = config[section][key];
                    }
                }
            });
        }
    });
    
    // ボード状態の復元
    updatePlacementBoardFromState();
    updatePieceCount();
}

function updatePlacementBoardFromState() {
    document.querySelectorAll('.board-cell').forEach(cell => {
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        const key = `${row}-${col}`;
        
        cell.classList.remove('placed-good', 'placed-bad');
        cell.textContent = '';
        
        if (boardState[key]) {
            if (boardState[key] === 'good') {
                cell.classList.add('placed-good');
                cell.textContent = 'G';
            } else {
                cell.classList.add('placed-bad');
                cell.textContent = 'B';
            }
        }
    });
}

// =================
// ユーティリティ関数
// =================
function getElementValue(id) {
    const element = document.getElementById(id);
    return element ? element.value : '';
}

function showAlert(message, type = 'info') {
    // 既存のアラートを削除
    const existingAlert = document.querySelector('.alert-temporary');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    // 新しいアラートを作成
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-temporary`;
    alert.textContent = message;
    
    // ページの上部に追加
    const container = document.querySelector('.container');
    container.insertBefore(alert, container.firstChild);
    
    // 3秒後に自動削除
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 3000);
}

function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

function saveCurrentStepData() {
    // 現在のステップのデータを保存
    moduleConfigs[`step${currentStep}`] = collectCurrentStepData();
}

function collectCurrentStepData() {
    const stepData = {};
    
    // 現在表示されているフォーム要素から値を収集
    const activeContent = document.querySelector('.step-content.active');
    if (activeContent) {
        const inputs = activeContent.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (input.id) {
                stepData[input.id] = input.value;
            }
        });
    }
    
    return stepData;
}

function updatePreview() {
    // リアルタイムプレビュー更新
    // パフォーマンス考慮で500ms後に実行
    clearTimeout(updatePreview.timeoutId);
    updatePreview.timeoutId = setTimeout(() => {
        // プレビュー更新ロジック
        console.log('🔄 プレビュー更新');
    }, 500);
}

function autosave() {
    // 自動保存（2秒後）
    clearTimeout(autosave.timeoutId);
    autosave.timeoutId = setTimeout(() => {
        saveConfiguration();
    }, 2000);
}

// エクスポート用コード生成
function downloadCode() {
    const code = document.getElementById('codeOutput')?.textContent;
    if (!code) {
        showAlert('生成されたコードがありません', 'warning');
        return;
    }
    
    const blob = new Blob([code], { type: 'text/python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantum_geister_ai_${Date.now()}.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showAlert('コードファイルをダウンロードしました', 'success');
}

function copyToClipboard() {
    const code = document.getElementById('codeOutput')?.textContent;
    if (!code) {
        showAlert('生成されたコードがありません', 'warning');
        return;
    }
    
    navigator.clipboard.writeText(code).then(() => {
        showAlert('クリップボードにコピーしました', 'success');
    }).catch(() => {
        showAlert('クリップボードへのコピーに失敗しました', 'error');
    });
}

// デバッグ用
window.QuantumBattleDebug = {
    getCurrentStep: () => currentStep,
    getSelectedMethod: () => selectedMethod,
    getBoardState: () => boardState,
    getPlacedPieces: () => placedPieces,
    getModuleConfigs: () => moduleConfigs,
    collectAllConfigurations: collectAllConfigurations
};
#!/bin/bash

# ================== 指定 Windows Python 解释器 ==================
# ✅ 在 Git Bash 中必须使用正斜杠路径
PYTHON_BIN="D:/ProgramData/Anaconda3/envs/ML39/python.exe"

# ================== 检查 Python 路径 ==================
echo "==> 使用 Python 解释器: $PYTHON_BIN"
"$PYTHON_BIN" --version || { echo "!!> Python 路径错误，请检查 PYTHON_BIN"; exit 1; }

# ================== 基本配置 ==================
SEED=42
PROJECT_ROOT=$(cd "$(dirname "$0")"/.. && pwd)
SRC_DIR="$PROJECT_ROOT/src"
RESULTS_DIR="$PROJECT_ROOT/results"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
LOG_FILE="$RESULTS_DIR/train_eval.log"

export PYTHONHASHSEED=$SEED

# ================== 创建目录 ==================
mkdir -p "$RESULTS_DIR" "$CHECKPOINTS_DIR"

# ================== 运行训练 ==================
echo "======================================"
echo "==> 开始训练..."
echo "日志文件: $LOG_FILE"
echo "======================================"

# -u 表示 unbuffered，Python 的 stdout/stderr 不缓冲，实时打印。
"$PYTHON_BIN" -u "$SRC_DIR/main_train.py" --seed $SEED 2>&1 | tee "$LOG_FILE"

# ================== 检查输出结果 ==================
if [ -f "$RESULTS_DIR/training_curve.png" ]; then
    echo "==> 训练曲线已生成: $RESULTS_DIR/training_curve.png"
else
    echo "!!> 未找到训练曲线，请检查训练日志"
fi

if [ -f "$CHECKPOINTS_DIR/best_model.pt" ]; then
    echo "==> 最优模型已保存: $CHECKPOINTS_DIR/best_model.pt"
else
    echo "!!> 未保存模型，请检查训练是否正常结束"
fi

echo "======================================"
echo "==> 全流程结束！"
echo "==> 日志文件: $LOG_FILE"
echo "==> 训练结果与曲线保存在: $RESULTS_DIR/"
echo "==> 模型权重保存在: $CHECKPOINTS_DIR/"
echo "======================================"

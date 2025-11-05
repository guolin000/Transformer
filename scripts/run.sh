#!/bin/bash

# ================== 使用 Windows Python 解释器 ==================
# 注意：路径中的反斜杠必须双写或改为正斜杠
WIN_PYTHON_PATH="D:\\ProgramData\\Anaconda3\\envs\\ML39\\python.exe"
# 或者使用这种写法：
# WIN_PYTHON_PATH="D:/ProgramData/Anaconda3/envs/ML39/python.exe"

# 通过 cmd.exe 调用 Windows Python
PYTHON_BIN="cmd.exe /C $WIN_PYTHON_PATH"

echo "==> 使用 Python 解释器: $WIN_PYTHON_PATH"
$PYTHON_BIN --version || { echo "!!> Python 路径错误，请检查 PYTHON_BIN"; exit 1; }

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

## ================== 安装依赖 ==================
#if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
#    echo "==> 安装依赖..."
#    $PYTHON_BIN -m pip install --upgrade pip
#    $PYTHON_BIN -m pip install -r "$PROJECT_ROOT/requirements.txt"
#else
#    echo "未找到 requirements.txt，跳过依赖安装"
#fi

# ================== 运行训练 ==================
echo "======================================"
echo "==> 开始训练..."
echo "日志文件: $LOG_FILE"
echo "======================================"

$PYTHON_BIN "$SRC_DIR/main_train.py" --seed $SEED 2>&1 | tee "$LOG_FILE"

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

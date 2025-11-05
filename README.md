# 手工实现 Transformer (Encoder + Decoder) 模型

包括：
- Multi-Head Self-Attention
- Position-wise FFN
- Residual + LayerNorm
- Positional Encoding
- 小规模文本建模
- 消融实验（去掉模块对比性能）

## 实验环境
- GPU: NVIDIA RTX 4060 8GB
- 内存: 32GB
- Python 3.8
- PyTorch 2.4.1

## 安装依赖
```bash
pip install -r requirements.txt

project_root/
│
├── src/                        # 💡 项目的源代码（核心目录）
│   ├── data/                   # 数据读取与预处理模块
│   │   ├── __init__.py
│   │   ├── vocab.py            # 构建词表
│   │   ├── dataset.py          # 定义 TranslationDataset、collate_fn
│   │
│   ├── models/                 # 模型定义模块
│   │   ├── __init__.py
│   │   ├── transformer.py      # Transformer 模型实现
│   │   ├── layers.py           # 多头注意力、前馈网络等子模块
│   │
│   ├── utils/                  # 工具模块（配置、训练辅助、日志）
│   │   ├── __init__.py
│   │   ├── config.py           # 命令行参数解析（argparse）
│   │   ├── train_utils.py      # 训练/评估通用函数
│   │   ├── logger.py           # 日志工具（可选）
│   │
│   ├── main_train.py           # 主训练脚本（训练逻辑入口）
│   ├── evaluate.py             # 评估 / 翻译测试脚本
│   └── __init__.py             # 表明这是一个 Python 包
│
├── data/                       # 数据文件（非 src）
│   ├── train.en
│   ├── train.zh
│   ├── valid.en
│   ├── valid.zh
│
├── checkpoints/                # 模型保存目录
│
├── results/                    # 训练曲线图、日志、表格
│
├── scripts/                    # shell 脚本或运行命令
│   ├── run.sh
│   ├── update_requirements.sh
│
├── requirements.txt            # 项目依赖文件
├── README.md                   # 项目说明文档
└── .gitignore                  # Git 忽略文件

#  Transformer for IWSLT2017 zh-en Translation

本项目实现了一个基于 **Transformer** 的中英翻译模型（IWSLT2017 zh-en 数据集）。  
支持从数据预处理、模型训练到评估的完整流程，并通过 `run.sh` 脚本在 Windows 环境下使用 **Git Bash** 一键运行。

---

## 📦 项目结构
```
project_root/
│
├── src/ # 源代码目录（核心模块）
│ ├── data/ # 数据读取与预处理（构建词表、加载数据）
│ ├── models/ # 模型定义（Transformer及子模块）
│ ├── utils/ # 工具函数（配置解析、训练工具、日志）
│ ├── main_train.py # 主训练脚本
│ ├── ablation.py # 消融实验训练脚本（去掉去掉位置编码与减小Encoder/Decoder层数）
│ └── evaluate.py # 模型评估脚本
├── data/ # 预处理后的数据文件（训练/验证集等）
├── checkpoints/ # 模型权重保存目录
├── results/ # 日志与可视化结果
├── scripts/ # 运行脚本目录（包含 run.sh）
├── requirements.txt # Python 依赖包列表
└── README.md # 项目说明文档
```
## ⚙️ 环境要求

### 💻 硬件配置建议
| 项目 | 推荐配置 |
|------|-----------|
| 操作系统 | Windows 10 / 11 |
| GPU | NVIDIA GeForce RTX 3060 及以上（显存 ≥ 8GB） |
| CPU | Intel i7 / Ryzen 7 及以上 |
| 内存 | ≥ 16 GB |
| 存储空间 | ≥ 10 GB（含数据与日志） |

> ⚠️ 若无 GPU，训练可在 CPU 上运行但速度较慢。

### 🧩 软件依赖
| 组件 | 版本 |
|------|------|
| Python | 3.9+ |
| PyTorch | 2.0+ |
| NumPy | ≥ 1.23 |
| tqdm | ≥ 4.65 |
| matplotlib | ≥ 3.7 |
| openpyxl | ≥ 3.1 |

可通过以下命令安装依赖：
```bash
pip install -r requirements.txt
   ```

## 🚀 训练与实验运行

本项目提供自动化脚本 **`run.sh`**，可直接在 **Git Bash** 中运行。

### 🔧 Windows 下运行方式

1. 打开 **Git Bash**
2. 切换到项目根目录：
3. 运行脚本（会自动使用项目内配置）：
脚本内容包括：

* 指定 Python 解释器路径
* 设置随机种子（42）
* 自动创建 `results/` 与 `checkpoints/`
* 执行训练
* 保存日志与曲线图

## 🔁 可复现实验命令

以下命令可在不使用 `run.sh` 的情况下手动执行，完全复现实验：

```bash
D:/ProgramData/Anaconda3/envs/ML39/python.exe ./src/main_train.py --seed 42
```


## 🧠 消融实验

可通过运行 `src/ablation.py` 中的脚本验证去掉位置编码（Positional Encoding）与减小 Transformer 层数（Encoder/Decoder 层数）的影响
例如
```bash
D:/ProgramData/Anaconda3/envs/ML39/python.exe ./src/ablation.py --seed 42
```




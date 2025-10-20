# Transformer From Scratch

手工实现 Transformer (Encoder + Decoder) 模型，支持：
- Multi-Head Self-Attention
- Position-wise FFN
- Residual + LayerNorm
- Positional Encoding
- 小规模文本建模
- 消融实验（去掉模块对比性能）

## 硬件要求
- GPU: NVIDIA RTX 4060 8GB
- 内存: ≥32GB
- Python 3.8
- PyTorch 2.4.1

## 安装依赖
```bash
pip install -r requirements.txt

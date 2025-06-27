This project is licensed for personal, non-commercial use only.
Any redistribution or commercial usage is strictly prohibited.

# FINALE FP16 分层训练项目（最低 125MB 显存）

![](https://github.com/Heart-of-Africa/finale-FluxProp/blob/main/mem-loss.png)

该项目允许在显存非常有限的情况下（最低 125MB）逐层训练模型。

## 使用方式

1. 安装依赖：
    pip install -r requirements.txt

2. 训练第 i 层（例如第 5 层）：
    python train_layer.py --layer 5

3. 训练完 i 层，即可尝试部署 i 层
    python infer_layer0.py

5. 所有层训练完后，可在 merge_checkpoints.py 中合并权重。

## 技术策略
用 LNN 架构（包含液体时间常数和非线性记忆单元）处理序列建模

基于简化 MLP / GRU-like 架构取代了 self-attention

替代了 Transformer 的核心模块：Multi-Head Self Attention（MHSA）

去除了 attention score 的计算、mask 操作、位置编码等传统机制

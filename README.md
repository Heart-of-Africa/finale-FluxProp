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
    python inference.py

btw 尝试部署时请将checkpoints中的文件(.pt)放入主目录文件夹中，在inference.py中修改第22行中文件的名字：
    def load_model(checkpoint_path="your file name.pt", train_txt="train.txt"):

5. 所有层训练完后，可在 merge_checkpoints.py 中合并权重。

## 技术策略
1. 模型架构：

基于自定义的 LNNLanguageModel 类，核心集成逻辑神经模块与马尔科夫转移机制；

每层支持反向传播时动态计算马尔科夫转移矩阵，嵌入“状态跳转”解释能力；

训练中每一层可单独冻结或激活，支持逐层精调。

2. 训练机制：

使用标准 PyTorch 训练框架；

支持 AMP 混合精度训练；

自定义的 train_layer.py 脚本通过传参 --layer 指定训练的层；

自动保存 checkpoint 文件，可断点恢复。

3. 推理系统：

独立的 inference.py 脚本，支持加载指定 checkpoint 模型进行文本生成；

支持 GUI 操作界面，统一封装为 fluxprop_app.py，便于部署打包成桌面应用。

4. 分词系统：

当前未使用自定义 BPE 分词器；

训练语料已进行统一分词或预处理；

分词器可切换为 transformers 中的预定义 GPT2Tokenizer，或后续替换为自定义 tokenizer。

5. 环境与部署：

支持在 Ubuntu + CUDA GPU 环境中部署；

可使用 PyInstaller 生成 GUI 可执行包；

多进程分布式训练准备完备，可扩展至多 GPU；

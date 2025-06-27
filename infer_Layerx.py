import torch
from LNNModel import LNNLanguageModel
from train_layer import CharTokenizer  # 假设 tokenizer 在训练文件中定义
import os

# 加载24个模型层（假设都保存在 checkpoints/layer_xx 目录）
def load_merged_model(vocab_size, device="cuda"):
    num_layers = 24
    alpha = 0.1
    hidden_size = 256
    embed_size = 256

    model = LNNLanguageModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        alpha=alpha,
        device=device
    ).to(device)

    # 手动加载每层参数
    for i in range(num_layers):
        path = f"./checkpoints/layer_{i:02d}/checkpoint_step_3000.pt"
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            layer = model.liquid_layers[i]
            state = ckpt["model_state"]
            # 加载该层的参数，假设 key 是有命名空间的（你可能需要做映射）
            for name, param in layer.named_parameters():
                full_key = f"liquid_layers.{i}.{name}"
                if full_key in state:
                    param.data.copy_(state[full_key])
                else:
                    print(f"[警告] {full_key} 不在 checkpoint 中")
        else:
            print(f"[警告] 缺少 checkpoint：{path}")

    return model


# 生成函数（字符级）
def generate(model, tokenizer, prompt, max_len=100):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    output = prompt

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_tensor)
            next_token = torch.argmax(logits[0]).item()
            output += tokenizer.decode([next_token])
            input_tensor = torch.cat([input_tensor[:, 1:], torch.tensor([[next_token]], device=device)], dim=1)

    return output


# 主流程
if __name__ == "__main__":
    with open("train.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = CharTokenizer(raw_text)
    vocab_size = tokenizer.vocab_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_merged_model(vocab_size, device)

    prompt = "你好，今天的天气"
    result = generate(model, tokenizer, prompt)
    print("🧠 模型生成：", result)


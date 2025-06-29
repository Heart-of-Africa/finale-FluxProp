import torch
import os
from LNNModel import LNNLanguageModel
from train_layer import CharTokenizer  # 假设 tokenizer 定义在此

def merge_checkpoints(checkpoint_root="checkpoints", save_path="merged_model.pt", num_layers=24):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 100  # 初始值，将在加载 tokenizer 后更新

    # 读取 tokenizer 以确定 vocab size
    with open("train.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = CharTokenizer(raw_text)
    vocab_size = tokenizer.vocab_size()

    # 初始化空模型结构
    model = LNNLanguageModel(vocab_size=vocab_size, num_layers=num_layers, device=device).to(device)

    for i in range(num_layers):
        path = os.path.join(checkpoint_root, f"layer_{i:02d}", "checkpoint_step_3000.pt")
        if not os.path.exists(path):
            print(f"[⚠️] 缺失权重：{path}")
            continue

        ckpt = torch.load(path, map_location=device)
        state = ckpt["model_state"]
        for name, param in model.liquid_layers[i].named_parameters():
            full_key = f"liquid_layers.{i}.{name}"
            if full_key in state:
                param.data.copy_(state[full_key])
            else:
                print(f"[⚠️] 缺失参数：{full_key}")

    # 保存合并好的模型权重
    torch.save(model.state_dict(), save_path)
    print(f"[✅] 已保存合并模型权重到：{save_path}")

if __name__ == "__main__":
    merge_checkpoints()

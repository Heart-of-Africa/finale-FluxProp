import torch
from LNNModel import LNNLanguageModel
import os

# 独立 tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens):
        return ''.join([self.itos[tok] for tok in tokens])

    def vocab_size(self):
        return len(self.chars)

# 加载模型和 tokenizer
def load_model(checkpoint_path="checkpoint_step_0610.pt", train_txt="train.txt"):
    with open(train_txt, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = CharTokenizer(raw_text)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LNNLanguageModel(tokenizer.vocab_size(), device=device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, device

# 文本生成函数
@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=100):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    for _ in range(max_new_tokens):
        logits = model(input_tensor)  # [1, T, vocab]
        next_token = torch.argmax(logits[0, -1], dim=-1).unsqueeze(0)  # [1]
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)  # [1, T+1]
    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)

# 主循环
if __name__ == "__main__":
    model, tokenizer, device = load_model()
    print("✅ 模型加载完毕，开始对话。输入 'exit' 可退出。")

    while True:
        prompt = input("你：").strip()
        if prompt.lower() == "exit":
            print("👋 再见！")
            break

        output = generate(model, tokenizer, prompt, device).strip()
        print(f"🤖：{output}")
        print("-" * 40)


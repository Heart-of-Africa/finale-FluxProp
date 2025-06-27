import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from LNNModel import LNNLanguageModel
from monitor import log_gpu_usage
from plot_utils import plot_loss_and_memory
from torch.nn.utils import clip_grad_norm_

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

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=32):
        self.data = tokenizer.encode(text)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+self.seq_len], dtype=torch.long)
        return x, y

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

# 准备数据
with open("train.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = CharTokenizer(raw_text)
vocab_size = tokenizer.vocab_size()
dataset = TextDataset(raw_text, tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 初始化模型和训练工具
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LNNLanguageModel(vocab_size, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

step = 0
loss_log = []
mem_log = []
max_steps = 3000

# 创建输出目录
image_dir = os.path.join("images", f"layer_{args.layer}")
ckpt_dir = os.path.join("checkpoints", f"layer_{args.layer}")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# 训练主循环
for epoch in range(1, 100):
    total_loss = 0
    for _, (x, y) in enumerate(loader):
        model.train()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        total_loss += loss.item()
        loss_log.append(loss.item())
        mem_log.append(torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0)

        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
            log_gpu_usage(step=step)

            # 保存 checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step:04d}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
            }, ckpt_path)

            # 保存图像
            plot_path = os.path.join(image_dir, f"loss_mem_step_{step:04d}.png")
            plot_loss_and_memory(loss_log, mem_log, save_path=plot_path)

        if step >= max_steps:
            break
    print(f"===> Epoch {epoch} Average Loss: {total_loss/len(loader):.4f}")
    if step >= max_steps:
        break


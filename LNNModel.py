import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def compute_transition_matrix(h):
    raw = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))  # outer product: [B, H, H]
    return torch.softmax(raw, dim=-1)

class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, alpha=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.U = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.alpha = alpha

    def forward(self, h, x):
        dh = -self.alpha * h + torch.tanh(torch.matmul(h, self.W.T) + torch.matmul(x, self.U.T) + self.b)
        return h + dh

    def markov_backward_hook(self, grad):
        M = compute_transition_matrix(grad)
        return torch.bmm(M, grad.unsqueeze(2)).squeeze(2)

class LNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=1024, num_layers=12, alpha=0.1, device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.liquid_layers = nn.ModuleList([
            LiquidNeuron(embed_size, hidden_size, alpha) for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.device = device

    def forward(self, input_ids):
        x = self.embedding(input_ids.to(self.device))  # [B, T, E]
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.liquid_layers[0].W.shape[0], device=self.device)
        for t in range(seq_len):
            xt = x[:, t, :]
            for i, layer in enumerate(self.liquid_layers):
                h = layer(h, xt)
                if self.training:
                    h.register_hook(self._record_grad_hook(i, xt))
        logits = self.decoder(h)
        return logits

    def _record_grad_hook(self, layer_id, xt):
        def hook(grad):
            grad_dir = "grad_paths"
            os.makedirs(grad_dir, exist_ok=True)
            path = os.path.join(grad_dir, f"layer_{layer_id}_grad.pt")
            torch.save(grad.detach().cpu(), path)
            return self.liquid_layers[layer_id].markov_backward_hook(grad)
        return hook
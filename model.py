
import torch
import torch.nn as nn

class LiquidTimeConstantModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w_ih = nn.Linear(input_size, hidden_size)
        self.w_hh = nn.Linear(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.ones(1) * 0.9)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.w_hh.out_features, device=x.device)
        for _ in range(1):
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(self.w_ih(x) + self.w_hh(h))
        return h

class MarkovGradientGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        probs = torch.sigmoid(self.logits)
        return x * probs

class LiquidMarkovModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768):
        super().__init__()
        self.lnn = LiquidTimeConstantModule(input_dim, hidden_dim)
        self.markov = MarkovGradientGate(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.lnn(x)
        gated = self.markov(h)
        return self.out(gated)

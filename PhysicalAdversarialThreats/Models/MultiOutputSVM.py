import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSVM(nn.Module):
    """Single-logit linear model"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
    def forward(self, x):
        x = x.view(x.size(0), -1)       # [B, 2000]
        return self.linear(x).squeeze(1)

class MultiOutputSVM(nn.Module):
    """Two-logit head: logits=[-f, +f] (symmetric, no bias shift)."""
    def __init__(self, input_dim, base_state_dict):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.out    = nn.Linear(1, 2, bias=True)
        self.load_base_and_fix_head(base_state_dict)

    def load_base_and_fix_head(self, state_dict):
        # copy base weights (w,b) into the first linear
        base = BaseSVM(self.linear.in_features)
        base.load_state_dict(state_dict)
        with torch.no_grad():
            self.linear.weight.copy_(base.linear.weight)
            self.linear.bias.copy_(base.linear.bias)
            # symmetric mapping to two logits
            self.out.weight.zero_(); self.out.bias.zero_()
            self.out.weight[0,0] = -1.0   # class 0 logit = -margin
            self.out.weight[1,0] =  1.0   # class 1 logit = +margin

    def forward(self, x):
        f = self.linear(x.view(x.size(0), -1)).squeeze(1)   # margin
        logits = self.out(f.unsqueeze(1))                   # [B,2]
        return logits   # (use CE or softmax outside as needed)
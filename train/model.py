"""The policy/value network. Small on purpose: it runs in plain JavaScript in the game, every
tick, so it has to stay a few million multiply-adds. nav/model.js mirrors this forward pass."""
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vecenv import ACTIONS, CHANNELS, SCALARS, VIEW_H, VIEW_W


class Net(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 32, 3, stride=2, padding=1)   # 21x33 -> 11x17
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)         # -> 6x9
        flat = 64 * 6 * 9
        self.fc1 = nn.Linear(flat + SCALARS, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, ACTIONS)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, grid, scal):
        x = grid.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.cat([x.flatten(1), scal], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.pi(x), self.v(x).squeeze(-1)


def export_json(net: Net):
    """Weights for nav/model.js: float32, base64, PyTorch layouts (conv [out,in,kh,kw], linear [out,in])."""
    def enc(t):
        return base64.b64encode(t.detach().cpu().float().numpy().astype("<f4").tobytes()).decode()

    layers = []
    for name, kind, extra in [("conv1", "conv", {"stride": 2, "pad": 1}), ("conv2", "conv", {"stride": 2, "pad": 1}),
                              ("fc1", "linear", {}), ("fc2", "linear", {}), ("pi", "linear", {}), ("v", "linear", {})]:
        m = getattr(net, name)
        layers.append({"name": name, "type": kind, "shape": list(m.weight.shape), "w": enc(m.weight), "b": enc(m.bias), **extra})
    return {"format": "tpbot-nav-1", "channels": CHANNELS, "view": [VIEW_H, VIEW_W], "scalars": SCALARS,
            "actions": ACTIONS, "layers": layers}


def reference_io(net: Net, seed=0):
    """A random input and the network's output, so nav/model.js can be checked against PyTorch."""
    g = np.random.default_rng(seed)
    grid = g.integers(0, 256, (1, CHANNELS, VIEW_H, VIEW_W), dtype=np.uint8)
    scal = g.standard_normal((1, SCALARS)).astype(np.float32)
    with torch.no_grad():
        logits, v = net(torch.from_numpy(grid), torch.from_numpy(scal))
    return {"grid": base64.b64encode(grid.tobytes()).decode(), "scalars": scal[0].tolist(),
            "logits": logits[0].tolist(), "value": float(v[0])}

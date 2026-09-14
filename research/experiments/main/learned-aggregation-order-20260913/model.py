"""Shared residual coordinate map; the only treatment is aggregation order."""
import torch
import torch.nn.functional as F

class Map(torch.nn.Module):
    def __init__(self, dimension=896, rank=32):
        super().__init__()
        self.down = torch.nn.Linear(dimension, rank, bias=False)
        self.up = torch.nn.Linear(rank, dimension, bias=False)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x, mode):
        if mode == 'map_after_mean':
            x = x.mean(-2)
        x = x + self.up(F.relu(self.down(x)))
        if mode == 'map_before_mean':
            x = x.mean(-2)
        return F.normalize(x, dim=-1)

def ridge(s, q, penalty):
    # s: batch,class,shot,dimension; q: batch,query,dimension.
    x = s.flatten(1, 2)
    y = F.one_hot(torch.arange(s.shape[1], device=s.device).repeat_interleave(s.shape[2]),
                  s.shape[1]).to(s.dtype)[None].expand(len(s), -1, -1)
    xm, ym = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
    xc, yc = x-xm, y-ym
    a = torch.linalg.solve(xc @ xc.transpose(-1,-2) + penalty*torch.eye(x.shape[1],device=s.device,dtype=s.dtype), yc)
    return (q-xm) @ xc.transpose(-1,-2) @ a + ym

def scores(net, s, q, mode, penalty=.1):
    return ridge(net(s,mode), net(q,mode), penalty)

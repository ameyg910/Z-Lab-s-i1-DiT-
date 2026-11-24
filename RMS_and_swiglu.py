import torch 
import torch.nn as nn
import torch.nn.functional as F 
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps 
        self.scale = nn.Parameter(torch.ones(dim)) 

    def forward(self, x): 
        dT = x.dtype 

        x = x.float()
        x = x*torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x*self.scale.float()).to(dT) 


class SwiGLUffn(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(2/3*int(dim*mlp_ratio))
        self.w12 = nn.Linear(dim, 2*hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x): 
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a)*b)
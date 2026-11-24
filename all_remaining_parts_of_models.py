import torch 
import torch.nn as nn 
import numpy as np
"""
step : Input -> 


Input: (B, 32, 32, 32): 32 channels on a 32*32 grid 
after the conv (kernel 2, stride 2) : (B, 512, 16, 16)
.flatten(2) : (B, 512, 256)
.transpose(1, 2): (B, 256, 512) : 256 tokens, each of a 512-d vector

"""

class PatchEmbed(nn.Module): 
    def __init__(self, patch, dim, in_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)



def sincos_1d(dim, pos):
        omega = 1.0 / (10000 ** (np.arange(dim // 2, dtype=np.float64)/(dim/2.0)))
        out = np.outer(pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

def sincos_2d(dim, grid, resolution, base_resolution=256):
     scale=base_resolution/resolution # 1.0 at 256px, 0.5 at 512px
     coords = np.arange(grid, dtype=np.float32)*scale
     gw, gh = np.meshgrid(coords, coords) # gw: column coordinate, gh: row coordinate
     return np.concatenate([sincos_1d(dim//2, gw), sincos_1d(dim//2, gh)], axis=1)
"""
1. Half the channels encode the column, half encode the row. 
   Each axis gets its own D/2-dim 1D embedding, and the two are glued together.
2. np.meshgrid(coords, coords) returns the column coordinates first. 
   gw[r, c] = c and gh[r, c] = r, both flattened in row-major order, which matches the patch order above.
3. The scale line is i1's resolution trick. 
   At 256px the coordinates are 0, 1, ..., 15. At 512px the grid is 32*32 and the coordinates become 0, 0.5, 1, ..., 15.5. The model sees the same position range, 
   just sampled more densely.
"""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from libs.factorization_module import FABlock3D_m, FABlock3D_m

from libs.positional_encoding_module import GaussianFourierFeatureTransform
    

class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 n_layer,
                 model,
                 **kwargs
             ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(3, dim // 2, 8),
                nn.Linear(dim, dim)
            ))
            if model=="IFactFormer_o":
                layer.append(FABlock3D_o(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs))
            elif model=="IFactFormer_m":
                layer.append(FABlock3D_m(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs))
            self.layers.append(layer)
            
        self.n_layer = n_layer

    def forward(self, u, pos_lst):
        b, nx, ny, nz, c = u.shape  # just want to make sure its shape
        nx, ny, nz = pos_lst[0].shape[0], pos_lst[1].shape[0], pos_lst[2].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1),
                                          pos_lst[1].squeeze(-1),
                                          pos_lst[2].squeeze(-1)]
                                         ), dim=-1).reshape(-1, 3)
        
        for l, (pos_enc, attn_layer) in enumerate(self.layers):
            u += rearrange(pos_enc(pos), '1 (nx ny nz) c -> 1 nx ny nz c', nx=nx, ny=ny, nz=nz)
            for i in range(self.n_layer):
                u = u + attn_layer(u, pos_lst) / self.n_layer
        return u
        
        
        
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.to_in = nn.Sequential(
            nn.Conv2d(config.in_dim, config.dim // 2, kernel_size=1, stride=1, padding=0, groups=config.in_dim),
            nn.GELU(),
            nn.Conv2d(config.dim // 2, config.dim, kernel_size=(config.in_time_window, 1), stride=1, padding=0, bias=False),
        )

        self.encoder = FactorizedTransformer(config.dim, config.dim_head, config.heads, config.dim, config.depth, config.n_layer, config.model)
        
        
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny nz c -> b c (nx ny nz)'),
            nn.Conv1d(config.dim, config.dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(config.dim // 2, config.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )       
        
    def forward(self,
                u,
                pos_lst,
                ):
        b, t, nx, ny, nz, c = u.shape
        
        u = rearrange(u, 'b t nx ny nz c -> b c t (nx ny nz)')
        u = self.to_in(u)
        u = rearrange(u, 'b c 1 (nx ny nz) -> b nx ny nz c', nx=nx, ny=ny, nz=nz)
        
        u = self.encoder(u, pos_lst)
        
        u = self.simple_to_out(u)
        u = rearrange(u, 'b c (nx ny nz) -> b nx ny nz c', nx=nx, ny=ny)
        
        return u
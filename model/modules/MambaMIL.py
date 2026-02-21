"""
MambaMIL
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
import numpy as np
from time import time


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, mamba_dim, dropout, n_layer=2,agg=True):
        super(MambaMIL, self).__init__()
        self.attn_agg=ABMIL(mamba_dim,return_attn=True)
        self.layers = nn.ModuleList()
        self.drop_out = nn.Dropout(dropout) #added by me
        self.agg=agg

        for _ in range(n_layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(mamba_dim),
                    Mamba(
                        d_model=mamba_dim,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]
        
        for layer in self.layers:
            h_ = h
            h = layer[0](h)
            h = self.drop_out(h) # added by me
            h = layer[1](h)
            h = h + h_

        if self.agg:
            h, A = self.attn_agg(h)
            self.attn_buffer = A.detach().cpu().squeeze()
            return h
        else:
            return h

    #call after making a prediction
    def get_last_attn(self):
        return self.attn_buffer.numpy()
    
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


class ABMIL(nn.Module):
    def __init__(self, dim, return_attn=False):
        super(ABMIL, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.return_attn = return_attn

        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]

        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(1)
        if(self.return_attn):
            return h, A
        else:
            return h
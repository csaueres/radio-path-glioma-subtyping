import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.modules.MambaMIL import MambaMIL

#TODO: make these compatible with unimodal datasets

class UnimodalModel(nn.Module):
    def __init__(self, in_dim, n_classes, dropout):
        super().__init__()
        self.multi_out=False
        self.hidden_dim=64

        self.adapter = nn.Linear(in_dim, self.hidden_dim)
        self.expert = None
        self.classifier = nn.Sequential(nn.Dropout(dropout),nn.Linear(self.hidden_dim, 64),nn.ReLU(),nn.Linear(64, n_classes))
    def forward(self,x):
        h = self.adapter(x)
        h = self.expert(h)
        logits = self.classifier(h)
        return logits

    def pred_w_attn(self,x):
        logits = self.classifier(self.expert(self.adapter(x)))
        return logits, {'unimodal_attn':self.expert.get_last_attn()}

class HistoOnlyMamba(UnimodalModel):
    def __init__(self, in_dim, n_classes, dropout, n_layer):
        super().__init__(in_dim, n_classes, dropout)
        self.expert = MambaMIL(self.hidden_dim, dropout,n_layer)
   
    def forward(self,histo_x,mri_x):
        return super().forward(histo_x)


class MRIOnlyMamba(UnimodalModel):
    def __init__(self, in_dim, n_classes, dropout, n_layer):
        super().__init__(in_dim, n_classes, dropout)
        self.expert = MambaMIL(self.hidden_dim, dropout,n_layer)
   
    def forward(self,histo_x,mri_x):
        return super().forward(mri_x)

class HistoMLP(UnimodalModel):
    def __init__(self, in_dim, n_classes, dropout):
        super().__init__(in_dim, n_classes, dropout)
        self.expert = nn.Sequential(nn.ReLU(),nn.Linear(self.hidden_dim,self.hidden_dim))

   
    def forward(self,histo_x,mri_x):
        histo_x = torch.mean(histo_x,dim=1,keepdims=False)
        return super().forward(histo_x)

class MRIMLP(UnimodalModel):
    def __init__(self, in_dim, n_classes, dropout):
        super().__init__(in_dim, n_classes, dropout)
        self.expert = nn.Sequential(nn.ReLU(),nn.Linear(self.hidden_dim,self.hidden_dim))
   
    def forward(self,histo_x,mri_x):
        return super().forward(mri_x)
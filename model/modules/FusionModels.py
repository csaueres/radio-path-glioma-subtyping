import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.modules.MambaMIL import MambaMIL, ABMIL
from model.modules.UnimodalModels import MRIOnlyMamba, HistoOnlyMamba
import random
import math


class AbstractEarlyFusion(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, n_layer, multi_out=True):
        super().__init__()
        self.multi_out=multi_out
        self.shared_size = 16
        self.histo_adapter = nn.Linear(in_dim, self.shared_size)
        self.mri_adapter = nn.Linear(self.mri_dim_in,self.shared_size)
        self.shared_expert=None
        #self.class_head = nn.Linear(self.shared_size,n_classes)
        self.class_head = nn.Sequential(nn.Dropout(dropout),nn.Linear(self.shared_size, 64),nn.ReLU(),nn.Linear(64, n_classes))


    def forward(self,histo_x,mri_x):
        histo_h = self.histo_adapter(histo_x)
        mri_h = self.mri_adapter(mri_x)
        combined_adapted_patches = self.combine_modalities(histo_h,mri_h)
        joint_logits = self.class_head(self.shared_expert(combined_adapted_patches))
        mri_logits = self.class_head(self.shared_expert(mri_h))
        histo_logits = self.class_head(self.shared_expert(histo_h))
        return histo_logits,mri_logits,joint_logits

class SequenceEarlyFusion(AbstractEarlyFusion):
    def __init__(self, in_dim, n_classes, dropout, n_layer):
        self.mri_dim_in = 768
        super().__init__(in_dim, n_classes, dropout)
        self.shared_expert=MambaMIL(self.shared_size, dropout,n_layer)

    def combine_modalities(self,histo_h,mri_h):
        return torch.cat([histo_h,mri_h],dim=1)

    #combines all the modalities that are present
    #linear model doesnt have attn so doesnt need this method
    def pred_w_attn(self,histo_x,mri_x):
        assert(not(histo_x is None and mri_x is None))
        adapted_patches_list = []
        if(histo_x is not None):
            adapted_patches_list.append(self.histo_adapter(histo_x))
        if(mri_x is not None):
            adapted_patches_list.append(self.mri_adapter(mri_x))
        joint_patches = torch.cat(adapted_patches_list,dim=1)
        out = self.class_head(self.shared_expert(joint_patches))
        attn = self.shared_expert[1].get_last_attn()
        if(histo_x is not None and mri_x is not None):
            attn_dic = {'histo_attn':attn[:197],'mri_attn':attn[-197:]}
        else:
            attn_dic = {'unimodal_attn':attn}
        return out,  attn_dic

class LinearEarlyFusion(AbstractEarlyFusion):
    def __init__(self, in_dim, n_classes, dropout, n_layer):
        self.mri_dim_in = 1536
        super().__init__(in_dim, n_classes, dropout)
        self.shared_expert = nn.Sequential(nn.ReLU(),nn.Linear(self.shared_size,self.shared_size))

    #early fusion implemented with meaning
    def combine_modalities(self,histo_h,mri_h):
        return (histo_h+mri_h)*0.5

    def forward(self,histo_x,mri_x):
        histo_x = torch.mean(histo_x,dim=1,keepdims=False)
        return super().forward(histo_x,mri_x)


#abstract implementation of a late fusion network with unimodal and fused heads
#dont instantiate
class AbstractTripleHeadNet(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, n_layer):
        super().__init__()
        self.multi_out=True
        self.histo_adapter = nn.Linear(in_dim, self.histo_dim_out)
        self.mri_adapter = nn.Linear(self.mri_dim_in,self.mri_dim_out)
        self.histo_expert = None
        self.mri_expert = None
        self.norm = nn.BatchNorm1d(self.histo_dim_out+self.mri_dim_out)
        self.joint_head = nn.Sequential(nn.Dropout(dropout),nn.Linear(self.histo_dim_out+self.mri_dim_out, 64),nn.ReLU(),nn.Linear(64, n_classes))
        self.histo_head=nn.Linear(self.histo_dim_out, 3)
        self.mri_head=nn.Linear(self.mri_dim_out, 3)

    def forward(self,histo_x,mri_x):
        histo_adapted = self.histo_adapter(histo_x)
        mri_adapted = self.mri_adapter(mri_x)
        histo_h = self.histo_expert(histo_adapted)
        mri_h = self.mri_expert(mri_adapted)
        combined_rep=torch.cat([mri_h,histo_h],dim=1)
        combined_rep=self.norm(combined_rep)
        out_joint = self.joint_head(combined_rep)
        out_h = self.histo_head(histo_h)
        out_m = self.mri_head(mri_h)
        #NOTE: activate for mean logit for unpaired evaluation
        #out_joint = (out_h+out_m)/2
        return (out_h,out_m,out_joint)

    def pred_histo(self,x):
        histo_adapted = self.histo_adapter(x)
        histo_h = self.histo_expert(histo_adapted)
        out_h = self.histo_head(histo_h)
        return out_h
        
    def pred_mri(self,x):
        h = self.mri_expert(self.mri_adapter(x))
        out = self.mri_head(h)
        return out


class LinearTripleHead(AbstractTripleHeadNet):
    def __init__(self, in_dim, n_classes, dropout):
        self.mri_dim_in = 1536
        self.mri_dim_out = 64
        self.histo_dim_out = 64
        super().__init__(in_dim, n_classes, dropout)
        self.histo_expert = nn.Sequential(nn.ReLU(),nn.Linear(self.histo_dim_out,self.histo_dim_out))
        self.mri_expert = nn.Sequential(nn.ReLU(),nn.Linear(self.mri_dim_out,self.mri_dim_out))
    def forward(self,histo_x,mri_x):
        histo_x = torch.mean(histo_x,dim=1,keepdims=False)
        return super().forward(histo_x,mri_x)

#expects sequence of patches for both
class MambaTripleHead(AbstractTripleHeadNet):
    def __init__(self, in_dim, n_classes, dropout, n_layer):
        self.histo_dim_in = in_dim
        self.mri_dim_in = 768
        self.mri_dim_out = 64
        self.histo_dim_out = 64
        super().__init__(in_dim, n_classes, dropout)
        self.histo_expert=MambaMIL(self.histo_dim_out, dropout,n_layer/2)
        self.mri_expert=MambaMIL(self.mri_dim_out, dropout,n_layer/2)

#To address reviewer comment
class LogitLevelFusion(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, n_layer):
        self.multi_out=False
        self.histo_dim_in = in_dim
        self.mri_dim_in = 768
        super().__init__()
        self.histo_model=HistoOnlyMamba(self.histo_dim_in,n_classes,dropout,n_layer/2)
        self.mri_model=MRIOnlyMamba(self.mri_dim_in,n_classes,dropout,n_layer/2)

    def forward(self,histo_x,mri_x):
        histo_out = self.histo_model(histo_x,mri_x)
        mri_out = self.mri_model(histo_x,mri_x)
        mean_out = histo_out*0.5 + mri_out*0.5
        return mean_out


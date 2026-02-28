
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.modules.MambaMIL import MambaMIL
from utils.utils import device


class MoE(nn.Module):
    def __init__(self, histo_dim, mri_dim, n_classes, dropout, n_layer, linear):
        super().__init__()
        self.multi_out=True
        self.linear = linear
        self.shared_size=16
        self.expert_out_dim=64

        self.histo_adapter = nn.Linear(histo_dim, self.shared_size)
        self.mri_adapter = nn.Linear(mri_dim,self.shared_size)

        if(self.linear):
            self.histo_expert=nn.Sequential(nn.ReLU(),nn.Linear(self.shared_size, self.expert_out_dim))
            self.mri_expert=nn.Sequential(nn.ReLU(),nn.Linear(self.shared_size, self.expert_out_dim))
        else:
            self.histo_expert=nn.Sequential(nn.Linear(self.shared_size, self.expert_out_dim), MambaMIL(self.expert_out_dim, dropout,n_layer//2))
            self.mri_expert=nn.Sequential(nn.Linear(self.shared_size, self.expert_out_dim), MambaMIL(self.expert_out_dim, dropout,n_layer//2))
        self.h_norm = nn.BatchNorm1d(self.expert_out_dim)
        self.m_norm = nn.BatchNorm1d(self.expert_out_dim)
        self.router = nn.Sequential(
            nn.Linear(self.shared_size, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
        self.joint_class_head = nn.Sequential(nn.Dropout(dropout),nn.Linear(self.expert_out_dim*2, 64),nn.ReLU(),nn.Linear(64, n_classes))

    def forward(self,histo_x,mri_x,mixing=True):
        if(self.linear):
            #use mean patch.
            histo_x = None if histo_x is None else torch.mean(histo_x,dim=1,keepdims=False)
        if(not mixing):
            return self.unmixed_forward(histo_x,mri_x)
        else:
            return self.paired_mixing_all(histo_x,mri_x)


    #same as paired mixing all except doesnt use mixing
    def unmixed_forward(self,histo_x,mri_x):
        histo_h = self.histo_adapter(histo_x)
        mri_h = self.mri_adapter(mri_x)

        histo_rep = self.h_norm(self.histo_expert(histo_h))
        mri_rep = self.m_norm(self.mri_expert(mri_h))
        missing_mod_embed = torch.zeros(histo_rep.shape,device=histo_rep.device)
        pseudo_joint_histo = torch.cat([missing_mod_embed,histo_rep],dim=1)
        pseudo_joint_mri = torch.cat([mri_rep,missing_mod_embed],dim=1)
        histo_out = self.joint_class_head(pseudo_joint_histo)
        mri_out = self.joint_class_head(pseudo_joint_mri)
        joint_rep = torch.cat([mri_rep,histo_rep],dim=1)
        joint_out = self.joint_class_head(joint_rep)
        return histo_out,mri_out,joint_out

        
    #assumes entire batch has all modalities
    def paired_mixing_all(self,histo_x,mri_x):
        histo_h = self.histo_adapter(histo_x)
        mri_h = self.mri_adapter(mri_x)
        histo_rep = self.h_norm(self.mix_experts(histo_h))
        mri_rep = self.m_norm(self.mix_experts(mri_h))
        joint_rep = torch.cat([mri_rep,histo_rep],dim=1)
        joint_out = self.joint_class_head(joint_rep)
        pseudo_joint1 = torch.cat([torch.zeros(mri_rep.shape,device=mri_rep.device),histo_rep],dim=1)
        histo_out = self.joint_class_head(pseudo_joint1)
        pseudo_joint2 = torch.cat([mri_rep,torch.zeros(histo_rep.shape,device=histo_rep.device)],dim=1)
        mri_out = self.joint_class_head(pseudo_joint2)
        return histo_out,mri_out,joint_out

    def mix_experts(self,mod_h,ret_weight=False):
        histo_out = self.histo_expert(mod_h)
        mri_out = self.mri_expert(mod_h)
        weight = self.get_router_weighting(mod_h) #B, 2
        stacked_experts = torch.stack([mri_out,histo_out],dim=-1) #B,S,2
        # mixed_out = torch.matmul(stacked_experts,weight)
        mixed_out = torch.einsum('bsm,bm->bs',stacked_experts,weight).squeeze(-1)
        if ret_weight:
            return mixed_out,weight
        return mixed_out

    def get_router_weighting(self,mod_h):
        #router operates on mean patch for simplicity
        if(not self.linear):
            mod_h = torch.mean(mod_h,dim=1,keepdims=False)
        w = self.router(mod_h)
        w = F.softmax(w,dim=-1)
        return w

    def pred_w_attn(self,histo_x,mri_x):
        assert(not(histo_x is None and mri_x is None))
        attn_dic = {}
        if(mri_x is not None):
            mri_h = self.mri_adapter(mri_x)
            mri_h, mri_router_weights = self.mix_experts(mri_h,ret_weight=True)
            attn_dic['mri_router_weight']=mri_router_weights.detach().cpu().numpy()
            attn_dic['me_mri_attn'] = self.mri_expert[1].get_last_attn()
            attn_dic['he_mri_attn'] = self.histo_expert[1].get_last_attn()
            mri_rep = self.m_norm(mri_h)
        else:
            mri_rep = None
        if(histo_x is not None):
            histo_h = self.histo_adapter(histo_x)
            histo_h,histo_router_weights = self.mix_experts(histo_h,ret_weight=True)
            attn_dic['histo_router_weight']=histo_router_weights.detach().cpu().numpy()
            attn_dic['me_histo_attn'] = self.mri_expert[1].get_last_attn()
            attn_dic['he_histo_attn'] = self.histo_expert[1].get_last_attn()
            histo_rep = self.h_norm(histo_h)
        else:
            histo_rep = None
        if(mri_rep is None): mri_rep = torch.zeros(histo_rep.shape,device=histo_x.device)
        if(histo_rep is None): histo_rep = torch.zeros(mri_rep.shape,device=mri_x.device)
        joint_rep = torch.cat([mri_rep,histo_rep],dim=1)
        out = self.joint_class_head(joint_rep)
        return out,  attn_dic

    def get_embedding(self,histo_x,mri_x):
        histo_h = self.histo_adapter(histo_x)
        mri_h = self.mri_adapter(mri_x)
        histo_rep = self.h_norm(self.mix_experts(histo_h))
        mri_rep = self.m_norm(self.mix_experts(mri_h))
        joint_rep = torch.cat([mri_rep,histo_rep],dim=1)
        return joint_rep.detach().cpu()

    def get_patch_embeddings(self,histo_x,mri_x):
        self.histo_expert[1].agg=False
        self.mri_expert[1].agg=False
        histo_h = self.histo_adapter(histo_x)
        mri_h = self.mri_adapter(mri_x)
        histo_rep = self.histo_expert(histo_h)
        mri_rep = self.mri_expert(mri_h)
        joint_rep = torch.cat([mri_rep,histo_rep],dim=1)
        return joint_rep.detach().cpu()

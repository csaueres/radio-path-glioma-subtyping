import torch
import torch.nn as nn
from typing import Optional


class CLIPLoss(nn.Module):
    def __init__(self,temperature):
        super().__init__()
        self.temp = temperature

    #labels just for compatability
    def forward(self,v1,v2,labels):
        v1 = nn.functional.normalize(v1, dim=-1)
        v2 = nn.functional.normalize(v2, dim=-1)
        # print("MRI Embed: ", v2[:,:4])
        logits = (v1 @ v2.T) / self.temp
        # print(logits)
        targets=torch.arange(len(v1)).to(logits.get_device())
        l1 = nn.functional.cross_entropy(logits, targets, reduction='none')
        l2 = nn.functional.cross_entropy(logits.T, targets, reduction='none')
        loss =  (l1+l2) / 2.0 # shape: (batch_size)
        return loss.mean()

#can use either clip loss (unsupervised, only pairs are considered same) or supervisedInfoNCE
class CoMMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.info_nce = MySupervisedInfoNCE(temperature=0.1)

    def forward(self,x1,x2,y1,y2,labels):
        loss = self.info_nce(x1,x2,labels)
        loss+= self.info_nce(y1,y2,labels)
        loss+= self.info_nce(x1,y2,labels)
        loss+= self.info_nce(x2,y1,labels)
        loss+= self.info_nce(x2,y2,labels)
        #skip x1,y1
        return loss

        
class SupervisedInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

        self.temp = temperature

    def forward(self, view1,view2, target):

        view1 = nn.functional.normalize(view1, dim=-1)
        view2 = nn.functional.normalize(view2, dim=-1)

        # Calculate similarities
        sim = (view1 @ view2.T) / self.temp

        # print(sim)
        # print("MRI Embed: ", view2[:,:4])
        # print(target)


        #just copy clip loss but with torch.nn.BCEWithLogitsLoss or torch.nn.functional.binary_cross_entropy_with_logits instead of regular ce?

        # Build positive and negative masks
        mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))
        neg_mask = 1 - mask
        
        # Things with mask = 0 should be ignored in the sum.
        # If we just gave a zero, it would be log sum exp(0) != 0
        # So we need to give them a small value, with log sum exp(-1000) \approx 0
        pos_mask_add = neg_mask * (-1000)
        neg_mask_add = pos_mask * (-1000)

        # calculate the standard log contrastive loss for each vmf sample ([batch])
        log_infonce_per_example = (sim * pos_mask + pos_mask_add).logsumexp(-1) - (sim * neg_mask + neg_mask_add).logsumexp(-1)

        # Calculate loss ([1])
        log_infonce = torch.mean(log_infonce_per_example)
        return -log_infonce

class MySupervisedInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

        self.temp = temperature

    def forward(self, view1,view2, target):

        view1 = nn.functional.normalize(view1, dim=-1)
        view2 = nn.functional.normalize(view2, dim=-1)

        # Calculate similarities
        sim = (view1 @ view2.T) / self.temp

        # unlike og implementation we dont want to mask out diagonals since they are from two different modalities in our case
        match_mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()

        # print(match_mask)
        # print(sim)

        log_infonce = nn.functional.binary_cross_entropy_with_logits(sim,match_mask)
        
        return log_infonce

# binary versions of the loss
class SoftMCCLoss(nn.Module):

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        #i added this
        preds = torch.softmax(preds, dim=1)
        
        tp = torch.sum(preds * labels)
        tn = torch.sum((1 - preds) * (1 - labels))
        fp = torch.sum(preds * (1 - labels))
        fn = torch.sum((1 - preds) * labels)

        numerator = tp * tn - fp * fn
        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
        soft_mcc = numerator / denom

        loss = 1 - soft_mcc
        return loss


class SoftMCCWithLogitsLoss(SoftMCCLoss):

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        preds_sigmoid = torch.sigmoid(preds)
        return super().forward(preds_sigmoid, labels)


# multi-class versions of the loss
class SoftMCCLossMulti(nn.Module):
    """With logits."""

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        # create soft confusion matrix
        preds = torch.softmax(preds, dim=1)

        # total number of correct predictions, softened by the probability of each class
        c = torch.sum(preds * labels)

        # total number of samples
        s = preds.size(0)

        # number of times each class occured in the labels
        t_k = torch.sum(labels, dim=0).to(dtype=torch.float32)

        # number of times each class was predicted
        p_k = torch.sum(preds, dim=0).to(dtype=torch.float32)

        numerator = c * s - (t_k * p_k).sum()
        denom = (
            torch.sqrt(s**2 - p_k.square().sum())
            * torch.sqrt(s**2 - t_k.square().sum())
            + 1e-8
        )

        soft_mcc = numerator / denom
        if(torch.isnan(soft_mcc)):
            soft_mcc=-1
        # print("Soft MCC: ", soft_mcc)
        # print("Denom: ", denom)
        # print("Numerator: ",numerator)
        # print("Labels: ", t_k)
        # print("Preds: ", p_k)
        return 1 - soft_mcc

class WeightedCombinedLosses(nn.Module):

    def __init__(
        self, losses: list[nn.Module], weights: Optional[list[float]] = None
    ) -> None:
        super().__init__()
        self.losses = losses
        # equal weights if not provided
        self.weights = (
            weights
            if weights is not None
            else [1 / len(self.losses)] * len(self.losses)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=preds.device)
        for w, l in zip(self.weights, self.losses):
            loss += w * l(preds, targets)

        return loss



import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef, f1_score, balanced_accuracy_score

from utils.utils import device, calculate_error


def predict_logits(model, loader, n_class):
    multi_out = model.multi_out
    joint_logits = []
    histo_logits=[]
    mri_logits=[]
    all_labels = []

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch_idx, (data, label) in enumerate(loader):
            if(label.item()==99):
                continue
            data= [None if d is None else d.to(device,non_blocking=True) for d in data]

            out = model(*data)
            # print(out)
            if(multi_out):
                jl= out[2].detach().cpu()
                histo_logits.append(out[0].detach().cpu())
                mri_logits.append(out[1].detach().cpu())
            else:
                jl = out.detach().cpu()
            joint_logits.append(jl)
            all_labels.append(label)

    joint_logits = torch.cat(joint_logits,dim=0)
    if(multi_out):
        histo_logits = torch.cat(histo_logits,dim=0)
        mri_logits = torch.cat(mri_logits,dim=0)
    else:
        histo_logits=None; mri_logits=None
    all_labels = torch.LongTensor(all_labels)
    return (histo_logits,mri_logits,joint_logits), all_labels

def compute_summary_stats(model, loader, n_class):
    print("computing summary metrics")
    model.eval()
    case_ids = loader.dataset.case_data['case_id']
    patient_results = {}
    #patient_results.update({case_id: {'prob': probs, 'label': label.item()}})

    (_,_,joint_logits), all_labels = predict_logits(model,loader,n_class)
    all_probs = F.softmax(joint_logits,dim=-1).numpy()
    all_labels = all_labels.numpy()
    acc_logger, metrics = calc_all_metrics(all_probs,all_labels,n_class)

    return patient_results, acc_logger, metrics

def calc_val_metrics(probs,labels,n_class):
    acc_logger = Accuracy_Logger(n_classes=n_class)
    preds = np.argmax(probs,axis=-1)
    acc_logger.log_batch(preds,labels)
    error = 1. - (np.sum(preds==labels)/len(labels))
    return acc_logger, error

def calc_all_metrics(probs,labels,n_class):
    acc_logger, test_error = calc_val_metrics(probs,labels,n_class)
    probs/=np.sum(probs,axis=-1,keepdims=True)
    try:
        if n_class == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class='ovr')
    except ValueError as e:
        print(e)
        auc=0.
    preds = np.argmax(probs,axis=-1)
    ba=balanced_accuracy_score(labels,preds)
    mcc=matthews_corrcoef(labels,preds,sample_weight=None)

    return acc_logger, (test_error, auc, ba, mcc)


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_metric, model, ckpt_name = 'checkpoint.pt'):

        score = -val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation metric decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class ResultsTracker:
    def __init__(self,expected_columns):
        self.column_names=expected_columns
        self.results_list=[]

    def add(self,fold_name,fold_results,model_name="NA"):
        acc_logger, (error, auc, ba, mcc) = fold_results
        acc=1-error
        class_accs=[]
        for i in range(acc_logger.n_classes):
            c_acc, correct, count = acc_logger.get_summary(i)
            class_accs.append(round(c_acc,3))
        self.results_list.append((model_name,fold_name,round(acc,3),*class_accs,round(auc,3),round(ba,3),round(mcc,3)))

    def make_df(self, summary_stats=True):
        results_df = pd.DataFrame(self.results_list,columns=self.column_names)
        if(summary_stats):
            averages = np.round(results_df.mean(numeric_only=True),3)
            stds = np.round(results_df.std(ddof=0,numeric_only=True),3)
            results_df=pd.concat([results_df,averages.to_frame().T,stds.to_frame().T], ignore_index=True)
            row_names = [str(i) for i in range(len(self.results_list))]+ ["Avg","StdDev"]
            results_df['Fold']=pd.Series(row_names)
        return results_df

    def __len__(self):
        return len(self.results_list)

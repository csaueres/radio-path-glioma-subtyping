import numpy as np
import torch
import torch.nn.functional as F

import os
import argparse
from utils.utils import *
from dataio.mm_dataset import get_dataset
from model.common import get_classifier

from model.evaluation import ResultsTracker, calc_all_metrics, predict_logits


def get_multihead_results(model,loader,n_class,n_heads):
    logits_by_head, labels = predict_logits(model,loader,n_class)
    probs_by_head = []
    if(n_heads>1):
        for i in range(n_heads):
            if(logits_by_head[i] is None):
                probs_by_head.append(None)
                continue 
            probs = F.softmax(logits_by_head[i], dim=1).numpy()
            probs_by_head.append(probs)
    else:
        probs = F.softmax(logits_by_head[2], dim=1).numpy()
        probs_by_head=[probs]
    return probs_by_head, labels.numpy()

def get_results_for_fold(model, eval_data, f, ckpt_dir,splits_dir,n_class,n_heads):
    print("Fold ",f)
    ckpt_path = os.path.join(ckpt_dir, "s_{}_checkpoint.pt".format(f))
    model.load_state_dict(torch.load(ckpt_path))
    # cph = os.path.join(ckpt_dir,"resub","histo_mamba_class_mm-dino_r1_tcga171","s_{}_checkpoint.pt".format(f))
    # cpm = os.path.join(ckpt_dir,"resub","mri_mamba_class_mm-dino_r1_tcga171","s_{}_checkpoint.pt".format(f))
    # model.histo_model.load_state_dict(torch.load(cph))
    # model.mri_model.load_state_dict(torch.load(cpm))
    model.to(device)
    model.eval()
    #print_network(model)
    if(splits_dir==None):
        split = eval_data.return_as_eval_dataset()
    else:
        _,split,_ = eval_data.return_splits(os.path.join(splits_dir,f"split_{f%5}.csv"))
    dataloader = get_split_loader(split,training=False)

    metrics_per_head=[]
    head_probs, labels = get_multihead_results(model,dataloader,n_class,n_heads)
    for head_r in head_probs:
        metrics = calc_all_metrics(head_r,labels,n_class) if head_r is not None else None
        metrics_per_head.append(metrics)

    return metrics_per_head

#used when only evaluating one model
def eval_and_print_results(args,dataset):
    model, _,_ = get_classifier(args)
    results_trackers = [ResultsTracker(["Model","Fold","Accuracy","Class 0 Acc.", "Class 1 Acc.", "Class 2 Acc.", "AUC","BA","MCC"]) for i in range(args.n_heads)]
    for f in range(args.k):
        metrics_listed_by_head = get_results_for_fold(model,dataset,f,args.checkpoint_dir,args.split_dir,args.n_class,args.n_heads)
        for i in range(args.n_heads):
            m = metrics_listed_by_head[i]
            results_trackers[i].add(f+1,m,args.model_type)
    head_names={0:'histo',1:'mri',2:'joint'}
    for i in range(args.n_heads):
        results_df = results_trackers[i].make_df(summary_stats=True)
        print("Head ",i)
        print(results_df.to_string())
        out_path = os.path.join(args.checkpoint_dir,f"results_{head_names[i]}.csv")
        results_df.to_csv(out_path)


#Computes Results for all model checkpoints in a directory and saves as csv
def eval_and_save_results_all(args,dataset):
    ckpt_dir = args.checkpoint_dir
    model_checkpoint_dirs = [ f.name for f in os.scandir(ckpt_dir) if f.is_dir() ]
    #model_checkpoint_dirs = ['late-fusion3h_mamba_unpaired_final']
    print(model_checkpoint_dirs)
    results_trackers = [ResultsTracker(["Model","Fold","Accuracy","Class 0 Acc.", "Class 1 Acc.", "Class 2 Acc.", "AUC","BA","MCC"]) for i in range(args.n_heads)]
    for model_name in model_checkpoint_dirs:
        model_type='_'.join(model_name.split('_')[:2])
        print(model_name)
        args.model_type=model_type
        model, _,_ = get_classifier(args)
        ckpt_path = os.path.join(ckpt_dir,model_name)
        for f in range(args.k):
            metrics_listed_by_head = get_results_for_fold(model,dataset,f,ckpt_path,args.split_dir,args.n_class,args.n_heads)
            for i in range(args.n_heads):
                m = metrics_listed_by_head[i]
                if(m is not None):
                    results_trackers[i].add(f+1,m,model_name)
    
    head_names={0:'histo',1:'mri',2:'joint'}
    for i in range(args.n_heads):
        out_path = os.path.join(ckpt_dir,f"results_{head_names[i]}.csv")
        print(f"Results saved to {out_path}")
        results_df = results_trackers[i].make_df(summary_stats=False)
        results_df.to_csv(out_path)






parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--checkpoint_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--case_csv', type=str, default=None, 
                    help='csv of items to be evaluated on')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='splits')
parser.add_argument('--histo_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--mri_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--histo_csv', type=str, default='')
parser.add_argument('--mri_csv', type=str, default='')
parser.add_argument('--histo_embed_dim', type=int, default=1536)
parser.add_argument('--mri_embed_dim', type=int, default=1536,help='size of the MRI patch/slice embeddings')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--patch_frac',type=float,default=1.0,help='random patches per case')
parser.add_argument('--mri_context_width',type=int,default=0,help='number of additional slices of mri centered around center of mass (max 5)')
parser.add_argument('--model_type', type=str,default="moe-mamba")
parser.add_argument('--mri_embedder', type=str, default='default', 
                    help='options=[default,]. Different preprocessing pipelines for different MRI encoders.')
parser.add_argument('--n_block', type=int, default=24,help='number of mamba blocks to include in the model')
parser.add_argument('--n_heads', type=int, default=1, help='How many output scenarios to evaluate model on. Use 1 for unimodal models and 3 for bimodal models.')
parser.add_argument('--load_data_in_mem', action='store_true', default=False, help='whether to load all WSI features in memory (requires large amounts of RAM)')
parser.add_argument('--task', type=str,default='idh_1p19q_class', help='classification task to perform.')
args = parser.parse_args()



if __name__ == "__main__":
    args.n_class=3
    args.drop_out=0.0
    args.return_attn=False

    if args.task == 'idh_1p19q_class':
        args.n_class=3
        args.label_dict = {'gbm':0, 'astro':1, 'oligo':2}
    elif args.task == '5way_class':
        args.n_class=5
        args.label_dict = {'GBM_MES':0,'GBM_RTK1':1,'GBM_RTK2':2,'ASTRO':3,'OLIGO':4}
    else:
        raise NotImplementedError("Please select either idh_1p19q_class or 5way_class or define a new task.")

    if(args.model_type=='all'):
        args.load_data_in_mem=True
        dataset = get_dataset(args)
        eval_and_save_results_all(args,dataset)
    else:
        dataset = get_dataset(args)
        eval_and_print_results(args,dataset)


    






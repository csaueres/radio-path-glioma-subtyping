import argparse
import os
import numpy as np
import torch

# internal imports
from model.common import train
from model.evaluation import ResultsTracker
from dataio.mm_dataset import get_dataset



def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    folds = np.arange(args.k_start, args.k)
    results_tracker = ResultsTracker(["Model","Fold","Accuracy","GBM Acc.", "Astro Acc.", "Oligo Acc.", "AUC","BA","MCC"])
    for i in folds:
        #if i>5 (the number of splits, then it will cycle)
        j = i%5
        train_dataset, val_dataset, test_dataset = dataset.return_splits(csv_path='{}/split_{}.csv'.format(args.split_dir, j))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        try:
            results,acc_logger, metrics  = train(datasets, i, args)
        except UnboundLocalError as e:
            #skip this fold
            print(e)
            print("Fold didn't converge but skipping")
            continue

        results_tracker.add(i+1,(acc_logger, metrics),args.model_type)

    final_df = results_tracker.make_df()
    save_name = 'results.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name),index=False)

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for multimodal Training')
parser.add_argument('--histo_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--mri_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--histo_embed_dim', type=int, default=1536,help='size of the histology patch embeddings')
parser.add_argument('--mri_embed_dim', type=int, default=1536,help='size of the MRI patch/slice embeddings')
parser.add_argument('--max_epochs', type=int, default=30,
                    help='maximum number of epochs to train (default: 30)')
parser.add_argument('--n_block', type=int, default=24,help='number of mamba blocks to include in the model')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=0.01,
                    help='weight decay (default: 0.01)')
parser.add_argument('--seed', type=int, default=7, 
                    help='random seed for reproducible experiment (default: 7)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=0, help='what fold to start at')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--case_csv', type=str, default=None,help='case csv. all cases with labels, regardless of which modalities are present')
parser.add_argument('--histo_csv', type=str, default=None)
parser.add_argument('--mri_csv', type=str, default=None)
parser.add_argument('--split_dir', type=str, default=None, 
                    help='specify the directory storing the splits to use')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.5, help='dropout')
parser.add_argument('--model_type', type=str, default='moe-mamba', 
                    help='type of model. see common for options')
parser.add_argument('--mri_embedder', type=str, default='default', 
                    help='options=[default,]. Different preprocessing pipelines for different MRI encoders.')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str,default='idh_1p19q_class', help='classification task to perform (defined below).')
parser.add_argument('--patch_frac',type=float,default=1.0,help='what fraction of random patches to use for each image for training')
parser.add_argument('--mri_context_width',type=int,default=0,help='how many additional slices to include on each side of the center slice (0 is only center slice)')
parser.add_argument('--load_data_in_mem', action='store_true', default=False, help='whether to load all Histo Features in memory (requires 100GB+ RAM)')
parser.add_argument('--unpaired_train', action='store_true', default=False, help='whether to train the model separately for each modality')
args = parser.parse_args()
args.return_attn=False
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'experiment': args.exp_code,
            'task': args.task,
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'reg': args.reg,
            'seed': args.seed,
            'n_block': args.n_block,
            "drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

print('\nLoad Dataset')


if args.task == 'idh_1p19q_class':
    args.n_class=3
    label_dict = {'gbm':0, 'astro':1, 'oligo':2}
    dataset = get_dataset(args,label_dict)
elif args.task == '5way_class':
    args.n_class=5
    label_dict = {'GBM_MES':0,'GBM_RTK1':1,'GBM_RTK2':2,'ASTRO':3,'OLIGO':4}
    dataset = get_dataset(args,label_dict)               
else:
    raise NotImplementedError("Please select either idh_1p19q_class or 5way_class or define a new task.")
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")



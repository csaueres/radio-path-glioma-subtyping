import numpy as np
import torch
import os
from time import time

from model.modules.FusionModels import *
from model.modules.UnimodalModels import *
from model.modules.MixtureModel import MoE
from utils.custom_losses import SoftMCCLoss, SoftMCCLossMulti, WeightedCombinedLosses

from utils.utils import *
from model.evaluation import compute_summary_stats, EarlyStopping

#sets up the model and a loss function. Ignore the loss function at eval time. For eval the checkpoint needs to be loaded in a subsequent step
def get_classifier(args):
    if(args.n_class==2):
        losses = [torch.nn.BCEWithLogitsLoss(),
                    SoftMCCLoss(),]
    else:
        losses = [torch.nn.CrossEntropyLoss(reduction='mean'),
                    SoftMCCLossMulti(),]
    weights = [1.0, 1.0]
    classification_loss = WeightedCombinedLosses(losses, weights)
    contrastive_loss = None
    if(args.model_type=='early-fusion_mamba'):
        model = SequenceEarlyFusion(histo_dim = args.histo_embed_dim, mri_dim=args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out, n_layer=args.n_block)
    elif(args.model_type=='early-fusion_mlp'):
        model = LinearEarlyFusion(histo_dim = args.histo_embed_dim, mri_dim=args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out)
    # elif(args.model_type=='linear1h'):
    #     model = LinearFusion(in_dim = args.embed_dim, n_classes=args.n_class, dropout=args.drop_out)
    elif(args.model_type=='late-fusion3h_mlp'):
        model = LinearTripleHead(histo_dim = args.histo_embed_dim, mri_dim=args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out)
    elif(args.model_type=='late-fusion3h_mamba'):
        model = MambaTripleHead(histo_dim = args.histo_embed_dim, mri_dim=args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out, n_layer=args.n_block)
    elif(args.model_type=='logit-fusion_mamba'):
        model = LogitLevelFusion(histo_dim = args.histo_embed_dim, mri_dim=args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out, n_layer=args.n_block)
    elif(args.model_type=='moe_mlp'):
        model = MoE(histo_dim = args.histo_embed_dim, mri_dim=args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out,n_layer=1, linear=True)
    elif(args.model_type=='moe_mamba'):
        model = MoE(histo_dim = args.histo_embed_dim, mri_dim=args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out, n_layer=args.n_block, linear=False)
    elif(args.model_type=='histo_mamba'):
        model = HistoOnlyMamba(in_dim = args.histo_embed_dim, n_classes=args.n_class, dropout=args.drop_out, n_layer=args.n_block)
    elif(args.model_type=='mri_mamba'):
        model = MRIOnlyMamba(in_dim = args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out, n_layer=args.n_block)
    elif(args.model_type=='histo_mlp'):
        model = HistoMLP(in_dim = args.histo_embed_dim, n_classes=args.n_class, dropout=args.drop_out)
    elif(args.model_type=='mri_mlp'):
        model = MRIMLP(in_dim = args.mri_embed_dim, n_classes=args.n_class, dropout=args.drop_out)
    else:
        raise NotImplementedError("No such model type :", args.model_type)
    return model, classification_loss, contrastive_loss


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))


    print('\nInit Model...', end=' ')
    model, classification_loss_fn, contrastive_loss_fn = get_classifier(args)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    early_stopping = EarlyStopping(patience = 5, stop_epoch=15, verbose = True)

    if(args.model_type=='moe_mamba' or args.model_type=='moe_mlp'):
        from model.train_utils_moe import train_loop, validate
    else:
        from model.train_utils_vanilla import train_loop, validate

    for epoch in range(args.max_epochs):
        train_loop(epoch,args.unpaired_train, model, train_loader, optimizer, args.n_class, writer, classification_loss_fn, contrastive_loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_class, 
            early_stopping, writer, classification_loss_fn, args.results_dir)
        
        if stop: 
            break

    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))

    results_dict, acc_logger,metrics = compute_summary_stats(model, val_loader, args.n_class)
    val_error, val_auc, val_ba, val_mcc = metrics
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    for i in range(args.n_class):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))


    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.close()
    return results_dict,acc_logger, metrics

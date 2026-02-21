import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import calculate_error,print_train_results,device
from model.evaluation import Accuracy_Logger

from model.train_utils_vanilla import calc_loss_multi_head, calc_loss_one_head
from model.train_utils_vanilla import  validate as vanilla_validate

UNMIXED_WARMUP=5


def train_loop(epoch,unpaired_train, model, loader, optimizer,n_class, writer=None,classification_loss_fn = None,contrastive_loss_fn=None):
    model.train()
    model.multi_out=True
    acc_logger = Accuracy_Logger(n_classes=n_class)
    train_error=0
    train_loss=0
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for step, (data, label) in enumerate(loader):

            data= [d.to(device,non_blocking=True) for d in data]
            labels_one_hot = torch.nn.functional.one_hot(label,n_class).float().to(device)

            #trains each expert on one modality individually
            if(epoch<UNMIXED_WARMUP):
                out_all = model(*data,mixing=False)
                loss=calc_loss_multi_head(classification_loss_fn,out_all[0:2],labels_one_hot)
            else:
                out_all = model(*data,mixing=True)
                if(unpaired_train):
                    #ignores joint output for loss
                    loss=calc_loss_multi_head(classification_loss_fn,out_all[0:2],labels_one_hot)
                else:
                    loss=calc_loss_multi_head(classification_loss_fn,out_all,labels_one_hot)
            #only for visualization purposes, doesn't optimize model
            out_joint = out_all[-1]
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

            Y_hat = torch.topk(out_joint.detach().cpu(), 1, dim=1)[1].squeeze(1)
            acc_logger.log_batch(Y_hat, label)
            error = calculate_error(Y_hat, label)
            train_error += error

    train_loss /=len(loader)
    train_error /= len(loader)
    print_train_results(epoch,train_loss,train_error,acc_logger,writer)


def validate(cur, epoch, model, loader, n_class, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    #model.multi_out=False
    return vanilla_validate(cur, epoch, model, loader, n_class, early_stopping, writer, loss_fn, results_dir)

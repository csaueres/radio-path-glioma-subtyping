import numpy as np
import torch
import torch.nn.functional as F
import os

from model.evaluation import Accuracy_Logger,predict_logits,calc_val_metrics
from sklearn.metrics import balanced_accuracy_score

from utils.utils import calculate_error,print_train_results,device



def train_loop(epoch,unpaired_train, model, loader, optimizer,n_class, writer=None,classification_loss_fn = None,contrastive_loss_fn=None):
    multi_out = model.multi_out
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_class)
    train_error=0
    train_loss=0
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for step, (data, label) in enumerate(loader):

            data= [d.to(device,non_blocking=True) for d in data]
            labels_one_hot = torch.nn.functional.one_hot(label,n_class).float().to(device)

            out_all = model(*data)

            if(multi_out):
                if(unpaired_train):
                    loss=calc_loss_multi_head(classification_loss_fn,out_all[0:2],labels_one_hot)
                else:
                    loss=calc_loss_multi_head(classification_loss_fn,out_all,labels_one_hot)
                #joint output is expected to be last. not used for optimization, only printing
                out_joint = out_all[-1]
            else:
                loss=calc_loss_one_head(classification_loss_fn,out_all,labels_one_hot)
                out_joint = out_all
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

def calc_loss_one_head(criterion,logits,labels):
    return criterion(logits,labels)

def calc_loss_multi_head(criterion,out,labels):
    loss = torch.zeros(1, device=labels.device)
    for head_logits in out:
        loss+=criterion(head_logits,labels)
    total_loss = loss
    return total_loss

def validate(cur, epoch, model, loader, n_class, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    
    loader.dataset.histo_fetcher.toggle_eval(True)
    (histo_logits,mri_logits,joint_logits), all_labels = predict_logits(model,loader,n_class)
    loader.dataset.histo_fetcher.toggle_eval(False)

    all_labels_one_hot = torch.nn.functional.one_hot(all_labels,n_class).float()
    val_loss = loss_fn(joint_logits,all_labels_one_hot).item()
    all_probs = F.softmax(joint_logits,dim=-1).numpy()
    acc_logger, val_error =calc_val_metrics(all_probs,all_labels.numpy(),n_class)
    
    if writer:
        writer.add_scalar('val/joint_loss', val_loss, epoch)
        #writer.add_scalar('val/ba', ba, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        if(model.multi_out):
            histo_only_loss = loss_fn(histo_logits,all_labels_one_hot).item()
            mri_only_loss = loss_fn(mri_logits,all_labels_one_hot).item()
            writer.add_scalar('val/h_loss', histo_only_loss, epoch)
            writer.add_scalar('val/m_loss', mri_only_loss, epoch)


    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}'.format(val_loss, val_error))
    for i in range(n_class):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False
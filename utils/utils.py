import pickle
import torch
import numpy as np
import torch.nn as nn

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_multimodal(batch):
	n_modalities=len(batch[0])
	mod1 = torch.cat([item[0] for item in batch], dim = 0)
	#mod1 = torch.nested.nested_tensor([item[0] for item in batch], layout=torch.jagged)
	mod2 = torch.cat([item[1] for item in batch], dim = 0)
	label = torch.LongTensor([item[2] for item in batch])
	return (mod1,mod2), label


#assumes batch size of 1
#can also handle unimodal case
def collate_val(batch):
	case = batch[0]
	label = torch.LongTensor([case[-1]])
	return case[:-1], label


def get_split_loader(split_dataset, training = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4, 'pin_memory':True} if device.type == "cuda" else {}
	if training:
		if weighted:
			weights = make_balanced_sample_weights(split_dataset)
			loader = DataLoader(split_dataset, batch_size=32,drop_last=True, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_multimodal, **kwargs)	
		else:
			loader = DataLoader(split_dataset, batch_size=32,drop_last=True, sampler = RandomSampler(split_dataset), collate_fn = collate_multimodal, **kwargs)
			#loader = DataLoader(split_dataset, batch_size=64,drop_last=True, sampler = RandomSampler(split_dataset), collate_fn = collate_possibly_missing_multimodal, **kwargs)
	else:
		#loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_multimodal, **kwargs)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_val, **kwargs)

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	#print(optimizer.state_dict()['param_groups'])
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_balanced_sample_weights(dataset):
	case_labels = []		
	for i in range(3):
		case_labels.append(np.where(dataset.case_data['label'] == i)[0])
	N = float(len(dataset))                                           
	# weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]       
	weight_per_class = [N/len(case_labels[c]) for c in range(len(case_labels))]                                                                                                 
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = int(dataset.get_label_by_idx(idx))                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def print_train_results(epoch,loss,error,acc_logger,writer):

    print('\nEpoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, loss, error))
    for i in range(acc_logger.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/error', error, epoch)


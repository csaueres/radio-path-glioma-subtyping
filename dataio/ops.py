import os
import numpy as np
import pickle
import h5py
import torch

import pandas as pd
import random
import time

def load_histo_features(data_dir,slide_id,embedder,aug):
	full_path = os.path.join(data_dir,aug,'h5_files','{}.h5'.format(slide_id))
	try:
		with h5py.File(full_path,'r') as hdf5_file:
			features = hdf5_file['features'][:]
			coords = None #hdf5_file['coords'][:]
	except Exception as e:
		print(f"caught Exception on slide {slide_id}: {e}")
		raise FileExistsError(f"file {full_path} likely doesnt exist")

	return features

def preload_histo_features(data_dir,samples,embedder,aug):
	t0=time.time()
	feature_dic = {}
	# exp_dic = {}
	i=0
	for s in samples:
		feats = load_histo_features(data_dir,s,embedder,aug)
		feature_dic[s]=feats
		# exp_dic[s]=torch.as_tensor(feats,dtype=torch.bfloat16)
		i+=1
		if(i%100==0):
			print(i)
			print(round(calc_mem_size(feature_dic,'n')/1000000000,3), ' GB')

	print(f"{len(feature_dic)} slide features loaded into memory")
	print(f"Feature Loading took {time.time()-t0} seconds")
	return feature_dic

def calc_mem_size(dic,l='n'):
	sum=0
	if(l=='n'):
		for t in dic.values():
			sum+=t.nbytes
	else:
		for t in dic.values():
			sum+=t.element_size() * t.nelement()
	return sum

#store all embeddings in memory
def preload_mri_features(data_dir,samples,embedder='mm-dino',n_surround_slices=0):
	print("MRI Embedder: ", embedder)
	feature_dic = {}
	if(embedder=='none'):
		return
	elif(embedder=='mm-dino'):
		handle = torch.load(os.path.join(data_dir,"daniel_tumor_embeds.pth"))

		subjects = handle["id"]['train'] + handle["id"]['val']+ handle["id"]['test']
		patch_and_tumor_embeds = torch.cat([handle["features"]['train'],handle["features"]['val'],handle["features"]['test']],dim=0)
		print("Shape:", patch_and_tumor_embeds.shape)
		patch_features = patch_and_tumor_embeds[:,0:,:]
		tumor_embeddings=torch.cat([patch_and_tumor_embeds[:,0,:],torch.mean(patch_features,dim=1)],dim=-1)
		#print("Slice Embeddings: ", tumor_embeddings.shape)

		assert(len(subjects)==len(patch_features))
		for i in range(len(subjects)):
			sample_id = subjects[i]
			if(sample_id in samples):
				#feature_dic[sample_id]=tumor_embeddings[i]
				feature_dic[sample_id]=patch_features[i]

	elif(embedder=='mm-dino-multislice'):
		handle = torch.load(os.path.join(data_dir,"mm-dino_multislice.pth"))
		subjects = handle["id"]['train'] + handle["id"]['val']+ handle["id"]['test']
		patch_and_tumor_embeds = torch.cat([handle["features"]['train'],handle["features"]['val'],handle["features"]['test']],dim=0)
		print("Shape:", patch_and_tumor_embeds.shape)

		#only center slice
		if(n_surround_slices==0):
			#samples,slices,patches,features
			patch_features = patch_and_tumor_embeds[:,5,:,:]
		else:
			patch_features = patch_and_tumor_embeds[:,5-n_surround_slices:6+n_surround_slices,:,:]
			patch_features = torch.flatten(patch_features,1,2)

		assert(len(subjects)==len(patch_features))
		for i in range(len(subjects)):
			sample_id = subjects[i]
			if(sample_id in samples):
				feature_dic[sample_id]=patch_features[i]

	elif(embedder=='brainiac'):
		handle = np.load(os.path.join(data_dir,"brainiac_features_t1c.npz"))
		modality_extensions = ["_t1","_t1ce","_t2","_flair"]
		subjects = handle['subjects'][:,0]
		token_and_cls_embeds = handle['features']
		patch_features = torch.as_tensor(token_and_cls_embeds)
		print(patch_features.shape)
		#currently only adding flair sequences, since they are last
		for i in range(len(subjects)):
			sample_id = subjects[i]
			# print(sample_id)
			sample_id = sample_id.split('_')[0]
			if(sample_id in samples):
				#could also make list and combine them later, imo better to do in gen if works at all
				feature_dic[sample_id]=patch_features[i]
		print(feature_dic.keys())
	else:
		raise (NotImplementedError("Unrecognized embedder {embedder}"))
	print(f"MRI Features Foootprint: {calc_mem_size(feature_dic,l='n')/1000000} MB")
	return feature_dic


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


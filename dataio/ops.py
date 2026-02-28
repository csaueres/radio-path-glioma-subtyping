import os
import numpy as np
import pickle
import h5py
import torch
import time

def load_histo_features(data_dir,slide_id,aug):
	full_path = os.path.join(data_dir,aug,'h5_files','{}.h5'.format(slide_id))
	try:
		with h5py.File(full_path,'r') as hdf5_file:
			features = hdf5_file['features'][:]
	except Exception as e:
		print(f"caught Exception on slide {slide_id}: {e}")
		raise FileExistsError(f"file {full_path} likely doesnt exist")

	return features

def preload_histo_features(data_dir,samples,aug):
	t0=time.time()
	feature_dic = {}
	i=0
	for s in samples:
		feats = load_histo_features(data_dir,s,aug)
		feature_dic[s]=feats
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
	elif(embedder=='default' or embedder=='mm-dino'):
		handle = torch.load(os.path.join(data_dir,embedder,"mri_features.pth"))
		#possibly given in this format
		try:
			subjects = handle["id"]['train'] + handle["id"]['val']+ handle["id"]['test']
			patch_and_tumor_embeds = torch.cat([handle["features"]['train'],handle["features"]['val'],handle["features"]['test']],dim=0)
		except Exception as e:
			subjects = handle["ids"]
			patch_and_tumor_embeds = handle["features"]
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
		handle = torch.load(os.path.join(data_dir,embedder,"mri_features.pth"))
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


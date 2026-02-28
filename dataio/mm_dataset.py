import os
import numpy as np
import pandas as pd
import random

from torch.utils.data import Dataset
from torch import from_numpy, as_tensor
import h5py

from dataio.ops import *

def get_dataset(args):
    if(args.histo_csv!='' and args.mri_csv!=''):
        dataset = MultimodalDataset(case_csv = args.case_csv,
                    histo_csv=args.histo_csv,
                    mri_csv=args.mri_csv,
                    histo_dir= args.histo_root_dir,
                    mri_dir= args.mri_root_dir,
                    mri_embedder= args.mri_embedder,
                    label_dict = args.label_dict,
                    histo_patch_frac=args.patch_frac,
                    mri_slices=args.mri_context_width,
                    in_memory=args.load_data_in_mem)
    elif(args.histo_csv!=''):
        dataset = UnimodalDataset(
                    modality = 'histo',
                    case_csv = args.case_csv,
                    mod_csv=args.histo_csv,
                    mod_dir= args.histo_root_dir,
                    patch_frac = args.patch_frac,
                    mod_embedder= 'gigapath_20x',
                    label_dict = args.label_dict,
                    in_memory=args.load_data_in_mem
        )
    elif(args.mri_csv!=''):
        dataset = UnimodalDataset(
                    modality = 'mri',
                    case_csv = args.case_csv,
                    mod_csv=args.mri_csv,
                    mod_dir= args.mri_root_dir,
                    patch_frac = args.patch_frac,
                    mod_embedder= args.mri_embedder,
                    label_dict = args.label_dict,
                    in_memory=True
        )
    return dataset



class ModalityFetcher(Dataset):
    def __init__(self,sample_csv,data_dir,embedder,patch_frac,all_samples_per_case=False,in_memory=False):
        self.sample_df = pd.read_csv(sample_csv,dtype={'case_id': str})
        self.sample_df.set_index('case_id',inplace=True,drop=True)
        self.sample_df.sort_index(inplace=True)
        self.cases = self.sample_df.index.unique().to_list()
        self.data_dir=data_dir
        self.rng = np.random.default_rng()
        self.all_samples_per_case=all_samples_per_case
        self.in_memory=in_memory
        self.embedder=embedder
        self.patch_frac=patch_frac
        self.eval_mode=False

    def get_by_cid(self,cid):
        samples_for_cid = self.sample_df.loc[cid]['sample_id']
        if(isinstance(samples_for_cid,str)):
            selected_sample = samples_for_cid
            feats = self._get_by_sid(selected_sample)
        else:
            if(self.all_samples_per_case):
                combined_feats = [self._get_by_sid(s) for s in samples_for_cid]
                feats = torch.cat(combined_feats,dim=-2)
            else:
               selected_sample = random.choice(samples_for_cid.to_list())
               feats = self._get_by_sid(selected_sample)
        # print("Selected sample", selected_sample)
        #during validation want to always use all patches
        if(self.patch_frac!=1.0 and not self.eval_mode):
            feats = self._sample_patches(feats)
        return feats.unsqueeze(0)
    

    #gets item by sid. internal method, should be overriden based on modality
    def _get_by_sid(self,sid):
        print("Getting ",sid)
        pass

    def contains(self,cid):
        return cid in self.cases

    def toggle_eval(self,b):
        self.eval_mode=b

    def __getitem__(self,cid):
        return self.get_by_cid(cid)

class HistoFetcher(ModalityFetcher):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        print(f"Sampling {self.patch_frac} patches")
        self.aug='og'
        print(f"Sampling {self.patch_frac} patches per slide")
        if(self.in_memory):
            all_slide_ids =self.sample_df['sample_id'].values
            self.preloaded_slide_features = preload_histo_features(self.data_dir,all_slide_ids,self.aug)

    def _get_by_sid(self,sid):
        if(self.in_memory):
            histo_features = self.preloaded_slide_features[sid]
        else:
            histo_features = load_histo_features(self.data_dir,sid,self.aug)
        histo_features = as_tensor(histo_features)
        return histo_features

    def _sample_patches(self,features):
        assert(len(features.shape)<3)
        n_patch = features.shape[0]
        if(self.patch_frac<1.0):
            subset_indices=np.sort(self.rng.choice(n_patch,int(n_patch*self.patch_frac),replace=False))
            features = features[subset_indices]
        #coords = coords[subset_indices]
        elif(self.patch_frac>1):
            subset_indices=self.rng.choice(n_patch,int(self.patch_frac),replace=True)
            features = features[subset_indices]
        return features

    

class MriFetcher(ModalityFetcher):
    def __init__(self,*args,**kwargs,):
        super().__init__(*args,**kwargs)
        samples=self.sample_df['sample_id'].values
        if(self.in_memory):
            self.all_mri_features=preload_mri_features(self.data_dir,samples,self.embedder,self.patch_frac)
        else:
            raise NotImplementedError("MRI features currently always loaded into mem")
        # print("MRI Samples")
        # print(self.all_mri_features.keys())
        # print(self.cases)

    def _get_by_sid(self,sid):
        mri_features = as_tensor(self.all_mri_features[sid])
        return mri_features

    def _sample_patches(self,features):
        return features


class MultimodalDataset(Dataset):
    def __init__(self,case_csv,label_dict,histo_csv,histo_dir,histo_patch_frac,mri_csv,mri_dir,mri_embedder,mri_slices,in_memory):
        case_data = pd.read_csv(case_csv, dtype={'case_id': str})
        self.mri_fetcher=MriFetcher(mri_csv,mri_dir,mri_embedder,mri_slices,all_samples_per_case=False,in_memory=True)
        self.histo_fetcher=HistoFetcher(histo_csv,histo_dir,'gigapath_20x',histo_patch_frac,all_samples_per_case=True,in_memory=in_memory)
        print("Histo Dataset Loaded in Memory: ",in_memory)
        self.case_data = df_prep(case_data,label_dict)
        #print(self.case_data.to_string())
        self.mod_str = 'mh'
        

    def return_splits(self, csv_path=None):
        all_splits = pd.read_csv(csv_path, dtype=self.case_data['case_id'].dtype)
        train_split = self.get_split_from_df(all_splits, 'train')
        val_split = self.get_split_from_df(all_splits, 'val')
        return train_split, val_split, val_split

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        mask = self.case_data['case_id'].isin(split.tolist())
        df_slice = self.case_data[mask].reset_index(drop=True)
        if split_key=='train':
            split = RandomlyPairedMultimodalSplit(df_slice, self.histo_fetcher,self.mri_fetcher)
            #only works if dataset is entirely paired
            #split = PairedMultimodalSplit(df_slice, self.histo_fetcher,self.mri_fetcher)
        else:
            #split = PairedMultimodalSplit(df_slice, self.histo_fetcher,self.mri_fetcher)
            split = RandomlyPairedMultimodalSplit(df_slice, self.histo_fetcher,self.mri_fetcher)
        return split

    def return_as_eval_dataset(self):
        return PairedMultimodalSplit(self.case_data,self.histo_fetcher,self.mri_fetcher)

    def get_list(self, idxs):
        return self.case_data['case_id'][idxs]

    def get_label_by_idx(self, i):
        return self.case_data['label'][i]

    def __len__(self):
        return len(self.case_data['case_id'])



#dataframe based rather than csv
class RandomlyPairedMultimodalSplit(MultimodalDataset):
    def __init__(self, case_data, histo_fetcher,mri_fetcher):
        self.case_data=case_data
        self.histo_fetcher=histo_fetcher
        self.mri_fetcher=mri_fetcher
        self.cids_by_label = self.stratify_by_label(self.case_data,self.mri_fetcher,self.histo_fetcher)
        self.mod_str = 'mh'
        #NOTE: toggle to print case dict
        #print(self.cids_by_label)

    #returns a dict[(label,modality):list]
    #intersects the cases and the modality metadata to only use ones which are in both
    @staticmethod
    def stratify_by_label(case_data,mod1,mod2):
        grouped_cids = {}
        for l in case_data['label'].unique():
            grouped_cids[(l,'mri')]=[]; grouped_cids[(l,'histo')]=[]
        for case_id,l in zip(case_data['case_id'],case_data['label']):
            if(mod1.contains(case_id)):
                grouped_cids[(l,'mri')].append(case_id)
            if(mod2.contains(case_id)):
                grouped_cids[(l,'histo')].append(case_id)

        return grouped_cids


    def __getitem__(self,idx):
        #get next case id and label
        case_id = self.case_data.at[idx,'case_id']
        label = self.case_data.at[idx,'label']
        return *self.get_case_semi_paired(case_id,label),label
        #return *self.get_case_random_pairing(case_id,label), label
        #return *self.get_case_non_paired(case_id,label),label

    #This one will return paired samples if available
    def get_case_semi_paired(self,case_id,label):
        missing_mods=0
        if(self.histo_fetcher.contains(case_id)):
            histo_sample=self.histo_fetcher[case_id]
        else:
            random_case = random.choice(self.cids_by_label[(label,'histo')])
            histo_sample=self.histo_fetcher[random_case]
            missing_mods+=1
        if(self.mri_fetcher.contains(case_id)):
            mri_sample = self.mri_fetcher[case_id]
        else:
            random_case = random.choice(self.cids_by_label[(label,'mri')])
            mri_sample=self.mri_fetcher[random_case]
            missing_mods+=1
        # if(missing_mods>1):
        #     print(f"Warning: Case {case_id} is missing both modalities")
        return histo_sample, mri_sample

    #this one is always random, even if the case has paired data
    def get_case_random_pairing(self,case_id,label):
        if(self.histo_fetcher.contains(case_id)):
            histo_sample=self.histo_fetcher[case_id]
            random_case = random.choice(self.cids_by_label[(label,'mri')])
            mri_sample=self.mri_fetcher[random_case]
        else:
            mri_sample = self.mri_fetcher[case_id]
            random_case = random.choice(self.cids_by_label[(label,'histo')])
            histo_sample=self.histo_fetcher[random_case]
        return histo_sample, mri_sample

    #ignores label, just returns a random instance of the other modality
    #sanity check
    def get_case_non_paired(self,case_id,label):
        label=None
        mri_cases = list(set(self.mri_fetcher.cases) & set(self.case_data['case_id']))
        histo_cases = list(set(self.histo_fetcher.cases) & set(self.case_data['case_id']))
        if(self.histo_fetcher.contains(case_id)):
                histo_sample=self.histo_fetcher[case_id]
                random_case = random.choice(mri_cases)
                mri_sample=self.mri_fetcher[random_case]
        else:
            mri_sample = self.mri_fetcher[case_id]
            random_case = random.choice(histo_cases)
            histo_sample=self.histo_fetcher[random_case]
        return histo_sample, mri_sample



class PairedMultimodalSplit(MultimodalDataset):
    def __init__(self, case_data, histo_fetcher,mri_fetcher):
        paired_mask = (case_data['histo']==1) & (case_data['mri']==1)
        self.case_data=case_data[paired_mask].reset_index(drop=True, inplace=False)
        print(f"Removed {len(case_data[~paired_mask])} cases because unpaired")
        self.histo_fetcher=histo_fetcher
        self.mri_fetcher=mri_fetcher
        self.mod_str = 'mh'

    def __getitem__(self,idx):
        #get next case id and label
        case_id = self.case_data.at[idx,'case_id']
        label = self.case_data.at[idx,'label']
        histo=self.histo_fetcher[case_id]
        mri = self.mri_fetcher[case_id]
        return histo,mri,label


#only meant to be used at eval time
class UnimodalDataset(Dataset):
    def __init__(self,modality,case_csv,label_dict,mod_csv,mod_dir,patch_frac,mod_embedder,in_memory):
        case_data = pd.read_csv(case_csv)
        case_data = df_prep(case_data,label_dict)
        if(modality=='histo'):
            self.data_fetcher=HistoFetcher(mod_csv,mod_dir,mod_embedder,patch_frac,all_samples_per_case=True,in_memory=in_memory)
            #alias to maintain consistency with above
            self.histo_fetcher=self.data_fetcher
            modality_mask = case_data['histo']==1
            self.mod_str = 'h'
        elif(modality=='mri'):
            self.data_fetcher=MriFetcher(mod_csv,mod_dir,mod_embedder,patch_frac,all_samples_per_case=False,in_memory=in_memory)
            self.mri_fetcher=self.data_fetcher
            modality_mask = case_data['mri']==1
            self.mod_str = 'm'
        else:
            raise NotImplementedError("Unknown modality ",modality)

        self.case_data=case_data[modality_mask].reset_index(drop=True, inplace=False)
        print(f"Kept {sum(modality_mask)} unimodal {modality} cases out of {len(modality_mask)}")
        self.rng = np.random.default_rng()


    def get_list(self, idxs):
        return self.case_data['case_id'][idxs]

    def get_label_by_idx(self, i):
        return self.case_data['label'][i]

    def __len__(self):
        return len(self.case_data['case_id'])
    
    def __getitem__(self,idx):
        case_id = self.case_data.at[idx,'case_id']
        label = self.case_data.at[idx,'label']
        if(self.data_fetcher.contains(case_id)):
            sample=self.data_fetcher[case_id]
        else:
            print(case_id, " Missing")
            new_idx = self.rng.integers(self.__len__())
            return self.__getitem__(new_idx)
        return sample,label
        

def df_prep(data, label_dict, ignore=[], label_col='label'):
    if label_col != 'label':
        data['label'] = data[label_col].copy()

    mask = data['label'].isin(ignore)
    data = data[~mask]
    data.reset_index(drop=True, inplace=True)
    for i in data.index:
        key = data.loc[i, 'label']
        data.at[i, 'label'] = label_dict[key]
    return data
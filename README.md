# This repository accompanies *Multimodal Fusion of Pathology and Radiology Foundation Models for WHO2021 Glioma Subtyping* (under review).

## Setup

### Installation
Required packages are listed in env.yml. A conda environment can be created by running the following command:
```
conda env create -n mmgs -f env.yml
```

### Generate Histology Embeddings
Our pipeline assumes that patch embeddings have been pre-extracted and stored in .h5 files.
Tools such as [Trident](https://github.com/mahmoodlab/TRIDENT) can perform the patching and embedding.
In this work we use GigaPath embeddings ([GigaPath](https://github.com/prov-gigapath/prov-gigapath)), but any should work (you may need to adjust the embedding dimensionality).

### Generate MRI Embeddings
Similarly, our pipeline expects MRI patch embeddings as .pth files. The format should be a dictionary of format:

"features"-> tensor of shape [n,p,e], describing number of samples, number of patches, and embedding size of each patch, respectively

"ids"-> associated sample_ids in the same order. 

In our experiments we employ [MM-DINOv2](https://github.com/daniel-scholz/mm-dinov2) generated embeddings.


## Pipeline

### Required Metadata
Training a model requires 3 .csvs and a splits directory.

Case CSV: Details cases, their diagnosis, and which modalities are present.

Histology CSV: Links the case_id to the name of the associated WSIs (can be multiple).

MRI CSV: Links the case_id to the name of the associated MRI study (tested with 1:1 pairing).

See metadata for examples.

### Training Model

```
python train.py --lr 5e-5 --patch_frac 2000 --k 10 --seed 7 --max_epochs 30 --load_data_in_mem \
 --model_type <MODEL> --mri_embed_dim 768 --histo_embed_dim 1536 --exp_code run1 \
 --results_dir results --case_csv metadata/train_cases.csv --split_dir metadata/splits/train_cases_5f\
 --histo_csv metadata/histo_train.csv --mri_csv metadata/mri_train.csv --histo_root_dir data/histo_features --mri_root_dir data/mri_features
 ```

 MODEL can be any of: early-fusion_mamba, late-fusion3h_mamba, moe_mamba, histo_mamba, mri_mamba. We recommend moe_mamba. See common.py for additional possible models. For patch-sequence (mamba) models using MM-DINOv2 the mri_embedding_dim should be 768, while for patch-mean (linear) models it should be twice that, so 1536. 

 load_data_in_mem should only be passed if enough RAM is present to load the entire dataset into memory.

### Evaluating Model

```
python eval.py --checkpoint_dir /ckpts/moe-mamba_pretrained.pt --case_csv metadata/test_cases.csv \
     --model_type <MODEL> --k 10 --patch_frac 1.0 --histo_embed_dim 1536 --n_heads 3 \
     --histo_csv metadata/histo_test.csv  --histo_root_dir data/wsi \
     --mri_csv metadata/mri_test.csv --mri_root_dir data/mri --mri_embed_dim 768 \
```
Number of heads is how many output scenarios to evaluate model on. Use 1 for unimodal models and 3 for bimodal models.
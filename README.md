# GOmarin: 

Adpatation of PlasmoFP model (https://github.com/harshstava/PlasmoFP_public/tree/main) for cyanobacterial sequences. Only tested on the 'Molecular Function' subontology.

start: `srun --pty bash` opens a bash session

`. /local/env/envconda.sh` to use conda

create the conda environment: `conda create -n PlasmoFP -f PlasmoFP.yaml`

then: `conda activate PlasmoFP`

## Downloading bacterial data from Uniprot: 

Protein sequences of bacteria from Uniprot in TSV format: 

Features : Entry_name, Organism, Sequence, Gene Ontology (MF),  Gene Ontology (CP), Gene Ontology (CC), Pfam, Interpro
(taxonomy_id:2) AND ((existence:3) OR (existence:2) OR (existence:1)) AND (length:[* TO 1200]) 

~330 000 sequences

--> trembl & swissprot

## Usage of Cluster

We used the capabilities of the IFB core cluster to run the jupyter notebooks using GPU.

https://doc.cluster.france-bioinformatique.fr/software/jupyter/


## Creation of training, test, and validation datasets + Generation of embeddings: 

Download the go-basic.obo file from https://current.geneontology.org/ontology/go-basic.obo.

`create_data_split/create_data_split_function_only.ipynb`

## Training models for predicting GO 'MF' terms and selecting parameters for the final model: 

model_function/MF_architecture_tune.ipynb

## Predicting GO terms using the model : 

model_function/Inference.ipynb


# TM-Vec embedding creation

[link to github](https://github.com/valentynbez/tmvec)

Conda environment
```
conda create -n tmvec python=3.10 -c pytorch
conda activate tmvec
pip install git+https://github.com/valentynbez/tmvec.git
conda install click
```

## Use TM-Vec to generate embeddings

Depending on the size of the fasta file, generatnig embeddings can take several hours. It may be necessary to use a GPU. Only the npz (and npy) format output is used.  

Download the TM-Vec model files:
- `tm_vec_cath_model.ckpt` - TM-Vec model checkpoint
- `tm_vec_cath_model_params.json` - TM-Vec configuration file

Available on : https://figshare.com/articles/dataset/TMvec_DeepBLAST_models/25810099?file=46296310

### For PlasmoFP:

This does not filter out long sequences
Also creates the .npy file used by PlasmoFP, instead of just the .npz file created by native tmvec.

#### Method 1 : 
Generate embeddings directly in the file create_data_splits_function_only.ipynb

#### Method 2 : 

Use the shell script (run_generate_embeddings.sh) by modifying it to call the corresponding files. 
Refer to the generate_embeddings.py script for formatting.  

### For general use

Filters out sequences with length > 1024

You can use split_fasta.sh to separate the fasta file into several smaller ones, then combine_embeddings.py to merge the .npz files.


Command to create embeddings for a fasta file:

```
tmvec build-db \
    --input-fasta sequences.fasta \
    --output db_test/small_fasta
```

The slurm script for this is available in generate_embeddings/tmvec.sh

## Use of TM-Vec Search

```
tmvec search \
    --input-fasta cyano.original.fasta \
    --database big_chunk_0_1_2_combined.npz \
    --output results/folder \
    --k-nearest 1
```

## Use of fdr_plot_ia.py

```
python plot/fdr_plot_ia.py \
    --original_tsv bacteria_sequences/function_only/model_training/cyanobact.original.predictions.tsv \
    --shuffled_tsv bacteria_sequences/function_only/model_training/cyanobact.shuffled.predictions.tsv \
    --ia_tsv bacteria_sequences/function_only/IA_all.tsv \
    --ia_threshold 3 \
    --out_prefix plot/fdr_plot_ia/fdr_ia_thres
```

## Use of fdr_plot_v2.py

```
python plot/fdr_plot_v2.py \
    --original_tsv bacteria_sequences/function_only/model_training/cyanobact.original.predictions.tsv \
    --shuffled_tsv bacteria_sequences/function_only/model_training/cyanobact.shuffled.predictions.tsv \
    --json_dict bacteria_sequences/function_only/function_data_json_dict.json \
    --out_prefix plot/fdr_plot_v2/fdr_dist_4
```

## Use of fdr_plot_v2_multi.py

```
python plot/fdr_plot_v2_multi.py \
    --original_tsv bacteria_sequences/function_only/model_training/cyanobact.original.predictions.tsv \
    --shuffled_tsv bacteria_sequences/function_only/model_training/cyanobact.shuffled.predictions.tsv \
    --json_dict bacteria_sequences/function_only/function_data_json_dict.json \
    --out_prefix plot/fdr_plot_v2/fdr_dist_345
```

## Use of plot_go_dist_freq.py

```
python plot/plot_go_dist_freq.py \
    --terms_tsv bacteria_sequences/function_only/terms_for_IA_calculation.tsv \
    --json_dict bacteria_sequences/function_only/function_data_json_dict.json \
    --out_prefix plot/go_dist_vs_freq/test_1 \
    --threshold 4
```


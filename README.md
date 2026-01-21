# GOmarin: 

Adpatation of PlasmoFP model (https://github.com/harshstava/PlasmoFP_public/tree/main) for cyanobacterial sequences. Only tested on the 'Molecular Function' subontology.

start: `srun --pty bash` opens a bash session

`. /local/env/envconda.sh` to use conda

create the conda environment: `conda create -n PlasmoFP python=3.12 -f PlasmoFP.yaml`

then: `conda activate PlasmoFP`

## Downloading bacterial data from Uniprot: 

Protein sequences of bacteria from Uniprot in TSV format: 

Features : Entry_name, Organism, Sequence, Gene Ontology (MF),  Gene Ontology (CP), Gene Ontology (CC), Pfam, Interpro
(taxonomy_id:2) AND ((existence:3) OR (existence:2) OR (existence:1)) AND (length:[* TO 1200]) 

~330 000 sequences

--> trembl & swissprot

## Creation of training, test, and validation datasets + Generation of embeddings: 

Download the go-basic.obo file from https://current.geneontology.org/ontology/go-basic.obo.
create_data_split/create_data_split_function_only.ipynb

## Usage of Cluster

We used the capabilities of the IFB core cluster to run the jupyter notebooks using GPU.

https://doc.cluster.france-bioinformatique.fr/software/jupyter/

## Training models for predicting GO 'MF' terms and selecting parameters for the final model: 

model_function/MF_architecture_tune.ipynb

## Predicting GO terms using the model : 

model_function/Inference.ipynb


# TM-Vec embedding creation

[link to github](https://github.com/valentynbez/tmvec)

```
conda create -n tmvec python=3.10 -c pytorch
conda activate tmvec
pip install git+https://github.com/valentynbez/tmvec.git
conda install click
```

```
tmvec build-db \
    --input-fasta small_embed.fasta \
    --output db_test/small_fasta
```

## Utiliser Tm-Vec pour récupèrer les embeddings

En fonction de la taille du fichier fasta, la génération des embeddings peut prendre plusieurs heures. Il peut être nécessaire d'utiliser un GPU. Seulement la sortie au format npz est utillisée. 

Télécharger les fichiers du modèle TM-Vec:
- `tm_vec_cath_model.ckpt` - TM-Vec model checkpoint
- `tm_vec_cath_model_params.json` - TM-Vec configuration file

Disponible sur : https://figshare.com/articles/dataset/TMvec_DeepBLAST_models/25810099?file=46296310

Besoin du dossier tm_vec téléchargé

Méthode 1 : 
Générer les embeddings directement dans le create_data_splits_function_only.ipynb

Méthode 2 : 
Utiliser le script shell (run_generate_embeddings.sh) en le modifiant pour bien utiliser les fichiers correspondant.

Voir le script de generate_embeddings pour la mise en forme. 

## Utilisation du TM-Vec Search 

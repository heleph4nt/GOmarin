# GOmarin: 

Reprise et adpatation du modèle de PlasmoFP (https://github.com/harshstava/PlasmoFP_public/tree/main) pour les séquences de cyanobactéries. Uniquement testé sur le sous-ensemble 'Molécular Function'.

start: `srun --pty bash` opens a bash session

`. /local/env/envconda.sh` to use conda

create the conda environment: `conda create -n PlasmoFP python=3.12 -r requirements.txt`

then: `conda activate PlasmoFP`

PlasmoFP folder contains: re-written PlasmoFP code, so that we do not need jupyter notebooks (it is a hassle to get them to work on a server)

## Downloading bacterial data from Uniprot: 

Protein sequences of bacteria from Uniprot in TSV format: 

Features : Entry_name, Organism, Sequence, Gene Ontology (MF),  Gene Ontology (CP), Gene Ontology (CC), Pfam, Interpro
(taxonomy_id:2) AND ((existence:3) OR (existence:2) OR (existence:1)) AND (length:[* TO 1200]) 

~330 000 sequences

--> trembl & swissprot

## Creation of training, test, and validation datasets + Generation of embeddings: 

create_data_split/create_data_split_function_only.ipynb

## Training models for predicting GO 'MF' terms and selecting parameters for the final model: 

model_function/MF_architecture_tune.ipynb

## Predicting GO terms using the model : 

model_function/Inference.ipynb

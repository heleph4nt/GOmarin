# GOmarin : 

Reprise et adpatation du modèle de PlasmoFP (https://github.com/harshstava/PlasmoFP_public/tree/main) pour les séquences de cyanobactéries. Uniquement testé sur le sous-ensemble 'Molécular Function'.

start: `srun --pty bash` opens a bash session

`. /local/env/envconda.sh` to use conda

create the conda environment: `conda create -n PlasmoFP python=3.12 -r requirements.txt`

then: `conda activate PlasmoFP`

PlasmoFP folder contains: re-written PlasmoFP code, so that we do not need jupyter notebooks (it is a hassle to get them to work on a server)

## Récupération des données : 

Séquences protéiques de bactéries sur Uniprot en format tsv : 

## Création des jeux de données d'entrainement, de test et de validation + Generation des embeddings : 

create_data_split/create_data_split_function_only.ipynb

## Entrainenement des modèles pour la prédiction de termes G0 'MF' et choix des paramètres du modèle final: 

model_function/MF_architecture_tune.ipynb

## Prédiction des terles GO grâce au modèle : 

model_function/Inference.ipynb

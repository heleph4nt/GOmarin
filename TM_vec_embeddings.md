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

## Bacteriota from Uniprot 

Features : Entry_name, Organism, Sequence, Gene Ontology (MF),  Gene Ontology (CP), Gene Ontology (CC), Pfam, Interpro
(taxonomy_id:2) AND ((existence:3) OR (existence:2) OR (existence:1)) AND (length:[* TO 1200]) 

~330 000 séquences

--> trembl & swissprot

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

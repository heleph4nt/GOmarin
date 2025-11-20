# TM-Vec embedding creation


[link to github](https://github.com/valentynbez/tmvec)

```
conda create -n tmvec python=3.10 -c pytorch
pip install git+https://github.com/valentynbez/tmvec.git
conda install click
```

```
tmvec build-db \
    --input-fasta small_embed.fasta \
    --output db_test/small_fasta
```

## Source for swissprot sequences:
Uniprot -> taxid 1117 (cyanobacteriota) -> Status: reviewed (Swissprot) --> 13000 proteins

## Bacteriota
(taxonomy_id:2) AND ((existence:3) OR (existence:2) OR (existence:1)) AND (length:[* TO 1200]) 
--> swissprot 
--> trembl & swissprot

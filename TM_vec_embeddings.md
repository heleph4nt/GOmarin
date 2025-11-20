# TM-Vec embedding creation


[link to github](https://github.com/valentynbez/tmvec)

```
conda create -n tmvec python -c pytorch
pip install git+https://github.com/valentynbez/tmvec.git
```

```
tmvec build-db \
    --input-fasta small_embed.fasta \
    --output db_test/small_fasta
```

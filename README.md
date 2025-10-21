# GOmarin


start: `srun --pty bash` opens a bash session

`. /local/env/envconda.sh` to use conda

create the conda environment: `conda create -n PlasmoFP python=3.12 -r requirements.txt`

then: `conda activate PlasmoFP`

PlasmoFP folder contains: re-written PlasmoFP code, so that we do not need jupyter notebooks (it is a hassle to get them to work on a server)

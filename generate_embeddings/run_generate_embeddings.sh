#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --account=deepmar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --output=output_embeddings_fct.txt
#SBATCH --job-name=calc_embeddings_fct
#SBATCH --time=30:00:00

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /shared/projects/deepmar/conda_environments/conda_envs/PlasmoFP


/shared/projects/deepmar/conda_environments/conda_envs/PlasmoFP/bin/python /shared/projects/deepmar/PlasmoFP_public/src/generate_embeddings.py --input reduced_90_function.fasta --output embeddings_16_01_2026/ --output_format npz --tm_vec_model /shared/projects/deepmar/data/tmvec_model_weights/tm_vec_cath_model.ckpt --device cuda --tm_vec_config /shared/projects/deepmar/data/tmvec_model_weights/tm_vec_cath_model_params.json


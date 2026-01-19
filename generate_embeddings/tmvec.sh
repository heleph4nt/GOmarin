#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
module load conda
source activate tmvec



tmvec build-db --input-fasta uniprot_taxonomy_bacteria_20_11_2025.fasta --output tmvec_embeddings 

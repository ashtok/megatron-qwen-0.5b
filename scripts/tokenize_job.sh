#!/bin/bash
#SBATCH --job-name=tokenize_gpt2_de
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # satisfy QOSMinGRES
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/tokenize_%j.out
#SBATCH --error=logs/tokenize_%j.err

set -euo pipefail

cd /data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2

srun apptainer exec --nv ../nemo_sandbox bash scripts/tokenize.sh

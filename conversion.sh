#!/bin/bash
#SBATCH --job-name=convert_qwen25
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/convert_qwen25_%j.out
#SBATCH --error=logs/convert_qwen25_%j.err

echo "=========================================="
echo "Converting Qwen checkpoint to HF format"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Load Singularity/Apptainer if needed
module load apptainer

# Sanity check: GPUs
nvidia-smi

# Run the conversion inside container
apptainer exec --nv ../nemo_sandbox \
    python /data/42-julia-hpc-rz-wuenlp/s472389/modalities_test/convert_to_hf_2gpu.py

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

#!/bin/bash

# Slurm sbatch options
#SBATCH -o runLlamaGuard.sh.log-%j

# Loading the required module(s)
module load anaconda/2023b

# Run the script
python test_llama_guard.py
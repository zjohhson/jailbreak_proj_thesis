#!/bin/bash

# Slurm sbatch options
#SBATCH -o download.sh.log-%j

# Loading the required module(s)
module load anaconda/2023b

# Run the script
python download.py
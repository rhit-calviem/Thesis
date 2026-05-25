#!/bin/bash
#SBATCH --job-name=OmniSR_Train
#SBATCH --output=Logs/job_%j.log
#SBATCH --nodelist=gus
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# needed to move to the directory
cd /home/calviem/Thesis

uv run nvidia-smi
# run code
hostname
uv run which python
uv run which python3
# uv run python run_pipeline.py
uv run python3 images_visualizations.py
# uv run python3 train_gan.py
# uv run python3 gan_visualization.py

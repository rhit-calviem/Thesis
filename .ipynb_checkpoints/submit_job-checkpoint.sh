#!/bin/bash
#SBATCH --job-name=OmniSR_Train       # Job name
#SBATCH --output=Logs/job_%j.log      # Standard output and error log (%j = JobID)
#SBATCH --partition=gpu               # Name of the partition (change if needed)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Match NUM_WORKERS in config.py
#SBATCH --mem=16G                     # Memory limit
#SBATCH --time=24:00:00               # Time limit (hrs:min:sec)

# Create Logs directory if it doesn't exist
mkdir -p Logs

# Load your environment (uncomment and adjust based on your cluster)
# module load cuda/11.8
# source activate your_env_name

# Run the pipeline
# We point to 'Models/hrmodel_ckpt.pth' (the full state) as the resume path
python run_pipeline.py
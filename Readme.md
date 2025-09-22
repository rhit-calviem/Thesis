# Lightweight Super-Resolution Model Pipeline
This repository contains the code for a complete, end-to-end pipeline for training and evaluating a lightweight single-image super-resolution (SISR) model, based on the findings from the paper "Efficient Attention-Sharing Information Distillation Transformer for Lightweight Single Image Super-Resolution".

Currently, it features a robust data pipeline with a placeholder model, which can be easily replaced with a more complex architecture like the ASID Transformer.

## Project Structure
Thesis/
├── main.py             # Main script to run training, evaluation, and visualization
├── config.py           # Central configuration for all hyperparameters and paths
├── models.py           # Contains the SR model architecture (currently a placeholder)
├── requirements.txt    # Lists all Python dependencies
├── .gitignore          # Specifies files and folders for Git to ignore
└── data/               # (Not tracked by Git) Holds all datasets
    ├── DIV2K/
    ├── Set5/
    └── ...

##Setup and Installation
1. Clone the Repository
git clone <your-repository-url>
cd thesis

2. Create and Activate Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

### Create the environment
python3 -m venv env

### Activate the environment
source env/bin/activate

3. Install Dependencies
Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt

(Note: If you have not created this file yet, run pip freeze > requirements.txt after installing your packages.)

4. Download Datasets
This pipeline requires the DIV2K dataset for training and standard benchmark datasets like Set5 for testing. Download them and place them in the data/ directory according to the structure specified in the project files. The data directory itself is not tracked by Git.

## How to Run
1. Configure Your Run
Adjust the settings in config.py to set the desired upscale factor, number of epochs, learning rate, and data paths.

2. Start Training and Evaluation
Execute the main script from the terminal:

python main.py

The script will:

Train the model on the DIV2K dataset.

Save the trained model weights to placeholder_model.pth.

Evaluate the model on the specified test set (e.g., Set5) and print the average PSNR.

Generate and save a visualization of a random test sample to sr_visualization.png.
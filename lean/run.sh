#!/bin/bash

# Setup script for Mathematical Reasoning in Latent Space

echo "Setting up Mathematical Reasoning in Latent Space..."

# Create necessary directories
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Download LeanDojo data (example)
echo "Setting up LeanDojo data..."
# You would need to download and setup LeanDojo data here
# python -c "from lean_dojo import LeanGitRepo; LeanGitRepo.from_path('path/to/lean/repo')"

# Run training
echo "Starting training..."
python main.py --mode train --use_lean_engine

# Run evaluation
echo "Running evaluation..."
python main.py --mode eval --model_path best_model.pt

echo "Setup and execution completed!"
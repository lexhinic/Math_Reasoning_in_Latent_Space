# main.py
import os
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from config import Config
from models.model import RewriteReasoningModel
from training.train import train
from data_processing.dataset import generate_multi_step_datasets
from evaluation.metrics import evaluate_multi_step_reasoning
from evaluation.visualize import (
    plot_roc_curves,
    plot_auc_vs_steps,
    plot_l2_distances,
    plot_embedding_visualization,
    plot_histogram_of_scores
)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate rewrite reasoning model")
    
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to save/load model")
    parser.add_argument("--results_path", type=str, default="results.pt", help="Path to save/load results")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save visualizations")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--max_steps", type=int, default=1, help="Maximum number of rewrite steps for evaluation")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    config = Config()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = RewriteReasoningModel(config).to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.train:
        print("Training model...")
        model = train(model, config, device)
        
        # Save model
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
    
    if args.evaluate or args.visualize:
        print("Loading model...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
    
    if args.evaluate:
        print("Generating multi-step datasets...")
        datasets = generate_multi_step_datasets(config, max_steps=args.max_steps)
        
        print("Evaluating multi-step reasoning...")
        results = evaluate_multi_step_reasoning(model, datasets, device, max_steps=args.max_steps)
        
        # Save results
        torch.save(results, args.results_path)
        print(f"Results saved to {args.results_path}")
    
    if args.visualize:
        print("Loading results...")
        results = torch.load(args.results_path, map_location=device)
        
        print("Creating visualizations...")
        
        # Plot ROC curves for different steps
#        for step in [1, 5, 9]:
        for step in [1]:
            if step < len(results["true_auc"]):
                fig = plot_roc_curves(results, step=step)
                fig.savefig(os.path.join(args.output_dir, f"roc_curves_step{step}.png"))
        
        # Plot AUC versus steps
        fig = plot_auc_vs_steps(results, max_steps=args.max_steps)
        fig.savefig(os.path.join(args.output_dir, "auc_vs_steps.png"))
        
        # Plot L2 distances
        fig = plot_l2_distances(results, max_steps=args.max_steps)
        fig.savefig(os.path.join(args.output_dir, "l2_distances.png"))
        
        # Plot histogram of scores for first step
        if results["true_success_probs"] and results["all_success_labels"]:
            fig = plot_histogram_of_scores(
                results["true_success_probs"][0], 
                results["all_success_labels"][0], 
                step=1
            )
            fig.savefig(os.path.join(args.output_dir, "histogram_step1.png"))
        
        # Plot histogram of scores for last step
        last_step = min(args.max_steps, len(results["true_success_probs"]) - 1)
        if results["true_success_probs"] and results["all_success_labels"] and last_step >= 0:
            fig = plot_histogram_of_scores(
                results["true_success_probs"][last_step], 
                results["all_success_labels"][last_step], 
                step=last_step + 1
            )
            fig.savefig(os.path.join(args.output_dir, f"histogram_step{last_step+1}.png"))
        
        print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
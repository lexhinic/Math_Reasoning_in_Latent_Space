"""
Main entry point for mathematical reasoning in latent space
"""
import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from typing import List

# Add project root to path
import sys 
sys.path.append('/home/stu4/formal_reasoning/baseline/Math_Reasoning_in_Latent_Space/')
from baseline.Math_Reasoning_in_Latent_Space.lean.config import config
from baseline.Math_Reasoning_in_Latent_Space.lean.data.ori.lean_interface import load_leandojo_theorems, LeanRewriteEngine, generate_rewrite_pairs
from data.formula_parser import LeanFormulaParser
from baseline.Math_Reasoning_in_Latent_Space.lean.data.ori.dataset import create_data_loaders, save_processed_data, load_processed_data
from models.reasoning_model import ReasoningModel
from training.trainer import create_trainer
from evaluation.evaluator import run_evaluation

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'main.log')),
            logging.StreamHandler()
        ]
    )

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(args):
    """Prepare training data"""
    logger = logging.getLogger(__name__)
    
    # Check if processed data exists
    processed_data_path = os.path.join(config.data.data_path, "processed_data.pkl")
    
    if os.path.exists(processed_data_path) and not args.reprocess_data:
        logger.info("Loading existing processed data...")
        data = load_processed_data(processed_data_path)
        theorems = data['theorems']
        rewrite_pairs = data['rewrite_pairs']
        parser = data['parser']
        
    else:
        logger.info("Processing data from scratch...")
        
        # Load theorems from LeanDojo
        logger.info("Loading theorems from LeanDojo...")
        theorems = load_leandojo_theorems(config.data.data_path)
        
        if not theorems:
            logger.error("No theorems loaded. Please check your data path and ensure LeanDojo data is available.")
            sys.exit(1)
        
        logger.info(f"Loaded {len(theorems)} theorems")
        
        # Initialize parser
        parser = LeanFormulaParser()
        
        # Generate rewrite pairs
        logger.info("Generating rewrite pairs...")
        
        if args.use_lean_engine:
            # Use actual Lean rewrite engine (requires Lean installation)
            with LeanRewriteEngine(args.lean_repo_path, config.data.max_rewrite_time) as engine:
                rewrite_pairs = generate_rewrite_pairs(theorems, engine)
        else:
            # Use simulated rewrite pairs for testing
            rewrite_pairs = generate_simulated_rewrite_pairs(theorems)
        
        logger.info(f"Generated {len(rewrite_pairs)} rewrite pairs")
        
        # Process formulas to build vocabularies
        logger.info("Building vocabularies...")
        for theorem in theorems:
            parser.parse_formula(theorem.statement)
        
        # Save processed data
        data = {
            'theorems': theorems,
            'rewrite_pairs': rewrite_pairs,
            'parser': parser
        }
        save_processed_data(data, processed_data_path)
        logger.info(f"Processed data saved to {processed_data_path}")
    
    return theorems, rewrite_pairs, parser

def generate_simulated_rewrite_pairs(theorems):
    """Generate simulated rewrite pairs for testing without Lean engine"""
    logger = logging.getLogger(__name__)
    
    rewrite_pairs = []
    
    for i, target_theorem in enumerate(theorems):
        for j, param_theorem in enumerate(theorems):
            if j < i:  # Only use earlier theorems as parameters
                # Simulate rewrite success based on simple heuristics
                target_tokens = set(target_theorem.statement.split())
                param_tokens = set(param_theorem.statement.split())
                
                intersection = target_tokens.intersection(param_tokens)
                
                # Success probability based on token overlap
                if len(intersection) > 3:
                    success = np.random.random() > 0.2  # 80% success
                elif len(intersection) > 1:
                    success = np.random.random() > 0.7  # 30% success
                else:
                    success = np.random.random() > 0.95  # 5% success
                
                # Generate result formula for successful rewrites
                result = None
                if success:
                    result = f"rewritten({target_theorem.statement})"
                
                rewrite_pairs.append((
                    target_theorem.statement,
                    param_theorem.statement,
                    success,
                    result
                ))
    
    logger.info(f"Generated {len(rewrite_pairs)} simulated rewrite pairs")
    return rewrite_pairs

def train_model(args, theorems, rewrite_pairs, parser):
    """Train the reasoning model"""
    logger = logging.getLogger(__name__)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(theorems, rewrite_pairs, parser)
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Get vocabulary sizes
    node_type_vocab_size, node_value_vocab_size = parser.get_vocab_sizes()
    
    logger.info(f"Node type vocabulary size: {node_type_vocab_size}")
    logger.info(f"Node value vocabulary size: {node_value_vocab_size}")
    
    # Create model
    logger.info("Creating model...")
    model = ReasoningModel(
        node_type_vocab_size=node_type_vocab_size,
        node_value_vocab_size=node_value_vocab_size,
        formula_embedding_dim=config.model.formula_embedding_dim,
        hidden_dim=config.model.node_embedding_dim,
        num_gnn_layers=config.model.gnn_hops,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        dropout=config.model.dropout,
        noise_std=config.model.noise_std
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = create_trainer(model, train_loader, val_loader)
    
    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
        logger.info(f"Resumed training from {args.resume_from_checkpoint}")
    
    # Train model
    logger.info(f"Starting training for {config.model.num_epochs} epochs...")
    training_results = trainer.train(config.model.num_epochs)
    
    logger.info("Training completed!")
    
    return model, test_loader, training_results

def evaluate_model(args, model, theorems, parser):
    """Evaluate the trained model"""
    logger = logging.getLogger(__name__)
    
    # Split theorems for evaluation
    num_test = int(len(theorems) * config.data.test_split)
    test_theorems = theorems[-num_test:]
    
    logger.info(f"Evaluating on {len(test_theorems)} test theorems")
    
    # Run evaluation
    eval_results = run_evaluation(
        model=model,
        test_theorems=test_theorems,
        parser=parser,
        save_dir=config.results_dir,
        max_steps=config.data.max_reasoning_steps
    )
    
    return eval_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Mathematical Reasoning in Latent Space")
    
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both',
                       help='Mode to run: train, eval, or both')
    parser.add_argument('--reprocess_data', action='store_true',
                       help='Reprocess data even if processed data exists')
    parser.add_argument('--use_lean_engine', action='store_true',
                       help='Use actual Lean rewrite engine (requires Lean installation)')
    parser.add_argument('--lean_repo_path', type=str, default='./lean_repo',
                       help='Path to Lean repository for rewrite engine')
    parser.add_argument('--resume_from_checkpoint', type=str,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                       help='Path to trained model for evaluation')
    parser.add_argument('--config', type=str,
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_random_seeds(config.seed)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Mathematical Reasoning in Latent Space")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {config.device}")
    
    # Prepare data
    theorems, rewrite_pairs, parser = prepare_data(args)
    
    model = None
    
    if args.mode in ['train', 'both']:
        # Train model
        model, test_loader, training_results = train_model(args, theorems, rewrite_pairs, parser)
        
    if args.mode in ['eval', 'both']:
        # Load model if not trained in this run
        if model is None:
            logger.info(f"Loading model from {args.model_path}")
            
            # Get vocabulary sizes
            node_type_vocab_size, node_value_vocab_size = parser.get_vocab_sizes()
            
            # Create model
            model = ReasoningModel(
                node_type_vocab_size=node_type_vocab_size,
                node_value_vocab_size=node_value_vocab_size,
                formula_embedding_dim=config.model.formula_embedding_dim,
                hidden_dim=config.model.node_embedding_dim,
                num_gnn_layers=config.model.gnn_hops,
                mlp_hidden_dims=config.model.mlp_hidden_dims,
                dropout=config.model.dropout,
                noise_std=config.model.noise_std
            )
            
            # Load checkpoint
            checkpoint_path = os.path.join(config.checkpoint_dir, args.model_path)
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {checkpoint_path}")
        
        # Evaluate model
        eval_results = evaluate_model(args, model, theorems, parser)
        
        logger.info("Evaluation completed!")
        logger.info(f"Results saved to {config.results_dir}")
    
    logger.info("Program completed successfully!")

if __name__ == "__main__":
    main()
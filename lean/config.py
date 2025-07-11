"""
Configuration file for mathematical reasoning in latent space
Adapted for Lean/LeanDojo from the original HOL Light implementation
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Graph Neural Network parameters
    gnn_hops: int = 16
    node_embedding_dim: int = 128
    formula_embedding_dim: int = 1024
    
    # MLP parameters
    mlp_hidden_dims: list = None
    mlp_activation: str = "relu"
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_clip: float = 1.0
    
    # Regularization
    noise_std: float = 1e-3  # β parameter from paper
    dropout: float = 0.1
    
    # Loss weights
    success_prediction_weight: float = 1.0
    embedding_prediction_weight: float = 1.0
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [1024, 1024, 1024]

@dataclass
class DataConfig:
    """Data processing configuration"""
    # LeanDojo dataset paths
    data_path: str = "./leandojo_data"
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    
    # Preprocessing
    max_formula_length: int = 512
    min_formula_length: int = 5
    
    # Rewrite generation
    max_rewrite_time: float = 5.0  # seconds
    max_rewrites_per_theorem: int = 1000
    
    # Multi-step evaluation
    max_reasoning_steps: int = 9
    eval_batch_size: int = 32

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

# Global config instance
config = Config()
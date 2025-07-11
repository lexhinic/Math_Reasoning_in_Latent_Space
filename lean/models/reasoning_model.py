import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from .graph_encoder import FormulaEncoder
import sys 
sys.path.append('/home/stu4/formal_reasoning/baseline/Math_Reasoning_in_Latent_Space/')
from baseline.Math_Reasoning_in_Latent_Space.lean.config import config

class ReasoningModel(nn.Module):
    """Main model for mathematical reasoning in latent space"""
    
    def __init__(self,
                 node_type_vocab_size: int,
                 node_value_vocab_size: int,
                 formula_embedding_dim: int = 1024,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 16,
                 mlp_hidden_dims: list = None,
                 dropout: float = 0.1,
                 noise_std: float = 1e-3):
        """
        Initialize the reasoning model
        
        Args:
            node_type_vocab_size: Size of node type vocabulary
            node_value_vocab_size: Size of node value vocabulary
            formula_embedding_dim: Dimension of formula embeddings
            hidden_dim: Hidden dimension for GNN
            num_gnn_layers: Number of GNN layers
            mlp_hidden_dims: Hidden dimensions for MLP layers
            dropout: Dropout probability
            noise_std: Standard deviation for training noise
        """
        super().__init__()
        
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [1024, 1024, 1024]
        
        self.formula_embedding_dim = formula_embedding_dim
        self.noise_std = noise_std
        
        # Formula encoders (γ and π in the paper)
        self.target_encoder = FormulaEncoder(
            node_type_vocab_size=node_type_vocab_size,
            node_value_vocab_size=node_value_vocab_size,
            output_dim=formula_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        self.parameter_encoder = FormulaEncoder(
            node_type_vocab_size=node_type_vocab_size,
            node_value_vocab_size=node_value_vocab_size,
            output_dim=formula_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # Combiner network (c in the paper)
        combiner_input_dim = formula_embedding_dim * 3  # Concatenation + element-wise product
        combiner_layers = []
        
        prev_dim = combiner_input_dim
        for hidden_dim in mlp_hidden_dims:
            combiner_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.combiner = nn.Sequential(*combiner_layers)
        self.combiner_output_dim = prev_dim
        
        # Success prediction head (p in the paper)
        self.success_predictor = nn.Linear(self.combiner_output_dim, 1)
        
        # Embedding prediction head (e in the paper)
        self.embedding_predictor = nn.Linear(self.combiner_output_dim, formula_embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_target(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Encode target formula
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            
        Returns:
            Target formula embeddings
        """
        embedding = self.target_encoder(x, edge_index, batch)
        
        # Add noise during training (as mentioned in the paper)
        if self.training:
            noise = torch.randn_like(embedding) * self.noise_std
            embedding = embedding + noise
        
        return embedding
    
    def encode_parameter(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Encode parameter formula
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            
        Returns:
            Parameter formula embeddings
        """
        return self.parameter_encoder(x, edge_index, batch)
    
    def combine_embeddings(self, target_emb: torch.Tensor, param_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine target and parameter embeddings
        
        Args:
            target_emb: Target formula embeddings
            param_emb: Parameter formula embeddings
            
        Returns:
            Combined embeddings
        """
        # Concatenate embeddings and their element-wise product
        elementwise_product = target_emb * param_emb
        combined = torch.cat([target_emb, param_emb, elementwise_product], dim=-1)
        
        return self.combiner(combined)
    
    def predict_success(self, combined_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict rewrite success probability
        
        Args:
            combined_emb: Combined embeddings from combiner network
            
        Returns:
            Success probabilities
        """
        return self.success_predictor(combined_emb).squeeze(-1)
    
    def predict_embedding(self, combined_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict embedding of rewrite result
        
        Args:
            combined_emb: Combined embeddings from combiner network
            
        Returns:
            Predicted result embeddings
        """
        return self.embedding_predictor(combined_emb)
    
    def forward(self, target_data, parameter_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            target_data: Target formula batch data
            parameter_data: Parameter formula batch data
            
        Returns:
            Tuple of (success_logits, predicted_embeddings)
        """
        # Encode formulas
        target_emb = self.encode_target(
            target_data.x, target_data.edge_index, target_data.batch
        )
        param_emb = self.encode_parameter(
            parameter_data.x, parameter_data.edge_index, parameter_data.batch
        )
        
        # Combine embeddings
        combined_emb = self.combine_embeddings(target_emb, param_emb)
        
        # Make predictions
        success_logits = self.predict_success(combined_emb)
        predicted_embeddings = self.predict_embedding(combined_emb)
        
        return success_logits, predicted_embeddings
    
    def reason_in_latent_space(self, 
                              initial_embedding: torch.Tensor,
                              parameter_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Perform reasoning steps in latent space
        
        Args:
            initial_embedding: Initial formula embedding [batch_size, embedding_dim]
            parameter_embeddings: Sequence of parameter embeddings [batch_size, num_steps, embedding_dim]
            
        Returns:
            Final predicted embedding after all reasoning steps
        """
        current_embedding = initial_embedding
        batch_size, num_steps, embedding_dim = parameter_embeddings.shape
        
        for step in range(num_steps):
            param_emb = parameter_embeddings[:, step, :]
            
            # Combine current embedding with parameter
            combined_emb = self.combine_embeddings(current_embedding, param_emb)
            
            # Predict next embedding
            next_embedding = self.predict_embedding(combined_emb)
            current_embedding = next_embedding
        
        return current_embedding
    
    def evaluate_rewrite_success(self, 
                                target_embedding: torch.Tensor,
                                parameter_embedding: torch.Tensor) -> torch.Tensor:
        """
        Evaluate rewrite success probability given embeddings
        
        Args:
            target_embedding: Target formula embedding
            parameter_embedding: Parameter formula embedding
            
        Returns:
            Success probabilities
        """
        combined_emb = self.combine_embeddings(target_embedding, parameter_embedding)
        return torch.sigmoid(self.predict_success(combined_emb))

class ReasoningLoss(nn.Module):
    """Combined loss function for reasoning model"""
    
    def __init__(self, 
                 success_weight: float = 1.0,
                 embedding_weight: float = 1.0,
                 use_stop_gradients: bool = True):
        """
        Initialize the loss function
        
        Args:
            success_weight: Weight for success prediction loss
            embedding_weight: Weight for embedding prediction loss
            use_stop_gradients: Whether to stop gradients for embedding prediction
        """
        super().__init__()
        
        self.success_weight = success_weight
        self.embedding_weight = embedding_weight
        self.use_stop_gradients = use_stop_gradients
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                success_logits: torch.Tensor,
                predicted_embeddings: torch.Tensor,
                success_targets: torch.Tensor,
                target_embeddings: torch.Tensor,
                success_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            success_logits: Predicted success logits
            predicted_embeddings: Predicted result embeddings
            success_targets: Ground truth success labels
            target_embeddings: Ground truth result embeddings
            success_mask: Mask for successful rewrites
            
        Returns:
            Dictionary containing loss components
        """
        # Success prediction loss
        success_loss = self.bce_loss(success_logits, success_targets)
        
        # Embedding prediction loss (only for successful rewrites)
        if success_mask.sum() > 0:
            successful_predicted = predicted_embeddings[success_mask]
            successful_targets = target_embeddings[success_mask]
            
            # Stop gradients as mentioned in the paper
            if self.use_stop_gradients:
                successful_targets = successful_targets.detach()
            
            embedding_loss = self.mse_loss(successful_predicted, successful_targets)
        else:
            embedding_loss = torch.tensor(0.0, device=success_logits.device)
        
        # Combined loss
        total_loss = (self.success_weight * success_loss + 
                     self.embedding_weight * embedding_loss)
        
        return {
            'total_loss': total_loss,
            'success_loss': success_loss,
            'embedding_loss': embedding_loss
        }
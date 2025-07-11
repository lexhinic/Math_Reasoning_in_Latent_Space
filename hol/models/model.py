# models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import EmbeddingNetwork
from .predictor import CombinerNetwork, RewriteSuccessPredictor, EmbeddingPredictor

class RewriteReasoningModel(nn.Module):
    """
    Full model for rewrite reasoning in latent space.
    Combines embedding networks and prediction networks.
    """
    
    def __init__(self, config):
        """
        Initialize the rewrite reasoning model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Embedding networks
        self.theorem_embed = EmbeddingNetwork(
            config.NODE_DIM,
            config.NODE_DIM,
            config.EMBED_DIM,
            config.NUM_HOPS
        )
        
        self.param_embed = EmbeddingNetwork(
            config.NODE_DIM,
            config.NODE_DIM,
            config.EMBED_DIM,
            config.NUM_HOPS
        )
        
        # Combiner network
        self.combiner = CombinerNetwork(
            config.EMBED_DIM,
            config.HIDDEN_DIM
        )
        
        # Prediction networks
        self.success_pred = RewriteSuccessPredictor(
            config.HIDDEN_DIM
        )
        
        self.embed_pred = EmbeddingPredictor(
            config.HIDDEN_DIM,
            config.EMBED_DIM
        )
    
    def forward(self, batch, add_noise=True):
        """
        Forward pass through the model.
        
        Args:
            batch: Batch of data
            add_noise: Whether to add noise to the theorem embeddings
            
        Returns:
            success_logits: Rewrite success logits
            embed_pred: Predicted embeddings for successful rewrites
            theorem_embed: Theorem embeddings
            param_embed: Parameter embeddings
        """
        # Handle batch processing
        theorem_embeds = []
        param_embeds = []
        
        # Process each example in the batch
        for i in range(len(batch["t_nodes"])):
            # Compute theorem embedding
            t_embed = self.theorem_embed(
                batch["t_nodes"][i],
                batch["t_edges"][i],
                batch["t_types"][i]
            )
            theorem_embeds.append(t_embed)
            
            # Compute parameter embedding
            p_embed = self.param_embed(
                batch["p_nodes"][i],
                batch["p_edges"][i],
                batch["p_types"][i]
            )
            param_embeds.append(p_embed)
        
        # Stack embeddings
        theorem_embed = torch.stack(theorem_embeds)
        param_embed = torch.stack(param_embeds)
        
        # Add noise to theorem embeddings during training
        if add_noise and self.training:
            theorem_embed = theorem_embed + torch.randn_like(theorem_embed) * self.config.NOISE_STD
        
        # Combine embeddings
        combined = self.combiner(theorem_embed, param_embed)
        
        # Predict rewrite success
        success_logits = self.success_pred(combined)
        
        # Predict result embeddings
        embed_pred = self.embed_pred(combined)
        
        return success_logits, embed_pred, theorem_embed, param_embed
    
    def compute_loss(self, batch):
        """
        Compute the loss for a batch of data.
        
        Args:
            batch: Batch of data
            
        Returns:
            loss: Total loss
            success_loss: Rewrite success prediction loss
            embed_loss: Embedding prediction loss
        """
        # Forward pass
        success_logits, embed_pred, theorem_embed, param_embed = self.forward(batch)
        
        # Compute rewrite success prediction loss
        success_loss = F.binary_cross_entropy_with_logits(
            success_logits,
            batch["success"]
        )
        
        # Compute embedding prediction loss for successful rewrites
        embed_loss = torch.tensor(0.0, device=success_logits.device)
        
        if "r_nodes" in batch:
            # Process each successful rewrite
            result_embeds = []
            success_mask = batch["success"].bool()
            
            if success_mask.sum() > 0:
                # Get indices of successful rewrites
                success_indices = torch.where(success_mask)[0].tolist()
                
                # Compute result embeddings
                for idx in success_indices:
                    r_embed = self.theorem_embed(
                        batch["r_nodes"][idx],
                        batch["r_edges"][idx],
                        batch["r_types"][idx]
                    )
                    result_embeds.append(r_embed)
                
                if result_embeds:
                    # Stack result embeddings
                    result_embed = torch.stack(result_embeds)
                    
                    # Detach result embeddings to prevent gradient flow
                    result_embed = result_embed.detach()
                    
                    # Compute loss only for successful rewrites
                    embed_loss = F.mse_loss(
                        embed_pred[success_mask],
                        result_embed
                    )
        
        # Compute total loss
        loss = success_loss + embed_loss
        
        return loss, success_loss, embed_loss
    
    def predict_rewrite_success(self, theorem_embed, param_embed):
        """
        Predict whether a rewrite will succeed.
        
        Args:
            theorem_embed: Theorem embedding
            param_embed: Parameter embedding
            
        Returns:
            success_prob: Probability of rewrite success
        """
        # Combine embeddings
        combined = self.combiner(theorem_embed, param_embed)
        
        # Predict rewrite success
        success_logits = self.success_pred(combined)
        
        # Convert logits to probabilities
        success_prob = torch.sigmoid(success_logits)
        
        return success_prob
    
    def predict_result_embedding(self, theorem_embed, param_embed):
        """
        Predict the embedding of the rewrite result.
        
        Args:
            theorem_embed: Theorem embedding
            param_embed: Parameter embedding
            
        Returns:
            embed_pred: Predicted embedding of the rewrite result
        """
        # Combine embeddings
        combined = self.combiner(theorem_embed, param_embed)
        
        # Predict result embedding
        embed_pred = self.embed_pred(combined)
        
        return embed_pred
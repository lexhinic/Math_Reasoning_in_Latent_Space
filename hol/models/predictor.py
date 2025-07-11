# models/predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinerNetwork(nn.Module):
    """
    Combiner network for theorem and parameter embeddings.
    Processes the concatenated embeddings to produce a combined representation.
    """
    
    def __init__(self, embed_dim, hidden_dim):
        """
        Initialize the combiner network.
        
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Three-layer perceptron
        self.fc1 = nn.Linear(3 * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, theorem_embed, param_embed):
        """
        Forward pass through the combiner network.
        
        Args:
            theorem_embed: Theorem embedding [batch_size, embed_dim]
            param_embed: Parameter embedding [batch_size, embed_dim]
            
        Returns:
            combined: Combined embedding [batch_size, hidden_dim]
        """
        # Concatenate embeddings with element-wise multiplication
        combined = torch.cat([
            theorem_embed,
            param_embed,
            theorem_embed * param_embed
        ], dim=1)
        
        # Pass through the three-layer perceptron
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        
        return combined

class RewriteSuccessPredictor(nn.Module):
    """
    Predictor for rewrite success.
    Predicts whether a rewrite operation will succeed.
    """
    
    def __init__(self, hidden_dim):
        """
        Initialize the rewrite success predictor.
        
        Args:
            hidden_dim: Dimension of input features
        """
        super().__init__()
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, combined):
        """
        Forward pass through the rewrite success predictor.
        
        Args:
            combined: Combined embedding [batch_size, hidden_dim]
            
        Returns:
            logits: Rewrite success logits [batch_size, 1]
        """
        return self.fc(combined).squeeze(-1)

class EmbeddingPredictor(nn.Module):
    """
    Predictor for the embedding of the rewrite result.
    Predicts the embedding vector of the formula that results from a successful rewrite.
    """
    
    def __init__(self, hidden_dim, embed_dim):
        """
        Initialize the embedding predictor.
        
        Args:
            hidden_dim: Dimension of input features
            embed_dim: Dimension of output embeddings
        """
        super().__init__()
        
        self.fc = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, combined):
        """
        Forward pass through the embedding predictor.
        
        Args:
            combined: Combined embedding [batch_size, hidden_dim]
            
        Returns:
            embed_pred: Predicted embedding [batch_size, embed_dim]
        """
        return self.fc(combined)
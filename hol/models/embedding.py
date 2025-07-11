# models/embedding.py
import torch
import torch.nn as nn
from .gnn import GraphNeuralNetwork

class EmbeddingNetwork(nn.Module):
    """
    Embedding network for HOL Light formulas.
    Uses a GNN to convert formula graphs into fixed-dimensional embeddings.
    """
    
    def __init__(self, node_dim, hidden_dim, embed_dim, num_hops):
        """
        Initialize the embedding network.
        
        Args:
            node_dim: Dimension of node features
            hidden_dim: Dimension of hidden layers
            embed_dim: Dimension of output embeddings
            num_hops: Number of message passing hops
        """
        super().__init__()
        
        self.gnn = GraphNeuralNetwork(node_dim, hidden_dim, embed_dim, num_hops)
    
    def forward(self, nodes, edges, node_types):
        """
        Forward pass through the embedding network.
        
        Args:
            nodes: Node features [num_nodes, node_features]
            edges: Edge indices [2, num_edges]
            node_types: Node type indices [num_nodes]
            
        Returns:
            embedding: Formula embedding [embed_dim]
        """
        return self.gnn(nodes, edges, node_types)
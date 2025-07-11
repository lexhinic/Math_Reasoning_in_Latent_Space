"""
Graph Neural Network encoder for mathematical formulas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
import math

class GraphConvLayer(MessagePassing):
    """Single layer of graph convolution"""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.1):
        """
        Initialize graph convolution layer
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension  
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Linear transformations for message passing
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention mechanism
        self.att_src = nn.Parameter(torch.randn(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.randn(1, heads, out_channels))
        
        # Output transformation
        self.lin_out = nn.Linear(heads * out_channels, out_channels)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters"""
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Message passing
        out = self.propagate(edge_index, x=x)
        
        # Output transformation
        out = self.lin_out(out)
        out = self.norm(out + x if x.size(-1) == out.size(-1) else out)
        out = F.relu(out)
        out = self.dropout_layer(out)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Create messages
        
        Args:
            x_i: Target node features
            x_j: Source node features  
            index: Edge indices
            
        Returns:
            Messages
        """
        # Transform features
        src_features = self.lin_src(x_j).view(-1, self.heads, self.out_channels)
        dst_features = self.lin_dst(x_i).view(-1, self.heads, self.out_channels)
        
        # Compute attention scores
        alpha_src = (src_features * self.att_src).sum(dim=-1)
        alpha_dst = (dst_features * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, dim=-1)
        
        # Apply attention
        out = src_features * alpha.unsqueeze(-1)
        out = out.view(-1, self.heads * self.out_channels)
        
        return out

class GraphEncoder(nn.Module):
    """Multi-layer graph neural network encoder"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 1024,
                 num_layers: int = 16,
                 heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize graph encoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of graph convolution layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Input embedding
        x = self.input_embedding(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        
        # Graph convolution layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
        
        # Global pooling
        x = global_max_pool(x, batch)
        
        # Output projection
        x = self.output_projection(x)
        
        return x

class NodeEmbedding(nn.Module):
    """Learnable node embeddings for different node types and values"""
    
    def __init__(self, 
                 node_type_vocab_size: int,
                 node_value_vocab_size: int,
                 embedding_dim: int = 64):
        """
        Initialize node embedding layer
        
        Args:
            node_type_vocab_size: Size of node type vocabulary
            node_value_vocab_size: Size of node value vocabulary  
            embedding_dim: Embedding dimension for each component
        """
        super().__init__()
        
        self.node_type_embedding = nn.Embedding(node_type_vocab_size, embedding_dim)
        self.node_value_embedding = nn.Embedding(node_value_vocab_size, embedding_dim)
        self.output_dim = embedding_dim * 2  # Concatenated embeddings
        
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: Node features [num_nodes, 2] where features are [type_id, value_id]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        type_ids = node_features[:, 0].long()
        value_ids = node_features[:, 1].long()
        
        type_emb = self.node_type_embedding(type_ids)
        value_emb = self.node_value_embedding(value_ids)
        
        return torch.cat([type_emb, value_emb], dim=-1)

class FormulaEncoder(nn.Module):
    """Complete formula encoder combining node embeddings and graph encoding"""
    
    def __init__(self,
                 node_type_vocab_size: int,
                 node_value_vocab_size: int,
                 node_embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 output_dim: int = 1024,
                 num_layers: int = 16,
                 heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize formula encoder
        
        Args:
            node_type_vocab_size: Size of node type vocabulary
            node_value_vocab_size: Size of node value vocabulary
            node_embedding_dim: Dimension for node embeddings
            hidden_dim: Hidden dimension for graph encoder
            output_dim: Output embedding dimension
            num_layers: Number of graph convolution layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_embedding = NodeEmbedding(
            node_type_vocab_size,
            node_value_vocab_size, 
            node_embedding_dim
        )
        
        self.graph_encoder = GraphEncoder(
            input_dim=self.node_embedding.output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, 2]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Formula embeddings [batch_size, output_dim]
        """
        # Node embeddings
        node_emb = self.node_embedding(x)
        
        # Graph encoding
        graph_emb = self.graph_encoder(node_emb, edge_index, batch)
        
        return graph_emb
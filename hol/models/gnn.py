import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for processing HOL Light formulas.
    Uses message passing to compute embeddings of formulas.
    """
    
    def __init__(self, node_dim, hidden_dim, output_dim, num_hops):
        """
        Initialize the GNN.
        
        Args:
            node_dim: Dimension of node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_hops: Number of message passing hops
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hops = num_hops
        
        # Node type embedding
        self.type_embedding = nn.Embedding(10, node_dim)  # Allow for 10 types
        
        # Initial node feature projection
        self.node_proj = nn.Linear(32, node_dim)  # From parser feature size
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(2 * node_dim, node_dim) for _ in range(num_hops)
        ])
        
        # Update layers
        self.update_layers = nn.ModuleList([
            nn.Linear(2 * node_dim, node_dim) for _ in range(num_hops)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(node_dim, output_dim)
    
    def forward(self, nodes, edges, node_types):
        """
        Forward pass through the GNN.
        
        Args:
            nodes: Node features [num_nodes, node_features]
            edges: Edge indices [2, num_edges]
            node_types: Node type indices [num_nodes]
            
        Returns:
            graph_embedding: Graph embedding [output_dim]
        """
        # Handle empty graphs
        if nodes.shape[0] == 0:
            return torch.zeros(self.output_dim, device=nodes.device)
        
        # Project node features
        h = self.node_proj(nodes)
        
        # Add node type embeddings
        h = h + self.type_embedding(node_types)
        
        # Message passing
        for i in range(self.num_hops):
            if edges.shape[1] == 0:
                # No edges in the graph
                continue
                
            # Get source and destination nodes
            src, dst = edges
            
            # Compute messages
            src_h = h[src]
            dst_h = h[dst]
            messages = torch.cat([src_h, dst_h], dim=1)
            messages = self.message_layers[i](messages)
            
            # Aggregate messages (sum)
            agg_messages = torch.zeros_like(h)
            for j in range(edges.shape[1]):
                agg_messages[dst[j]] += messages[j]
            
            # Update node representations
            h_new = torch.cat([h, agg_messages], dim=1)
            h_new = self.update_layers[i](h_new)
            h = F.relu(h_new)
        
        # Global pooling (max)
        graph_embedding = torch.max(h, dim=0)[0]
        
        # Project to output dimension
        graph_embedding = self.output_proj(graph_embedding)
        
        return graph_embedding
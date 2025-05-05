import torch
import numpy as np
from typing import Tuple, Dict, Any


def static_reduction(graph: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Perform static reduction on the graph by pruning edges in the farthest α-percentile.
    
    Args:
        graph: Node coordinates tensor of shape [batch_size, num_nodes, 2]
        alpha: Percentage of farthest edges to prune (default: 0.1)
        
    Returns:
        Adjacency matrix of shape [batch_size, num_nodes, num_nodes] with pruned edges
    """
    batch_size, num_nodes, _ = graph.shape
    
    # Compute pairwise Euclidean distances
    x = graph.unsqueeze(2)  # [batch_size, num_nodes, 1, 2]
    y = graph.unsqueeze(1)  # [batch_size, 1, num_nodes, 2]
    distances = torch.sqrt(torch.sum((x - y) ** 2, dim=-1))  # [batch_size, num_nodes, num_nodes]
    
    # For each node, identify the farthest alpha-percentile of nodes
    threshold = int((1 - alpha) * num_nodes)
    
    # Sort distances and create an adjacency matrix (1 for valid edges, 0 for pruned)
    _, indices = torch.sort(distances, dim=-1)
    
    # Create masks where 1s indicate valid edges (not in the farthest alpha-percentile)
    adjacency = torch.zeros_like(distances)
    for b in range(batch_size):
        for i in range(num_nodes):
            # Get indices of valid neighbors
            valid_neighbors = indices[b, i, :threshold]
            # Set valid edges to 1
            adjacency[b, i, valid_neighbors] = 1.0
    
    return adjacency 
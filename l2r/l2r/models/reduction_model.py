import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional


class ReductionEmbedding(nn.Module):
    """Embedding layer for the reduction model."""
    
    def __init__(self, embedding_dim: int = 128, problem_type: str = 'tsp'):
        """
        Args:
            embedding_dim: Dimension of the embedding
            problem_type: Type of the problem ('tsp' or 'cvrp')
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.problem_type = problem_type
        
        if problem_type == 'tsp':
            # For TSP, we only have coordinates
            self.embedding = nn.Linear(2, embedding_dim)
        elif problem_type == 'cvrp':
            # For CVRP, we have separate embeddings for depot and nodes
            self.depot_embedding = nn.Linear(2, embedding_dim)  # Depot has only coordinates
            self.node_embedding = nn.Linear(3, embedding_dim)   # Nodes have coordinates and demand
            self.last_embedding = nn.Linear(embedding_dim + 1, embedding_dim)  # Last node + remaining capacity
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def forward(self, 
                nodes: torch.Tensor, 
                first_node_idx: torch.Tensor = None, 
                last_node_idx: torch.Tensor = None,
                demands: Optional[torch.Tensor] = None,
                remaining_capacity: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            nodes: Node features tensor [batch_size, num_nodes, feature_dim]
            first_node_idx: Index of the first node in partial solution [batch_size]
            last_node_idx: Index of the last node in partial solution [batch_size]
            demands: Node demands tensor [batch_size, num_nodes] (for CVRP)
            remaining_capacity: Remaining capacity [batch_size] (for CVRP)
            
        Returns:
            Tuple containing:
                - node_embeddings: Embeddings for all nodes [batch_size, num_nodes, embedding_dim]
                - context_embedding: Context embedding for the partial solution [batch_size, embedding_dim]
        """
        batch_size, num_nodes, _ = nodes.shape
        
        if self.problem_type == 'tsp':
            # Embed all nodes
            node_embeddings = self.embedding(nodes)
            
            # Get embeddings for first and last nodes in the partial solution
            first_embeddings = torch.gather(
                node_embeddings, 1, 
                first_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, self.embedding_dim)
            ).squeeze(1)
            
            last_embeddings = torch.gather(
                node_embeddings, 1,
                last_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, self.embedding_dim)
            ).squeeze(1)
            
            # Context embedding is a function of first and last node embeddings
            context_embedding = first_embeddings + last_embeddings
            
        elif self.problem_type == 'cvrp':
            # Process depot separately (index 0)
            depot_features = nodes[:, 0, :2]  # First node is the depot
            depot_embedding = self.depot_embedding(depot_features).unsqueeze(1)
            
            # Process other nodes (coordinates and demands)
            node_features = torch.cat([nodes[:, 1:, :2], demands[:, 1:].unsqueeze(-1)], dim=-1)
            other_nodes_embedding = self.node_embedding(node_features)
            
            # Combine depot and other nodes
            node_embeddings = torch.cat([depot_embedding, other_nodes_embedding], dim=1)
            
            # Get last node embedding and combine with remaining capacity
            last_embeddings = torch.gather(
                node_embeddings, 1,
                last_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, self.embedding_dim)
            ).squeeze(1)
            
            # Combine last node embedding with remaining capacity
            last_with_capacity = torch.cat([last_embeddings, remaining_capacity.unsqueeze(-1)], dim=-1)
            context_embedding = self.last_embedding(last_with_capacity)
            
        return node_embeddings, context_embedding


class AttentionLayer(nn.Module):
    """Attention layer for the reduction model."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.compatibility_matrix = nn.Linear(embedding_dim, embedding_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, 
                context_embedding: torch.Tensor, 
                node_embeddings: torch.Tensor,
                distances: torch.Tensor) -> torch.Tensor:
        """
        Calculate the potential scores for all feasible nodes.
        
        Args:
            context_embedding: Context embedding of the partial solution [batch_size, embedding_dim]
            node_embeddings: Node embeddings [batch_size, num_nodes, embedding_dim]
            distances: Distances from the last node to all other nodes [batch_size, num_nodes]
            
        Returns:
            potential_scores: Scores for all nodes [batch_size, num_nodes]
        """
        batch_size, num_nodes, embedding_dim = node_embeddings.shape
        
        # Project node embeddings into key-value pairs
        keys = self.key_projection(node_embeddings)  # [batch_size, num_nodes, embedding_dim]
        values = self.value_projection(node_embeddings)  # [batch_size, num_nodes, embedding_dim]
        
        # Attention mechanism
        # Reshape context embedding for broadcasting [batch_size, 1, embedding_dim]
        context = context_embedding.unsqueeze(1)
        
        # Calculate attention scores
        attention_scores = torch.matmul(context, keys.transpose(1, 2)) / (embedding_dim ** 0.5)  # [batch_size, 1, num_nodes]
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Compute context attention
        context_attention = torch.matmul(attention_weights, values).squeeze(1)  # [batch_size, embedding_dim]
        
        # Calculate potential scores using the attention outputs
        compatibility = self.compatibility_matrix(node_embeddings)  # [batch_size, num_nodes, embedding_dim]
        
        # Calculate raw scores
        raw_scores = torch.matmul(context_attention.unsqueeze(1), compatibility.transpose(1, 2)).squeeze(1)
        raw_scores = raw_scores / (embedding_dim ** 0.5)
        
        # Apply sigmoid activation and subtract normalized distances
        # Normalize distances to [0, 1] by dividing by sqrt(2)
        normalized_distances = distances / (2.0 ** 0.5)
        
        potential_scores = self.sigmoid(raw_scores) - normalized_distances
        
        return potential_scores


class ReductionModel(nn.Module):
    """
    Learning-based reduction model to dynamically evaluate the potential of feasible nodes
    and adaptively reduce the search space at each construction step.
    """
    
    def __init__(self, 
                embedding_dim: int = 128, 
                problem_type: str = 'tsp'):
        """
        Args:
            embedding_dim: Dimension of the embedding
            problem_type: Type of the problem ('tsp' or 'cvrp')
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.problem_type = problem_type
        
        # Embedding layer
        self.embedding = ReductionEmbedding(embedding_dim, problem_type)
        
        # Attention layer
        self.attention = AttentionLayer(embedding_dim)
        
    def forward(self, 
                nodes: torch.Tensor, 
                adjacency: torch.Tensor,
                first_node_idx: torch.Tensor, 
                last_node_idx: torch.Tensor,
                visited_mask: torch.Tensor,
                demands: Optional[torch.Tensor] = None,
                remaining_capacity: Optional[torch.Tensor] = None,
                k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            nodes: Node features tensor [batch_size, num_nodes, feature_dim]
            adjacency: Adjacency matrix from static reduction [batch_size, num_nodes, num_nodes]
            first_node_idx: Index of the first node in partial solution [batch_size]
            last_node_idx: Index of the last node in partial solution [batch_size]
            visited_mask: Mask for visited nodes [batch_size, num_nodes]
            demands: Node demands tensor [batch_size, num_nodes] (for CVRP)
            remaining_capacity: Remaining capacity [batch_size] (for CVRP)
            k: Number of candidate nodes to select
            
        Returns:
            Tuple containing:
                - candidate_indices: Indices of the top-k candidate nodes [batch_size, k]
                - candidate_scores: Scores of the top-k candidate nodes [batch_size, k]
        """
        batch_size, num_nodes, _ = nodes.shape
        
        # Get node embeddings and context embedding
        node_embeddings, context_embedding = self.embedding(
            nodes, first_node_idx, last_node_idx, demands, remaining_capacity
        )
        
        # Calculate distances from the last node to all other nodes
        last_node_coords = torch.gather(
            nodes, 1, 
            last_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, 2)
        ).squeeze(1)
        
        # Calculate Euclidean distances
        distances = torch.sqrt(torch.sum((nodes - last_node_coords.unsqueeze(1)) ** 2, dim=-1))
        
        # Calculate potential scores for all nodes
        potential_scores = self.attention(context_embedding, node_embeddings, distances)
        
        # Create feasibility mask based on:
        # 1. Adjacency matrix from static reduction
        # 2. Visited nodes
        # 3. For CVRP: demand vs remaining capacity constraints
        feasibility_mask = torch.ones_like(visited_mask, dtype=torch.float)
        
        # Apply adjacency constraint from static reduction
        # For each node, get the adjacency row for the last node
        adjacency_rows = torch.gather(
            adjacency, 1,
            last_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, num_nodes)
        ).squeeze(1)
        feasibility_mask = feasibility_mask * adjacency_rows
        
        # Mask visited nodes (0 for visited, 1 for unvisited)
        feasibility_mask = feasibility_mask * (~visited_mask).float()
        
        # For CVRP, apply capacity constraints
        if self.problem_type == 'cvrp' and demands is not None and remaining_capacity is not None:
            capacity_mask = (demands <= remaining_capacity.unsqueeze(1)).float()
            feasibility_mask = feasibility_mask * capacity_mask
        
        # Apply the feasibility mask to potential scores
        masked_scores = potential_scores * feasibility_mask - 1e9 * (1 - feasibility_mask)
        
        # Select top-k candidate nodes
        effective_k = min(k, int(feasibility_mask.sum(dim=1).max().item()))
        
        # If there are no feasible nodes, return empty tensors
        if effective_k == 0:
            return torch.zeros(batch_size, 0, dtype=torch.long, device=nodes.device), \
                   torch.zeros(batch_size, 0, device=nodes.device)
        
        # Get top-k candidate nodes
        candidate_scores, candidate_indices = torch.topk(masked_scores, effective_k, dim=1)
        
        # If effective_k < k, pad with -1 indices and -inf scores
        if effective_k < k:
            padding_indices = torch.full(
                (batch_size, k - effective_k), -1, 
                dtype=torch.long, device=nodes.device
            )
            candidate_indices = torch.cat([candidate_indices, padding_indices], dim=1)
            
            padding_scores = torch.full(
                (batch_size, k - effective_k), float('-inf'), 
                device=nodes.device
            )
            candidate_scores = torch.cat([candidate_scores, padding_scores], dim=1)
        
        return candidate_indices, candidate_scores 
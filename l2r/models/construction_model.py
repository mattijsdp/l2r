import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional

from l2r.models.attention import AAFM


class NormalizationLayer(nn.Module):
    """
    Coordinate normalization layer for the local construction model.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, nodes: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates based on the candidate nodes.
        
        Args:
            nodes: Node coordinates [batch_size, num_nodes, 2]
            candidate_indices: Indices of the candidate nodes [batch_size, k]
            
        Returns:
            normalized_nodes: Normalized node coordinates [batch_size, num_nodes, 2]
        """
        batch_size, num_nodes, _ = nodes.shape
        
        # Create a mask for valid indices (not -1 padding)
        valid_mask = (candidate_indices != -1)
        
        # Initialize normalized nodes
        normalized_nodes = torch.zeros_like(nodes)
        
        # Process each instance in the batch separately
        for b in range(batch_size):
            # Get valid candidate indices
            valid_candidates = candidate_indices[b][valid_mask[b]]
            
            if valid_candidates.numel() == 0:
                # No valid candidates, return original coordinates
                normalized_nodes[b] = nodes[b]
                continue
            
            # Get coordinates of valid candidates
            candidate_coords = nodes[b, valid_candidates]
            
            # Calculate min and max coordinates
            x_min, _ = candidate_coords[:, 0].min(dim=0)
            x_max, _ = candidate_coords[:, 0].max(dim=0)
            y_min, _ = candidate_coords[:, 1].min(dim=0)
            y_max, _ = candidate_coords[:, 1].max(dim=0)
            
            # Calculate scaling factor
            r = 1.0 / max(x_max - x_min, y_max - y_min).clamp(min=1e-6)
            
            # Normalize coordinates
            normalized_nodes[b, :, 0] = r * (nodes[b, :, 0] - x_min)
            normalized_nodes[b, :, 1] = r * (nodes[b, :, 1] - y_min)
            
            # Ensure coordinates are within [0, 1]
            normalized_nodes[b] = torch.clamp(normalized_nodes[b], 0.0, 1.0)
        
        return normalized_nodes


class ConstructionEmbedding(nn.Module):
    """
    Embedding layer for the local construction model.
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
        
        # Basic coordinate embedding
        self.coord_embedding = nn.Linear(2, embedding_dim)
        
        # Demand and capacity embeddings for CVRP
        if problem_type == 'cvrp':
            self.demand_embedding = nn.Linear(1, embedding_dim)
            self.load_embedding = nn.Linear(1, embedding_dim)
        
        # Learnable matrices for first and last nodes
        self.W1 = nn.Linear(embedding_dim, embedding_dim)
        self.W2 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, 
                nodes: torch.Tensor,
                first_node_idx: torch.Tensor,
                last_node_idx: torch.Tensor,
                candidate_indices: torch.Tensor,
                demands: Optional[torch.Tensor] = None,
                remaining_capacity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            nodes: Node coordinates [batch_size, num_nodes, 2]
            first_node_idx: Index of the first node in partial solution [batch_size]
            last_node_idx: Index of the last node in partial solution [batch_size]
            candidate_indices: Indices of the candidate nodes [batch_size, k]
            demands: Node demands tensor [batch_size, num_nodes] (for CVRP)
            remaining_capacity: Remaining capacity [batch_size] (for CVRP)
            
        Returns:
            graph_embedding: Graph embedding [batch_size, 2+k, embedding_dim]
        """
        batch_size = nodes.shape[0]
        k = candidate_indices.shape[1]
        
        # Basic coordinate embedding
        all_coord_embeddings = self.coord_embedding(nodes)
        
        if self.problem_type == 'tsp':
            # Extract first and last node embeddings
            first_node_emb = torch.gather(
                all_coord_embeddings, 1,
                first_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, self.embedding_dim)
            ).squeeze(1)
            
            last_node_emb = torch.gather(
                all_coord_embeddings, 1,
                last_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, self.embedding_dim)
            ).squeeze(1)
            
            # Apply special treatment to first and last nodes
            first_node_emb = self.W1(first_node_emb)
            last_node_emb = self.W2(last_node_emb)
            
            # Extract candidate embeddings
            candidate_embeddings = torch.zeros(batch_size, k, self.embedding_dim, device=nodes.device)
            
            # Create a mask for valid indices (not -1 padding)
            valid_mask = (candidate_indices != -1)
            
            # Fill in embeddings for valid candidates
            for b in range(batch_size):
                valid_candidates = candidate_indices[b][valid_mask[b]]
                if valid_candidates.numel() > 0:
                    candidate_embeddings[b, :valid_candidates.numel()] = all_coord_embeddings[b, valid_candidates]
            
            # Combine all embeddings
            graph_embedding = torch.cat([
                first_node_emb.unsqueeze(1),
                last_node_emb.unsqueeze(1),
                candidate_embeddings
            ], dim=1)
            
        elif self.problem_type == 'cvrp':
            # Process demands and capacity for CVRP
            norm_demands = demands / (remaining_capacity.unsqueeze(1) + 1e-6)
            demand_embeddings = self.demand_embedding(norm_demands.unsqueeze(-1))
            
            # Combine coordinate and demand embeddings
            all_embeddings = all_coord_embeddings
            for b in range(batch_size):
                # Add demand embeddings to all nodes except depot
                all_embeddings[b, 1:] = all_embeddings[b, 1:] + demand_embeddings[b, 1:]
            
            # Extract first and last node embeddings
            first_node_emb = torch.gather(
                all_embeddings, 1,
                first_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, self.embedding_dim)
            ).squeeze(1)
            
            last_node_emb = torch.gather(
                all_embeddings, 1,
                last_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, self.embedding_dim)
            ).squeeze(1)
            
            # Add remaining capacity information to first and last nodes
            load_embedding = self.load_embedding(remaining_capacity.unsqueeze(-1))
            first_node_emb = self.W1(first_node_emb) + load_embedding
            last_node_emb = self.W2(last_node_emb) + load_embedding
            
            # Extract candidate embeddings
            candidate_embeddings = torch.zeros(batch_size, k, self.embedding_dim, device=nodes.device)
            
            # Create a mask for valid indices (not -1 padding)
            valid_mask = (candidate_indices != -1)
            
            # Fill in embeddings for valid candidates
            for b in range(batch_size):
                valid_candidates = candidate_indices[b][valid_mask[b]]
                if valid_candidates.numel() > 0:
                    candidate_embeddings[b, :valid_candidates.numel()] = all_embeddings[b, valid_candidates]
            
            # Combine all embeddings
            graph_embedding = torch.cat([
                first_node_emb.unsqueeze(1),
                last_node_emb.unsqueeze(1),
                candidate_embeddings
            ], dim=1)
        
        return graph_embedding


class AttentionLayer(nn.Module):
    """
    Attention layer for the local construction model.
    """
    
    def __init__(self, embedding_dim: int = 128):
        """
        Args:
            embedding_dim: Dimension of the embedding
        """
        super().__init__()
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Attention mechanism
        self.attention = AAFM(embedding_dim)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, 
                x: torch.Tensor,
                distances: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            distances: Distance matrix [batch_size, seq_len, seq_len]
            
        Returns:
            output: Output tensor [batch_size, seq_len, embedding_dim]
        """
        # Self-attention sub-layer with layer normalization and residual connection
        h = self.layer_norm1(x)
        h = x + self.attention(h, h, h, distances)
        
        # Feed-forward sub-layer with layer normalization and residual connection
        output = h + self.ff_network(self.layer_norm2(h))
        
        return output


class ConstructionModel(nn.Module):
    """
    Local Solution Construction Model for the L2R framework.
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 num_layers: int = 6,
                 problem_type: str = 'tsp',
                 clipping: float = 10.0):
        """
        Args:
            embedding_dim: Dimension of the embedding
            num_layers: Number of attention layers
            problem_type: Type of the problem ('tsp' or 'cvrp')
            clipping: Clipping parameter for compatibility calculation
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.problem_type = problem_type
        self.clipping = clipping
        
        # Normalization layer
        self.normalization = NormalizationLayer()
        
        # Embedding layer
        self.embedding = ConstructionEmbedding(embedding_dim, problem_type)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(embedding_dim) for _ in range(num_layers)
        ])
        
        # Compatibility projection
        self.compatibility_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, 
                nodes: torch.Tensor,
                first_node_idx: torch.Tensor,
                last_node_idx: torch.Tensor,
                candidate_indices: torch.Tensor,
                demands: Optional[torch.Tensor] = None,
                remaining_capacity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            nodes: Node coordinates [batch_size, num_nodes, 2]
            first_node_idx: Index of the first node in partial solution [batch_size]
            last_node_idx: Index of the last node in partial solution [batch_size]
            candidate_indices: Indices of the candidate nodes [batch_size, k]
            demands: Node demands tensor [batch_size, num_nodes] (for CVRP)
            remaining_capacity: Remaining capacity [batch_size] (for CVRP)
            
        Returns:
            log_probs: Log probabilities for selecting candidate nodes [batch_size, k]
        """
        batch_size, num_nodes, _ = nodes.shape
        k = candidate_indices.shape[1]
        
        # Normalize coordinates
        normalized_nodes = self.normalization(nodes, candidate_indices)
        
        # Get initial embeddings
        graph_embedding = self.embedding(
            normalized_nodes, first_node_idx, last_node_idx, candidate_indices,
            demands, remaining_capacity
        )  # [batch_size, 2+k, embedding_dim]
        
        # Compute pairwise distances for the attention mechanism
        # Include first, last, and candidate nodes
        seq_len = 2 + k
        positions = torch.zeros(batch_size, seq_len, 2, device=nodes.device)
        
        # Fill in coordinates for first and last nodes
        positions[:, 0] = torch.gather(
            normalized_nodes, 1,
            first_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, 2)
        ).squeeze(1)
        
        positions[:, 1] = torch.gather(
            normalized_nodes, 1,
            last_node_idx.view(batch_size, 1, 1).expand(batch_size, 1, 2)
        ).squeeze(1)
        
        # Fill in coordinates for candidate nodes
        valid_mask = (candidate_indices != -1)
        for b in range(batch_size):
            valid_candidates = candidate_indices[b][valid_mask[b]]
            if valid_candidates.numel() > 0:
                positions[b, 2:2+valid_candidates.numel()] = normalized_nodes[b, valid_candidates]
        
        # Compute pairwise distances
        x = positions.unsqueeze(2)  # [batch_size, seq_len, 1, 2]
        y = positions.unsqueeze(1)  # [batch_size, 1, seq_len, 2]
        distances = torch.sqrt(torch.sum((x - y) ** 2, dim=-1))  # [batch_size, seq_len, seq_len]
        
        # Process through attention layers
        h = graph_embedding
        for layer in self.attention_layers:
            h = layer(h, distances)
        
        # Extract embeddings for compatibility calculation
        first_emb = h[:, 0]  # [batch_size, embedding_dim]
        last_emb = h[:, 1]   # [batch_size, embedding_dim]
        candidate_embs = h[:, 2:]  # [batch_size, k, embedding_dim]
        
        # Calculate context embedding
        context_emb = first_emb + last_emb  # [batch_size, embedding_dim]
        
        # Project candidate embeddings
        projected_candidates = self.compatibility_projection(candidate_embs)  # [batch_size, k, embedding_dim]
        
        # Calculate compatibility scores
        compatibility = torch.matmul(
            context_emb.unsqueeze(1),
            projected_candidates.transpose(1, 2)
        ).squeeze(1) / (self.embedding_dim ** 0.5)  # [batch_size, k]
        
        # Get distances from last node to candidates
        candidate_distances = torch.zeros(batch_size, k, device=nodes.device)
        for b in range(batch_size):
            valid_candidates = candidate_indices[b][valid_mask[b]]
            if valid_candidates.numel() > 0:
                last_coords = torch.gather(
                    nodes[b], 0,
                    last_node_idx[b].view(1, 1).expand(1, 2)
                ).squeeze(0)
                
                candidate_coords = nodes[b, valid_candidates]
                candidate_distances[b, :valid_candidates.numel()] = torch.sqrt(
                    torch.sum((candidate_coords - last_coords.unsqueeze(0)) ** 2, dim=-1)
                )
        
        # Calculate adaptation bias based on number of candidates and distances
        f_value = k * candidate_distances
        
        # Calculate raw logits with clipping
        raw_logits = self.clipping * torch.tanh(compatibility + f_value)
        
        # Mask invalid candidates
        logits = raw_logits - 1e9 * (~valid_mask).float()
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs 
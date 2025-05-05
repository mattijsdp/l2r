import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class AAFM(nn.Module):
    """
    Adaptation Attention Free Module for local construction model.
    Based on the paper's Appendix C.
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: Dimension of the model
        """
        super().__init__()
        self.d_model = d_model
        
        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Learnable scaling parameter for adaptation bias
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                distances: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q: Query tensor [batch_size, seq_len_q, d_model]
            k: Key tensor [batch_size, seq_len_k, d_model]
            v: Value tensor [batch_size, seq_len_v, d_model]
            distances: Distance matrix [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q, _ = q.shape
        seq_len_k = k.shape[1]
        
        # Project query, key, and value
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Apply sigmoid to query
        q_sigmoid = torch.sigmoid(q)
        
        # Calculate adaptation bias based on distances
        if distances is not None:
            # Calculate adaptation bias: a_ij = -alpha * log2(N) * d_ij
            n = seq_len_k  # Total number of nodes
            adaptation_bias = -self.alpha * torch.log2(torch.tensor(n, dtype=torch.float)) * distances
        else:
            adaptation_bias = torch.zeros(batch_size, seq_len_q, seq_len_k, device=q.device)
        
        # Calculate attention weights
        exp_k = torch.exp(k)
        exp_a = torch.exp(adaptation_bias)
        
        # Element-wise product of exp(k) and v
        weighted_values = exp_k.unsqueeze(1) * v.unsqueeze(1)  # [batch_size, 1, seq_len_k, d_model]
        
        # Numerator: exp(A)(exp(K) ⊙ V)
        numerator = torch.matmul(exp_a, weighted_values)  # [batch_size, seq_len_q, d_model]
        
        # Denominator: exp(A)exp(K)
        denominator = torch.matmul(exp_a, exp_k.unsqueeze(-1))  # [batch_size, seq_len_q, 1]
        
        # Calculate attention
        attention = numerator / (denominator + 1e-9)  # Add epsilon for numerical stability
        
        # Element-wise product with sigmoid query
        output = q_sigmoid * attention
        
        return output 
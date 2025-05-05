import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import mlflow
from typing import Dict, Any, List, Tuple, Optional

from l2r.models.reduction_model import ReductionModel
from l2r.models.construction_model import ConstructionModel
from l2r.utils.static_reduction import static_reduction


class L2RModule(pl.LightningModule):
    """
    PyTorch Lightning module for the L2R framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Problem settings
        self.problem_type = config.get('problem_type', 'tsp')
        self.static_reduction_alpha = config.get('static_reduction_alpha', 0.1)
        self.max_search_space_size = config.get('max_search_space_size', 20 if self.problem_type == 'tsp' else 50)
        
        # Model settings
        self.embedding_dim = config.get('embedding_dim', 128)
        self.num_attention_layers = config.get('num_attention_layers', 6)
        self.clipping = config.get('clipping', 10.0)
        
        # Training settings
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.lr_decay = config.get('lr_decay', 0.98)
        self.grad_clip_val = config.get('grad_clip_val', 1.0)
        self.baseline_exp_beta = config.get('baseline_exp_beta', 0.8)
        
        # Initialize models
        self.reduction_model = ReductionModel(
            embedding_dim=self.embedding_dim,
            problem_type=self.problem_type
        )
        
        self.construction_model = ConstructionModel(
            embedding_dim=self.embedding_dim,
            num_layers=self.num_attention_layers,
            problem_type=self.problem_type,
            clipping=self.clipping
        )
        
        # Baseline parameters for REINFORCE
        self.register_buffer('baseline', torch.zeros(1))
        self.baseline_exp_beta = 0.8
        
        # Initialize automatic logging
        mlflow.pytorch.autolog()
        
    def forward(self, 
                nodes: torch.Tensor, 
                demands: Optional[torch.Tensor] = None, 
                capacities: Optional[torch.Tensor] = None,
                greedy: bool = True) -> torch.Tensor:
        """
        Forward pass through the L2R model.
        
        Args:
            nodes: Node coordinates [batch_size, num_nodes, 2]
            demands: Node demands [batch_size, num_nodes] (for CVRP)
            capacities: Vehicle capacities [batch_size] (for CVRP)
            greedy: Whether to use greedy decoding (True) or sampling (False)
            
        Returns:
            tours: Constructed tours [batch_size, num_nodes]
        """
        batch_size, num_nodes, _ = nodes.shape
        device = nodes.device
        
        # Static reduction - prune the graph
        adjacency = static_reduction(nodes, self.static_reduction_alpha)
        
        # Initialize tours with first node (randomly for training, fixed for inference)
        tours = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
        
        if self.problem_type == 'tsp':
            # For TSP, start from random node during training, node 0 during inference
            if self.training:
                tours[:, 0] = torch.randint(0, num_nodes, (batch_size,), device=device)
            else:
                tours[:, 0] = torch.zeros(batch_size, dtype=torch.long, device=device)
                
            # Initialize remaining capacity (not used for TSP)
            remaining_capacities = None
            
        elif self.problem_type == 'cvrp':
            # For CVRP, always start from depot (node 0)
            tours[:, 0] = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Initialize remaining capacity
            remaining_capacities = capacities.clone()
        
        # Initialize visited mask
        visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        for b in range(batch_size):
            visited_mask[b, tours[b, 0]] = True
        
        # Construct solution step-by-step
        for step in range(1, num_nodes):
            # Get current partial solution
            first_node_idx = tours[:, 0]
            last_node_idx = tours[:, step-1]
            
            # Get candidate nodes from reduction model
            candidate_indices, _ = self.reduction_model(
                nodes, adjacency, first_node_idx, last_node_idx, visited_mask,
                demands, remaining_capacities, self.max_search_space_size
            )
            
            # Get log probabilities from construction model
            log_probs = self.construction_model(
                nodes, first_node_idx, last_node_idx, candidate_indices,
                demands, remaining_capacities
            )
            
            # Select next node (greedy or sampling)
            if greedy:
                _, selected_idx = log_probs.max(dim=-1)
            else:
                selected_idx = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
            
            # Convert local indices (within candidates) to global indices
            next_node_idx = torch.gather(
                candidate_indices, 1, 
                selected_idx.view(batch_size, 1)
            ).squeeze(-1)
            
            # Handle case when no valid candidates are available
            invalid_mask = (next_node_idx == -1)
            if invalid_mask.any():
                # Find unvisited nodes for instances with no valid candidates
                for b in range(batch_size):
                    if invalid_mask[b]:
                        # Find first unvisited node
                        unvisited = (~visited_mask[b]).nonzero(as_tuple=True)[0]
                        if len(unvisited) > 0:
                            next_node_idx[b] = unvisited[0]
                        else:
                            # All nodes visited, go back to start node
                            next_node_idx[b] = tours[b, 0]
            
            # Add selected node to the tour
            tours[:, step] = next_node_idx
            
            # Update visited mask
            for b in range(batch_size):
                visited_mask[b, next_node_idx[b]] = True
            
            # Update remaining capacity for CVRP
            if self.problem_type == 'cvrp' and demands is not None and remaining_capacities is not None:
                for b in range(batch_size):
                    next_node = next_node_idx[b].item()
                    if next_node > 0:  # Not the depot
                        remaining_capacities[b] -= demands[b, next_node]
                    else:  # Return to depot, reset capacity
                        remaining_capacities[b] = capacities[b]
        
        # For CVRP, ensure all tours end at the depot
        if self.problem_type == 'cvrp':
            # Add depot as the final node
            depot_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            tours = torch.cat([tours, depot_indices], dim=1)
        
        return tours
    
    def calculate_tour_length(self, 
                              nodes: torch.Tensor, 
                              tours: torch.Tensor) -> torch.Tensor:
        """
        Calculate the total length of each tour.
        
        Args:
            nodes: Node coordinates [batch_size, num_nodes, 2]
            tours: Tour indices [batch_size, tour_len]
            
        Returns:
            lengths: Tour lengths [batch_size]
        """
        batch_size, tour_len = tours.shape
        
        # Gather coordinates of the tour nodes
        tour_nodes = torch.gather(
            nodes, 1, 
            tours.unsqueeze(-1).expand(batch_size, tour_len, 2)
        )
        
        # Compute distances between consecutive nodes
        shifted_nodes = torch.roll(tour_nodes, -1, dims=1)
        segment_lengths = torch.sqrt(torch.sum((tour_nodes - shifted_nodes) ** 2, dim=2))
        
        # For TSP, compute full cycle length
        if self.problem_type == 'tsp':
            # Sum all segments to get the tour length
            tour_lengths = segment_lengths.sum(dim=1)
        
        # For CVRP, don't include the final return-to-depot connection
        else:
            # Sum all segments except the last one
            tour_lengths = segment_lengths[:, :-1].sum(dim=1)
        
        return tour_lengths
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.config.get('weight_decay', 0.0)
        )
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self.lr_decay
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_reward'
        }
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            loss: Reinforcement learning loss
        """
        # Unpack batch
        if self.problem_type == 'tsp':
            nodes = batch
            demands = None
            capacities = None
        else:  # CVRP
            nodes, demands, capacities = batch
            
        # Greedy rollout (baseline)
        with torch.no_grad():
            baseline_tours = self.forward(nodes, demands, capacities, greedy=True)
            baseline_lengths = self.calculate_tour_length(nodes, baseline_tours)
            baseline_reward = -baseline_lengths  # Negative length as reward
        
        # Policy rollout (sampled)
        sampled_tours = self.forward(nodes, demands, capacities, greedy=False)
        sampled_lengths = self.calculate_tour_length(nodes, sampled_tours)
        sampled_reward = -sampled_lengths  # Negative length as reward
        
        # Update baseline using exponential moving average
        if self.training:
            self.baseline = self.baseline_exp_beta * self.baseline + \
                           (1 - self.baseline_exp_beta) * baseline_reward.mean()
        
        # Calculate advantage
        advantage = sampled_reward - baseline_reward
        
        # Calculate loss using REINFORCE
        loss = -(advantage.detach() * sampled_reward).mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_reward', sampled_reward.mean(), prog_bar=True)
        self.log('train_baseline_reward', baseline_reward.mean())
        self.log('train_advantage', advantage.mean())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            reward: Negative tour length
        """
        # Unpack batch
        if self.problem_type == 'tsp':
            nodes = batch
            demands = None
            capacities = None
        else:  # CVRP
            nodes, demands, capacities = batch
            
        # Greedy rollout
        tours = self.forward(nodes, demands, capacities, greedy=True)
        tour_lengths = self.calculate_tour_length(nodes, tours)
        reward = -tour_lengths.mean()  # Negative length as reward
        
        # Log metrics
        self.log('val_reward', reward, prog_bar=True)
        self.log('val_tour_length', tour_lengths.mean())
        
        # Log example tour for visualization
        if batch_idx == 0:
            example_idx = 0
            example_nodes = nodes[example_idx].cpu().numpy()
            example_tour = tours[example_idx].cpu().numpy()
            
            # Log coordinates and tour as artifacts
            mlflow.log_dict(
                {
                    'nodes': example_nodes.tolist(),
                    'tour': example_tour.tolist(),
                    'length': tour_lengths[example_idx].item()
                },
                'example_tour.json'
            )
        
        return reward
    
    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            reward: Negative tour length
        """
        # Unpack batch
        if self.problem_type == 'tsp':
            nodes = batch
            demands = None
            capacities = None
        else:  # CVRP
            nodes, demands, capacities = batch
            
        # Greedy rollout
        tours = self.forward(nodes, demands, capacities, greedy=True)
        tour_lengths = self.calculate_tour_length(nodes, tours)
        reward = -tour_lengths.mean()  # Negative length as reward
        
        # Log metrics
        self.log('test_reward', reward)
        self.log('test_tour_length', tour_lengths.mean())
        
        return reward 
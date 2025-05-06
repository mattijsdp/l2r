#!/usr/bin/env python3
"""Test script for the L2R model components."""

import os
import sys
import pytest
import torch
import numpy as np
from typing import Dict, Any, Tuple, cast
from omegaconf import OmegaConf

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l2r.utils.static_reduction import static_reduction
from l2r.models.reduction_model import ReductionModel
from l2r.models.construction_model import ConstructionModel
from l2r.models.l2r_module import L2RModule


@pytest.fixture
def test_data() -> Tuple[torch.Tensor, int, int]:
    """Create test data for testing."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a batch of test data (4 instances, 10 nodes each)
    batch_size = 4
    num_nodes = 10
    nodes = torch.rand(batch_size, num_nodes, 2)
    
    return nodes, batch_size, num_nodes


@pytest.fixture
def adjacency_data(test_data: Tuple[torch.Tensor, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate adjacency matrix from test data."""
    nodes, _, _ = test_data
    adjacency = static_reduction(nodes, alpha=0.2)
    return nodes, adjacency


@pytest.fixture
def candidate_data(adjacency_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate candidate indices for testing construction model."""
    nodes, adjacency = adjacency_data
    batch_size, num_nodes, _ = nodes.shape
    
    # Initialize reduction model
    reduction_model = ReductionModel(embedding_dim=64, problem_type='tsp')
    
    # Create test inputs
    first_node_idx = torch.zeros(batch_size, dtype=torch.long)
    last_node_idx = torch.ones(batch_size, dtype=torch.long)
    visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool)
    
    # Mark the first and last nodes as visited
    for i in range(batch_size):
        visited_mask[i, first_node_idx[i]] = True
        visited_mask[i, last_node_idx[i]] = True
    
    # Forward pass
    candidate_indices, _ = reduction_model(
        nodes, adjacency, first_node_idx, last_node_idx, visited_mask, k=3
    )
    
    return nodes, candidate_indices, first_node_idx, last_node_idx


@pytest.fixture
def tsp_config() -> Dict[str, Any]:
    """Create a basic config for TSP."""
    # Create a basic config
    config = {
        'problem_type': 'tsp',
        'static_reduction_alpha': 0.2,
        'max_search_space_size': 5,
        'embedding_dim': 64,
        'num_attention_layers': 2,
        'clipping': 10.0,
        'learning_rate': 1e-4,
        'lr_decay': 0.98,
    }
    return config


@pytest.fixture
def cvrp_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create CVRP test data."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a batch of test data (4 instances, 20 nodes each including depot)
    batch_size = 4
    num_nodes = 20  # Including depot
    nodes = torch.rand(batch_size, num_nodes, 2)
    demands = torch.zeros(batch_size, num_nodes)
    
    # Set demands for non-depot nodes (1-10 units)
    demands[:, 1:] = torch.rand(batch_size, num_nodes - 1) * 9 + 1
    
    # Set vehicle capacities (sum of demands / 4)
    capacities = torch.sum(demands, dim=1) / 4
    
    return nodes, demands, capacities


def test_static_reduction(test_data: Tuple[torch.Tensor, int, int]) -> None:
    """Test the static reduction function."""
    nodes, batch_size, num_nodes = test_data
    
    # Apply static reduction
    adjacency = static_reduction(nodes, alpha=0.2)
    
    # Check the output shape
    expected_shape = (batch_size, num_nodes, num_nodes)
    assert adjacency.shape == expected_shape, f"Expected shape {expected_shape}, got {adjacency.shape}"
    
    # Check that the output is a binary matrix (0s and 1s)
    assert torch.all((adjacency == 0) | (adjacency == 1)), "Adjacency matrix should contain only 0s and 1s"
    
    # Check that the diagonal is all 0s (no self-loops)
    for i in range(batch_size):
        for j in range(num_nodes):
            assert adjacency[i, j, j] == 0, f"Diagonal element ({i}, {j}, {j}) should be 0"
    
    # Check that we've pruned approximately alpha% of the edges
    expected_ones = int(num_nodes * (num_nodes - 1) * (1 - 0.2))
    actual_ones = int(adjacency.sum().item())
    tolerance = int(num_nodes * 0.5)  # Allow some tolerance
    assert abs(expected_ones - actual_ones) <= tolerance, f"Expected ~{expected_ones} edges, got {actual_ones}"


def test_reduction_model(adjacency_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test the learning-based reduction model."""
    nodes, adjacency = adjacency_data
    batch_size, num_nodes, _ = nodes.shape
    
    # Initialize reduction model
    reduction_model = ReductionModel(embedding_dim=64, problem_type='tsp')
    
    # Create test inputs
    first_node_idx = torch.zeros(batch_size, dtype=torch.long)
    last_node_idx = torch.ones(batch_size, dtype=torch.long)
    visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool)
    
    # Mark the first and last nodes as visited
    for i in range(batch_size):
        visited_mask[i, first_node_idx[i]] = True
        visited_mask[i, last_node_idx[i]] = True
    
    # Forward pass
    candidate_indices, candidate_scores = reduction_model(
        nodes, adjacency, first_node_idx, last_node_idx, visited_mask, k=3
    )
    
    # Check output shapes
    assert candidate_indices.shape == (batch_size, 3), f"Expected shape {(batch_size, 3)}, got {candidate_indices.shape}"
    assert candidate_scores.shape == (batch_size, 3), f"Expected shape {(batch_size, 3)}, got {candidate_scores.shape}"
    
    # Check that the candidate indices are valid (in range [0, num_nodes-1] or -1 for padding)
    valid_indices = (candidate_indices == -1) | ((candidate_indices >= 0) & (candidate_indices < num_nodes))
    assert torch.all(valid_indices), "Invalid candidate indices found"
    
    # Check that visited nodes are not in the candidates
    for i in range(batch_size):
        for j in range(candidate_indices.shape[1]):
            idx = candidate_indices[i, j].item()
            if idx != -1:  # Skip padding indices
                assert not visited_mask[i, idx], f"Visited node {idx} should not be in candidates"


def test_construction_model(candidate_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    """Test the local construction model."""
    nodes, candidate_indices, first_node_idx, last_node_idx = candidate_data
    batch_size = nodes.shape[0]
    
    # Initialize construction model
    construction_model = ConstructionModel(embedding_dim=64, num_layers=2, problem_type='tsp')
    
    # Forward pass
    log_probs = construction_model(nodes, first_node_idx, last_node_idx, candidate_indices)
    
    # Check output shape
    assert log_probs.shape == (batch_size, candidate_indices.shape[1]), \
        f"Expected shape {(batch_size, candidate_indices.shape[1])}, got {log_probs.shape}"
    
    # Check that log probs sum to 1 for each instance
    for i in range(batch_size):
        probs = torch.exp(log_probs[i])
        assert abs(probs.sum().item() - 1.0) < 1e-5, f"Probabilities should sum to 1, got {probs.sum().item()}"


def test_l2r_module(tsp_config: Dict[str, Any]) -> None:
    """Test the complete L2R module."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a batch of test data (4 instances, 20 nodes each)
    batch_size = 4
    num_nodes = 20
    nodes = torch.rand(batch_size, num_nodes, 2)
    
    # Convert to OmegaConf and then back to dict to fix type compatibility
    cfg = OmegaConf.create(tsp_config)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # Ensure we have a Dict[str, Any]
    typed_config = cast(Dict[str, Any], config_dict)
    
    # Initialize L2R module
    l2r_module = L2RModule(typed_config)
    
    # Forward pass (greedy decoding)
    tours = l2r_module(nodes, greedy=True)
    
    # Check output shape
    assert tours.shape == (batch_size, num_nodes), f"Expected shape {(batch_size, num_nodes)}, got {tours.shape}"
    
    # Check that tours contain each node exactly once
    for i in range(batch_size):
        tour = tours[i]
        unique_nodes = torch.unique(tour)
        assert unique_nodes.numel() == num_nodes, \
            f"Tour should contain each node exactly once, got {unique_nodes.numel()} unique nodes"
    
    # Calculate tour lengths
    tour_lengths = l2r_module.calculate_tour_length(nodes, tours)
    
    # Check tour lengths shape
    assert tour_lengths.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {tour_lengths.shape}"
    
    # Forward pass (sampling)
    sampled_tours = l2r_module(nodes, greedy=False)
    
    # Check that sampled tours are different from greedy tours
    assert not torch.all(tours == sampled_tours), "Sampled tours should be different from greedy tours"


def test_cvrp(tsp_config: Dict[str, Any], cvrp_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    """Test L2R on CVRP instances."""
    nodes, demands, capacities = cvrp_data
    batch_size, num_nodes = nodes.shape[:2]
    
    # Update config for CVRP
    config = tsp_config.copy()
    config['problem_type'] = 'cvrp'
    
    # Convert to OmegaConf and then back to dict to fix type compatibility
    cfg = OmegaConf.create(config)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # Ensure we have a Dict[str, Any]
    typed_config = cast(Dict[str, Any], config_dict)
    
    # Initialize L2R module
    l2r_module = L2RModule(typed_config)
    
    # Forward pass (greedy decoding)
    tours = l2r_module(nodes, demands, capacities, greedy=True)
    
    # For CVRP, the tour length should be num_nodes + 1 (includes return to depot)
    assert tours.shape == (batch_size, num_nodes + 1), \
        f"Expected shape {(batch_size, num_nodes + 1)}, got {tours.shape}"
    
    # Check that tours start and end at the depot (node 0)
    for i in range(batch_size):
        assert tours[i, 0] == 0, f"Tour should start at the depot (node 0), got {tours[i, 0]}"
        assert tours[i, -1] == 0, f"Tour should end at the depot (node 0), got {tours[i, -1]}"
    
    # Calculate tour lengths
    tour_lengths = l2r_module.calculate_tour_length(nodes, tours)
    
    # Check tour lengths shape
    assert tour_lengths.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {tour_lengths.shape}"


if __name__ == "__main__":
    """Run all tests using pytest."""
    print("Starting L2R model tests...")
    sys.exit(pytest.main([__file__, "-v"])) 
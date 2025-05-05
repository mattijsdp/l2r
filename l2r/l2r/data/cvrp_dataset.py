"""CVRP dataset and data module implementation."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, List, Dict, Any


class CVRPDataset(Dataset):
    """
    Dataset for the Capacitated Vehicle Routing Problem (CVRP).
    Generates random instances with uniformly distributed coordinates and demands.
    """
    
    def __init__(self, 
                 num_samples: int = 10000, 
                 num_nodes: int = 100, 
                 capacity: float = 50.0,
                 demand_min: float = 1.0,
                 demand_max: float = 10.0,
                 seed: Optional[int] = None):
        """
        Initialize the CVRP dataset.
        
        Args:
            num_samples: Number of problem instances
            num_nodes: Number of nodes in each instance (excluding depot)
            capacity: Vehicle capacity
            demand_min: Minimum demand
            demand_max: Maximum demand
            seed: Random seed
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes  # Not including depot
        self.capacity = capacity
        self.demand_min = demand_min
        self.demand_max = demand_max
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random node coordinates
        # Node 0 is the depot, nodes 1 to num_nodes are the customers
        self.nodes = np.random.uniform(0, 1, size=(num_samples, num_nodes + 1, 2))
        
        # Generate random demands for each node (except depot)
        self.demands = np.zeros((num_samples, num_nodes + 1))
        self.demands[:, 1:] = np.random.uniform(
            demand_min, demand_max, size=(num_samples, num_nodes)
        )
        
        # Create vehicle capacities
        self.capacities = np.full(num_samples, capacity)
    
    def __len__(self) -> int:
        """Return the number of instances in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a CVRP instance by index.
        
        Args:
            idx: Instance index
            
        Returns:
            Tuple containing:
                - nodes: Node coordinates [num_nodes+1, 2] (including depot)
                - demands: Node demands [num_nodes+1] (including depot, which has 0 demand)
                - capacity: Vehicle capacity [1]
        """
        nodes = self.nodes[idx]
        demands = self.demands[idx]
        capacity = self.capacities[idx]
        
        return (
            torch.tensor(nodes, dtype=torch.float),
            torch.tensor(demands, dtype=torch.float),
            torch.tensor(capacity, dtype=torch.float)
        )


class CVRPDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for CVRP."""
    
    def __init__(self, 
                 batch_size: int = 64, 
                 num_nodes: int = 100, 
                 capacity: float = 50.0,
                 demand_min: float = 1.0,
                 demand_max: float = 10.0,
                 num_samples: int = 10000,
                 num_workers: int = 4,
                 val_size: int = 1000,
                 test_size: int = 1000,
                 seed: Optional[int] = None):
        """
        Initialize the CVRP data module.
        
        Args:
            batch_size: Batch size
            num_nodes: Number of nodes in each instance (excluding depot)
            capacity: Vehicle capacity
            demand_min: Minimum demand
            demand_max: Maximum demand
            num_samples: Number of training samples
            num_workers: Number of workers for data loading
            val_size: Number of validation samples
            test_size: Number of test samples
            seed: Random seed
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.capacity = capacity
        self.demand_min = demand_min
        self.demand_max = demand_max
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up the data module.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = CVRPDataset(
                num_samples=self.num_samples,
                num_nodes=self.num_nodes,
                capacity=self.capacity,
                demand_min=self.demand_min,
                demand_max=self.demand_max,
                seed=self.seed
            )
            
            self.val_dataset = CVRPDataset(
                num_samples=self.val_size,
                num_nodes=self.num_nodes,
                capacity=self.capacity,
                demand_min=self.demand_min,
                demand_max=self.demand_max,
                seed=self.seed + 1 if self.seed is not None else None
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = CVRPDataset(
                num_samples=self.test_size,
                num_nodes=self.num_nodes,
                capacity=self.capacity,
                demand_min=self.demand_min,
                demand_max=self.demand_max,
                seed=self.seed + 2 if self.seed is not None else None
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 
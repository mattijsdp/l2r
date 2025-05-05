"""TSP dataset and data module implementation."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, List, Dict, Any


class TSPDataset(Dataset):
    """
    Dataset for the Traveling Salesman Problem (TSP).
    Generates random instances with uniformly distributed coordinates.
    """
    
    def __init__(self, num_samples: int = 10000, num_nodes: int = 100, seed: Optional[int] = None):
        """
        Initialize the TSP dataset.
        
        Args:
            num_samples: Number of problem instances
            num_nodes: Number of nodes in each instance
            seed: Random seed
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random node coordinates
        self.nodes = np.random.uniform(0, 1, size=(num_samples, num_nodes, 2))
    
    def __len__(self) -> int:
        """Return the number of instances in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a TSP instance by index.
        
        Args:
            idx: Instance index
            
        Returns:
            nodes: Node coordinates [num_nodes, 2]
        """
        nodes = self.nodes[idx]
        return torch.tensor(nodes, dtype=torch.float)


class TSPDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for TSP."""
    
    def __init__(self, 
                 batch_size: int = 128, 
                 num_nodes: int = 100, 
                 num_samples: int = 10000,
                 num_workers: int = 4,
                 val_size: int = 1000,
                 test_size: int = 1000,
                 seed: Optional[int] = None):
        """
        Initialize the TSP data module.
        
        Args:
            batch_size: Batch size
            num_nodes: Number of nodes in each instance
            num_samples: Number of training samples
            num_workers: Number of workers for data loading
            val_size: Number of validation samples
            test_size: Number of test samples
            seed: Random seed
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
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
            self.train_dataset = TSPDataset(
                num_samples=self.num_samples,
                num_nodes=self.num_nodes,
                seed=self.seed
            )
            
            self.val_dataset = TSPDataset(
                num_samples=self.val_size,
                num_nodes=self.num_nodes,
                seed=self.seed + 1 if self.seed is not None else None
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = TSPDataset(
                num_samples=self.test_size,
                num_nodes=self.num_nodes,
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
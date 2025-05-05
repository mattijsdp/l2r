#!/usr/bin/env python3
"""Training script for the L2R model"""

import os
import random
import numpy as np
import torch
import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from l2r.models import L2RModule
from l2r.data import TSPDataModule, CVRPDataModule


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function"""
    # Print config
    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    set_random_seed(cfg.seed)
    
    # Initialize MLFlow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=f"l2r_{cfg.problem_type}",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"),
    )
    
    # Log config to MLFlow
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg))
    
    # Initialize data module
    if cfg.problem_type == "tsp":
        data_module = TSPDataModule(
            batch_size=cfg.training.batch_size,
            num_nodes=cfg.training.training_scale,
            num_samples=cfg.training.batches_per_epoch * cfg.training.batch_size,
            num_workers=os.cpu_count(),
        )
    elif cfg.problem_type == "cvrp":
        data_module = CVRPDataModule(
            batch_size=cfg.training.batch_size,
            num_nodes=cfg.training.training_scale,
            capacity=cfg.model.training_capacity,
            num_samples=cfg.training.batches_per_epoch * cfg.training.batch_size,
            num_workers=os.cpu_count(),
        )
    else:
        raise ValueError(f"Unsupported problem type: {cfg.problem_type}")
    
    # Initialize model
    model = L2RModule(cfg)
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_reward:.2f}",
        monitor="val_reward",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    early_stopping = EarlyStopping(
        monitor="val_reward",
        mode="max",
        patience=10,
        verbose=True,
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=cfg.training.get("gpus_per_node", 1),
        strategy=cfg.training.get("strategy", "auto"),
        precision=cfg.training.get("precision", 32),
        gradient_clip_val=cfg.training.grad_clip_val,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=mlflow_logger,
        log_every_n_steps=cfg.training.log_interval,
        num_sanity_val_steps=2,
    )
    
    # Train model
    trainer.fit(model, datamodule=data_module)
    
    # Test model
    trainer.test(model, datamodule=data_module)
    
    # Save final model
    trainer.save_checkpoint("final_model.ckpt")
    

if __name__ == "__main__":
    main() 
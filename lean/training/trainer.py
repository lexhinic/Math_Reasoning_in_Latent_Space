import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from models.reasoning_model import ReasoningModel, ReasoningLoss
import sys 
sys.path.append('/home/stu4/formal_reasoning/baseline/Math_Reasoning_in_Latent_Space/')
from baseline.Math_Reasoning_in_Latent_Space.lean.config import config

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for the reasoning model"""
    
    def __init__(self,
                 model: ReasoningModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = "cuda"):
        """
        Initialize trainer
        
        Args:
            model: The reasoning model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Loss function
        self.criterion = ReasoningLoss(
            success_weight=config.model.success_prediction_weight,
            embedding_weight=config.model.embedding_prediction_weight
        )
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        self.global_step = 0
        self.epoch = 0
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_success_loss = 0.0
        total_embedding_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            success_logits, predicted_embeddings = self.model(
                batch['target_batch'], 
                batch['parameter_batch']
            )
            
            # Get target embeddings for successful rewrites
            target_embeddings = None
            if batch['has_results'].any():
                # Encode result formulas to get target embeddings
                target_embeddings = self.model.encode_target(
                    batch['result_batch'].x,
                    batch['result_batch'].edge_index,
                    batch['result_batch'].batch
                )
            else:
                # Create dummy target embeddings
                target_embeddings = torch.zeros_like(predicted_embeddings)
            
            # Compute loss
            loss_dict = self.criterion(
                success_logits=success_logits,
                predicted_embeddings=predicted_embeddings,
                success_targets=batch['success'],
                target_embeddings=target_embeddings,
                success_mask=batch['has_results']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.model.gradient_clip)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_success_loss += loss_dict['success_loss'].item()
            total_embedding_loss += loss_dict['embedding_loss'].item()
            num_batches += 1
            
            # Collect predictions for AUC computation
            predictions = torch.sigmoid(success_logits).cpu().numpy()
            targets = batch['success'].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/BatchLoss', loss_dict['total_loss'].item(), self.global_step)
                self.writer.add_scalar('Train/SuccessLoss', loss_dict['success_loss'].item(), self.global_step)
                self.writer.add_scalar('Train/EmbeddingLoss', loss_dict['embedding_loss'].item(), self.global_step)
            
            self.global_step += 1
        
        # Compute AUC
        train_auc = roc_auc_score(all_targets, all_predictions)
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_success_loss = total_success_loss / num_batches
        avg_embedding_loss = total_embedding_loss / num_batches
        
        return {
            'loss': avg_loss,
            'success_loss': avg_success_loss,
            'embedding_loss': avg_embedding_loss,
            'auc': train_auc
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_success_loss = 0.0
        total_embedding_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                success_logits, predicted_embeddings = self.model(
                    batch['target_batch'], 
                    batch['parameter_batch']
                )
                
                # Get target embeddings
                target_embeddings = None
                if batch['has_results'].any():
                    target_embeddings = self.model.encode_target(
                        batch['result_batch'].x,
                        batch['result_batch'].edge_index,
                        batch['result_batch'].batch
                    )
                else:
                    target_embeddings = torch.zeros_like(predicted_embeddings)
                
                # Compute loss
                loss_dict = self.criterion(
                    success_logits=success_logits,
                    predicted_embeddings=predicted_embeddings,
                    success_targets=batch['success'],
                    target_embeddings=target_embeddings,
                    success_mask=batch['has_results']
                )
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item()
                total_success_loss += loss_dict['success_loss'].item()
                total_embedding_loss += loss_dict['embedding_loss'].item()
                num_batches += 1
                
                # Collect predictions
                predictions = torch.sigmoid(success_logits).cpu().numpy()
                targets = batch['success'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute AUC
        val_auc = roc_auc_score(all_targets, all_predictions)
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_success_loss = total_success_loss / num_batches
        avg_embedding_loss = total_embedding_loss / num_batches
        
        return {
            'loss': avg_loss,
            'success_loss': avg_success_loss,
            'embedding_loss': avg_embedding_loss,
            'auc': val_auc,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary containing training history
        """
        train_history = {
            'loss': [], 'success_loss': [], 'embedding_loss': [], 'auc': []
        }
        val_history = {
            'loss': [], 'success_loss': [], 'embedding_loss': [], 'auc': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train AUC: {train_metrics['auc']:.4f} "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val AUC: {val_metrics['auc']:.4f} "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Tensorboard logging
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/AUC', train_metrics['auc'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save history
            for key in train_history:
                train_history[key].append(train_metrics[key])
                val_history[key].append(val_metrics[key])
            
            # Save best model
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                logger.info(f"New best model saved with validation AUC: {self.best_val_auc:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        return {
            'train_history': train_history,
            'val_history': val_history
        }
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_auc': self.best_val_auc,
            'global_step': self.global_step
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = os.path.join(config.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint
        
        Args:
            filename: Name of the checkpoint file to load
        """
        load_path = os.path.join(config.checkpoint_dir, filename)
        
        if not os.path.exists(load_path):
            logger.error(f"Checkpoint file not found: {load_path}")
            return
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_auc = checkpoint['best_val_auc']
        self.global_step = checkpoint['global_step']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {load_path}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch data to the specified device
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Batch with data moved to device
        """
        batch['target_batch'] = batch['target_batch'].to(self.device)
        batch['parameter_batch'] = batch['parameter_batch'].to(self.device)
        batch['result_batch'] = batch['result_batch'].to(self.device)
        batch['success'] = batch['success'].to(self.device)
        batch['has_results'] = batch['has_results'].to(self.device)
        batch['indices'] = batch['indices'].to(self.device)
        
        return batch
    
    def plot_roc_curve(self, predictions: np.ndarray, targets: np.ndarray, save_path: str):
        """
        Plot and save ROC curve
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(targets, predictions)
        auc = roc_auc_score(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")

def create_trainer(model: ReasoningModel,
                  train_loader: DataLoader,
                  val_loader: DataLoader) -> Trainer:
    """
    Create a trainer with default configuration
    
    Args:
        model: The reasoning model
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        Configured trainer
    """
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.model.num_epochs,
        eta_min=1e-6
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device
    )
    
    return trainer
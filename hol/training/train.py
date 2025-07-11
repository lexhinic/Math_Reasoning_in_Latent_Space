import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from data_processing.dataset import get_dataloader

def train_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        device: Device to use for training
        
    Returns:
        avg_loss: Average loss for the epoch
        avg_success_loss: Average rewrite success prediction loss
        avg_embed_loss: Average embedding prediction loss
    """
    model.train()
    
    total_loss = 0.0
    total_success_loss = 0.0
    total_embed_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], list):
                # Handle lists of tensors
                for i in range(len(batch[key])):
                    if isinstance(batch[key][i], torch.Tensor):
                        batch[key][i] = batch[key][i].to(device)
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss
        loss, success_loss, embed_loss = model.compute_loss(batch)
        
        # Backpropagate
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_success_loss += success_loss.item()
        total_embed_loss += embed_loss.item()
        num_batches += 1
    
    # Compute averages
    avg_loss = total_loss / max(1, num_batches)
    avg_success_loss = total_success_loss / max(1, num_batches)
    avg_embed_loss = total_embed_loss / max(1, num_batches)
    
    return avg_loss, avg_success_loss, avg_embed_loss

def validate(model, dataloader, device):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        device: Device to use for validation
        
    Returns:
        avg_loss: Average loss for the validation set
        avg_success_loss: Average rewrite success prediction loss
        avg_embed_loss: Average embedding prediction loss
    """
    model.eval()
    
    total_loss = 0.0
    total_success_loss = 0.0
    total_embed_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None:
                continue
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], list):
                    # Handle lists of tensors
                    for i in range(len(batch[key])):
                        if isinstance(batch[key][i], torch.Tensor):
                            batch[key][i] = batch[key][i].to(device)
                elif isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Compute loss
            loss, success_loss, embed_loss = model.compute_loss(batch)
            
            # Update statistics
            total_loss += loss.item()
            total_success_loss += success_loss.item()
            total_embed_loss += embed_loss.item()
            num_batches += 1
    
    # Compute averages
    avg_loss = total_loss / max(1, num_batches)
    avg_success_loss = total_success_loss / max(1, num_batches)
    avg_embed_loss = total_embed_loss / max(1, num_batches)
    
    return avg_loss, avg_success_loss, avg_embed_loss

def train(model, config, device):
    """
    Train the model.
    
    Args:
        model: Model to train
        config: Configuration object
        device: Device to use for training
        
    Returns:
        model: Trained model
    """
    # Get dataloaders
    train_dataloader = get_dataloader(config, "train")
    val_dataloader = get_dataloader(config, "val")
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train for specified number of epochs
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train for one epoch
        train_loss, train_success_loss, train_embed_loss = train_epoch(
            model, train_dataloader, optimizer, device
        )
        
        # Validate
        val_loss, val_success_loss, val_embed_loss = validate(
            model, val_dataloader, device
        )
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f} (Success: {train_success_loss:.4f}, Embed: {train_embed_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Success: {val_success_loss:.4f}, Embed: {val_embed_loss:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
            print(f"New best model with validation loss: {best_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model
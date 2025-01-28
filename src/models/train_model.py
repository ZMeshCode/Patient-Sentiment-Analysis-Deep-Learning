import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Tuple

from sentiment_model import PatientSentimentModel, PatientFeedbackDataset
from config.training_config import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        """
        Initialize the model trainer.
        
        Args:
            config (TrainingConfig): Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize model
        self.model = PatientSentimentModel(
            num_classes=config.num_classes,
            dropout=config.dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare training and validation data.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders
        """
        # Load datasets
        train_df = pd.read_csv(self.config.train_data)
        val_df = pd.read_csv(self.config.val_data)
        
        # Create datasets
        train_dataset = PatientFeedbackDataset(
            texts=train_df['feedback'].tolist(),
            labels=train_df['sentiment'].tolist(),
            max_length=self.config.max_length
        )
        
        val_dataset = PatientFeedbackDataset(
            texts=val_df['feedback'].tolist(),
            labels=val_df['sentiment'].tolist(),
            max_length=self.config.max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        # Verify device placement
        logging.info(f"Training on device: {self.device}")
        logging.info(f"Model is on device: {next(self.model.parameters()).device}")
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Verify tensor device placement (only for first batch)
            if batch_idx == 0:
                logging.info(f"Input tensor device: {input_ids.device}")
                logging.info(f"Labels tensor device: {labels.device}")
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            
            # Log more frequently
            if batch_idx % (self.config.log_interval // 4) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })
                logging.info(
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        """
        # Create model directory if it doesn't exist
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        # Create learning rate scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            logging.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }
            training_history.append(metrics)
            
            logging.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.config.model_dir, 'best_model.pth')
                )
            
            # Save checkpoint if needed
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.model_dir,
                    f'checkpoint_epoch_{epoch + 1}.pth'
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }, checkpoint_path)
        
        # Save training history
        history_path = os.path.join(
            self.config.model_dir,
            f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        
        logging.info(f"\nTraining completed! History saved to {history_path}")

def main():
    """
    Main function to train the model.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer with config
    trainer = ModelTrainer(TrainingConfig)
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main() 
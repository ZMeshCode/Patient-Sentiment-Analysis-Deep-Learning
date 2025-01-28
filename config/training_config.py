import torch

"""
Configuration for model training.
"""

class TrainingConfig:
    # Model parameters
    model_name = 'bert-base-uncased'
    num_classes = 3
    max_length = 512
    dropout = 0.1
    
    # Training parameters
    batch_size = 32  # Increased batch size since we'll use MPS
    learning_rate = 2e-5
    num_epochs = 5
    warmup_steps = 0
    weight_decay = 0.01
    
    # Data parameters
    train_split = 0.8
    random_seed = 42
    
    # Paths
    model_dir = 'data/models'
    train_data = 'data/processed/patient_feedback_train_20250128_112659.csv'
    val_data = 'data/processed/patient_feedback_val_20250128_112659.csv'
    
    # Training device - Use MPS if available, otherwise CPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Logging
    log_interval = 100  # Print metrics every N steps
    save_interval = 1   # Save model every N epochs 
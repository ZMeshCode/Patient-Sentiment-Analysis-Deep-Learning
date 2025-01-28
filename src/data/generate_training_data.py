import os
import pandas as pd
from synthetic_data_generator import PatientFeedbackGenerator
import logging
from datetime import datetime

def generate_and_save_data(
    num_samples: int = 10000,
    output_dir: str = 'data/processed',
    train_split: float = 0.8
):
    """
    Generate synthetic patient feedback data and save train/val splits.
    
    Args:
        num_samples (int): Number of samples to generate
        output_dir (str): Directory to save the datasets
        train_split (float): Proportion of data to use for training
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = PatientFeedbackGenerator()
    
    # Generate dataset
    logging.info(f"Generating {num_samples} synthetic feedback samples...")
    df = generator.generate_dataset(num_samples=num_samples)
    
    # Calculate split sizes
    train_size = int(len(df) * train_split)
    
    # Split the data
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Save datasets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    train_path = os.path.join(output_dir, f'patient_feedback_train_{timestamp}.csv')
    val_path = os.path.join(output_dir, f'patient_feedback_val_{timestamp}.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    # Log statistics
    logging.info("\nDataset Statistics:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Training samples: {len(train_df)}")
    logging.info(f"Validation samples: {len(val_df)}")
    
    logging.info("\nSentiment distribution in training set:")
    logging.info(train_df['sentiment'].value_counts(normalize=True))
    
    logging.info("\nSentiment distribution in validation set:")
    logging.info(val_df['sentiment'].value_counts(normalize=True))
    
    logging.info(f"\nDatasets saved to:")
    logging.info(f"Training: {train_path}")
    logging.info(f"Validation: {val_path}")
    
    # Save metadata about the generation
    metadata = {
        'timestamp': timestamp,
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'train_split': train_split,
        'sentiment_weights': generator.sentiment_weights,
        'train_file': os.path.basename(train_path),
        'val_file': os.path.basename(val_path)
    }
    
    metadata_path = os.path.join(output_dir, f'dataset_metadata_{timestamp}.json')
    pd.Series(metadata).to_json(metadata_path)
    logging.info(f"Metadata saved to: {metadata_path}")

def main():
    """
    Main function to generate training data.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Generate data with default parameters
    generate_and_save_data()

if __name__ == "__main__":
    main() 
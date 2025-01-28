# Patient Sentiment Analysis System

A deep learning-based system for analyzing patient satisfaction and feedback in healthcare settings.

## Overview
This project implements a sophisticated sentiment analysis system designed to process and analyze patient feedback data. It uses deep learning techniques to understand patient sentiments and provide actionable insights for healthcare providers.

## Features
- Data collection and preprocessing pipeline
- Deep learning model for sentiment analysis
- Interactive dashboard for visualizing insights
- API endpoints for real-time sentiment analysis
- Automated report generation

## Project Structure
```
├── config/               # Configuration files
├── data/                # Data directory
│   ├── processed/       # Processed datasets
│   └── models/         # Trained models
├── src/                # Source code
│   ├── data/          # Data processing scripts
│   └── models/        # Model architecture and training
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Environment Setup
1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Generate synthetic training data:
```bash
python src/data/generate_training_data.py
```

2. Train the model:
```bash
python src/models/train_model.py
```

## Model Architecture
- Base: BERT (bert-base-uncased)
- Classification head for sentiment categories
- Regression head for fine-grained sentiment scores
- Optimized for M1/M2 Mac GPUs using MPS

## Training
- Batch size: 32
- Learning rate: 2e-5
- Optimizer: AdamW with weight decay
- Device: GPU (MPS) on Apple Silicon, CPU fallback

## License
MIT License

## Contributors
- [Your Name]

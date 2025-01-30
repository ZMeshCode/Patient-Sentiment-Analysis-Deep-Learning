# Patient Sentiment Analysis System

A deep learning-based system for analyzing patient satisfaction and feedback in healthcare settings.

## Screenshots

### Single Analysis View
![Single Analysis](docs/images/single_analysis.png)
*Real-time sentiment analysis of individual feedback with detailed probability breakdown and confidence scores. The interface provides immediate insights into patient sentiment with color-coded results and probability distributions.*

### Batch Analysis View
![Batch Analysis](docs/images/batch_analysis.png)
*Bulk analysis of multiple feedback entries with word cloud visualization and sentiment distribution. Features include:*
- Interactive word cloud showing most frequent terms
- Sentiment distribution charts
- Timeline analysis
- Detailed metrics and statistics
- Export functionality for further analysis

### Historical Analysis View
![Historical Analysis](docs/images/historical_analysis.png)
*Comprehensive timeline view showing sentiment trends and statistics over time. This view enables healthcare providers to:*
- Track sentiment patterns over time
- Identify trends in patient satisfaction
- Export historical data for reporting
- Generate insights from aggregated feedback

## Overview
This project implements a sophisticated sentiment analysis system designed to process and analyze patient feedback data. It uses deep learning techniques to understand patient sentiments and provide actionable insights for healthcare providers.

## Data Generation
The model is trained on synthetically generated patient feedback data to ensure privacy and data protection. The synthetic data generator creates realistic patient feedback scenarios while maintaining the statistical properties of real healthcare feedback data. This approach allows us to:
- Protect patient privacy by not using real patient data
- Generate large-scale training datasets
- Create diverse feedback scenarios
- Control the distribution of sentiment categories

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

## Technologies & Skills
### Deep Learning & ML
- PyTorch - Deep learning framework
- Transformers (Hugging Face) - BERT model implementation
- scikit-learn - Data preprocessing and evaluation
- MPS (Metal Performance Shaders) - Apple Silicon GPU acceleration

### Data Processing
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- NLTK - Natural Language Processing
- spaCy - Advanced NLP tasks
- TextBlob - Sentiment analysis utilities

### Web & API
- FastAPI - REST API development
- Streamlit - Interactive dashboard
- Uvicorn - ASGI web server

### Development Tools
- Python 3.12 - Primary programming language
- Git - Version control
- Virtual Environment - Dependency management
- Black - Code formatting
- Flake8 - Code linting
- pytest - Testing framework

### Database
- SQLAlchemy - ORM for database operations
- Alembic - Database migrations
- PostgreSQL - Database (via psycopg2)

## Training
- Batch size: 32
- Learning rate: 2e-5
- Optimizer: AdamW with weight decay
- Device: GPU (MPS) on Apple Silicon, CPU fallback

## Contributors
- Bobby Meshesha

# Patient Sentiment Analysis Dashboard

A powerful tool for analyzing patient feedback using natural language processing and sentiment analysis.

## Features

### Dashboard Views
1. **Single Analysis**
   - Real-time analysis of individual feedback
   - Detailed sentiment probabilities
   - Confidence scores
   - Visual probability distribution

2. **Batch Analysis**
   - Upload CSV files with multiple feedback entries
   - Bulk sentiment analysis
   - Summary statistics and metrics
   - Interactive visualizations
   - Export functionality

3. **Historical Analysis**
   - Track all analyzed feedback over time
   - Trend analysis and patterns
   - Comprehensive statistics
   - Data export capabilities

### Visualizations
1. **Word Cloud**
   - Visual representation of most frequent terms
   - Size indicates word frequency
   - Excludes common stop words
   - Interactive display

2. **Sentiment Distribution**
   - Bar charts showing sentiment breakdown
   - Color-coded categories (Positive, Neutral, Negative)
   - Percentage distributions

3. **Timeline Analysis**
   - Sentiment trends over time
   - Interactive time-series plot
   - Track sentiment evolution

### Export Features
- Download analysis results as CSV
- Complete feedback history export
- Detailed metrics and probabilities
- Timestamp and classification data

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install additional NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

1. Start the dashboard:
```bash
cd src/dashboard
PYTHONPATH=$PYTHONPATH:../.. streamlit run app.py
```

2. Access the dashboard in your web browser at `http://localhost:8501`

3. Choose your preferred analysis mode:
   - Single Analysis: Enter individual feedback
   - Batch Analysis: Upload CSV file with feedback
   - Historical Analysis: View all analyzed feedback

## Data Format

For batch analysis, prepare your CSV file with the following format:
```csv
feedback
"Your patient feedback text here"
"Another feedback entry"
...
```

## Model Information

The sentiment analysis model uses BERT architecture fine-tuned on healthcare feedback data, providing:
- Three-class classification (Positive, Neutral, Negative)
- Confidence scores
- Sentiment probability distribution
- Numerical sentiment scores

## Key Features of the Dashboard

### Visualization Tools
- **Word Cloud**: Visual representation of frequently occurring terms in feedback
- **Sentiment Distribution**: Color-coded charts showing the breakdown of positive, neutral, and negative feedback
- **Timeline Analysis**: Track sentiment trends over time with interactive plots
- **Confidence Metrics**: Probability scores for each sentiment category

### Analysis Capabilities
- Real-time individual feedback analysis
- Batch processing of multiple feedback entries
- Historical trend analysis
- Export functionality for further analysis

### User Interface
- Clean, intuitive design
- Easy navigation between different views
- Interactive visualizations
- Responsive layout for different screen sizes

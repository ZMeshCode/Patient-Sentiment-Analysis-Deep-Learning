import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, Tuple, List
import logging
from torch.utils.data import Dataset

class PatientFeedbackDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], max_length: int = 512):
        """
        Dataset for patient feedback data.
        
        Args:
            texts (List[str]): List of feedback texts
            labels (List[int]): List of sentiment labels
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PatientSentimentModel(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.1):
        """
        Initialize the sentiment analysis model based on BERT.
        
        Args:
            num_classes (int): Number of sentiment classes (default: 3 for negative, neutral, positive)
            dropout (float): Dropout rate for regularization
        """
        super(PatientSentimentModel, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Sentiment regression head for fine-grained sentiment score
        self.sentiment_regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tokenized input sequences
            attention_mask (torch.Tensor): Attention mask for padding
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Classification logits and sentiment scores
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get classification logits and sentiment score
        logits = self.classifier(pooled_output)
        sentiment_score = self.sentiment_regressor(pooled_output)
        
        return logits, sentiment_score

class SentimentAnalyzer:
    def __init__(self, model_path: str = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_path (str, optional): Path to saved model weights
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = PatientSentimentModel().to(self.device)
        
        if model_path:
            try:
                print(f"Loading model from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        
        self.model.eval()
        
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict sentiment for a list of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Dict[str, float]]: List of predictions with class probabilities and sentiment scores
        """
        # Tokenize inputs
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            logits, sentiment_scores = self.model(
                encoded['input_ids'],
                encoded['attention_mask']
            )
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=1)
            predictions = []
            
            for prob, score in zip(probs, sentiment_scores):
                pred_dict = {
                    'negative_prob': float(prob[0].item()),
                    'neutral_prob': float(prob[1].item()),
                    'positive_prob': float(prob[2].item()),
                    'sentiment_score': float(score.item()),
                    'predicted_class': int(torch.argmax(prob).item())
                }
                print(f"Prediction details: {pred_dict}")
                predictions.append(pred_dict)
                
        return predictions

def main():
    """
    Main function to demonstrate model usage.
    """
    # Example usage
    analyzer = SentimentAnalyzer()
    
    sample_texts = [
        "The medical staff was extremely helpful and caring.",
        "I had to wait for hours without any updates.",
        "The treatment was okay, but could be better."
    ]
    
    predictions = analyzer.predict(sample_texts)
    
    for text, pred in zip(sample_texts, predictions):
        print(f"\nText: {text}")
        print(f"Prediction: {pred}")

if __name__ == "__main__":
    main() 
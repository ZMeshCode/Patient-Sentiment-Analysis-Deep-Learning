import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from typing import List, Dict, Union
import logging

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    logging.warning("NLTK data download failed. Please download manually if needed.")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def get_sentiment_score(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment scores using TextBlob.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Dictionary containing polarity and subjectivity scores
        """
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def process_feedback(self, feedback_data: Union[pd.DataFrame, List[str]]) -> pd.DataFrame:
        """
        Process a batch of patient feedback.
        
        Args:
            feedback_data: DataFrame or list of feedback texts
            
        Returns:
            pd.DataFrame: Processed feedback with sentiment scores
        """
        if isinstance(feedback_data, list):
            feedback_data = pd.DataFrame({'feedback': feedback_data})
        
        # Clean text
        feedback_data['cleaned_text'] = feedback_data['feedback'].apply(self.clean_text)
        
        # Remove stopwords
        feedback_data['processed_text'] = feedback_data['cleaned_text'].apply(self.remove_stopwords)
        
        # Calculate sentiment scores
        sentiment_scores = feedback_data['processed_text'].apply(self.get_sentiment_score)
        feedback_data['polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
        feedback_data['subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])
        
        return feedback_data

def main():
    """
    Main function to demonstrate preprocessing functionality.
    """
    # Example usage
    preprocessor = TextPreprocessor()
    
    # Sample data
    sample_feedback = [
        "The doctor was very professional and caring.",
        "Had to wait too long in the emergency room!",
        "Great experience with the nursing staff.",
    ]
    
    # Process feedback
    processed_data = preprocessor.process_feedback(sample_feedback)
    print("Processed feedback with sentiment scores:")
    print(processed_data)

if __name__ == "__main__":
    main() 
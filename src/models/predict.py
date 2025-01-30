from sentiment_model import SentimentAnalyzer
import argparse
import json

def analyze_feedback(text: str, model_path: str = "data/models/best_model.pth") -> dict:
    """
    Analyze a single piece of patient feedback.
    
    Args:
        text (str): Patient feedback text
        model_path (str): Path to the trained model
        
    Returns:
        dict: Analysis results including sentiment probabilities and scores
    """
    # Initialize analyzer with trained model
    analyzer = SentimentAnalyzer(model_path=model_path)
    
    # Get prediction
    predictions = analyzer.predict([text])
    result = predictions[0]
    
    # Map sentiment class to label
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    result["sentiment"] = sentiment_map[result["predicted_class"]]
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Analyze patient feedback sentiment")
    parser.add_argument("--text", type=str, help="Patient feedback text to analyze")
    parser.add_argument("--model", type=str, default="data/models/best_model.pth", 
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    if args.text:
        result = analyze_feedback(args.text, args.model)
        print("\nAnalysis Results:")
        print(f"Input Text: {args.text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence Scores:")
        print(f"  Positive: {result['positive_prob']:.2%}")
        print(f"  Neutral:  {result['neutral_prob']:.2%}")
        print(f"  Negative: {result['negative_prob']:.2%}")
        print(f"Sentiment Score: {result['sentiment_score']:.2f}")
    else:
        # Example usage if no text provided
        sample_texts = [
            "The medical staff was extremely helpful and caring.",
            "I had to wait for hours without any updates.",
            "The treatment was okay, but could be better."
        ]
        
        print("\nAnalyzing sample feedback:")
        for text in sample_texts:
            result = analyze_feedback(text, args.model)
            print("\n---")
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence Scores:")
            print(f"  Positive: {result['positive_prob']:.2%}")
            print(f"  Neutral:  {result['neutral_prob']:.2%}")
            print(f"  Negative: {result['negative_prob']:.2%}")
            print(f"Sentiment Score: {result['sentiment_score']:.2f}")

if __name__ == "__main__":
    main() 
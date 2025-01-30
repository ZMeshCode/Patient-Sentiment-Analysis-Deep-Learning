import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.sentiment_model import SentimentAnalyzer
except ImportError as e:
    st.error(f"Failed to import SentimentAnalyzer: {str(e)}")
    st.error(f"Current sys.path: {sys.path}")
    raise

# Initialize the sentiment analyzer
@st.cache_resource
def load_analyzer():
    model_path = os.path.join(project_root, "data/models/best_model.pth")
    st.write(f"Loading model from: {model_path}")
    st.write(f"Model file exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    analyzer = SentimentAnalyzer(model_path=model_path)
    return analyzer

# Initialize session state for feedback history
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

def analyze_and_store_feedback(analyzer, text):
    """Analyze feedback and store results in session state"""
    result = analyzer.predict([text])[0]
    
    # Debug information
    st.write("Raw prediction results:", result)
    
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    result["sentiment"] = sentiment_map[result["predicted_class"]]
    
    # Add timestamp and text
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["text"] = text
    
    # Store in history
    st.session_state.feedback_history.append(result)
    return result

def create_sentiment_chart(history):
    """Create a bar chart of sentiment distribution"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        color_discrete_map={
            'positive': '#2ecc71',
            'neutral': '#f1c40f',
            'negative': '#e74c3c'
        }
    )
    fig.update_layout(title="Sentiment Distribution")
    return fig

def create_timeline_chart(history):
    """Create a timeline of sentiment scores"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.line(
        df,
        x='timestamp',
        y='sentiment_score',
        title="Sentiment Score Timeline",
        labels={'sentiment_score': 'Sentiment Score', 'timestamp': 'Time'}
    )
    return fig

def generate_word_cloud(texts):
    """Generate word cloud from texts"""
    try:
        # Ensure NLTK data is downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Combine all texts
        text = " ".join(texts)
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        try:
            words = word_tokenize(text)
        except LookupError:
            # Fallback to simple splitting if tokenization fails
            words = text.split()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback to a basic set of stopwords if loading fails
            stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                         'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                         'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        
        words = [word for word in words if word not in stop_words]
        
        if not words:
            st.warning("No meaningful words found for word cloud generation.")
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(" ".join(words))
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def get_table_download_link(df, filename="feedback_analysis.csv"):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def main():
    st.set_page_config(
        page_title="Patient Feedback Analyzer",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Patient Feedback Sentiment Analyzer")
    
    # Load the analyzer
    analyzer = load_analyzer()
    
    # Sidebar
    st.sidebar.header("Dashboard Options")
    view_mode = st.sidebar.selectbox(
        "Select View",
        ["Single Analysis", "Batch Analysis", "Historical Analysis"]
    )
    
    if view_mode == "Single Analysis":
        st.subheader("Analyze Individual Feedback")
        
        # Text input
        text = st.text_area(
            "Enter patient feedback:",
            height=100,
            placeholder="Type or paste patient feedback here..."
        )
        
        if st.button("Analyze Sentiment"):
            if text:
                with st.spinner("Analyzing feedback..."):
                    result = analyze_and_store_feedback(analyzer, text)
                
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"Overall Sentiment: {result['sentiment'].upper()}")
                
                with col2:
                    st.metric(
                        "Sentiment Score",
                        f"{result['sentiment_score']:.2f}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Confidence",
                        f"{max(result['positive_prob'], result['neutral_prob'], result['negative_prob']):.1%}"
                    )
                
                # Detailed probabilities
                st.subheader("Sentiment Probabilities")
                probs_df = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Probability': [
                        result['positive_prob'],
                        result['neutral_prob'],
                        result['negative_prob']
                    ]
                })
                
                fig = px.bar(
                    probs_df,
                    x='Sentiment',
                    y='Probability',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#f1c40f',
                        'Negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig)
            
            else:
                st.warning("Please enter some feedback text to analyze.")
    
    elif view_mode == "Batch Analysis":
        st.subheader("Analyze Multiple Feedback Entries")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with feedback (column name should be 'feedback' or 'Feedback')",
            type=['csv']
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # Case-insensitive column check
            feedback_col = None
            for col in df.columns:
                if col.lower() == 'feedback':
                    feedback_col = col
                    break
            
            if feedback_col is None:
                st.error("CSV file must contain a column named 'feedback' or 'Feedback'!")
                st.write("Available columns:", ", ".join(df.columns))
            else:
                if st.button("Analyze All"):
                    with st.spinner("Analyzing all feedback entries..."):
                        results = []
                        for text in df[feedback_col]:
                            result = analyze_and_store_feedback(analyzer, text)
                            results.append(result)
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display summary statistics
                        st.subheader("Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Total Analyzed",
                                len(results)
                            )
                        
                        with col2:
                            positive_pct = (results_df['sentiment'] == 'positive').mean()
                            st.metric(
                                "Positive Feedback",
                                f"{positive_pct:.1%}"
                            )
                        
                        with col3:
                            avg_score = results_df['sentiment_score'].mean()
                            st.metric(
                                "Average Sentiment Score",
                                f"{avg_score:.2f}"
                            )
                        
                        # Word Cloud
                        st.subheader("Word Cloud Visualization")
                        wordcloud_image = generate_word_cloud(results_df['text'].tolist())
                        if wordcloud_image:
                            st.image(wordcloud_image)
                        
                        # Display charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_chart = create_sentiment_chart(results)
                            if sentiment_chart:
                                st.plotly_chart(sentiment_chart)
                        
                        with col2:
                            timeline_chart = create_timeline_chart(results)
                            if timeline_chart:
                                st.plotly_chart(timeline_chart)
                        
                        # Export functionality
                        st.subheader("Export Results")
                        st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)
                        
                        # Display detailed results table
                        st.subheader("Detailed Results")
                        st.dataframe(
                            results_df[[
                                'text', 'sentiment', 'sentiment_score',
                                'positive_prob', 'neutral_prob', 'negative_prob'
                            ]]
                        )
    
    else:  # Historical Analysis
        st.subheader("Historical Analysis")
        
        if not st.session_state.feedback_history:
            st.info("No historical data available yet. Analyze some feedback to see the history!")
        else:
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            
            history_df = pd.DataFrame(st.session_state.feedback_history)
            
            with col1:
                st.metric(
                    "Total Analyzed",
                    len(st.session_state.feedback_history)
                )
            
            with col2:
                positive_pct = sum(1 for x in st.session_state.feedback_history if x['sentiment'] == 'positive') / len(st.session_state.feedback_history)
                st.metric(
                    "Positive Feedback",
                    f"{positive_pct:.1%}"
                )
            
            with col3:
                avg_score = sum(x['sentiment_score'] for x in st.session_state.feedback_history) / len(st.session_state.feedback_history)
                st.metric(
                    "Average Sentiment Score",
                    f"{avg_score:.2f}"
                )
            
            # Word Cloud
            st.subheader("Word Cloud Visualization")
            wordcloud_image = generate_word_cloud(history_df['text'].tolist())
            st.image(wordcloud_image)
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_chart = create_sentiment_chart(st.session_state.feedback_history)
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart)
            
            with col2:
                timeline_chart = create_timeline_chart(st.session_state.feedback_history)
                if timeline_chart:
                    st.plotly_chart(timeline_chart)
            
            # Export functionality
            st.subheader("Export Analysis History")
            st.markdown(get_table_download_link(history_df, "feedback_history.csv"), unsafe_allow_html=True)
            
            # Display history table
            st.subheader("Analysis History")
            st.dataframe(
                history_df[[
                    'timestamp', 'text', 'sentiment',
                    'sentiment_score', 'positive_prob',
                    'neutral_prob', 'negative_prob'
                ]]
            )

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
from datetime import datetime, timedelta

class PatientFeedbackGenerator:
    def __init__(self):
        # Define common topics in healthcare feedback
        self.topics = {
            'staff': {
                'positive': [
                    "The {staff_type} was extremely professional and caring",
                    "{staff_type} provided excellent care and attention",
                    "Very impressed with the {staff_type}'s expertise",
                    "The {staff_type} took time to explain everything clearly",
                    "Exceptional service from the {staff_type}"
                ],
                'neutral': [
                    "The {staff_type} was okay",
                    "{staff_type} did their job as expected",
                    "Regular interaction with the {staff_type}",
                    "The {staff_type} was present when needed"
                ],
                'negative': [
                    "The {staff_type} seemed rushed and inattentive",
                    "Poor communication from the {staff_type}",
                    "The {staff_type} was not very helpful",
                    "Disappointed with the {staff_type}'s attitude",
                    "The {staff_type} didn't address my concerns"
                ]
            },
            'waiting_time': {
                'positive': [
                    "Minimal waiting time",
                    "Got seen right away",
                    "Very efficient process",
                    "Impressed with how quickly I was seen",
                    "No delays in the appointment"
                ],
                'neutral': [
                    "Average waiting time",
                    "Waited as expected",
                    "Normal wait times",
                    "The wait was reasonable"
                ],
                'negative': [
                    "Had to wait too long",
                    "Excessive waiting time",
                    "Spent hours waiting",
                    "Unacceptable delays",
                    "The waiting time was frustrating"
                ]
            },
            'facility': {
                'positive': [
                    "The facility was clean and modern",
                    "Well-maintained hospital environment",
                    "Excellent facilities",
                    "Very comfortable environment",
                    "State-of-the-art equipment"
                ],
                'neutral': [
                    "Standard hospital environment",
                    "Regular facility conditions",
                    "Acceptable cleanliness",
                    "Basic but functional facility"
                ],
                'negative': [
                    "The facility needs updating",
                    "Poor maintenance of the premises",
                    "Cleanliness issues in the facility",
                    "Outdated equipment",
                    "Uncomfortable environment"
                ]
            },
            'treatment': {
                'positive': [
                    "Excellent treatment results",
                    "The procedure went very well",
                    "Very effective treatment",
                    "Great outcome from the treatment",
                    "Completely satisfied with the care received"
                ],
                'neutral': [
                    "Treatment was as expected",
                    "Standard procedure",
                    "Regular treatment process",
                    "The treatment was okay"
                ],
                'negative': [
                    "Unsatisfactory treatment outcome",
                    "The procedure didn't help much",
                    "Poor treatment experience",
                    "Not happy with the results",
                    "The treatment wasn't effective"
                ]
            }
        }
        
        self.staff_types = [
            "doctor", "nurse", "receptionist", "specialist",
            "surgeon", "staff member", "medical team",
            "healthcare provider", "physician", "care team"
        ]
        
        # Sentiment weights for generating balanced/imbalanced datasets
        self.sentiment_weights = {
            'positive': 0.4,  # 40% positive
            'neutral': 0.3,   # 30% neutral
            'negative': 0.3   # 30% negative
        }

    def generate_feedback(self) -> Tuple[str, int]:
        """
        Generate a single piece of patient feedback.
        
        Returns:
            Tuple[str, int]: Generated feedback text and sentiment label
        """
        # Select sentiment based on weights
        sentiment = random.choices(
            list(self.sentiment_weights.keys()),
            weights=list(self.sentiment_weights.values())
        )[0]
        
        # Convert sentiment to label
        sentiment_label = {'negative': 0, 'neutral': 1, 'positive': 2}[sentiment]
        
        # Randomly select topics to include
        num_topics = random.randint(1, 3)
        selected_topics = random.sample(list(self.topics.keys()), num_topics)
        
        feedback_parts = []
        for topic in selected_topics:
            template = random.choice(self.topics[topic][sentiment])
            if topic == 'staff':
                staff_type = random.choice(self.staff_types)
                feedback_parts.append(template.format(staff_type=staff_type))
            else:
                feedback_parts.append(template)
        
        # Add connecting words
        connectors = [". ", "! ", ". Additionally, ", ". Also, ", ". Moreover, "]
        feedback = random.choice(connectors).join(feedback_parts) + "."
        
        return feedback, sentiment_label

    def generate_dataset(
        self,
        num_samples: int,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Generate a dataset of synthetic patient feedback.
        
        Args:
            num_samples (int): Number of feedback samples to generate
            include_metadata (bool): Whether to include additional metadata
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        data = []
        
        # Generate feedback samples
        for _ in range(num_samples):
            feedback, sentiment = self.generate_feedback()
            
            sample = {
                'feedback': feedback,
                'sentiment': sentiment
            }
            
            if include_metadata:
                # Add metadata
                sample.update({
                    'timestamp': datetime.now() - timedelta(
                        days=random.randint(0, 365)
                    ),
                    'department': random.choice([
                        'Emergency', 'Cardiology', 'Pediatrics',
                        'Orthopedics', 'General Medicine', 'Surgery',
                        'Oncology', 'Neurology', 'Obstetrics'
                    ]),
                    'patient_age_group': random.choice([
                        '18-30', '31-45', '46-60', '61-75', '75+'
                    ]),
                    'visit_type': random.choice([
                        'Routine Checkup', 'Emergency',
                        'Follow-up', 'Procedure', 'Consultation'
                    ])
                })
            
            data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if include_metadata:
            # Sort by timestamp
            df = df.sort_values('timestamp')
        
        return df

def main():
    """
    Main function to demonstrate the synthetic data generation.
    """
    # Initialize generator
    generator = PatientFeedbackGenerator()
    
    # Generate a sample dataset
    print("Generating synthetic patient feedback dataset...")
    df = generator.generate_dataset(num_samples=1000)
    
    # Display sample statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts(normalize=True))
    
    # Display some sample feedback
    print("\nSample feedback:")
    samples = df.sample(5)
    for _, row in samples.iterrows():
        print(f"\nFeedback: {row['feedback']}")
        print(f"Sentiment: {row['sentiment']}")
        print(f"Department: {row['department']}")
        print(f"Visit Type: {row['visit_type']}")

if __name__ == "__main__":
    main() 
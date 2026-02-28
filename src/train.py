import os
import pandas as pd
import numpy as np
import re
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from src.preprocessing import clean_text

def map_sentiment(label):
    """
    Yelp Review Full dataset has labels 0-4.
    0, 1 -> Negative
    2 -> Neutral
    3, 4 -> Positive
    """
    if label <= 1:
        return 'Negative'
    elif label == 2:
        return 'Neutral'
    else:
        return 'Positive'

def generate_urgency_labels(text):
    """
    Rule-based urgency classification.
    """
    text_lower = str(text).lower()
    
    high_keywords = r'\b(urgent|immediately|refund now|cancel|legal)\b'
    medium_keywords = r'\b(issue|not working|error|problem)\b'
    
    if re.search(high_keywords, text_lower):
        return 'High'
    elif re.search(medium_keywords, text_lower):
        return 'Medium'
    else:
        # Default to Low for general feedback
        return 'Low'

def main():
    print("Fetching dataset...")
    # Load 50,000 rows from yelp_review_full as a proxy for support tickets
    # It has 'label' (0-4) and 'text'
    dataset = load_dataset("yelp_review_full", split="train[:50000]")
    df = pd.DataFrame(dataset)
    
    print("Mapping sentiment labels...")
    df['sentiment'] = df['label'].apply(map_sentiment)
    
    print("Generating rule-based urgency labels...")
    df['urgency'] = df['text'].apply(generate_urgency_labels)
    
    print("Preprocessing text... this may take a few minutes depending on CPU.")
    # To speed up, we are applying the clean_text function over 50k rows
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    y_sentiment = df['sentiment']
    y_urgency = df['urgency']
    
    # Optional: Save processed data
    os.makedirs('data', exist_ok=True)
    df.to_parquet('data/processed_tickets.parquet', index=False)
    
    # --- Train Sentiment Model ---
    print("Training Sentiment Classification Model...")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)
    
    sentiment_model = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced')
    sentiment_model.fit(X_train_s, y_train_s)
    
    s_preds = sentiment_model.predict(X_test_s)
    print("\n--- Sentiment Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test_s, s_preds):.4f}")
    print(classification_report(y_test_s, s_preds))
    
    # --- Train Urgency Model ---
    print("Training Urgency Classification Model...")
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y_urgency, test_size=0.2, random_state=42)
    
    # Using class_weight='balanced' since 'High' urgency might be minority class
    urgency_model = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced')
    urgency_model.fit(X_train_u, y_train_u)
    
    u_preds = urgency_model.predict(X_test_u)
    print("\n--- Urgency Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test_u, u_preds):.4f}")
    print(classification_report(y_test_u, u_preds))
    
    # --- Save Models ---
    print("\nSaving models and vectorizer...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(sentiment_model, 'models/sentiment_model.joblib')
    joblib.dump(urgency_model, 'models/urgency_model.joblib')
    
    print("Training complete! Models saved to models/ directory.")

if __name__ == "__main__":
    main()

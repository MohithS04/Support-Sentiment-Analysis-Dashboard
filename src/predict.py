import os
import joblib

from src.preprocessing import clean_text

_vectorizer = None
_sentiment_model = None
_urgency_model = None

def load_models():
    global _vectorizer, _sentiment_model, _urgency_model
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    vec_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    s_path = os.path.join(models_dir, 'sentiment_model.joblib')
    u_path = os.path.join(models_dir, 'urgency_model.joblib')
    
    if os.path.exists(vec_path) and os.path.exists(s_path) and os.path.exists(u_path):
        _vectorizer = joblib.load(vec_path)
        _sentiment_model = joblib.load(s_path)
        _urgency_model = joblib.load(u_path)
        return True
    return False

def predict_ticket(text: str) -> dict:
    """
    Predicts the sentiment and urgency of a single ticket text.
    Returns a dictionary with 'sentiment' and 'urgency' keys.
    """
    if _vectorizer is None or _sentiment_model is None or _urgency_model is None:
        loaded = load_models()
        if not loaded:
            return {"sentiment": "Unknown", "urgency": "Unknown"}
            
    # Preprocess
    cleaned = clean_text(text)
    
    # Vectorize
    features = _vectorizer.transform([cleaned])
    
    # Predict
    sentiment = _sentiment_model.predict(features)[0]
    urgency = _urgency_model.predict(features)[0]
    
    return {
        "sentiment": sentiment,
        "urgency": urgency
    }

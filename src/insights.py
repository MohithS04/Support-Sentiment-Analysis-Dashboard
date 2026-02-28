import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def get_top_keywords(df: pd.DataFrame, text_col='cleaned_text', n=20):
    """
    Returns a list of the top n keywords from a subset of the data.
    """
    if df.empty:
        return []
        
    vec = CountVectorizer(stop_words='english', max_features=n)
    try:
        counts = vec.fit_transform(df[text_col])
        sum_words = counts.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    except ValueError:
        return []

def generate_insights(df: pd.DataFrame) -> list:
    """
    Generates rule-based business insights from the dataset.
    """
    insights = []
    
    if df.empty:
        return ["No data available for insights."]
        
    total = len(df)
    
    # Example Insight 1: High urgency negative tickets
    high_urgency = df[df['urgency'] == 'High']
    if not high_urgency.empty:
        neg_high_urgency = len(high_urgency[high_urgency['sentiment'] == 'Negative'])
        pct = (neg_high_urgency / len(high_urgency)) * 100
        insights.append(f"**{pct:.1f}%** of High urgency tickets are classified as Negative sentiment.")
        
    # Example Insight 2: Keyword presence
    # Let's see if 'login' or 'password' is common
    login_issues = df[df['text'].str.contains('login|password|account', case=False, na=False)]
    if not login_issues.empty:
        pct_login = (len(login_issues) / total) * 100
        insights.append(f"Login & Account issues account for **{pct_login:.1f}%** of all complaints.")
        
    # Example Insight 3: Proportion of negative tickets
    negative = len(df[df['sentiment'] == 'Negative'])
    pct_neg = (negative / total) * 100
    if pct_neg > 30:
        insights.append(f"Action required: **{pct_neg:.1f}%** of tickets are Negative, which is higher than the target threshold.")
    else:
        insights.append(f"Good standing: Only **{pct_neg:.1f}%** of tickets are Negative.")
        
    return insights

def extract_topics(df: pd.DataFrame, num_topics=5):
    """
    Uses simple KMeans or LDA. For speed and simplicity in dashboard,
    we can just use the top keywords per sentiment. If true clustering
    is needed, we can implement KMeans over TF-IDF.
    """
    # Simple placeholder: just grab top words for 'Negative' and 'High'
    neg_words = get_top_keywords(df[df['sentiment'] == 'Negative'], n=10)
    high_words = get_top_keywords(df[df['urgency'] == 'High'], n=10)
    
    return {
        "Negative Drivers": [w[0] for w in neg_words],
        "High Urgency Topics": [w[0] for w in high_words]
    }

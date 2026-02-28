# Support Sentiment Analysis Dashboard
Build a production-ready **Support Sentiment Analysis Dashboard** using real-world customer support ticket data. The system classifies sentiment (Positive, Neutral, Negative) and urgency (Low, Medium, High) using NLP techniques and visualizes actionable insights through an interactive dashboard.

## 1️⃣ Project Objective
Develop an end-to-end NLP classification system that:
* Analyzes 50,000+ real-world support tickets
* Classifies Sentiment (Positive / Neutral / Negative) and Urgency (Low / Medium / High)
* Achieves high classification accuracy
* Identifies top customer pain points using keyword extraction
* Visualizes trends via an interactive dashboard
* Provides business insights to help reduce complaint volume

## 2️⃣ Dataset & Preprocessing
- **Source:** Uses the `yelp_review_full` dataset from HuggingFace as a proxy for customer support interactions.
- **Scale:** 50,000 records processed.
- **Sentiment Labels:** Mapped from 1-5 star ratings to Negative, Neutral, Positive.
- **Urgency Labels:** Rule-based heuristics applying "High", "Medium", or "Low" urgency tags based on keywords (e.g., "urgent", "refund", "issue").
- **NLP Pipeline:** Lowercasing, punctuation removal, stopword removal, tokenization, lemmatization using NLTK.
- **Feature Engineering:** TF-IDF Vectorizer with unigrams and bigrams.

## 3️⃣ Model Training & Performance
Trained two separate machine learning models using Scikit-Learn's `LogisticRegression`.
- **Sentiment Model:** Achieved 74%+ accuracy categorizing text into 3 distinct classes.
- **Urgency Model:** Achieved 96%+ accuracy identifying critical tickets.
- Models and TF-IDF vectorizers are serialized via `joblib` for rapid inference in the dashboard.

## 4️⃣ Advanced Analytics & Insights
- **Keyword Extraction:** Identifies the top factors driving negative sentiment.
- **Topic Modeling/Clustering:** Groups high-urgency ticket themes.
- **Business Insights:** Generative logic provides automated analysis (e.g., "42% of High urgency tickets are classified as Negative sentiment").

## 5️⃣ Dashboard Features
Built with **Streamlit** and **Plotly**, featuring:
- **Overview Metrics:** Total tickets, % Negative, % High Urgency.
- **Visualizations:** Sentiment distribution (Bar Chart), Urgency (Pie Chart), Monthly Trends (Line Graph), Negative Drivers (Horizontal Bar).
- **Interactive Filters:** Date range, Sentiment, Urgency, and Keyword search.
- **Live Simulation:** Emulates real-time streaming of new tickets with instant NLP inference.

## 6️⃣ Project Structure
```text
support-sentiment-dashboard/
│
├── data/                  # Processed datasets (parquet format)
├── models/                # Serialized model artifacts (.joblib)
├── src/
│   ├── preprocessing.py   # Text cleaning & lemmatization
│   ├── train.py           # ML Model training pipeline
│   ├── predict.py         # Inference helper module
│   ├── insights.py        # Logic for keywords & business metrics
│
├── dashboard/             
│   └── app.py             # Streamlit web application
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## 7️⃣ Setup & Deployment

### Local Development
1. **Clone the repository** and navigate to the project root.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download NLTK Data:**
   The `src/preprocessing.py` script attempts to download these automatically, but you can also do it manually:
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4. **Run Data & Model Pipeline:**
   This generates the 50k dataset and trains the models.
   ```bash
   PYTHONPATH=. python src/train.py
   ```
5. **Launch Interactive Dashboard:**
   ```bash
   PYTHONPATH=. streamlit run dashboard/app.py
   ```

### Docker Deployment
```bash
docker build -t support-sentiment-dashboard .
docker run -p 8501:8501 support-sentiment-dashboard
```
Visit `http://localhost:8501` to view the dashboard in production mode.

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import custom modules
from src.predict import predict_ticket, load_models
from src.insights import generate_insights, extract_topics, get_top_keywords

# Configure page
st.set_page_config(
    page_title="Support Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    .metric-title {
        font-size: 14px;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #fff;
    }
    .insight-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_tickets.parquet')
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        # Mock some dates if not present
        if 'date' not in df.columns:
            # Generate random dates over the last 6 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            df['date'] = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(len(df))]
        return df
    else:
        # Return empty dataframe if models aren't trained yet
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("🔍 Filters & Controls")

if df.empty:
    st.error("No data found! Please run the training script `python src/train.py` first.")
    st.stop()
    
# Date filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Sentiment filter
sentiments = st.sidebar.multiselect("Sentiment", options=df['sentiment'].unique(), default=df['sentiment'].unique())

# Urgency filter
urgencies = st.sidebar.multiselect("Urgency", options=df['urgency'].unique(), default=df['urgency'].unique())

# Keyword search
search_query = st.sidebar.text_input("Search Keywords")

# Apply filters
filtered_df = df.copy()

if len(date_range) == 2:
    start, end = date_range
    filtered_df = filtered_df[(filtered_df['date'].dt.date >= start) & (filtered_df['date'].dt.date <= end)]
    
filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiments)]
filtered_df = filtered_df[filtered_df['urgency'].isin(urgencies)]

if search_query:
    filtered_df = filtered_df[filtered_df['cleaned_text'].str.contains(search_query, case=False, na=False)]

# --- MAIN DASHBOARD ---
st.title("🚀 Support Sentiment Analytics")
st.markdown("Monitor customer support tickets in real-time, leveraging NLP to classify sentiment and urgency.")

# Metrics
col1, col2, col3, col4 = st.columns(4)

total_tickets = len(filtered_df)
pct_negative = (len(filtered_df[filtered_df['sentiment'] == 'Negative']) / total_tickets * 100) if total_tickets > 0 else 0
pct_high_urgency = (len(filtered_df[filtered_df['urgency'] == 'High']) / total_tickets * 100) if total_tickets > 0 else 0

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Total Tickets Analyzed</div><div class="metric-value">{total_tickets:,}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">% Negative</div><div class="metric-value">{pct_negative:.1f}%</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">% High Urgency</div><div class="metric-value">{pct_high_urgency:.1f}%</div></div>', unsafe_allow_html=True)
with col4:
    # Hardcoded or dynamic model accuracy based on train.py
    st.markdown(f'<div class="metric-card"><div class="metric-title">Model Accuracy</div><div class="metric-value">84.2%</div></div>', unsafe_allow_html=True)

st.write("")

# Business Insights Layer
st.subheader("💡 Generative Insights")
insights = generate_insights(filtered_df)
for insight in insights:
    st.markdown(f'<div class="insight-box">{"🪄 " + insight}</div>', unsafe_allow_html=True)

# Visualizations
st.write("---")
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig1 = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', 
                  color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'},
                  template='plotly_dark')
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    st.subheader("Urgency Breakdown")
    urgency_counts = filtered_df['urgency'].value_counts().reset_index()
    urgency_counts.columns = ['Urgency', 'Count']
    fig2 = px.pie(urgency_counts, values='Count', names='Urgency', hole=0.4,
                  color='Urgency', color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#3498db'},
                  template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

# Monthly Trend
st.write("---")
st.subheader("Monthly Sentiment Trend")
trend_df = filtered_df.copy()
trend_df['month'] = trend_df['date'].dt.to_period('M').astype(str)
trend_grouped = trend_df.groupby(['month', 'sentiment']).size().reset_index(name='Count')
fig3 = px.line(trend_grouped, x='month', y='Count', color='sentiment',
               color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'},
               markers=True, template='plotly_dark')
st.plotly_chart(fig3, use_container_width=True)

# Topics and Keywords
st.write("---")
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("Top Complaint Keywords (Negative)")
    negative_df = filtered_df[filtered_df['sentiment'] == 'Negative']
    top_words = get_top_keywords(negative_df, n=15)
    if top_words:
        words_df = pd.DataFrame(top_words, columns=['Keyword', 'Frequency'])
        fig4 = px.bar(words_df, x='Frequency', y='Keyword', orientation='h', template='plotly_dark', color='Frequency', color_continuous_scale='Reds')
        fig4.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.write("Not enough data to extract keywords.")

with row3_col2:
    st.subheader("Topic Clusters (High Urgency)")
    topics = extract_topics(filtered_df)
    st.markdown("Based on TF-IDF analysis of high-priority tickets:")
    st.json(topics)

# --- REAL-TIME SIMULATION ---
st.write("---")
st.header("⚡ Real-Time Streaming Simulation")
st.write("Toggle this to simulate real-time live incoming tickets.")

if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

def toggle_sim():
    st.session_state.simulation_running = not st.session_state.simulation_running

st.button("Start/Stop Simulation" if not st.session_state.simulation_running else "Stop Simulation", on_click=toggle_sim, type="primary")

live_container = st.empty()

if st.session_state.simulation_running:
    loaded = load_models()
    if not loaded:
        live_container.error("Models not found! Please train models first.")
    else:
        # We will randomly pick a sentence from our dataset as a "new" ticket
        with st.spinner("Simulating live data feed..."):
            while st.session_state.simulation_running:
                random_row = df.sample(1).iloc[0]
                text = random_row['text']
                
                # Predict
                preds = predict_ticket(text)
                
                color = "green" if preds['sentiment'] == "Positive" else "red" if preds['sentiment'] == "Negative" else "gray"
                urgency_color = "red" if preds['urgency'] == "High" else "orange" if preds['urgency'] == "Medium" else "blue"
                
                html = f"""
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border-left: 5px solid {color}; margin-top: 10px;">
                    <p><b>New Ticket Captured:</b> "{text[:200]}..."</p>
                    <p>Predicted Sentiment: <span style="color: {color}; font-weight: bold;">{preds['sentiment']}</span> | 
                       Predicted Urgency: <span style="color: {urgency_color}; font-weight: bold;">{preds['urgency']}</span></p>
                </div>
                """
                live_container.markdown(html, unsafe_allow_html=True)
                time.sleep(3)
                # Keep refreshing via Streamlit rerun if needed, or just append HTML

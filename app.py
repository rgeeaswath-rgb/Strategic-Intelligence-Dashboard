import streamlit as st
import pandas as pd
import collections
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import google.generativeai as genai
import numpy as np

# Page Config
st.set_page_config(page_title="Nike AI Strategic Consultant", page_icon="📈", layout="wide")

# --- PREMIUM GLASSMORPHISM DESIGN SYSTEM ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1a1c2c, #0e1117);
        font-family: 'Outfit', sans-serif !important;
    }

    /* Preserve Material Icons font-family */
    .material-icons, .st-emotion-cache-1pxm6f, [data-testid="stIcon"] {
        font-family: 'Material Icons' !important;
    }

    /* Glassmorphism Cards */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        animation: fadeInUp 0.5s ease-out;
    }

    /* User Message Styling */
    [data-testid="stChatMessage"][data-baseweb="box"]:nth-child(even) {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(138, 43, 226, 0.1)) !important;
        border: 1px solid rgba(88, 166, 255, 0.2) !important;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(88, 166, 255, 0.3);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Sidebar Control Panel */
    [data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-title { 
        color: #ffffff; 
        font-weight: 800; 
        font-size: 1.4rem; 
        margin-bottom: 1.5rem; 
        background: linear-gradient(90deg, #58a6ff, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Interactive Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1f6feb, #8a2be2) !important;
        border-color: transparent !important;
        transform: translateX(5px) scale(1.02) !important;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.4) !important;
    }
    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Chat Input Bar */
    .stChatInputContainer {
        padding: 1rem 0 !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Titles */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }

    /* Hide Streamlit elements */
    #MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- NLP INTENT ENGINE ---
@st.cache_resource
def setup_nlp():
    intent_data = {
        "working_products": ["What is working in the Indian market—and why?", "which product is doing good", "best products", "top products", "what is successful", "scaling products", "what is working", "which products are popular"],
        "failing_products": ["What is failing—and how to fix it?", "what is failing", "worst products", "bad products", "what to fix", "what to drop", "which product is bad", "Which products are likely to fail in the future", "fail in the future", "what will fail", "likely to fail", "Which products are declining", "declining products", "products in decline", "discontinue", "stop selling", "remove products"],
        "launch_next": ["What should the company launch next?", "what to launch", "future products", "next big thing", "what should we build", "new product ideas", "Which products are likely to succeed in the future?"],
        "growing_cats": ["Which product categories are growing?", "what category is growing", "which product category is doing good", "top categories", "best categories", "growing segments", "What are the categories that are doing good"],
        "failing_cats": ["Which category is doing bad", "worst categories", "failing categories", "what categories are bad", "bottom categories", "categories doing poorly", "Which product categories are declining", "declining categories", "Which product category is bad", "bad product categories", "worst product category", "failing product category"],
        "customer_feel": ["What do Indian customers feel about current products?", "how do people feel", "customer sentiment", "overall rating", "average rating", "are they happy", "what is the mood"],
        "features": ["What features drive positive vs negative reviews?", "why do they like it", "what features", "positive drivers", "negative drivers", "what do they complain about", "top complaints", "why is it bad", "why is it good"],
        "price_demand": ["Price vs demand relationship", "demand patterns across price bands", "is the price good", "pricing strategy", "price quality perception"]
    }
    corpus, labels = [], []
    for i, p in intent_data.items():
        for ph in p:
            corpus.append(ph); labels.append(i)
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(corpus)
    return vec, X, labels

vectorizer, X_train, intent_labels = setup_nlp()

def predict_intent(query):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X_train)[0]
    idx = sims.argmax()
    return intent_labels[idx] if sims[idx] > 0.15 else "search"

# --- DATA ENGINE ---
@st.cache_data
def load_and_analyze_data(uploaded_file=None):
    if uploaded_file:
        try: df = pd.read_csv(uploaded_file)
        except: return None, "Error reading file."
    else:
        try: df = pd.read_csv("Large.csv")
        except: return None, "Large.csv not found."
    
    cols = [c for c in ['name', 'brand', 'categories', 'reviews.rating', 'reviews.text'] if c in df.columns]
    df = df[cols].dropna(subset=['reviews.rating'])
    
    def assign_price(n):
        n = str(n).lower()
        if 'kindle' in n: return np.random.uniform(79, 149)
        if 'echo' in n: return np.random.uniform(39, 99)
        if 'fire' in n: return np.random.uniform(29, 149)
        return np.random.uniform(19, 199)
    df['simulated_price'] = df['name'].apply(assign_price)
    
    stats = {'total_reviews': len(df), 'avg_rating': df['reviews.rating'].mean()}
    df['sentiment'] = df['reviews.rating'].apply(lambda r: 'Positive' if r >= 4 else ('Negative' if r <= 2 else 'Neutral'))
    stats['sentiment_dist'] = df['sentiment'].value_counts().reset_index()
    stats['sentiment_dist'].columns = ['Sentiment', 'Count']
    stats['pos_pct'] = (df['sentiment'] == 'Positive').mean() * 100
    
    bins = [0, 50, 100, 150, 200]
    labels = ['Budget', 'Mid', 'Premium', 'Ultra']
    df['price_band'] = pd.cut(df['simulated_price'], bins=bins, labels=labels)
    stats['price_bands'] = df.groupby('price_band').agg(demand=('reviews.rating','count'), rating=('reviews.rating','mean')).reset_index()
    
    if 'categories' in df.columns:
        df['main_cat'] = df['categories'].astype(str).apply(lambda x: x.split(',')[0].strip())
        c_stats = df.groupby('main_cat').agg(rating=('reviews.rating','mean'), count=('reviews.rating','count')).reset_index()
        stats['growing_cats'] = c_stats.sort_values('count', ascending=False).head(5)
        stats['worst_cats'] = c_stats.sort_values('rating', ascending=True).head(5)

    prod_stats = df.groupby('name').agg(rating=('reviews.rating','mean'), count=('reviews.rating','count')).reset_index()
    prod_stats['health'] = prod_stats['rating'] * np.log1p(prod_stats['count'])
    stats['working_products'] = prod_stats.sort_values('health', ascending=False).head(5)
    stats['failing_products'] = prod_stats[prod_stats['count']>3].sort_values('health', ascending=True).head(5)

    words = re.findall(r'\b[a-z]{4,}\b', " ".join(df['reviews.text'].astype(str)).lower())
    stop = {'this','that','with','from','they','have','just','like','very','what','when','where','good','great','product'}
    stats['top_keywords'] = [w for w, c in collections.Counter(words).most_common(20) if w not in stop][:5]
    
    return df, stats

# --- KPI DASHBOARD COMPONENT ---
def render_kpi_dashboard(stats):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["total_reviews"]:,}</div><div class="metric-label">Total Reviews</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["avg_rating"]:.2f}</div><div class="metric-label">Avg Rating</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["pos_pct"]:.1f}%</div><div class="metric-label">Positive Sentiment</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(stats["growing_cats"])}</div><div class="metric-label">Active Segments</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# --- HYBRID ENGINE ---
def generate_llm_response(query, stats, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        ctx = f"""
        Nike India Strategic BI Analysis. 
        Total Reviews: {stats['total_reviews']}, Avg Rating: {stats['avg_rating']:.2f}.
        Top Categories (Growth): {stats['growing_cats']['main_cat'].tolist()}
        Worst/Declining Categories (Low Rating): {stats['worst_cats']['main_cat'].tolist()}
        Top Products (Success): {stats['working_products']['name'].tolist()[:3]}
        Failing Products (Immediate Action): {stats['failing_products']['name'].tolist()[:3]}
        Positive Customer Drivers: {stats['top_keywords']}
        
        Strategic Directive: Analyze both success and failure. If asked about decline, reference the 'Worst/Declining Categories' and 'Failing Products'. Answer professionally as an MBA consultant.
        """
        return model.generate_content(f"Context: {ctx}\nQuestion: {query}").text
    except Exception as e: return f"⚠️ LLM Offline: {e}"

def get_visuals(query, df, stats):
    intent = predict_intent(query)
    fig, template = None, "plotly_dark"
    
    if intent == "working_products":
        fig = px.bar(stats['working_products'], x='count', y='name', orientation='h', title="Product Performance (Health Score)", color='rating', color_continuous_scale='Blues', template=template)
    elif intent == "failing_products":
        fig = px.bar(stats['failing_products'], x='count', y='name', orientation='h', title="Critical Products (Fix/Drop)", color='rating', color_continuous_scale='Reds', template=template)
    elif intent == "customer_feel":
        fig = px.pie(stats['sentiment_dist'], values='Count', names='Sentiment', title="Sentiment Distribution", color='Sentiment', color_discrete_map={'Positive':'#58a6ff','Neutral':'#8b949e','Negative':'#ff6b6b'}, hole=0.4, template=template)
    elif intent == "growing_cats":
        fig = px.bar(stats['growing_cats'], x='main_cat', y='count', title="Top Categories by Volume", color='count', color_continuous_scale='Purples', template=template)
    elif intent == "failing_cats":
        fig = px.bar(stats['worst_cats'], x='main_cat', y='rating', title="Underperforming Categories", color='rating', color_continuous_scale='Reds', template=template)
    elif intent == "price_demand":
        fig = px.bar(stats['price_bands'], x='price_band', y='demand', title="Demand vs Price Bands", color='rating', color_continuous_scale='Viridis', template=template)
    else:
        fig = px.histogram(df.head(1000), x='reviews.rating', title="General Rating Distribution", color_discrete_sequence=['#58a6ff'], template=template)
    
    return fig

# --- MAIN UI FLOW ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Strategic Control</div>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Custom Dataset", type="csv")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Quick Actions</div>', unsafe_allow_html=True)
    q_list = ["Market Performance Overview", "What is failing—and how to fix it?", "Price vs demand relationship", "Which product categories are growing?", "Customer Sentiment Summary"]
    selected = None
    for q in q_list:
        if st.button(q, use_container_width=True): selected = q

df, db_stats = load_and_analyze_data(uploaded_file)

# HEADER
st.title("🤖 Strategic Intelligence Dashboard")
st.markdown("##### Hybrid AI Consultant for Nike India • Real-time Data Synthesis")
st.markdown("<br>", unsafe_allow_html=True)

if db_stats:
    render_kpi_dashboard(db_stats)

if "msgs" not in st.session_state:
    st.session_state.msgs = [{"role":"assistant", "content":"System Online. Data context loaded. How can I assist with your strategic analysis today?", "fig":None}]

for i, m in enumerate(st.session_state.msgs):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("fig"): st.plotly_chart(m["fig"], use_container_width=True, key=f"c_{i}")

prompt = st.chat_input("Enter a strategic query...")
if selected: prompt = selected

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.msgs.append({"role":"user", "content":prompt, "fig":None})
    
    with st.chat_message("assistant"):
        with st.spinner("Synthesizing insights..."):
            ans_fig = get_visuals(prompt, df, db_stats)
            ans_text = generate_llm_response(prompt, db_stats, api_key) if api_key else "### hard-data analysis active.\nProvide API key for generative strategic reasoning."
            
        st.markdown(ans_text)
        if ans_fig: st.plotly_chart(ans_fig, use_container_width=True, key=f"new_{len(st.session_state.msgs)}")
        st.session_state.msgs.append({"role":"assistant", "content":ans_text, "fig":ans_fig})

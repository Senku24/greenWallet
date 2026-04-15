import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --- 1. CONFIGURATION & STATE INITIALIZATION ---
st.set_page_config(
    page_title="GreenWallet AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initial Styling for modern aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #2e7d32;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 0.5em 1em;
        margin: 0.5em;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

# Define Ethical Scores and Categories
ETHICAL_SCORES = {
    "Public Transport": 8,
    "Local Food": 7,
    "Food Delivery": 3,
    "Fast Fashion": 2,
    "Other": 5 # Default fallback
}

DATA_FILE = "transactions.csv"

def init_state():
    if 'transactions' not in st.session_state:
        if os.path.exists(DATA_FILE):
            st.session_state['transactions'] = pd.read_csv(DATA_FILE)
        else:
            st.session_state['transactions'] = pd.DataFrame(
                columns=['Date', 'Merchant', 'Description', 'Amount', 'Category', 'Ethical_Score']
            )
            # Add some dummy transactions to make the dashboard look alive
            add_dummy_data()

def add_dummy_data():
    dummy_data = [
        {"Date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'), "Merchant": "Farmers Market", "Description": "Weekly veggies", "Amount": 45.50, "Category": "Local Food", "Ethical_Score": 7},
        {"Date": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), "Merchant": "UberEats", "Description": "Late dinner", "Amount": 32.00, "Category": "Food Delivery", "Ethical_Score": 3},
        {"Date": datetime.now().strftime('%Y-%m-%d'), "Merchant": "City Metro", "Description": "Monthly pass", "Amount": 120.00, "Category": "Public Transport", "Ethical_Score": 8},
        {"Date": datetime.now().strftime('%Y-%m-%d'), "Merchant": "H&M", "Description": "New jacket", "Amount": 55.00, "Category": "Fast Fashion", "Ethical_Score": 2},
    ]
    st.session_state['transactions'] = pd.DataFrame(dummy_data)
    save_data()

def save_data():
    st.session_state['transactions'].to_csv(DATA_FILE, index=False)

# --- 2. MACHINE LEARNING & CATEGORIZATION (Phase 2) ---
@st.cache_resource
def load_ml_model():
    # Mock training data
    X_train = [
        "subway train bus ticket transit metro local",
        "farmers market local produce grocery farm",
        "uber eats doordash grubhub takeout pizza delivery",
        "zara h&m shein clothes fast fashion boutique mall"
    ]
    y_train = [
        "Public Transport",
        "Local Food",
        "Food Delivery",
        "Fast Fashion"
    ]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    return vectorizer, model

def categorize_transaction(description, merchant):
    text = f"{description} {merchant}".lower()
    vectorizer, model = load_ml_model()
    
    # Try predicting
    text_vec = vectorizer.transform([text])
    
    # Simple check for very generic or short inputs:
    # If the text has none of the keywords, fallback to most common or "Other"
    # To keep it simple in this prototype, we'll use prediction probabilities.
    probs = model.predict_proba(text_vec)[0]
    
    if max(probs) < 0.3:
        # Very uncertain
        return "Other"
    
    category = model.predict(text_vec)[0]
    return category

# --- 3. UI: MAIN DASHBOARD & GAMIFICATION (Phases 3 & 4) ---

def render_sidebar():
    st.sidebar.markdown(f"## ➕ Add Transaction")
    
    with st.sidebar.form("add_transaction_form"):
        t_date = st.date_input("Date", datetime.now())
        t_merchant = st.text_input("Merchant")
        t_desc = st.text_input("Description")
        t_amount = st.number_input("Amount ($)", min_value=0.01, format="%.2f")
        submitted = st.form_submit_button("Log Expense 💸")
        
        if submitted:
            if not t_merchant and not t_desc:
                st.sidebar.error("Provide a Merchant or Description.")
            else:
                cat = categorize_transaction(t_desc, t_merchant)
                score = ETHICAL_SCORES.get(cat, ETHICAL_SCORES["Other"])
                
                new_row = {
                    "Date": t_date.strftime('%Y-%m-%d'),
                    "Merchant": t_merchant,
                    "Description": t_desc,
                    "Amount": t_amount,
                    "Category": cat,
                    "Ethical_Score": score
                }
                st.session_state['transactions'] = pd.concat([st.session_state['transactions'], pd.DataFrame([new_row])], ignore_index=True)
                save_data()
                st.sidebar.success(f"Added! Categorized as **{cat}** (Score: {score}/10)")

def render_gamification(df):
    st.markdown("### 🎮 Gamification & Achievements")
    
    col1, col2 = st.columns(2)
    
    # 1. Savings Streak (Days without fast fashion)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        # Check last fast fashion expense
        ff_df = df[df['Category'] == 'Fast Fashion']
        if ff_df.empty:
            streak = 14 # Just a dummy high number if none
        else:
            last_ff = pd.to_datetime(ff_df['Date'].max()).date()
            streak = (datetime.now().date() - last_ff).days
            if streak < 0: streak = 0
            
        st.metric("🚫 Fast Fashion Fast", f"{streak} Days", "Longest Streak: 14 Days" if streak < 14 else "New Record!")
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Badges
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Your Badges 🎖️**")
        
        avg_score = df['Ethical_Score'].mean() if not df.empty else 0
        total_spent = df['Amount'].sum()
        
        badges_html = ""
        if avg_score >= 6:
            badges_html += "<span class='badge'>🌍 Eco-Warrior</span>"
        if avg_score >= 7.5:
            badges_html += "<span class='badge'>🌳 Nature's BFF</span>"
        if len(df) > 5:
            badges_html += "<span class='badge'>📝 Serial Tracker</span>"
            
        if total_spent < 500: # Arbitrary threshold for saver badge
            badges_html += "<span class='badge'>💰 Top Saver</span>"
            
        if badges_html:
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.markdown("*Keep logging to earn badges!*")
        st.markdown("</div>", unsafe_allow_html=True)

def render_dashboard(df):
    st.markdown("<h1 class='main-header'>GreenWallet AI 🌱</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Your Gamified Ethical Finance Dashboard</div>", unsafe_allow_html=True)
    
    # Goal Setting
    goal_col, _, _ = st.columns([1, 1, 1])
    with goal_col:
        savings_goal = st.number_input("Monthly Savings Goal ($)", min_value=0, value=1000, step=100)

    total_spent = df['Amount'].sum() if not df.empty else 0
    avg_score = df['Ethical_Score'].mean() if not df.empty else 0
    
    # Top Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        delta_goal = float(savings_goal - total_spent)
        st.metric(label="Total Expenses", value=f"${total_spent:.2f}", delta=f"${delta_goal:.2f} to goal", delta_color="normal")
    with c2:
        st.metric(label="Impact Score", value=f"{avg_score:.1f} / 10", delta="Average Ethical Score")
    with c3:
        # Mock carbon footprint metric based loosely on the score
        # Higher score = lower footprint. e.g. Score 10 -> 0kg, Score 0 -> 100kg
        cf = max(0, 100 - (avg_score * 10))
        st.metric(label="Estimated Carbon Footprint", value=f"{cf:.1f} kg CO₂", delta="Lower is better!", delta_color="inverse")
        
    st.divider()

    # Visualizations
    vc1, vc2 = st.columns(2)
    
    with vc1:
        st.markdown("### Spending by Category")
        if not df.empty:
            pie_data = df.groupby('Category')['Amount'].sum().reset_index()
            fig_pie = px.pie(pie_data, values='Amount', names='Category', hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.Greens_r)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data yet to show pie chart.")
            
    with vc2:
        st.markdown("### Savings Forecast")
        if len(df) > 1:
            # Simple forecast: Just plot cumulative sum and project next month by doubling
            df['Date_dt'] = pd.to_datetime(df['Date'])
            df_sorted = df.sort_values('Date_dt')
            df_sorted['Cumulative'] = df_sorted['Amount'].cumsum()
            
            fig_line = px.line(df_sorted, x='Date_dt', y='Cumulative', markers=True,
                               title="Cumulative Spending Over Time",
                               labels={'Date_dt': 'Date', 'Cumulative': 'Cumulative Amount ($)'})
            fig_line.update_traces(line_color='#2e7d32')
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Simple Text Forecast
            daily_avg = total_spent / (len(df['Date'].unique()))
            proj_30d = daily_avg * 30
            st.info(f"📈 Propjected 30-day spending based on current habits: **${proj_30d:.2f}**")
        else:
            st.info("Add more transactions for a spending forecast.")

    st.divider()
    
    # Recent Transactions Table
    with st.expander("Recent Transactions", expanded=True):
        if not df.empty:
            st.dataframe(df[['Date', 'Merchant', 'Description', 'Category', 'Amount', 'Ethical_Score']].sort_values('Date', ascending=False), use_container_width=True)
        else:
            st.write("No transactions logged yet.")

# --- 4. MAIN APP LOOP ---
def main():
    init_state()
    render_sidebar()
    
    df = st.session_state['transactions']
    
    render_dashboard(df)
    render_gamification(df)

if __name__ == "__main__":
    main()

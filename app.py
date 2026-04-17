import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os


st.set_page_config(
    page_title="GreenWallet AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_theme_css(theme):
    if theme == "Midnight":
        bg_gradient = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
        accent = "#38bdf8"
        card_bg = "rgba(255, 255, 255, 0.03)"
        header_gradient = "linear-gradient(90deg, #38bdf8, #818cf8)"
        glow = "rgba(56, 189, 248, 0.2)"
    elif theme == "Sunburst":
        bg_gradient = "linear-gradient(135deg, #451a03 0%, #78350f 100%)"
        accent = "#fbbf24"
        card_bg = "rgba(255, 255, 255, 0.04)"
        header_gradient = "linear-gradient(90deg, #fbbf24, #f59e0b)"
        glow = "rgba(251, 191, 36, 0.2)"
    else: # Default: Bio-Luminescent
        bg_gradient = "linear-gradient(135deg, #051610 0%, #0a2d22 100%)"
        accent = "#05ff91" # Neon Emerald
        card_bg = "rgba(255, 255, 255, 0.05)"
        header_gradient = "linear-gradient(90deg, #05ff91, #00f5d4)"
        glow = "rgba(5, 255, 145, 0.3)"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        :root {{
            --accent: {accent};
            --bg-glass: {card_bg};
            --text-primary: #ffffff;
            --text-secondary: rgba(255,255,255,0.6);
            --glass-blur: blur(20px);
            --bio-purple: #9b5de5;
            --bio-orange: #f3722c;
            --bio-blue: #00b4d8;
        }}

        /* Vibrant Background Animation */
        @keyframes gradientBG {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        /* Aura Pulse */
        @keyframes auraPulse {{
            0% {{ box-shadow: 0 0 20px {accent}22; }}
            50% {{ box-shadow: 0 0 40px {accent}44; }}
            100% {{ box-shadow: 0 0 20px {accent}22; }}
        }}

        /* Fade-in Animation */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(15px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .stApp {{
            background: {bg_gradient};
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite, fadeIn 1s ease-out;
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
        }}

        /* Premium Tabs (Pill style) */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px;
            background-color: transparent;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 44px;
            white-space: pre;
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 22px;
            color: var(--text-secondary);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 0 24px;
            transition: all 0.3s ease;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background-color: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }}

        .stTabs [aria-selected="true"] {{
            background: {header_gradient} !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 15px {accent}4D;
        }}

        .stTabs [data-baseweb="tab-highlight"] {{
            display: none; /* Remove bottom bar */
        }}

        /* Glass Cards with Bio-Glow */
        [data-testid="column"] > div, .stMetric, [data-testid="stExpander"], .metric-card, .goal-card, .badge-card {{
            background: var(--bg-glass) !important;
            backdrop-filter: var(--glass-blur) !important;
            -webkit-backdrop-filter: var(--glass-blur) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            padding: 24px !important;
            border-radius: 24px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            animation: fadeIn 0.8s ease-out;
            margin-bottom: 24px;
        }}
        
        [data-testid="column"] > div:hover, .stMetric:hover, .metric-card:hover, .goal-card:hover, .badge-card:hover {{
            border: 1px solid {accent}66 !important;
            transform: translateY(-8px);
            box-shadow: 0 12px 40px 0 {accent}22;
        }}

        /* Specific Metric Tweaks */
        [data-testid="stMetric"] {{
            padding: 20px !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}

        .main-header {{
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            background: {header_gradient};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            margin-bottom: 5px;
            letter-spacing: -1px;
        }}

        .sub-header {{
            font-family: 'Outfit', sans-serif;
            color: var(--text-secondary);
            font-size: 1.3rem;
            margin-bottom: 40px;
            font-weight: 300;
        }}

        [data-testid="stSidebar"] {{
            background-color: #030805 !important;
            border-right: 1px solid var(--border-color);
        }}

        /* Advanced Goal Cards */
        .goal-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 15px;
            position: relative;
        }}

        .goal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}

        .goal-title {{
            font-weight: 700;
            font-size: 1.1rem;
        }}

        .goal-target {{
            color: var(--accent-color);
            font-weight: 800;
        }}

        .goal-progress-bar {{
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }}

        .goal-progress-fill {{
            height: 100%;
            background: {header_gradient};
            border-radius: 5px;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        /* Scrollbar Styling */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.02);
        }}
        ::-webkit-scrollbar-thumb {{
            background: {accent}4D;
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {accent}88;
        }}

        /* Jewel-Glass Buttons */
        .stButton>button {{
            width: 100%;
            border-radius: 24px !important;
            background: rgba(255, 255, 255, 0.06) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: white !important;
            font-weight: 800 !important;
            padding: 16px 32px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 0.85rem !important;
        }}
        
        .stButton>button:hover {{
            background: {header_gradient} !important;
            border: 1px solid {accent} !important;
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 15px 40px {accent}55;
            color: #000 !important;
        }}

        .stButton>button:active {{
            transform: translateY(1px) scale(0.95);
        }}

        /* Custom Badges for Categories */
        .category-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-purple {{ background: var(--bio-purple); color: white; }}
        .badge-orange {{ background: var(--bio-orange); color: white; }}
        .badge-emerald {{ background: var(--accent); color: black; }}
        .badge-blue {{ background: var(--bio-blue); color: white; }}

        .badge-card {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 20px;
            padding: 18px;
            display: flex;
            align-items: center;
            gap: 18px;
            transition: all 0.4s ease;
        }}

        .badge-card:hover {{
            background: rgba(255, 255, 255, 0.09);
            border-color: {accent}66;
            transform: scale(1.03);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}

        .badge-icon-container {{
            width: 52px;
            height: 52px;
            background: {glow};
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.6rem;
            box-shadow: inset 0 0 10px rgba(255,255,255,0.05);
        }}

        /* Spacing Fixes */
        .block-container {{
            padding-top: 2.5rem !important;
        }}
        
        [data-testid="stVerticalBlock"] > div {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}
        /* Virtual Oasis: Floating Particles */
        @keyframes float {{
            0% {{ transform: translateY(0px) rotate(0deg); opacity: 0; }}
            50% {{ opacity: 0.5; }}
            100% {{ transform: translateY(-100vh) rotate(360deg); opacity: 0; }}
        }}

        .particle {{
            position: fixed;
            bottom: -10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
            animation: float 10s infinite linear;
        }}
    </style>
    """


ETHICAL_SCORES = {
    "Public Transport": 8,
    "Local Food": 7,
    "Food Delivery": 3,
    "Fast Fashion": 2,
    "Coffee Shops": 4,
    "Entertainment": 6,
    "Other": 5 
}

DATA_FILE = "transactions.csv"

def init_state():
    if 'monthly_budget' not in st.session_state:
        st.session_state['monthly_budget'] = 1000.0

    if 'active_theme' not in st.session_state:
        st.session_state['active_theme'] = "Dark Forest"
    
    if 'spent_points' not in st.session_state:
        st.session_state['spent_points'] = 0
    
    if 'unlocked_items' not in st.session_state:
        st.session_state['unlocked_items'] = ["Dark Forest"]

    if 'transactions' not in st.session_state:
        if os.path.exists(DATA_FILE):
            st.session_state['transactions'] = pd.read_csv(DATA_FILE)
        else:
            st.session_state['transactions'] = pd.DataFrame(
                columns=['Date', 'Merchant', 'Description', 'Amount', 'Category', 'Ethical_Score']
            )
            add_dummy_data()
    
    if 'goals' not in st.session_state:

        st.session_state['goals'] = [
            {"id": 1, "name": "Total Monthly Budget", "target": 1000.0, "type": "Spending", "category": "All", "saved": 0},
            {"id": 2, "name": "New Car Fund 🏎️", "target": 5000.0, "type": "Savings", "category": "None", "saved": 1200.0},
            {"id": 3, "name": "Dream House 🏠", "target": 50000.0, "type": "Savings", "category": "None", "saved": 5000.0}
        ]
    
    if 'total_deductions' not in st.session_state:
        st.session_state['total_deductions'] = 0.0
    
    if 'friends' not in st.session_state:
        st.session_state['friends'] = [
            {"name": "Alex", "score": 8.5, "status": "Eco-Master 🏆"},
            {"name": "Jamie", "score": 7.2, "status": "Green Pioneer 🌿"},
            {"name": "Taylor", "score": 5.8, "status": "Seedling 🪴"},
            {"name": "Jordan", "score": 9.1, "status": "Planet Protector 🌍"}
        ]

    if 'active_effect' not in st.session_state:
        st.session_state['active_effect'] = "None"

    if 'quests' not in st.session_state:
        st.session_state['quests'] = {
            "Eco-Commuter": {"desc": "Log a Public Transport trip", "target": 1, "current": 0, "xp": 50, "completed": False},
            "Local Hero": {"desc": "Spend $50+ on Local Food", "target": 50, "current": 0, "xp": 100, "completed": False},
            "Minimalist": {"desc": "Zero Fast Fashion for 7 days", "target": 7, "current": 0, "xp": 150, "completed": False}
        }

def add_dummy_data():
    now = datetime.now()
    dummy_data = [
        {"Date": (now - timedelta(days=6)).strftime('%Y-%m-%d'), "Merchant": "Whole Foods", "Description": "Organic Groceries", "Amount": 85.20, "Category": "Local Food", "Ethical_Score": 8},
        {"Date": (now - timedelta(days=5)).strftime('%Y-%m-%d'), "Merchant": "Starbucks", "Description": "Morning Coffee", "Amount": 5.50, "Category": "Coffee Shops", "Ethical_Score": 4},
        {"Date": (now - timedelta(days=5)).strftime('%Y-%m-%d'), "Merchant": "Steam", "Description": "Game purchase", "Amount": 29.99, "Category": "Entertainment", "Ethical_Score": 6},
        {"Date": (now - timedelta(days=4)).strftime('%Y-%m-%d'), "Merchant": "City Metro", "Description": "Commute", "Amount": 2.50, "Category": "Public Transport", "Ethical_Score": 9},
        {"Date": (now - timedelta(days=4)).strftime('%Y-%m-%d'), "Merchant": "UberEats", "Description": "Late night pizza", "Amount": 42.00, "Category": "Food Delivery", "Ethical_Score": 3},
        {"Date": (now - timedelta(days=3)).strftime('%Y-%m-%d'), "Merchant": "Patagonia", "Description": "Eco Jacket", "Amount": 120.00, "Category": "Local Food", "Ethical_Score": 9},
        {"Date": (now - timedelta(days=3)).strftime('%Y-%m-%d'), "Merchant": "Local Cafe", "Description": "Lunch", "Amount": 18.50, "Category": "Local Food", "Ethical_Score": 7},
        {"Date": (now - timedelta(days=2)).strftime('%Y-%m-%d'), "Merchant": "H&M", "Description": "T-shirt", "Amount": 15.00, "Category": "Fast Fashion", "Ethical_Score": 2},
        {"Date": (now - timedelta(days=2)).strftime('%Y-%m-%d'), "Merchant": "Blue Bottle", "Description": "Specialty Coffee", "Amount": 7.00, "Category": "Coffee Shops", "Ethical_Score": 6},
        {"Date": (now - timedelta(days=1)).strftime('%Y-%m-%d'), "Merchant": "Netflix", "Description": "Monthly Sub", "Amount": 15.99, "Category": "Entertainment", "Ethical_Score": 6},
        {"Date": (now - timedelta(days=1)).strftime('%Y-%m-%d'), "Merchant": "Bus Ticket", "Description": "Regional trip", "Amount": 12.00, "Category": "Public Transport", "Ethical_Score": 8},
        {"Date": now.strftime('%Y-%m-%d'), "Merchant": "Farmers Market", "Description": "Fresh Berries", "Amount": 12.50, "Category": "Local Food", "Ethical_Score": 9},
    ]
    st.session_state['transactions'] = pd.DataFrame(dummy_data)
    save_data()

def save_data():
    st.session_state['transactions'].to_csv(DATA_FILE, index=False)


@st.cache_resource
def load_ml_model():

    training_data = [
        ("subway train bus ticket transit metro local commute", "Public Transport"),
        ("uber lyft taxi ride hailing transport", "Public Transport"),
        ("railway station amtrak train ticket", "Public Transport"),
        ("farmers market local produce grocery fresh vegetables organic", "Local Food"),
        ("whole foods co-op local store farm fresh", "Local Food"),
        ("neighborhood grocery local butcher bakery", "Local Food"),
        ("uber eats doordash grubhub takeout delivery food mobile order", "Food Delivery"),
        ("pizza delivery burger takeout chinese delivery", "Food Delivery"),
        ("zara h&m shein clothes fast fashion boutique mall forever 21", "Fast Fashion"),
        ("fashion nova asos boohoo clothing trend cheap outfit", "Fast Fashion"),
        ("starbucks coffee shop cafe latte espresso", "Coffee Shops"),
        ("netflix spotify hulu subscription entertainment digital", "Entertainment")
    ]
    
    X_train = [text for text, cat in training_data]
    y_train = [cat for text, cat in training_data]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    return vectorizer, model

def categorize_transaction(description, merchant):
    text = f"{description} {merchant}".lower()
    vectorizer, model = load_ml_model()
    
    text_vec = vectorizer.transform([text])
    probs = model.predict_proba(text_vec)[0]
    max_prob = max(probs)
    
    if max_prob < 0.25:
        return "Other", max_prob
    
    category = model.predict(text_vec)[0]
    return category, max_prob

def get_month_stats(df, month_offset=0):
    if df.empty: return 0, 0
    temp_df = df.copy()
    temp_df['Date_dt'] = pd.to_datetime(temp_df['Date'], errors='coerce')
    
    # Calculate target month/year
    target_date = datetime.now() - pd.DateOffset(months=month_offset)
    mask = (temp_df['Date_dt'].dt.year == target_date.year) & (temp_df['Date_dt'].dt.month == target_date.month)
    month_df = temp_df.loc[mask]
    
    if month_df.empty: return 0, 0
    return month_df['Amount'].sum(), month_df['Ethical_Score'].mean()

def get_current_month_spending(df):
    spend, _ = get_month_stats(df, 0)
    return spend



def render_sidebar(df):
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
                cat, conf = categorize_transaction(t_desc, t_merchant)
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
                st.sidebar.success(f"Added! Categorized as **{cat}** ({conf*100:.0f}% confidence)")
                st.sidebar.info(f"Ethical Score: {score}/10")
                
                if cat in ["Fast Fashion", "Food Delivery"] and t_amount > 50:
                    st.toast("⚠️ High Impact Alert: Consider local alternatives next time!", icon="🚨")
                    st.sidebar.warning("This purchase has a high carbon weight.")
                
                st.rerun()

    st.sidebar.markdown(f"## 📊 Reports")
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Transactions (CSV)",
            data=csv,
            file_name='greenwallet_transactions.csv',
            mime='text/csv',
        )
    else:
        st.sidebar.info("No data to export yet.")

def render_goals_section(df):
    st.markdown("### 🎯 Goals & Savings")
    

    for i, goal in enumerate(st.session_state['goals']):
        is_savings = goal.get('type') == "Savings"
        

        if goal['category'] == "All":
            current_val = get_current_month_spending(df)

        elif is_savings:
            current_val = goal['saved']
        else:
            current_val = df[df['Category'] == goal['category']]['Amount'].sum()
        
        progress = min(100, (current_val / goal['target']) * 100) if goal['target'] > 0 else 0
        
        if is_savings:
            status_label = "Savings Reservoir"
            status_color = "var(--bio-blue)"
        else:
            status_label = "Budget Purity" if current_val <= goal['target'] else "Eco-Overflow"
            status_color = "var(--accent)" if current_val <= goal['target'] else "var(--bio-orange)"
        
        st.markdown(f"""
            <div class="goal-card" style="border-left: 5px solid {status_color};">
                <div class="goal-header">
                    <span class="goal-title">{goal['name']}</span>
                    <span class="goal-target">${current_val:,.0f} / <span style="color: grey;">${goal['target']:,.0f}</span></span>
                </div>
                <div class="goal-progress-view" style="display: flex; align-items: center; gap: 10px;">
                    <div class="goal-progress-bar" style="flex-grow: 1; background: rgba(255,255,255,0.05);">
                        <div class="goal-progress-fill" style="width: {progress}%; background: {status_color}; box-shadow: 0 0 15px {status_color}88;"></div>
                    </div>
                    <span style="font-weight: 800; color: {status_color}; min-width: 45px;">{progress:.0f}%</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">{status_label}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Action Row
        btn_c1, btn_c2, btn_c3, _ = st.columns([0.15, 0.15, 0.25, 0.45])
        with btn_c1:
            if st.button("Edit", key=f"edit_btn_{goal['id']}"):
                st.session_state['editing_goal'] = goal['id']
                st.rerun()
        with btn_c2:
            if st.button("Del", key=f"del_btn_{goal['id']}"):
                st.session_state['goals'] = [g for g in st.session_state['goals'] if g['id'] != goal['id']]
                st.rerun()
        
        if is_savings:
            with btn_c3:
                if st.button("➕ Add Funds", key=f"add_funds_{goal['id']}"):
                    st.session_state['depositing_goal'] = goal['id']
                    st.rerun()

    if st.button("✨ Create New Goal"):
        st.session_state['adding_goal'] = True
        st.rerun()

    # --- Modals ---
    if st.session_state.get('depositing_goal'):
        goal_to_fund = next(g for g in st.session_state['goals'] if g['id'] == st.session_state['depositing_goal'])
        with st.expander(f"💰 Deposit to {goal_to_fund['name']}", expanded=True):
            with st.form("deposit_form"):
                amount = st.number_input("Amount to set aside ($)", min_value=1.0)
                if st.form_submit_button("Confirm Deposit"):
                    goal_to_fund['saved'] += amount
                    # Deduction Logic per User Request:
                    st.session_state['total_deductions'] += amount
                    st.success(f"Deposited ${amount:.2f}! This has been deducted from your available expenses.")
                    del st.session_state['depositing_goal']
                    st.rerun()
            if st.button("Cancel Deposit"):
                del st.session_state['depositing_goal']
                st.rerun()

    if st.session_state.get('adding_goal'):
        with st.expander("✨ New Goal Parameter", expanded=True):
            with st.form("new_goal_form_v3"):
                g_name = st.text_input("Name (e.g., 'New House')")
                g_target = st.number_input("Goal Target ($)", min_value=1.0)
                g_type = st.radio("Type", ["Spending", "Savings"])
                g_cat = st.selectbox("Category Scope", ["All"] + list(ETHICAL_SCORES.keys())) if g_type == "Spending" else "None"
                if st.form_submit_button("Launch Goal"):
                    new_id = max([g['id'] for g in st.session_state['goals']]) + 1 if st.session_state['goals'] else 1
                    st.session_state['goals'].append({
                        "id": new_id, "name": g_name, "target": g_target, "type": g_type, "category": g_cat, "saved": 0.0
                    })
                    st.session_state['adding_goal'] = False
                    st.rerun()
            if st.button("Close"):
                st.session_state['adding_goal'] = False
                st.rerun()

    if 'editing_goal' in st.session_state:
        target_goal = next((g for g in st.session_state['goals'] if g['id'] == st.session_state['editing_goal']), None)
        if target_goal:
            with st.expander(f"📝 Adjust: {target_goal['name']}", expanded=True):
                new_target = st.number_input("Update Target Limit ($)", value=float(target_goal['target']))
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Save Changes"):
                        target_goal['target'] = new_target
                        if target_goal['name'] == "Total Monthly Budget":
                            st.session_state['monthly_budget'] = new_target
                        del st.session_state['editing_goal']
                        st.rerun()
                with c2:
                    if st.button("Cancel Edit"):
                        del st.session_state['editing_goal']
                        st.rerun()
        else:
            del st.session_state['editing_goal']
            st.rerun()

def render_social_tab():
    st.markdown("## 👯 Eco-Social Hub")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🏆 Community Leaderboard")
        
        # Combine User with Friends
        me = {"name": "You (Seedling)", "score": st.session_state.get('avg_score', 0), "status": "Current Streak 🔥"}
        leaderboard = sorted(st.session_state['friends'] + [me], key=lambda x: x['score'], reverse=True)
        
        for i, friend in enumerate(leaderboard):
            rank = i + 1
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}th"
            color = "#10b981" if friend['name'].startswith("You") else "#94a3b8"
            
            st.markdown(f"""
                <div class="badge-card" style="border-left: 4px solid {color};">
                    <div style="font-size: 1.2rem; margin-right: 15px;">{emoji}</div>
                    <div class="badge-info">
                        <div class="badge-card-title">{friend['name']}</div>
                        <div class="badge-card-status">{friend['status']}</div>
                    </div>
                    <div style="font-weight: 800; font-size: 1.2rem; color: #10b981;">{friend['score']:.1f}</div>
                </div>
                <div style="margin-bottom: 10px;"></div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🤝 Connect & Invite")
        with st.container():
            st.write("Share your green wins with friends and earn +100 XP per invite!")
            st.button("🔗 Copy Invite Link")
            st.button("👥 Sync Contacts 🤝")
        
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        
        st.markdown("#### 🌟 Active Challenges")
        with st.container():
            st.markdown("**No-Fast-Fashion Week** 👗")
            st.caption("320 friends participating")
            st.progress(0.45)
            st.button("Join Challenge")

def render_badge_card(icon, title, status, progress_pct, unlocked=True):
    alpha = "1" if unlocked else "0.3"
    lock_class = "" if unlocked else "badge-card-locked"
    
    st.markdown(f"""
        <div class="badge-card {lock_class}">
            <div class="badge-icon-container">
                {icon}
            </div>
            <div class="badge-info">
                <div class="badge-card-title">{title}</div>
                <div class="badge-card-status">{status}</div>
                <div style="width: 100%; background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px; margin-top: 8px;">
                    <div style="width: {progress_pct}%; background: #10b981; height: 100%; border-radius: 2px; transition: width 0.5s ease;"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_gamification(df):
    # Recalculate Quests
    if not df.empty:
        # Eco-Commuter
        pub_trans_count = len(df[df['Category'] == 'Public Transport'])
        st.session_state['quests']['Eco-Commuter']['current'] = pub_trans_count
        if pub_trans_count >= 1: st.session_state['quests']['Eco-Commuter']['completed'] = True
        
        # Local Hero
        local_spend = df[df['Category'] == 'Local Food']['Amount'].sum()
        st.session_state['quests']['Local Hero']['current'] = local_spend
        if local_spend >= 50: st.session_state['quests']['Local Hero']['completed'] = True
        
        # Minimalist (Simplified for local stay)
        high_carbon_df = df[df['Category'].isin(['Fast Fashion', 'Food Delivery'])].copy()
        if high_carbon_df.empty:
            st.session_state['quests']['Minimalist']['current'] = 7
            st.session_state['quests']['Minimalist']['completed'] = True
        else:
            last_date = pd.to_datetime(high_carbon_df['Date']).max().date()
            diff = (datetime.now().date() - last_date).days
            st.session_state['quests']['Minimalist']['current'] = min(7, diff)
            if diff >= 7: st.session_state['quests']['Minimalist']['completed'] = True

    # Wrap in container for card effect
    with st.container():
        st.markdown("## 🎮 Gamification & Achievements")
        
        col1, col2, col3 = st.columns(3)
        
        # 1. Green XP & Leveling System
        with col1:
            st.markdown("#### Green XP Progress 🌱")
            
            quest_xp = sum([q['xp'] for q in st.session_state['quests'].values() if q['completed']])
            base_xp = len(df) * 10
            bonus_xp = len(df[df['Ethical_Score'] >= 5]) * 5
            total_xp = base_xp + bonus_xp + quest_xp
            
            level = (total_xp // 200) + 1
            xp_in_level = total_xp % 200
            progress_val = xp_in_level / 200.0
            
            # Visual Reward
            glow_intensity = min(0.4, 0.1 + (level * 0.05))
            st.session_state['glow_intensity'] = glow_intensity # Store for main loop

            st.markdown(f"<h2 style='margin-bottom: 0;'>Level {level}</h2>", unsafe_allow_html=True)
            st.markdown(f"Total XP: **{total_xp}**")
            st.progress(progress_val)
            
            remaining_xp = 200 - xp_in_level
            st.caption(f"✨ {remaining_xp} XP needed to reach Level {level + 1}")

    # 2. Quests
    with col2:
        st.markdown("#### Active Quests 🎯")
        for q_name, q_data in st.session_state['quests'].items():
            status = "✅" if q_data['completed'] else "⏳"
            color = "#10b981" if q_data['completed'] else "#64748b"
            st.markdown(f"""
                <div style='margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 5px;'>
                    <span style='color: {color}; font-weight: 600;'>{status} {q_name}</span><br/>
                    <small style='color: #94a3b8;'>{q_data['desc']} ({q_data['xp']} XP)</small>
                </div>
            """, unsafe_allow_html=True)

    # 3. Progress Badges
    with col3:
        st.markdown("#### Your Badges 🎖️")
        
        avg_score = df['Ethical_Score'].mean() if not df.empty else 0.0
        current_month_spending = get_current_month_spending(df)
        budget = st.session_state['monthly_budget']
        num_tx = len(df)
        
        # Eco-Warrior
        badge_name = "Eco-Warrior"
        if "Virtual Oak" in st.session_state['unlocked_items']: badge_name += " 🌳"
        
        ew_unlocked = avg_score >= 6
        ew_prog = min(100, (avg_score / 6.0) * 100)
        render_badge_card("🌍", badge_name, f"Avg Score: {avg_score:.1f}/6.0", ew_prog, ew_unlocked)
        
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True) # Spacer
        
        # Serial Tracker
        st_unlocked = num_tx >= 10
        st_prog = min(100, (num_tx / 10.0) * 100)
        render_badge_card("📝", "Serial Tracker", f"Logged: {num_tx}/10", st_prog, st_unlocked)
        
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True) # Spacer
            
        # Top Saver
        ts_unlocked = current_month_spending < budget and not df.empty
        ts_prog = min(100, (current_month_spending / budget) * 100) if ts_unlocked else 0
        render_badge_card("💰", "Top Saver", f"Spend: ${current_month_spending:.0f}/${budget:.0f}", ts_prog, ts_unlocked)

def render_reward_store(total_xp):
    with st.container():
        st.markdown("### 🏪 Green Rewards Store")
        
        spent = st.session_state['spent_points']
        available = total_xp - spent
        
        st.markdown(f"#### Available to Spend: **{available} XP**")
        
        # Store Items
        items = [
            {"name": "Midnight", "type": "Theme", "price": 500, "desc": "Sleek Midnight Blue skin", "icon": "🌌"},
            {"name": "Sunburst", "type": "Theme", "price": 1000, "desc": "High-contrast Solar theme", "icon": "☀️"},
            {"name": "Virtual Oak", "type": "Badge", "price": 200, "desc": "An Oak tree emoji badge", "icon": "🌳"},
            {"name": "Virtual Oasis", "type": "Effect", "price": 1500, "desc": "Floating leaves & particles", "icon": "🍃"},
            {"name": "Green NFT", "type": "Badge", "price": 2000, "desc": "Golden 'Legendary' badge", "icon": "✨"},
        ]
        
        cols = st.columns(len(items))
        for i, item in enumerate(items):
            with cols[i]:
                st.markdown(f"""
                    <div class="badge-card">
                        <div class="badge-icon-container">{item['icon']}</div>
                        <div class="badge-info">
                            <div class="badge-card-title">{item['name']}</div>
                            <small style="color: grey;">{item['price']} XP</small>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                is_unlocked = item['name'] in st.session_state['unlocked_items']
                
                if is_unlocked:
                    if st.button(f"Equip", key=f"equip_{i}"):
                        if item['type'] == "Theme": st.session_state['active_theme'] = item['name']
                        elif item['type'] == "Effect": st.session_state['active_effect'] = item['name']
                        st.success(f"Equipped {item['name']}!")
                        st.rerun()
                else:
                    can_afford = available >= item['price']
                    if st.button(f"Unlock", key=f"unlock_{i}", disabled=not can_afford):
                        st.session_state['spent_points'] += item['price']
                        st.session_state['unlocked_items'].append(item['name'])
                        st.success(f"Unlocked {item['name']}!")
                        st.rerun()

def render_ai_coach(df):
    st.markdown("### 🌿 Seedling AI Coach")
    
    with st.container():
        if df.empty:
            st.info("Hello! I'm Seedling. Log some transactions so I can start coaching you on your ethical spending journey!")
        else:
            # 1. Historical Trend Analysis
            curr_spend, curr_score = get_month_stats(df, 0)
            prev_spend, prev_score = get_month_stats(df, 1)
            
            if prev_score > 0:
                diff = curr_score - prev_score
                perc = (diff / prev_score) * 100
                if perc > 0:
                    st.success(f"📈 **Trending Up:** You're **{perc:.1f}% more ethical** than last month! Outstanding progress.")
                elif perc < 0:
                    st.warning(f"📉 **Trending Down:** Your ethical score is **{abs(perc):.1f}% lower** than last month. Let's get back on track!")
                else:
                    st.info("📊 **Steady Pace:** You're maintaining your ethical score from last month. Consistency is key!")
            else:
                st.info("🌱 **First Month:** Great start! I'll compare your progress next month once we have more history.")

            st.markdown("<div style='margin: 20px 0; border-top: 1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)

            # 2. Category Insights
            # ... (Category logic)
            delivery_spend = df[df['Category'] == 'Food Delivery']['Amount'].sum()
            insights = []
            if delivery_spend > 100:
                insights.append(f"🍔 Your **Food Delivery** spend is quite high (${delivery_spend:.2f}). Consider ordering from **Local Food** vendors!")
            
            if not insights:
                st.write("You're doing great! Keep logging to get more personalized tips.")
            else:
                for insight in insights[:1]:
                    st.info(f"💡 {insight}")
                
def render_dashboard(df):
    st.markdown(f"""
        <div style='margin-bottom: -20px;'>
            <h1 class='main-header'>GreenWallet AI 🌱</h1>
            <div class='sub-header' style='margin-top: -15px;'>Your Gamified Ethical Finance Dashboard</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate Total XP here for the store
    quest_xp = sum([q['xp'] for q in st.session_state['quests'].values() if q['completed']])
    base_xp = len(df) * 10
    bonus_xp = len(df[df['Ethical_Score'] >= 5]) * 5
    total_xp = base_xp + bonus_xp + quest_xp

    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Dashboard", "🎮 Quests & Badges", "🏪 Reward Store", "👯 Eco-Social"])
    
    with tab1:
        current_month_spent = get_current_month_spending(df)
        budget = st.session_state['monthly_budget']
        avg_score = df['Ethical_Score'].mean() if not df.empty else 0
        st.session_state['avg_score'] = avg_score # Store for Social
        
        # Top Metrics — Custom Stat Cards
        current_month_spent = get_current_month_spending(df) + st.session_state.get('total_deductions', 0)
        delta_goal = float(budget - current_month_spent)
        delta_color = "var(--accent)" if delta_goal >= 0 else "var(--bio-orange)"
        
        budget_pct = min(100, (current_month_spent / budget) * 100) if budget > 0 else 0
        score_pct = (avg_score / 10) * 100

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid var(--accent);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Monthly Expenses</div>
                            <div style="font-size: 2.5rem; font-weight: 800; letter-spacing: -1px;">${current_month_spent:.2f}</div>
                        </div>
                        <div style="font-size: 2.5rem; opacity: 0.3;">💸</div>
                    </div>
                    <div style="margin-top: 16px;">
                        <div style="height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden;">
                            <div style="width: {budget_pct}%; height: 100%; background: {delta_color}; border-radius: 4px; transition: width 0.8s ease; box-shadow: 0 0 12px {delta_color};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.8rem;">
                            <span style="color: {delta_color}; font-weight: 700;">${abs(delta_goal):.2f} {'remaining' if delta_goal >= 0 else 'over'}</span>
                            <span style="color: var(--text-secondary);">of ${budget:.0f} budget</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid var(--bio-purple);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Impact Score</div>
                            <div style="font-size: 2.5rem; font-weight: 800; letter-spacing: -1px;">{avg_score:.1f}<span style="font-size: 1.2rem; color: var(--text-secondary);"> / 10</span></div>
                        </div>
                        <div style="font-size: 2.5rem; opacity: 0.3;">🌿</div>
                    </div>
                    <div style="margin-top: 16px;">
                        <div style="height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden;">
                            <div style="width: {score_pct}%; height: 100%; background: var(--bio-purple); border-radius: 4px; transition: width 0.8s ease; box-shadow: 0 0 12px var(--bio-purple);"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.8rem;">
                            <span style="color: var(--bio-purple); font-weight: 700;">Average Ethical Score</span>
                            <span style="color: var(--text-secondary);">{len(df)} transactions</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Advanced Goals Section (New)
        render_goals_section(df)
        
        # AI Coach Section
        render_ai_coach(df)

        # Visualizations
        vc1, vc2 = st.columns(2)
        
        chart_theme = {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font_color': '#f8fafc',
            'margin': dict(t=40, b=40, l=40, r=40)
        }

        with vc1:
            st.markdown("### Spending by Category")
            if not df.empty:
                pie_data = df.groupby('Category')['Amount'].sum().reset_index()
                # Emerald/Forest palette
                fig_pie = px.pie(pie_data, values='Amount', names='Category', hole=0.6, 
                                 color_discrete_sequence=['#05ff91', '#00f5d4', '#9b5de5', '#f3722c', '#fee440'])
                fig_pie.update_layout(**chart_theme)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data yet to show pie chart.")
                
        with vc2:
            st.markdown("### Savings Forecast")
            if len(df) > 1:
                temp_df = df.copy()
                temp_df['Date_dt'] = pd.to_datetime(temp_df['Date'], errors='coerce')
                temp_df = temp_df.dropna(subset=['Date_dt'])
                df_sorted = temp_df.sort_values('Date_dt')
                df_sorted['Cumulative'] = df_sorted['Amount'].cumsum()
                
                fig_line = px.line(df_sorted, x='Date_dt', y='Cumulative', markers=True,
                                   labels={'Date_dt': 'Date', 'Cumulative': 'Cumulative Amount ($)'})
                fig_line.update_traces(line_color='#05ff91', line_width=4, marker=dict(size=10, color='#00f5d4'))
                fig_line.update_layout(**chart_theme)
                st.plotly_chart(fig_line, use_container_width=True)
                
                total_spent_all = df['Amount'].sum()
                unique_days = len(temp_df['Date_dt'].dt.date.unique())
                if unique_days > 0:
                    daily_avg = total_spent_all / unique_days
                    proj_30d = daily_avg * 30
                    st.info(f"📈 Projected 30-day spending based on current habits: **${proj_30d:.2f}**")
            else:
                st.info("Add more transactions for a spending forecast.")

    with tab2:
        render_gamification(df)
        
    with tab3:
        render_reward_store(total_xp)

    with tab4:
        render_social_tab()

    # Advanced Ledger Editor
    if not df.empty:
        with st.expander("📝 Advanced Ledger Editor", expanded=True):
            display_df = df.copy()
            display_df = display_df[['Date', 'Merchant', 'Description', 'Category', 'Amount', 'Ethical_Score']].sort_values('Date', ascending=False)
            edited_df = st.data_editor(display_df, key="ledger_v2", use_container_width=True)
            if not edited_df.equals(display_df):
                for index, r in edited_df.iterrows():
                    orig_match = (df['Date'] == r['Date']) & (df['Merchant'] == r['Merchant']) & (df['Amount'] == r['Amount'])
                    if any(orig_match):
                        df.loc[orig_match, 'Category'] = r['Category']
                        df.loc[orig_match, 'Ethical_Score'] = ETHICAL_SCORES.get(r['Category'], 5)
                st.session_state['transactions'] = df
                save_data()
                st.rerun()
    else:
        st.write("No transactions logged yet.")

# --- 4. MAIN APP LOOP ---
def main():
    init_state()
    st.markdown(get_theme_css(st.session_state['active_theme']), unsafe_allow_html=True)
    
    # Render Global Effects
    if st.session_state.get('active_effect') == "Virtual Oasis":
        for i in range(10):
            st.markdown(f'<div class="particle" style="left: {np.random.randint(0, 100)}%; width: {np.random.randint(5, 15)}px; height: {np.random.randint(5, 15)}px; animation-delay: {np.random.randint(0, 10)}s; background: {st.session_state.get("accent_color", "#10b981")}4D;"></div>', unsafe_allow_html=True)

    df = st.session_state['transactions']
    render_sidebar(df)
    
    render_dashboard(df)

    # Dynamic Background Glow
    if 'glow_intensity' in st.session_state:
        st.markdown(f"""
            <style>
                .stApp {{
                    background: radial-gradient(circle at 50% 0%, rgba(5, 255, 145, {st.session_state['glow_intensity'] * 0.4}) 0%, #051610 80%) !important;
                }}
            </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

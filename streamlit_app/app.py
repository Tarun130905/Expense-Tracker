"""
AI-Enhanced Expense Tracker - Main Streamlit Application
Interactive dashboard for expense tracking, categorization, and forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import ExpenseDatabase
from src.models import CategoryClassifier, SpendingForecaster, AnomalyDetector
from src.features import FeatureEngineer


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; margin-bottom: 1rem; }
    .stat-card { background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .metric-label { font-size: 0.9rem; color: #666; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1f77b4; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if 'db' not in st.session_state:
    st.session_state.db = ExpenseDatabase()

if 'classifier' not in st.session_state:
    st.session_state.classifier = CategoryClassifier()

if 'forecaster' not in st.session_state:
    st.session_state.forecaster = SpendingForecaster()

if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data():
    """Load transactions from database."""
    return st.session_state.db.get_transactions(limit=1000)


def get_categories():
    """Get list of categories."""
    return st.session_state.db.get_categories()


def train_models_on_data():
    """Train ML models on current database."""
    df = st.session_state.db.get_transactions(limit=10000)
    
    if len(df) < 10:
        st.warning("Need at least 10 transactions to train models")
        return False
    
    # Filter to labeled transactions
    df_labeled = df[df['category'].notna()].copy()
    
    if len(df_labeled) < 5:
        st.warning("Need at least 5 labeled transactions to train categorizer")
        return False
    
    try:
        # Train categorizer
        from src.features import FeatureEngineer
        X, feature_cols = FeatureEngineer.prepare_categorization_data(df_labeled)
        cat_metrics = st.session_state.classifier.train(X.values, df_labeled['category'].values, feature_cols)
        
        # Train anomaly detector
        st.session_state.anomaly_detector.fit(df_labeled)
        
        # Train forecaster
        monthly = FeatureEngineer.aggregate_transactions_to_monthly(df)
        if len(monthly) >= 3:
            X_forecast, feat_cols = FeatureEngineer.prepare_forecast_features(monthly)
            y_forecast = monthly.iloc[len(monthly) - len(X_forecast):]['amount'].values
            if len(X_forecast) > 0 and len(y_forecast) > 0:
                forecast_metrics = st.session_state.forecaster.train(X_forecast, y_forecast, feat_cols, n_rounds=50)
        
        return True
    except Exception as e:
        st.error(f"Error training models: {e}")
        return False


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# ⚙️ Navigation")
    page = st.radio(
        "Select page:",
        ["📊 Dashboard", "➕ Add Expense", "🔄 Import CSV", "🤖 AI Features", "📈 Analytics", "⚙️ Settings"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    df_all = load_data()
    if len(df_all) > 0:
        total_spending = df_all['amount'].sum()
        num_transactions = len(df_all)
        avg_transaction = total_spending / num_transactions if num_transactions > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Spent", f"₹{total_spending:,.2f}")
        with col2:
            st.metric("Transactions", num_transactions)
        
        st.metric("Avg/Transaction", f"₹{avg_transaction:.2f}")
    
    st.markdown("---")
    st.markdown("### 🔧 Tools")
    
    if st.button("🔄 Train Models", use_container_width=True):
        with st.spinner("Training models..."):
            if train_models_on_data():
                st.success("✓ Models trained successfully!")
            else:
                st.error("Failed to train models")
    
    if st.button("📥 Load Sample Data", use_container_width=True):
        try:
            count = st.session_state.db.bulk_insert_from_csv('sample_transactions.csv')
            st.success(f"✓ Loaded {count} sample transactions!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading sample data: {e}")


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "📊 Dashboard":
    st.markdown("# 💰 Expense Tracker Dashboard")
    
    df = load_data()
    
    if len(df) == 0:
        st.info("👋 No transactions yet. Start by adding expenses or loading sample data!")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = df['amount'].sum()
            st.metric("Total Spending", f"₹{total:,.2f}")
        
        with col2:
            avg = df['amount'].mean()
            st.metric("Avg Transaction", f"₹{avg:.2f}")
        
        with col3:
            num = len(df)
            st.metric("Total Transactions", num)
        
        with col4:
            max_trans = df['amount'].max()
            st.metric("Max Transaction", f"₹{max_trans:.2f}")
        
        st.markdown("---")
        
        # Category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📂 Spending by Category")
            cat_stats = st.session_state.db.get_category_stats()
            
            if len(cat_stats) > 0:
                fig_cat = px.pie(
                    cat_stats,
                    values='total_spending',
                    names='category',
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_cat.update_traces(textposition='inside', textinfo='label+percent')
                st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Top Categories")
            cat_stats_sorted = cat_stats.sort_values('total_spending', ascending=False).head(10)
            
            fig_bar = px.bar(
                cat_stats_sorted,
                x='total_spending',
                y='category',
                orientation='h',
                labels={'total_spending': 'Total Spending (₹)', 'category': 'Category'},
                color='total_spending',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # Monthly trend
        st.markdown("### 📈 Monthly Spending Trend")
        monthly_stats = st.session_state.db.get_monthly_stats()
        
        if len(monthly_stats) > 0:
            monthly_stats_sorted = monthly_stats.sort_values('month')
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=monthly_stats_sorted['month'],
                y=monthly_stats_sorted['total_spending'],
                mode='lines+markers',
                name='Monthly Spending',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            fig_line.update_layout(
                title="Monthly Total Spending",
                xaxis_title="Month",
                yaxis_title="Total Spending (₹)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        st.markdown("---")
        
        # Recent transactions
        st.markdown("### 📝 Recent Transactions")
        df_display = df[['date', 'vendor', 'description', 'category', 'amount']].head(20)
        st.dataframe(df_display, use_container_width=True)


# ============================================================================
# PAGE: ADD EXPENSE
# ============================================================================

elif page == "➕ Add Expense":
    st.markdown("# ➕ Add New Expense")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date = st.date_input("Date", datetime.now())
        amount = st.number_input("Amount (₹)", min_value=0.01, step=0.01)
        vendor = st.text_input("Vendor/Store", placeholder="e.g., Walmart")
    
    with col2:
        description = st.text_area("Description", placeholder="e.g., Weekly groceries")
        category = st.selectbox("Category", ["Auto-suggest"] + get_categories())
    
    if st.button("💾 Add Expense", use_container_width=True, type="primary"):
        # Predict category if needed
        if category == "Auto-suggest":
            if st.session_state.classifier.is_trained():
                try:
                    X, _ = FeatureEngineer.prepare_categorization_data(
                        pd.DataFrame({
                            'amount': [amount],
                            'vendor': [vendor],
                            'description': [description]
                        })
                    )
                    pred_cat, confidence = st.session_state.classifier.predict_single(X.iloc[0].values)
                    category = pred_cat
                    st.info(f"🤖 Predicted category: **{category}** (confidence: {confidence:.1%})")
                except Exception as e:
                    st.warning(f"Could not auto-predict: {e}")
                    category = None
            else:
                st.warning("Model not trained yet. Please train or select category manually.")
                category = None
        
        if category and category != "Auto-suggest":
            try:
                trans_id = st.session_state.db.add_transaction(
                    date=str(date),
                    amount=amount,
                    vendor=vendor,
                    description=description,
                    category=category
                )
                st.success(f"✓ Expense added (ID: {trans_id})")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding expense: {e}")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - **Auto-suggest**: System will predict category based on description
    - **Better descriptions**: More detailed descriptions improve auto-categorization
    - **Train models**: Add 10+ labeled transactions and train models for better predictions
    """)


# ============================================================================
# PAGE: IMPORT CSV
# ============================================================================

elif page == "🔄 Import CSV":
    st.markdown("# 🔄 Import Expenses from CSV")
    
    st.info("""
    Upload a CSV file with the following columns:
    - `date` (YYYY-MM-DD format)
    - `amount` (numeric, in ₹)
    - `vendor` (optional)
    - `description` (optional)
    - `category` (optional)
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Preview
            df_preview = pd.read_csv(temp_path, nrows=5)
            st.markdown("### 📋 Preview")
            st.dataframe(df_preview, use_container_width=True)
            
            # Import button
            if st.button("📥 Import All", use_container_width=True, type="primary"):
                try:
                    count = st.session_state.db.bulk_insert_from_csv(temp_path)
                    st.success(f"✓ Imported {count} transactions!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing: {e}")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ============================================================================
# PAGE: AI FEATURES
# ============================================================================

elif page == "🤖 AI Features":
    st.markdown("# 🤖 AI Features")
    
    tab1, tab2, tab3 = st.tabs(["🏷️ Categorization", "📊 Forecast", "🚨 Anomalies"])
    
    # TAB 1: Categorization
    with tab1:
        st.markdown("### 🏷️ Auto-Categorization")
        
        if not st.session_state.classifier.is_trained():
            st.warning("⚠️ Classifier not trained. Please train models first.")
        else:
            st.success("✓ Classifier trained and ready!")
            
            # Show uncategorized transactions
            df = load_data()
            df_uncategorized = df[df['category'].isna()]
            
            if len(df_uncategorized) > 0:
                st.markdown(f"#### Found {len(df_uncategorized)} uncategorized transactions")
                
                for idx, row in df_uncategorized.head(5).iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        try:
                            X, _ = FeatureEngineer.prepare_categorization_data(
                                pd.DataFrame({
                                    'amount': [row['amount']],
                                    'vendor': [row['vendor']],
                                    'description': [row['description']]
                                })
                            )
                            pred_cat, confidence = st.session_state.classifier.predict_single(X.iloc[0].values)
                            st.write(f"**{row['description']}** - {row['amount']}")
                            st.caption(f"Suggested: {pred_cat} ({confidence:.1%})")
                        except:
                            st.write(f"**{row['description']}** - ${row['amount']:.2f}")
                    
                    with col2:
                        if st.button("✓", key=f"accept_{idx}"):
                            st.session_state.db.update_transaction(row['id'], category=pred_cat)
                            st.rerun()
                    
                    with col3:
                        manual_cat = st.selectbox("Or", [""] + get_categories(), key=f"cat_{idx}")
                        if manual_cat:
                            st.session_state.db.update_transaction(row['id'], category=manual_cat)
                            st.rerun()
            else:
                st.info("✓ All transactions are categorized!")
    
    # TAB 2: Forecast
    with tab2:
        st.markdown("### 📊 Spending Forecast")
        
        if not st.session_state.forecaster.is_trained():
            st.warning("⚠️ Forecaster not trained. Train models to enable forecasting.")
        else:
            st.success("✓ Forecaster ready!")
            
            df = load_data()
            monthly = FeatureEngineer.aggregate_transactions_to_monthly(df)
            
            if len(monthly) >= 3:
                try:
                    from predict_expenses_commented import predict_next_month_spending
                    pred, artifacts = predict_next_month_spending(monthly, verbose=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Next Month", f"₹{pred:,.2f}", delta=None)
                    
                    with col2:
                        recent_avg = monthly['amount'].tail(3).mean()
                        delta = pred - recent_avg
                        st.metric("vs. Recent Average", f"₹{recent_avg:,.2f}", delta=f"₹{delta:+,.2f}")
                    
                    st.markdown("---")
                    st.markdown("#### 📈 Historical vs. Predicted")
                    
                    # Plot
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(
                        x=monthly['date'],
                        y=monthly['amount'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#1f77b4')
                    ))
                    
                    # Add prediction point
                    last_date = monthly['date'].max()
                    next_date = last_date + pd.Timedelta(days=30)
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=[next_date],
                        y=[pred],
                        mode='markers',
                        name='Predicted',
                        marker=dict(color='#ff7f0e', size=12)
                    ))
                    
                    fig_forecast.update_layout(height=400, hovermode='x unified')
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in forecasting: {e}")
            else:
                st.info("Need at least 3 months of data for forecasting.")
    
    # TAB 3: Anomalies
    with tab3:
        st.markdown("### 🚨 Anomaly Detection")
        
        df = load_data()
        
        if len(df) > 0:
            # Run anomaly detection
            df_anom = st.session_state.anomaly_detector.detect(df, threshold=2.0)
            df_anomalies = df_anom[df_anom['is_anomaly'] == 1]
            
            if len(df_anomalies) > 0:
                st.warning(f"⚠️ Found {len(df_anomalies)} anomalous transactions")
                
                st.dataframe(
                    df_anomalies[['date', 'vendor', 'category', 'amount', 'anomaly_score']],
                    use_container_width=True
                )
            else:
                st.success("✓ No anomalies detected!")
        else:
            st.info("No transactions to analyze.")


# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif page == "📈 Analytics":
    st.markdown("# 📈 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📅 By Category", "⏱️ By Month", "📊 Statistics"])
    
    df = load_data()
    
    with tab1:
        st.markdown("### 📅 Category Analysis")
        
        if len(df) > 0:
            # Category selector
            categories = get_categories()
            selected_cat = st.selectbox("Select category", categories)
            
            df_cat = df[df['category'] == selected_cat]
            
            if len(df_cat) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total", f"₹{df_cat['amount'].sum():.2f}")
                with col2:
                    st.metric("Count", len(df_cat))
                with col3:
                    st.metric("Average", f"₹{df_cat['amount'].mean():.2f}")
                with col4:
                    st.metric("Max", f"₹{df_cat['amount'].max():.2f}")
                
                st.dataframe(
                    df_cat[['date', 'vendor', 'description', 'amount']].sort_values('date', ascending=False),
                    use_container_width=True
                )
    
    with tab2:
        st.markdown("### ⏱️ Monthly Analysis")
        
        if len(df) > 0:
            monthly_cat = st.session_state.db.get_monthly_by_category()
            
            if len(monthly_cat) > 0:
                fig_stack = px.bar(
                    monthly_cat,
                    x='month',
                    y='amount',
                    color='category',
                    title="Monthly Spending by Category",
                    labels={'amount': 'Amount (₹)', 'month': 'Month'}
                )
                st.plotly_chart(fig_stack, use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Summary Statistics")
        
        if len(df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Overall")
                st.write(df['amount'].describe())
            
            with col2:
                st.markdown("#### By Category")
                cat_stats = st.session_state.db.get_category_stats()
                st.dataframe(cat_stats, use_container_width=True)


# ============================================================================
# PAGE: SETTINGS
# ============================================================================

elif page == "⚙️ Settings":
    st.markdown("# ⚙️ Settings & Configuration")
    
    tab1, tab2, tab3 = st.tabs(["📤 Export", "🔧 Model Status", "ℹ️ Info"])
    
    with tab1:
        st.markdown("### 📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start date")
        with col2:
            end_date = st.date_input("End date", datetime.now())
        
        if st.button("📥 Download CSV", use_container_width=True):
            try:
                df = st.session_state.db.get_transactions_by_date_range(
                    str(start_date), str(end_date)
                )
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Click to download",
                    data=csv,
                    file_name=f"expenses_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        st.markdown("### 🔧 Model Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.classifier.is_trained():
                st.success("✓ Categorizer: Trained")
            else:
                st.error("✗ Categorizer: Not trained")
        
        with col2:
            if st.session_state.forecaster.is_trained():
                st.success("✓ Forecaster: Trained")
            else:
                st.error("✗ Forecaster: Not trained")
        
        with col3:
            st.info(f"✓ Anomaly Detector: Ready")
        
        st.markdown("---")
        
        if st.button("🔄 Train All Models", use_container_width=True, type="primary"):
            with st.spinner("Training models..."):
                if train_models_on_data():
                    st.success("✓ Models trained successfully!")
                    st.rerun()
    
    with tab3:
        st.markdown("### ℹ️ About")
        st.info("""
        **AI-Enhanced Expense Tracker v1.0**
        
        Built with:
        - Streamlit for UI
        - RandomForest for categorization
        - XGBoost for forecasting
        - Statistical methods for anomalies
        
        📖 [Documentation](https://github.com)
        """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💰 AI-Enhanced Expense Tracker | Built with Streamlit & ML")

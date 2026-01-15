import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Metric container */
div[data-testid="metric-container"] {
    background-color: #0f172a;  /* dark slate */
    border: 1px solid #1e293b;
    padding: 16px;
    border-radius: 14px;
}

/* Metric label */
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 14px;
}

/* Metric value */
div[data-testid="metric-container"] div {
    color: #ffffff !important;
    font-size: 26px;
    font-weight: 700;
}

/* Delta text */
div[data-testid="metric-container"] span {
    font-size: 14px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Title
st.title("₿ Bitcoin Price Prediction with Machine Learning")
st.markdown("---")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("### Data Settings")
start_date = st.sidebar.date_input(
    "Start Date", 
    value=datetime(2020, 1, 1),
    help="Select the starting date for historical data"
)

st.sidebar.markdown("### Model Parameters")
lookback_days = st.sidebar.slider(
    "Lookback Days", 
    min_value=3, 
    max_value=30, 
    value=7,
    help="Number of previous days to use as features"
)

prediction_days = st.sidebar.slider(
    "Days to Predict Ahead", 
    min_value=1, 
    max_value=7, 
    value=1,
    help="How many days into the future to predict"
)

model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["Random Forest", "Linear Regression", "Both"],
    help="Select which ML model to use"
)

test_size = st.sidebar.slider(
    "Test Set Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    help="Percentage of data to use for testing"
) / 100


# Functions
@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def fetch_bitcoin_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    btc_data = yf.download(
        'BTC-USD',
        start=start_date,
        end=end_date,
        progress=False
    )

    #flatten columns & remove duplicates
    if isinstance(btc_data.columns, pd.MultiIndex):
        btc_data.columns = btc_data.columns.get_level_values(0)

    btc_data = btc_data.loc[:, ~btc_data.columns.duplicated()]

    #Ensure numeric
    btc_data['Close'] = pd.to_numeric(btc_data['Close'], errors='coerce')
    btc_data['Volume'] = pd.to_numeric(btc_data['Volume'], errors='coerce')

    return btc_data.dropna()


def create_features(df, lookback=7):
    """Create technical indicators and features"""
    df = df.copy()
    
    # Moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    
    # Price and volume changes
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Lagged features
    for i in range(1, lookback + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    df = df.dropna()
    return df


def prepare_data(df, target_days=1):
    """Prepare features and target variable"""
    df['Target'] = df['Close'].shift(-target_days)
    df = df.dropna()
    
    feature_cols = [col for col in df.columns 
                   if col not in ['Target', 'Close', 'Open', 'High', 'Low', 'Adj Close']]
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y, df.index


def train_model(X_train, X_test, y_train, y_test, model_type):
    """Train and evaluate a model"""
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, mse, mae, r2


def plot_predictions(y_test, predictions, dates, model_name):
    """Create prediction visualization"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, y_test.values, label='Actual Price', linewidth=2.5, color='#1f77b4')
    ax.plot(dates, predictions, label=f'{model_name} Prediction', 
            linewidth=2.5, color='#ff7f0e', alpha=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bitcoin Price (USD)', fontsize=12)
    ax.set_title(f'Bitcoin Price Prediction - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        'Feature': [str(f) for f in feature_names],  # 🔥 tuple-safe
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        importance_df['Feature'],
        importance_df['Importance']
    )
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


# Main app logic
if st.sidebar.button(" Run Prediction", type="primary", use_container_width=True):
    
    with st.spinner(" Fetching Bitcoin data..."):
        btc_data = fetch_bitcoin_data(start_date.strftime('%Y-%m-%d'))
    
    if len(btc_data) < 100:
        st.error(" Not enough data. Please select an earlier start date.")
        st.stop()
    
    st.success(f" Loaded {len(btc_data)} days of Bitcoin data")
    
    # Display current metrics
    current_price = float(btc_data['Close'].iloc[-1])
    price_change_24h = float(btc_data['Close'].iloc[-1] - btc_data['Close'].iloc[-2])
    price_change_pct = (price_change_24h / btc_data['Close'].iloc[-2]) * 100
    avg_7d = float(btc_data['Close'].tail(7).mean())
    volume_24h = float(btc_data['Volume'].iloc[-1])

    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Current BTC Price", 
        f"${current_price:,.2f}",
        f"{price_change_pct:+.2f}%"
    )
    col2.metric("24h Change", f"${price_change_24h:,.2f}")
    col3.metric("7-Day Average", f"${avg_7d:,.2f}")
    col4.metric("24h Volume", f"{volume_24h:,.0f}")
    
    st.markdown("---")
    
    # Feature engineering
    with st.spinner(" Engineering features..."):
        btc_features = create_features(btc_data, lookback=lookback_days)
    
    # Prepare data
    X, y, dates = prepare_data(btc_features, target_days=prediction_days)
    
    # Train-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = dates[split_idx:]
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Info about data split
    st.info(f" Training: {len(X_train)} samples | Testing: {len(X_test)} samples")
    
    # Train models
    models_to_run = ["Random Forest", "Linear Regression"] if model_choice == "Both" else [model_choice]
    
    for model_name in models_to_run:
        st.markdown(f"##  {model_name} Results")
        
        with st.spinner(f"Training {model_name}..."):
            model, y_pred, mse, mae, r2 = train_model(
                X_train_scaled, X_test_scaled, y_train, y_test, model_name
            )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Squared Error", f"${mse:,.2f}")
        col2.metric("Mean Absolute Error", f"${mae:,.2f}")
        col3.metric("R² Score", f"{r2:.4f}")
        
        # Prediction plot
        fig = plot_predictions(y_test, y_pred, test_dates, model_name)
        st.pyplot(fig)
        
        # Feature importance for Random Forest
        if model_name == "Random Forest":
            with st.expander(" View Feature Importance"):
                fig_importance = plot_feature_importance(model, X.columns)
                st.pyplot(fig_importance)
        
        st.markdown("---")
    
    # Raw data viewer
    with st.expander(" View Raw Bitcoin Data"):
        st.dataframe(btc_data.tail(50), use_container_width=True)
    
    # Download predictions
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Price': y_test.values,
        'Predicted Price': y_pred
    })
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label=" Download Predictions as CSV",
        data=csv,
        file_name=f"bitcoin_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    # Welcome screen
    st.info(" Configure settings in the sidebar and click **'Run Prediction'** to start!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### How to Use This App
        
        1. **Set the start date** for historical data
        2. **Choose lookback days** - how many previous days to use
        3. **Select prediction horizon** - days ahead to predict
        4. **Pick a model** - Random Forest typically performs better
        5. **Click 'Run Prediction'** to see results!
        """)
    
    with col2:
        st.markdown("""
        ### About the Models
        
        - **Linear Regression**: Simple baseline model
        - **Random Forest**: Ensemble method, usually more accurate
        
        ### 🔍 Features Used
        - Moving averages (7, 30 days)
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Price and volume lags
        - Volatility indicators
        """)
    
    st.warning("""
     **Disclaimer**: This tool is for educational purposes only. 
    Cryptocurrency markets are highly volatile and unpredictable. 
    Never make investment decisions based solely on ML predictions. 
    Always do your own research and consider professional financial advice.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("###  About")
st.sidebar.info("""
This app uses machine learning to predict Bitcoin prices based on historical data and technical indicators.

**Built with:**
- Python
- Streamlit
- scikit-learn
- yfinance
""")

"""
Expense Prediction Function - Fully Commented Implementation

This module demonstrates XGBoost-based forecasting for next-month spending prediction.
All major steps are documented with explanations of inputs, feature engineering, training,
and return values.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple, Dict, Any


def predict_next_month_spending(
    df: pd.DataFrame,
    xgb_params: Dict[str, Any] = None,
    n_rounds: int = 100,
    verbose: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Predict next month's total spending using XGBoost regression on historical data.
    
    This function takes historical transaction or monthly spending data, performs
    feature engineering (lag features, rolling statistics, time encoding), trains
    an XGBoost regressor, and returns a point forecast for the next month's spending.
    
    INPUTS:
    -------
    df : pd.DataFrame
        Historical transactions or pre-aggregated monthly data.
        
        If transaction-level (raw expenses):
            Required columns: 'date' (datetime-like), 'amount' (numeric, positive)
            The function will aggregate to monthly totals internally.
            
        If monthly-level (pre-aggregated):
            Columns: 'date' (end of month), 'amount' (monthly total)
            Already aggregated data is used directly.
            
        Example transaction-level DataFrame:
            date       | amount
            2025-01-05 | 42.50
            2025-01-15 | 18.75
            2025-02-03 | 55.20
            ...
            
        Example monthly-level DataFrame:
            date       | amount
            2025-01-31 | 1250.00
            2025-02-28 | 1320.50
            2025-03-31 | 1180.75
            ...
    
    xgb_params : dict, optional
        XGBoost hyperparameters. If None, sensible defaults are used.
        
        Supported keys:
        - 'objective': loss function (default: 'reg:squarederror' for regression)
        - 'learning_rate' (eta): 0.01-0.3 (default: 0.05, slower=more stable)
        - 'max_depth': 2-10 (default: 4, higher=more complex patterns)
        - 'subsample': 0.5-1.0 (default: 0.8, fraction of samples per tree)
        - 'colsample_bytree': 0.5-1.0 (default: 0.8, fraction of features per tree)
        - 'seed': random seed (default: 42)
        - 'gamma': minimum loss reduction (default: 0)
        - 'lambda': L2 regularization (default: 1)
        - 'alpha': L1 regularization (default: 0)
        
        Example high-capacity params (may overfit):
            {'learning_rate': 0.1, 'max_depth': 8, 'subsample': 0.9}
        
        Example conservative params (underfitting risk):
            {'learning_rate': 0.01, 'max_depth': 2, 'subsample': 0.6}
    
    n_rounds : int, optional
        Number of boosting rounds (iterations). Default: 100.
        Typical range: 50-500. More rounds = longer training but potentially
        better fit (until overfitting).
    
    verbose : bool, optional
        If True, prints training progress and final prediction. Default: False.
    
    RETURNS:
    --------
    Tuple[float, Dict]
        - float: Predicted spending amount for the next month (point forecast)
          Range: typically $500-$3000 for personal expenses (units: same as input)
        
        - dict: Artifacts dictionary containing:
          {
            'model': xgboost.Booster,           # Trained model object
            'feature_columns': list,            # Features used in prediction
            'last_features': pd.DataFrame,      # Feature values for next month
            'training_periods': int,            # Number of months in training
            'rmse': float                       # Root mean squared error on validation
          }
    
    FEATURE ENGINEERING DETAILS:
    ----------------------------
    
    1. DATA AGGREGATION (if input is transaction-level):
       - Group transactions by month (resample 'MS' or 'M')
       - Sum amounts per month
       - Result: DataFrame with 1 row per calendar month
       
       Code logic:
           monthly = df.set_index('date').resample('MS')['amount'].sum().reset_index()
       
       Example output:
           date       | amount (sum)
           2025-01-01 | 1200.50
           2025-02-01 | 1350.00
           2025-03-01 | 1180.75
    
    2. TIME FEATURES:
       - Extract 'month' (1-12) and 'year' from date for seasonality
       - Month helps capture recurring patterns (e.g., higher spending in Dec)
       
       Code logic:
           df['month'] = df['date'].dt.month        # 1-12
           df['year'] = df['date'].dt.year          # 2025, 2026, etc.
           df['day_of_year'] = df['date'].dt.dayofyear  # 1-365
    
    3. LAG FEATURES (autoregressive terms):
       - Lag-1: spending from 1 month ago
       - Lag-2: spending from 2 months ago
       - Lag-3: spending from 3 months ago
       
       Why useful:
         - Captures momentum: if you spent $1000 last month, likely $1000± this month
         - Autoregressive relationship: y(t) ≈ f(y(t-1), y(t-2), y(t-3))
       
       Code logic:
           for lag in [1, 2, 3]:
               df[f'lag_{lag}'] = df['amount'].shift(lag)
       
       Shifted data example:
           amount | lag_1 | lag_2 | lag_3
           1200.5 | NaN   | NaN   | NaN      <- month 1 (no history)
           1350.0 | 1200.5| NaN   | NaN      <- month 2 (1 lag available)
           1180.7 | 1350.0| 1200.5| NaN      <- month 3 (2 lags available)
           1250.0 | 1180.7| 1350.0| 1200.5   <- month 4 (all lags available)
           
       After dropna(): training starts from month 4 (earliest reliable row)
    
    4. ROLLING STATISTICS (adaptive baseline):
       - Rolling mean (3-month window): recent average spending
       - Rolling std (3-month window): recent spending variability
       
       Why useful:
         - Adapts to increasing/decreasing trends
         - Std indicates spending stability (low = predictable, high = volatile)
         - Helps model understand context (e.g., "spending is trending up")
       
       Code logic:
           df['rolling_mean_3'] = df['amount'].rolling(window=3).mean()
           df['rolling_std_3'] = df['amount'].rolling(window=3).std()
       
       Rolling statistics example (on 'amount' column):
           amount | rolling_mean_3 | rolling_std_3
           1200.5 | NaN            | NaN           <- need 3 values
           1350.0 | NaN            | NaN
           1180.7 | 1243.73        | 85.39         <- avg of 1200.5, 1350, 1180.7
           1250.0 | 1260.23        | 79.81         <- avg of 1350, 1180.7, 1250
    
    5. SEASONALITY ENCODING (optional advanced):
       - Could add day-of-week or holiday flags (e.g., Christmas, Black Friday)
       - Month sin/cos transforms for circular nature of seasons
       
       Code logic (sin/cos for smoothness):
           df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
           df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
       
       Advantage: smooth transition from Dec (month 12) to Jan (month 1)
    
    TRAINING PROCESS:
    ------------------
    
    1. SPLIT DATA (time-series aware):
       - Do NOT shuffle! Time order matters.
       - Use last 1-2 months as test/validation set
       - Use all prior months as training
       
       Example (n=24 months total):
           Train: months 1-22 (X_train, y_train)
           Test:  month 23 (X_test, y_test)
           Hold:  month 24 (we predict this)
       
       Code logic:
           train = monthly_feat.iloc[:-1]
           test = monthly_feat.iloc[-1:]
    
    2. PREPARE FEATURE MATRIX:
       - Select all feature columns (exclude 'date', 'amount', 'year')
       - X_train: features only, y_train: target (amount)
       
       Features used:
           ['month', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3']
       
       Code logic:
           feature_cols = [c for c in df.columns if c not in ('date', 'amount', 'year')]
           X_train = train[feature_cols]
           y_train = train['amount']
    
    3. CONVERT TO XGBOOST FORMAT:
       - Create DMatrix (XGBoost's optimized data container)
       
       Code logic:
           dtrain = xgb.DMatrix(X_train, label=y_train)
    
    4. TRAIN MODEL:
       - Call xgb.train() with:
         * params: hyperparameters dict
         * dtrain: training data
         * num_boost_round: iterations (100 typically)
         * early_stopping_rounds: stop if val_score doesn't improve
       
       Code logic:
           model = xgb.train(params, dtrain, num_boost_round=100)
       
       Output: Booster object (trained model)
    
    5. EVALUATE ON TEST SET (optional):
       - Compute RMSE, MAE, MAPE on held-out month
       
       Code logic:
           preds_test = model.predict(xgb.DMatrix(X_test))
           rmse = np.sqrt(np.mean((y_test - preds_test) ** 2))
           mae = np.mean(np.abs(y_test - preds_test))
           mape = 100 * np.mean(np.abs((y_test - preds_test) / y_test))
    
    PREDICTION (Next Month):
    -------------------------
    
    1. BUILD FEATURE VECTOR FOR NEXT MONTH:
       - Use most recent values to create next month's features
       - lag_1 = last month's amount (y[t])
       - lag_2 = amount from 2 months ago (y[t-1])
       - lag_3 = amount from 3 months ago (y[t-2])
       - rolling_mean_3 = last row's rolling_mean_3
       - rolling_std_3 = last row's rolling_std_3
       - month = (current month % 12) + 1 (for seasonality)
       
       Code logic:
           last_row = monthly_feat.iloc[-1]
           next_features = {
               'lag_1': last_row['amount'],
               'lag_2': last_row['lag_1'],
               'lag_3': last_row['lag_2'],
               'rolling_mean_3': last_row['rolling_mean_3'],
               'rolling_std_3': last_row['rolling_std_3'],
               'month': (last_row['month'] % 12) + 1
           }
       
       Example:
           If last month (Oct) spent $1250, and 3-month avg is $1243.73:
           
           lag_1=1250, lag_2=1180.7, lag_3=1350, 
           rolling_mean_3=1243.73, rolling_std_3=79.81, month=11
    
    2. PREDICT:
       - Convert to DMatrix and call model.predict()
       - Returns single float (predicted spending amount)
       
       Code logic:
           X_next = pd.DataFrame([next_features])[feature_cols]
           dnext = xgb.DMatrix(X_next)
           pred_next = float(model.predict(dnext)[0])
       
       Output example:
           pred_next = 1268.45  # Predicted November spending: $1268.45
    
    RETURN VALUE INTERPRETATION:
    ----------------------------
    
    Returns: (1268.45, {'model': ..., 'feature_columns': [...], ...})
    
    - pred_next = 1268.45
      Interpretation: "Expected spending next month is ~$1268.45"
      
      Use for:
      - Budget planning ("I should set aside $1268 for next month")
      - Anomaly detection (if next month > 1268 * 1.5, flag as high)
      - Comparison (vs. user's budget target)
    
    - Confidence interval (optional enhancement):
      Could compute via ensemble predictions or quantile regression:
      - Lower bound (10th percentile): $1100
      - Point forecast (median): $1268
      - Upper bound (90th percentile): $1400
    
    COMPLETE FUNCTION EXAMPLE CALL:
    --------------------------------
    
    # Load 6 months of transaction data
    df = pd.read_csv('transactions.csv')  # columns: date, amount
    
    # Predict next month
    pred, artifacts = predict_next_month_spending(
        df,
        xgb_params={'learning_rate': 0.05, 'max_depth': 4},
        n_rounds=100,
        verbose=True
    )
    
    print(f"Predicted next month: ${pred:.2f}")
    print(f"Model trained on {artifacts['training_periods']} months")
    print(f"Validation RMSE: ${artifacts['rmse']:.2f}")
    
    # Example output:
    # Predicted next month: $1268.45
    # Model trained on 5 months
    # Validation RMSE: $45.30
    
    CONSIDERATIONS FOR PRODUCTION:
    -------------------------------
    
    - MINIMUM DATA: Ideally 12+ months (more data = better seasonality capture)
    - RETRAINING: Retrain monthly or quarterly with new data
    - FEATURE DRIFT: If spending patterns change (job change, relocation),
      retrain with recent 12-month window only
    - ERROR METRICS: Monitor forecast error (MAPE) over time
    - USER CONTEXT: Combine with user-provided adjustments
      (e.g., "I'm going on vacation next month, add 20%")
    """
    
    # =========================================================================
    # STEP 1: INPUT VALIDATION
    # =========================================================================
    
    # Verify required columns exist
    if 'date' not in df.columns or 'amount' not in df.columns:
        raise ValueError(
            "Input DataFrame must contain 'date' and 'amount' columns. "
            f"Got columns: {df.columns.tolist()}"
        )
    
    # Create working copy and convert date to datetime
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    
    if verbose:
        print(f"[Input] {len(data)} rows from {data['date'].min()} to {data['date'].max()}")
    
    # =========================================================================
    # STEP 2: AGGREGATE TO MONTHLY TOTALS (if needed)
    # =========================================================================
    
    # Check if data is already monthly (detect by row count and date spacing)
    # If more than 1-2 rows per month, it's likely transaction-level
    date_diffs = data['date'].diff().dt.days
    avg_day_spacing = date_diffs[1:].mean()  # Skip first NaN
    
    if avg_day_spacing > 15:  # Likely already monthly
        monthly = data.copy()
    else:
        # Aggregate transaction-level data to monthly
        monthly = (
            data.set_index('date')
            .resample('MS')['amount']
            .sum()
            .reset_index()
            .rename(columns={'amount': 'amount'})
        )
        monthly['date'] = monthly['date'].dt.to_period('M').dt.end_time
    
    if verbose:
        print(f"[Monthly] {len(monthly)} months aggregated")
    
    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    
    # Extract time features from date
    monthly['month'] = monthly['date'].dt.month  # 1-12
    monthly['year'] = monthly['date'].dt.year
    monthly['quarter'] = monthly['date'].dt.quarter  # 1-4
    
    # Create lag features (previous spending amounts)
    # These capture autoregressive patterns: spending this month depends on last month
    for lag in [1, 2, 3]:
        monthly[f'lag_{lag}'] = monthly['amount'].shift(lag)
    
    # Rolling window statistics (3-month moving average and std)
    # These capture short-term trends and volatility
    monthly['rolling_mean_3'] = monthly['amount'].rolling(window=3, min_periods=1).mean()
    monthly['rolling_std_3'] = monthly['amount'].rolling(window=3, min_periods=1).std()
    monthly['rolling_std_3'] = monthly['rolling_std_3'].fillna(0)  # First value std=0
    
    # Additional aggregate features
    monthly['rolling_mean_6'] = monthly['amount'].rolling(window=6, min_periods=1).mean()
    
    # Sinusoidal encoding for monthly seasonality (circular representation)
    # Jan (1) and Dec (12) should be close in feature space
    monthly['month_sin'] = np.sin(2 * np.pi * monthly['month'] / 12)
    monthly['month_cos'] = np.cos(2 * np.pi * monthly['month'] / 12)
    
    # Year-over-year growth rate (if enough data)
    if len(monthly) > 12:
        monthly['yoy_growth'] = (monthly['amount'] / monthly['amount'].shift(12) - 1) * 100
        monthly['yoy_growth'] = monthly['yoy_growth'].fillna(0)
    else:
        monthly['yoy_growth'] = 0
    
    if verbose:
        print(f"[Features] Engineered {len(monthly.columns)} columns")
    
    # =========================================================================
    # STEP 4: PREPARE TRAINING DATA (handle NaN from lag/rolling)
    # =========================================================================
    
    # Remove rows with NaN values (created by lag operations)
    # Only keep rows where we have sufficient history
    monthly_feat = monthly.dropna(subset=['lag_2']).copy()
    
    if len(monthly_feat) < 3:
        raise ValueError(
            f"Insufficient data after feature engineering. "
            f"Need at least 12+ months, got {len(monthly_feat)} valid rows."
        )
    
    # Split into train and test (time-series aware)
    # Use all but last month for training, last month for validation
    train = monthly_feat.iloc[:-1]
    test = monthly_feat.iloc[-1:]
    
    # Identify feature columns (exclude target and metadata)
    feature_cols = [
        c for c in monthly_feat.columns 
        if c not in ('date', 'amount', 'year', 'quarter')
    ]
    
    if verbose:
        print(f"[Train/Test] Train: {len(train)} months, Test: {len(test)} months")
        print(f"[Features] Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    # =========================================================================
    # STEP 5: XGBOOST MODEL TRAINING
    # =========================================================================
    
    X_train = train[feature_cols].values
    y_train = train['amount'].values
    
    # Convert to XGBoost's optimized DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    
    # Set default hyperparameters if not provided
    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:squarederror',    # Mean squared error for regression
            'learning_rate': 0.05,               # Slower learning = more stable
            'max_depth': 4,                      # Tree depth (4 is moderate complexity)
            'subsample': 0.8,                    # Use 80% of samples per tree
            'colsample_bytree': 0.8,             # Use 80% of features per tree
            'seed': 42,                          # Reproducibility
            'gamma': 0,                          # No minimum loss reduction
            'lambda': 1.0,                       # L2 regularization
            'alpha': 0,                          # No L1 regularization
        }
    
    if verbose:
        print(f"[XGBoost] Training with params: {xgb_params}")
    
    # Train the model
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False
    )
    
    if verbose:
        print(f"[Training] Completed {n_rounds} boosting rounds")
    
    # =========================================================================
    # STEP 6: EVALUATE ON TEST SET (optional)
    # =========================================================================
    
    X_test = test[feature_cols].values
    y_test = test['amount'].values
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    preds_test = model.predict(dtest)
    
    rmse = np.sqrt(np.mean((y_test - preds_test) ** 2))
    mae = np.mean(np.abs(y_test - preds_test))
    mape = 100 * np.mean(np.abs((y_test - preds_test) / y_test))
    
    if verbose:
        print(f"[Validation] RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MAPE: {mape:.2f}%")
    
    # =========================================================================
    # STEP 7: BUILD FEATURES FOR NEXT MONTH
    # =========================================================================
    
    last_row = monthly_feat.iloc[-1]
    
    # Build feature dict for next month
    # lag_1: this month's amount becomes next month's lag_1
    # rolling stats: use last computed values
    # month: increment and wrap (Dec 12 -> Jan 1)
    next_features = {
        'lag_1': last_row['amount'],
        'lag_2': last_row['lag_1'],
        'lag_3': last_row['lag_2'],
        'rolling_mean_3': last_row['rolling_mean_3'],
        'rolling_std_3': last_row['rolling_std_3'],
        'rolling_mean_6': last_row['rolling_mean_6'],
        'month': ((last_row['month'] % 12) + 1),
        'quarter': (((last_row['month'] - 1 + 1) // 3) % 4) + 1,
        'month_sin': np.sin(2 * np.pi * ((last_row['month'] % 12) + 1) / 12),
        'month_cos': np.cos(2 * np.pi * ((last_row['month'] % 12) + 1) / 12),
        'yoy_growth': last_row['yoy_growth'],
    }
    
    # Create DataFrame with same feature order as training
    X_next = pd.DataFrame([next_features])[feature_cols]
    dnext = xgb.DMatrix(X_next, feature_names=feature_cols)
    
    if verbose:
        print(f"[Next Month Features]")
        for col, val in next_features.items():
            if col in feature_cols[:5]:  # Show first 5 features
                print(f"  {col}: {val:.4f}")
    
    # =========================================================================
    # STEP 8: PREDICT NEXT MONTH SPENDING
    # =========================================================================
    
    pred_next = float(model.predict(dnext)[0])
    
    if verbose:
        print(f"[Prediction] Next month spending: ${pred_next:.2f}")
    
    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    
    artifacts = {
        'model': model,
        'feature_columns': feature_cols,
        'last_features': X_next,
        'training_periods': len(train),
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'last_month_amount': float(last_row['amount']),
        'rolling_mean_3': float(last_row['rolling_mean_3']),
    }
    
    return pred_next, artifacts

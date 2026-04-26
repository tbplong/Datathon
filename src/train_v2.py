import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df = pd.read_csv('data/revenue_features_full.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"Raw data: {df['date'].min().date()} -> {df['date'].max().date()}, shape={df.shape}")

TARGETS = ['Revenue', 'COGS']

# ══════════════════════════════════════════════════════════════════════
# 2. REMOVE WEAK & CONSTANT FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Removing weak-correlation & constant features")
print("=" * 70)

# Features with |correlation| < 0.1 with BOTH Revenue and COGS
# Identified via correlation analysis on the raw dataset
WEAK_FEATURES = [
    # Constant-value features (zero variance → NaN correlation)
    'items_reorder',           # always 0
    'unique_sizes',            # always 4

    # Promo features (|corr| < 0.1 with both targets)
    'active_promos_count',
    'promo_type_fixed',
    'promo_categories_covered',
    'promo_type_percentage',
    'promo_channel_in_store',
    'promo_channel_social_media',
    'promo_channel_email',
    'has_stackable',
    'avg_min_order_value',

    # Traffic session breakdown (|corr| < 0.1 with both targets)
    'traffic_sessions_social_media',
    'traffic_sessions_paid_search',
    'traffic_sessions_organic_search',
    'traffic_sessions_email_campaign',
    'traffic_sessions_referral',
    'traffic_sessions_direct',

    # Other weak features
    'items_overstock',
    'avg_installments',
    'avg_fill_rate',
    'avg_bounce_rate',
    'avg_session_duration_sec',
    'avg_delivery_days',
    'avg_rating',
]

existing_weak = [f for f in WEAK_FEATURES if f in df.columns]
df = df.drop(columns=existing_weak)
print(f"  Removed {len(existing_weak)} weak/constant features:")
for f in existing_weak:
    print(f"    - [REMOVED] {f}")
print(f"  Remaining shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Feature engineering")
print("=" * 70)

# -- Calendar features --
df['year']         = df['date'].dt.year
df['month']        = df['date'].dt.month
df['day_of_week']  = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['day_of_year']  = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding for month and day_of_week
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Interaction: day_of_week × month
df['dow_x_month'] = df['day_of_week'] * 12 + df['month']

# -- Trend features --
df['year_trend']      = df['year'] - 2012
df['years_since_2012'] = df['year'] - 2012
df['trend_sq']        = df['years_since_2012'] ** 2  # quadratic trend

# -- Lag features (target lags) --
LAG_DAYS = [1, 2, 3, 7, 14, 30, 365]
for t in TARGETS:
    col = t
    for lag in LAG_DAYS:
        df[f'{t.lower()}_lag_{lag}d'] = df[col].shift(lag)

# -- Lag features (operational metrics) --
# Key operational columns that are available same-day (need to lag for no-leak)
OP_COLS = [
    'order_count', 'total_quantity', 'unique_customers', 'unique_zips',
    'total_payment_value', 'total_shipping_fee', 'total_discount',
    'total_sessions', 'total_unique_visitors', 'total_page_views',
    'new_customers_count', 'return_quantity', 'refund_amount',
    'total_reviews', 'total_stock_on_hand', 'total_units_received',
]
OP_COLS = [c for c in OP_COLS if c in df.columns]

for c in OP_COLS:
    df[f'{c}_lag_1d'] = df[c].shift(1)
    df[f'{c}_lag_7d'] = df[c].shift(7)

# -- Rolling statistics (on lagged values to prevent leakage) --
ROLLING_WINDOWS = [7, 14, 30]
for t in TARGETS:
    lag1 = f'{t.lower()}_lag_1d'
    if lag1 not in df.columns:
        continue
    for w in ROLLING_WINDOWS:
        df[f'{t.lower()}_rmean_{w}d'] = df[lag1].rolling(w, min_periods=1).mean()
        df[f'{t.lower()}_rstd_{w}d']  = df[lag1].rolling(w, min_periods=1).std()
    # Expanding mean (all history)
    df[f'{t.lower()}_exp_mean'] = df[lag1].expanding(min_periods=1).mean()

# Rolling stats for key operational features
for c in ['order_count', 'total_quantity', 'total_payment_value']:
    lag1 = f'{c}_lag_1d'
    if lag1 not in df.columns:
        continue
    for w in [7, 30]:
        df[f'{c}_rmean_{w}d'] = df[lag1].rolling(w, min_periods=1).mean()

# -- Momentum & deviation features --
for t in TARGETS:
    tl = t.lower()
    lag1  = f'{tl}_lag_1d'
    r7    = f'{tl}_rmean_7d'
    r14   = f'{tl}_rmean_14d'
    r30   = f'{tl}_rmean_30d'

    # Momentum: short-term vs long-term rolling mean
    if r7 in df.columns and r30 in df.columns:
        df[f'{tl}_momentum_7_30'] = df[r7] / df[r30].replace(0, np.nan)
    if r7 in df.columns and r14 in df.columns:
        df[f'{tl}_momentum_7_14'] = df[r7] / df[r14].replace(0, np.nan)

    # Deviation from rolling mean
    if lag1 in df.columns and r7 in df.columns:
        df[f'{tl}_dev_from_7d'] = df[lag1] / df[r7].replace(0, np.nan)
    if lag1 in df.columns and r30 in df.columns:
        df[f'{tl}_dev_from_30d'] = df[lag1] / df[r30].replace(0, np.nan)

    # Lag differences (acceleration)
    lag7  = f'{tl}_lag_7d'
    lag30 = f'{tl}_lag_30d'
    if lag1 in df.columns and lag7 in df.columns:
        df[f'{tl}_diff_1d_7d'] = df[lag1] - df[lag7]
    if lag1 in df.columns and lag30 in df.columns:
        df[f'{tl}_diff_1d_30d'] = df[lag1] - df[lag30]

# Revenue-to-COGS ratio (gross margin proxy)
if 'revenue_lag_1d' in df.columns and 'cogs_lag_1d' in df.columns:
    df['rev_cogs_ratio_lag1'] = df['revenue_lag_1d'] / df['cogs_lag_1d'].replace(0, np.nan)

# Revenue coefficient of variation (7-day)
if 'revenue_rstd_7d' in df.columns and 'revenue_rmean_7d' in df.columns:
    df['revenue_cv_7d'] = df['revenue_rstd_7d'] / df['revenue_rmean_7d'].replace(0, np.nan)

# -- Fill NaNs from lagged/rolling features --
df = df.fillna(0)

# ══════════════════════════════════════════════════════════════════════
# 4. FINAL FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Final feature selection & correlation check")
print("=" * 70)

# Exclude raw same-day operational columns (only use their lagged versions)
# These would cause data leakage if used directly
RAW_SAME_DAY = [
    'order_count', 'total_quantity', 'unique_customers', 'unique_zips',
    'total_payment_value', 'total_shipping_fee', 'total_discount',
    'total_sessions', 'total_unique_visitors', 'total_page_views',
    'new_customers_count', 'return_quantity', 'refund_amount',
    'total_reviews', 'total_stock_on_hand', 'total_units_received',
    'avg_unit_price', 'unique_items', 'avg_product_price', 'avg_product_cogs',
    'unique_colors',
]
# Also exclude detailed breakdowns that are same-day
SAME_DAY_BREAKDOWN = [c for c in df.columns if any(c.startswith(p) for p in [
    'orders_payment_', 'orders_device_', 'orders_source_',
    'orders_status_', 'orders_region_',
    'returns_reason_', 'new_cust_acq_', 'new_cust_age_', 'new_cust_gen_',
    'qty_category_', 'qty_segment_',
    'promo_channel_', 'max_discount_value',
])]

EXCLUDE_COLS = set(['date', 'year'] + TARGETS + RAW_SAME_DAY + SAME_DAY_BREAKDOWN)
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

# Post-engineering correlation check — drop features still weakly correlated
corr_rev  = df[feature_cols].corrwith(df['Revenue']).abs()
corr_cogs = df[feature_cols].corrwith(df['COGS']).abs()
max_corr  = pd.concat([corr_rev, corr_cogs], axis=1).max(axis=1)

CORR_THRESHOLD = 0.05
weak_engineered = max_corr[max_corr < CORR_THRESHOLD].index.tolist()
if weak_engineered:
    print(f"  Removing {len(weak_engineered)} weakly-correlated engineered features (|corr| < {CORR_THRESHOLD}):")
    for f in weak_engineered:
        print(f"    - [REMOVED] {f} (max|corr|={max_corr[f]:.4f})")
    feature_cols = [c for c in feature_cols if c not in weak_engineered]

# Print final correlation summary
corr_rev  = df[feature_cols].corrwith(df['Revenue']).abs()
corr_cogs = df[feature_cols].corrwith(df['COGS']).abs()
max_corr  = pd.concat([corr_rev, corr_cogs], axis=1).max(axis=1).sort_values(ascending=False)

print(f"\n  Final feature count: {len(feature_cols)}")
print(f"\n  Top 20 features by max |correlation|:")
for feat, val in max_corr.head(20).items():
    print(f"    {val:.4f}  {feat}")
print(f"\n  Bottom 10 features by max |correlation|:")
for feat, val in max_corr.tail(10).items():
    print(f"    {val:.4f}  {feat}")

X_all = df[feature_cols].copy()

# ══════════════════════════════════════════════════════════════════════
# 5. SAMPLE WEIGHTS (recency bias)
# ══════════════════════════════════════════════════════════════════════
years = df['year'].values
max_year = years.max()
decay_rate = 0.15
sample_weights = np.exp(-decay_rate * (max_year - years))
print(f"\n  Sample weight range: {sample_weights.min():.4f} -> {sample_weights.max():.4f}")

# ══════════════════════════════════════════════════════════════════════
# 6. HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════
best_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "colsample_bylevel": 0.6,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "early_stopping_rounds": 150,
}

# ══════════════════════════════════════════════════════════════════════
# 7. LOG-TRANSFORM HELPERS
# ══════════════════════════════════════════════════════════════════════
def log_t(y):   return np.log1p(y)
def inv_t(y):   return np.expm1(y)

# ══════════════════════════════════════════════════════════════════════
# 8. TIME-SERIES CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
tscv = TimeSeriesSplit(n_splits=5)
print("\n" + "=" * 70)
print("STEP 5: Time-Series Cross-Validation")
print("=" * 70)

for target in TARGETS:
    y_raw = df[target]
    y_log = log_t(y_raw)
    rmses, maes, r2s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_vl  = y_log.iloc[train_idx], y_log.iloc[val_idx]
        w_tr = sample_weights[train_idx]

        model = XGBRegressor(**best_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_vl)],
                  sample_weight=w_tr, verbose=0)

        preds = inv_t(model.predict(X_val))
        y_val_raw = y_raw.iloc[val_idx]

        rmse = np.sqrt(mean_squared_error(y_val_raw, preds))
        mae  = mean_absolute_error(y_val_raw, preds)
        r2   = r2_score(y_val_raw, preds)
        rmses.append(rmse); maes.append(mae); r2s.append(r2)
        print(f"  [{target}] Fold {fold+1}: RMSE={rmse:>14,.2f}  MAE={mae:>14,.2f}  R²={r2:.4f}")

    print(f"\n  {target} CV Summary:")
    print(f"    RMSE:  {np.mean(rmses):>14,.2f}  (±{np.std(rmses):>12,.2f})")
    print(f"    MAE:   {np.mean(maes):>14,.2f}  (±{np.std(maes):>12,.2f})")
    print(f"    R²:    {np.mean(r2s):>14.4f}  (±{np.std(r2s):>12.4f})")
    print()

# ══════════════════════════════════════════════════════════════════════
# 9. HOLD-OUT VALIDATION (2022)
# ══════════════════════════════════════════════════════════════════════
val_mask  = df['year'] == 2022
X_tr_ho   = X_all[~val_mask]
X_val_ho  = X_all[val_mask]

print("=" * 70)
print("STEP 6: Hold-out Validation (2022)")
print("=" * 70)

for target in TARGETS:
    y_raw = df[target]
    y_log = log_t(y_raw)
    w_tr  = sample_weights[~val_mask]

    model_ho = XGBRegressor(**best_params)
    model_ho.fit(X_tr_ho, y_log[~val_mask],
                 eval_set=[(X_val_ho, y_log[val_mask])],
                 sample_weight=w_tr, verbose=0)

    preds = inv_t(model_ho.predict(X_val_ho))
    y_val_raw = y_raw[val_mask]
    rmse = np.sqrt(mean_squared_error(y_val_raw, preds))
    mae  = mean_absolute_error(y_val_raw, preds)
    r2   = r2_score(y_val_raw, preds)
    print(f"\n  {target}:")
    print(f"    RMSE:  {rmse:>14,.2f}")
    print(f"    MAE:   {mae:>14,.2f}")
    print(f"    R²:    {r2:>14.4f}")

# ══════════════════════════════════════════════════════════════════════
# 10. TRAIN FINAL MODELS ON ALL DATA
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Training final models on ALL data")
print("=" * 70)

final_params = {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'}

model_revenue = XGBRegressor(**final_params)
model_revenue.fit(X_all, log_t(df['Revenue']), sample_weight=sample_weights, verbose=500)

model_cogs = XGBRegressor(**final_params)
model_cogs.fit(X_all, log_t(df['COGS']), sample_weight=sample_weights, verbose=500)

# Feature importance
print("\nTop 20 features (Revenue):")
imp_rev = pd.Series(model_revenue.feature_importances_, index=feature_cols).nlargest(20)
for feat, val in imp_rev.items():
    print(f"  {val:.4f}  {feat}")

print("\nTop 20 features (COGS):")
imp_cogs = pd.Series(model_cogs.feature_importances_, index=feature_cols).nlargest(20)
for feat, val in imp_cogs.items():
    print(f"  {val:.4f}  {feat}")

# ══════════════════════════════════════════════════════════════════════
# 11. BUILD TEST FEATURES & PREDICT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Building test features & predicting")
print("=" * 70)

test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
test_df = pd.DataFrame({'date': test_dates})

# Calendar features
test_df['month']        = test_df['date'].dt.month
test_df['day_of_month'] = test_df['date'].dt.day
test_df['day_of_week']  = test_df['date'].dt.dayofweek
test_df['day_of_year']  = test_df['date'].dt.dayofyear
test_df['week_of_year'] = test_df['date'].dt.isocalendar().week.astype(int)
test_df['is_weekend']   = (test_df['day_of_week'] >= 5).astype(int)
test_df['month_sin']    = np.sin(2 * np.pi * test_df['month'] / 12)
test_df['month_cos']    = np.cos(2 * np.pi * test_df['month'] / 12)
test_df['dow_sin']      = np.sin(2 * np.pi * test_df['day_of_week'] / 7)
test_df['dow_cos']      = np.cos(2 * np.pi * test_df['day_of_week'] / 7)
test_df['dow_x_month']  = test_df['day_of_week'] * 12 + test_df['month']

# Trend features
test_df['year']            = test_df['date'].dt.year
test_df['year_trend']      = test_df['year'] - 2012
test_df['years_since_2012'] = test_df['year'] - 2012
test_df['trend_sq']        = test_df['years_since_2012'] ** 2

# Seasonal profile: use historical (month, day_of_month) averages
# for lag/rolling features that we can't compute directly for future dates
group_keys = ['month', 'day_of_month']
lag_rolling_cols = [c for c in feature_cols if c not in
    group_keys + ['day_of_week', 'day_of_year', 'week_of_year', 'is_weekend',
                  'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'dow_x_month',
                  'year_trend', 'years_since_2012', 'trend_sq',
                  'year']]

# Build seasonal averages from training data
seasonal_profile = df.groupby(group_keys)[lag_rolling_cols].mean().reset_index()
test_df = test_df.merge(seasonal_profile, on=group_keys, how='left')

# Fill any NaN (e.g., Feb 29 in non-leap-year training data)
for col in lag_rolling_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].interpolate(method='linear').bfill().ffill()
    else:
        test_df[col] = 0

# Trend scaling — use 2019–2022 CAGR (same macro regime)
# annual_rev_recent = df[df['year'] >= 2019].groupby('year')['Revenue'].sum()
# if len(annual_rev_recent) >= 2:
#     cagr = (annual_rev_recent.iloc[-1] / annual_rev_recent.iloc[0]) ** (1 / (len(annual_rev_recent) - 1))
#     years_ahead = test_df['year'] - 2022
#     trend_mult = cagr ** years_ahead
#     print(f"  Trend CAGR (2019–2022): {cagr:.4f}")

#     # Scale lag/rolling features that represent absolute values
#     scale_kws = ['revenue_', 'cogs_', 'order_count', 'total_quantity',
#                  'total_payment_value', 'total_shipping_fee', 'total_discount',
#                  'total_sessions', 'total_unique_visitors', 'total_page_views',
#                  'new_customers_count', 'return_quantity', 'refund_amount',
#                  'total_reviews', 'total_stock', 'total_units']
#     scale_cols = [c for c in lag_rolling_cols if any(kw in c.lower() for kw in scale_kws)]
#     for col in scale_cols:
#         if col in test_df.columns:
#             test_df[col] = test_df[col] * trend_mult

# Ensure all feature columns exist
for col in feature_cols:
    if col not in test_df.columns:
        test_df[col] = 0

X_test = test_df[feature_cols]
print(f"  Test feature shape: {X_test.shape}")

# ══════════════════════════════════════════════════════════════════════
# 12. PREDICT & SAVE SUBMISSION
# ══════════════════════════════════════════════════════════════════════
revenue_pred = np.maximum(inv_t(model_revenue.predict(X_test)), 0)
cogs_pred    = np.maximum(inv_t(model_cogs.predict(X_test)), 0)

submission = pd.DataFrame({
    'Date': test_dates.strftime('%Y-%m-%d'),
    'Revenue': np.round(revenue_pred, 2),
    'COGS': np.round(cogs_pred, 2),
})

submission.to_csv('data/submission.csv', index=False)

print(f"\nSubmission saved -- {len(submission)} rows")
print(submission.head(10))
print(f"\nRevenue -- mean: {revenue_pred.mean():,.2f}, min: {revenue_pred.min():,.2f}, max: {revenue_pred.max():,.2f}")
print(f"COGS    -- mean: {cogs_pred.mean():,.2f}, min: {cogs_pred.min():,.2f}, max: {cogs_pred.max():,.2f}")
print(f"Margin  -- mean: {(revenue_pred - cogs_pred).mean():,.2f}")

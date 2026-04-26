import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. Load data — keep ALL rows & ALL original features
# ══════════════════════════════════════════════════════════════
df = pd.read_csv('data/revenue_features_v4.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df['year'] = df['date'].dt.year

TARGETS = ['Revenue', 'COGS']
print(f"Data: {df['date'].min().date()} → {df['date'].max().date()}, shape={df.shape}")

# ══════════════════════════════════════════════════════════════
# 2. Add NEW engineered features (on top of existing)
# ══════════════════════════════════════════════════════════════
for t in TARGETS:
    tl = t.lower()
    lag1 = f'{tl}_lag_1d'
    r7  = f'{tl}_rmean_7d'  if f'{tl}_rmean_7d'  in df.columns else None
    r14 = f'{tl}_rmean_14d' if f'{tl}_rmean_14d' in df.columns else None
    r30 = f'{tl}_rmean_30d' if f'{tl}_rmean_30d' in df.columns else None

    # Momentum: short-term vs long-term rolling mean
    if r7 and r30:
        df[f'{tl}_momentum_7_30'] = df[r7] / df[r30].replace(0, np.nan)
    if r7 and r14:
        df[f'{tl}_momentum_7_14'] = df[r7] / df[r14].replace(0, np.nan)

    # Deviation from rolling mean (how unusual is yesterday?)
    if lag1 in df.columns and r7:
        df[f'{tl}_dev_from_7d'] = df[lag1] / df[r7].replace(0, np.nan)
    if lag1 in df.columns and r30:
        df[f'{tl}_dev_from_30d'] = df[lag1] / df[r30].replace(0, np.nan)

    # Lag differences (acceleration)
    if lag1 in df.columns:
        lag7 = f'{tl}_lag_7d' if f'{tl}_lag_7d' in df.columns else None
        lag30 = f'{tl}_lag_30d' if f'{tl}_lag_30d' in df.columns else None
        if lag7:
            df[f'{tl}_diff_1d_7d'] = df[lag1] - df[lag7]
        if lag30:
            df[f'{tl}_diff_1d_30d'] = df[lag1] - df[lag30]

# Revenue-to-COGS ratio (gross margin proxy)
if 'revenue_lag_1d' in df.columns and 'COGS_lag_1d' in df.columns:
    df['rev_cogs_ratio_lag1'] = df['revenue_lag_1d'] / df['COGS_lag_1d'].replace(0, np.nan)

# Day-of-week × month interaction
if 'day_of_week' in df.columns and 'month' in df.columns:
    df['dow_x_month'] = df['day_of_week'] * 12 + df['month']

# Piecewise trend — separate pre/post 2019 behavior
df['years_since_2012'] = df['year'] - 2012
df['trend_post_2019'] = np.maximum(df['year'] - 2019, 0)
df['trend_sq'] = df['years_since_2012'] ** 2  # quadratic to capture the inverted-U

# Rolling volatility ratio
if 'revenue_rstd_7d' in df.columns and 'revenue_rmean_7d' in df.columns:
    df['revenue_cv_7d'] = df['revenue_rstd_7d'] / df['revenue_rmean_7d'].replace(0, np.nan)

# Fill NaN from new features
df = df.fillna(0)

# Feature list — everything except date, year, and targets
feature_cols = [c for c in df.columns if c not in ['date', 'year'] + TARGETS]
X_all = df[feature_cols].copy()
print(f"Total features: {len(feature_cols)}")

# ══════════════════════════════════════════════════════════════
# 3. Sample weights (gentle recency bias)
# ══════════════════════════════════════════════════════════════
# Exponential decay: recent years weighted more, but old data still contributes
years = df['year'].values
max_year = years.max()
decay_rate = 0.15  # gentle decay
sample_weights = np.exp(-decay_rate * (max_year - years))
print(f"Weight range: {sample_weights.min():.3f} → {sample_weights.max():.3f}")

# ══════════════════════════════════════════════════════════════
# 4. Hyperparameters
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# 5. Log-transform helpers
# ══════════════════════════════════════════════════════════════
def log_t(y):   return np.log1p(y)
def inv_t(y):   return np.expm1(y)

# ══════════════════════════════════════════════════════════════
# 6. Time-series cross-validation
# ══════════════════════════════════════════════════════════════
tscv = TimeSeriesSplit(n_splits=5)
print("\n📊 Time-Series Cross-Validation:")
print("=" * 70)

for target in TARGETS:
    y_raw = df[target]
    y_log = log_t(y_raw)
    rmses, maes, r2s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_vl = y_log.iloc[train_idx], y_log.iloc[val_idx]
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
        print(f"  Fold {fold+1}: RMSE={rmse:>14,.2f}  MAE={mae:>14,.2f}  R²={r2:.4f}")

    print(f"\n  {target}:")
    print(f"    RMSE:  {np.mean(rmses):>14,.2f}  (±{np.std(rmses):>12,.2f})")
    print(f"    MAE:   {np.mean(maes):>14,.2f}  (±{np.std(maes):>12,.2f})")
    print(f"    R²:    {np.mean(r2s):>14.4f}  (±{np.std(r2s):>12.4f})")

# ══════════════════════════════════════════════════════════════
# 7. Hold-out validation (2022)
# ══════════════════════════════════════════════════════════════
val_mask = df['year'] == 2022
X_tr_ho = X_all[~val_mask]
X_val_ho = X_all[val_mask]

print("\n" + "=" * 70)
print("📈 Hold-out Validation (2022):")
print("=" * 70)

for target in TARGETS:
    y_raw = df[target]
    y_log = log_t(y_raw)
    w_tr = sample_weights[~val_mask]

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

# ══════════════════════════════════════════════════════════════
# 8. Train final models on ALL data
# ══════════════════════════════════════════════════════════════
print("\n🚀 Training final models on ALL data...")
final_params = {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'}

model_revenue = XGBRegressor(**final_params)
model_revenue.fit(X_all, log_t(df['Revenue']), sample_weight=sample_weights, verbose=500)

model_cogs = XGBRegressor(**final_params)
model_cogs.fit(X_all, log_t(df['COGS']), sample_weight=sample_weights, verbose=500)

# Feature importance
print("\n📊 Top 15 features (Revenue):")
imp = pd.Series(model_revenue.feature_importances_, index=feature_cols).nlargest(15)
for feat, val in imp.items():
    print(f"  {val:.4f}  {feat}")

# ══════════════════════════════════════════════════════════════
# 9. Build test features & predict
# ══════════════════════════════════════════════════════════════
print("\n🔮 Building test features...")
test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
test_df = pd.DataFrame({'date': test_dates})
test_df['month'] = test_df['date'].dt.month
test_df['day_of_month'] = test_df['date'].dt.day

# Seasonal profile from training data
group_keys = ['month', 'day_of_month']
agg_cols = [c for c in feature_cols if c not in group_keys]
seasonal_profile = df.groupby(group_keys)[agg_cols].mean().reset_index()
test_df = test_df.merge(seasonal_profile, on=group_keys, how='left')

# Fill NaN (e.g., Feb 29)
for col in feature_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].interpolate(method='linear').bfill().ffill()
    else:
        test_df[col] = 0

# Update trend features for test period
test_df['year'] = test_df['date'].dt.year
test_df['years_since_2012'] = test_df['year'] - 2012
test_df['trend_post_2019'] = np.maximum(test_df['year'] - 2019, 0)
test_df['trend_sq'] = test_df['years_since_2012'] ** 2
if 'year_trend' in feature_cols:
    test_df['year_trend'] = test_df['year'] - 2012
if 'dow_x_month' in feature_cols and 'day_of_week' in test_df.columns:
    test_df['dow_x_month'] = test_df['day_of_week'] * 12 + test_df['month']

# Trend scaling — use 2019–2022 CAGR (same regime)
annual_rev_recent = df[df['year'] >= 2019].groupby('year')['Revenue'].sum()
if len(annual_rev_recent) >= 2:
    cagr = (annual_rev_recent.iloc[-1] / annual_rev_recent.iloc[0]) ** (1 / (len(annual_rev_recent) - 1))
    years_ahead = test_df['year'] - 2022
    trend_mult = cagr ** years_ahead
    print(f"Trend CAGR (2019–2022): {cagr:.4f}")

    scale_kws = ['revenue_', 'cogs_', 'total_', 'order_count', 'quantity', 'sessions',
                 'visitors', 'views', 'payment_value', 'shipping', 'refund', 'return_',
                 'reviews', 'avg_order_value']
    scale_cols = [c for c in feature_cols if any(kw in c.lower() for kw in scale_kws)]
    for col in scale_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col] * trend_mult

X_test = test_df[feature_cols]
print(f"Test feature shape: {X_test.shape}")

# ══════════════════════════════════════════════════════════════
# 10. Predict & save submission
# ══════════════════════════════════════════════════════════════
revenue_pred = np.maximum(inv_t(model_revenue.predict(X_test)), 0)
cogs_pred    = np.maximum(inv_t(model_cogs.predict(X_test)), 0)

submission = pd.DataFrame({
    'Date': test_dates.strftime('%Y-%m-%d'),
    'Revenue': np.round(revenue_pred, 2),
    'COGS': np.round(cogs_pred, 2),
})

submission.to_csv('data/submission.csv', index=False)

print(f"\n✅ Submission saved — {len(submission)} rows")
print(submission.head(10))
print(f"\nRevenue — mean: {revenue_pred.mean():,.2f}, min: {revenue_pred.min():,.2f}, max: {revenue_pred.max():,.2f}")
print(f"COGS    — mean: {cogs_pred.mean():,.2f}, min: {cogs_pred.min():,.2f}, max: {cogs_pred.max():,.2f}")

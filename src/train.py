import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. Load & prepare data
# ══════════════════════════════════════════════════════════════
df = pd.read_csv('data/revenue_features_full.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"Training data: {df['date'].min().date()} → {df['date'].max().date()}, shape={df.shape}")

TARGETS = ['Revenue', 'COGS']

# ══════════════════════════════════════════════════════════════
# 2. Feature engineering
# ══════════════════════════════════════════════════════════════

def add_date_features(data):
    """Calendar-based features."""
    d = data.copy()
    d['year']            = d['date'].dt.year
    d['month']           = d['date'].dt.month
    d['day']             = d['date'].dt.day
    d['day_of_week']     = d['date'].dt.dayofweek
    d['day_of_year']     = d['date'].dt.dayofyear
    d['week_of_year']    = d['date'].dt.isocalendar().week.astype(int)
    d['quarter']         = d['date'].dt.quarter
    d['is_weekend']      = (d['day_of_week'] >= 5).astype(int)
    d['is_month_start']  = d['date'].dt.is_month_start.astype(int)
    d['is_month_end']    = d['date'].dt.is_month_end.astype(int)
    d['is_quarter_start'] = d['date'].dt.is_quarter_start.astype(int)
    d['is_quarter_end']  = d['date'].dt.is_quarter_end.astype(int)
    d['is_year_start']   = d['date'].dt.is_year_start.astype(int)
    d['is_year_end']     = d['date'].dt.is_year_end.astype(int)
    d['days_in_month']   = d['date'].dt.days_in_month
    # Cyclical encodings
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['dow_sin']   = np.sin(2 * np.pi * d['day_of_week'] / 7)
    d['dow_cos']   = np.cos(2 * np.pi * d['day_of_week'] / 7)
    d['doy_sin']   = np.sin(2 * np.pi * d['day_of_year'] / 365.25)
    d['doy_cos']   = np.cos(2 * np.pi * d['day_of_year'] / 365.25)
    d['woy_sin']   = np.sin(2 * np.pi * d['week_of_year'] / 52)
    d['woy_cos']   = np.cos(2 * np.pi * d['week_of_year'] / 52)
    # Interactions
    d['month_x_dow'] = d['month'] * 10 + d['day_of_week']
    d['quarter_x_dow'] = d['quarter'] * 10 + d['day_of_week']
    return d


def add_lag_features(data, hist_df):
    """
    Add lag features using historical Revenue/COGS data.
    hist_df must have columns: date, Revenue, COGS (sorted by date).
    """
    d = data.copy()

    # Build a lookup: date → Revenue, COGS
    hist_lookup = hist_df.set_index('date')[TARGETS].to_dict('index')

    for target in TARGETS:
        t_lower = target.lower()

        # --- Lag features (same calendar day in previous years) ---
        for years_back in [1, 2, 3]:
            col_name = f'{t_lower}_lag_{years_back}y'
            d[col_name] = d['date'].apply(
                lambda dt: hist_lookup.get(dt - pd.DateOffset(years=years_back), {}).get(target, np.nan)
            )

        # --- Lag features (days back) ---
        for days_back in [364, 371, 728]:  # ~1yr, ~1yr+1wk, ~2yr (week-aligned)
            col_name = f'{t_lower}_lag_{days_back}d'
            d[col_name] = d['date'].apply(
                lambda dt: hist_lookup.get(dt - pd.Timedelta(days=days_back), {}).get(target, np.nan)
            )

    return d


def add_rolling_target_features(data, hist_df):
    """
    Add rolling statistics of Revenue/COGS from historical data.
    For each date, compute stats from the PRIOR historical window.
    """
    d = data.copy()
    hist = hist_df.set_index('date')[TARGETS].sort_index()

    for target in TARGETS:
        t_lower = target.lower()
        series = hist[target]

        # For each row, find the rolling stats from historical data
        # ending the day BEFORE the current date
        for window in [7, 14, 30, 60, 90]:
            mean_col = f'{t_lower}_roll_{window}d_mean'
            std_col  = f'{t_lower}_roll_{window}d_std'

            means = []
            stds  = []
            for dt in d['date']:
                mask = (series.index < dt) & (series.index >= dt - pd.Timedelta(days=window))
                window_data = series[mask]
                means.append(window_data.mean() if len(window_data) > 0 else np.nan)
                stds.append(window_data.std() if len(window_data) > 1 else np.nan)

            d[mean_col] = means
            d[std_col]  = stds

        # Same-month historical averages (all prior years)
        monthly_avg = series.groupby([series.index.month]).agg(['mean', 'std', 'median'])
        d[f'{t_lower}_month_hist_mean']   = d['month'].map(monthly_avg['mean']).values
        d[f'{t_lower}_month_hist_std']    = d['month'].map(monthly_avg['std']).values
        d[f'{t_lower}_month_hist_median'] = d['month'].map(monthly_avg['median']).values

        # Same day-of-week historical averages
        dow_avg = series.groupby(series.index.dayofweek).agg(['mean', 'std'])
        d[f'{t_lower}_dow_hist_mean'] = d['day_of_week'].map(dow_avg['mean']).values
        d[f'{t_lower}_dow_hist_std']  = d['day_of_week'].map(dow_avg['std']).values

        # Month × day_of_week historical average
        month_dow_avg = series.groupby([series.index.month, series.index.dayofweek]).mean()
        d[f'{t_lower}_month_dow_mean'] = d.apply(
            lambda row: month_dow_avg.get((row['month'], row['day_of_week']), np.nan), axis=1
        )

        # Same (month, day) historical average — seasonal profile
        md_avg = series.groupby([series.index.month, series.index.day]).agg(['mean', 'median'])
        d[f'{t_lower}_md_hist_mean']   = d.apply(
            lambda row: md_avg['mean'].get((row['month'], row['day']), np.nan), axis=1
        )
        d[f'{t_lower}_md_hist_median'] = d.apply(
            lambda row: md_avg['median'].get((row['month'], row['day']), np.nan), axis=1
        )

    return d


def add_trend_features(data, hist_df):
    """Year-over-year growth and trend features."""
    d = data.copy()
    annual = hist_df.groupby(hist_df['date'].dt.year)[TARGETS].sum()

    for target in TARGETS:
        t_lower = target.lower()
        yoy = annual[target].pct_change().dropna()
        d[f'{t_lower}_avg_yoy_growth'] = yoy.mean()
        d[f'{t_lower}_recent_yoy_growth'] = yoy.iloc[-1] if len(yoy) > 0 else 0
        # Weighted recent growth (last 3 years weighted more)
        if len(yoy) >= 3:
            weights = np.array([1, 2, 3])
            d[f'{t_lower}_weighted_yoy'] = np.average(yoy.iloc[-3:], weights=weights)
        else:
            d[f'{t_lower}_weighted_yoy'] = yoy.mean()

    return d

# ══════════════════════════════════════════════════════════════
# 3. Build features for training data
# ══════════════════════════════════════════════════════════════
print("Engineering features for training data...")

# Keep original operational features
op_cols = [c for c in df.columns if c not in ['date'] + TARGETS]

df = add_date_features(df)
df = add_lag_features(df, df[['date'] + TARGETS])
df = add_rolling_target_features(df, df[['date'] + TARGETS])
df = add_trend_features(df, df[['date'] + TARGETS])

# Fill NaN from lag/rolling features
df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

# ══════════════════════════════════════════════════════════════
# 4. Time-series cross-validation to evaluate & tune
# ══════════════════════════════════════════════════════════════
feature_cols = [c for c in df.columns if c not in ['date'] + TARGETS]
X_all = df[feature_cols]
print(f"Total features: {len(feature_cols)}")

best_params = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "early_stopping_rounds": 100,
}

# Evaluate with time-series CV (train on older data, validate on newer)
tscv = TimeSeriesSplit(n_splits=4)
print("\n📊 Time-Series Cross-Validation:")
print("=" * 70)

for target in TARGETS:
    y = df[target]
    rmses, maes, r2s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**best_params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=0)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae  = mean_absolute_error(y_val, preds)
        r2   = r2_score(y_val, preds)
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

    print(f"\n  {target}:")
    print(f"    RMSE:  {np.mean(rmses):>14,.2f}  (±{np.std(rmses):>12,.2f})")
    print(f"    MAE:   {np.mean(maes):>14,.2f}  (±{np.std(maes):>12,.2f})")
    print(f"    R²:    {np.mean(r2s):>14.4f}  (±{np.std(r2s):>12.4f})")

# ══════════════════════════════════════════════════════════════
# 5. Final validation on last year (2022) for reporting
# ══════════════════════════════════════════════════════════════
val_mask = df['year'] == 2022
X_tr_final = X_all[~val_mask]
X_val_final = X_all[val_mask]

print("\n" + "=" * 70)
print("📈 Final Hold-out Validation (2022):")
print("=" * 70)

for target in TARGETS:
    y_tr = df.loc[~val_mask, target]
    y_val = df.loc[val_mask, target]

    model = XGBRegressor(**best_params)
    model.fit(X_tr_final, y_tr,
              eval_set=[(X_val_final, y_val)],
              verbose=0)

    preds = model.predict(X_val_final)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)
    r2   = r2_score(y_val, preds)

    print(f"\n  {target}:")
    print(f"    RMSE:  {rmse:>14,.2f}")
    print(f"    MAE:   {mae:>14,.2f}")
    print(f"    R²:    {r2:>14.4f}")

# ══════════════════════════════════════════════════════════════
# 6. Train final models on ALL data
# ══════════════════════════════════════════════════════════════
print("\n🚀 Training final models on ALL data...")

# Remove early stopping for final training (use all data, no eval set)
final_params = {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'}

model_revenue = XGBRegressor(**final_params)
model_revenue.fit(X_all, df['Revenue'], verbose=200)

model_cogs = XGBRegressor(**final_params)
model_cogs.fit(X_all, df['COGS'], verbose=200)

# ══════════════════════════════════════════════════════════════
# 7. Build test features & predict
# ══════════════════════════════════════════════════════════════
print("\n🔮 Building test features...")

test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
test_df = pd.DataFrame({'date': test_dates})
test_df = add_date_features(test_df)

# --- Operational features: use seasonal (month, day) averages from training ---
seasonal_profile = df.groupby(['month', 'day'])[op_cols].mean().reset_index()
test_df = test_df.merge(seasonal_profile, on=['month', 'day'], how='left')
for col in op_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].interpolate(method='linear').bfill().ffill()
    else:
        test_df[col] = 0

# --- Apply trend scaling to volume features ---
annual_rev = df.groupby('year')['Revenue'].sum()
full_years = annual_rev.loc[2013:2022]
growth_rev = (full_years.iloc[-1] / full_years.iloc[0]) ** (1 / (len(full_years) - 1))

annual_cogs_s = df.groupby('year')['COGS'].sum()
full_cogs = annual_cogs_s.loc[2013:2022]
growth_cogs = (full_cogs.iloc[-1] / full_cogs.iloc[0]) ** (1 / (len(full_cogs) - 1))

years_ahead = test_df['year'] - 2022
trend_mult = growth_rev ** years_ahead

volume_kws = ['total', 'count', 'order', 'quantity', 'sessions', 'visitors',
              'views', 'payment_value', 'shipping', 'stock', 'units',
              'customers', 'refund', 'return_quantity', 'reviews']
volume_features = [c for c in op_cols if any(kw in c.lower() for kw in volume_kws)]
for col in volume_features:
    if col in test_df.columns:
        test_df[col] = test_df[col] * trend_mult

# --- Lag & rolling features from historical data ---
hist_data = df[['date'] + TARGETS].copy()
test_df = add_lag_features(test_df, hist_data)
test_df = add_rolling_target_features(test_df, hist_data)
test_df = add_trend_features(test_df, hist_data)

# Fill NaN
test_df = test_df.fillna(method='bfill').fillna(method='ffill').fillna(0)

# Ensure column alignment
X_test = test_df[feature_cols]

print(f"Test feature shape: {X_test.shape}")

# ══════════════════════════════════════════════════════════════
# 8. Predict & save
# ══════════════════════════════════════════════════════════════
revenue_pred = model_revenue.predict(X_test)
cogs_pred    = model_cogs.predict(X_test)

# Ensure non-negative
revenue_pred = np.maximum(revenue_pred, 0)
cogs_pred    = np.maximum(cogs_pred, 0)

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

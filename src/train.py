import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/revenue_features_v4.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Training data: {df['date'].min().date()} → {df['date'].max().date()}, shape={df.shape}")

TARGETS = ['Revenue', 'COGS']

# Separate features from targets and date
feature_cols = [c for c in df.columns if c not in ['date'] + TARGETS]
X_all = df[feature_cols]
print(f"Total features: {len(feature_cols)}")

best_params = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
}

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
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae  = mean_absolute_error(y_val, preds)
        r2   = r2_score(y_val, preds)
        rmses.append(rmse); maes.append(mae); r2s.append(r2)

    print(f"\n  {target}:")
    print(f"    RMSE:  {np.mean(rmses):>14,.2f}  (±{np.std(rmses):>12,.2f})")
    print(f"    MAE:   {np.mean(maes):>14,.2f}  (±{np.std(maes):>12,.2f})")
    print(f"    R²:    {np.mean(r2s):>14.4f}  (±{np.std(r2s):>12.4f})")

df['year'] = df['date'].dt.year
val_mask = df['year'] == 2022
X_tr_ho = X_all[~val_mask]
X_val_ho = X_all[val_mask]

print("\n" + "=" * 70)
print("📈 Hold-out Validation (2022):")
print("=" * 70)

for target in TARGETS:
    y_tr_ho = df.loc[~val_mask, target]
    y_val_ho = df.loc[val_mask, target]

    model_ho = XGBRegressor(**best_params)
    model_ho.fit(X_tr_ho, y_tr_ho, eval_set=[(X_val_ho, y_val_ho)], verbose=0)

    preds_ho = model_ho.predict(X_val_ho)
    rmse = np.sqrt(mean_squared_error(y_val_ho, preds_ho))
    mae  = mean_absolute_error(y_val_ho, preds_ho)
    r2   = r2_score(y_val_ho, preds_ho)
    print(f"\n  {target}:")
    print(f"    RMSE:  {rmse:>14,.2f}")
    print(f"    MAE:   {mae:>14,.2f}")
    print(f"    R²:    {r2:>14.4f}")

print("\n🚀 Training final models on ALL data...")

final_params = {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'}

model_revenue = XGBRegressor(**final_params)
model_revenue.fit(X_all, df['Revenue'], verbose=200)

model_cogs = XGBRegressor(**final_params)
model_cogs.fit(X_all, df['COGS'], verbose=200)

print("\n🔮 Building test features...")

test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
test_df = pd.DataFrame({'date': test_dates})

group_keys = ['month', 'day_of_month']
agg_cols = [c for c in feature_cols if c not in group_keys]

test_df['month'] = test_df['date'].dt.month
test_df['day_of_month'] = test_df['date'].dt.day

seasonal_profile = df.groupby(group_keys)[agg_cols].mean().reset_index()
test_df = test_df.merge(seasonal_profile, on=group_keys, how='left')

for col in feature_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].interpolate(method='linear').bfill().ffill()
    else:
        test_df[col] = 0

annual_rev = df.groupby('year')['Revenue'].sum()
full_years = annual_rev.loc[2013:2022]
if len(full_years) >= 2:
    cagr = (full_years.iloc[-1] / full_years.iloc[0]) ** (1 / (len(full_years) - 1))
    years_ahead = test_df['date'].dt.year - 2022
    trend_mult = cagr ** years_ahead

    scale_kws = ['revenue_', 'cogs_', 'total_', 'order_count', 'quantity', 'sessions',
                 'visitors', 'views', 'payment_value', 'shipping', 'refund', 'return_',
                 'reviews', 'avg_order_value']
    scale_cols = [c for c in feature_cols if any(kw in c.lower() for kw in scale_kws)]
    for col in scale_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col] * trend_mult

X_test = test_df[feature_cols]
print(f"Test feature shape: {X_test.shape}")

revenue_pred = model_revenue.predict(X_test)
cogs_pred    = model_cogs.predict(X_test)

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

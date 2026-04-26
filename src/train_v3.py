import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings, time
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGETS = ['Revenue', 'COGS']
def log_t(y): return np.log1p(y)
def inv_t(y): return np.expm1(np.clip(y, -20, 20))

# ====================================================================
# 1. LOAD & CLEAN
# ====================================================================
print("=" * 70); print("STEP 1: Load & clean"); print("=" * 70)
df = pd.read_csv('data/revenue_features_full.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"Shape: {df.shape}")

WEAK = [
    'items_reorder','unique_sizes','active_promos_count','promo_type_fixed',
    'promo_categories_covered','promo_type_percentage','promo_channel_in_store',
    'promo_channel_social_media','promo_channel_email','has_stackable',
    'avg_min_order_value','traffic_sessions_social_media','traffic_sessions_paid_search',
    'traffic_sessions_organic_search','traffic_sessions_email_campaign',
    'traffic_sessions_referral','traffic_sessions_direct','items_overstock',
    'avg_installments','avg_fill_rate','avg_bounce_rate','avg_session_duration_sec',
    'avg_delivery_days','avg_rating',
]
# Multicollinear with order_count (corr>0.999)
MULTICOLL = ['unique_customers', 'unique_zips']
to_drop = [f for f in WEAK + MULTICOLL if f in df.columns]
df = df.drop(columns=to_drop)
print(f"  Dropped {len(to_drop)} weak/multicollinear features -> {df.shape}")

# ====================================================================
# 2. FEATURE ENGINEERING
# ====================================================================
print("\n" + "=" * 70); print("STEP 2: Feature engineering"); print("=" * 70)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_x_month'] = df['day_of_week'] * 12 + df['month']
df['year_trend'] = df['year'] - 2012
df['trend_sq'] = (df['year'] - 2012) ** 2

# Target lags
for t in TARGETS:
    for lag in [1, 2, 3, 7, 14, 30, 365]:
        df[f'{t.lower()}_lag_{lag}d'] = df[t].shift(lag)

# Operational lags
OP_COLS = [c for c in ['order_count','total_quantity','total_payment_value',
    'total_shipping_fee','total_discount','total_sessions','total_unique_visitors',
    'total_page_views','new_customers_count','return_quantity','refund_amount',
    'total_reviews','total_stock_on_hand','total_units_received'] if c in df.columns]
for c in OP_COLS:
    df[f'{c}_lag_1d'] = df[c].shift(1)
    df[f'{c}_lag_7d'] = df[c].shift(7)

# Rolling stats on target lags
for t in TARGETS:
    lag1 = f'{t.lower()}_lag_1d'
    for w in [7, 14, 30]:
        df[f'{t.lower()}_rmean_{w}d'] = df[lag1].rolling(w, min_periods=1).mean()
        df[f'{t.lower()}_rstd_{w}d'] = df[lag1].rolling(w, min_periods=1).std()
    df[f'{t.lower()}_exp_mean'] = df[lag1].expanding(min_periods=1).mean()

for c in ['order_count','total_quantity','total_payment_value']:
    lag1 = f'{c}_lag_1d'
    if lag1 in df.columns:
        for w in [7, 30]:
            df[f'{c}_rmean_{w}d'] = df[lag1].rolling(w, min_periods=1).mean()

# Momentum & deviation
for t in TARGETS:
    tl = t.lower()
    l1, r7, r14, r30 = f'{tl}_lag_1d', f'{tl}_rmean_7d', f'{tl}_rmean_14d', f'{tl}_rmean_30d'
    if r7 in df.columns and r30 in df.columns:
        df[f'{tl}_momentum_7_30'] = df[r7] / df[r30].replace(0, np.nan)
    if r7 in df.columns and r14 in df.columns:
        df[f'{tl}_momentum_7_14'] = df[r7] / df[r14].replace(0, np.nan)
    if l1 in df.columns and r7 in df.columns:
        df[f'{tl}_dev_from_7d'] = df[l1] / df[r7].replace(0, np.nan)
    if l1 in df.columns and r30 in df.columns:
        df[f'{tl}_dev_from_30d'] = df[l1] / df[r30].replace(0, np.nan)
    l7, l30 = f'{tl}_lag_7d', f'{tl}_lag_30d'
    if l1 in df.columns and l7 in df.columns:
        df[f'{tl}_diff_1d_7d'] = df[l1] - df[l7]
    if l1 in df.columns and l30 in df.columns:
        df[f'{tl}_diff_1d_30d'] = df[l1] - df[l30]

if 'revenue_lag_1d' in df.columns and 'cogs_lag_1d' in df.columns:
    df['rev_cogs_ratio_lag1'] = df['revenue_lag_1d'] / df['cogs_lag_1d'].replace(0, np.nan)
if 'revenue_rstd_7d' in df.columns and 'revenue_rmean_7d' in df.columns:
    df['revenue_cv_7d'] = df['revenue_rstd_7d'] / df['revenue_rmean_7d'].replace(0, np.nan)

df = df.fillna(0)

# ====================================================================
# 3. FEATURE SELECTION
# ====================================================================
RAW_SAME_DAY = OP_COLS + ['avg_unit_price','unique_items','avg_product_price',
    'avg_product_cogs','unique_colors']
BREAKDOWN_PREFIXES = ['orders_payment_','orders_device_','orders_source_',
    'orders_status_','orders_region_','returns_reason_','new_cust_acq_',
    'new_cust_age_','new_cust_gen_','qty_category_','qty_segment_',
    'promo_channel_','max_discount_value']
SAME_DAY_BD = [c for c in df.columns if any(c.startswith(p) for p in BREAKDOWN_PREFIXES)]
EXCLUDE = set(['date','year'] + TARGETS + RAW_SAME_DAY + SAME_DAY_BD)
feature_cols = [c for c in df.columns if c not in EXCLUDE]

# Drop features with |corr| < 0.05
cr = df[feature_cols].corrwith(df['Revenue']).abs()
cc = df[feature_cols].corrwith(df['COGS']).abs()
mc = pd.concat([cr, cc], axis=1).max(axis=1)
weak_eng = mc[mc < 0.05].index.tolist()
feature_cols = [c for c in feature_cols if c not in weak_eng]
print(f"  Features: {len(feature_cols)} (dropped {len(weak_eng)} weak engineered)")

X_all = df[feature_cols].copy()
years = df['year'].values
sample_weights = np.exp(-0.15 * (years.max() - years))

df.to_csv('data/revenue_features_v3_eng.csv', index=False)
print("Data saved to data/revenue_features_v3_eng.csv")

# ====================================================================
# 4. OPTUNA HYPERPARAMETER TUNING
# ====================================================================
print("\n" + "=" * 70); print("STEP 3: Optuna hyperparameter tuning"); print("=" * 70)
t0 = time.time()

def objective(trial):
    params = {
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 30),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
        'objective': 'reg:squarederror',
        'tree_method': 'hist', 'device': 'cuda',
        'early_stopping_rounds': 100, 'random_state': 42,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for target in TARGETS:
        y_log = log_t(df[target])
        for tr_idx, vl_idx in tscv.split(X_all):
            m = XGBRegressor(**params)
            m.fit(X_all.iloc[tr_idx], y_log.iloc[tr_idx],
                  eval_set=[(X_all.iloc[vl_idx], y_log.iloc[vl_idx])],
                  sample_weight=sample_weights[tr_idx], verbose=0)
            preds = inv_t(m.predict(X_all.iloc[vl_idx]))
            scores.append(np.sqrt(mean_squared_error(df[target].iloc[vl_idx], preds)))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective, n_trials=30, show_progress_bar=True)
print(f"  XGB best RMSE: {study_xgb.best_value:,.2f} ({time.time()-t0:.0f}s)")

# Build best params
bp_xgb = {
    'n_estimators': 5000, 'objective': 'reg:squarederror',
    'tree_method': 'hist', 'device': 'cuda', 'random_state': 42,
    'early_stopping_rounds': 150,
    'learning_rate': study_xgb.best_params['lr'],
    'max_depth': study_xgb.best_params['max_depth'],
    'subsample': study_xgb.best_params['subsample'],
    'colsample_bytree': study_xgb.best_params['colsample_bytree'],
    'min_child_weight': study_xgb.best_params['min_child_weight'],
    'gamma': study_xgb.best_params['gamma'],
    'reg_alpha': study_xgb.best_params['reg_alpha'],
    'reg_lambda': study_xgb.best_params['reg_lambda'],
}
print(f"\n  XGB best params: {bp_xgb}")

# ====================================================================
# 5. CV EVALUATION
# ====================================================================
print("\n" + "=" * 70); print("STEP 4: 5-Fold Time-Series CV"); print("=" * 70)
tscv5 = TimeSeriesSplit(n_splits=5)



# for target in TARGETS:
#     y_raw, y_log = df[target], log_t(df[target])
#     rmses, maes, r2s = [], [], []
#     for fold, (tr, vl) in enumerate(tscv5.split(X_all)):
#         mx = XGBRegressor(**bp_xgb)
#         mx.fit(X_all.iloc[tr], y_log.iloc[tr], eval_set=[(X_all.iloc[vl], y_log.iloc[vl])],
#                sample_weight=sample_weights[tr], verbose=0)
#         preds = np.maximum(inv_t(mx.predict(X_all.iloc[vl])), 0)
#         actual = y_raw.iloc[vl]
#         rmse = np.sqrt(mean_squared_error(actual, preds))
#         mae = mean_absolute_error(actual, preds)
#         r2 = r2_score(actual, preds)
#         rmses.append(rmse); maes.append(mae); r2s.append(r2)
#         print(f"  [{target}] Fold {fold+1}: RMSE={rmse:>14,.2f}  MAE={mae:>14,.2f}  R2={r2:.4f}")
#     print(f"  {target} Avg: RMSE={np.mean(rmses):>14,.2f} MAE={np.mean(maes):>14,.2f} R2={np.mean(r2s):.4f}\n")

# # ====================================================================
# # 6. TRAIN FINAL MODELS
# # ====================================================================
# print("=" * 70); print("STEP 5: Train final models"); print("=" * 70)
# fp_xgb = {k:v for k,v in bp_xgb.items() if k != 'early_stopping_rounds'}
# models = {}
# for target in TARGETS:
#     y_log = log_t(df[target])
#     mx = XGBRegressor(**fp_xgb)
#     mx.fit(X_all, y_log, sample_weight=sample_weights, verbose=500)
#     models[target] = mx
#     print(f"  {target}: XGB trained")

# # ====================================================================
# # 7. BUILD TEST FEATURES & PREDICT (seasonal profile, recent years)
# # ====================================================================
# print("\n" + "=" * 70); print("STEP 6: Build test features & predict"); print("=" * 70)

# test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
# test_df = pd.DataFrame({'date': test_dates})

# # Calendar features
# test_df['month'] = test_df['date'].dt.month
# test_df['day_of_month'] = test_df['date'].dt.day
# test_df['day_of_week'] = test_df['date'].dt.dayofweek
# test_df['day_of_year'] = test_df['date'].dt.dayofyear
# test_df['week_of_year'] = test_df['date'].dt.isocalendar().week.astype(int)
# test_df['is_weekend'] = (test_df['day_of_week'] >= 5).astype(int)
# test_df['month_sin'] = np.sin(2 * np.pi * test_df['month'] / 12)
# test_df['month_cos'] = np.cos(2 * np.pi * test_df['month'] / 12)
# test_df['dow_sin'] = np.sin(2 * np.pi * test_df['day_of_week'] / 7)
# test_df['dow_x_month'] = test_df['day_of_week'] * 12 + test_df['month']

# # Trend features
# test_df['year'] = test_df['date'].dt.year
# test_df['year_trend'] = test_df['year'] - 2012
# test_df['trend_sq'] = (test_df['year'] - 2012) ** 2

# # Seasonal profile from ALL years (more stable averages)
# group_keys = ['month', 'day_of_month']
# lag_rolling_cols = [c for c in feature_cols if c not in
#     group_keys + ['day_of_week', 'day_of_year', 'week_of_year', 'is_weekend',
#                   'month_sin', 'month_cos', 'dow_sin', 'dow_x_month',
#                   'year_trend', 'trend_sq', 'year']]

# seasonal_profile = df.groupby(group_keys)[lag_rolling_cols].mean().reset_index()
# test_df = test_df.merge(seasonal_profile, on=group_keys, how='left')

# # Fill NaN (e.g., Feb 29)
# for col in lag_rolling_cols:
#     if col in test_df.columns:
#         test_df[col] = test_df[col].interpolate(method='linear').bfill().ffill()
#     else:
#         test_df[col] = 0

# # Ensure all feature columns exist
# for col in feature_cols:
#     if col not in test_df.columns:
#         test_df[col] = 0

# X_test = test_df[feature_cols]
# print(f"  Test feature shape: {X_test.shape}")

# # XGB-only prediction
# rev_preds = np.maximum(inv_t(models['Revenue'].predict(X_test)), 0)
# cogs_preds = np.maximum(inv_t(models['COGS'].predict(X_test)), 0)

# # ====================================================================
# # 8. SAVE SUBMISSION
# # ====================================================================
# submission = pd.DataFrame({
#     'Date': test_dates.strftime('%Y-%m-%d'),
#     'Revenue': np.round(rev_preds, 2),
#     'COGS': np.round(cogs_preds, 2),
# })
# submission.to_csv('data/submission.csv', index=False)

# print(f"\nSubmission saved -- {len(submission)} rows")
# print(submission.head(10))
# print(f"\nRevenue -- mean: {rev_preds.mean():,.2f}, min: {rev_preds.min():,.2f}, max: {rev_preds.max():,.2f}")
# print(f"COGS    -- mean: {cogs_preds.mean():,.2f}, min: {cogs_preds.min():,.2f}, max: {cogs_preds.max():,.2f}")
# print(f"Margin  -- mean: {(rev_preds - cogs_preds).mean():,.2f}")
# print(f"\nTotal time: {time.time()-t0:.0f}s")

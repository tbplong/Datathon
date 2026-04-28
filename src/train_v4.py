"""train_v4.py - Horizon-Specific Ensembling for Revenue & COGS Forecasting
Model A: Short-term  (Days 1-14)  - Recursive with all lag features
Model B: Medium-term (Days 15-30) - Direct, lag_30d+ only
Model C: Long-term   (Day 31+)   - Direct, calendar/promo only
"""
import pandas as pd, numpy as np, time, gc, warnings
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGETS = ['Revenue', 'COGS']
TEST_START, TEST_END = '2023-01-01', '2024-07-01'
HORIZON_A_END, HORIZON_B_END = '2023-01-14', '2023-01-30'

def log_t(y): return np.log1p(y)
def inv_t(y): return np.expm1(np.clip(y, -20, 20))

# ── Feature classification helpers ──────────────────────────────────
SHORT_LAG_PATS = ['_lag_1d','_lag_2d','_lag_3d','_lag_7d','_lag_14d',
                  '_rmean_7d','_rstd_7d','_rmean_14d','_rstd_14d']
SHORT_DERIVED = ['momentum_7_14','momentum_7_30','dev_from_7d','dev_from_30d',
                 'diff_1d_7d','diff_1d_30d','rev_cogs_ratio_lag1','revenue_cv_7d']
MEDIUM_LAG_PATS = ['_lag_30d','_lag_365d','_rmean_30d','_rstd_30d','_exp_mean']

PRUNE_SET = {
    'items_reorder','unique_sizes','promo_type_fixed','is_weekend',
    'refund_amount_lag_1d','cogs_momentum_7_14','cogs_lag_14d',
    'cogs_lag_3d','revenue_diff_1d_30d','revenue_momentum_7_14',
    'cogs_rmean_7d','items_stockout','dow_sin','return_quantity_lag_1d',
    'cogs_rmean_14d','revenue_diff_1d_7d','revenue_lag_30d',
}

def is_short_lag(f):
    return any(f.endswith(s) for s in SHORT_LAG_PATS) or any(p in f for p in SHORT_DERIVED)

def is_medium_lag(f):
    return any(f.endswith(s) for s in MEDIUM_LAG_PATS)

def classify_features(feature_cols):
    """Split features into short-lag, medium-lag, and calendar/promo sets."""
    short, medium, calendar = [], [], []
    for f in feature_cols:
        if is_short_lag(f):   short.append(f)
        elif is_medium_lag(f): medium.append(f)
        else:                  calendar.append(f)
    return short, medium, calendar

# ====================================================================
# 1. LOAD & FEATURE ENGINEERING
# ====================================================================
print("=" * 70); print("STEP 1: Load & engineer features"); print("=" * 70)
t0 = time.time()
df = pd.read_csv('data/revenue_features_full.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"  Raw shape: {df.shape}")

# Calendar
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week']  = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['day_of_year']  = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin']      = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_x_month']  = df['day_of_week'] * 12 + df['month']
df['year_trend']   = df['year'] - 2012
df['trend_sq']     = (df['year'] - 2012) ** 2

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
        df[f'{t.lower()}_rstd_{w}d']  = df[lag1].rolling(w, min_periods=1).std()
    df[f'{t.lower()}_exp_mean'] = df[lag1].expanding(min_periods=1).mean()

for c in ['order_count','total_quantity','total_payment_value']:
    lag1 = f'{c}_lag_1d'
    if lag1 in df.columns:
        for w in [7, 30]:
            df[f'{c}_rmean_{w}d'] = df[lag1].rolling(w, min_periods=1).mean()

# Momentum & deviation
for t in TARGETS:
    tl = t.lower()
    l1  = f'{tl}_lag_1d'
    r7, r14, r30 = f'{tl}_rmean_7d', f'{tl}_rmean_14d', f'{tl}_rmean_30d'
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
# 2. FEATURE SELECTION & HORIZON SPLITTING
# ====================================================================
print("\n" + "=" * 70); print("STEP 2: Feature selection & horizon split"); print("=" * 70)

RAW_SAME_DAY = OP_COLS + ['avg_unit_price','unique_items','avg_product_price',
    'avg_product_cogs','unique_colors']
BD_PREFIXES = ['orders_payment_','orders_device_','orders_source_','orders_status_',
    'orders_region_','returns_reason_','new_cust_acq_','new_cust_age_',
    'new_cust_gen_','qty_category_','qty_segment_','promo_channel_','max_discount_value']
SAME_DAY_BD = [c for c in df.columns if any(c.startswith(p) for p in BD_PREFIXES)]
EXCLUDE = set(['date','year'] + TARGETS + RAW_SAME_DAY + SAME_DAY_BD)
all_feats = [c for c in df.columns if c not in EXCLUDE]

# Correlation filter
cr = df[all_feats].corrwith(df['Revenue']).abs()
cc = df[all_feats].corrwith(df['COGS']).abs()
mc = pd.concat([cr, cc], axis=1).max(axis=1)
all_feats = [c for c in all_feats if mc.get(c, 0) >= 0.05]
all_feats = [c for c in all_feats if c not in PRUNE_SET]

# Classify
short_feats, medium_feats, calendar_feats = classify_features(all_feats)
feats_A = all_feats                          # all features
feats_B = medium_feats + calendar_feats      # no short lags
feats_C = calendar_feats                     # calendar/promo only
print(f"  Model A features: {len(feats_A)}")
print(f"  Model B features: {len(feats_B)} (dropped {len(short_feats)} short-lag)")
print(f"  Model C features: {len(feats_C)} (calendar/promo only)")

years = df['year'].values
sample_weights = np.exp(-0.15 * (years.max() - years))

# ====================================================================
# 3. OPTUNA TUNING (per-horizon)
# ====================================================================
print("\n" + "=" * 70); print("STEP 3: Optuna tuning per horizon"); print("=" * 70)

def make_objective(X, feats, df_ref, sw):
    """Factory: returns an Optuna objective for the given feature set."""
    X_sub = X[feats]
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
            y_log = log_t(df_ref[target])
            for tr, vl in tscv.split(X_sub):
                m = XGBRegressor(**params)
                m.fit(X_sub.iloc[tr], y_log.iloc[tr],
                      eval_set=[(X_sub.iloc[vl], y_log.iloc[vl])],
                      sample_weight=sw[tr], verbose=0)
                preds = inv_t(m.predict(X_sub.iloc[vl]))
                scores.append(np.sqrt(mean_squared_error(df_ref[target].iloc[vl], preds)))
        return np.mean(scores)
    return objective

def run_optuna(label, X, feats, df_ref, sw, n_trials=20):
    print(f"\n  Tuning {label} ({len(feats)} features, {n_trials} trials)...")
    t = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(make_objective(X, feats, df_ref, sw), n_trials=n_trials, show_progress_bar=True)
    bp = study.best_params
    best = {
        'n_estimators': 5000, 'objective': 'reg:squarederror',
        'tree_method': 'hist', 'device': 'cuda', 'random_state': 42,
        'early_stopping_rounds': 150,
        'learning_rate': bp['lr'], 'max_depth': bp['max_depth'],
        'subsample': bp['subsample'], 'colsample_bytree': bp['colsample_bytree'],
        'min_child_weight': bp['min_child_weight'], 'gamma': bp['gamma'],
        'reg_alpha': bp['reg_alpha'], 'reg_lambda': bp['reg_lambda'],
    }
    print(f"    {label} best RMSE: {study.best_value:,.2f} ({time.time()-t:.0f}s)")
    return best

X_all = df[all_feats].copy()
params_A = run_optuna("Model A", X_all, feats_A, df, sample_weights, n_trials=20)
params_B = run_optuna("Model B", X_all, feats_B, df, sample_weights, n_trials=20)
params_C = run_optuna("Model C", X_all, feats_C, df, sample_weights, n_trials=15)

# ====================================================================
# 4. TRAIN FINAL MODELS
# ====================================================================
print("\n" + "=" * 70); print("STEP 4: Train final models"); print("=" * 70)

def train_models(label, feats, params):
    """Train Revenue + COGS models for a given feature set."""
    fp = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
    models = {}
    for target in TARGETS:
        y_log = log_t(df[target])
        m = XGBRegressor(**fp)
        m.fit(df[feats], y_log, sample_weight=sample_weights, verbose=0)
        models[target] = m
        print(f"  {label} [{target}] trained ({len(feats)} features)")
    return models

models_A = train_models("Model A", feats_A, params_A)
models_B = train_models("Model B", feats_B, params_B)
models_C = train_models("Model C", feats_C, params_C)
gc.collect()

# ====================================================================
# 5. INFERENCE
# ====================================================================
print("\n" + "=" * 70); print("STEP 5: Inference"); print("=" * 70)

test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq='D')

# ── Helper: build calendar features for a date ─────────────────────
def make_calendar_row(d):
    """Return a dict of calendar features for a single date."""
    m = d.month
    dow = d.dayofweek
    return {
        'month': m, 'day_of_week': dow, 'day_of_month': d.day,
        'day_of_year': d.dayofyear,
        'week_of_year': d.isocalendar()[1],
        'is_weekend': int(dow >= 5),
        'month_sin': np.sin(2 * np.pi * m / 12),
        'month_cos': np.cos(2 * np.pi * m / 12),
        'dow_sin': np.sin(2 * np.pi * dow / 7),
        'dow_x_month': dow * 12 + m,
        'year_trend': d.year - 2012,
        'trend_sq': (d.year - 2012) ** 2,
    }

# ── Seasonal profile for filling operational lags ───────────────────
seasonal_cols = [c for c in all_feats if c not in ['month', 'day_of_month']]
seasonal = df.groupby(['month', 'day_of_month'])[seasonal_cols].mean().reset_index()

# ── 5a. MODEL A: Recursive short-term (days 1-14) ──────────────────
print("\n  [Model A] Recursive prediction (days 1-14)...")

# Build history buffers from last 400 days of training data
# to look up lag values during recursive prediction
train_end_idx = len(df) - 1
hist = {}
for t in TARGETS:
    hist[t.lower()] = list(df[t].values)  # full history

# Also store operational column histories for lag lookups
op_hist = {}
for c in OP_COLS:
    op_hist[c] = list(df[c].values)

dates_A = pd.date_range(TEST_START, HORIZON_A_END, freq='D')
preds_A = {t: [] for t in TARGETS}

for d in dates_A:
    row = make_calendar_row(d)
    n = len(hist['revenue'])  # current length = train_len + predictions so far

    # Target lag features (using actual history + predictions)
    for t in TARGETS:
        tl = t.lower()
        h = hist[tl]
        for lag in [1, 2, 3, 7, 14, 30, 365]:
            idx = n - lag
            row[f'{tl}_lag_{lag}d'] = h[idx] if idx >= 0 else 0

        # Rolling stats from the lag_1d series
        for w in [7, 14, 30]:
            vals = [h[n - 1 - i] for i in range(min(w, n))]
            row[f'{tl}_rmean_{w}d'] = np.mean(vals) if vals else 0
            row[f'{tl}_rstd_{w}d']  = np.std(vals, ddof=1) if len(vals) > 1 else 0
        row[f'{tl}_exp_mean'] = np.mean(h) if h else 0

    # Operational lag features (use last known actuals — no leakage)
    for c in OP_COLS:
        oh = op_hist[c]
        # lag_1d: last actual = last training value (no future ops data)
        row[f'{c}_lag_1d'] = oh[-1] if oh else 0
        row[f'{c}_lag_7d'] = oh[-7] if len(oh) >= 7 else (oh[0] if oh else 0)

    # Operational rolling features
    for c in ['order_count','total_quantity','total_payment_value']:
        if f'{c}_lag_1d' in row:
            for w in [7, 30]:
                vals = op_hist.get(c, [])
                vals_w = vals[-w:] if len(vals) >= w else vals
                row[f'{c}_rmean_{w}d'] = np.mean(vals_w) if vals_w else 0

    # Momentum & deviation features
    for t in TARGETS:
        tl = t.lower()
        l1 = row.get(f'{tl}_lag_1d', 0)
        r7 = row.get(f'{tl}_rmean_7d', 0)
        r14 = row.get(f'{tl}_rmean_14d', 0)
        r30 = row.get(f'{tl}_rmean_30d', 0)
        row[f'{tl}_momentum_7_30'] = r7 / r30 if r30 != 0 else 0
        row[f'{tl}_momentum_7_14'] = r7 / r14 if r14 != 0 else 0
        row[f'{tl}_dev_from_7d']   = l1 / r7 if r7 != 0 else 0
        row[f'{tl}_dev_from_30d']  = l1 / r30 if r30 != 0 else 0
        l7 = row.get(f'{tl}_lag_7d', 0)
        l30 = row.get(f'{tl}_lag_30d', 0)
        row[f'{tl}_diff_1d_7d']  = l1 - l7
        row[f'{tl}_diff_1d_30d'] = l1 - l30

    if row.get('cogs_lag_1d', 0) != 0:
        row['rev_cogs_ratio_lag1'] = row.get('revenue_lag_1d', 0) / row['cogs_lag_1d']
    else:
        row['rev_cogs_ratio_lag1'] = 0
    r7v = row.get('revenue_rstd_7d', 0)
    m7v = row.get('revenue_rmean_7d', 0)
    row['revenue_cv_7d'] = r7v / m7v if m7v != 0 else 0

    # Fill any remaining features from seasonal profile
    md, dd = d.month, d.day
    sp = seasonal[(seasonal['month'] == md) & (seasonal['day_of_month'] == dd)]
    for f in feats_A:
        if f not in row:
            row[f] = sp[f].values[0] if (len(sp) > 0 and f in sp.columns) else 0

    # Predict
    x = pd.DataFrame([{f: row.get(f, 0) for f in feats_A}])
    for t in TARGETS:
        pred = max(inv_t(models_A[t].predict(x)[0]), 0)
        preds_A[t].append(pred)
        hist[t.lower()].append(pred)  # append for next day's lags

print(f"    Predicted {len(dates_A)} days")

# ── 5b. MODEL B: Direct medium-term (days 15-30) ───────────────────
print("  [Model B] Direct prediction (days 15-30)...")

dates_B = pd.date_range(pd.Timestamp(HORIZON_A_END) + pd.Timedelta(days=1),
                        HORIZON_B_END, freq='D')
preds_B = {t: [] for t in TARGETS}

for d in dates_B:
    row = make_calendar_row(d)

    # Medium-term lags: use actual training data only (no leakage)
    for t in TARGETS:
        tl = t.lower()
        h = list(df[t].values)  # only actuals
        n = len(h)
        for lag in [30, 365]:
            # days_ahead = how far d is from last training date
            days_ahead = (d - df['date'].iloc[-1]).days
            actual_idx = n - lag + days_ahead
            row[f'{tl}_lag_{lag}d'] = h[actual_idx] if 0 <= actual_idx < n else 0

        # Rolling 30d from actuals only
        for w in [30]:
            end_idx = n - 1  # last training index
            start_idx = max(0, end_idx - w + 1)
            vals = [h[i] for i in range(start_idx, end_idx + 1)]
            row[f'{tl}_rmean_{w}d'] = np.mean(vals) if vals else 0
            row[f'{tl}_rstd_{w}d']  = np.std(vals, ddof=1) if len(vals) > 1 else 0
        row[f'{tl}_exp_mean'] = np.mean(list(df[t].values))

    # Fill remaining from seasonal profile
    md, dd = d.month, d.day
    sp = seasonal[(seasonal['month'] == md) & (seasonal['day_of_month'] == dd)]
    for f in feats_B:
        if f not in row:
            row[f] = sp[f].values[0] if (len(sp) > 0 and f in sp.columns) else 0

    x = pd.DataFrame([{f: row.get(f, 0) for f in feats_B}])
    for t in TARGETS:
        pred = max(inv_t(models_B[t].predict(x)[0]), 0)
        preds_B[t].append(pred)

print(f"    Predicted {len(dates_B)} days")

# ── 5c. MODEL C: Direct long-term (day 31+) ────────────────────────
print("  [Model C] Direct prediction (day 31+)...")

dates_C = pd.date_range(pd.Timestamp(HORIZON_B_END) + pd.Timedelta(days=1),
                        TEST_END, freq='D')

# For Model C, only calendar features - can batch predict
rows_C = [make_calendar_row(d) for d in dates_C]
test_C = pd.DataFrame(rows_C)

# Fill any calendar features that might be missing
for f in feats_C:
    if f not in test_C.columns:
        test_C[f] = 0

preds_C = {}
for t in TARGETS:
    preds_C[t] = np.maximum(inv_t(models_C[t].predict(test_C[feats_C])), 0).tolist()

print(f"    Predicted {len(dates_C)} days")

# ====================================================================
# 6. CONCATENATE & SAVE
# ====================================================================
print("\n" + "=" * 70); print("STEP 6: Concatenate & save submission"); print("=" * 70)

all_dates = list(dates_A) + list(dates_B) + list(dates_C)
all_rev   = preds_A['Revenue'] + preds_B['Revenue'] + preds_C['Revenue']
all_cogs  = preds_A['COGS'] + preds_B['COGS'] + preds_C['COGS']

submission = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in all_dates],
    'Revenue': np.round(all_rev, 2),
    'COGS': np.round(all_cogs, 2),
})
submission.to_csv('data/submission.csv', index=False)

rev = np.array(all_rev)
cogs = np.array(all_cogs)

print(f"\nSubmission saved -- {len(submission)} rows")
print(f"  Horizon A (1-14):  {len(dates_A)} days")
print(f"  Horizon B (15-30): {len(dates_B)} days")
print(f"  Horizon C (31+):   {len(dates_C)} days")
print(submission.head(10))

for label, arr in [('Revenue', rev), ('COGS', cogs)]:
    print(f"\n{label} -- mean: {arr.mean():,.2f}, min: {arr.min():,.2f}, max: {arr.max():,.2f}")
print(f"Margin  -- mean: {(rev - cogs).mean():,.2f}")
print(f"\nTotal time: {time.time()-t0:.0f}s")

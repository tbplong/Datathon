import pandas as pd
import numpy as np

df = pd.read_csv('data/revenue_features_v4.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

TARGETS = ['Revenue', 'COGS']
feature_cols = [c for c in df.columns if c not in ['date'] + TARGETS]
df['year'] = df['date'].dt.year

# 1. Yearly Revenue Trend
print("=" * 70)
print("YEARLY REVENUE TREND")
print("=" * 70)
for y in range(2012, 2023):
    mask = df['year'] == y
    rev = df.loc[mask, 'Revenue']
    cogs = df.loc[mask, 'COGS']
    print(f"  {y}: Rev mean={rev.mean():>12,.0f}  COGS mean={cogs.mean():>12,.0f}  rows={mask.sum()}")

# 2. Check lag features vs actual
print("\n" + "=" * 70)
print("LAG FEATURE SANITY CHECK")
print("=" * 70)
check = df[['date', 'Revenue', 'revenue_lag_1d', 'COGS', 'COGS_lag_1d']].head(10)
print(check.to_string())
# Verify lag_1d is shifted correctly
df['expected_rev_lag1'] = df['Revenue'].shift(1)
mismatch = (df['revenue_lag_1d'] - df['expected_rev_lag1']).abs()
print(f"\nrevenue_lag_1d vs shift(1) mismatch: mean={mismatch.mean():.4f}, max={mismatch.max():.4f}")

# 3. When do operational features start?
print("\n" + "=" * 70)
print("OPERATIONAL FEATURE AVAILABILITY")
print("=" * 70)
op_cols = ['order_count_lag_1d', 'total_quantity_lag_1d', 'total_sessions_lag_1d',
           'unique_customers_lag_1d', 'total_payment_value_lag_1d']
for col in op_cols:
    if col in df.columns:
        first_nonzero_idx = df[df[col] > 0].index
        if len(first_nonzero_idx) > 0:
            idx = first_nonzero_idx[0]
            print(f"  {col}: first nonzero at row {idx} ({df.loc[idx, 'date'].date()})")
        else:
            print(f"  {col}: ALL ZEROS")

# 4. Data by year - order_count availability
print("\n" + "=" * 70)
print("ORDER COUNT BY YEAR")
print("=" * 70)
for y in range(2012, 2023):
    mask = df['year'] == y
    ocl = df.loc[mask, 'order_count_lag_1d']
    nonzero = (ocl > 0).sum()
    print(f"  {y}: rows={mask.sum():>4}, order_count_lag_1d mean={ocl.mean():>8.1f}, nonzero={nonzero:>4}/{mask.sum()}")

# 5. Revenue distribution shifts
print("\n" + "=" * 70)
print("REVENUE CV BY YEAR (coefficient of variation)")
print("=" * 70)
for y in range(2012, 2023):
    mask = df['year'] == y
    rev = df.loc[mask, 'Revenue']
    cv = rev.std() / rev.mean()
    print(f"  {y}: mean={rev.mean():>12,.0f}  std={rev.std():>12,.0f}  CV={cv:.3f}")

# 6. Feature importance quick check with correlation
print("\n" + "=" * 70)
print("FEATURE GROUPS CORRELATION WITH REVENUE")
print("=" * 70)
groups = {
    'lag_1d': [c for c in feature_cols if c.endswith('_lag_1d')],
    'lag_7d': [c for c in feature_cols if c.endswith('_lag_7d')],
    'lag_30d': [c for c in feature_cols if c.endswith('_lag_30d')],
    'lag_365d': [c for c in feature_cols if 'lag_365d' in c],
    'rolling_mean': [c for c in feature_cols if 'rmean' in c],
    'rolling_std': [c for c in feature_cols if 'rstd' in c],
    'calendar': ['day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_tet_window', 'is_mega_sale', 'is_payday'],
    'promo': [c for c in feature_cols if 'promo' in c or 'discount' in c],
    'wow/mom': [c for c in feature_cols if 'wow' in c or 'mom' in c],
}
for name, cols in groups.items():
    valid = [c for c in cols if c in feature_cols]
    if valid:
        avg_corr = df[valid].corrwith(df['Revenue']).abs().mean()
        print(f"  {name:15s}: {len(valid):>3} features, avg |corr|={avg_corr:.4f}")

# 7. Check for early period with no operational data
print("\n" + "=" * 70)
print("EARLY PERIOD ANALYSIS")
print("=" * 70)
early_mask = df['order_count_lag_1d'] == 0
print(f"Rows with order_count_lag_1d=0: {early_mask.sum()} ({early_mask.mean():.1%})")
if early_mask.sum() > 0:
    print(f"  Date range: {df.loc[early_mask, 'date'].min().date()} to {df.loc[early_mask, 'date'].max().date()}")
    # Revenue stats for these rows
    rev_early = df.loc[early_mask, 'Revenue']
    rev_late = df.loc[~early_mask, 'Revenue']
    print(f"  Revenue (zero ops): mean={rev_early.mean():,.0f}, std={rev_early.std():,.0f}")
    print(f"  Revenue (has ops):  mean={rev_late.mean():,.0f}, std={rev_late.std():,.0f}")

# 8. Check multicollinearity among top features
print("\n" + "=" * 70)
print("MULTICOLLINEARITY CHECK (top correlated feature pairs)")
print("=" * 70)
top_feats = df[feature_cols].corrwith(df['Revenue']).abs().nlargest(20).index.tolist()
corr_matrix = df[top_feats].corr()
pairs = []
for i in range(len(top_feats)):
    for j in range(i+1, len(top_feats)):
        pairs.append((top_feats[i], top_feats[j], abs(corr_matrix.iloc[i, j])))
pairs.sort(key=lambda x: x[2], reverse=True)
for f1, f2, c in pairs[:10]:
    print(f"  {c:.4f}  {f1}  <->  {f2}")

# 9. Year_trend feature check
print("\n" + "=" * 70)
print("YEAR_TREND FEATURE CHECK")
print("=" * 70)
if 'year_trend' in feature_cols:
    print(f"  year_trend unique values: {df['year_trend'].unique()}")
    print(f"  year_trend correlation with Revenue: {df['year_trend'].corr(df['Revenue']):.4f}")

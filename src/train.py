from __future__ import annotations

from pathlib import Path
import gc
import time
import warnings

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_PATH = DATA_DIR / "revenue_features_full.csv"
ENGINEERED_PATH = DATA_DIR / "revenue_features_eng.csv"
SUBMISSION_PATH = ROOT/"submission" / "submission.csv"
TEST_START = "2023-01-01"
TEST_END = "2024-07-01"
TARGETS = ["Revenue", "COGS"]
LAG_DAYS = [1, 2, 3, 7, 14, 30, 365]
ROLLING_WINDOWS = [7, 14, 30]
OPERATIONAL_COLUMNS = [
    "order_count",
    "total_quantity",
    "total_payment_value",
    "total_shipping_fee",
    "total_discount",
    "total_sessions",
    "total_unique_visitors",
    "total_page_views",
    "new_customers_count",
    "return_quantity",
    "refund_amount",
    "total_reviews",
    "total_stock_on_hand",
    "total_units_received",
]
BASE_EXCLUSIONS = {"date", "year", *TARGETS}
RAW_SAME_DAY = {
    "avg_unit_price",
    "unique_items",
    "avg_product_price",
    "avg_product_cogs",
    "unique_colors",
}
BREAKDOWN_PREFIXES = (
    "orders_payment_",
    "orders_device_",
    "orders_source_",
    "orders_status_",
    "orders_region_",
    "returns_reason_",
    "new_cust_acq_",
    "new_cust_age_",
    "new_cust_gen_",
    "qty_category_",
    "qty_segment_",
    "promo_channel_",
    "max_discount_value",
)
PRUNE_SET = {
    "items_reorder",
    "unique_sizes",
    "promo_type_fixed",
    "is_weekend",
    "refund_amount_lag_1d",
    "cogs_momentum_7_14",
    "cogs_lag_14d",
    "cogs_lag_3d",
    "revenue_diff_1d_30d",
    "revenue_momentum_7_14",
    "cogs_rmean_7d",
    "items_stockout",
    "dow_sin",
    "return_quantity_lag_1d",
    "cogs_rmean_14d",
    "revenue_diff_1d_7d",
    "revenue_lag_30d",
}
MODEL_PARAMS = {
    "n_estimators": 5000,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": SEED,
    "seed": SEED,
    "n_jobs": 1,
    "verbosity": 0,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def log_t(y: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    result = np.log1p(y)
    if isinstance(y, pd.Series):
        return pd.Series(result, index=y.index)
    return result


def inv_t(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(y, -20, 20))


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_x_month"] = df["day_of_week"] * 12 + df["month"]
    df["year_trend"] = df["year"] - 2012
    df["trend_sq"] = (df["year"] - 2012) ** 2
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for target in TARGETS:
        target_key = target.lower()
        for lag in LAG_DAYS:
            df[f"{target_key}_lag_{lag}d"] = df[target].shift(lag)

    operational_cols = [column for column in OPERATIONAL_COLUMNS if column in df.columns]
    for column in operational_cols:
        df[f"{column}_lag_1d"] = df[column].shift(1)
        df[f"{column}_lag_7d"] = df[column].shift(7)

    for target in TARGETS:
        target_key = target.lower()
        lag_1d = f"{target_key}_lag_1d"
        for window in ROLLING_WINDOWS:
            df[f"{target_key}_rmean_{window}d"] = df[lag_1d].rolling(window, min_periods=1).mean()
            df[f"{target_key}_rstd_{window}d"] = df[lag_1d].rolling(window, min_periods=1).std()
        df[f"{target_key}_exp_mean"] = df[lag_1d].expanding(min_periods=1).mean()

    for column in ["order_count", "total_quantity", "total_payment_value"]:
        lag_1d = f"{column}_lag_1d"
        if lag_1d in df.columns:
            for window in [7, 30]:
                df[f"{column}_rmean_{window}d"] = df[lag_1d].rolling(window, min_periods=1).mean()

    for target in TARGETS:
        target_key = target.lower()
        lag_1d = f"{target_key}_lag_1d"
        lag_7d = f"{target_key}_lag_7d"
        lag_30d = f"{target_key}_lag_30d"
        rmean_7d = f"{target_key}_rmean_7d"
        rmean_14d = f"{target_key}_rmean_14d"
        rmean_30d = f"{target_key}_rmean_30d"

        if rmean_7d in df.columns and rmean_30d in df.columns:
            df[f"{target_key}_momentum_7_30"] = df[rmean_7d] / df[rmean_30d].replace(0, np.nan)
        if rmean_7d in df.columns and rmean_14d in df.columns:
            df[f"{target_key}_momentum_7_14"] = df[rmean_7d] / df[rmean_14d].replace(0, np.nan)
        if lag_1d in df.columns and rmean_7d in df.columns:
            df[f"{target_key}_dev_from_7d"] = df[lag_1d] / df[rmean_7d].replace(0, np.nan)
        if lag_1d in df.columns and rmean_30d in df.columns:
            df[f"{target_key}_dev_from_30d"] = df[lag_1d] / df[rmean_30d].replace(0, np.nan)
        if lag_1d in df.columns and lag_7d in df.columns:
            df[f"{target_key}_diff_1d_7d"] = df[lag_1d] - df[lag_7d]
        if lag_1d in df.columns and lag_30d in df.columns:
            df[f"{target_key}_diff_1d_30d"] = df[lag_1d] - df[lag_30d]

    if "revenue_lag_1d" in df.columns and "cogs_lag_1d" in df.columns:
        df["rev_cogs_ratio_lag1"] = df["revenue_lag_1d"] / df["cogs_lag_1d"].replace(0, np.nan)
    if "revenue_rstd_7d" in df.columns and "revenue_rmean_7d" in df.columns:
        df["revenue_cv_7d"] = df["revenue_rstd_7d"] / df["revenue_rmean_7d"].replace(0, np.nan)

    return df.fillna(0)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    same_day_breakdowns = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in BREAKDOWN_PREFIXES)
    ]
    excluded = set(BASE_EXCLUSIONS) | RAW_SAME_DAY | set(same_day_breakdowns)
    feature_cols = [column for column in df.columns if column not in excluded]

    correlation_revenue = df[feature_cols].corrwith(df["Revenue"]).abs()
    correlation_cogs = df[feature_cols].corrwith(df["COGS"]).abs()
    max_correlation = pd.concat([correlation_revenue, correlation_cogs], axis=1).max(axis=1)
    weak_features = max_correlation[max_correlation < 0.05].index.tolist()
    feature_cols = [column for column in feature_cols if column not in weak_features]

    pruned_features = [column for column in feature_cols if column in PRUNE_SET]
    feature_cols = [column for column in feature_cols if column not in PRUNE_SET]

    print(f"  Features kept: {len(feature_cols)}")
    print(f"  Dropped weak-correlation features: {len(weak_features)}")
    print(f"  Dropped manual prune features: {len(pruned_features)}")
    if pruned_features:
        print(f"    Removed: {pruned_features}")

    return feature_cols


def build_sample_weights(years: np.ndarray, decay_rate: float = 0.15) -> np.ndarray:
    max_year = years.max()
    return np.exp(-decay_rate * (max_year - years))


def optimize_hyperparameters(
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_weights: np.ndarray,
    n_trials: int = 30,
) -> dict:
    """Use Optuna to find optimal XGBoost hyperparameters via time-series CV."""
    print("\n" + "=" * 70)
    print("STEP 3: Optuna hyperparameter tuning")
    print("=" * 70)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": SEED,
            "seed": SEED,
            "early_stopping_rounds": 100,
            "n_jobs": 1,
            "verbosity": 0,
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores: list[float] = []
        
        for target in TARGETS:
            y_raw = df[target]
            y_log = pd.Series(np.log1p(y_raw.values), index=y_raw.index)
            for train_idx, val_idx in tscv.split(df[feature_cols]):
                model = XGBRegressor(**params)
                model.fit(
                    df[feature_cols].iloc[train_idx],
                    y_log.iloc[train_idx],
                    eval_set=[(df[feature_cols].iloc[val_idx], y_log.iloc[val_idx])],
                    sample_weight=sample_weights[train_idx],
                    verbose=False,
                )
                preds = inv_t(model.predict(df[feature_cols].iloc[val_idx]))
                rmse = np.sqrt(mean_squared_error(df[target].iloc[val_idx], preds))
                scores.append(float(rmse))
        
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = {
        "n_estimators": 5000,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": SEED,
        "seed": SEED,
        "early_stopping_rounds": 150,
        "n_jobs": 1,
        "verbosity": 0,
        "learning_rate": study.best_params["lr"],
        "max_depth": study.best_params["max_depth"],
        "subsample": study.best_params["subsample"],
        "colsample_bytree": study.best_params["colsample_bytree"],
        "min_child_weight": study.best_params["min_child_weight"],
        "gamma": study.best_params["gamma"],
        "reg_alpha": study.best_params["reg_alpha"],
        "reg_lambda": study.best_params["reg_lambda"],
    }
    
    print(f"  Best RMSE: {study.best_value:,.2f}")
    print(f"  Best params: {best_params}")
    
    return best_params


def evaluate_time_series_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_weights: np.ndarray,
    params: dict,
) -> None:
    print("\n" + "=" * 70)
    print("STEP 4: Time-series cross-validation")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)
    for target in TARGETS:
        y_raw = df[target]
        y_log = pd.Series(np.log1p(y_raw.values), index=y_raw.index)
        rmses: list[float] = []
        maes: list[float] = []
        r2s: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df[feature_cols])):
            model = XGBRegressor(**params)
            model.fit(
                df[feature_cols].iloc[train_idx],
                y_log.iloc[train_idx],
                eval_set=[(df[feature_cols].iloc[val_idx], y_log.iloc[val_idx])],
                sample_weight=sample_weights[train_idx],
                verbose=False,
            )

            predictions = np.maximum(inv_t(model.predict(df[feature_cols].iloc[val_idx])), 0)
            actual = y_raw.iloc[val_idx]
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)
            print(f"  [{target}] Fold {fold + 1}: RMSE={rmse:>14,.2f}  MAE={mae:>14,.2f}  R2={r2:.4f}")

        print(
            f"  {target} Avg: RMSE={np.mean(rmses):>14,.2f} "
            f"MAE={np.mean(maes):>14,.2f} R2={np.mean(r2s):.4f}\n"
        )


def train_final_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_weights: np.ndarray,
    params: dict,
) -> dict[str, XGBRegressor]:
    print("=" * 70)
    print("STEP 5: Train final models")
    print("=" * 70)

    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    models: dict[str, XGBRegressor] = {}
    for target in TARGETS:
        model = XGBRegressor(**final_params)
        model.fit(
            df[feature_cols],
            log_t(df[target]),
            sample_weight=sample_weights,
            verbose=False,
        )
        models[target] = model
        print(f"  {target}: trained")

    return models


def compute_feature_importance_and_shap(
    df: pd.DataFrame,
    X_all: pd.DataFrame,
    models: dict[str, XGBRegressor],
    feature_cols: list[str],
) -> None:
    print("\n" + "=" * 70)
    print("STEP 6: Feature importance & SHAP analysis")
    print("=" * 70)

    for target in TARGETS:
        model = models[target]
        target_key = target.lower()
        print(f"\n  [{target}] Computing feature importance and SHAP values...")

        # XGBoost built-in importance
        importance_data: dict[str, list] = {"feature": feature_cols}
        for importance_type in ["weight", "gain", "cover"]:
            booster_scores = model.get_booster().get_score(importance_type=importance_type)
            importance_data[importance_type] = [booster_scores.get(feature, 0.0) for feature in feature_cols]

        importance_df = pd.DataFrame(importance_data).sort_values("gain", ascending=False).reset_index(drop=True)
        importance_path = DATA_DIR / f"feature_importance_{target_key}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"    Importance saved -> {importance_path.name}")
        print("    Top 15 by gain:")
        print(importance_df.head(15)[["feature", "gain", "weight", "cover"]].to_string(index=False))

        # SHAP values
        print(f"    Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_all)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
        shap_path = DATA_DIR / f"shap_importance_{target_key}.csv"
        shap_df.to_csv(shap_path, index=False)
        print(f"    SHAP importance saved -> {shap_path.name}")
        print("    Top 15 by mean |SHAP|:")
        print(shap_df.head(15).to_string(index=False))

        # SHAP summary plot (beeswarm)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_all, feature_names=feature_cols, max_display=30, show=False)
        plt.title(f"SHAP Summary - {target}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        beeswarm_path = DATA_DIR / f"shap_summary_{target_key}.png"
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    SHAP beeswarm plot saved -> {beeswarm_path.name}")

        # SHAP bar plot (global importance)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_all, feature_names=feature_cols, plot_type="bar", max_display=30, show=False)
        plt.title(f"SHAP Feature Importance (Bar) - {target}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        bar_path = DATA_DIR / f"shap_bar_{target_key}.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    SHAP bar plot saved -> {bar_path.name}")

        # Combined ranking table
        combined = importance_df[["feature", "gain"]].merge(shap_df, on="feature", how="outer").fillna(0)
        combined["gain_rank"] = combined["gain"].rank(ascending=False).astype(int)
        combined["shap_rank"] = combined["mean_abs_shap"].rank(ascending=False).astype(int)
        combined["avg_rank"] = (combined["gain_rank"] + combined["shap_rank"]) / 2
        combined = combined.sort_values("avg_rank").reset_index(drop=True)
        combined_path = DATA_DIR / f"feature_ranking_{target_key}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"    Combined ranking saved -> {combined_path.name}")


def build_test_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq="D")
    test_df = pd.DataFrame({"date": test_dates})
    test_df = add_calendar_features(test_df)

    group_keys = ["month", "day_of_month"]
    feature_defaults = [column for column in feature_cols if column not in {"month", "day_of_month"}]
    seasonal_profile = df.groupby(group_keys)[feature_defaults].mean(numeric_only=True).reset_index()
    test_df = test_df.merge(seasonal_profile, on=group_keys, how="left")

    for column in feature_defaults:
        if column in test_df.columns:
            test_df[column] = test_df[column].interpolate(method="linear").bfill().ffill()
        else:
            test_df[column] = 0

    for column in feature_cols:
        if column not in test_df.columns:
            test_df[column] = 0

    return test_df[feature_cols]


def main() -> None:
    set_seed(SEED)
    start_time = time.time()

    print("=" * 70)
    print("STEP 1: Load data")
    print("=" * 70)
    df = pd.read_csv(SOURCE_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Raw shape: {df.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: Feature engineering")
    print("=" * 70)
    df = add_calendar_features(df)
    df = add_lag_features(df)

    feature_cols = select_feature_columns(df)
    df.to_csv(ENGINEERED_PATH, index=False)
    print(f"  Engineered data saved -> {ENGINEERED_PATH.name}")

    X_all = df[feature_cols].copy()
    years = df["year"].to_numpy()
    sample_weights = build_sample_weights(years)
    print(f"  Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    # Optimize hyperparameters using Optuna
    best_params = optimize_hyperparameters(df, feature_cols, sample_weights, n_trials=30)

    # Evaluate with time-series CV using tuned params
    evaluate_time_series_cv(df, feature_cols, sample_weights, best_params)

    # Train final models with tuned params
    models = train_final_models(df, feature_cols, sample_weights, best_params)
    gc.collect()

    # Compute feature importance and SHAP values
    compute_feature_importance_and_shap(df, X_all, models, feature_cols)

    print("\n" + "=" * 70)
    print("STEP 7: Build test features and predict")
    print("=" * 70)
    X_test = build_test_features(df, feature_cols)
    print(f"  Test feature shape: {X_test.shape}")

    revenue_pred = np.maximum(inv_t(models["Revenue"].predict(X_test)), 0)
    cogs_pred = np.maximum(inv_t(models["COGS"].predict(X_test)), 0)

    submission = pd.DataFrame(
        {
            "Date": pd.date_range(start=TEST_START, end=TEST_END, freq="D").strftime("%Y-%m-%d"),
            "Revenue": np.round(revenue_pred, 2),
            "COGS": np.round(cogs_pred, 2),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"\n  Submission saved -> {SUBMISSION_PATH.name} ({len(submission)} rows)")
    print(submission.head(10).to_string(index=False))
    print(
        f"\n  Revenue mean={revenue_pred.mean():,.2f} min={revenue_pred.min():,.2f} max={revenue_pred.max():,.2f}"
    )
    print(
        f"  COGS    mean={cogs_pred.mean():,.2f} min={cogs_pred.min():,.2f} max={cogs_pred.max():,.2f}"
    )
    print(f"  Margin  mean={(revenue_pred - cogs_pred).mean():,.2f}")
    print(f"\n  Total time: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
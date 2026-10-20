"""Features, models and walk-forward backtest for day-ahead gas demand.

Forecast setup
--------------
On gas day D at 12:00 we forecast total demand of gas day D+1 (the target).
Instead of predicting the level directly, the models predict the change
between the target and today's intraday estimate published by Snam:

    y = demand(D+1) - estimate_of_demand(D) at 12:00

The level is then reconstructed by adding the estimate back. Predicting the
change lets tree models handle the strong seasonal trend, which they cannot
extrapolate on their own.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

FEATURES = [
    "demand_today_at_issue",  # Snam intraday estimate of today's demand
    "hdd_forecast",           # forecast heating degree-days for the target day
    "delta_hdd",              # change in HDD between today and the target day
    "delta_lag_7",            # same day-over-day change one week earlier
    "dow_sin", "dow_cos",     # day of week of the target (cyclical)
    "is_off", "prev_is_off", "off_transition",
    "is_bridge", "is_august_break",
]


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    dow = pd.Series(df.index.dayofweek, index=df.index)

    df["delta_hdd"] = df["hdd_forecast"] - df["hdd_today_at_issue"]
    # change from day t-8 to t-7: both settled actuals at issue time
    df["delta_lag_7"] = df["demand"].shift(7) - df["demand"].shift(8)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # non-working days (weekend or national holiday) and transitions
    df["is_off"] = ((dow >= 5) | (df["is_holiday"] == 1)).astype(int)
    df["prev_is_off"] = df["is_off"].shift(1)
    df["off_transition"] = df["is_off"] - df["prev_is_off"]
    next_is_off = df["is_off"].shift(-1, fill_value=0)  # calendar, known in advance
    df["is_bridge"] = ((df["is_off"] == 0) & (df["prev_is_off"] == 1) & (next_is_off == 1)).astype(int)

    # Christmas week and mid-August: many industrial sites close
    md = df.index.month * 100 + df.index.day
    df.loc[(md >= 1224) | (md <= 106), "is_bridge"] = 1
    df["is_august_break"] = ((md >= 805) & (md <= 820)).astype(int)

    df["target_delta"] = df["demand"] - df["demand_today_at_issue"]
    return df


MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=42),
}


def walk_forward(df: pd.DataFrame, test_start: str, refit_every: int = 7) -> pd.DataFrame:
    """Expanding-window backtest.

    Every `refit_every` days the models are re-trained on all data strictly
    before the block, then predict each day of the block one day ahead.
    """
    data = df.dropna(subset=FEATURES + ["target_delta"])
    test_days = data.index[data.index >= test_start]
    preds = {name: pd.Series(index=test_days, dtype=float) for name in MODELS}

    for i in range(0, len(test_days), refit_every):
        block = test_days[i:i + refit_every]
        train = data[data.index < block[0]]
        for name, make in MODELS.items():
            model = make().fit(train[FEATURES], train["target_delta"])
            delta = model.predict(data.loc[block, FEATURES])
            preds[name].loc[block] = delta + data.loc[block, "demand_today_at_issue"]

    out = pd.DataFrame(preds)
    out["Naive (today's estimate)"] = data.loc[test_days, "demand_today_at_issue"]
    out["Seasonal naive (t-7)"] = data.loc[test_days, "demand"].index.map(
        lambda d: df["demand"].get(d - pd.Timedelta(days=7)))
    out["Snam forecast"] = data.loc[test_days, "snam_forecast"]
    out["actual"] = data.loc[test_days, "demand"]
    return out


def score(results: pd.DataFrame) -> pd.DataFrame:
    """MAE in GWh/day (data are in MWh), MAPE and R² for every column."""
    y = results["actual"]
    rows = []
    for col in results.columns.drop("actual"):
        p = results[col]
        rows.append({
            "model": col,
            "MAE (GWh/day)": mean_absolute_error(y, p) / 1e3,
            "MAPE (%)": 100 * mean_absolute_percentage_error(y, p),
            "R²": r2_score(y, p),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("MAE (GWh/day)")


def fit_final(df: pd.DataFrame, name: str = "Gradient Boosting"):
    """Fit a model on all available data (used for feature importance)."""
    data = df.dropna(subset=FEATURES + ["target_delta"])
    return MODELS[name]().fit(data[FEATURES], data["target_delta"]), data

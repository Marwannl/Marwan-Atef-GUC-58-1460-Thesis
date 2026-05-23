import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent.parent / "cox_model.pkl"
DAYS = [1, 2, 3, 5, 7, 10, 14]
MIN_EPISODE = 3

UP_FEATURES = ["rsi", "volume_declining", "near_upper_band", "bull_regime"]
DOWN_FEATURES = ["rsi", "volume_declining", "near_lower_band", "bear_regime"]


def _rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def _bollinger_bands(prices, window=20):
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return mid + 2 * std, mid - 2 * std


def _market_regime(closes) -> int:
    # returns 1 for bull, -1 for bear, 0 for sideways
    if len(closes) < 50:
        return 0
    ma20 = float(closes.rolling(20).mean().iloc[-1])
    ma50 = float(closes.rolling(50).mean().iloc[-1])
    current = float(closes.iloc[-1])
    slope = (float(closes.iloc[-1]) - float(closes.iloc[-20])) / float(closes.iloc[-20]) * 100
    if current > ma20 > ma50 and slope > 3:
        return 1
    if current < ma20 < ma50 and slope < -3:
        return -1
    return 0


def build_trend_episodes(ticker: str, period: str = "5y") -> pd.DataFrame:
    import yfinance as yf
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, auto_adjust=True)
    if hist.empty or len(hist) < 50:
        return pd.DataFrame()

    close = hist["Close"]
    volume = hist["Volume"]
    rsi_series = _rsi(close)
    upper_bb, lower_bb = _bollinger_bands(close)
    daily_up = (close.diff() > 0).astype(int)

    episodes = []
    i = 1
    n = len(close)

    while i < n:
        d = int(daily_up.iloc[i])
        start = i
        while i < n and int(daily_up.iloc[i]) == d:
            i += 1
        duration = i - start
        if duration < MIN_EPISODE:
            continue

        # trend reached end of data without reversing, so it's right-censored
        event = 0 if i >= n else 1

        s = start
        rsi_val = float(rsi_series.iloc[s])
        if np.isnan(rsi_val):
            continue

        vol_window = volume.iloc[max(0, s - 3):s].values
        vol_declining = int(
            len(vol_window) >= 3 and
            float(vol_window[-1]) < float(vol_window[-2]) < float(vol_window[-3])
        )

        c = float(close.iloc[s])
        ub_val = float(upper_bb.iloc[s])
        lb_val = float(lower_bb.iloc[s])
        if np.isnan(ub_val) or np.isnan(lb_val):
            continue

        near_upper = int(c > ub_val * 0.98)
        near_lower = int(c < lb_val * 1.02)
        regime = _market_regime(close.iloc[:s + 1])

        if d == 1:
            episodes.append({
                "duration": duration,
                "event": event,
                "direction": 1,
                "rsi": rsi_val,
                "volume_declining": vol_declining,
                "near_upper_band": near_upper,
                "bull_regime": int(regime == 1),
            })
        else:
            episodes.append({
                "duration": duration,
                "event": event,
                "direction": 0,
                "rsi": rsi_val,
                "volume_declining": vol_declining,
                "near_lower_band": near_lower,
                "bear_regime": int(regime == -1),
            })

    return pd.DataFrame(episodes)


def fit_cox(tickers: list[str], period: str = "5y") -> dict:
    from lifelines import CoxPHFitter

    all_up, all_down = [], []

    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        df = build_trend_episodes(ticker, period)
        if df.empty:
            continue
        up = df[df["direction"] == 1][["duration", "event"] + UP_FEATURES]
        down = df[df["direction"] == 0][["duration", "event"] + DOWN_FEATURES]
        all_up.append(up)
        all_down.append(down)

    if not all_up or not all_down:
        raise ValueError("Insufficient data to fit Cox models")

    df_up = pd.concat(all_up, ignore_index=True).dropna()
    df_down = pd.concat(all_down, ignore_index=True).dropna()

    print(f"\nFitting uptrend model on {len(df_up)} episodes...")
    cph_up = CoxPHFitter()
    cph_up.fit(df_up, duration_col="duration", event_col="event")
    cph_up.print_summary()

    print(f"\nFitting downtrend model on {len(df_down)} episodes...")
    cph_down = CoxPHFitter()
    cph_down.fit(df_down, duration_col="duration", event_col="event")
    cph_down.print_summary()

    return {"up": cph_up, "down": cph_down}


def save_model(models: dict, path: Path = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(models, f)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _predict_sf(model, covariates: dict, times: list[int]) -> pd.Series:
    row = pd.DataFrame([covariates])
    sf_df = model.predict_survival_function(row, times=times)
    return sf_df.iloc[:, 0]


def cox_survival_curve(models: dict, direction: str, covariates: dict, current_age: int = 0) -> list:
    model = models["up"] if direction == "up" else models["down"]
    future_times = [current_age + d for d in DAYS]
    all_times = sorted(set([max(0, current_age)] + future_times))
    sf = _predict_sf(model, covariates, all_times)

    s_now = float(sf.loc[current_age]) if current_age > 0 else 1.0
    s_now = max(s_now, 1e-9)

    return [
        {
            "day": d,
            "probability": round(min(1.0, float(sf.loc[current_age + d]) / s_now), 4),
        }
        for d in DAYS
    ]


def cox_median_survival(models: dict, direction: str, covariates: dict, current_age: int = 0) -> int:
    model = models["up"] if direction == "up" else models["down"]
    times = list(range(max(0, current_age), current_age + 61))
    sf = _predict_sf(model, covariates, times)

    s_now = float(sf.loc[current_age]) if current_age > 0 else 1.0
    s_now = max(s_now, 1e-9)

    for d in range(1, 61):
        if float(sf.loc[current_age + d]) / s_now < 0.5:
            return d
    return 60

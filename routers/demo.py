import asyncio
import io
import re
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pdfplumber
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import Stock, DailySnapshot, TrendRecord, Prediction
from routers.survival import (
    compute_hazard_rate, survival_curve, median_survival_days, generate_explanation,
)
from routers.cox_survival import (
    load_model as _load_cox_model,
    cox_survival_curve, cox_median_survival,
)

_cox_models = _load_cox_model()

router = APIRouter(prefix="/demo", tags=["demo"])

COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp.",
    "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.", "META": "Meta Platforms",
    "AMZN": "Amazon.com Inc.", "NFLX": "Netflix Inc.", "AMD": "AMD Inc.",
    "SPY": "S&P 500 ETF",
}


def _rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def _bollinger(prices, window=20):
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return mid, mid + 2 * std, mid - 2 * std


def _market_regime(closes) -> str:
    if len(closes) < 50:
        return "Sideways"
    ma20 = closes.rolling(20).mean().iloc[-1]
    ma50 = closes.rolling(50).mean().iloc[-1]
    current = closes.iloc[-1]
    slope = (closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20] * 100
    if current > ma20 > ma50 and slope > 3:
        return "Bull"
    if current < ma20 < ma50 and slope < -3:
        return "Bear"
    return "Sideways"


def _linear_forecast(closes) -> dict:
    from sklearn.linear_model import LinearRegression
    c = closes.values[-90:]
    x = np.arange(len(c)).reshape(-1, 1)
    model = LinearRegression().fit(x, c)
    fx = np.arange(len(c), len(c) + 7).reshape(-1, 1)
    fp = model.predict(fx)
    pct = (fp[-1] - c[-1]) / c[-1] * 100
    return {"prices": fp.tolist(), "pct_change": float(pct)}


def _sentiment(news: list) -> dict:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        ana = SentimentIntensityAnalyzer()
        scores = [ana.polarity_scores(n.get("title", ""))["compound"]
                  for n in news if n.get("title")]
        avg = float(np.mean(scores)) if scores else 0.0
    except Exception:
        avg = 0.0
    label = "positive" if avg > 0.05 else "negative" if avg < -0.05 else "neutral"
    return {"score": avg, "label": label}


def _yf_session():
    import requests
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })
    s.verify = False
    return s


def _fetch_yfinance(ticker: str):
    import time, pandas as pd
    s = _yf_session()
    end = int(time.time())
    start = end - 86400 * 400  # ~13 months
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={start}&period2={end}&interval=1d&includePrePost=false")
    r = s.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    chart = data.get("chart", {})
    result = (chart.get("result") or [None])[0]
    if result is None:
        err = (chart.get("error") or {}).get("description", f"No data for {ticker}")
        raise ValueError(err)

    meta = result["meta"]
    timestamps = result.get("timestamp", [])
    q = result.get("indicators", {}).get("quote", [{}])[0]
    opens  = q.get("open",   [])
    highs  = q.get("high",   [])
    lows   = q.get("low",    [])
    closes = q.get("close",  [])
    vols   = q.get("volume", [])

    if not closes:
        raise ValueError(f"No data for {ticker}")

    idx = pd.to_datetime(timestamps, unit="s", utc=True)
    hist = pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                          "Close": closes, "Volume": vols}, index=idx)
    hist = hist.dropna(subset=["Close"])

    live_price = float(meta.get("regularMarketPrice") or closes[-1])
    # use second-to-last daily close as previous day's close
    prev_close = float(closes[-2]) if len(closes) >= 2 and closes[-2] is not None else live_price

    # company name straight from Yahoo — no hardcoded dict needed
    company_name = meta.get("longName") or meta.get("shortName") or ticker

    # fetch news headlines for sentiment analysis
    news = []
    try:
        news_url = (f"https://query1.finance.yahoo.com/v1/finance/search"
                    f"?q={ticker}&quotesCount=0&newsCount=10&enableFuzzyQuery=false")
        nr = s.get(news_url, timeout=8)
        news = nr.json().get("news", [])[:10]
    except Exception:
        pass

    return hist, news, live_price, prev_close, company_name


def _resolve_ticker_sync(ticker: str):
    import time, pandas as pd
    s = _yf_session()
    # Try Yahoo Finance search API
    search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}&quotesCount=8&newsCount=0"
    try:
        sq = s.get(search_url, timeout=10).json()
        quotes = sq.get("quotes", [])
    except Exception:
        quotes = []
    _US_EXCHANGES = frozenset({"NMS", "NYQ", "NCM", "NGM", "PCX", "ASE"})
    for q in quotes:
        if q.get("exchange") not in _US_EXCHANGES or q.get("quoteType") != "EQUITY":
            continue
        resolved = q["symbol"]
        if resolved == ticker:
            continue
        hist, news, live_price, prev_close, _cn = _fetch_yfinance(resolved)
        company = q.get("shortname") or q.get("longname") or resolved
        return resolved, hist, news, company, live_price, prev_close
    raise ValueError(f"No US equity found for '{ticker}'")


def _detect_current_trend(snapshots: list) -> dict:
    if len(snapshots) < 2:
        snap = snapshots[-1] if snapshots else None
        return {
            "direction": "up",
            "started_at": snap.date if snap else str(date.today()),
            "duration_days": 1,
            "start_price": snap.close if snap else 100.0,
        }

    closes = [s.close for s in snapshots]
    dates = [s.date for s in snapshots]

    direction = "up" if closes[-1] >= closes[-2] else "down"
    start_idx = len(closes) - 1
    for i in range(len(closes) - 2, 0, -1):
        day_dir = "up" if closes[i] >= closes[i - 1] else "down"
        if day_dir != direction:
            break
        start_idx = i

    return {
        "direction": direction,
        "started_at": dates[start_idx],
        "duration_days": len(closes) - start_idx,
        "start_price": closes[start_idx],
    }


def _upsert_snapshots(db: Session, ticker: str, hist):
    closes = hist["Close"]
    rsi_series = _rsi(closes)
    ma7_series = closes.rolling(7).mean()
    ma30_series = closes.rolling(30).mean()
    _, upper_bb, lower_bb = _bollinger(closes)

    idx_list = list(hist.index)
    for i, (idx, row) in enumerate(hist.iterrows()):
        d = idx.date().isoformat()
        if db.query(DailySnapshot).filter_by(ticker=ticker, date=d).first():
            continue

        def safe(val):
            v = float(val)
            return None if (v != v) else round(v, 4)  # NaN check

        db.add(DailySnapshot(
            ticker=ticker, date=d,
            open=round(float(row["Open"]), 4),
            high=round(float(row["High"]), 4),
            low=round(float(row["Low"]), 4),
            close=round(float(row["Close"]), 4),
            volume=int(row["Volume"]),
            rsi=safe(rsi_series.iloc[i]),
            ma7=safe(ma7_series.iloc[i]),
            ma30=safe(ma30_series.iloc[i]),
            upper_bb=safe(upper_bb.iloc[i]),
            lower_bb=safe(lower_bb.iloc[i]),
        ))
    db.commit()


def _build_chart_json(hist, forecast_data: dict) -> dict:
    hist1y = hist.tail(365)
    closes = hist1y["Close"]
    mid, upper_bb, lower_bb = _bollinger(closes)
    rsi_series = _rsi(closes)

    def _ts(idx):
        try:
            return idx.date().isoformat()
        except Exception:
            return str(idx)[:10]

    ohlcv = [
        {
            "time": _ts(idx),
            "open": round(float(r["Open"]), 4),
            "high": round(float(r["High"]), 4),
            "low": round(float(r["Low"]), 4),
            "close": round(float(r["Close"]), 4),
            "volume": int(r["Volume"]),
        }
        for idx, r in hist1y.iterrows()
    ]

    bb = []
    for i, (idx, _) in enumerate(hist1y.iterrows()):
        ub, lb, mb = float(upper_bb.iloc[i]), float(lower_bb.iloc[i]), float(mid.iloc[i])
        if ub == ub and lb == lb:  # not NaN
            bb.append({"time": _ts(idx), "upper": round(ub, 4),
                       "mid": round(mb, 4), "lower": round(lb, 4)})

    rsi_pts = []
    for i, (idx, _) in enumerate(hist1y.iterrows()):
        v = float(rsi_series.iloc[i])
        if v == v:  # not NaN
            rsi_pts.append({"time": _ts(idx), "value": round(v, 2)})

    last_ts = hist.index[-1]
    forecast_pts = [{"time": _ts(last_ts), "value": round(float(hist["Close"].iloc[-1]), 4)}]
    for i, price in enumerate(forecast_data["prices"]):
        forecast_pts.append({
            "time": (last_ts + timedelta(days=i + 1)).date().isoformat(),
            "value": round(float(price), 4),
        })

    current = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2])
    daily = current - prev
    return {
        "ohlcv": ohlcv, "bb": bb, "rsi": rsi_pts, "forecast": forecast_pts,
        "current_price": round(current, 2),
        "daily_change": round(daily, 2),
        "daily_pct": round(daily / prev * 100, 2),
    }


def _fetch_live_price(ticker: str):
    try:
        import time
        s = _yf_session()
        end = int(time.time())
        start = end - 86400 * 5
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
               f"?period1={start}&period2={end}&interval=1d&includePrePost=false")
        r = s.get(url, timeout=10)
        r.raise_for_status()
        result = r.json()["chart"]["result"][0]
        meta = result["meta"]
        closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        live = float(meta["regularMarketPrice"])
        prev = float(closes[-2]) if len(closes) >= 2 and closes[-2] is not None else float(closes[-1])
        return ticker, live, prev
    except Exception:
        return ticker, None, None


@router.get("/price/{ticker}")
async def get_live_price_only(ticker: str, _: str = Depends(get_current_user)):
    ticker = ticker.upper()
    t, price, prev = await asyncio.to_thread(_fetch_live_price, ticker)
    if price is None:
        raise HTTPException(status_code=404, detail=f"No live price for {ticker}")
    daily_pct = round((price - prev) / prev * 100, 2) if prev and prev > 0 else None
    return {
        "ticker": t,
        "price": round(price, 2),
        "prev_close": round(prev, 2) if prev else None,
        "daily_change_pct": daily_pct,
    }


@router.get("/stocks")
async def get_stocks(db: Session = Depends(get_db), _: str = Depends(get_current_user)):
    stocks = db.query(Stock).all()
    if not stocks:
        return {"stocks": []}

    # fetch all live prices in parallel
    live_results = await asyncio.gather(
        *[asyncio.to_thread(_fetch_live_price, s.ticker) for s in stocks]
    )
    live_map = {ticker: (price, prev) for ticker, price, prev in live_results}

    result = []
    for s in stocks:
        snaps = (db.query(DailySnapshot)
                 .filter_by(ticker=s.ticker)
                 .order_by(DailySnapshot.date.asc())
                 .all())
        trend = _detect_current_trend(snaps) if snaps else {"direction": None, "duration_days": None}

        live_price, prev_close = live_map.get(s.ticker, (None, None))

        # fall back to DB if live fetch failed
        if live_price is None and snaps:
            live_price = snaps[-1].close
            prev_close = snaps[-2].close if len(snaps) >= 2 else None

        daily_pct = None
        if live_price is not None and prev_close and prev_close > 0:
            daily_pct = round((live_price - prev_close) / prev_close * 100, 2)

        result.append({
            "ticker": s.ticker,
            "company": s.company_name,
            "sector": s.sector,
            "price": round(live_price, 2) if live_price is not None else None,
            "daily_change_pct": daily_pct,
            "trend_direction": trend["direction"],
            "trend_age": trend["duration_days"],
        })
    return {"stocks": result}


@router.post("/analyze/{ticker}")
async def analyze_ticker(
    ticker: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    ticker = ticker.upper()

    try:
        hist, news, live_price, live_prev_close, api_company = await asyncio.to_thread(_fetch_yfinance, ticker)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    closes = hist["Close"]
    rsi_val = float(_rsi(closes).iloc[-1])
    regime = _market_regime(closes)

    await asyncio.to_thread(_upsert_snapshots, db, ticker, hist)

    snaps = (db.query(DailySnapshot)
             .filter_by(ticker=ticker)
             .order_by(DailySnapshot.date.asc())
             .all())
    trend = _detect_current_trend(snaps)

    vol_snaps = snaps[-4:] if len(snaps) >= 4 else snaps
    vols = [s.volume for s in vol_snaps]
    vol_declining = bool(len(vols) >= 3 and vols[-1] < vols[-2] < vols[-3])
    vol_decline_pct = 0.0
    if len(vols) >= 2 and vols[-2] > 0:
        vol_decline_pct = (vols[-2] - vols[-1]) / vols[-2] * 100

    latest = snaps[-1] if snaps else None
    near_upper = bool(latest and latest.upper_bb is not None and latest.close > latest.upper_bb * 0.98)
    near_lower = bool(latest and latest.lower_bb is not None and latest.close < latest.lower_bb * 1.02)

    sentiment = await asyncio.to_thread(_sentiment, news)
    mood = sentiment["label"]

    forecast_data = await asyncio.to_thread(_linear_forecast, closes)

    hazard = compute_hazard_rate(
        direction=trend["direction"], rsi=rsi_val,
        volume_declining=vol_declining,
        near_band=(near_upper if trend["direction"] == "up" else near_lower),
        sentiment_label=mood,
        trend_age=trend["duration_days"],
        regime=regime,
    )

    if trend["direction"] == "up":
        cox_covs = {"rsi": rsi_val, "volume_declining": int(vol_declining),
                    "near_upper_band": int(near_upper), "bull_regime": int(regime == "Bull")}
    else:
        cox_covs = {"rsi": rsi_val, "volume_declining": int(vol_declining),
                    "near_lower_band": int(near_lower), "bear_regime": int(regime == "Bear")}

    if _cox_models is not None:
        try:
            curve = cox_survival_curve(_cox_models, trend["direction"], cox_covs, trend["duration_days"])
            median_days = cox_median_survival(_cox_models, trend["direction"], cox_covs, trend["duration_days"])
            cd = {p["day"]: p["probability"] for p in curve}
            prob_3d = round(1 - cd[3], 4)
            prob_5d = round(1 - cd[5], 4)
            prob_7d = round(1 - cd[7], 4)
        except Exception:
            curve = survival_curve(hazard)
            median_days = median_survival_days(hazard)
            prob_3d = round(1 - (1 - hazard) ** 3, 4)
            prob_5d = round(1 - (1 - hazard) ** 5, 4)
            prob_7d = round(1 - (1 - hazard) ** 7, 4)
    else:
        curve = survival_curve(hazard)
        median_days = median_survival_days(hazard)
        prob_3d = round(1 - (1 - hazard) ** 3, 4)
        prob_5d = round(1 - (1 - hazard) ** 5, 4)
        prob_7d = round(1 - (1 - hazard) ** 7, 4)

    explanation = generate_explanation(
        direction=trend["direction"], rsi=rsi_val,
        volume_declining=vol_declining, vol_decline_pct=vol_decline_pct,
        near_upper_band=near_upper, near_lower_band=near_lower,
        sentiment_label=mood, trend_age=trend["duration_days"],
        hazard_rate=hazard,
    )

    current_price = live_price if live_price is not None else float(closes.iloc[-1])
    prev_price = live_prev_close if live_prev_close is not None else float(closes.iloc[-2])
    daily_change = current_price - prev_price
    daily_change_pct = daily_change / prev_price * 100 if prev_price else 0.0

    chart_json = await asyncio.to_thread(_build_chart_json, hist, forecast_data)
    if live_price is not None:
        chart_json["current_price"] = round(live_price, 2)
        chart_json["daily_change"] = round(daily_change, 2)
        chart_json["daily_pct"] = round(daily_change_pct, 2)

    ma7 = float(closes.tail(7).mean())
    ma30 = float(closes.tail(30).mean())
    score = 0
    if rsi_val < 30: score += 2
    elif rsi_val > 70: score -= 2
    if current_price > ma7 > ma30: score += 1
    elif current_price < ma7 < ma30: score -= 1
    if forecast_data["pct_change"] > 2: score += 1
    elif forecast_data["pct_change"] < -2: score -= 1
    signal = "Buy" if score >= 2 else "Sell" if score <= -2 else "Hold"

    db.add(Prediction(
        ticker=ticker,
        trend_direction=trend["direction"],
        predicted_duration_days=median_days,
        survival_prob_3d=prob_3d,
        survival_prob_5d=prob_5d,
        survival_prob_7d=prob_7d,
    ))
    db.commit()

    company = api_company or COMPANY_NAMES.get(ticker, ticker)
    return {
        "ticker": ticker,
        "company": company,
        "current_price": round(current_price, 2),
        "daily_change": round(daily_change, 2),
        "daily_change_pct": round(daily_change_pct, 2),
        "trend": trend,
        "traditional": {
            "signal": signal,
            "rsi": round(rsi_val, 1),
            "ma7": round(ma7, 2),
            "ma30": round(ma30, 2),
            "regime": regime,
            "forecast_pct": round(forecast_data["pct_change"], 2),
            "news_mood": mood,
        },
        "survival": {
            "curve": curve,
            "median_survival_days": median_days,
            "prob_reversal_3d": prob_3d,
            "prob_reversal_5d": prob_5d,
            "prob_reversal_7d": prob_7d,
        },
        "explanation": explanation,
        "chart_json": chart_json,
    }


@router.get("/trend-clock/{ticker}")
def trend_clock(
    ticker: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    ticker = ticker.upper()
    snaps = (db.query(DailySnapshot)
             .filter_by(ticker=ticker)
             .order_by(DailySnapshot.date.asc())
             .all())
    if not snaps:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    latest = snaps[-1]
    trend = _detect_current_trend(snaps)

    rsi_val = latest.rsi or 50.0
    vol_snaps = snaps[-4:] if len(snaps) >= 4 else snaps
    vols = [s.volume for s in vol_snaps]
    vol_declining = bool(len(vols) >= 3 and vols[-1] < vols[-2] < vols[-3])
    near_upper = latest.upper_bb is not None and latest.close > latest.upper_bb * 0.98
    near_lower = latest.lower_bb is not None and latest.close < latest.lower_bb * 1.02

    hazard = compute_hazard_rate(
        direction=trend["direction"], rsi=rsi_val,
        volume_declining=vol_declining,
        near_band=(near_upper if trend["direction"] == "up" else near_lower),
        sentiment_label="neutral",
        trend_age=trend["duration_days"],
        regime="Sideways",
    )

    current_age = trend["duration_days"]
    if trend["direction"] == "up":
        cox_covs = {"rsi": rsi_val, "volume_declining": int(vol_declining),
                    "near_upper_band": int(near_upper), "bull_regime": 0}
    else:
        cox_covs = {"rsi": rsi_val, "volume_declining": int(vol_declining),
                    "near_lower_band": int(near_lower), "bear_regime": 0}

    if _cox_models is not None:
        try:
            _curve = cox_survival_curve(_cox_models, trend["direction"], cox_covs, current_age)
            _cd = {p["day"]: p["probability"] for p in _curve}
            return {
                "ticker": ticker,
                "trend": trend,
                "prob_reversal_3d": round(1 - _cd[3], 4),
                "prob_reversal_5d": round(1 - _cd[5], 4),
                "prob_reversal_7d": round(1 - _cd[7], 4),
                "median_survival_days": cox_median_survival(
                    _cox_models, trend["direction"], cox_covs, current_age
                ),
            }
        except Exception:
            pass

    return {
        "ticker": ticker,
        "trend": trend,
        "prob_reversal_3d": round(1 - (1 - hazard) ** 3, 4),
        "prob_reversal_5d": round(1 - (1 - hazard) ** 5, 4),
        "prob_reversal_7d": round(1 - (1 - hazard) ** 7, 4),
        "median_survival_days": median_survival_days(hazard),
    }


@router.get("/history/{ticker}")
def prediction_history(
    ticker: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    ticker = ticker.upper()
    preds = (db.query(Prediction)
             .filter_by(ticker=ticker)
             .order_by(Prediction.predicted_at.desc())
             .limit(20)
             .all())
    return {
        "ticker": ticker,
        "predictions": [
            {
                "predicted_at": str(p.predicted_at),
                "trend_direction": p.trend_direction,
                "predicted_duration_days": p.predicted_duration_days,
                "survival_prob_3d": p.survival_prob_3d,
                "survival_prob_5d": p.survival_prob_5d,
                "survival_prob_7d": p.survival_prob_7d,
                "was_correct": p.was_correct,
            }
            for p in preds
        ],
    }


@router.get("/snapshot/{ticker}")
def get_snapshot(
    ticker: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    ticker = ticker.upper()
    snap = (db.query(DailySnapshot)
            .filter_by(ticker=ticker)
            .order_by(DailySnapshot.date.desc())
            .first())
    if snap:
        return {
            "ticker": ticker, "date": snap.date,
            "open": snap.open, "high": snap.high,
            "low": snap.low, "close": snap.close,
            "volume": snap.volume, "rsi": snap.rsi,
            "ma7": snap.ma7, "ma30": snap.ma30,
            "upper_bb": snap.upper_bb, "lower_bb": snap.lower_bb,
            "source": "cache",
        }
    try:
        hist, _, _lp, _pc, _cn = _fetch_yfinance(ticker)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    closes = hist["Close"]
    rsi_s = _rsi(closes)
    _, ubb, lbb = _bollinger(closes)
    return {
        "ticker": ticker,
        "date": hist.index[-1].date().isoformat(),
        "open": round(float(hist["Open"].iloc[-1]), 4),
        "high": round(float(hist["High"].iloc[-1]), 4),
        "low": round(float(hist["Low"].iloc[-1]), 4),
        "close": round(float(hist["Close"].iloc[-1]), 4),
        "volume": int(hist["Volume"].iloc[-1]),
        "rsi": round(float(rsi_s.iloc[-1]), 2),
        "ma7": round(float(closes.tail(7).mean()), 4),
        "ma30": round(float(closes.tail(30).mean()), 4),
        "upper_bb": round(float(ubb.iloc[-1]), 4),
        "lower_bb": round(float(lbb.iloc[-1]), 4),
        "source": "live",
    }


class AddStockBody(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None


_TICKER_DENYLIST = frozenset({
    # currencies / fx
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "HKD", "CNY", "INR",
    "NZD", "SEK", "NOK", "DKK", "SGD", "MXN", "BRL", "ZAR", "RUB",
    # market / exchange terms
    "ETF", "NYSE", "AMEX", "TSX", "SEC", "FINRA", "OTC", "ADR", "REIT",
    # financial metrics
    "EPS", "PE", "PB", "PEG", "ROE", "ROA", "YTD", "MTD", "QTD", "NAV",
    "AUM", "FCF", "EBIT", "IRR", "NPV", "DCF", "CAGR", "GAAP",
    # titles / org roles
    "CEO", "CFO", "COO", "CTO", "CIO", "VP", "SVP", "EVP", "MD", "GM",
    # legal suffixes
    "INC", "LLC", "LTD", "CORP", "CO", "PLC", "AG", "SA", "NV", "BV", "LP",
    # time / calendar
    "JAN", "FEB", "MAR", "APR", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
    "DEC", "QTR", "YR", "FY", "TTM",
    # common words from report documents
    "PDF", "CSV", "XLS", "HTML", "JSON", "US", "UK", "EU", "UN",
    "NO", "NA", "N", "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "AT", "BY", "IN", "IT", "IS", "TO", "OR", "AND", "THE", "FOR",
    "NOT", "WITH", "FROM", "NET", "NEW", "ALL", "ANY", "AS",
    # common financial doc words
    "TOTAL", "CASH", "FUND", "BOND", "STOCK", "SHARE", "PRICE", "VALUE",
    "RATE", "RETURN", "GAIN", "LOSS", "TAX", "FEE", "COST", "RISK",
    "DATE", "NOTE", "PAGE", "REF", "TYPE", "CODE", "PLAN",
})


def _extract_tickers_from_text(text: str) -> list:
    found = set()

    # high-confidence patterns first (explicit ticker notation)
    for pattern in [
        r'\$([A-Z]{1,5})\b',
        r'\b(?:NASDAQ|NYSE|AMEX|TSX):\s*([A-Z]{1,5})\b',
        r'\bTicker(?:\s+Symbol)?[:\s]+([A-Z]{1,5})\b',
        r'\bSymbol[:\s]+([A-Z]{1,5})\b',
        r'\(([A-Z]{1,5})\)',
    ]:
        for m in re.finditer(pattern, text):
            found.add(m.group(1))

    # broader fallback - any 1-5 uppercase word, relies on denylist + yfinance to filter noise
    for m in re.finditer(r'\b([A-Z]{1,5})\b', text):
        candidate = m.group(1)
        if len(candidate) >= 2 and candidate not in _TICKER_DENYLIST:
            found.add(candidate)

    return sorted(found)[:40]  # cap at 40 candidates to keep fetch time reasonable


@router.post("/stocks")
async def add_stock(
    body: AddStockBody,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    ticker = body.ticker.upper().strip()
    if not ticker or not ticker.isalpha() or len(ticker) > 5:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    existing = db.query(Stock).filter_by(ticker=ticker).first()
    if existing:
        return {"ticker": ticker, "status": "already_exists", "company": existing.company_name}

    try:
        hist, _, _lp, _pc, _cn = await asyncio.to_thread(_fetch_yfinance, ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for {ticker}: {e}")

    company = body.company_name or COMPANY_NAMES.get(ticker, ticker)
    sector = body.sector or "Unknown"
    db.add(Stock(ticker=ticker, company_name=company, sector=sector))
    db.commit()
    await asyncio.to_thread(_upsert_snapshots, db, ticker, hist)
    return {"ticker": ticker, "status": "added", "company": company}


@router.delete("/stocks/{ticker}")
def delete_stock(
    ticker: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    ticker = ticker.upper()
    stock = db.query(Stock).filter_by(ticker=ticker).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{ticker} not found")
    db.query(DailySnapshot).filter_by(ticker=ticker).delete()
    db.query(TrendRecord).filter_by(ticker=ticker).delete()
    db.query(Prediction).filter_by(ticker=ticker).delete()
    db.delete(stock)
    db.commit()
    return {"ticker": ticker, "status": "deleted"}


@router.post("/stocks/from-pdf")
async def add_stocks_from_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF too large (max 10 MB)")

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse PDF")

    tickers = _extract_tickers_from_text(text)
    if not tickers:
        raise HTTPException(status_code=422, detail="No stock tickers found in PDF")

    results = []
    for ticker in tickers:
        existing = db.query(Stock).filter_by(ticker=ticker).first()
        if existing:
            results.append({"ticker": ticker, "status": "already_exists"})
            continue
        resolved = ticker
        company = COMPANY_NAMES.get(ticker, ticker)
        hist = None
        try:
            hist, _, _lp, _pc, _cn = await asyncio.to_thread(_fetch_yfinance, ticker)
        except Exception:
            try:
                resolved, hist, _, company, _lp, _pc = await asyncio.to_thread(_resolve_ticker_sync, ticker)
                existing_r = db.query(Stock).filter_by(ticker=resolved).first()
                if existing_r:
                    results.append({"ticker": resolved, "status": "already_exists", "resolved_from": ticker})
                    continue
            except Exception:
                results.append({"ticker": ticker, "status": "failed"})
                continue

        db.add(Stock(ticker=resolved, company_name=company, sector="Unknown"))
        db.commit()
        await asyncio.to_thread(_upsert_snapshots, db, resolved, hist)
        entry: dict = {"ticker": resolved, "status": "added"}
        if resolved != ticker:
            entry["resolved_from"] = ticker
        results.append(entry)

    return {"results": results}


class StockChatMessage(BaseModel):
    role: str
    content: str


class StockChatRequest(BaseModel):
    ticker: str
    messages: list[StockChatMessage]
    stock_context: Optional[dict] = None


@router.post("/chat")
async def stock_demo_chat(
    body: StockChatRequest,
    _: str = Depends(get_current_user),
):
    from routers.chat import _stream_from_openrouter, _stream_from_ollama, _is_ollama_model, DEFAULT_MODEL

    ticker = body.ticker.upper()
    ctx = body.stock_context or {}
    company = ctx.get("company", ticker)

    system_lines = [
        f"You are a sharp, concise financial analysis assistant. The user is viewing live analysis for {ticker} ({company}).",
        "",
        f"LIVE ANALYSIS DATA — {ticker}:",
    ]

    if ctx.get("current_price") is not None:
        chg = ctx.get("daily_change_pct") or 0
        sign = "+" if chg >= 0 else ""
        system_lines.append(f"  Price: ${ctx['current_price']} ({sign}{chg:.2f}% today)")

    trend = ctx.get("trend") or {}
    if trend:
        system_lines.append(
            f"  Trend: {trend.get('direction', '?')} for {trend.get('duration_days', '?')} days"
            f" · started ${trend.get('start_price', '?')}"
        )

    trad = ctx.get("traditional") or {}
    if trad:
        fc = trad.get("forecast_pct") or 0
        system_lines.append(
            f"  Signal: {trad.get('signal')} · RSI(14): {trad.get('rsi')} · Regime: {trad.get('regime')}"
            f" · 7d forecast: {fc:+.2f}%"
        )

    surv = ctx.get("survival") or {}
    if surv:
        p3 = (surv.get("prob_reversal_3d") or 0) * 100
        p5 = (surv.get("prob_reversal_5d") or 0) * 100
        p7 = (surv.get("prob_reversal_7d") or 0) * 100
        system_lines.append(
            f"  Trend survival: median {surv.get('median_survival_days', '?')}d remaining"
            f" · reversal risk 3d={p3:.0f}% / 5d={p5:.0f}% / 7d={p7:.0f}%"
        )

    if ctx.get("explanation"):
        system_lines.append(f"  Note: {ctx['explanation']}")

    system_lines += [
        "",
        "Rules: be concise and data-driven. Cite actual numbers. Under 120 words unless asked for more.",
    ]

    messages = [{"role": "system", "content": "\n".join(system_lines)}]
    messages += [{"role": m.role, "content": m.content} for m in body.messages]

    async def stream():
        streamer = _stream_from_ollama if _is_ollama_model(DEFAULT_MODEL) else _stream_from_openrouter
        try:
            async for token in streamer(DEFAULT_MODEL, messages):
                yield token
        except RuntimeError as e:
            yield str(e)

    return StreamingResponse(stream(), media_type="text/plain")

# run this once to load stock data into the database: python seed.py
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database import engine, Base, SessionLocal
from models import Stock, DailySnapshot, TrendRecord

Base.metadata.create_all(bind=engine)

STOCKS = [
    ("AAPL", "Apple Inc.", "Technology"),
    ("TSLA", "Tesla Inc.", "Automotive"),
    ("NVDA", "NVIDIA Corp.", "Technology"),
    ("MSFT", "Microsoft Corp.", "Technology"),
    ("GOOGL", "Alphabet Inc.", "Technology"),
    ("META", "Meta Platforms", "Technology"),
    ("AMZN", "Amazon.com Inc.", "Consumer"),
    ("NFLX", "Netflix Inc.", "Media"),
    ("AMD", "AMD Inc.", "Technology"),
    ("SPY", "S&P 500 ETF", "Index"),
]


def _rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def _bollinger(prices, window=20):
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return mid + 2 * std, mid - 2 * std


def _detect_trends(dates, closes):
    s = pd.Series(closes)
    ma5 = s.rolling(5).mean()
    slopes = ma5.diff().fillna(0)

    trends = []
    if len(slopes) < 6:
        return trends

    direction = "up" if slopes.iloc[5] >= 0 else "down"
    start_i = 0

    for i in range(6, len(slopes)):
        day_dir = "up" if slopes.iloc[i] >= 0 else "down"
        if day_dir != direction:
            duration = i - start_i
            if duration >= 3:
                trends.append({
                    "direction": direction,
                    "started_at": dates[start_i],
                    "ended_at": dates[i - 1],
                    "start_price": closes[start_i],
                    "end_price": closes[i - 1],
                    "duration_days": duration,
                })
            direction = day_dir
            start_i = i

    return trends


def seed_ticker(db: Session, ticker: str, company: str, sector: str):
    import yfinance as yf

    print(f"  {ticker}...", end=" ", flush=True)

    if not db.query(Stock).filter_by(ticker=ticker).first():
        db.add(Stock(ticker=ticker, company_name=company, sector=sector))
        db.commit()

    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty:
        print("SKIP (no data)")
        return

    closes = hist["Close"]
    rsi_s = _rsi(closes)
    ma7_s = closes.rolling(7).mean()
    ma30_s = closes.rolling(30).mean()
    ubb_s, lbb_s = _bollinger(closes)

    snap_count = 0
    for i, (idx, row) in enumerate(hist.iterrows()):
        d = idx.date().isoformat()
        if db.query(DailySnapshot).filter_by(ticker=ticker, date=d).first():
            continue

        def safe(val):
            v = float(val)
            return None if (v != v) else round(v, 4)

        db.add(DailySnapshot(
            ticker=ticker, date=d,
            open=round(float(row["Open"]), 4),
            high=round(float(row["High"]), 4),
            low=round(float(row["Low"]), 4),
            close=round(float(row["Close"]), 4),
            volume=int(row["Volume"]),
            rsi=safe(rsi_s.iloc[i]),
            ma7=safe(ma7_s.iloc[i]),
            ma30=safe(ma30_s.iloc[i]),
            upper_bb=safe(ubb_s.iloc[i]),
            lower_bb=safe(lbb_s.iloc[i]),
        ))
        snap_count += 1

    db.commit()

    dates_list = [idx.date().isoformat() for idx in hist.index]
    closes_list = [float(c) for c in closes.values]
    trends = _detect_trends(dates_list, closes_list)

    trend_count = 0
    for t in trends:
        if db.query(TrendRecord).filter_by(ticker=ticker, started_at=t["started_at"]).first():
            continue
        db.add(TrendRecord(
            ticker=ticker,
            direction=t["direction"],
            started_at=t["started_at"],
            ended_at=t["ended_at"],
            start_price=t["start_price"],
            end_price=t["end_price"],
            duration_days=t["duration_days"],
            reversal_confirmed=True,
        ))
        trend_count += 1

    db.commit()
    print(f"{snap_count} snapshots, {trend_count} trends")


def main():
    db = SessionLocal()
    try:
        print("Seeding stock data (this takes ~60s)...")
        for ticker, company, sector in STOCKS:
            seed_ticker(db, ticker, company, sector)
        print("\nDone. Start the app: uvicorn main:app --reload --port 8080")
    finally:
        db.close()


if __name__ == "__main__":
    main()

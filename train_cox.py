#!/usr/bin/env python3
# Trains Cox proportional hazard models on yfinance data and saves cox_model.pkl.
# Usage: python train_cox.py [TICKER TICKER ...]
# If no tickers are provided, a default list is used.
import sys

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "NFLX", "SPY",
    "JPM", "BAC", "XOM", "JNJ", "PG",
    "KO", "WMT", "DIS", "BA", "GS",
]


def main():
    from routers.cox_survival import fit_cox, save_model

    tickers = sys.argv[1:] or DEFAULT_TICKERS
    print(f"Training Cox survival models on {len(tickers)} tickers (5y history each)...\n")
    models = fit_cox(tickers, period="5y")
    save_model(models)
    print("\nDone. Start the server and the Cox model will be loaded automatically.")


if __name__ == "__main__":
    main()

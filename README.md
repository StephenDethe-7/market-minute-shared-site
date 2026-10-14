# Market Minute — shared website

A Streamlit stock research and paper-trading practice app. The **Practice bot** only records pretend trades; it does not connect to a brokerage or place real orders.

## Share the app with a friend

This folder is a clean deployment copy. It intentionally excludes the local database, CSV exports, logs, and Python virtual environment.

1. Put this folder in a **private GitHub repository** so the source is not published openly.
2. Add your friend to the repository with write access if you both want to edit the app. Give them viewer access to the hosted app if they only need to use it.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io), choose **Create app**, then select the repository, branch, and `app.py` entry point.
4. Choose private sharing for the app and invite your friend. After deployment, share the generated `streamlit.app` link with them.
5. If you want Finnhub news or live quotes on the hosted app, add a newly rotated key in the app's Cloud secrets. Do not put the key in this repository.

For a Finnhub secret, use this format in the hosting dashboard:

```toml
FINNHUB_API_KEY = "your-new-key"
```

The free Yahoo Finance source works without a key and may be delayed or incomplete.

## Important shared-data limit

The app creates `market_minute.sqlite` beside `app.py`. Streamlit Community Cloud does not guarantee that files written by the app will survive restarts, so saved scans and practice-bot history may disappear. The app is suitable for trying the shared site, but durable shared history needs an external database before you rely on it.

The automatic practice bot checks candles while the app session is open. It uses estimated costs and next-candle paper fills; it is not a live execution system, and its results do not establish future profitability.

## App files

- `app.py` — Streamlit screens and practice bot
- `research_engine.py` — market screening and paper-research storage
- `requirements.txt` — pinned Python dependencies

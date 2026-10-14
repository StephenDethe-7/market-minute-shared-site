from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import quote
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import websockets
import research_engine as research


st.set_page_config(page_title="Market Minute", page_icon="📈", layout="wide")
st.title("Market Minute")
st.caption("Intraday stock research, live quotes, broad market screening, and daily paper-review tracking.")
st.warning("For research, not trading instructions. Historical ranges are comparisons, not guaranteed forecasts. Data may be delayed or incomplete.")

DEFAULT_WATCHLIST = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, SPY, QQQ, AVGO, JPM, XOM, WMT, COST, LLY, NFLX, ORCL, PLTR, INTC, COIN, SMCI, BA, DIS, TSM"
DATABASE_PATH = Path(os.environ.get("MARKET_MINUTE_DB", Path(__file__).with_name("market_minute.sqlite")))


def _set_live_status(manager: dict, status: str) -> None:
    with manager["lock"]:
        manager["status"] = status


async def _run_finnhub_stream(manager: dict, api_key: str) -> None:
    url = f"wss://ws.finnhub.io?token={quote(api_key, safe='')}"
    try:
        async with websockets.connect(url, ping_interval=20, close_timeout=3) as ws:
            for symbol in manager["symbols"]:
                await ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
            _set_live_status(manager, "Connected · waiting for trades")
            while not manager["stop"].is_set():
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1)
                except asyncio.TimeoutError:
                    continue
                payload = json.loads(message)
                if payload.get("type") == "error":
                    _set_live_status(manager, "Finnhub stream error · check key and account access")
                    continue
                if payload.get("type") != "trade":
                    continue
                with manager["lock"]:
                    for trade in payload.get("data", []):
                        symbol = trade.get("s")
                        price = trade.get("p")
                        timestamp = trade.get("t")
                        if symbol not in manager["ticks"] or price is None or timestamp is None:
                            continue
                        ticks = manager["ticks"][symbol]
                        ticks.append({"time": int(timestamp), "price": float(price), "volume": float(trade.get("v") or 0)})
                        cutoff = int(timestamp) - 20 * 60 * 1000
                        while ticks and ticks[0]["time"] < cutoff:
                            ticks.pop(0)
                    manager["status"] = "Live · receiving trades"
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            _set_live_status(manager, "Finnhub denied live streaming for this key or account plan.")
        else:
            _set_live_status(manager, f"Live connection stopped · {type(exc).__name__}")
    finally:
        if manager["stop"].is_set():
            _set_live_status(manager, "Stopped")


def start_live_stream(api_key: str, symbols: list[str]) -> dict:
    old = st.session_state.get("live_stream")
    if old:
        old["stop"].set()
        if old.get("thread") and old["thread"].is_alive():
            old["thread"].join(timeout=2)
    manager = {"symbols": symbols, "ticks": {symbol: [] for symbol in symbols},
               "status": "Connecting to Finnhub…", "stop": threading.Event(), "lock": threading.Lock()}

    def worker() -> None:
        asyncio.run(_run_finnhub_stream(manager, api_key))

    manager["thread"] = threading.Thread(target=worker, name="finnhub-live-watchlist", daemon=True)
    st.session_state.live_stream = manager
    manager["thread"].start()
    return manager


def live_table(manager: dict) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    rows = []
    with manager["lock"]:
        for symbol in manager["symbols"]:
            ticks = list(manager["ticks"].get(symbol, []))
            if not ticks:
                rows.append({"Ticker": symbol, "Live price": np.nan, "1-minute move": np.nan,
                             "5-minute move": np.nan, "15-minute move": np.nan,
                             "Last trade": "Waiting for first trade"})
                continue
            latest = ticks[-1]
            row = {"Ticker": symbol, "Live price": latest["price"],
                   "Last trade": datetime.fromtimestamp(latest["time"] / 1000, timezone.utc).astimezone().strftime("%I:%M:%S %p %Z")}
            for seconds, label in ((60, "1-minute move"), (300, "5-minute move"), (900, "15-minute move")):
                window_start = latest["time"] - seconds * 1000
                prior = next((tick for tick in ticks if tick["time"] <= window_start), None)
                if prior is None or prior["price"] == 0:
                    row[label] = np.nan
                else:
                    row[label] = (latest["price"] / prior["price"] - 1) * 100
            rows.append(row)
    return pd.DataFrame(rows)


def start_market_research(state: dict) -> None:
    with state["lock"]:
        if state["running"]:
            return
        state["running"] = True
        state["status"] = "Preparing market universe…"
        state["progress"] = (0, 1, "")
        state["error"] = ""
        state["source_errors"] = []

    def update_progress(done: int, total: int, symbol: str) -> None:
        with state["lock"]:
            state["progress"] = (done, total, symbol)
            state["status"] = f"Reviewing {symbol} ({done}/{total})"

    def worker() -> None:
        try:
            run_at, _, flags, errors = research.run_market_research(progress=update_progress)
            with state["lock"]:
                state["last_run"] = run_at
                state["flag_count"] = len(flags)
                state["source_errors"] = errors
                state["status"] = "Scan complete"
        except Exception as exc:
            with state["lock"]:
                state["error"] = f"{type(exc).__name__}: {exc}"
                state["status"] = "Scan failed"
        finally:
            with state["lock"]:
                state["running"] = False

    state["thread"] = threading.Thread(target=worker, name="daily-market-research", daemon=True)
    state["thread"].start()


def clean_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        fields = {"open", "high", "low", "close", "volume"}
        field_level = next((level for level in range(raw.columns.nlevels)
                            if fields.issubset({str(x).lower() for x in raw.columns.get_level_values(level)})), None)
        if field_level is None:
            return pd.DataFrame()
        raw.columns = raw.columns.get_level_values(field_level)
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(raw.columns):
        return pd.DataFrame()
    df = raw[required].apply(pd.to_numeric, errors="coerce").dropna(subset=["close"])
    try:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return pd.DataFrame()
    df = df.loc[~df.index.isna()]
    if df.empty:
        return pd.DataFrame()
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    return df


def save_bars(symbol: str, interval: str, provider: str, bars: pd.DataFrame) -> None:
    if bars.empty:
        return
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS intraday_bars (
            provider TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            timestamp TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (provider, symbol, interval, timestamp)
        )""")
        records = [(provider, symbol, interval, idx.tz_convert("UTC").isoformat(),
                    float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume))
                   for idx, row in bars.iterrows()]
        conn.executemany("INSERT OR REPLACE INTO intraday_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        conn.execute("DELETE FROM intraday_bars WHERE timestamp < ?",
                     ((datetime.now(timezone.utc) - timedelta(days=120)).isoformat(),))


def read_saved_bars(symbol: str, interval: str, provider: str) -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            bars = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume FROM intraday_bars WHERE provider=? AND symbol=? AND interval=? ORDER BY timestamp",
                conn, params=(provider, symbol, interval),
            )
    except sqlite3.OperationalError:
        return pd.DataFrame()
    if bars.empty:
        return bars
    bars.index = pd.to_datetime(bars.pop("timestamp"), utc=True)
    return bars


@st.cache_data(ttl=30, max_entries=500, show_spinner=False)
def load_bars(symbol: str, interval: str, provider: str, api_key: str = "") -> pd.DataFrame:
    try:
        if provider == "Finnhub" and api_key:
            now = int(time.time())
            days = 5 if interval == "1m" else 30
            response = requests.get(
                "https://finnhub.io/api/v1/stock/candle",
                params={"symbol": symbol, "resolution": interval[:-1], "from": now - days * 86400, "to": now},
                headers={"X-Finnhub-Token": api_key}, timeout=20,
            )
            if response.status_code == 401:
                raise ValueError("Finnhub rejected the API key. Check the key in Settings.")
            if response.status_code == 429:
                raise ValueError("The Finnhub request limit was reached. Wait a minute and try again.")
            if response.status_code == 403:
                raise ValueError("Finnhub denied this request. Check that your plan allows intraday candles for this symbol.")
            response.raise_for_status()
            payload = response.json()
            fetched = pd.DataFrame()
            if payload.get("s") == "ok":
                fetched = clean_history(pd.DataFrame(
                    {"open": payload["o"], "high": payload["h"], "low": payload["l"],
                     "close": payload["c"], "volume": payload["v"]},
                    index=pd.to_datetime(payload["t"], unit="s", utc=True)))
            elif payload.get("s") == "no_data":
                raise ValueError("Finnhub returned no candles for this ticker and interval. Check the symbol, market hours, or plan access.")
            elif payload.get("s") == "error":
                raise ValueError(f"Finnhub candle request failed: {payload.get('error', 'provider error')}")
        else:
            period = {"1m": "5d", "5m": "30d", "15m": "60d"}.get(interval, "30d")
            fetched = clean_history(yf.download(symbol, period=period, interval=interval, auto_adjust=True,
                                                progress=False, prepost=False, threads=False))
    except ValueError:
        raise
    except Exception:
        saved = read_saved_bars(symbol, interval, provider)
        if not saved.empty:
            return saved
        raise

    if not fetched.empty:
        save_bars(symbol, interval, provider, fetched)
    saved = read_saved_bars(symbol, interval, provider)
    if saved.empty:
        return fetched
    if fetched.empty:
        return saved
    return pd.concat([saved, fetched]).loc[lambda frame: ~frame.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=300, max_entries=100, show_spinner=False)
def load_news(api_key: str, symbol: str = "") -> list[dict]:
    today = datetime.now(timezone.utc).date()
    headers = {"X-Finnhub-Token": api_key}
    if symbol:
        params = {"symbol": symbol, "from": (today - timedelta(days=7)).isoformat(), "to": today.isoformat()}
        url = "https://finnhub.io/api/v1/company-news"
    else:
        params = {"category": "general"}
        url = "https://finnhub.io/api/v1/news"
    response = requests.get(url, params=params, headers=headers, timeout=20)
    if response.status_code == 401:
        raise ValueError("Finnhub rejected the API key. Check the key in Settings.")
    if response.status_code == 429:
        raise ValueError("The Finnhub request limit was reached. Wait a minute and try again.")
    response.raise_for_status()
    return response.json()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Be defensive here too: cached/provider frames must always use real timestamps
    # before building day-based VWAP groups or displaying the latest bar time.
    d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
    d = d.loc[~d.index.isna()].sort_index()
    d = d.loc[~d.index.duplicated(keep="last")]
    if d.empty:
        return d
    c = d["close"]
    d["ema9"] = c.ewm(span=9, adjust=False).mean()
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    typical = (d["high"] + d["low"] + c) / 3
    session = pd.Series(d.index.date, index=d.index)
    pv = typical * d["volume"].fillna(0)
    d["vwap"] = pv.groupby(session).cumsum() / d["volume"].fillna(0).groupby(session).cumsum().replace(0, np.nan)
    d["vol_avg"] = d["volume"].rolling(20, min_periods=5).mean()
    d["vol_ratio"] = d["volume"] / d["vol_avg"].replace(0, np.nan)
    d["fwd"] = c.shift(-1) / c - 1
    # Keep a compact current-condition label for comparisons with prior bars.
    d["trend"] = np.where(d["ema9"] >= d["ema20"], "up", "down")
    d["rsi_band"] = pd.cut(d["rsi"], [0, 35, 50, 65, 100], labels=["weak", "soft", "firm", "hot"], include_lowest=True)
    return d


def outlook(d: pd.DataFrame, bars: int) -> dict:
    c = d["close"]
    fwd = c.shift(-bars) / c - 1
    current = d.iloc[-1]
    # Compare past bars with similar short-term trend and momentum conditions.
    eligible = fwd.notna() & d["rsi_band"].notna()
    similar = d.loc[eligible & (d["trend"] == current["trend"]) & (d["rsi_band"] == current["rsi_band"]), :].copy()
    returns = fwd.loc[similar.index].dropna()
    if len(returns) < 5:
        returns = fwd.loc[eligible].dropna()
        basis = "all available bars (too few similar examples)"
    else:
        basis = "past bars with a similar trend and RSI range"
    if returns.empty:
        return {"n": 0, "up_count": 0, "up": np.nan, "median": np.nan, "low": np.nan, "high": np.nan, "basis": basis}
    up_count = int((returns > 0).sum())
    return {"n": int(len(returns)), "up_count": up_count, "up": float(up_count / len(returns)),
            "median": float(returns.median()), "low": float(returns.quantile(.2)),
            "high": float(returns.quantile(.8)), "basis": basis}


def outlook_with_market(d: pd.DataFrame, market: pd.DataFrame | None, bars: int) -> dict:
    estimate = outlook(d, bars)
    if market is None or market.empty:
        estimate["market_status"] = "SPY data unavailable · showing stock-only history"
        estimate["market_note"] = "The broad-market feed could not be loaded, so this rate does not include market conditions."
        return estimate

    features = market[["close", "ema9", "ema20", "vwap"]].rename(
        columns={"close": "spy_close", "ema9": "spy_ema9", "ema20": "spy_ema20", "vwap": "spy_vwap"}
    ).copy()
    features["spy_trend"] = np.where(features["spy_ema9"] >= features["spy_ema20"], "up", "down")
    features["spy_vwap_state"] = np.where(
        features["spy_vwap"].isna(), "unknown",
        np.where(features["spy_close"] >= features["spy_vwap"], "above", "below")
    )
    aligned = d.join(features, how="inner")
    if aligned.empty:
        estimate["market_status"] = "SPY bars do not line up · showing stock-only history"
        estimate["market_note"] = "The stock and SPY feeds had no matching timestamps."
        return estimate

    current = aligned.iloc[-1]
    market_direction = "upward" if current["spy_trend"] == "up" else "downward"
    vwap_text = f"{current['spy_vwap_state']} VWAP" if current["spy_vwap_state"] != "unknown" else "VWAP unavailable"
    market_status = f"SPY {market_direction} momentum · {vwap_text} · bar {aligned.index[-1].strftime('%I:%M %p %Z')}"
    eligible = aligned["fwd"].notna() & aligned["rsi_band"].notna()
    same_conditions = eligible & (aligned["trend"] == current["trend"]) & \
        (aligned["rsi_band"] == current["rsi_band"]) & \
        (aligned["spy_trend"] == current["spy_trend"]) & \
        (aligned["spy_vwap_state"] == current["spy_vwap_state"])
    returns = aligned.loc[same_conditions, "fwd"].dropna()
    if len(returns) < 10:
        estimate["market_status"] = market_status
        estimate["market_note"] = f"Only {len(returns)} matching stock-and-market examples; the displayed rate uses stock history without market matching."
        return estimate

    up_count = int((returns > 0).sum())
    estimate.update({
        "n": int(len(returns)), "up_count": up_count, "up": float(up_count / len(returns)),
        "median": float(returns.median()), "low": float(returns.quantile(.2)),
        "high": float(returns.quantile(.8)),
        "basis": "past bars with similar stock trend/RSI and matching SPY trend/VWAP",
        "market_status": market_status,
        "market_note": "Matched the stock's trend and RSI range plus SPY's trend and position versus VWAP.",
    })
    return estimate


def ema_crossover_paper_test(
    bars: pd.DataFrame,
    short_period: int,
    long_period: int,
    units: float,
    cost_bps_per_side: float,
    starting_cash: float = 10_000.0,
) -> tuple[dict, pd.DataFrame]:
    """Backtest a long-only EMA crossover with next-bar-open paper fills."""
    data = bars[["open", "close"]].copy().dropna()
    data["ema_short"] = data["close"].ewm(span=short_period, adjust=False, min_periods=long_period).mean()
    data["ema_long"] = data["close"].ewm(span=long_period, adjust=False, min_periods=long_period).mean()
    data = data.dropna(subset=["ema_short", "ema_long"])
    if len(data) < 3:
        return {"error": "There are not enough candles for these EMA periods."}, pd.DataFrame()

    cross_up = (data["ema_short"] > data["ema_long"]) & \
        (data["ema_short"].shift(1) <= data["ema_long"].shift(1))
    cross_down = (data["ema_short"] < data["ema_long"]) & \
        (data["ema_short"].shift(1) >= data["ema_long"].shift(1))
    cost_rate = max(float(cost_bps_per_side), 0.0) / 10_000
    trades: list[dict] = []
    position: dict | None = None

    def close_position(exit_index: int, reason: str) -> None:
        nonlocal position
        if position is None:
            return
        raw_exit = float(data["open"].iloc[exit_index]) if reason == "EMA cross down" else float(data["close"].iloc[exit_index])
        entry_fill = position["entry_price"] * (1 + cost_rate)
        exit_fill = raw_exit * (1 - cost_rate)
        pnl = (exit_fill - entry_fill) * units
        trades.append({
            "Entry time": position["entry_time"], "Entry price": position["entry_price"],
            "Exit time": data.index[exit_index], "Exit price": raw_exit,
            "Exit reason": reason, "Shares": units,
            "Net P&L": pnl, "Net return %": (exit_fill / entry_fill - 1) * 100,
        })
        position = None

    # A signal is known only after its candle closes; simulate the fill at the next candle's open.
    for signal_index in range(1, len(data) - 1):
        fill_index = signal_index + 1
        if position is None and bool(cross_up.iloc[signal_index]):
            position = {"entry_time": data.index[fill_index], "entry_price": float(data["open"].iloc[fill_index])}
        elif position is not None and bool(cross_down.iloc[signal_index]):
            close_position(fill_index, "EMA cross down")
    if position is not None:
        close_position(len(data) - 1, "End of available data")

    trades_df = pd.DataFrame(trades)
    total_pnl = float(trades_df["Net P&L"].sum()) if not trades_df.empty else 0.0
    wins = int((trades_df["Net P&L"] > 0).sum()) if not trades_df.empty else 0
    equity = starting_cash + trades_df["Net P&L"].cumsum() if not trades_df.empty else pd.Series(dtype=float)
    drawdown_pct = float(((equity / equity.cummax()) - 1).min() * 100) if not equity.empty else 0.0
    benchmark_entry = float(data["open"].iloc[0])
    benchmark_exit = float(data["close"].iloc[-1])
    benchmark_net = ((benchmark_exit * (1 - cost_rate)) / (benchmark_entry * (1 + cost_rate)) - 1) * 100
    latest = data.iloc[-1]
    summary = {
        "trade_count": len(trades_df), "wins": wins,
        "win_rate": (wins / len(trades_df)) if not trades_df.empty else None,
        "net_pnl": total_pnl, "strategy_return_pct": total_pnl / starting_cash * 100,
        "max_drawdown_pct": drawdown_pct, "buy_hold_return_pct": benchmark_net,
        "current_alignment": "Fast EMA above slow EMA" if latest["ema_short"] > latest["ema_long"] else "Fast EMA below slow EMA",
        "latest_price": float(latest["close"]),
        "latest_short_ema": float(latest["ema_short"]), "latest_long_ema": float(latest["ema_long"]),
        "bars_used": len(data), "first_bar": data.index[0], "last_bar": data.index[-1],
    }
    return summary, trades_df


def init_forward_paper_db() -> None:
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS ema_forward_paper_runs (
            run_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            provider TEXT NOT NULL, short_period INTEGER NOT NULL, long_period INTEGER NOT NULL,
            units REAL NOT NULL, starting_cash REAL NOT NULL, cost_bps REAL NOT NULL,
            running INTEGER NOT NULL, last_signal_bar TEXT, position_entry_time TEXT,
            position_entry_price REAL, realized_pnl REAL NOT NULL DEFAULT 0,
            latest_price REAL, status TEXT, started_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS ema_forward_paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            entry_time TEXT NOT NULL, entry_price REAL NOT NULL,
            exit_time TEXT NOT NULL, exit_price REAL NOT NULL, exit_reason TEXT NOT NULL,
            shares REAL NOT NULL, net_pnl REAL NOT NULL, net_return_pct REAL NOT NULL
        )""")


def save_forward_paper_run(state: dict) -> None:
    init_forward_paper_db()
    fields = ("run_id", "symbol", "interval", "provider", "short_period", "long_period",
              "units", "starting_cash", "cost_bps", "running", "last_signal_bar",
              "position_entry_time", "position_entry_price", "realized_pnl", "latest_price",
              "status", "started_at", "updated_at")
    values = [state.get(field) for field in fields]
    values[9] = int(bool(state.get("running")))
    values[-1] = datetime.now(timezone.utc).isoformat()
    state["updated_at"] = values[-1]
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO ema_forward_paper_runs ({', '.join(fields)}) "
            f"VALUES ({', '.join('?' for _ in fields)})", values,
        )


def load_running_forward_paper_run() -> dict | None:
    init_forward_paper_db()
    with sqlite3.connect(DATABASE_PATH) as conn:
        row = conn.execute(
            "SELECT run_id, symbol, interval, provider, short_period, long_period, units, "
            "starting_cash, cost_bps, running, last_signal_bar, position_entry_time, "
            "position_entry_price, realized_pnl, latest_price, status, started_at, updated_at "
            "FROM ema_forward_paper_runs WHERE running=1 ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    fields = ("run_id", "symbol", "interval", "provider", "short_period", "long_period",
              "units", "starting_cash", "cost_bps", "running", "last_signal_bar",
              "position_entry_time", "position_entry_price", "realized_pnl", "latest_price",
              "status", "started_at", "updated_at")
    state = dict(zip(fields, row))
    state["running"] = bool(state["running"])
    last_update = pd.Timestamp(state["updated_at"])
    if state["running"] and pd.Timestamp.now(tz="UTC") - last_update > pd.Timedelta(seconds=120):
        stop_forward_paper_bot(state)
        state["status"] = "Paused because the app stopped checking. Any open pretend trade was closed at its last saved price; missed candles were not replayed."
        save_forward_paper_run(state)
    return state


def save_forward_paper_trade(run_id: str, trade: dict) -> None:
    init_forward_paper_db()
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            "INSERT INTO ema_forward_paper_trades "
            "(run_id, entry_time, entry_price, exit_time, exit_price, exit_reason, shares, net_pnl, net_return_pct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, trade["Entry time"], trade["Entry price"], trade["Exit time"], trade["Exit price"],
             trade["Exit reason"], trade["Shares"], trade["Net P&L"], trade["Net return %"]),
        )


def load_forward_paper_trades(run_id: str) -> pd.DataFrame:
    init_forward_paper_db()
    with sqlite3.connect(DATABASE_PATH) as conn:
        return pd.read_sql_query(
            "SELECT entry_time AS 'Entry time', entry_price AS 'Entry price', "
            "exit_time AS 'Exit time', exit_price AS 'Exit price', exit_reason AS 'Exit reason', "
            "shares AS Shares, net_pnl AS 'Net P&L', net_return_pct AS 'Net return %' "
            "FROM ema_forward_paper_trades WHERE run_id=? ORDER BY id DESC",
            conn, params=(run_id,),
        )


def completed_candles(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    if bars.empty:
        return bars
    candle_length = pd.Timedelta(minutes=int(interval[:-1]))
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=5)
    return bars.loc[(bars.index + candle_length) <= cutoff]


def update_forward_paper_bot(state: dict, api_key: str) -> None:
    try:
        bars = load_bars(state["symbol"], state["interval"], state["provider"],
                         api_key if state["provider"] == "Finnhub" else "")
    except Exception as exc:
        state["status"] = str(exc)
        save_forward_paper_run(state)
        return
    closed = completed_candles(bars, state["interval"])
    if len(closed) < state["long_period"] + 2:
        state["status"] = "Waiting for enough completed candles to calculate the averages."
        save_forward_paper_run(state)
        return

    data = closed[["close"]].copy()
    data["ema_fast"] = data["close"].ewm(span=state["short_period"], adjust=False,
                                           min_periods=state["long_period"]).mean()
    data["ema_slow"] = data["close"].ewm(span=state["long_period"], adjust=False,
                                           min_periods=state["long_period"]).mean()
    data = data.dropna(subset=["ema_fast", "ema_slow"])
    if len(data) < 2:
        state["status"] = "Waiting for more completed candles."
        save_forward_paper_run(state)
        return

    cross_up = (data["ema_fast"] > data["ema_slow"]) & (data["ema_fast"].shift(1) <= data["ema_slow"].shift(1))
    cross_down = (data["ema_fast"] < data["ema_slow"]) & (data["ema_fast"].shift(1) >= data["ema_slow"].shift(1))
    cursor = pd.Timestamp(state["last_signal_bar"]) if state.get("last_signal_bar") else data.index[0]
    cost_rate = float(state["cost_bps"]) / 10_000
    status = "Watching for a new completed candle."

    for signal_index in range(1, len(data)):
        signal_time = data.index[signal_index]
        if signal_time <= cursor:
            continue
        wants_entry = state.get("position_entry_price") is None and bool(cross_up.iloc[signal_index])
        wants_exit = state.get("position_entry_price") is not None and bool(cross_down.iloc[signal_index])
        if wants_entry or wants_exit:
            next_bars = bars.loc[bars.index > signal_time]
            if next_bars.empty:
                status = "A crossover appeared; waiting for the next candle's opening price."
                break
            fill_time = next_bars.index[0]
            fill_price = float(next_bars.iloc[0]["open"])
            if wants_entry:
                required_cash = fill_price * (1 + cost_rate) * float(state["units"])
                available_cash = float(state["starting_cash"]) + float(state["realized_pnl"])
                if required_cash <= available_cash:
                    state["position_entry_time"] = fill_time.isoformat()
                    state["position_entry_price"] = fill_price
                    status = f"Paper entry recorded at ${fill_price:,.2f}."
                else:
                    status = "Entry skipped: the selected share amount is larger than the paper balance."
            elif wants_exit:
                entry_price = float(state["position_entry_price"])
                entry_fill = entry_price * (1 + cost_rate)
                exit_fill = fill_price * (1 - cost_rate)
                net_pnl = (exit_fill - entry_fill) * float(state["units"])
                trade = {
                    "Entry time": state["position_entry_time"], "Entry price": entry_price,
                    "Exit time": fill_time.isoformat(), "Exit price": fill_price,
                    "Exit reason": "EMA cross down", "Shares": float(state["units"]),
                    "Net P&L": net_pnl, "Net return %": (exit_fill / entry_fill - 1) * 100,
                }
                save_forward_paper_trade(state["run_id"], trade)
                state["realized_pnl"] = float(state["realized_pnl"]) + net_pnl
                state["position_entry_time"] = None
                state["position_entry_price"] = None
                status = f"Paper exit recorded at ${fill_price:,.2f}; net result ${net_pnl:+,.2f}."
        cursor = signal_time

    state["last_signal_bar"] = cursor.isoformat()
    state["latest_price"] = float(bars.iloc[-1]["close"])
    state["latest_fast_ema"] = float(data["ema_fast"].iloc[-1])
    state["latest_slow_ema"] = float(data["ema_slow"].iloc[-1])
    state["latest_candle_time"] = data.index[-1].isoformat()
    state["status"] = status
    save_forward_paper_run(state)


def stop_forward_paper_bot(state: dict) -> None:
    if state.get("position_entry_price") is not None and state.get("latest_price") is not None:
        entry_price = float(state["position_entry_price"])
        exit_price = float(state["latest_price"])
        cost_rate = float(state["cost_bps"]) / 10_000
        entry_fill = entry_price * (1 + cost_rate)
        exit_fill = exit_price * (1 - cost_rate)
        net_pnl = (exit_fill - entry_fill) * float(state["units"])
        trade = {
            "Entry time": state["position_entry_time"], "Entry price": entry_price,
            "Exit time": datetime.now(timezone.utc).isoformat(), "Exit price": exit_price,
            "Exit reason": "Bot stopped · last observed price", "Shares": float(state["units"]),
            "Net P&L": net_pnl, "Net return %": (exit_fill / entry_fill - 1) * 100,
        }
        save_forward_paper_trade(state["run_id"], trade)
        state["realized_pnl"] = float(state["realized_pnl"]) + net_pnl
        state["position_entry_time"] = None
        state["position_entry_price"] = None
    state["running"] = False
    state["status"] = "Stopped. Any open pretend trade was closed at the last observed price."
    save_forward_paper_run(state)


def scan_summary(d: pd.DataFrame, horizon: int) -> str:
    last = d.iloc[-1]
    direction = "upward" if last.ema9 >= last.ema20 else "downward"
    vwap_position = "above VWAP" if pd.notna(last.vwap) and last.close >= last.vwap else "below VWAP"
    volume = "elevated volume" if pd.notna(last.vol_ratio) and last.vol_ratio >= 1.5 else "ordinary volume"
    estimate = outlook(d, horizon)
    if estimate["n"]:
        odds = f"{estimate['up']:.0%} historical up rate ({estimate['n']} examples)"
    else:
        odds = "not enough history for a comparison"
    return f"{direction.capitalize()} momentum, {vwap_position}, {volume}; {odds}."


def analyze(symbol: str, interval: str, provider: str, api_key: str) -> tuple[pd.DataFrame, str | None]:
    try:
        raw = load_bars(symbol, interval, provider, api_key if provider == "Finnhub" else "")
    except Exception as exc:
        return pd.DataFrame(), str(exc)
    if raw.empty:
        return raw, "No intraday data came back for this ticker. Check the symbol or try again later."
    return add_indicators(raw), None


def chart(d: pd.DataFrame, symbol: str, interval: str, bars: int) -> go.Figure:
    view = d.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view.index, open=view.open, high=view.high, low=view.low,
                                 close=view.close, name=symbol,
                                 increasing_line_color="#159477", decreasing_line_color="#db6561"))
    fig.add_trace(go.Scatter(x=view.index, y=view.ema9, name="EMA 9", line=dict(color="#3989cf", width=1.5)))
    fig.add_trace(go.Scatter(x=view.index, y=view.ema20, name="EMA 20", line=dict(color="#e9a542", width=1.5)))
    fig.add_trace(go.Scatter(x=view.index, y=view.vwap, name="Session VWAP", line=dict(color="#8c66c5", width=1.5, dash="dot")))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=25, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08), template="plotly_white",
                      yaxis_title="Price", xaxis_title=f"Time (each candle = {interval})")
    return fig


def show_analysis(symbol: str, interval: str, horizon: int, provider: str, api_key: str):
    d, error = analyze(symbol, interval, provider, api_key)
    if error:
        st.warning(f"{symbol}: {error}")
        return
    if len(d) < 30:
        st.warning(f"{symbol}: Not enough recent intraday bars to summarize yet.")
        return
    now = d.iloc[-1]
    market_data, market_error = analyze("SPY", interval, "Yahoo Finance", "")
    estimate = outlook_with_market(d, None if market_error else market_data, horizon)
    market_matched = "matching SPY" in estimate["basis"]
    if not market_matched or estimate["n"] < 10:
        prediction = "Not enough market history"
    elif estimate["up"] >= 0.55:
        prediction = "Leans up"
    elif estimate["up"] <= 0.45:
        prediction = "Leans down"
    else:
        prediction = "No clear direction"
    trend = "Upward" if now.ema9 >= now.ema20 else "Downward"
    st.subheader(f"{symbol.upper()} · {interval} candles")
    st.caption(f"Source: {provider} · Latest available bar: {d.index[-1].strftime('%b %d, %Y %I:%M %p %Z')} · Checked {datetime.now().astimezone().strftime('%I:%M %p %Z')}")
    a, b, c, e = st.columns(4)
    a.metric("Latest price", f"${now.close:,.2f}")
    b.metric("Short trend", trend, "EMA 9 vs EMA 20")
    b2 = float(now.rsi) if pd.notna(now.rsi) else 0
    c.metric("RSI (14)", f"{b2:.0f}" if b2 else "—", "momentum gauge")
    e.metric("Volume vs recent avg", f"{now.vol_ratio:.1f}×" if pd.notna(now.vol_ratio) else "—")
    st.plotly_chart(chart(d, symbol.upper(), interval, horizon), alt=f"{symbol.upper()} intraday price and moving-average chart")

    st.markdown(f"#### Prediction · next {horizon} bar{'s' if horizon != 1 else ''}")
    st.caption(f"Market context: {estimate['market_status']}. {estimate['market_note']} SPY data is fetched from Yahoo Finance.")
    if estimate["n"]:
        target = float(now.close) * (1 + estimate["median"])
        low = float(now.close) * (1 + estimate["low"])
        high = float(now.close) * (1 + estimate["high"])
        p1, p2, p3 = st.columns(3)
        probability_note = (f"{estimate['up']:.0%} up · {estimate['up_count']}/{estimate['n']} matches"
                            if market_matched else "SPY-matched sample is too small")
        p1.metric("Prediction", prediction, probability_note)
        p2.metric("Median move in past examples", f"{estimate['median']:+.2%}", f"if repeated: about ${target:,.2f}")
        p3.metric("Range in past examples", f"${low:,.2f} – ${high:,.2f}", "middle 60%, not a target")
        st.caption(f"Direction rule: up-lean at 55% or higher, down-lean at 45% or lower; otherwise no clear direction. The percentage is a historical frequency, not a guaranteed chance. Basis: {estimate['basis']}. Examples can overlap. Horizon: {horizon} × {interval} bars.")
    else:
        st.info("There is not enough history to calculate a comparison for this interval.")


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Intraday price source", ["Yahoo Finance", "Finnhub"])
    try:
        configured_api_key = st.secrets.get("FINNHUB_API_KEY", "")
    except Exception:
        configured_api_key = ""
    api_key = st.text_input("Finnhub API key (for headlines and Finnhub prices)", type="password",
                            value=configured_api_key,
                            help="Optional. You can enter it here for this session, or set FINNHUB_API_KEY in your hosting secrets.")
    st.caption("Yahoo Finance is the free no-key choice for 1-minute, 5-minute, and 15-minute candles. Finnhub may deny intraday candles unless your account plan includes them; its news feed is available separately here.")
    st.link_button("Finnhub account and plan details", "https://finnhub.io/pricing")
    interval = st.selectbox("Candle size", ["1m", "5m", "15m"],
                            format_func=lambda x: {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes"}[x])
    horizon = st.slider("Compare the next number of candles", 1, 5, 5)
    st.caption(f"With {interval} candles, comparing the next {horizon} candle(s) covers about {int(interval[:-1]) * horizon} minute(s).")
    auto_refresh = st.checkbox("Refresh open analysis every 30 seconds", value=True, disabled=(provider == "Finnhub" and not api_key))
    if provider == "Yahoo Finance":
        st.caption("Yahoo Finance requires no key. Its intraday prices may be delayed and are not an exchange-certified live feed.")
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

tab_stock, tab_scan, tab_news, tab_live, tab_research, tab_strategy, tab_forward_bot = st.tabs(
    ["Analyze a stock", "Scan a watchlist", "Market news", "Live watchlist", "Automated research", "EMA paper bot", "Practice bot"]
)

refresh_seconds = "30s" if auto_refresh and (provider == "Yahoo Finance" or api_key) else None

@st.fragment(run_every=refresh_seconds)
def live_analysis(symbol: str, candle_interval: str, future_bars: int, source: str, token: str):
    show_analysis(symbol, candle_interval, future_bars, source, token)


@st.fragment(run_every="2s")
def show_live_watchlist(manager: dict):
    with manager["lock"]:
        status = manager["status"]
    st.caption(f"Finnhub WebSocket · {status}")
    table = live_table(manager)
    st.dataframe(table, hide_index=True, alt="Live stock prices from Finnhub trade updates",
                 column_config={"Live price": st.column_config.NumberColumn(format="$%.2f"),
                                "1-minute move": st.column_config.NumberColumn(format="%+.2f%%"),
                                "5-minute move": st.column_config.NumberColumn(format="%+.2f%%"),
                                "15-minute move": st.column_config.NumberColumn(format="%+.2f%%")})
    st.caption("Moves are calculated from the first observed trade at or before each time window. A blank move means the stream has not collected enough ticks yet. Prices update only when Finnhub sends trades.")


@st.fragment(run_every="30s")
def show_forward_paper_bot(api_key: str):
    state = st.session_state.get("forward_ema_paper_bot")
    if state is None:
        state = load_running_forward_paper_run()
        if state is not None:
            st.session_state["forward_ema_paper_bot"] = state
    if state is None:
        st.info("No practice bot is running. Set its rules below and start it when you are ready.")
        return

    if state.get("running"):
        last_update = pd.Timestamp(state["updated_at"]) if state.get("updated_at") else pd.Timestamp.now(tz="UTC")
        if pd.Timestamp.now(tz="UTC") - last_update > pd.Timedelta(seconds=120):
            stop_forward_paper_bot(state)
            state["status"] = "Paused because the app stopped checking. Any open pretend trade was closed at its last saved price; missed candles were not replayed."
            save_forward_paper_run(state)
        else:
            update_forward_paper_bot(state, api_key)
    if state.get("running") and st.button("Stop bot and close pretend trade", key=f"stop_forward_bot_{state['run_id']}"):
        stop_forward_paper_bot(state)
        st.session_state["forward_ema_paper_bot"] = state

    st.markdown(f"#### {state['symbol']} · {state['interval']} · {state['short_period']}/{state['long_period']} EMA")
    if state.get("running"):
        st.caption("Practice bot is running while this app stays open.")
    else:
        st.caption("This practice bot is stopped. Its saved trade log is still available below.")
    if state.get("status"):
        if any(word in str(state["status"]).lower() for word in ("denied", "rejected", "failed", "error")):
            st.warning(state["status"])
        else:
            st.info(state["status"])

    trades = load_forward_paper_trades(state["run_id"])
    closed_count = len(trades)
    wins = int((trades["Net P&L"] > 0).sum()) if closed_count else 0
    realized = float(state.get("realized_pnl", 0.0))
    unrealized = 0.0
    if state.get("position_entry_price") is not None and state.get("latest_price") is not None:
        cost_rate = float(state["cost_bps"]) / 10_000
        entry_fill = float(state["position_entry_price"]) * (1 + cost_rate)
        exit_fill = float(state["latest_price"]) * (1 - cost_rate)
        unrealized = (exit_fill - entry_fill) * float(state["units"])
    p1, p2, p3 = st.columns(3)
    if state.get("latest_fast_ema") is not None and state.get("latest_slow_ema") is not None:
        prediction = "Upward lean" if state["latest_fast_ema"] > state["latest_slow_ema"] else "Downward lean"
    else:
        prediction = "Waiting for candles"
    p1.metric("Prediction", prediction)
    p2.metric("Pretend profit/loss", f"${realized + unrealized:+,.2f}",
              f"closed ${realized:+,.2f} · open ${unrealized:+,.2f}")
    p3.metric("Winning closed trades", f"{wins} of {closed_count}",
              f"{wins / closed_count:.0%}" if closed_count else "No closed trades yet")
    if state.get("position_entry_price") is not None:
        st.caption(f"Pretend position: {state['units']:g} shares from ${float(state['position_entry_price']):,.2f}; latest observed price ${float(state.get('latest_price') or 0):,.2f}.")
    if state.get("latest_candle_time"):
        st.caption(f"Latest completed candle used: {state['latest_candle_time']} · source: {state['provider']}.")
    st.caption("The bot checks about every 30 seconds while this app is open. It acts only on a completed EMA crossover and assumes a fill at the next candle's open. Price feeds may be delayed or incomplete.")
    if not trades.empty:
        st.dataframe(trades, hide_index=True, alt="Saved practice trades from the running EMA bot",
                     column_config={
                         "Entry price": st.column_config.NumberColumn(format="$%.2f"),
                         "Exit price": st.column_config.NumberColumn(format="$%.2f"),
                         "Net P&L": st.column_config.NumberColumn(format="$%+.2f"),
                         "Net return %": st.column_config.NumberColumn(format="%+.2f%%"),
                     })
        st.download_button("Download practice bot trades", trades.to_csv(index=False),
                           file_name=f"{state['symbol'].lower()}-forward-paper-trades.csv",
                           mime="text/csv", key=f"download_forward_bot_{state['run_id']}")


@st.fragment(run_every="10s")
def research_status_monitor(state: dict, auto_enabled: bool, scheduled_hhmm: str):
    if auto_enabled:
        now_local = datetime.now().astimezone()
        due = now_local.strftime("%H:%M") >= scheduled_hhmm
        latest_run, _, _ = research.latest_research()
        latest_local_day = datetime.fromisoformat(latest_run).astimezone().date() if latest_run else None
        attempted_today = state.get("auto_attempt_date") == now_local.date().isoformat()
        if due and latest_local_day != now_local.date() and not attempted_today:
            state["auto_attempt_date"] = now_local.date().isoformat()
            start_market_research(state)
    with state["lock"]:
        running = state["running"]
        status = state["status"]
        progress = state["progress"]
        error = state["error"]
        source_errors = list(state["source_errors"])
    st.caption(f"Automation status: {status}")
    if running and progress[1]:
        st.progress(min(progress[0] / progress[1], 1), text=f"{status}")
    if error:
        st.error(error)
    if source_errors:
        st.warning("Some data requests were unavailable: " + "; ".join(source_errors[:8]))

with tab_stock:
    with st.form("analyze_form"):
        ticker = st.text_input("Stock ticker", "AAPL", max_chars=12).strip().upper()
        submitted = st.form_submit_button("Analyze", type="primary")
    if submitted and ticker:
        if provider == "Finnhub" and not api_key:
            st.info("Add a Finnhub API key in Settings, or select Yahoo Finance as the source.")
        else:
            live_analysis(ticker, interval, horizon, provider, api_key)
    else:
        st.write("Enter a ticker such as AAPL, MSFT, NVDA, or SPY, then select **Analyze**.")

with tab_scan:
    st.write("Scan up to 100 symbols for intraday momentum and volume. Use commas, spaces, or paste a ticker list. Larger scans may take longer or hit provider limits.")
    with st.form("scan_form"):
        tickers_text = st.text_area("Tickers separated by commas or spaces", DEFAULT_WATCHLIST, height=90)
        scan_submitted = st.form_submit_button("Scan watchlist", type="primary")
    if scan_submitted:
        symbols = list(dict.fromkeys(x.strip().upper() for x in tickers_text.replace(",", " ").split() if x.strip()))[:100]
        if provider == "Finnhub" and not api_key:
            st.info("Add a Finnhub API key in Settings, or select Yahoo Finance as the source.")
        elif not symbols:
            st.info("Add at least one ticker to scan.")
        else:
            rows = []
            progress = st.progress(0, text="Loading intraday bars…")
            for i, sym in enumerate(symbols):
                d, error = analyze(sym, interval, provider, api_key)
                fatal_provider_error = bool(error and any(term in error.lower() for term in ("request limit", "rejected the api key", "denied this request")))
                if fatal_provider_error:
                    status = "Provider limit reached" if "request limit" in error.lower() else "Provider/key error"
                    rows.append({"Ticker": sym, "Status": status, "Why": error, "Latest bar": "—", "Price": np.nan, "1-bar move": np.nan, "5-bar move": np.nan, "Trend": "—", "Vs VWAP": "—", "RSI": np.nan, "Volume vs avg": np.nan, "Up in similar periods": np.nan, "Median move": np.nan, "Historical range": "—", "Examples": 0, "Readout": "—"})
                    break
                if error:
                    rows.append({"Ticker": sym, "Status": "Data unavailable", "Why": error, "Latest bar": "—", "Price": np.nan, "1-bar move": np.nan, "5-bar move": np.nan, "Trend": "—", "Vs VWAP": "—", "RSI": np.nan, "Volume vs avg": np.nan, "Up in similar periods": np.nan, "Median move": np.nan, "Historical range": "—", "Examples": 0, "Readout": "—"})
                elif len(d) < 30:
                    latest = d.index[-1].strftime("%b %d %I:%M %p %Z") if not d.empty else "—"
                    rows.append({"Ticker": sym, "Status": "Need more bars", "Why": f"Only {len(d)} bars; at least 30 are needed for this summary.", "Latest bar": latest, "Price": float(d.iloc[-1].close) if not d.empty else np.nan, "1-bar move": np.nan, "5-bar move": np.nan, "Trend": "—", "Vs VWAP": "—", "RSI": np.nan, "Volume vs avg": np.nan, "Up in similar periods": np.nan, "Median move": np.nan, "Historical range": "—", "Examples": 0, "Readout": "—"})
                else:
                    last = d.iloc[-1]
                    o = outlook(d, horizon)
                    range_text = f"{o['low']:+.2%} to {o['high']:+.2%}" if o["n"] else "—"
                    rows.append({"Ticker": sym, "Status": "Ready", "Why": "—", "Latest bar": d.index[-1].strftime("%b %d %I:%M %p %Z"), "Price": float(last.close),
                                 "1-bar move": float((last.close / d.iloc[-2].close - 1) * 100),
                                 "5-bar move": float((last.close / d.iloc[-6].close - 1) * 100),
                                 "Trend": "Up" if last.ema9 >= last.ema20 else "Down",
                                 "Vs VWAP": "Above" if pd.notna(last.vwap) and last.close >= last.vwap else "Below",
                                 "RSI": float(last.rsi) if pd.notna(last.rsi) else np.nan,
                                 "Volume vs avg": float(last.vol_ratio) if pd.notna(last.vol_ratio) else np.nan,
                                 "Up in similar periods": o["up"], "Median move": o["median"] * 100 if o["n"] else np.nan,
                                 "Historical range": range_text, "Examples": o["n"], "Readout": scan_summary(d, horizon)})
                progress.progress((i + 1) / len(symbols), text=f"Checking {sym} ({i + 1}/{len(symbols)})")
            progress.empty()
            table = pd.DataFrame(rows)
            st.dataframe(table, hide_index=True, alt="Intraday stock scan with recent momentum, VWAP, volume, and historical outcomes",
                         column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                                        "1-bar move": st.column_config.NumberColumn(format="%+.2f%%"),
                                        "5-bar move": st.column_config.NumberColumn(format="%+.2f%%"),
                                        "RSI": st.column_config.NumberColumn(format="%.0f"),
                                        "Volume vs avg": st.column_config.NumberColumn(format="%.1f×"),
                                        "Up in similar periods": st.column_config.NumberColumn(format="percent"),
                                        "Median move": st.column_config.NumberColumn(format="%+.2f%%")})
            st.caption(f"Watchlist scan · {interval} bars · next {horizon} candle(s). The historical range is the 20th–80th percentile of later moves in the sample; odds and ranges describe past data, not guaranteed predictions.")
            if (table.Status == "Provider limit reached").any():
                st.info("The selected provider stopped this scan at its request limit. Try a smaller batch or wait before scanning again.")
            if not (table.Status == "Ready").any():
                st.error("This scan returned no usable ticker data. Read the **Why** column for the provider's response. For free, no-key data choose Yahoo Finance; for Finnhub, add a valid API key and scan a smaller batch to stay within its limits.")
            ready = table[table.Status == "Ready"]
            if not ready.empty:
                best = ready.loc[ready["5-bar move"].idxmax()]
                above_vwap = int((ready["Vs VWAP"] == "Above").sum())
                elevated_volume = int((ready["Volume vs avg"] >= 1.5).sum())
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Tickers with usable data", f"{len(ready)} / {len(symbols)}")
                s2.metric("Strongest 5-bar move", f"{best.Ticker} · {best['5-bar move']:+.2f}%")
                s3.metric("Above session VWAP", f"{above_vwap} / {len(ready)}")
                s4.metric("Volume ≥ 1.5× recent avg", f"{elevated_volume} / {len(ready)}")
                selected = st.selectbox("Open a ticker from this scan", ready.Ticker.tolist())
                if st.button("Show detailed analysis", key="show_scan_detail"):
                    live_analysis(selected, interval, horizon, provider, api_key)

with tab_news:
    st.write("Browse recent headlines from Finnhub's aggregated market-news feed or filter headlines for one ticker. News availability and publisher coverage vary by source.")
    if not api_key:
        st.info("Enter your Finnhub API key in Settings to load market and company headlines. This can be used while Yahoo Finance remains selected for prices.")
    else:
        with st.form("news_form"):
            news_ticker = st.text_input("Optional ticker filter", "", max_chars=12).strip().upper()
            news_submitted = st.form_submit_button("Load headlines", type="primary")
        if news_submitted:
            try:
                articles = load_news(api_key, news_ticker)
                if not articles:
                    st.info("No recent headlines were returned.")
                for article in articles[:30]:
                    title = article.get("headline") or article.get("title") or "Untitled story"
                    source_name = article.get("source", "News source")
                    published = article.get("datetime")
                    when = datetime.fromtimestamp(published, timezone.utc).astimezone().strftime("%b %d · %I:%M %p %Z") if isinstance(published, (int, float)) and published else "Recent"
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(f"{source_name} · {when}")
                        story_url = article.get("url")
                        if story_url:
                            st.link_button("Read story", story_url)
            except Exception as exc:
                st.error(f"Could not load headlines: {exc}")

with tab_live:
    st.write("Stream live trades for a watchlist and calculate price changes over the latest 1 and 5 minutes. This uses Finnhub WebSocket access, separate from the candle endpoint.")
    with st.form("live_watchlist_form"):
        live_symbols_text = st.text_area("Live tickers separated by commas or spaces", DEFAULT_WATCHLIST, height=90)
        start_live = st.form_submit_button("Start live watchlist", type="primary")
    if start_live:
        if not api_key:
            st.error("Enter your Finnhub API key in Settings first.")
        else:
            symbols = list(dict.fromkeys(x.strip().upper() for x in live_symbols_text.replace(",", " ").split() if x.strip()))[:50]
            if not symbols:
                st.info("Add at least one ticker.")
            else:
                start_live_stream(api_key, symbols)
                st.rerun()
    manager = st.session_state.get("live_stream")
    if manager:
        if st.button("Stop live watchlist", key="stop_live_watchlist"):
            manager["stop"].set()
            st.rerun()
        show_live_watchlist(manager)

with tab_research:
    st.subheader("Market-wide research and daily tracking")
    st.write("This workflow gathers a broad Yahoo Finance screen, compares volume with each ticker's own recent history, checks options for a limited sample, flags up to five unusual names, then waits for your review.")
    st.info("Coverage is limited by the free data source: this combines up to 50 active, gaining, losing, and heavily shorted stocks per screen, up to 50 ETFs, and a curated index/sector ETF list. It does not cover every listed security. Options are sampled for up to 25 names; unavailable fields stay labeled UNAVAILABLE.")
    research_state = st.session_state.setdefault("market_research_state", {
        "running": False, "status": "Not started", "progress": (0, 1, ""),
        "error": "", "source_errors": [], "lock": threading.Lock(), "thread": None,
    })
    schedule_default = datetime.strptime("15:10", "%H:%M").time()
    auto_enabled = st.checkbox("Run once each day while this app is open", key="auto_market_scan")
    scheduled_time = st.time_input("Daily scan time (your computer's local time)", schedule_default, key="market_scan_time")
    manual_col, sched_col = st.columns([1, 2])
    with manual_col:
        if st.button("Run market scan now", type="primary", disabled=research_state["running"]):
            research_state["error"] = ""
            research_state["source_errors"] = []
            start_market_research(research_state)
            st.rerun()
    with sched_col:
        st.caption("The app must stay open for the daily schedule to run. The default is 3:10 PM local time, shortly after the U.S. regular session closes.")
    research_status_monitor(research_state, auto_enabled, scheduled_time.strftime("%H:%M"))

    st.markdown("#### Prediction track record · daily volume alerts")
    paper_results = research.paper_test_results()
    if paper_results.empty:
        st.info("No completed 25-minute paper outcomes yet. After an eligible daily volume alert, the next scan will try to measure the next session's first 25 minutes using Yahoo 5-minute bars.")
    else:
        result_count = len(paper_results)
        up_count = int((paper_results["25-minute return %"] > 0).sum())
        beat_count = int((paper_results["Difference vs SPY %"] > 0).sum())
        up_rate = up_count / result_count
        distinct_days = int(paper_results["Signal date"].nunique())
        chronological = paper_results.sort_values(["Signal date", "Ticker"]).reset_index(drop=True)
        holdout_start = int(result_count * 2 / 3)
        holdout = chronological.iloc[holdout_start:]
        holdout_excess = float(holdout["Difference vs SPY %"].mean()) if not holdout.empty else float("nan")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Measured up rate from past tests", f"{up_rate:.0%}", f"{up_count} of {result_count} paper tests")
        p2.metric("Median 25-minute move", f"{paper_results['25-minute return %'].median():+.2f}%")
        p3.metric("Positive vs SPY", f"{beat_count / result_count:.0%}", f"{beat_count} of {result_count} tests")
        p4.metric("Latest third avg vs SPY", f"{holdout_excess:+.2f}%" if pd.notna(holdout_excess) else "—",
                  f"{len(holdout)} tests · before costs")
        if result_count < 100 or distinct_days < 40:
            st.warning(f"Early sample: this has {result_count}/100 paper tests across {distinct_days}/40 signal dates. Do not treat the percentage as a dependable future probability yet.")
        elif holdout_excess <= 0:
            st.warning("The latest third did not beat SPY on average before costs. The saved history does not support an edge under this test so far.")
        else:
            st.caption("The latest third beat SPY on average before costs. This is still historical, may change, and is not proof of a tradable edge.")
        st.caption("Rule: US equities only; price ≥ $5; average daily dollar volume ≥ $20 million; share volume ≥ 2× its baseline. The same ticker is counted at most once per 7 calendar days (about 5 trading days). Entry is the close of the first 5-minute bar next session (about 9:35 a.m. ET); exit is the close at 10:00 a.m. ET. Return is gross, before spread and fees. Yahoo data can be incomplete or delayed. The matching SPY return is over the same window.")
        with st.expander("See completed paper tests"):
            st.dataframe(paper_results, hide_index=True, alt="Completed next-session 25-minute paper outcomes compared with SPY")
            st.download_button("Download paper-test results", paper_results.to_csv(index=False),
                               file_name="market-minute-paper-test.csv", mime="text/csv")

    with st.expander("How to read an alert and test it fairly"):
        st.markdown("""
**What an alert says:** trading volume was unusual. It does not say the stock should go up or down.

**Score:** the larger of share-volume multiple and options-volume multiple. A bigger score means a larger activity ratio, not a better opportunity or a higher chance of profit. Small usual volume can make a ticker look unusually active.

**Data checks:** `AVAILABLE` means the app received the inputs needed for that calculation. It does not independently verify the provider. Options intent, opening versus closing trades, and the reason for a move may remain unknown. Missing baselines are not treated as evidence.

**Fair test:** the paper results above use one fixed next-session 25-minute window, compare with SPY, and keep the signal rules fixed. Collect at least 100 results across 40 different signal dates, include realistic spread and fees before judging it, and check the newest third separately. A small or changing sample is not a proven edge.

**Timeframe reminder:** the alert test uses a daily volume screen and a next-session 25-minute outcome. The stock-analysis tab has a separate 1-minute, 5-minute, or 15-minute historical up-rate; the two percentages answer different questions.
""")

    latest_run, universe_results, flags = research.latest_research()
    if latest_run:
        local_run = datetime.fromisoformat(latest_run).astimezone().strftime("%b %d, %Y · %I:%M:%S %p %Z")
        st.markdown(f"#### Latest run · {local_run}")
        if flags.empty:
            st.info("No symbols met the unusual-activity thresholds in this run.")
        else:
            st.markdown("#### Up to five research flags")
            st.caption("These flags describe unusual activity only. Options volume cannot show whether trades were opened, closed, bought, sold, or used as a hedge.")
            flag_columns = [c for c in ["Rank", "Ticker", "Type", "Score", "Why flagged", "Boring explanation",
                                        "Options volume vs own history", "Options baseline data", "OI change",
                                        "OI change data", "Data quality", "Unknowns", "Decision"] if c in flags]
            st.dataframe(flags[flag_columns], hide_index=True, alt="Ranked unusual market activity flags with explanation and review status",
                         column_config={"Score": st.column_config.NumberColumn(format="%.1f×"),
                                        "Options volume vs own history": st.column_config.NumberColumn(format="%.1f×"),
                                        "OI change": st.column_config.NumberColumn(format="%d")})
            candidate = st.selectbox("Choose a flag to approve for paper tracking or reject", flags["Ticker"].tolist())
            note = st.text_input("Optional review note", key="market_review_note")
            approve_col, reject_col = st.columns(2)
            with approve_col:
                if st.button("Approve for paper tracking", key="approve_paper_flag"):
                    research.review_candidate(research.DATABASE_PATH, latest_run, candidate,
                                              "APPROVED FOR PAPER TRACKING", note)
                    st.success(f"{candidate} approved for simulated tracking. No order was placed.")
                    st.rerun()
            with reject_col:
                if st.button("Reject this flag", key="reject_paper_flag"):
                    research.review_candidate(research.DATABASE_PATH, latest_run, candidate, "REJECTED", note)
                    st.info(f"{candidate} rejected and logged. No order was placed.")
                    st.rerun()
        if not universe_results.empty:
            st.markdown("#### Scanned market sample")
            view_cols = [c for c in ["Ticker", "Type", "Price", "Day change %", "Day volume", "30d avg volume",
                                     "Relative volume", "Relative volume data", "30d move %", "90d move %", "Calls volume", "Puts volume",
                                     "Put/call", "Open interest", "OI change", "OI change data", "Options volume vs own history", "Options baseline data",
                                     "Options expiry", "Earnings", "Data quality"] if c in universe_results]
            st.dataframe(universe_results[view_cols], hide_index=True, alt="Market screening sample with historical price, volume, and available options fields",
                         column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                                        "Day change %": st.column_config.NumberColumn(format="%+.2f%%"),
                                        "30d move %": st.column_config.NumberColumn(format="%+.2f%%"),
                                        "90d move %": st.column_config.NumberColumn(format="%+.2f%%"),
                                        "Relative volume": st.column_config.NumberColumn(format="%.2f×"),
                                        "Put/call": st.column_config.NumberColumn(format="%.2f"),
                                        "Options volume vs own history": st.column_config.NumberColumn(format="%.1f×")})
            st.download_button("Download this research run", universe_results.to_csv(index=False),
                               file_name=f"market-research-{latest_run[:10]}.csv", mime="text/csv")
        followups = research.recent_followups()
        if not followups.empty:
            st.markdown("#### What happened after earlier flags")
            st.caption("This records the price at the next available scan; it is an audit trail, not proof a flag predicted the move.")
            st.dataframe(followups, hide_index=True, alt="Follow-up price changes after previously flagged candidates",
                         column_config={"Flag price": st.column_config.NumberColumn(format="$%.2f"),
                                        "Follow-up price": st.column_config.NumberColumn(format="$%.2f"),
                                        "Move after flag %": st.column_config.NumberColumn(format="%+.2f%%")})
    else:
        st.info("No market-wide research run has been saved yet. Select **Run market scan now** to create the first snapshot and begin the daily baseline.")

with tab_strategy:
    st.subheader("EMA crossover · paper test")
    st.write("See how a short moving average crossing a longer one would have performed on recent 1-minute, 5-minute, or 15-minute candles.")
    st.caption("This is a historical simulation. It does not connect to a broker or place orders.")
    with st.form("ema_paper_strategy_form"):
        strategy_symbol = st.text_input("Ticker", "AAPL", max_chars=12).strip().upper()
        setting_a, setting_b, setting_c = st.columns(3)
        with setting_a:
            short_period = st.number_input("Fast EMA candles", min_value=2, max_value=200, value=10, step=1)
        with setting_b:
            long_period = st.number_input("Slow EMA candles", min_value=3, max_value=500, value=50, step=1)
        with setting_c:
            strategy_units = st.number_input("Paper shares", min_value=0.01, max_value=100000.0, value=1.0, step=1.0)
        setting_d, setting_e = st.columns(2)
        with setting_d:
            paper_cash = st.number_input("Starting paper balance", min_value=100.0, max_value=10000000.0,
                                         value=10000.0, step=1000.0, format="%.2f")
        with setting_e:
            assumed_cost = st.number_input("Estimated cost per side (basis points)", min_value=0.0,
                                            max_value=200.0, value=5.0, step=1.0,
                                            help="One basis point is 0.01%. This estimates trading friction; it is not a live quote.")
        run_ema_test = st.form_submit_button("Run paper test", type="primary")

    if run_ema_test:
        if not strategy_symbol:
            st.error("Enter a ticker, such as AAPL.")
        elif short_period >= long_period:
            st.error("The fast EMA period must be smaller than the slow EMA period.")
        elif provider == "Finnhub" and not api_key:
            st.error("Enter your Finnhub key in Settings, or choose Yahoo Finance as the price source.")
        else:
            with st.spinner(f"Loading recent {interval} candles for {strategy_symbol}…"):
                strategy_bars, strategy_error = analyze(strategy_symbol, interval, provider, api_key)
                if strategy_error:
                    st.session_state.pop("ema_paper_result", None)
                    st.error(f"{strategy_symbol}: {strategy_error}")
                else:
                    strategy_summary, strategy_trades = ema_crossover_paper_test(
                        strategy_bars, int(short_period), int(long_period), float(strategy_units),
                        float(assumed_cost), float(paper_cash),
                    )
                    if strategy_summary.get("error"):
                        st.session_state.pop("ema_paper_result", None)
                        st.warning(strategy_summary["error"])
                    else:
                        st.session_state["ema_paper_result"] = {
                            "symbol": strategy_symbol, "interval": interval, "provider": provider,
                            "summary": strategy_summary, "trades": strategy_trades,
                            "units": float(strategy_units), "cost_bps": float(assumed_cost),
                        }

    strategy_result = st.session_state.get("ema_paper_result")
    if strategy_result:
        summary = strategy_result["summary"]
        trades = strategy_result["trades"]
        st.markdown(f"#### {strategy_result['symbol']} · {strategy_result['interval']} candles")
        up_lean = summary["latest_short_ema"] > summary["latest_long_ema"]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Prediction", "Upward lean" if up_lean else "Downward lean",
                  summary["current_alignment"])
        p2.metric("Past paper result", f"${summary['net_pnl']:+,.2f}",
                  f"{summary['strategy_return_pct']:+.2f}% of starting balance")
        p3.metric("Winning closed trades", f"{summary['wins']} of {summary['trade_count']}",
                  f"{summary['win_rate']:.0%} of past trades" if summary["win_rate"] is not None else "No closed trades")
        p4.metric("Buy-and-hold over same candles", f"{summary['buy_hold_return_pct']:+.2f}%",
                  "with the same estimated costs")
        st.caption(
            f"Used {summary['bars_used']} candles from {summary['first_bar']} to {summary['last_bar']} "
            f"via {strategy_result['provider']}. Current price: ${summary['latest_price']:,.2f}; "
            f"fast EMA: ${summary['latest_short_ema']:,.2f}; slow EMA: ${summary['latest_long_ema']:,.2f}. "
            f"Estimated trading cost: {strategy_result['cost_bps']:g} basis points per side."
        )
        if summary["trade_count"] == 0:
            st.info("No complete crossover trades appeared in the available candles. Try a shorter EMA period or a different ticker.")
        else:
            st.dataframe(trades, hide_index=True, alt="Historical paper trades from the EMA crossover simulation",
                         column_config={
                             "Entry price": st.column_config.NumberColumn(format="$%.2f"),
                             "Exit price": st.column_config.NumberColumn(format="$%.2f"),
                             "Net P&L": st.column_config.NumberColumn(format="$%+.2f"),
                             "Net return %": st.column_config.NumberColumn(format="%+.2f%%"),
                         })
            st.download_button("Download simulated trades", trades.to_csv(index=False),
                               file_name=f"{strategy_result['symbol'].lower()}-ema-paper-trades.csv",
                               mime="text/csv")
        st.caption(
            "How it simulates: after a candle closes and the fast EMA crosses the slow EMA, it assumes a fill at the next candle's open. "
            "It enters on an upward cross and exits on a downward cross; any open trade is closed at the last available candle. "
            "The win rate is the share of these past simulated trades that made money after the selected estimated costs. "
            "It is not the chance the next trade will win, and this simulation does not model every spread, delay, or market interruption."
        )

with tab_forward_bot:
    st.subheader("Automatic practice bot")
    st.write("This watches for new EMA crossings and records pretend trades using your selected candle size.")
    st.warning("Practice only: it cannot place real orders. It runs while this app is open, and price data may be delayed or incomplete.")
    active_bot = st.session_state.get("forward_ema_paper_bot")
    if active_bot is None:
        active_bot = load_running_forward_paper_run()
        if active_bot is not None:
            st.session_state["forward_ema_paper_bot"] = active_bot

    if active_bot and active_bot.get("running"):
        st.info("A practice bot is already running. Stop it below before starting another one.")
    else:
        with st.form("forward_ema_paper_bot_form"):
            forward_symbol = st.text_input("Ticker to watch", "AAPL", max_chars=12).strip().upper()
            forward_a, forward_b, forward_c = st.columns(3)
            with forward_a:
                forward_short = st.number_input("Fast EMA candles", min_value=2, max_value=200, value=10, step=1,
                                                key="forward_ema_fast")
            with forward_b:
                forward_long = st.number_input("Slow EMA candles", min_value=3, max_value=500, value=50, step=1,
                                               key="forward_ema_slow")
            with forward_c:
                forward_units = st.number_input("Pretend shares", min_value=0.01, max_value=100000.0,
                                                value=1.0, step=1.0, key="forward_ema_shares")
            forward_d, forward_e = st.columns(2)
            with forward_d:
                forward_cash = st.number_input("Pretend starting balance", min_value=100.0,
                                               max_value=10000000.0, value=10000.0, step=1000.0,
                                               format="%.2f", key="forward_ema_cash")
            with forward_e:
                forward_cost = st.number_input("Estimated cost per side (basis points)", min_value=0.0,
                                               max_value=200.0, value=5.0, step=1.0,
                                               help="One basis point is 0.01%. This is an estimate, not a live spread.",
                                               key="forward_ema_cost")
            start_forward_bot = st.form_submit_button("Start practice bot", type="primary")

        if start_forward_bot:
            if not forward_symbol:
                st.error("Enter a ticker, such as AAPL.")
            elif forward_short >= forward_long:
                st.error("The fast EMA period must be smaller than the slow EMA period.")
            elif provider == "Finnhub" and not api_key:
                st.error("Choose Yahoo Finance in Settings or enter a Finnhub key that can access intraday candles.")
            else:
                with st.spinner(f"Preparing the practice bot for {forward_symbol}…"):
                    initial_error = None
                    try:
                        initial_bars = load_bars(forward_symbol, interval, provider,
                                                 api_key if provider == "Finnhub" else "")
                        initial_closed = completed_candles(initial_bars, interval)
                    except Exception as exc:
                        initial_closed = pd.DataFrame()
                        initial_error = str(exc)
                    if not initial_closed.empty:
                        initial_ema_fast = initial_closed["close"].ewm(
                            span=int(forward_short), adjust=False, min_periods=int(forward_long)
                        ).mean()
                        initial_ema_slow = initial_closed["close"].ewm(
                            span=int(forward_long), adjust=False, min_periods=int(forward_long)
                        ).mean()
                        valid_ema = pd.DataFrame({"fast": initial_ema_fast, "slow": initial_ema_slow}).dropna()
                    else:
                        valid_ema = pd.DataFrame()
                    if initial_error:
                        st.error(f"Could not load candles: {initial_error}")
                    elif not initial_closed.empty and len(valid_ema) < 2:
                        st.error("There are not enough completed candles yet for these EMA periods. Try a shorter slow EMA.")
                    elif not initial_closed.empty:
                        bot_started = datetime.now(timezone.utc).isoformat()
                        bot_state = {
                            "run_id": uuid.uuid4().hex, "symbol": forward_symbol, "interval": interval,
                            "provider": provider, "short_period": int(forward_short), "long_period": int(forward_long),
                            "units": float(forward_units), "starting_cash": float(forward_cash),
                            "cost_bps": float(forward_cost), "running": True,
                            "last_signal_bar": valid_ema.index[-1].isoformat(),
                            "position_entry_time": None, "position_entry_price": None, "realized_pnl": 0.0,
                            "latest_price": float(initial_bars["close"].iloc[-1]),
                            "latest_fast_ema": float(valid_ema["fast"].iloc[-1]),
                            "latest_slow_ema": float(valid_ema["slow"].iloc[-1]),
                            "latest_candle_time": valid_ema.index[-1].isoformat(),
                            "status": "Started. Waiting for a new completed candle and crossover.",
                            "started_at": bot_started,
                        }
                        save_forward_paper_run(bot_state)
                        st.session_state["forward_ema_paper_bot"] = bot_state
                    elif initial_closed.empty:
                        st.error("No usable candles came back. Try Yahoo Finance or check the ticker.")

    with st.expander("What this bot does and what its numbers mean"):
        st.markdown("""
It waits until a candle is finished. If the fast average crosses above the slow average, it records a pretend entry at the next candle's open. If the fast average crosses below, it records a pretend exit. It keeps at most one pretend position open and skips an entry that costs more than the pretend balance.

The bot starts watching from now; it does not turn an old crossover into a new trade. Its prediction is only the current EMA direction, not a reliable chance of profit. A trade can lose money while the bot waits for an opposite crossover. The selected cost estimate may be lower than real trading costs.
""")
    show_forward_paper_bot(api_key)

st.caption(f"Selected intraday price source: {provider}. Finnhub headlines can be loaded independently in the Market news tab. Third-party feeds may be delayed or incomplete; this research tool does not guarantee accuracy or provide financial advice.")

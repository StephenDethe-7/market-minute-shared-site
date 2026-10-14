from __future__ import annotations

from datetime import datetime, time as clock_time, timedelta, timezone
import os
import sqlite3
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


DATABASE_PATH = Path(os.environ.get("MARKET_MINUTE_DB", Path(__file__).with_name("market_minute.sqlite")))
SCREENERS = ("most_actives", "day_gainers", "day_losers", "most_shorted_stocks")
BENCHMARKS = {
    "^GSPC": "Index", "^IXIC": "Index", "^DJI": "Index", "^RUT": "Index", "^VIX": "Index",
    "SPY": "Index ETF", "QQQ": "Index ETF", "IWM": "Index ETF", "DIA": "Index ETF",
    "VTI": "Broad-market ETF", "VOO": "Broad-market ETF", "RSP": "Equal-weight ETF",
    "XLB": "Sector ETF", "XLC": "Sector ETF", "XLE": "Sector ETF", "XLF": "Sector ETF",
    "XLI": "Sector ETF", "XLK": "Sector ETF", "XLP": "Sector ETF", "XLRE": "Sector ETF",
    "XLU": "Sector ETF", "XLV": "Sector ETF", "XLY": "Sector ETF",
}


def init_research_db(path: Path = DATABASE_PATH) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS research_snapshots (
            run_at TEXT NOT NULL, ticker TEXT NOT NULL, asset_type TEXT NOT NULL,
            price REAL, day_change_pct REAL, day_volume REAL, avg_volume_30d REAL,
            relative_volume REAL, change_30d_pct REAL, change_90d_pct REAL,
            calls_volume REAL, puts_volume REAL, total_open_interest REAL,
            options_volume_multiple REAL, open_interest_change REAL,
            options_expiry TEXT, earnings_at TEXT, data_status TEXT NOT NULL,
            explanation TEXT, boring_explanation TEXT, unknowns TEXT,
            PRIMARY KEY (run_at, ticker)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS research_reviews (
            run_at TEXT NOT NULL, ticker TEXT NOT NULL, decision TEXT NOT NULL,
            reviewed_at TEXT NOT NULL, note TEXT DEFAULT '',
            PRIMARY KEY (run_at, ticker)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS research_outcomes (
            source_run_at TEXT NOT NULL, ticker TEXT NOT NULL, followup_run_at TEXT NOT NULL,
            flagged_price REAL, followup_price REAL, forward_return_pct REAL,
            PRIMARY KEY (source_run_at, ticker)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS paper_test_outcomes (
            source_run_at TEXT NOT NULL, ticker TEXT NOT NULL, signal_date TEXT NOT NULL,
            relative_volume REAL NOT NULL, avg_dollar_volume REAL NOT NULL,
            entry_at TEXT NOT NULL, entry_price REAL NOT NULL,
            exit_at TEXT NOT NULL, exit_price REAL NOT NULL,
            spy_entry REAL NOT NULL, spy_exit REAL NOT NULL,
            return_pct REAL NOT NULL, spy_return_pct REAL NOT NULL, excess_return_pct REAL NOT NULL,
            data_source TEXT NOT NULL,
            PRIMARY KEY (source_run_at, ticker)
        )""")


def _quote_fields(quote: dict) -> dict:
    def first(*keys):
        return next((quote[k] for k in keys if quote.get(k) is not None), None)
    return {
        "price": first("regularMarketPrice", "currentPrice", "postMarketPrice"),
        "day_change_pct": first("regularMarketChangePercent", "percentchange"),
        "day_volume": first("regularMarketVolume", "dayVolume", "dayvolume"),
        "avg_volume_30d": first("averageDailyVolume3Month", "averageDailyVolume10Day", "avgdailyvol3m"),
        "earnings_at": first("earningsTimestamp", "earningsTimestampStart"),
    }


def collect_universe(count: int = 50) -> tuple[dict[str, dict], list[str]]:
    universe: dict[str, dict] = {}
    errors: list[str] = []
    for screen_name in SCREENERS:
        try:
            result = yf.screen(screen_name, count=count)
            for quote in result.get("quotes", []):
                symbol = str(quote.get("symbol", "")).strip().upper()
                if symbol:
                    entry = universe.setdefault(symbol, {"asset_type": "US equity", "quote": {}})
                    entry["quote"].update(quote)
                    entry.setdefault("screens", []).append(screen_name)
        except Exception as exc:
            errors.append(f"{screen_name}: {type(exc).__name__}")
    try:
        result = yf.screen("top_etfs_us", count=count)
        for quote in result.get("quotes", []):
            symbol = str(quote.get("symbol", "")).strip().upper()
            if symbol:
                entry = universe.setdefault(symbol, {"asset_type": "ETF", "quote": {}})
                entry["quote"].update(quote)
                entry.setdefault("screens", []).append("top_etfs_us")
    except Exception as exc:
        errors.append(f"top_etfs_us: {type(exc).__name__}")
    if not any(item.get("quote") for item in universe.values()):
        raise RuntimeError("Yahoo Finance returned no stock or ETF screener data. Check the connection and try again. " + "; ".join(errors))
    for symbol, asset_type in BENCHMARKS.items():
        universe.setdefault(symbol, {"asset_type": asset_type, "quote": {}, "screens": ["curated benchmark list"]})
    return universe, errors


def _history_metrics(history: pd.DataFrame | None, quote_fields: dict) -> dict:
    result = {"history_status": "UNAVAILABLE", "relative_volume": None, "change_30d_pct": None, "change_90d_pct": None}
    try:
        if history is None or history.empty or "Close" not in history:
            return result
        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        volumes = pd.to_numeric(history["Volume"], errors="coerce").dropna() if "Volume" in history else pd.Series(dtype=float)
        if quote_fields.get("price") is None and not closes.empty:
            quote_fields["price"] = float(closes.iloc[-1])
        if quote_fields.get("day_volume") is None and not volumes.empty:
            quote_fields["day_volume"] = float(volumes.iloc[-1])
        if len(closes) >= 2:
            result["change_90d_pct"] = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)
        if len(closes) >= 22:
            result["change_30d_pct"] = float((closes.iloc[-1] / closes.iloc[-22] - 1) * 100)
        if len(volumes) >= 10:
            completed = volumes.iloc[:-1] if history.index[-1].date() == datetime.now().date() else volumes
            baseline = float(completed.tail(30).mean()) if not completed.empty else None
            current_volume = quote_fields.get("day_volume")
            if baseline and current_volume is not None and pd.notna(current_volume):
                result["relative_volume"] = float(current_volume) / baseline
                result["avg_volume_30d"] = baseline
        result["history_status"] = "OK"
    except Exception as exc:
        result["history_status"] = f"UNAVAILABLE ({type(exc).__name__})"
    return result


def _download_histories(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    histories: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for start in range(0, len(symbols), 40):
        batch = symbols[start:start + 40]
        try:
            data = yf.download(batch, period="3mo", interval="1d", auto_adjust=True,
                               group_by="ticker", progress=False, threads=False)
            if data is None or data.empty:
                errors.append(f"Daily history batch {start // 40 + 1}: no data returned")
                continue
            if isinstance(data.columns, pd.MultiIndex):
                ticker_level = next((level for level in range(data.columns.nlevels)
                                     if set(batch).intersection(map(str, data.columns.get_level_values(level)))), None)
                if ticker_level is None:
                    errors.append(f"Daily history batch {start // 40 + 1}: ticker columns were not recognized")
                    continue
                names = set(map(str, data.columns.get_level_values(ticker_level)))
                for symbol in batch:
                    if symbol in names:
                        histories[symbol] = data.xs(symbol, axis=1, level=ticker_level, drop_level=True)
            elif len(batch) == 1:
                histories[batch[0]] = data
        except Exception as exc:
            errors.append(f"Daily history batch {start // 40 + 1}: {type(exc).__name__}")
    return histories, errors


def _paper_test_history(data: pd.DataFrame, ticker_level: int | None, symbol: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        history = data.xs(symbol, axis=1, level=ticker_level, drop_level=True) if ticker_level is not None else data.copy()
    except (KeyError, ValueError):
        return pd.DataFrame()
    if isinstance(history.columns, pd.MultiIndex):
        levels = [level for level in range(history.columns.nlevels)
                  if "Close" in map(str, history.columns.get_level_values(level))]
        if not levels:
            return pd.DataFrame()
        history.columns = history.columns.get_level_values(levels[0])
    close_column = next((column for column in history.columns if str(column).lower() == "close"), None)
    if close_column is None:
        return pd.DataFrame()
    bars = pd.DataFrame({"close": pd.to_numeric(history[close_column], errors="coerce")}).dropna()
    index = pd.to_datetime(bars.index, errors="coerce")
    if getattr(index, "tz", None) is None:
        index = index.tz_localize("America/New_York")
    else:
        index = index.tz_convert("America/New_York")
    bars.index = index
    return bars.loc[~bars.index.isna()].sort_index()


def record_intraday_paper_outcomes(path: Path = DATABASE_PATH, now: datetime | None = None) -> int:
    """Measure the pre-set next-session 25-minute paper rule for eligible daily volume flags."""
    init_research_db(path)
    now_et = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    cutoff = (now_et.date() - timedelta(days=55)).isoformat()
    with sqlite3.connect(path) as conn:
        pending = conn.execute("""SELECT s.run_at,s.ticker,s.price,s.relative_volume,s.avg_volume_30d
            FROM research_snapshots s
            LEFT JOIN paper_test_outcomes p ON p.source_run_at=s.run_at AND p.ticker=s.ticker
            WHERE s.asset_type='US equity' AND s.explanation!='' AND s.relative_volume>=2
              AND s.price>=5 AND s.avg_volume_30d>0 AND s.price*s.avg_volume_30d>=20000000
              AND substr(s.run_at,1,10)>=? AND p.source_run_at IS NULL
            ORDER BY s.run_at ASC""", (cutoff,)).fetchall()
        completed = conn.execute("SELECT ticker,source_run_at FROM paper_test_outcomes ORDER BY source_run_at").fetchall()

    # Avoid repeatedly counting the same ticker during one short-lived volume event.
    last_counted: dict[str, datetime] = {}
    for ticker, run_at in completed:
        try:
            last_counted[ticker] = datetime.fromisoformat(run_at)
        except ValueError:
            continue
    eligible = []
    for row in pending:
        source_at = datetime.fromisoformat(row[0])
        signal_date = source_at.astimezone(ZoneInfo("America/New_York")).date()
        if signal_date >= now_et.date():
            continue
        prior = last_counted.get(row[1])
        if prior and (source_at - prior).days < 7:
            continue
        eligible.append((row, signal_date))
        last_counted[row[1]] = source_at
    if not eligible:
        return 0

    symbols = list(dict.fromkeys([row[0][1] for row in eligible] + ["SPY"]))
    try:
        data = yf.download(symbols, period="60d", interval="5m", auto_adjust=True,
                           group_by="ticker", progress=False, threads=False)
    except Exception:
        return 0
    if data is None or data.empty:
        return 0
    ticker_level = None
    if isinstance(data.columns, pd.MultiIndex):
        ticker_level = next((level for level in range(data.columns.nlevels)
                             if set(symbols).intersection(map(str, data.columns.get_level_values(level)))), None)
        if ticker_level is None:
            return 0
    bars_by_symbol = {symbol: _paper_test_history(data, ticker_level, symbol) for symbol in symbols}
    spy = bars_by_symbol.get("SPY", pd.DataFrame())
    if spy.empty:
        return 0

    made = 0
    with sqlite3.connect(path) as conn:
        for row, signal_date in eligible:
            run_at, ticker, _, relative_volume, avg_volume = row
            stock = bars_by_symbol.get(ticker, pd.DataFrame())
            if stock.empty:
                continue
            stock = stock.loc[stock.index.date > signal_date]
            if stock.empty:
                continue
            next_session = min(stock.index.date)
            spy_day = spy.loc[spy.index.date == next_session]
            stock_day = stock.loc[stock.index.date == next_session]
            def close_at_5m_close(day: pd.DataFrame, bar_start: clock_time) -> float | None:
                selected = day[(day.index.hour == bar_start.hour) & (day.index.minute == bar_start.minute)]
                if selected.empty:
                    return None
                value = float(selected.iloc[0]["close"])
                return value if np.isfinite(value) and value > 0 else None
            # Yahoo timestamps 5-minute bars by their start: 09:30 closes 09:35;
            # the 09:55 bar closes 10:00, 25 minutes after that entry close.
            entry = close_at_5m_close(stock_day, clock_time(9, 30))
            exit_price = close_at_5m_close(stock_day, clock_time(9, 55))
            spy_entry = close_at_5m_close(spy_day, clock_time(9, 30))
            spy_exit = close_at_5m_close(spy_day, clock_time(9, 55))
            if None in (entry, exit_price, spy_entry, spy_exit):
                continue
            ret = (exit_price / entry - 1) * 100
            spy_ret = (spy_exit / spy_entry - 1) * 100
            session_date = next_session.isoformat()
            entry_at = datetime.combine(next_session, clock_time(9, 35), ZoneInfo("America/New_York")).isoformat()
            exit_at = datetime.combine(next_session, clock_time(10, 0), ZoneInfo("America/New_York")).isoformat()
            conn.execute("""INSERT OR IGNORE INTO paper_test_outcomes VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_at, ticker, session_date, float(relative_volume), float(row[2] * avg_volume),
                 entry_at, entry, exit_at, exit_price,
                 spy_entry, spy_exit, ret, spy_ret, ret - spy_ret, "Yahoo Finance 5-minute bars"))
            made += 1
        conn.commit()
    return made


def _options_snapshot(symbol: str) -> dict:
    out = {"calls_volume": None, "puts_volume": None, "total_open_interest": None, "options_expiry": None,
           "options_status": "UNAVAILABLE"}
    try:
        ticker = yf.Ticker(symbol)
        expirations = list(ticker.options or [])
        if not expirations:
            out["options_status"] = "UNAVAILABLE (no listed options returned)"
            return out
        today = datetime.now(timezone.utc).date()
        future = [e for e in expirations if datetime.strptime(e, "%Y-%m-%d").date() >= today]
        expiry = future[0] if future else expirations[0]
        chain = ticker.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts
        def total(chain: pd.DataFrame, column: str) -> float:
            if column not in chain:
                return 0.0
            return float(pd.to_numeric(chain[column], errors="coerce").fillna(0).sum())
        out.update({
            "calls_volume": total(calls, "volume"),
            "puts_volume": total(puts, "volume"),
            "total_open_interest": total(calls, "openInterest") + total(puts, "openInterest"),
            "options_expiry": expiry,
            "options_status": "OK · nearest expiry only",
        })
    except Exception as exc:
        out["options_status"] = f"UNAVAILABLE ({type(exc).__name__})"
    return out


def _previous_options(path: Path, ticker: str, expiry: str, before_date: str) -> list[tuple[float | None, float | None, float | None]]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("""SELECT calls_volume, puts_volume, total_open_interest
            FROM research_snapshots WHERE ticker=? AND options_expiry=? AND substr(run_at,1,10) < ?
            ORDER BY run_at DESC LIMIT 20""", (ticker, expiry, before_date)).fetchall()
    return rows


def run_market_research(path: Path = DATABASE_PATH, option_limit: int = 25, screener_count: int = 50,
                        progress: Callable[[int, int, str], None] | None = None) -> tuple[str, pd.DataFrame, pd.DataFrame, list[str]]:
    init_research_db(path)
    started = datetime.now(timezone.utc)
    run_at = started.isoformat(timespec="microseconds")
    run_day = started.astimezone().date().isoformat()
    universe, source_errors = collect_universe(count=screener_count)
    histories, history_errors = _download_histories(list(universe))
    source_errors.extend(history_errors)
    rows: list[dict] = []
    total = len(universe)
    for index, (symbol, item) in enumerate(universe.items(), start=1):
        quote = item.get("quote", {})
        fields = _quote_fields(quote)
        history = _history_metrics(histories.get(symbol), fields)
        fields.update(history)
        fields["avg_volume_30d"] = history.get("avg_volume_30d", fields.get("avg_volume_30d"))
        earnings = fields.get("earnings_at")
        if earnings is not None:
            try:
                earnings = datetime.fromtimestamp(float(earnings), timezone.utc).astimezone().strftime("%b %d %I:%M %p %Z")
            except (TypeError, ValueError, OSError):
                earnings = "UNAVAILABLE"
        rows.append({"Ticker": symbol, "Type": item["asset_type"], "Price": fields.get("price"),
                     "Day change %": fields.get("day_change_pct"), "Day volume": fields.get("day_volume"),
                     "30d avg volume": fields.get("avg_volume_30d"), "Relative volume": fields.get("relative_volume"),
                     "Relative volume data": "UNAVAILABLE · 30-day history or current volume missing" if history.get("relative_volume") is None else "AVAILABLE · own 30-day average",
                     "30d move %": fields.get("change_30d_pct"), "90d move %": fields.get("change_90d_pct"),
                     "Calls volume": None, "Puts volume": None, "Put/call": None, "Open interest": None,
                     "OI change": None, "OI change data": "UNAVAILABLE · needs an earlier same-expiry snapshot",
                     "Options volume vs own history": None,
                     "Options baseline data": "UNAVAILABLE · needs 3 prior daily snapshots for the same expiry",
                     "Options expiry": "UNAVAILABLE",
                     "Earnings": earnings or "UNAVAILABLE", "Price data": history["history_status"],
                     "Options data": "UNAVAILABLE · sampled after stock screen", "Screen": ", ".join(item.get("screens", [])),
                     "Score": 0.0, "Why flagged": "", "Boring explanation": "",
                     "Unknowns": "Options intent, opening/closing side, counterpart position, and catalyst may be unknown."})
        if progress:
            progress(index, total, symbol)

    frame = pd.DataFrame(rows)
    # Options endpoints are queried only for the most active screened symbols to keep free-source usage bounded.
    option_candidates = frame.sort_values("Relative volume", ascending=False, na_position="last").head(option_limit)["Ticker"].tolist()
    for symbol in option_candidates:
        snapshot = _options_snapshot(symbol)
        match = frame["Ticker"] == symbol
        frame.loc[match, "Calls volume"] = snapshot["calls_volume"]
        frame.loc[match, "Puts volume"] = snapshot["puts_volume"]
        frame.loc[match, "Open interest"] = snapshot["total_open_interest"]
        frame.loc[match, "Options expiry"] = snapshot["options_expiry"] or "UNAVAILABLE"
        frame.loc[match, "Options data"] = snapshot["options_status"]
        calls, puts = snapshot["calls_volume"], snapshot["puts_volume"]
        frame.loc[match, "Put/call"] = (puts / calls) if calls and calls > 0 and puts is not None else None
        if calls is None or puts is None:
            frame.loc[match, "Options baseline data"] = "UNAVAILABLE · options chain missing"
            frame.loc[match, "OI change data"] = "UNAVAILABLE · options chain missing"
            continue
        history = _previous_options(path, symbol, snapshot["options_expiry"], run_day)
        valid_volume = [float(r[0] or 0) + float(r[1] or 0) for r in history if r[0] is not None and r[1] is not None]
        total_volume = float(calls + puts)
        if len(valid_volume) >= 3 and np.median(valid_volume) > 0:
            frame.loc[match, "Options volume vs own history"] = total_volume / float(np.median(valid_volume))
            frame.loc[match, "Options baseline data"] = f"AVAILABLE · {len(valid_volume)} prior daily snapshots"
        valid_oi = [float(r[2]) for r in history if r[2] is not None]
        if valid_oi:
            frame.loc[match, "OI change"] = float(snapshot["total_open_interest"] or 0) - valid_oi[0]
            frame.loc[match, "OI change data"] = "AVAILABLE · compared with prior same-expiry snapshot"

    # A transparent, deterministic flagger ranks unusual activity; it never labels options trades bullish/bearish.
    for idx, row in frame.iterrows():
        relative_volume = row["Relative volume"]
        options_multiple = row["Options volume vs own history"]
        relative_score = float(relative_volume) if pd.notna(relative_volume) else 0.0
        options_score = float(options_multiple) if pd.notna(options_multiple) else 0.0
        score = max(relative_score, options_score)
        reasons = []
        if pd.notna(relative_volume) and relative_volume >= 2:
            reasons.append(f"Share volume is {relative_volume:.1f}× its own 30-day average")
        if pd.notna(options_multiple) and options_multiple >= 2.5:
            reasons.append(f"Nearest-expiry options volume is {options_multiple:.1f}× this ticker's saved baseline")
        frame.at[idx, "Score"] = score
        if reasons:
            frame.at[idx, "Why flagged"] = "; ".join(reasons)
            frame.at[idx, "Boring explanation"] = "A market-wide move, scheduled news, routine hedging, or spread activity could explain the volume."

    flags = frame[frame["Why flagged"] != ""].sort_values("Score", ascending=False).head(5).copy()
    if flags.empty:
        frame["Flag rank"] = None
    else:
        flags.insert(0, "Rank", range(1, len(flags) + 1))
        rank_map = dict(zip(flags["Ticker"], flags["Rank"]))
        frame["Flag rank"] = frame["Ticker"].map(rank_map)

    with sqlite3.connect(path) as conn:
        for _, row in frame.iterrows():
            def numeric(key):
                value = row.get(key)
                return None if value is None or pd.isna(value) else float(value)
            conn.execute("""INSERT OR REPLACE INTO research_snapshots VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_at, row["Ticker"], row["Type"], numeric("Price"), numeric("Day change %"), numeric("Day volume"),
                 numeric("30d avg volume"), numeric("Relative volume"), numeric("30d move %"), numeric("90d move %"),
                 numeric("Calls volume"), numeric("Puts volume"), numeric("Open interest"),
                 numeric("Options volume vs own history"), numeric("OI change"),
                 None if row["Options expiry"] == "UNAVAILABLE" else row["Options expiry"],
                 None if row["Earnings"] == "UNAVAILABLE" else row["Earnings"],
                 f"price={row['Price data']}; relative_volume={row['Relative volume data']}; options={row['Options data']}; options_baseline={row['Options baseline data']}; oi_change={row['OI change data']}", row["Why flagged"],
                 row["Boring explanation"], row["Unknowns"]))
        conn.commit()
    _record_followups(path, run_at, frame)
    try:
        record_intraday_paper_outcomes(path, started)
    except Exception as exc:
        source_errors.append(f"Paper-test price history unavailable: {type(exc).__name__}")
    return run_at, frame, flags, source_errors


def _record_followups(path: Path, run_at: str, frame: pd.DataFrame) -> None:
    with sqlite3.connect(path) as conn:
        for _, row in frame.iterrows():
            price = row.get("Price")
            if price is None or pd.isna(price) or float(price) <= 0:
                continue
            previous = conn.execute("""SELECT s.run_at,s.price FROM research_snapshots s
                LEFT JOIN research_outcomes o ON o.source_run_at=s.run_at AND o.ticker=s.ticker
                WHERE s.ticker=? AND s.explanation!='' AND s.run_at<?
                AND substr(s.run_at,1,10)<substr(?,1,10) AND o.source_run_at IS NULL
                ORDER BY s.run_at DESC LIMIT 1""", (row["Ticker"], run_at, run_at)).fetchone()
            if not previous or previous[1] is None or float(previous[1]) <= 0:
                continue
            change = (float(price) / float(previous[1]) - 1) * 100
            conn.execute("""INSERT OR IGNORE INTO research_outcomes
                (source_run_at,ticker,followup_run_at,flagged_price,followup_price,forward_return_pct)
                VALUES (?,?,?,?,?,?)""", (previous[0], row["Ticker"], run_at, float(previous[1]), float(price), change))
        conn.commit()


def recent_runs(path: Path = DATABASE_PATH, limit: int = 20) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query("""SELECT run_at, ticker, asset_type, price, relative_volume,
                calls_volume, puts_volume, options_volume_multiple, open_interest_change,
                data_status, explanation, boring_explanation, unknowns
                FROM research_snapshots ORDER BY run_at DESC LIMIT ?""", conn, params=(limit,))
    except sqlite3.OperationalError:
        return pd.DataFrame()


def latest_research(path: Path = DATABASE_PATH) -> tuple[str | None, pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        return None, pd.DataFrame(), pd.DataFrame()
    try:
        with sqlite3.connect(path) as conn:
            latest = conn.execute("SELECT MAX(run_at) FROM research_snapshots").fetchone()[0]
            if not latest:
                return None, pd.DataFrame(), pd.DataFrame()
            data = pd.read_sql_query("""SELECT ticker AS Ticker, asset_type AS Type, price AS Price,
                day_change_pct AS 'Day change %', day_volume AS 'Day volume', avg_volume_30d AS '30d avg volume',
                relative_volume AS 'Relative volume', change_30d_pct AS '30d move %', change_90d_pct AS '90d move %',
                calls_volume AS 'Calls volume', puts_volume AS 'Puts volume', total_open_interest AS 'Open interest',
                options_volume_multiple AS 'Options volume vs own history', open_interest_change AS 'OI change',
                options_expiry AS 'Options expiry', earnings_at AS Earnings, data_status AS 'Data quality',
                explanation AS 'Why flagged', boring_explanation AS 'Boring explanation', unknowns AS Unknowns
                FROM research_snapshots WHERE run_at=?""", conn, params=(latest,))
            reviews = pd.read_sql_query("SELECT ticker AS Ticker, decision AS Decision FROM research_reviews WHERE run_at=?", conn, params=(latest,))
        if data.empty:
            return latest, data, pd.DataFrame()
        data["Score"] = data[["Relative volume", "Options volume vs own history"]].fillna(0).max(axis=1)
        data["Relative volume data"] = np.where(data["Relative volume"].notna(),
                                                "AVAILABLE · own 30-day average",
                                                "UNAVAILABLE · 30-day history or current volume missing")
        data["Options baseline data"] = np.where(data["Options volume vs own history"].notna(),
                                                  "AVAILABLE · 3+ prior snapshots",
                                                  "UNAVAILABLE · needs 3 prior daily snapshots for the same expiry")
        data["OI change data"] = np.where(data["OI change"].notna(),
                                          "AVAILABLE · prior same-expiry snapshot",
                                          "UNAVAILABLE · needs an earlier same-expiry snapshot")
        flags = data[data["Why flagged"].fillna("") != ""].sort_values("Score", ascending=False).head(5).copy()
        flags.insert(0, "Rank", range(1, len(flags) + 1))
        if not reviews.empty:
            flags = flags.merge(reviews, on="Ticker", how="left")
        else:
            flags["Decision"] = "AWAITING YOUR REVIEW"
        flags["Decision"] = flags["Decision"].fillna("AWAITING YOUR REVIEW")
        return latest, data, flags
    except sqlite3.OperationalError:
        return None, pd.DataFrame(), pd.DataFrame()


def recent_followups(path: Path = DATABASE_PATH, limit: int = 50) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query("""SELECT ticker AS Ticker, source_run_at AS 'Flag date',
                flagged_price AS 'Flag price', followup_run_at AS 'Follow-up date',
                followup_price AS 'Follow-up price', forward_return_pct AS 'Move after flag %'
                FROM research_outcomes ORDER BY followup_run_at DESC LIMIT ?""", conn, params=(limit,))
    except sqlite3.OperationalError:
        return pd.DataFrame()


def paper_test_results(path: Path = DATABASE_PATH, limit: int = 500) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        # Existing installs may have a database created before paper testing was added.
        # Initialize the additive tables before querying so the page opens cleanly.
        init_research_db(path)
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query("""SELECT signal_date AS 'Signal date', ticker AS Ticker,
                relative_volume AS 'Share volume multiple', avg_dollar_volume AS 'Average dollar volume',
                entry_at AS 'Paper entry', entry_price AS 'Entry price', exit_at AS 'Paper exit',
                exit_price AS 'Exit price', return_pct AS '25-minute return %',
                spy_return_pct AS 'SPY 25-minute return %', excess_return_pct AS 'Difference vs SPY %'
                FROM paper_test_outcomes ORDER BY signal_date DESC, ticker LIMIT ?""", conn, params=(limit,))
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame()


def review_candidate(path: Path, run_at: str, ticker: str, decision: str, note: str = "") -> None:
    if decision not in {"APPROVED FOR PAPER TRACKING", "REJECTED"}:
        raise ValueError("Paper review decision must be approved or rejected.")
    with sqlite3.connect(path) as conn:
        conn.execute("""INSERT OR REPLACE INTO research_reviews (run_at,ticker,decision,reviewed_at,note)
            VALUES (?,?,?,?,?)""", (run_at, ticker, decision, datetime.now(timezone.utc).isoformat(), note))

import io
import math
import zipfile
from datetime import datetime, time, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


APP_TITLE = "Dataset Binance API downloader (DBAD)"

ASSETS = ["BTC", "ETH", "BNB", "SOL", "LINK"]
STABLES = ["USDT", "USDC", "FDUSD", "TUSD", "DAI"]

# User-facing timeframes (target). Some are not directly supported by Binance.
TARGET_TIMEFRAMES = [
    "1s", "5s", "10s", "30s",
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

DATA_SOURCES = [
    "Binance API (REST)",
    "Binance Public Data (data.binance.vision)"
]

PUBLIC_BASE_URL = "https://data.binance.vision"

def interval_to_ms(interval: str) -> int | None:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "s":
        return value * 1000
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    if unit == "w":
        return value * 604_800_000
    if unit == "M":
        return value * 2_592_000_000  # approx 30d for estimation only
    return None


def interval_to_pandas_freq(interval: str) -> str:
    unit = interval[-1]
    value = interval[:-1]
    if unit == "s":
        return f"{value}S"
    if unit == "m":
        return f"{value}T"
    if unit == "h":
        return f"{value}H"
    if unit == "d":
        return f"{value}D"
    if unit == "w":
        return f"{value}W"
    if unit == "M":
        return f"{value}M"
    raise ValueError(f"Unsupported interval: {interval}")


def public_interval(interval: str) -> str:
    return "1mo" if interval == "1M" else interval


def daterange_days(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, market: str, progress_cb, log_cb):
    if market == "spot":
        endpoint = "https://api.binance.com/api/v3/klines"
    else:
        endpoint = "https://fapi.binance.com/fapi/v1/klines"

    rows = []
    current = start_ms
    step_ms = interval_to_ms(interval) or 1
    limit = 1000

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(endpoint, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance error {resp.status_code}: {resp.text}")
        data = resp.json()
        if isinstance(data, dict) and "code" in data:
            raise RuntimeError(f"Binance error {data.get('code')}: {data.get('msg')}")
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        current = last_open + step_ms
        if progress_cb:
            progress_cb(current)
        if log_cb:
            log_cb(f"Fetched {len(data)} rows (total {len(rows)})")
        if len(data) < limit:
            break

    return rows


def fetch_spot_agg_trades(symbol: str, start_ms: int, end_ms: int, progress_cb, log_cb) -> list:
    endpoint = "https://api.binance.com/api/v3/aggTrades"
    rows = []
    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }

    while True:
        resp = requests.get(endpoint, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance error {resp.status_code}: {resp.text}")
        data = resp.json()
        if isinstance(data, dict) and "code" in data:
            raise RuntimeError(f"Binance error {data.get('code')}: {data.get('msg')}")
        if not data:
            break
        rows.extend(data)
        last_trade = data[-1]
        last_id = last_trade["a"]
        last_time = last_trade["T"]
        if progress_cb:
            progress_cb(last_time)
        if log_cb:
            log_cb(f"Fetched {len(data)} trades (total {len(rows)})")
        if last_time >= end_ms or len(data) < 1000:
            break
        params = {"symbol": symbol, "fromId": last_id + 1, "limit": 1000}

    return rows


def build_df(rows: list) -> pd.DataFrame:
    columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df.set_index("Open time", inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_df_from_trades(rows: list, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    df = df.set_index("T").sort_index()
    if start_ms is not None:
        df = df[df.index >= pd.to_datetime(start_ms, unit="ms", utc=True)]
    if end_ms is not None:
        df = df[df.index <= pd.to_datetime(end_ms, unit="ms", utc=True)]
    ohlc = df["p"].resample("1S").ohlc()
    volume = df["q"].resample("1S").sum()
    ohlc.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    ohlc["Volume"] = volume
    ohlc = ohlc.dropna()
    return ohlc


def resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    freq = interval_to_pandas_freq(target_interval)
    return (
        df.resample(freq)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def download_public_zip(url: str) -> io.BytesIO | None:
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"Binance public data error {resp.status_code}: {resp.text}")
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
            buf.write(chunk)
    buf.seek(0)
    return buf


def read_kline_zip(buf: io.BytesIO) -> pd.DataFrame:
    with zipfile.ZipFile(buf) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, header=None)
    columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
    df = df.iloc[:, : len(columns)]
    df.columns = columns[: df.shape[1]]
    # Spot public data can use microseconds for timestamps from 2025 onward.
    sample_ts = pd.to_numeric(df["Open time"].iloc[0], errors="coerce")
    if pd.notna(sample_ts) and sample_ts >= 10**14:
        ts_unit = "us"
    else:
        ts_unit = "ms"
    df["Open time"] = pd.to_datetime(df["Open time"], unit=ts_unit, utc=True)
    df.set_index("Open time", inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_public_kline_url(market_key: str, symbol: str, interval: str, day: datetime) -> str:
    interval_pub = public_interval(interval)
    date_str = day.strftime("%Y-%m-%d")
    if market_key == "spot":
        base = f"{PUBLIC_BASE_URL}/data/spot/daily/klines/{symbol}/{interval_pub}"
    else:
        base = f"{PUBLIC_BASE_URL}/data/futures/um/daily/klines/{symbol}/{interval_pub}"
    return f"{base}/{symbol}-{interval_pub}-{date_str}.zip"


st.set_page_config(page_title=APP_TITLE, page_icon="📥", layout="wide")
st.title("📥 Dataset Binance API downloader (DBAD)")
st.caption("Télécharge l'historique OHLCV depuis l'API Binance (spot ou perpetual) et sauvegarde en CSV.")

with st.sidebar:
    st.header("⚙️ Configuration")
    data_source = st.selectbox("Data source", DATA_SOURCES, index=1)
    market = st.selectbox("Market", ["Spot", "Perpetual (USD-M)"])
    asset = st.selectbox("Asset", ASSETS, index=0)
    stable = st.selectbox("Stable coin", STABLES, index=0)
    symbol = f"{asset}{stable}"
    st.text_input("Symbol", value=symbol, disabled=True)

    start_date = st.date_input("Start date", value=None)
    end_date = st.date_input("End date", value=None)
    timeframe = st.selectbox("Timeframe", TARGET_TIMEFRAMES, index=1)

    st.divider()
    start_download = st.button("⬇️ Start download", type="primary", use_container_width=True)

status_box = st.empty()
progress_bar = st.progress(0)
log_box = st.empty()

if start_download:
    logs = []

    def log(msg: str):
        logs.append(msg)
        log_box.text("\n".join(logs[-20:]))

    if not start_date or not end_date:
        st.error("Veuillez sélectionner une date de début et une date de fin.")
    elif end_date < start_date:
        st.error("La date de fin doit être postérieure à la date de début.")
    else:
        market_key = "spot" if market == "Spot" else "futures"
        target_interval = timeframe
        fetch_interval = "1s"
        resample_needed = target_interval != "1s"

        start_dt = datetime.combine(start_date, time(0, 0), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, time(23, 59, 59), tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        interval_ms = interval_to_ms(fetch_interval)
        if interval_ms:
            est = max(1, math.ceil((end_ms - start_ms) / interval_ms))
            log(f"Estimated candles: {est:,} at {fetch_interval}")
            if est > 1_000_000:
                st.warning("Très grand volume de données. Réduisez la plage si nécessaire.")

        if resample_needed:
            st.info(f"Téléchargement en {fetch_interval} puis resampling vers {target_interval}.")

        status_box.info("Connexion à Binance...")

        def progress_cb(current_ms: int):
            progress = min(1.0, (current_ms - start_ms) / max(1, (end_ms - start_ms)))
            progress_bar.progress(progress)

        try:
            status_box.info("Téléchargement en cours...")
            if data_source == "Binance Public Data (data.binance.vision)":
                log("Source: data.binance.vision (archives daily).")
                day_start = datetime.combine(start_date, time(0, 0), tzinfo=timezone.utc)
                day_end = datetime.combine(end_date, time(0, 0), tzinfo=timezone.utc)
                total_days = max(1, (day_end - day_start).days + 1)
                dfs = []
                for idx, day in enumerate(daterange_days(day_start, day_end), start=1):
                    progress_bar.progress(min(1.0, idx / total_days))
                    url = build_public_kline_url(market_key, symbol, fetch_interval, day)
                    buf = download_public_zip(url)
                    if buf is None:
                        log(f"Missing archive for {day.date()} at {fetch_interval}")
                        if target_interval != fetch_interval:
                            fallback_url = build_public_kline_url(market_key, symbol, target_interval, day)
                            buf = download_public_zip(fallback_url)
                            if buf is None:
                                log(f"Missing fallback archive for {day.date()} at {target_interval}")
                                continue
                            log(f"Fallback to {target_interval} for {day.date()}")
                            df_part = read_kline_zip(buf)
                        else:
                            continue
                    else:
                        df_part = read_kline_zip(buf)
                    if not df_part.empty:
                        dfs.append(df_part)
                        log(f"Loaded {len(df_part)} rows for {day.date()}")
                if not dfs:
                    status_box.error("Aucune donnée reçue depuis data.binance.vision.")
                    st.stop()
                df = pd.concat(dfs).sort_index()
                df = df[(df.index >= pd.to_datetime(start_ms, unit="ms", utc=True)) & (df.index <= pd.to_datetime(end_ms, unit="ms", utc=True))]
                log(f"Rows fetched: {len(df)}")
            else:
                if market_key == "spot":
                    log("Mode spot: récupération des trades agrégés pour construire des bougies 1s.")
                    rows = fetch_spot_agg_trades(symbol, start_ms, end_ms, progress_cb, log)
                    if not rows:
                        status_box.error("Aucune donnée reçue. Vérifiez le symbole ou la période.")
                        st.stop()
                    df = build_df_from_trades(rows, start_ms=start_ms, end_ms=end_ms)
                    log(f"Rows fetched (1s bars): {len(df)}")
                else:
                    rows = fetch_klines(symbol, fetch_interval, start_ms, end_ms, market_key, progress_cb, log)
                    if not rows:
                        status_box.error("Aucune donnée reçue. Vérifiez le symbole ou la période.")
                        st.stop()
                    df = build_df(rows)
                    log(f"Rows fetched: {len(df)}")

            if resample_needed:
                status_box.info("Resampling...")
                df = resample_ohlcv(df, target_interval)
                log(f"Rows after resample: {len(df)}")

            base_dir = Path(__file__).resolve().parent
            folder = base_dir / "Data" / "DBAD" / market_key / symbol
            folder.mkdir(parents=True, exist_ok=True)
            file_name = f"Binance_{symbol}_{start_date}_{end_date}_{target_interval}.csv"
            output_path = folder / file_name
            df.to_csv(output_path)

            status_box.success("Téléchargement terminé ✅")
            st.success(f"Fichier sauvegardé: {output_path}")
            st.write("Aperçu des données:")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as exc:
            status_box.error(f"Erreur: {exc}")

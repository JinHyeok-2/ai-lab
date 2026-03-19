#!/usr/bin/env python3
# ETH 과거 데이터 수집 및 전처리
# 실거래 바이낸스 API로 장기 데이터 수집 (klines는 공개 데이터)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pandas_ta as ta
from binance.client import Client
from config import API_KEY, API_SECRET

# 데이터 수집은 항상 실거래 API 사용 (공개 klines, 1년치 가능)
def get_real_client() -> Client:
    return Client(API_KEY, API_SECRET, testnet=False)

INTERVAL_MAP = {
    "15m": {"label": "15분봉",  "days": 90},
    "30m": {"label": "30분봉",  "days": 365},
    "1h":  {"label": "1시간봉", "days": 365},
    "2h":  {"label": "2시간봉", "days": 365},
    "4h":  {"label": "4시간봉", "days": 500},
    "1d":  {"label": "일봉",    "days": 730},
}

def fetch_eth_data(interval: str = "1h", days: int = None) -> pd.DataFrame:
    """ETH/USDT 과거 데이터 수집"""
    client = get_real_client()

    if days is None:
        days = INTERVAL_MAP.get(interval, {}).get("days", 365)

    label = INTERVAL_MAP.get(interval, {}).get("label", interval)
    print(f"[{label}] 데이터 수집 중... ({days}일치)")

    from datetime import datetime, timedelta
    start_str = (datetime.now() - timedelta(days=days)).strftime("%d %b, %Y")

    raw = client.get_historical_klines(
        symbol="ETHUSDT",
        interval=interval,
        start_str=start_str
    )

    df = pd.DataFrame(raw, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df[["time","open","high","low","close","volume"]].copy()

    # 기술적 지표
    df["rsi"]       = ta.rsi(df["close"], length=14)
    macd            = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]      = macd["MACD_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    bb              = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"]  = bb["BBU_20_2.0_2.0"]
    df["bb_lower"]  = bb["BBL_20_2.0_2.0"]
    df["bb_pct"]    = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["ema20"]     = ta.ema(df["close"], length=20)
    df["ema50"]     = ta.ema(df["close"], length=50)
    df["atr"]       = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"수집 완료: {len(df)}개 캔들 ({df['time'].iloc[0].date()} ~ {df['time'].iloc[-1].date()})")
    return df


def save_data(df: pd.DataFrame, interval: str = "1h"):
    path = Path(__file__).parent / f"eth_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"저장: {path}")
    return path


def load_data(interval: str = "1h") -> pd.DataFrame:
    path = Path(__file__).parent / f"eth_{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 데이터 수집 필요")
    df = pd.read_csv(path, parse_dates=["time"])
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1h", choices=["1h","4h","1d"])
    args = parser.parse_args()

    df = fetch_eth_data(interval=args.interval)
    save_data(df, args.interval)
    print(df.tail(3).to_string())

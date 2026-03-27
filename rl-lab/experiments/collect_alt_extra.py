#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# 추가 알트코인 30m 데이터 수집 — 실거래 종목 + 주요 대형 알트
# conda run -n ai-lab python collect_alt_extra.py

import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "rl" / "alt_30m"
API_URL = "https://fapi.binance.com/fapi/v1/klines"

# 추가 수집 대상 (기존 14종 제외)
# 실거래 종목: ONT, ZEC, FIL, JTO, LIGHT
# 주요 대형 알트: AAVE, UNI, ATOM, FTM, ETC, TIA, SEI, DYDX, PEPE(재시도), MATIC→POL
EXTRA_SYMBOLS = [
    "ONTUSDT", "ZECUSDT", "FILUSDT", "JTOUSDT",
    "AAVEUSDT", "UNIUSDT", "ATOMUSDT", "FTMUSDT",
    "ETCUSDT", "TIAUSDT", "SEIUSDT", "DYDXUSDT",
    "PEPEUSDT", "MATICUSDT",
]

START_DATE = "2024-01-01"


def fetch_candles(symbol, start_ms, limit=1000):
    params = {"symbol": symbol, "interval": "30m", "startTime": start_ms, "limit": limit}
    resp = requests.get(API_URL, params=params, timeout=15)
    data = resp.json()
    if not data or isinstance(data, dict):
        return []
    return [{"time": pd.Timestamp(k[0], unit="ms"),
             "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4]),
             "volume": float(k[5])} for k in data]


def recalc_features(df):
    df["rsi"] = ta.rsi(df["close"], length=14).fillna(50)
    _macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = _macd["MACD_12_26_9"].fillna(0)
    df["macd_hist"] = _macd["MACDh_12_26_9"].fillna(0)
    _bb = ta.bbands(df["close"], length=20, std=2)
    bbu = [c for c in _bb.columns if c.startswith("BBU")][0]
    bbl = [c for c in _bb.columns if c.startswith("BBL")][0]
    df["bb_upper"] = _bb[bbu].fillna(df["close"])
    df["bb_lower"] = _bb[bbl].fillna(df["close"])
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan).fillna(1)
    df["bb_pct"] = ((df["close"] - df["bb_lower"]) / bb_range).fillna(0.5).clip(0, 1)
    df["ema20"] = ta.ema(df["close"], length=20).fillna(df["close"])
    df["ema50"] = ta.ema(df["close"], length=50).fillna(df["close"])
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14).fillna(0)
    df["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()).fillna(1.0)
    return df


def download_symbol(symbol):
    name = symbol.replace("USDT", "").lower()
    csv_path = DATA_DIR / f"{name}_30m.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["time"])
        print(f"  [{symbol}] 이미 존재 ({len(existing)}캔들) — 스킵")
        return csv_path, len(existing)

    start_ms = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    print(f"  [{symbol}] 신규 수집: {START_DATE}~")

    all_candles = []
    end_ms = int(time.time() * 1000)
    while start_ms < end_ms:
        candles = fetch_candles(symbol, start_ms)
        if not candles:
            break
        all_candles.extend(candles)
        start_ms = int(candles[-1]["time"].timestamp() * 1000) + 30 * 60 * 1000
        if len(candles) < 1000:
            break
        time.sleep(0.15)

    if not all_candles:
        print(f"  [{symbol}] 데이터 없음!")
        return csv_path, 0

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    df = recalc_features(df)
    df.to_csv(csv_path, index=False)
    print(f"  [{symbol}] 저장: {len(df)}캔들 ({df['time'].iloc[0]} ~ {df['time'].iloc[-1]})")
    return csv_path, len(df)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== 추가 알트코인 데이터 수집 ({len(EXTRA_SYMBOLS)}종) ===\n")

    ok, fail = 0, 0
    for sym in EXTRA_SYMBOLS:
        try:
            _, count = download_symbol(sym)
            if count > 0:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [{sym}] 에러: {e}")
            fail += 1

    total = len(list(DATA_DIR.glob("*_30m.csv")))
    print(f"\n결과: 성공 {ok}, 실패 {fail}")
    print(f"전체 alt_30m 데이터: {total}종")


if __name__ == "__main__":
    main()

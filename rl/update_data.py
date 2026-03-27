#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# RL 학습 데이터 자동 갱신 — 바이낸스에서 최신 30m 캔들 다운로드
# 크론 등록: crontab -e → 0 1 * * * conda run -n ai-lab python /home/hyeok/01.APCC/00.ai-lab/rl/update_data.py
# ETH + BTC + ALT 26종 동시 갱신

import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent
ALT_DATA_DIR = DATA_DIR / "alt_30m"

# 메인 심볼 (ETH/BTC)
SYMBOLS = {
    "ETHUSDT": DATA_DIR / "eth_30m_v41.csv",
    "BTCUSDT": DATA_DIR / "btc_30m.csv",
}

# 알트코인 26종 (alt_30m/ 내 모든 코인, PEPE/MATIC 제외)
ALT_SYMBOLS = [
    # 기존 14종
    "SOLUSDT", "DOGEUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "NEARUSDT", "ARBUSDT", "OPUSDT",
    "SUIUSDT", "APTUSDT", "WIFUSDT", "TAOUSDT",
    # 추가 12종
    "ONTUSDT", "ZECUSDT", "FILUSDT", "JTOUSDT",
    "AAVEUSDT", "UNIUSDT", "ATOMUSDT", "FTMUSDT",
    "ETCUSDT", "TIAUSDT", "SEIUSDT", "DYDXUSDT",
]
API_URL = "https://fapi.binance.com/fapi/v1/klines"


def fetch_candles(symbol, start_ms, limit=1000):
    """바이낸스 선물 30m 캔들 다운로드"""
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
    """파생 컬럼 계산 (rsi, macd, bb, ema, atr, vol_ratio)"""
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


def update_symbol(symbol, csv_path):
    """단일 심볼 데이터 갱신"""
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["time"])
        last_time = df["time"].iloc[-1]
        start_ms = int((last_time + pd.Timedelta(minutes=30)).timestamp() * 1000)
        print(f"[{symbol}] 기존 {len(df)}캔들, 마지막: {last_time}")
    else:
        # 신규 생성: 2023-12-21부터
        df = pd.DataFrame()
        start_ms = int(pd.Timestamp("2023-12-21").timestamp() * 1000)
        print(f"[{symbol}] 신규 생성")

    # 다운로드
    all_new = []
    end_ms = int(time.time() * 1000)
    while start_ms < end_ms:
        candles = fetch_candles(symbol, start_ms)
        if not candles:
            break
        all_new.extend(candles)
        start_ms = int(candles[-1]["time"].timestamp() * 1000) + 30 * 60 * 1000
        if len(candles) < 1000:
            break
        time.sleep(0.2)  # API 부하 방지

    if not all_new:
        print(f"[{symbol}] 새 데이터 없음")
        return

    new_df = pd.DataFrame(all_new)
    print(f"[{symbol}] 새 캔들: {len(new_df)}개 ({new_df['time'].iloc[0]} ~ {new_df['time'].iloc[-1]})")

    # 병합
    if len(df) > 0:
        combined = pd.concat([df[["time", "open", "high", "low", "close", "volume"]], new_df],
                             ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset="time", keep="first").sort_values("time").reset_index(drop=True)

    # 파생 컬럼 전체 재계산
    combined = recalc_features(combined)

    added = len(combined) - (len(df) if len(df) > 0 else 0)
    combined.to_csv(csv_path, index=False)
    print(f"[{symbol}] 저장: {csv_path} ({len(combined)}캔들, +{added})")


def main():
    print(f"=== RL 데이터 갱신 ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}) ===")

    # 1. 메인 심볼 (ETH/BTC)
    print("\n--- 메인 (ETH/BTC) ---")
    for symbol, csv_path in SYMBOLS.items():
        try:
            update_symbol(symbol, csv_path)
        except Exception as e:
            print(f"[{symbol}] 에러: {e}")

    # 2. 알트코인 14종
    print("\n--- 알트코인 (14종) ---")
    ALT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    alt_ok, alt_fail = 0, 0
    for symbol in ALT_SYMBOLS:
        csv_name = f"{symbol.replace('USDT', '').lower()}_30m.csv"
        csv_path = ALT_DATA_DIR / csv_name
        try:
            update_symbol(symbol, csv_path)
            alt_ok += 1
        except Exception as e:
            print(f"[{symbol}] 에러: {e}")
            alt_fail += 1
    print(f"\n알트 결과: 성공 {alt_ok}, 실패 {alt_fail}")
    print("완료")


if __name__ == "__main__":
    main()

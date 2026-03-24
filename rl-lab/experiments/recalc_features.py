#!/usr/bin/env python3
# eth_30m_v41.csv 파생 컬럼 전체 재계산
import pandas as pd
import pandas_ta as ta
import numpy as np

csv_path = '/home/hyeok/01.APCC/00.ai-lab/rl/eth_30m_v41.csv'
df = pd.read_csv(csv_path, parse_dates=['time'])
print(f"원본: {len(df)}캔들")

df["rsi"] = ta.rsi(df["close"], length=14).fillna(50)
_macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
df["macd"] = _macd["MACD_12_26_9"].fillna(0)
df["macd_hist"] = _macd["MACDh_12_26_9"].fillna(0)
_bb = ta.bbands(df["close"], length=20, std=2)
# 컬럼명 자동 탐지 (pandas_ta 버전에 따라 다름)
bbu_col = [c for c in _bb.columns if c.startswith("BBU")][0]
bbl_col = [c for c in _bb.columns if c.startswith("BBL")][0]
df["bb_upper"] = _bb[bbu_col].fillna(df["close"])
df["bb_lower"] = _bb[bbl_col].fillna(df["close"])
bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan).fillna(1)
df["bb_pct"] = ((df["close"] - df["bb_lower"]) / bb_range).fillna(0.5).clip(0, 1)
df["ema20"] = ta.ema(df["close"], length=20).fillna(df["close"])
df["ema50"] = ta.ema(df["close"], length=50).fillna(df["close"])
df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14).fillna(0)
df["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()).fillna(1.0)

print(f"마지막 3행 rsi: {df['rsi'].tail(3).tolist()}")
print(f"마지막 200행 rsi=0: {(df.tail(200)['rsi'] == 0).sum()}개")

df.to_csv(csv_path, index=False)
print(f"저장: {csv_path}")

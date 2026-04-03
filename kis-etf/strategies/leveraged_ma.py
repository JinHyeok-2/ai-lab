#!/usr/bin/env python3
# 레버리지 MA교차 — KODEX 레버리지(2X) 전용

import pandas as pd
from strategies.base import BaseStrategy


class LeveragedMAStrategy(BaseStrategy):
    """
    레버리지 ETF + 느린 MA교차:
    - EMA(10)/EMA(30) 사용 (레버리지 노이즈 대비 느리게)
    - 골든크로스 → 매수, 데드크로스 → 매도
    - KODEX 레버리지(122630) 전용
    """

    name = "레버리지MA"
    description = "KODEX레버리지 EMA(10)/EMA(30)"
    etf_universe = ["122630"]

    def __init__(self, short: int = 10, long: int = 30):
        self.short = short
        self.long  = long

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]
        ema_s = close.ewm(span=self.short, adjust=False).mean()
        ema_l = close.ewm(span=self.long,  adjust=False).mean()

        cur_above  = ema_s.iloc[-1] > ema_l.iloc[-1]
        prev_above = ema_s.iloc[-2] > ema_l.iloc[-2]

        if cur_above and not prev_above:
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"레버리지 골든크로스 EMA({self.short})>EMA({self.long})",
                    "confidence": 0.7}

        if not cur_above and prev_above:
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": f"레버리지 데드크로스", "confidence": 0.7}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def get_required_lookback(self) -> int:
        return self.long + 5

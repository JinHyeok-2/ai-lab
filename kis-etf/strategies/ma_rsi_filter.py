#!/usr/bin/env python3
# MA교차 + RSI 필터 — 가짜 크로스 필터링

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MARSIFilterStrategy(BaseStrategy):
    """
    MA교차에 RSI 필터 추가:
    - 매수: 골든크로스 + RSI(14) > 50 (상승 모멘텀 확인)
    - 매도: 데드크로스 OR RSI < 40
    """

    name = "MA+RSI필터"
    description = "EMA교차 + RSI>50 필터"
    etf_universe = ["069500", "229200", "091160", "305720"]

    def __init__(self, short: int = 5, long: int = 20, rsi_period: int = 14):
        self.short = short
        self.long  = long
        self.rsi_period = rsi_period

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]
        ema_s = close.ewm(span=self.short, adjust=False).mean()
        ema_l = close.ewm(span=self.long,  adjust=False).mean()
        rsi   = self._calc_rsi(close, self.rsi_period)

        cur_above  = ema_s.iloc[-1] > ema_l.iloc[-1]
        prev_above = ema_s.iloc[-2] > ema_l.iloc[-2]
        cur_rsi    = rsi.iloc[-1] if rsi is not None else 50

        # 골든크로스 + RSI > 50
        if cur_above and not prev_above and cur_rsi > 50:
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"골든크로스+RSI={cur_rsi:.0f}>50", "confidence": 0.8}

        # 데드크로스 OR RSI < 40
        if (not cur_above and prev_above) or (cur_above and cur_rsi < 40):
            reason = "데드크로스" if not cur_above else f"RSI={cur_rsi:.0f}<40"
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": reason, "confidence": 0.7}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def _calc_rsi(self, close, period):
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def get_required_lookback(self) -> int:
        return max(self.long, self.rsi_period) + 5

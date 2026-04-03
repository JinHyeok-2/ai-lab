#!/usr/bin/env python3
# 전략 5: RSI 역추세 (과매도 매수 / 과매수 매도)

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class RSIContrarianStrategy(BaseStrategy):
    """
    RSI 역추세 전략:
    - RSI(14) < 30 → 매수 (과매도 반등)
    - RSI(14) > 70 → 매도 (과매수 조정)
    - 추가 필터: 3일 연속 하락 + RSI < 30 → 강한 매수
    - 추세 반전 포착, 변동성 큰 섹터 ETF에 적합
    """

    name = "RSI역추세"
    description = "RSI(14) 과매도(<30) 매수, 과매수(>70) 매도"
    etf_universe = ["069500", "229200", "091160", "305720"]

    def __init__(self, period: int = 14, buy_level: int = 30, sell_level: int = 70):
        self.period     = period
        self.buy_level  = buy_level
        self.sell_level = sell_level

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]
        rsi   = self._calc_rsi(close, self.period)

        if rsi is None or np.isnan(rsi.iloc[-1]):
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        cur_rsi = rsi.iloc[-1]

        # 연속 하락일 계산
        consec_down = 0
        for i in range(-1, -6, -1):
            if len(close) + i < 1:
                break
            if close.iloc[i] < close.iloc[i-1]:
                consec_down += 1
            else:
                break

        if cur_rsi < self.buy_level:
            # 3일 연속 하락 + RSI < 30 → 강한 신호
            conf = 0.6 + (self.buy_level - cur_rsi) / 100 + min(consec_down * 0.05, 0.15)
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"RSI={cur_rsi:.0f} < {self.buy_level} (연속하락 {consec_down}일)",
                    "confidence": min(conf, 1.0)}

        if cur_rsi > self.sell_level:
            conf = 0.6 + (cur_rsi - self.sell_level) / 100
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": f"RSI={cur_rsi:.0f} > {self.sell_level} (과매수)",
                    "confidence": min(conf, 1.0)}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def _calc_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0)
        loss  = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_required_lookback(self) -> int:
        return self.period + 10

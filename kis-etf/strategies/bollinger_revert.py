#!/usr/bin/env python3
# 전략 3: 볼린저밴드 평균회귀

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class BollingerRevertStrategy(BaseStrategy):
    """
    볼린저밴드 평균회귀 전략:
    - BB%B < 0.05 (하단 이탈) → 매수 (과매도 반등 기대)
    - BB%B > 0.95 (상단 이탈) → 매도
    - 거래량 확인: 평균 대비 1.5배 이상 시 신호 강화
    - 횡보장에서 효과적, 추세장에서 약함
    """

    name = "볼린저밴드"
    description = "BB%B 극단값 평균회귀"
    etf_universe = ["069500", "229200", "091160", "305720"]

    def __init__(self, period: int = 20, std_mult: float = 2.0,
                 buy_threshold: float = 0.05, sell_threshold: float = 0.95):
        self.period   = period
        self.std_mult = std_mult
        self.buy_th   = buy_threshold
        self.sell_th  = sell_threshold

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]

        # 볼린저밴드 계산
        sma  = close.rolling(self.period).mean()
        std  = close.rolling(self.period).std()
        upper = sma + self.std_mult * std
        lower = sma - self.std_mult * std

        # %B = (종가 - 하단) / (상단 - 하단)
        band_width = upper.iloc[-1] - lower.iloc[-1]
        if band_width <= 0:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        pct_b = (close.iloc[-1] - lower.iloc[-1]) / band_width

        # 거래량 조건 (평균 대비)
        vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1] if df["volume"].rolling(20).mean().iloc[-1] > 0 else 1

        if pct_b < self.buy_th:
            conf = min(0.6 + (self.buy_th - pct_b) * 2 + (vol_ratio - 1) * 0.1, 1.0)
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"BB%B={pct_b:.2f} < {self.buy_th} (과매도, vol×{vol_ratio:.1f})",
                    "confidence": conf}

        if pct_b > self.sell_th:
            conf = min(0.6 + (pct_b - self.sell_th) * 2, 1.0)
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": f"BB%B={pct_b:.2f} > {self.sell_th} (과매수)",
                    "confidence": conf}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def get_required_lookback(self) -> int:
        return self.period + 5

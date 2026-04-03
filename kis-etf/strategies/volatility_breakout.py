#!/usr/bin/env python3
# 변동성 브레이크아웃 — 래리 윌리엄스 전략

import pandas as pd
from strategies.base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    변동성 브레이크아웃 (래리 윌리엄스):
    - 매수: 당일 종가 > 당일 시가 + 전일 변동폭 × K
    - 매도: 다음날 시가 (1일 보유)
    - K=0.5 (기본), 레버리지 ETF에 적합
    """

    name = "변동성돌파"
    description = "래리윌리엄스 변동성돌파 (1일보유)"
    etf_universe = ["122630", "069500"]  # 레버리지 + KODEX200

    def __init__(self, k: float = 0.5):
        self.k = k
        self._positions = {}

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        if len(df) < 3:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        # 보유 중이면 무조건 매도 (1일 보유)
        if ticker in self._positions:
            del self._positions[ticker]
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": "1일보유 청산", "confidence": 0.9}

        # 전일 변동폭
        prev_range = df["high"].iloc[-2] - df["low"].iloc[-2]
        # 당일 시가 + 전일 변동폭 × K
        breakout_level = df["open"].iloc[-1] + prev_range * self.k
        # 당일 종가가 돌파했는지
        cur_close = df["close"].iloc[-1]

        if cur_close > breakout_level and prev_range > 0:
            pct = (cur_close - df["open"].iloc[-1]) / df["open"].iloc[-1] * 100
            self._positions[ticker] = True
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"변동성돌파 K={self.k} (일중+{pct:.1f}%)",
                    "confidence": 0.6 + min(pct / 10, 0.3)}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def get_required_lookback(self) -> int:
        return 5

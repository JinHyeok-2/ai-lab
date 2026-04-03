#!/usr/bin/env python3
# 전략 1: 이동평균 교차 (EMA 골든/데드크로스)

import pandas as pd
from strategies.base import BaseStrategy


class MACrossStrategy(BaseStrategy):
    """
    EMA(5) × EMA(20) 교차 전략
    - 골든크로스(단기>장기) → 매수
    - 데드크로스(단기<장기) → 매도
    - 추세 추종형, 지수 ETF에 적합
    """

    name = "MA교차"
    description = "EMA(5)/EMA(20) 골든/데드크로스"
    etf_universe = ["069500", "229200", "091160", "305720"]  # 지수+섹터

    def __init__(self, short: int = 5, long: int = 20):
        self.short = short
        self.long  = long

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]
        ema_s = close.ewm(span=self.short, adjust=False).mean()
        ema_l = close.ewm(span=self.long,  adjust=False).mean()

        # 현재 + 전일 비교로 교차 판단
        cur_above  = ema_s.iloc[-1] > ema_l.iloc[-1]
        prev_above = ema_s.iloc[-2] > ema_l.iloc[-2]

        if cur_above and not prev_above:
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"골든크로스 EMA({self.short})>EMA({self.long})",
                    "confidence": 0.7}

        if not cur_above and prev_above:
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": f"데드크로스 EMA({self.short})<EMA({self.long})",
                    "confidence": 0.7}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def get_required_lookback(self) -> int:
        return self.long + 5

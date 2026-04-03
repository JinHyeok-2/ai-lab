#!/usr/bin/env python3
# RSI 트레일링 — 극단 과매도 매수 + 트레일링 스톱 매도

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class RSITrailingStrategy(BaseStrategy):
    """
    RSI 역추세 + 트레일링 스톱:
    - 매수: RSI(14) < 25 (극단 과매도)
    - 매도: 매수 후 고점 대비 -3% 하락 (트레일링)
    - 바닥에서 잡고, 추세가 끝날 때까지 보유
    """

    name = "RSI트레일링"
    description = "RSI<25 매수 + 고점-3% 트레일링"
    etf_universe = ["069500", "091160", "305720"]

    def __init__(self, rsi_period: int = 14, rsi_entry: int = 25, trail_pct: float = 0.03):
        self.rsi_period = rsi_period
        self.rsi_entry  = rsi_entry
        self.trail_pct  = trail_pct
        self._positions = {}  # {ticker: {"entry": P, "high": H}}

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]
        cur_price = close.iloc[-1]

        # 보유 중 → 트레일링 스톱 판단
        if ticker in self._positions:
            pos = self._positions[ticker]
            # 고점 갱신
            if cur_price > pos["high"]:
                pos["high"] = cur_price

            # 고점 대비 하락폭
            drawdown = (pos["high"] - cur_price) / pos["high"]

            if drawdown >= self.trail_pct:
                pnl_pct = (cur_price / pos["entry"] - 1) * 100
                del self._positions[ticker]
                return {"ticker": ticker, "action": "SELL", "qty": 0,
                        "reason": f"트레일링 고점{pos['high']:,.0f}→{cur_price:,.0f} (-{drawdown*100:.1f}%, 총{pnl_pct:+.1f}%)",
                        "confidence": 0.9}

            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        # 신규 진입: RSI < 25
        rsi = self._calc_rsi(close, self.rsi_period)
        if rsi is None or np.isnan(rsi.iloc[-1]):
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        cur_rsi = rsi.iloc[-1]
        if cur_rsi < self.rsi_entry:
            self._positions[ticker] = {"entry": cur_price, "high": cur_price}
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"RSI={cur_rsi:.0f}<{self.rsi_entry} 극단과매도",
                    "confidence": 0.7 + (self.rsi_entry - cur_rsi) / 100}

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
        return self.rsi_period + 10

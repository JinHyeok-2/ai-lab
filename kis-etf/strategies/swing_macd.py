#!/usr/bin/env python3
# 단기 전략: MACD + ADX 추세 확인 스윙 (5~20일 보유)

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class SwingMACDStrategy(BaseStrategy):
    """
    단기 스윙 전략 (5~20일 보유):

    매수 조건:
    1) MACD 히스토그램 음→양 전환 (상승 모멘텀 시작)
    2) ADX > 20 (추세 존재 확인)
    3) 가격 > EMA(20) (상승 추세 위)

    매도 조건:
    - MACD 히스토그램 양→음 전환
    - 또는 -3% 손절

    대상: 지수 + 섹터 ETF
    """

    name = "단기스윙"
    description = "MACD전환+ADX추세 스윙 (5~20일)"
    etf_universe = ["069500", "229200", "091160", "305720"]

    def __init__(self):
        self.fast   = 12
        self.slow   = 26
        self.signal = 9
        self.adx_period = 14
        self.adx_min    = 20
        self.sl_pct     = -0.03  # 손절 -3%
        self._positions = {}

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # MACD
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd     = ema_fast - ema_slow
        signal   = macd.ewm(span=self.signal, adjust=False).mean()
        hist     = macd - signal

        # ADX
        adx = self._calc_adx(high, low, close, self.adx_period)

        # EMA(20)
        ema20 = close.ewm(span=20, adjust=False).mean()

        # 보유 중 → 매도 판단
        if ticker in self._positions:
            pos = self._positions[ticker]
            cur_ret = (close.iloc[-1] / pos["price"]) - 1

            # 손절
            if cur_ret <= self.sl_pct:
                del self._positions[ticker]
                return {"ticker": ticker, "action": "SELL", "qty": 0,
                        "reason": f"손절 {cur_ret*100:+.1f}%",
                        "confidence": 0.9}

            # MACD 히스토그램 양→음 전환 → 매도
            if len(hist) >= 2 and hist.iloc[-2] > 0 and hist.iloc[-1] <= 0:
                del self._positions[ticker]
                return {"ticker": ticker, "action": "SELL", "qty": 0,
                        "reason": f"MACD 하락전환 ({cur_ret*100:+.1f}%)",
                        "confidence": 0.8}

            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        # 신규 진입 판단
        if len(hist) < 2 or adx is None:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        cond_macd = hist.iloc[-2] <= 0 and hist.iloc[-1] > 0    # 음→양 전환
        cond_adx  = adx.iloc[-1] > self.adx_min                 # 추세 존재
        cond_ema  = close.iloc[-1] > ema20.iloc[-1]              # 상승 추세

        if cond_macd and cond_adx and cond_ema:
            self._positions[ticker] = {"price": close.iloc[-1]}
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"MACD전환 ADX={adx.iloc[-1]:.0f} 가격>EMA20",
                    "confidence": 0.75}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def _calc_adx(self, high, low, close, period):
        """ADX 계산"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr      = tr.ewm(span=period, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx

    def get_required_lookback(self) -> int:
        return self.slow + self.signal + 5

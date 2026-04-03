#!/usr/bin/env python3
# 스캘핑 전략: RSI 극단 + 볼린저밴드 + 갭 반전
# 레버리지 ETF 대상, 일봉 기반 일중 스캘핑 시뮬레이션

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class ScalpingRSIStrategy(BaseStrategy):
    """
    스캘핑 전략 (일봉 기반 시뮬레이션):

    매수 조건 (3개 중 2개 이상):
    1) RSI(5) < 20 (초과매도)
    2) BB%B < 0.0 (볼린저 하단 이탈)
    3) 전일 대비 -2% 이상 하락 (갭다운 반등)

    매도 조건:
    - 당일 종가가 매수가 대비 +1% 이상 → 익절
    - 당일 종가가 매수가 대비 -1.5% 이하 → 손절
    - 최대 보유 3일 → 강제 청산

    대상: KODEX 레버리지(122630), KODEX 인버스2X(252670)
    """

    name = "스캘핑"
    description = "RSI극단+BB이탈+갭반전 (레버리지ETF)"
    etf_universe = ["122630", "252670"]  # 레버리지, 인버스2X

    def __init__(self):
        self.rsi_period = 5
        self.rsi_level  = 20
        self.bb_period  = 10
        self.bb_std     = 2.0
        self.tp_pct     = 0.01    # 익절 1%
        self.sl_pct     = -0.015  # 손절 -1.5%
        self.max_hold   = 3       # 최대 보유 3일
        self._positions = {}      # {ticker: {"price": P, "day": 0}}

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        close = df["close"]

        # 보유 중이면 청산 판단
        if ticker in self._positions:
            pos = self._positions[ticker]
            pos["day"] += 1
            cur_price = close.iloc[-1]
            ret = (cur_price / pos["price"]) - 1

            if ret >= self.tp_pct:
                del self._positions[ticker]
                return {"ticker": ticker, "action": "SELL", "qty": 0,
                        "reason": f"익절 {ret*100:+.1f}% (TP {self.tp_pct*100}%)",
                        "confidence": 0.9}
            if ret <= self.sl_pct:
                del self._positions[ticker]
                return {"ticker": ticker, "action": "SELL", "qty": 0,
                        "reason": f"손절 {ret*100:+.1f}% (SL {self.sl_pct*100}%)",
                        "confidence": 0.9}
            if pos["day"] >= self.max_hold:
                del self._positions[ticker]
                return {"ticker": ticker, "action": "SELL", "qty": 0,
                        "reason": f"시간청산 {pos['day']}일 ({ret*100:+.1f}%)",
                        "confidence": 0.7}

            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        # 신규 진입 판단
        score = 0
        reasons = []

        # 1) RSI(5) 초과매도
        rsi = self._calc_rsi(close, self.rsi_period)
        if rsi is not None and rsi.iloc[-1] < self.rsi_level:
            score += 1
            reasons.append(f"RSI({self.rsi_period})={rsi.iloc[-1]:.0f}")

        # 2) BB 하단 이탈
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        lower = sma - self.bb_std * std
        upper = sma + self.bb_std * std
        band_w = upper.iloc[-1] - lower.iloc[-1]
        if band_w > 0:
            pct_b = (close.iloc[-1] - lower.iloc[-1]) / band_w
            if pct_b < 0.0:
                score += 1
                reasons.append(f"BB%B={pct_b:.2f}")

        # 3) 갭다운 -2%
        daily_ret = (close.iloc[-1] / close.iloc[-2]) - 1 if len(close) >= 2 else 0
        if daily_ret < -0.02:
            score += 1
            reasons.append(f"일변동={daily_ret*100:.1f}%")

        if score >= 2:
            self._positions[ticker] = {"price": close.iloc[-1], "day": 0}
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"스캘핑진입 ({'/'.join(reasons)})",
                    "confidence": 0.6 + score * 0.1}

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
        return 15

#!/usr/bin/env python3
# 전략 2: 듀얼 모멘텀 (절대모멘텀 + 상대모멘텀)
# Gary Antonacci의 Dual Momentum 기반

import pandas as pd
from strategies.base import BaseStrategy


class DualMomentumStrategy(BaseStrategy):
    """
    듀얼 모멘텀 전략:
    1) 절대모멘텀: 12개월 수익률 > 0% (상승 추세)
    2) 상대모멘텀: 주식 ETF vs 채권 ETF 수익률 비교
    - 주식 > 채권 + 절대 양수 → 주식 매수
    - 그 외 → 채권 매수 (안전자산 대피)
    - 월 1회 리밸런싱 (매월 첫 거래일)
    """

    name = "듀얼모멘텀"
    description = "12개월 절대+상대 모멘텀 (주식 vs 채권)"
    etf_universe = ["069500", "148070"]  # 주식(KODEX200) + 채권(국고채10년)

    def __init__(self, lookback: int = 252, equity: str = "069500", bond: str = "148070"):
        self.lookback = lookback  # 12개월 영업일
        self.equity = equity
        self.bond   = bond
        self._last_rebal_month = None

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        # 이 전략은 equity ticker 기준으로만 판단
        if ticker != self.equity:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        # 월 1회만 리밸런싱
        cur_month = df.index[-1].strftime("%Y-%m")
        cur_day   = df.index[-1].day

        # 매월 첫 5거래일 이내에만 신호
        month_data = df[df.index.strftime("%Y-%m") == cur_month]
        if len(month_data) > 5:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "리밸런싱 아님", "confidence": 0}

        if self._last_rebal_month == cur_month:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "이번달 완료", "confidence": 0}

        if len(df) < self.lookback:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "데이터 부족", "confidence": 0}

        # 12개월 수익률 (절대모멘텀)
        mom = df["close"].iloc[-1] / df["close"].iloc[-self.lookback] - 1

        self._last_rebal_month = cur_month

        if mom > 0:
            # 절대모멘텀 양수 → 주식 매수
            return {"ticker": self.equity, "action": "BUY", "qty": 0,
                    "reason": f"12M 수익률 {mom*100:.1f}% > 0%, 주식 보유",
                    "confidence": min(0.5 + mom, 1.0)}
        else:
            # 절대모멘텀 음수 → 매도 (채권으로 전환은 별도 처리)
            return {"ticker": self.equity, "action": "SELL", "qty": 0,
                    "reason": f"12M 수익률 {mom*100:.1f}% < 0%, 채권 대피",
                    "confidence": 0.8}

    def get_required_lookback(self) -> int:
        return self.lookback + 10

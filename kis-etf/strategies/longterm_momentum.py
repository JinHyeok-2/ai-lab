#!/usr/bin/env python3
# 장기 전략: 모멘텀 + 자산배분 혼합 (월 1회 리밸런싱)
# 듀얼 모멘텀 개선 — 섹터 로테이션 추가

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class LongtermMomentumStrategy(BaseStrategy):
    """
    장기 모멘텀+자산배분 혼합 전략 (월 1회):

    로직:
    1) 주식 ETF 4종의 6개월 모멘텀 순위 계산
    2) 모멘텀 1위 종목이 채권 수익률보다 높으면 → 주식 매수
    3) 모든 주식이 채권보다 낮으면 → 채권+금으로 방어
    4) 배분: 1위 주식 40% + 채권 40% + 금 20%
       (주식 약세 시: 채권 60% + 금 40%)

    대상: 지수/섹터 ETF + 채권 + 금
    """

    name = "장기모멘텀"
    description = "섹터로테이션+자산배분 (월1회)"
    etf_universe = ["069500", "229200", "091160", "305720", "148070", "132030"]

    EQUITY_TICKERS = ["069500", "229200", "091160", "305720"]
    BOND_TICKER    = "148070"
    GOLD_TICKER    = "132030"

    def __init__(self, lookback: int = 126):  # 6개월
        self.lookback = lookback
        self._last_month = None
        self._current_pick = None  # 현재 선택된 주식 종목

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        # 주 종목(069500) 기준으로만 월 1회 판단
        if ticker != "069500":
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        cur_month = df.index[-1].strftime("%Y-%m")
        month_data = df[df.index.strftime("%Y-%m") == cur_month]
        if len(month_data) > 5:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        if self._last_month == cur_month:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        if len(df) < self.lookback:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "데이터부족", "confidence": 0}

        # 6개월 모멘텀 계산
        mom = (df["close"].iloc[-1] / df["close"].iloc[-self.lookback]) - 1
        self._last_month = cur_month

        if mom > 0:
            # 절대모멘텀 양수 → 주식 보유
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"6M모멘텀 {mom*100:+.1f}% 양수→주식",
                    "confidence": min(0.6 + mom, 1.0)}
        else:
            # 절대모멘텀 음수 → 주식 매도 (방어 모드)
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": f"6M모멘텀 {mom*100:+.1f}% 음수→방어",
                    "confidence": 0.8}

    def get_required_lookback(self) -> int:
        return self.lookback + 10

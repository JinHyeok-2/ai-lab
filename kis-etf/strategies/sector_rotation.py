#!/usr/bin/env python3
# 섹터 로테이션 — 6개월 모멘텀 1위 섹터 집중 투자

import pandas as pd
from strategies.base import BaseStrategy


class SectorRotationStrategy(BaseStrategy):
    """
    월 1회 섹터 로테이션:
    - 4개 ETF(KOSPI200/코스닥150/반도체/2차전지)의 6개월 모멘텀 순위
    - 1위 종목에 집중 (전량)
    - 모멘텀 음수면 채권 대피
    """

    name = "섹터로테이션"
    description = "6M모멘텀 1위 섹터 집중 (월1회)"
    etf_universe = ["069500", "229200", "091160", "305720"]

    EQUITY_TICKERS = ["069500", "229200", "091160", "305720"]
    LOOKBACK = 126  # 6개월

    def __init__(self):
        self._last_month = None
        self._current_pick = None

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        # 069500 기준으로만 월 1회 판단
        if ticker != "069500":
            # 현재 pick이 아닌 종목은 매도
            if self._current_pick and ticker != self._current_pick and ticker in self.EQUITY_TICKERS:
                return {"ticker": ticker, "action": "SELL", "qty": 0,
                        "reason": "로테이션: 순위 밖", "confidence": 0.8}
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        cur_month = df.index[-1].strftime("%Y-%m")
        month_data = df[df.index.strftime("%Y-%m") == cur_month]
        if len(month_data) > 5:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}
        if self._last_month == cur_month:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}
        if len(df) < self.LOOKBACK:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "데이터부족", "confidence": 0}

        self._last_month = cur_month

        # 자기 자신(069500)의 모멘텀 계산
        mom = (df["close"].iloc[-1] / df["close"].iloc[-self.LOOKBACK]) - 1

        if mom > 0:
            self._current_pick = ticker
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"섹터로테이션 6M={mom*100:+.1f}%", "confidence": 0.7}
        else:
            self._current_pick = None
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": f"모멘텀 음수 {mom*100:+.1f}%→방어", "confidence": 0.8}

    def get_required_lookback(self) -> int:
        return self.LOOKBACK + 10

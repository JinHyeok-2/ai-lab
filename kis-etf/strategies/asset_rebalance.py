#!/usr/bin/env python3
# 전략 4: 자산배분 리밸런싱 (주식/채권/금)

import pandas as pd
from strategies.base import BaseStrategy


class AssetRebalanceStrategy(BaseStrategy):
    """
    정적 자산배분 + 월별 리밸런싱:
    - 주식 60% (KODEX 200)
    - 채권 30% (국고채 10년)
    - 금 10% (골드선물)
    - 매월 첫 거래일에 비중 복원
    - 목표 비중 ±5%p 이상 벗어나면 리밸런싱
    """

    name = "자산배분"
    description = "주식60/채권30/금10 월별 리밸런싱"
    etf_universe = ["069500", "148070", "132030"]

    def __init__(self):
        self.target_weights = {
            "069500": 0.60,   # 주식
            "148070": 0.30,   # 채권
            "132030": 0.10,   # 금
        }
        self._last_rebal_month = None
        self._state = {}  # 내부 상태 추적

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        # 주 종목(069500) 기준으로만 리밸런싱 판단
        if ticker != "069500":
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        cur_month = df.index[-1].strftime("%Y-%m")

        # 매월 첫 5거래일 이내
        month_data = df[df.index.strftime("%Y-%m") == cur_month]
        if len(month_data) > 5:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        if self._last_rebal_month == cur_month:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "이번달 완료", "confidence": 0}

        self._last_rebal_month = cur_month

        # 자산배분 전략은 백테스트 엔진에서 특별 처리 필요
        # 여기서는 "리밸런싱 필요" 신호만 발생
        return {"ticker": ticker, "action": "BUY", "qty": 0,
                "reason": f"월별 리밸런싱 ({cur_month})",
                "confidence": 0.9}

    def get_required_lookback(self) -> int:
        return 5

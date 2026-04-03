#!/usr/bin/env python3
# 전략 추상 기반 클래스

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """ETF 매매 전략 기반 클래스"""

    name: str = "base"
    description: str = ""
    etf_universe: list = []   # 대상 ETF 티커 목록

    @abstractmethod
    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        """
        매매 신호 생성

        Args:
            ticker: ETF 종목코드
            df: 일봉 OHLCV DataFrame (컬럼: open, high, low, close, volume)

        Returns:
            {
                "ticker": str,
                "action": "BUY" | "SELL" | "HOLD",
                "qty": int (0이면 전량),
                "reason": str,
                "confidence": float (0~1),
            }
        """
        raise NotImplementedError

    @abstractmethod
    def get_required_lookback(self) -> int:
        """신호 생성에 필요한 최소 일봉 수"""
        raise NotImplementedError

    def __repr__(self):
        return f"<Strategy: {self.name}>"

#!/usr/bin/env python3
# 레짐 기반 전략 — 상승장/하락장 분기 매매

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class RegimeStrategy(BaseStrategy):
    """
    시장 레짐 판별 → 상승장/하락장별 전략 분기

    레짐 판별:
    - SMA(200): 종가 > 200일 이평 → 상승장
    - EMA(60):  종가 > 60일 EMA → 상승장

    상승장 전략:
    - MA교차: EMA(5)/EMA(20) 골든/데드크로스
    - 레버리지MA: EMA(10)/EMA(30) on KODEX 레버리지

    하락장 전략:
    - 변동성돌파: 래리윌리엄스 K=0.5, 1일 보유
    - 현금보유: 전량 매도
    """

    def __init__(self, regime_type: str = "SMA200",
                 bull_strategy: str = "MA교차",
                 bear_strategy: str = "현금보유",
                 label: str = ""):
        self.regime_type   = regime_type    # "SMA200" or "EMA60"
        self.bull_strategy = bull_strategy  # "MA교차" or "레버리지MA"
        self.bear_strategy = bear_strategy  # "변동성돌파" or "현금보유"

        self.name = label or f"{regime_type}_{bull_strategy}_{bear_strategy}"
        self.description = f"레짐({regime_type}) 상승:{bull_strategy} 하락:{bear_strategy}"

        # 대상 ETF
        if bull_strategy == "레버리지MA":
            self.etf_universe = ["122630"]  # 레버리지
        else:
            self.etf_universe = ["069500", "229200", "091160", "305720"]

        # 레짐용 기준 종목 (KODEX 200)
        self._regime_ticker = "069500"

        # 변동성돌파 상태
        self._vb_positions = {}

    def _is_bull(self, df: pd.DataFrame) -> bool:
        """레짐 판별"""
        close = df["close"]
        if self.regime_type == "SMA200":
            if len(close) < 200:
                return True  # 데이터 부족 시 상승장 가정
            sma200 = close.rolling(200).mean().iloc[-1]
            return close.iloc[-1] > sma200
        else:  # EMA60
            if len(close) < 60:
                return True
            ema60 = close.ewm(span=60, adjust=False).mean().iloc[-1]
            return close.iloc[-1] > ema60

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        is_bull = self._is_bull(df)

        if is_bull:
            return self._bull_signal(ticker, df)
        else:
            return self._bear_signal(ticker, df)

    def _bull_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        """상승장 전략"""
        # 변동성돌파 보유 중이면 먼저 청산
        if ticker in self._vb_positions:
            del self._vb_positions[ticker]
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": "레짐전환→상승장, 변동성돌파 청산", "confidence": 0.8}

        if self.bull_strategy == "MA교차":
            return self._ma_cross(ticker, df, short=5, long=20)
        else:  # 레버리지MA
            return self._ma_cross(ticker, df, short=10, long=30)

    def _bear_signal(self, ticker: str, df: pd.DataFrame) -> dict:
        """하락장 전략"""
        if self.bear_strategy == "현금보유":
            # 보유 중이면 전량 매도
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": "하락장→현금보유", "confidence": 0.9}

        else:  # 변동성돌파
            return self._volatility_breakout(ticker, df)

    def _ma_cross(self, ticker: str, df: pd.DataFrame, short: int, long: int) -> dict:
        """MA 교차 신호"""
        close = df["close"]
        if len(close) < long + 2:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        ema_s = close.ewm(span=short, adjust=False).mean()
        ema_l = close.ewm(span=long,  adjust=False).mean()

        cur_above  = ema_s.iloc[-1] > ema_l.iloc[-1]
        prev_above = ema_s.iloc[-2] > ema_l.iloc[-2]

        if cur_above and not prev_above:
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"상승장 골든크로스 EMA({short})>EMA({long})",
                    "confidence": 0.7}
        if not cur_above and prev_above:
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": f"데드크로스 EMA({short})<EMA({long})",
                    "confidence": 0.7}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def _volatility_breakout(self, ticker: str, df: pd.DataFrame) -> dict:
        """변동성 브레이크아웃"""
        if len(df) < 3:
            return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

        # 보유 중이면 1일 후 매도
        if ticker in self._vb_positions:
            del self._vb_positions[ticker]
            return {"ticker": ticker, "action": "SELL", "qty": 0,
                    "reason": "변동성돌파 1일청산", "confidence": 0.9}

        prev_range = df["high"].iloc[-2] - df["low"].iloc[-2]
        breakout = df["open"].iloc[-1] + prev_range * 0.5
        cur_close = df["close"].iloc[-1]

        if cur_close > breakout and prev_range > 0:
            self._vb_positions[ticker] = True
            return {"ticker": ticker, "action": "BUY", "qty": 0,
                    "reason": f"하락장 변동성돌파", "confidence": 0.6}

        return {"ticker": ticker, "action": "HOLD", "qty": 0, "reason": "", "confidence": 0}

    def get_required_lookback(self) -> int:
        return 210  # SMA(200) + 여유

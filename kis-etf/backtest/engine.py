#!/usr/bin/env python3
# 일봉 기반 백테스트 엔진
import sys; sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np


class BacktestEngine:
    """
    단순 일봉 백테스트 엔진

    - 당일 종가 기준 신호 → 다음날 시가 체결 (현실적)
    - 수수료 편도 0.015% (ETF)
    - 슬리피지 0.01% (시장가 주문 가정)
    """

    def __init__(self, initial_capital: int = 1_000_000,
                 fee_rate: float = 0.00015,
                 slippage: float = 0.0001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage

    def run(self, strategy, data: dict, start: str = None, end: str = None) -> dict:
        """
        백테스트 실행

        Args:
            strategy: BaseStrategy 인스턴스 (generate_signal 메서드 필수)
            data: {ticker: DataFrame} — 일봉 데이터
            start/end: 'YYYY-MM-DD' 형식 (선택)

        Returns:
            {
                "equity_curve": pd.Series,
                "trades": list,
                "signals": list,
                "positions": dict,  # 최종 보유
            }
        """
        # 공통 날짜 인덱스 생성
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        if start:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start)]
        if end:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end)]

        # 상태 초기화
        cash = self.initial_capital
        positions = {}       # {ticker: {"qty": N, "avg_price": P, "buy_date": D}}
        equity_list = []     # [(date, equity)]
        trades = []          # 거래 기록
        signals = []         # 신호 기록

        lookback = strategy.get_required_lookback()

        for i, today in enumerate(all_dates):
            # 포트폴리오 평가
            holdings_value = 0
            for ticker, pos in positions.items():
                if ticker in data and today in data[ticker].index:
                    cur_price = data[ticker].loc[today, "close"]
                    holdings_value += pos["qty"] * cur_price

            equity = cash + holdings_value
            equity_list.append((today, equity))

            # 각 종목에 대해 신호 생성
            for ticker in strategy.etf_universe:
                if ticker not in data:
                    continue

                df = data[ticker]
                # today까지의 데이터만 사용 (미래 정보 차단)
                hist = df[df.index <= today]
                if len(hist) < lookback:
                    continue

                signal = strategy.generate_signal(ticker, hist)
                if signal["action"] == "HOLD":
                    continue

                signals.append({
                    "date": today, "ticker": ticker,
                    "action": signal["action"], "reason": signal.get("reason", ""),
                })

                # 다음 거래일 시가로 체결
                next_idx = i + 1
                if next_idx >= len(all_dates):
                    continue
                next_day = all_dates[next_idx]
                if next_day not in df.index:
                    continue
                exec_price = df.loc[next_day, "open"]

                if signal["action"] == "BUY" and ticker not in positions:
                    # 매수: 자본의 일정 비율
                    alloc = cash * 0.2  # 종목당 20%
                    alloc = min(alloc, cash * 0.95)  # 최대 95% (수수료 여유)
                    if alloc < exec_price:
                        continue

                    # 슬리피지 적용
                    buy_price = int(exec_price * (1 + self.slippage))
                    qty = int(alloc / buy_price)
                    if qty <= 0:
                        continue

                    cost = qty * buy_price
                    fee  = int(cost * self.fee_rate)
                    total_cost = cost + fee

                    if total_cost > cash:
                        qty = int((cash * 0.95) / (buy_price * (1 + self.fee_rate)))
                        if qty <= 0:
                            continue
                        cost = qty * buy_price
                        fee  = int(cost * self.fee_rate)
                        total_cost = cost + fee

                    cash -= total_cost
                    positions[ticker] = {
                        "qty": qty, "avg_price": buy_price, "buy_date": next_day,
                    }
                    trades.append({
                        "date": next_day, "ticker": ticker, "side": "BUY",
                        "qty": qty, "price": buy_price, "fee": fee,
                    })

                elif signal["action"] == "SELL" and ticker in positions:
                    # 매도: 전량
                    pos = positions[ticker]
                    sell_price = int(exec_price * (1 - self.slippage))
                    revenue = pos["qty"] * sell_price
                    fee = int(revenue * self.fee_rate)
                    net = revenue - fee
                    pnl = net - (pos["qty"] * pos["avg_price"])
                    pnl_pct = (sell_price / pos["avg_price"] - 1) * 100

                    cash += net
                    trades.append({
                        "date": next_day, "ticker": ticker, "side": "SELL",
                        "qty": pos["qty"], "price": sell_price, "fee": fee,
                        "pnl": pnl, "pnl_pct": pnl_pct,
                        "hold_days": (next_day - pos["buy_date"]).days,
                    })
                    del positions[ticker]

        # equity curve → Series
        eq_series = pd.Series(
            [e[1] for e in equity_list],
            index=pd.DatetimeIndex([e[0] for e in equity_list]),
            name="equity",
        )

        return {
            "equity_curve": eq_series,
            "trades":       trades,
            "signals":      signals,
            "positions":    positions,
            "final_cash":   cash,
        }


def run_buy_and_hold(ticker: str, data: dict, initial_capital: int = 1_000_000) -> pd.Series:
    """Buy & Hold 벤치마크 (초기 전량 매수 → 보유)"""
    if ticker not in data:
        return pd.Series(dtype=float)

    df = data[ticker]
    first_price = df.iloc[0]["close"]
    qty = initial_capital // first_price
    remainder = initial_capital - qty * first_price

    equity = df["close"] * qty + remainder
    equity.name = "equity"
    return equity

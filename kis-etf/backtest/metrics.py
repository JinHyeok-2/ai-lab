#!/usr/bin/env python3
# 백테스트 성과 지표 계산
import sys; sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd


def calc_metrics(equity_curve: pd.Series, trades: list = None, risk_free: float = 0.03) -> dict:
    """
    성과 지표 계산

    Args:
        equity_curve: 일별 자산 가치 Series (index=날짜)
        trades: 거래 기록 리스트 (optional)
        risk_free: 무위험 수익률 (연율, 기본 3%)

    Returns:
        dict: 성과 지표
    """
    if equity_curve is None or len(equity_curve) < 2:
        return _empty_metrics()

    # 기본 수익률
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25

    # CAGR (연평균 복합 수익률)
    if years > 0 and equity_curve.iloc[0] > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    else:
        cagr = 0

    # 일별 수익률
    daily_returns = equity_curve.pct_change().dropna()

    # MDD (최대 낙폭)
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    mdd = drawdown.min()

    # 변동성 (연율화)
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0

    # Sharpe Ratio
    excess_return = cagr - risk_free
    sharpe = excess_return / volatility if volatility > 0 else 0

    # Sortino Ratio (하방 변동성만 사용)
    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
    sortino = excess_return / downside_vol if downside_vol > 0 else 0

    # Calmar Ratio (CAGR / |MDD|)
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # 승률 (거래 기록 있을 때)
    win_rate = 0
    profit_factor = 0
    trade_count = 0
    avg_profit = 0

    if trades:
        trade_count = len(trades)
        profits = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
        losses  = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0]
        win_rate = len(profits) / trade_count * 100 if trade_count else 0
        total_profit = sum(profits)
        total_loss   = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_profit = sum(t.get("pnl", 0) for t in trades) / trade_count if trade_count else 0

    return {
        "total_return":  total_return * 100,     # %
        "cagr":          cagr * 100,             # %
        "mdd":           mdd * 100,              # % (음수)
        "volatility":    volatility * 100,       # %
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "trade_count":   trade_count,
        "win_rate":      win_rate,               # %
        "profit_factor": profit_factor,
        "avg_profit":    avg_profit,             # 원
        "start_date":    equity_curve.index[0].strftime("%Y-%m-%d"),
        "end_date":      equity_curve.index[-1].strftime("%Y-%m-%d"),
        "days":          days,
        "final_value":   equity_curve.iloc[-1],
    }


def _empty_metrics() -> dict:
    return {
        "total_return": 0, "cagr": 0, "mdd": 0, "volatility": 0,
        "sharpe": 0, "sortino": 0, "calmar": 0,
        "trade_count": 0, "win_rate": 0, "profit_factor": 0,
        "avg_profit": 0, "start_date": "", "end_date": "",
        "days": 0, "final_value": 0,
    }


def format_metrics(name: str, m: dict) -> str:
    """지표를 한 줄 포맷 문자열로 변환"""
    return (
        f"{name:<16} "
        f"{m['total_return']:>+8.1f}%  "
        f"{m['cagr']:>+6.1f}%  "
        f"{m['mdd']:>+7.1f}%  "
        f"{m['sharpe']:>6.2f}  "
        f"{m['trade_count']:>5}  "
        f"{m['win_rate']:>5.0f}%  "
        f"{m['final_value']:>12,.0f}"
    )


def metrics_header() -> str:
    return (
        f"{'전략':<16} "
        f"{'총수익':>8}  "
        f"{'CAGR':>6}  "
        f"{'MDD':>7}  "
        f"{'Sharpe':>6}  "
        f"{'거래수':>5}  "
        f"{'승률':>5}  "
        f"{'최종자산':>12}"
    )

#!/usr/bin/env python3
# 전략 비교 백테스트 실행 + 리포트
import sys; sys.stdout.reconfigure(line_buffering=True)

from backtest.data_loader import download_universe, BACKTEST_UNIVERSE
from backtest.engine import BacktestEngine, run_buy_and_hold
from backtest.metrics import calc_metrics, format_metrics, metrics_header

from strategies.ma_cross import MACrossStrategy
from strategies.dual_momentum import DualMomentumStrategy
from strategies.bollinger_revert import BollingerRevertStrategy
from strategies.asset_rebalance import AssetRebalanceStrategy
from strategies.rsi_contrarian import RSIContrarianStrategy
from strategies.scalping_rsi import ScalpingRSIStrategy
from strategies.swing_macd import SwingMACDStrategy
from strategies.longterm_momentum import LongtermMomentumStrategy
from strategies.ma_rsi_filter import MARSIFilterStrategy
from strategies.sector_rotation import SectorRotationStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from strategies.leveraged_ma import LeveragedMAStrategy
from strategies.rsi_trailing import RSITrailingStrategy


def get_all_strategies():
    """전략 인스턴스 목록 반환"""
    return [
        # 스캘핑 (레버리지 ETF, 1~3일)
        ScalpingRSIStrategy(),
        VolatilityBreakoutStrategy(),
        # 단기 (지수/섹터, 5~20일)
        MACrossStrategy(),
        MARSIFilterStrategy(),
        SwingMACDStrategy(),
        LeveragedMAStrategy(),
        BollingerRevertStrategy(),
        RSIContrarianStrategy(),
        RSITrailingStrategy(),
        # 장기 (자산배분, 월 1회)
        SectorRotationStrategy(),
        LongtermMomentumStrategy(),
        DualMomentumStrategy(),
        AssetRebalanceStrategy(),
    ]


def run_comparison(initial_capital: int = 1_000_000,
                   start: str = "2019-01-02", end: str = "2026-04-03"):
    """전체 전략 비교 실행"""
    print("=" * 90)
    print(f"  KIS-ETF 전략 비교 백테스트")
    print(f"  기간: {start} ~ {end}  |  초기자본: {initial_capital:,}원")
    print("=" * 90)

    # 데이터 다운로드
    print("\n[1] 데이터 로드...")
    data = download_universe(BACKTEST_UNIVERSE)
    print(f"  → {len(data)}종목 로드 완료\n")

    # 백테스트 엔진
    engine = BacktestEngine(initial_capital=initial_capital)

    # 전략별 실행
    results = {}
    strategies = get_all_strategies()

    print("[2] 전략 백테스트 실행...")
    for strat in strategies:
        print(f"\n  ▶ {strat.name} ({strat.description})")
        result = engine.run(strat, data, start=start, end=end)
        sell_trades = [t for t in result["trades"] if t.get("side") == "SELL"]
        metrics = calc_metrics(result["equity_curve"], sell_trades)
        results[strat.name] = {
            "strategy": strat,
            "result":   result,
            "metrics":  metrics,
        }
        print(f"    → 거래 {metrics['trade_count']}건, 총수익 {metrics['total_return']:+.1f}%, MDD {metrics['mdd']:.1f}%")

    # Buy & Hold 벤치마크 (KODEX 200)
    print(f"\n  ▶ Buy&Hold (KODEX 200)")
    bh_equity = run_buy_and_hold("069500", data, initial_capital)
    bh_metrics = calc_metrics(bh_equity)
    results["Buy&Hold"] = {"metrics": bh_metrics}
    print(f"    → 총수익 {bh_metrics['total_return']:+.1f}%, MDD {bh_metrics['mdd']:.1f}%")

    # 비교표 출력
    print("\n" + "=" * 90)
    print("[3] 전략 비교 결과")
    print("=" * 90)
    print()
    print(metrics_header())
    print("-" * 90)

    # 카테고리별 정렬
    categories = {
        "스캘핑/초단기": ["스캘핑", "변동성돌파"],
        "단기":         ["MA교차", "MA+RSI필터", "단기스윙", "레버리지MA", "볼린저밴드", "RSI역추세", "RSI트레일링"],
        "장기":         ["섹터로테이션", "장기모멘텀", "듀얼모멘텀", "자산배분"],
        "벤치마크":     ["Buy&Hold"],
    }

    for cat_name, strat_names in categories.items():
        cat_results = [(n, results[n]) for n in strat_names if n in results]
        if not cat_results:
            continue
        cat_results.sort(key=lambda x: x[1]["metrics"]["cagr"], reverse=True)
        print(f"\n  ─── {cat_name} ───")
        for name, r in cat_results:
            print(format_metrics(name, r["metrics"]))

    print("-" * 90)

    # 전체 정렬 (상세 분석용)
    all_sorted = sorted(results.items(), key=lambda x: x[1]["metrics"]["cagr"], reverse=True)

    # 상세 분석
    print("\n[4] 상세 분석")
    print()
    best = all_sorted[0]
    print(f"  🏆 최고 CAGR: {best[0]} ({best[1]['metrics']['cagr']:+.1f}%)")

    # MDD 기준 최안전
    safest = min(results.items(), key=lambda x: abs(x[1]["metrics"]["mdd"]))
    print(f"  🛡️  최저 MDD: {safest[0]} ({safest[1]['metrics']['mdd']:.1f}%)")

    # Sharpe 기준 최효율
    best_sharpe = max(results.items(), key=lambda x: x[1]["metrics"]["sharpe"])
    print(f"  ⚡ 최고 Sharpe: {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe']:.2f})")

    # 거래 상세 (각 전략)
    print("\n[5] 전략별 거래 상세")
    print()
    for name, r in all_sorted:
        m = r["metrics"]
        if name == "Buy&Hold":
            continue
        result = r.get("result", {})
        trades = result.get("trades", [])
        sell_trades = [t for t in trades if t.get("side") == "SELL"]
        buy_trades  = [t for t in trades if t.get("side") == "BUY"]

        print(f"  [{name}]")
        print(f"    매수 {len(buy_trades)}건, 매도 {len(sell_trades)}건")
        if sell_trades:
            avg_hold = sum(t.get("hold_days", 0) for t in sell_trades) / len(sell_trades)
            print(f"    평균 보유: {avg_hold:.0f}일")
            wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
            losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
            if wins:
                print(f"    평균 수익: {sum(t['pnl'] for t in wins)/len(wins):+,.0f}원")
            if losses:
                print(f"    평균 손실: {sum(t['pnl'] for t in losses)/len(losses):+,.0f}원")
        # 미청산 포지션
        positions = result.get("positions", {})
        if positions:
            print(f"    미청산: {list(positions.keys())}")
        print()

    return results


if __name__ == "__main__":
    run_comparison()

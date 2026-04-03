#!/usr/bin/env python3
# 레짐 전략 8개 조합 비교 백테스트
import sys; sys.stdout.reconfigure(line_buffering=True)

from backtest.data_loader import download_universe, BACKTEST_UNIVERSE
from backtest.engine import BacktestEngine, run_buy_and_hold
from backtest.metrics import calc_metrics, format_metrics, metrics_header

from strategies.regime_strategy import RegimeStrategy


def get_regime_combos():
    """8개 레짐 조합 생성"""
    combos = []
    for regime in ["SMA200", "EMA60"]:
        for bull in ["MA교차", "레버리지MA"]:
            for bear in ["변동성돌파", "현금보유"]:
                r_short = "S200" if regime == "SMA200" else "E60"
                b_short = "MA" if bull == "MA교차" else "Lev"
                d_short = "VB" if bear == "변동성돌파" else "Cash"
                label = f"{r_short}_{b_short}_{d_short}"
                combos.append(RegimeStrategy(
                    regime_type=regime,
                    bull_strategy=bull,
                    bear_strategy=bear,
                    label=label,
                ))
    return combos


def run_regime_comparison(initial_capital: int = 1_000_000,
                          start: str = "2019-01-02", end: str = "2026-04-03"):
    """레짐 전략 8개 조합 비교"""
    print("=" * 95)
    print(f"  레짐 기반 전략 비교 (8개 조합)")
    print(f"  기간: {start} ~ {end}  |  초기자본: {initial_capital:,}원")
    print(f"  레짐: SMA(200) vs EMA(60)  |  상승: MA교차 vs 레버리지MA  |  하락: 변동성돌파 vs 현금")
    print("=" * 95)

    # 데이터
    print("\n[1] 데이터 로드...")
    data = download_universe(BACKTEST_UNIVERSE)
    print(f"  → {len(data)}종목 로드 완료\n")

    engine = BacktestEngine(initial_capital=initial_capital)
    combos = get_regime_combos()
    results = {}

    # 8개 조합 실행
    print("[2] 레짐 전략 백테스트...")
    for strat in combos:
        print(f"\n  ▶ {strat.name} ({strat.description})")
        result = engine.run(strat, data, start=start, end=end)
        sell_trades = [t for t in result["trades"] if t.get("side") == "SELL"]
        metrics = calc_metrics(result["equity_curve"], sell_trades)
        results[strat.name] = {"strategy": strat, "result": result, "metrics": metrics}
        print(f"    → 거래 {metrics['trade_count']}건, 총수익 {metrics['total_return']:+.1f}%, MDD {metrics['mdd']:.1f}%")

    # Buy & Hold 벤치마크
    print(f"\n  ▶ Buy&Hold (KODEX 200)")
    bh_equity = run_buy_and_hold("069500", data, initial_capital)
    bh_metrics = calc_metrics(bh_equity)
    results["Buy&Hold"] = {"metrics": bh_metrics}
    print(f"    → 총수익 {bh_metrics['total_return']:+.1f}%, MDD {bh_metrics['mdd']:.1f}%")

    # 비교표
    print("\n" + "=" * 95)
    print("[3] 레짐 전략 비교 결과")
    print("=" * 95)
    print()
    print(metrics_header())
    print("-" * 95)

    # 그룹별 정렬
    sma_results = [(n, r) for n, r in results.items() if n.startswith("S200")]
    ema_results = [(n, r) for n, r in results.items() if n.startswith("E60")]

    print("\n  ─── SMA(200) 레짐 ───")
    for n, r in sorted(sma_results, key=lambda x: x[1]["metrics"]["cagr"], reverse=True):
        print(format_metrics(n, r["metrics"]))

    print("\n  ─── EMA(60) 레짐 ───")
    for n, r in sorted(ema_results, key=lambda x: x[1]["metrics"]["cagr"], reverse=True):
        print(format_metrics(n, r["metrics"]))

    print("\n  ─── 벤치마크 ───")
    print(format_metrics("Buy&Hold", bh_metrics))
    print("-" * 95)

    # 순위표
    all_sorted = sorted(
        [(n, r) for n, r in results.items() if n != "Buy&Hold"],
        key=lambda x: x[1]["metrics"]["cagr"], reverse=True
    )

    print("\n[4] 종합 순위 (CAGR 기준)")
    print()
    for i, (name, r) in enumerate(all_sorted, 1):
        m = r["metrics"]
        # 레짐/상승/하락 분리
        parts = name.split("_")
        regime = "SMA(200)" if parts[0] == "S200" else "EMA(60)"
        bull   = "MA교차" if parts[1] == "MA" else "레버리지MA"
        bear   = "변동성돌파" if parts[2] == "VB" else "현금보유"
        print(f"  {i}위  {name:<18} CAGR {m['cagr']:>+6.1f}%  MDD {m['mdd']:>+7.1f}%  Sharpe {m['sharpe']:>5.2f}  거래 {m['trade_count']:>4}건")
        print(f"       레짐:{regime}  상승:{bull}  하락:{bear}")
        print()

    # 최적 조합
    best = all_sorted[0]
    print(f"  🏆 최적 조합: {best[0]}")
    print(f"     CAGR {best[1]['metrics']['cagr']:+.1f}%, MDD {best[1]['metrics']['mdd']:.1f}%, Sharpe {best[1]['metrics']['sharpe']:.2f}")

    return results


if __name__ == "__main__":
    run_regime_comparison()

#!/usr/bin/env python3
# KIS-ETF 자동매매 봇 — CLI 진입점
import sys; sys.stdout.reconfigure(line_buffering=True)

import argparse
from datetime import datetime


def cmd_test_connection():
    """API 연결 + 텔레그램 테스트"""
    print("=" * 50)
    print("KIS-ETF 봇 연결 테스트")
    print("=" * 50)

    # 1. KIS API 연결
    print("\n[1/3] KIS API 연결...")
    try:
        from kis_client import connect, get_balance
        connect()
        print("  ✓ KIS API 연결 성공")
    except Exception as e:
        print(f"  ✗ KIS API 연결 실패: {e}")
        return

    # 2. 잔고 조회
    print("\n[2/3] 잔고 조회...")
    try:
        bal = get_balance()
        print(f"  ✓ 예수금: {bal['deposit']:,}원")
        print(f"  ✓ 총 평가: {bal['total']:,}원")
        if bal['stocks']:
            print(f"  ✓ 보유 {len(bal['stocks'])}종목:")
            for s in bal['stocks']:
                print(f"    • {s['name']} ({s['ticker']}) {s['qty']}주 {s['pnl_pct']:+.1f}%")
        else:
            print("  ✓ 보유 종목 없음")
    except Exception as e:
        print(f"  ✗ 잔고 조회 실패: {e}")

    # 3. 텔레그램 테스트
    print("\n[3/3] 텔레그램 알림 테스트...")
    try:
        from notifier import test_connection
        ok = test_connection()
        if ok:
            print("  ✓ 텔레그램 알림 전송 성공")
        else:
            print("  △ 텔레그램 미설정 또는 전송 실패 (선택사항)")
    except Exception as e:
        print(f"  △ 텔레그램 오류: {e}")

    print("\n" + "=" * 50)
    print("연결 테스트 완료!")


def cmd_status():
    """현재 보유 현황"""
    from kis_client import get_balance
    from etf_db import get_trade_stats

    bal   = get_balance()
    stats = get_trade_stats(30)

    print("=" * 50)
    print(f"KIS-ETF 보유 현황 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 50)
    print(f"  예수금:     {bal['deposit']:>12,}원")
    print(f"  총 평가:    {bal['total']:>12,}원")
    print(f"  평가손익:   {bal['profit']:>+12,}원 ({bal['profit_rate']:+.2f}%)")
    print()

    if bal['stocks']:
        print("  보유 종목:")
        for s in bal['stocks']:
            print(f"    {s['name']:<20} {s['qty']:>4}주  {s['cur_price']:>8,}원  {s['pnl_pct']:>+6.1f}%  {s['pnl']:>+8,}원")
    else:
        print("  보유 종목 없음")

    print()
    print(f"  최근 30일: {stats['total']}건, 승률 {stats['win_rate']:.0f}%, 총손익 {stats['total_pnl']:+,}원")


def cmd_notify_status():
    """보유현황 텔레그램 발송"""
    from kis_client import get_balance
    import notifier

    bal = get_balance()
    holdings = []
    for s in bal["stocks"]:
        holdings.append({
            "name": s["name"], "qty": s["qty"],
            "pnl_pct": s["pnl_pct"],
        })

    ok = notifier.send_status(bal["deposit"], holdings, bal["profit"])
    if ok:
        print("텔레그램 보유현황 발송 완료")
    else:
        print("텔레그램 발송 실패")


def cmd_history(days: int):
    """최근 거래 내역"""
    from etf_db import _get_conn, _db_lock

    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE time >= date('now', ?)
            ORDER BY time DESC
        """, (f"-{days} days",)).fetchall()
        conn.close()

    print(f"최근 {days}일 거래 내역 ({len(rows)}건)")
    print("-" * 70)
    for r in rows:
        pnl_str = f"{r['pnl']:+,}원" if r['pnl'] else "-"
        print(f"  {r['time'][:16]}  {r['side']:<4} {r['name']:<16} {r['qty']:>4}주 × {r['price']:>8,}원  {pnl_str}")


def cmd_backtest(strategy_name: str = None, all_strategies: bool = False):
    """백테스트 실행"""
    from backtest.compare import run_comparison, get_all_strategies
    from backtest.data_loader import download_universe, BACKTEST_UNIVERSE
    from backtest.engine import BacktestEngine
    from backtest.metrics import calc_metrics, format_metrics, metrics_header

    if all_strategies or not strategy_name:
        # 전체 전략 비교
        run_comparison()
    else:
        # 단일 전략 실행
        strat_map = {s.name: s for s in get_all_strategies()}
        # 영문 이름 매핑
        alias = {"ma_cross": "MA교차", "dual_momentum": "듀얼모멘텀",
                 "bollinger": "볼린저밴드", "asset_rebalance": "자산배분",
                 "rsi": "RSI역추세"}
        name = alias.get(strategy_name, strategy_name)
        if name not in strat_map:
            print(f"전략 '{strategy_name}' 없음. 가능: {list(strat_map.keys())}")
            return
        strat = strat_map[name]
        data = download_universe(BACKTEST_UNIVERSE)
        engine = BacktestEngine()
        result = engine.run(strat, data)
        sell_trades = [t for t in result["trades"] if t.get("side") == "SELL"]
        m = calc_metrics(result["equity_curve"], sell_trades)
        print(f"\n{metrics_header()}")
        print("-" * 90)
        print(format_metrics(name, m))


def cmd_price(ticker: str):
    """ETF 현재가 조회"""
    from kis_client import get_price
    p = get_price(ticker)
    print(f"{p['name']} ({ticker})")
    print(f"  현재가: {p['price']:,}원 ({p['change']:+.2f}%)")
    print(f"  시가: {p['open']:,}  고가: {p['high']:,}  저가: {p['low']:,}")
    print(f"  거래량: {p['volume']:,}")


def main():
    parser = argparse.ArgumentParser(description="KIS-ETF 자동매매 봇")
    parser.add_argument("--test-connection", action="store_true", help="API 연결 테스트")
    parser.add_argument("--status", action="store_true", help="보유 현황 조회")
    parser.add_argument("--history", type=int, metavar="N", help="최근 N일 거래 내역")
    parser.add_argument("--price", type=str, metavar="TICKER", help="ETF 현재가 조회")
    parser.add_argument("--run-daily", action="store_true", help="데일리 자동매매 실행")
    parser.add_argument("--backtest", action="store_true", help="백테스트 실행")
    parser.add_argument("--strategy", type=str, help="백테스트 전략 이름")
    parser.add_argument("--all", action="store_true", help="전체 전략 백테스트")
    parser.add_argument("--dry-run", action="store_true", help="실제 주문 없이 시뮬레이션")
    parser.add_argument("--notify-status", action="store_true", help="보유현황 텔레그램 발송")
    parser.add_argument("--preview-signal", action="store_true", help="장전 신호 미리보기")
    parser.add_argument("--check-fills", action="store_true", help="체결 확인 알림")
    parser.add_argument("--backtest-regime", action="store_true", help="레짐 전략 8개 조합 비교")

    args = parser.parse_args()

    if args.test_connection:
        cmd_test_connection()
    elif args.status:
        cmd_status()
    elif args.notify_status:
        cmd_notify_status()
    elif args.preview_signal:
        from daily_runner import preview_signals
        preview_signals()
    elif args.check_fills:
        from daily_runner import check_fills
        check_fills()
    elif args.history:
        cmd_history(args.history)
    elif args.price:
        cmd_price(args.price)
    elif args.run_daily:
        from daily_runner import run_daily
        run_daily(dry_run=args.dry_run)
    elif args.backtest_regime:
        from backtest.compare_regime import run_regime_comparison
        run_regime_comparison()
    elif args.backtest:
        cmd_backtest(args.strategy, args.all)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

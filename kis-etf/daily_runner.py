#!/usr/bin/env python3
# 데일리 자동매매 실행기 — E60_MA_Cash 레짐 전략
# 상승장(EMA60↑): MA교차 4종목 매매 / 하락장(EMA60↓): 전량 매도→현금보유
import sys; sys.stdout.reconfigure(line_buffering=True)

import traceback
from datetime import datetime, date
from pykrx import stock as krx

from config import (
    INITIAL_CAPITAL, MAX_POSITION_PCT, MAX_POSITIONS, MAX_DAILY_BUYS,
    FEE_RATE, MA_SHORT, MA_LONG,
)
from kis_client import connect, get_account, get_balance, get_price, buy_market, sell_market, get_etf_name
import etf_db
import notifier


# ── E60_MA_Cash 전략 설정 ────────────────────────────────────────────
# 레짐 판별: KODEX200 종가 vs EMA(60)
REGIME_TICKER = "069500"   # 레짐 판별 기준 종목
REGIME_EMA    = 60         # EMA 기간

# MA교차 대상 (상승장에서만 활성)
MA_TICKERS = ["069500", "229200", "091160", "305720"]


def is_trading_day() -> bool:
    """오늘이 거래일인지 확인 (pykrx)"""
    today = date.today().strftime("%Y%m%d")
    try:
        # 오늘 날짜의 OHLCV가 있으면 거래일
        df = krx.get_market_ohlcv_by_date(today, today, "069500")
        return df is not None and len(df) > 0
    except Exception:
        # pykrx 실패 시 평일이면 거래일로 간주
        return date.today().weekday() < 5


def get_daily_ohlcv(ticker: str, days: int = 60):
    """최근 N일 일봉 조회 (pykrx)"""
    import pandas as pd
    from datetime import timedelta
    end   = date.today().strftime("%Y%m%d")
    start = (date.today() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
    df = krx.get_etf_ohlcv_by_date(start, end, ticker)
    if df is None or df.empty:
        df = krx.get_market_ohlcv_by_date(start, end, ticker)
    if df is None or df.empty:
        return None
    col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
    df = df.rename(columns=col_map)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df = df[df["volume"] > 0]
    return df.tail(days)


def calc_ema(series, span):
    """EMA 계산"""
    return series.ewm(span=span, adjust=False).mean()


def check_regime() -> tuple:
    """
    레짐 판별: KODEX200 종가 vs EMA(60)

    Returns:
        (is_bull: bool, ema_value: float, close_value: float)
    """
    df = get_daily_ohlcv(REGIME_TICKER, 100)
    if df is None or len(df) < REGIME_EMA + 5:
        print("  [레짐] 데이터 부족, 상승장 가정")
        return True, 0, 0

    close = df["close"]
    ema = close.ewm(span=REGIME_EMA, adjust=False).mean()
    cur_close = close.iloc[-1]
    cur_ema   = ema.iloc[-1]
    is_bull   = cur_close > cur_ema

    regime_str = "🟢 상승장" if is_bull else "🔴 하락장"
    diff_pct = (cur_close / cur_ema - 1) * 100
    print(f"  [레짐] {regime_str} | KODEX200 {cur_close:,.0f} vs EMA({REGIME_EMA}) {cur_ema:,.0f} ({diff_pct:+.1f}%)")

    return is_bull, cur_ema, cur_close


def run_bear_cash(holdings: dict) -> list:
    """하락장: 보유 종목 전량 매도 → 현금 보유"""
    orders = []
    for ticker, h in holdings.items():
        if ticker in MA_TICKERS:
            name = get_etf_name(ticker)
            orders.append({
                "ticker": ticker, "name": name, "action": "SELL",
                "qty": h["qty"], "price": h.get("cur_price", 0),
                "reason": "하락장→현금보유 (EMA60 하회)",
                "strategy": "E60_Cash",
            })
            print(f"  🔴 [{name}] 전량 매도 {h['qty']}주 (하락장 방어)")
    if not orders:
        print("  ⚪ 보유 종목 없음 (현금 대기 중)")
    return orders


def run_ma_cross(capital: int, holdings: dict) -> list:
    """
    MA교차 전략 실행

    Returns:
        [{"ticker", "name", "action", "qty", "price", "reason"}]
    """
    orders = []
    per_stock = capital * MAX_POSITION_PCT / 100  # 종목당 최대 배분

    for ticker in MA_TICKERS:
        name = get_etf_name(ticker)
        df = get_daily_ohlcv(ticker, 60)
        if df is None or len(df) < MA_LONG + 5:
            print(f"  [{name}] 데이터 부족, 스킵")
            continue

        close = df["close"]
        ema_s = calc_ema(close, MA_SHORT)
        ema_l = calc_ema(close, MA_LONG)

        cur_above  = ema_s.iloc[-1] > ema_l.iloc[-1]
        prev_above = ema_s.iloc[-2] > ema_l.iloc[-2]

        cur_price = int(close.iloc[-1])

        # 골든크로스 → 매수
        if cur_above and not prev_above and ticker not in holdings:
            qty = int(per_stock / cur_price)
            if qty > 0:
                orders.append({
                    "ticker": ticker, "name": name, "action": "BUY",
                    "qty": qty, "price": cur_price,
                    "reason": f"골든크로스 EMA({MA_SHORT})>EMA({MA_LONG})",
                    "strategy": "MA교차",
                })
                print(f"  🟢 [{name}] 매수 신호: EMA({MA_SHORT})={ema_s.iloc[-1]:.0f} > EMA({MA_LONG})={ema_l.iloc[-1]:.0f}")

        # 데드크로스 → 매도
        elif not cur_above and prev_above and ticker in holdings:
            qty = holdings[ticker]["qty"]
            orders.append({
                "ticker": ticker, "name": name, "action": "SELL",
                "qty": qty, "price": cur_price,
                "reason": f"데드크로스 EMA({MA_SHORT})<EMA({MA_LONG})",
                "strategy": "MA교차",
            })
            print(f"  🔴 [{name}] 매도 신호: EMA({MA_SHORT})={ema_s.iloc[-1]:.0f} < EMA({MA_LONG})={ema_l.iloc[-1]:.0f}")

        else:
            status = "보유중" if ticker in holdings else "관망"
            trend  = "상승" if cur_above else "하락"
            print(f"  ⚪ [{name}] {status} (추세: {trend})")

    return orders


def run_asset_rebalance(capital: int, holdings: dict) -> list:
    """
    자산배분 리밸런싱 (월 1회)

    Returns:
        [{"ticker", "name", "action", "qty", "price", "reason"}]
    """
    orders = []

    # 매월 첫 거래일에만 실행
    today = date.today()
    last_rebal = etf_db.get_bot_state("last_rebal_month", "")
    cur_month = today.strftime("%Y-%m")

    if last_rebal == cur_month:
        print(f"  [자산배분] 이번달({cur_month}) 리밸런싱 완료, 스킵")
        return orders

    # 첫 5거래일 이내인지 확인
    if today.day > 7:
        print(f"  [자산배분] 매월 초(7일 이내)에만 리밸런싱, 스킵")
        return orders

    print(f"  [자산배분] {cur_month} 월별 리밸런싱 실행")

    # 각 종목별 목표 금액 vs 현재 보유 금액
    for ticker, target_weight in REBAL_TICKERS.items():
        name = get_etf_name(ticker)
        try:
            price_info = get_price(ticker)
            cur_price = price_info["price"]
        except Exception:
            print(f"  [{name}] 시세 조회 실패, 스킵")
            continue

        target_amount = int(capital * target_weight)
        current_qty = holdings.get(ticker, {}).get("qty", 0)
        current_amount = current_qty * cur_price

        diff = target_amount - current_amount
        diff_pct = abs(diff) / target_amount * 100 if target_amount > 0 else 0

        # 5% 이상 차이나면 리밸런싱
        if diff_pct < 5:
            print(f"  ⚪ [{name}] 목표 {target_weight*100:.0f}% 유지 (차이 {diff_pct:.1f}%)")
            continue

        if diff > 0:
            # 매수 필요
            qty = int(diff / cur_price)
            if qty > 0:
                orders.append({
                    "ticker": ticker, "name": name, "action": "BUY",
                    "qty": qty, "price": cur_price,
                    "reason": f"리밸런싱 목표{target_weight*100:.0f}% (차이 {diff_pct:.0f}%)",
                    "strategy": "자산배분",
                })
                print(f"  🟢 [{name}] 매수 {qty}주 (목표 {target_weight*100:.0f}%, 부족 {diff:+,}원)")
        else:
            # 매도 필요
            sell_qty = min(int(abs(diff) / cur_price), current_qty)
            if sell_qty > 0:
                orders.append({
                    "ticker": ticker, "name": name, "action": "SELL",
                    "qty": sell_qty, "price": cur_price,
                    "reason": f"리밸런싱 목표{target_weight*100:.0f}% (초과 {diff_pct:.0f}%)",
                    "strategy": "자산배분",
                })
                print(f"  🔴 [{name}] 매도 {sell_qty}주 (목표 {target_weight*100:.0f}%, 초과 {abs(diff):,}원)")

    etf_db.save_bot_state("last_rebal_month", cur_month)
    return orders


def execute_orders(orders: list, dry_run: bool = False) -> list:
    """
    주문 실행

    Args:
        orders: 주문 리스트
        dry_run: True면 실제 주문 안 하고 로그만

    Returns:
        실행된 주문 리스트
    """
    executed = []
    buy_count = 0

    for o in orders:
        if o["action"] == "BUY":
            buy_count += 1
            if buy_count > MAX_DAILY_BUYS:
                print(f"  ⚠️ 일일 매수 한도({MAX_DAILY_BUYS}건) 초과, 스킵: {o['name']}")
                continue

        try:
            if dry_run:
                print(f"  [DRY] {o['action']} {o['name']} {o['qty']}주 × {o['price']:,}원")
                executed.append(o)
                continue

            if o["action"] == "BUY":
                result = buy_market(o["ticker"], o["qty"])
            else:
                result = sell_market(o["ticker"], o["qty"])

            o["order_number"] = result.get("number", "")
            executed.append(o)

            # DB 기록
            etf_db.add_trade(
                ticker=o["ticker"], name=o["name"], side=o["action"],
                qty=o["qty"], price=o["price"],
                strategy=o.get("strategy", ""), reason=o.get("reason", ""),
            )

            # 보유 종목 DB 업데이트
            if o["action"] == "BUY":
                etf_db.upsert_holding(o["ticker"], o["name"], o["qty"], o["price"], o.get("strategy", ""))
            else:
                etf_db.remove_holding(o["ticker"])

            # 텔레그램 알림
            if o["action"] == "BUY":
                notifier.send_order(o["ticker"], o["name"], "매수", o["qty"], o["price"], o.get("reason", ""))
            else:
                notifier.send_order(o["ticker"], o["name"], "매도", o["qty"], o["price"], o.get("reason", ""))

        except Exception as e:
            print(f"  ❌ 주문 실패 [{o['name']}]: {e}")
            notifier.send_error(f"주문 실패: {o['name']} {o['action']} - {e}")

    return executed


def run_daily(dry_run: bool = False):
    """데일리 자동매매 메인 루프"""
    now = datetime.now()
    print("=" * 60)
    print(f"  KIS-ETF 데일리 자동매매")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')} {'[DRY RUN]' if dry_run else ''}")
    print("=" * 60)

    # 0. 거래일 확인
    if not is_trading_day():
        print("\n오늘은 휴장일, 종료")
        return

    # 1. KIS 연결 + 잔고 조회
    print("\n[1] 잔고 조회...")
    try:
        connect()
        bal = get_balance()
        deposit = bal["deposit"]
        total   = bal["total"]
        print(f"  예수금: {deposit:,}원 | 총평가: {total:,}원 | 보유 {len(bal['stocks'])}종목")
    except Exception as e:
        print(f"  ❌ 잔고 조회 실패: {e}")
        notifier.send_error(f"잔고 조회 실패: {e}")
        return

    # 현재 보유 종목 → dict
    holdings = {}
    for s in bal["stocks"]:
        holdings[s["ticker"]] = s

    # 2. 레짐 판별
    print(f"\n[2] 레짐 판별 (EMA{REGIME_EMA})...")
    is_bull, ema_val, close_val = check_regime()

    # 3. 전략 실행 (레짐 분기)
    if is_bull:
        print(f"\n[3] 🟢 상승장 → MA교차 전략 (자본 {total:,}원)...")
        all_orders = run_ma_cross(total, holdings)
    else:
        print(f"\n[3] 🔴 하락장 → 전량 매도 → 현금 보유...")
        all_orders = run_bear_cash(holdings)
    print(f"\n[4] 주문 실행 ({len(all_orders)}건)...")
    if not all_orders:
        print("  주문 없음 (모두 HOLD)")
    else:
        executed = execute_orders(all_orders, dry_run=dry_run)
        print(f"  실행 완료: {len(executed)}건")

    # 5. 신호 기록 (DB)
    today_str = now.strftime("%Y-%m-%d")
    for o in all_orders:
        etf_db.add_signal(today_str, o["ticker"], o.get("strategy", ""),
                         o["action"], o.get("reason", ""), o.get("price", 0))

    # 6. 잔고 기록
    try:
        bal_after = get_balance()
        etf_db.save_balance(
            bal_after["total"],
            bal_after["deposit"],
            bal_after["holdings_value"],
        )
    except Exception:
        pass

    # 7. 요약 출력
    print(f"\n{'='*60}")
    buy_cnt  = len([o for o in all_orders if o["action"] == "BUY"])
    sell_cnt = len([o for o in all_orders if o["action"] == "SELL"])
    print(f"  매수 {buy_cnt}건 / 매도 {sell_cnt}건 / 보유 {len(holdings)}종목")
    print(f"{'='*60}")

    # 텔레그램 요약
    if all_orders:
        summary = f"📊 <b>[ETF 데일리 요약]</b> {today_str}\n"
        summary += f"매수 {buy_cnt}건 / 매도 {sell_cnt}건\n"
        for o in all_orders:
            emoji = "🟢" if o["action"] == "BUY" else "🔴"
            summary += f"{emoji} {o['name']} {o['action']} {o['qty']}주\n"
        notifier.send_message(summary)


def preview_signals():
    """장전 신호 미리보기 (08:55) — 주문 없이 신호만 계산 + 텔레그램 발송"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    print(f"[장전 미리보기] {today_str}")

    if not is_trading_day():
        print("  오늘은 휴장일")
        return

    try:
        connect()
        bal = get_balance()
        total = bal["total"]
    except Exception as e:
        print(f"  잔고 조회 실패: {e}")
        return

    holdings = {s["ticker"]: s for s in bal["stocks"]}

    # 레짐 판별
    print(f"  레짐 판별...")
    is_bull, _, _ = check_regime()

    if is_bull:
        print(f"  🟢 상승장 → MA교차 신호 확인...")
        all_signals = run_ma_cross(total, holdings)
    else:
        print(f"  🔴 하락장 → 전량매도 예정...")
        all_signals = run_bear_cash(holdings)
    notifier.send_preview(all_signals, today_str)
    print(f"  텔레그램 발송 완료 ({len(all_signals)}건)")


def check_fills():
    """체결 확인 (09:10) — 당일 주문 내역 조회 + 텔레그램 발송"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    print(f"[체결 확인] {today_str}")

    try:
        connect()
        acct = get_account()
        orders = acct.daily_orders(start=date.today())
    except Exception as e:
        print(f"  주문 내역 조회 실패: {e}")
        notifier.send_error(f"체결 확인 실패: {e}")
        return

    fills = []
    for o in orders:
        try:
            fills.append({
                "name":       o.name,
                "side":       "매수" if str(o.market).startswith("buy") or "매수" in str(o) else "매도",
                "qty":        int(o.qty) if hasattr(o, 'qty') else 0,
                "filled":     True,  # daily_orders는 체결 내역
                "fill_price": int(o.price) if hasattr(o, 'price') else 0,
            })
        except Exception:
            continue

    notifier.send_fill_report(fills, today_str)
    print(f"  텔레그램 발송 완료 ({len(fills)}건)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="실제 주문 없이 시뮬레이션")
    parser.add_argument("--preview", action="store_true", help="장전 신호 미리보기")
    parser.add_argument("--check-fills", action="store_true", help="체결 확인")
    args = parser.parse_args()

    if args.preview:
        preview_signals()
    elif args.check_fills:
        check_fills()
    else:
        run_daily(dry_run=args.dry_run)

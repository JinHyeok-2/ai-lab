#!/usr/bin/env python3
# 백테스트 엔진 + 페이퍼 트레이딩 시뮬레이터
# updater score_symbol 로직 그대로 재사용하여 과거 데이터 검증

import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from binance_client import get_klines, get_price, get_client
from indicators import calc_indicators
from alt_scanner import get_alt_futures_symbols

# ── 설정 (updater와 동일) ──
MIN_SCORE = 5
MIN_SCORE_SHORT = 7
SL_MULT = {'eth': 1.5, 'alt': 2.0}
TP_MULT = {'eth': 3.0, 'alt': 4.0}
MAX_SL_PCT = 5.0
FEE_PCT = 0.07
MIN_TP_PCT = 0.3
BLACKLIST = {'BRUSDT', 'SIRENUSDT', 'XAUUSDT', 'XAGUSDT', 'RIVERUSDT'}
RESULT_DIR = Path('/home/hyeok/01.APCC/00.ai-lab/backtest_results')
RESULT_DIR.mkdir(exist_ok=True)


def score_at(i15, i1h, i4h, btc_up):
    """스코어링 (updater 동일)"""
    sc = 0
    for ind, w in [(i15, 1), (i1h, 2), (i4h, 3)]:
        e20 = ind.get('ema20', 0) or 0
        e50 = ind.get('ema50', 0) or 0
        if e20 and e50:
            sc += w if e20 > e50 else -w
    rsi = i15.get('rsi', 50) or 50
    if rsi < 35: sc += 3
    elif rsi < 45: sc += 1
    elif rsi > 65: sc -= 3
    elif rsi > 55: sc -= 1
    rsi1h = i1h.get('rsi', 50) or 50
    if rsi1h < 40: sc += 2
    elif rsi1h > 60: sc -= 2
    if (i15.get('macd_hist', 0) or 0) > 0: sc += 1
    else: sc -= 1
    if (i1h.get('macd_hist', 0) or 0) > 0: sc += 2
    else: sc -= 2
    adx = i15.get('adx', 0) or 0
    if adx < 15: sc = int(sc * 0.5)
    if btc_up: sc += 2
    else: sc -= 2
    if sc >= MIN_SCORE: d = 'long'
    elif sc <= -MIN_SCORE_SHORT: d = 'short'
    else: d = 'wait'
    return sc, d, rsi, adx


def calc_sl_tp(entry, direction, atr, is_eth):
    """SL/TP 계산"""
    if atr <= 0: atr = entry * 0.02
    sl_m = SL_MULT['eth'] if is_eth else SL_MULT['alt']
    tp_m = TP_MULT['eth'] if is_eth else TP_MULT['alt']
    if direction == 'long':
        sl = entry - atr * sl_m
        tp = entry + atr * tp_m
    else:
        sl = entry + atr * sl_m
        tp = entry - atr * tp_m
    sl_dist = abs(entry - sl) / entry * 100
    if sl_dist > MAX_SL_PCT:
        sl = entry * (1 - MAX_SL_PCT / 100) if direction == 'long' else entry * (1 + MAX_SL_PCT / 100)
    risk = abs(entry - sl)
    if risk > 0 and abs(tp - entry) / risk < 2.0:
        tp = entry + risk * 2.0 if direction == 'long' else entry - risk * 2.0
    min_tp = entry * MIN_TP_PCT / 100
    if abs(tp - entry) < min_tp:
        tp = entry + min_tp if direction == 'long' else entry - min_tp
    return sl, tp


def run_backtest(symbols, days=7):
    """과거 데이터 백테스트 — 각 종목 독립적으로 분석"""
    print(f"=== 백테스트 시작 (최근 {days}일, {len(symbols)}종목) ===")
    print(f"설정: 롱≥{MIN_SCORE} 숏≤-{MIN_SCORE_SHORT} SL={SL_MULT} TP={TP_MULT}\n")

    trades = []

    for sym in symbols:
        if sym in BLACKLIST or sym == 'BTCUSDT':
            continue
        print(f"{sym} 분석 중...", end=" ")
        try:
            # 각 TF별 전체 데이터 로드 (get_klines가 DataFrame 반환)
            df15 = get_klines(sym, '15m', 500)
            df1h = get_klines(sym, '1h', 500)
            df4h = get_klines(sym, '4h', 200)
            df_btc = get_klines('BTCUSDT', '1h', 500)
            time.sleep(0.5)

            if len(df15) < 100 or len(df1h) < 60 or len(df4h) < 30:
                print(f"데이터 부족")
                continue

            # 저유동성 필터: 최근 100캔들의 가격 변동률이 1% 미만이면 스킵
            recent = df15.tail(100)
            price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean() * 100
            if price_range < 1.0:
                print(f"저유동성 스킵 (변동 {price_range:.1f}%)")
                continue

            sym_trades = 0
            cooldown_until = 0

            # 15분봉 순회 (60번째부터, 4캔들 간격 = 1시간)
            for i in range(60, len(df15) - 20, 4):
                # 쿨다운 중이면 스킵
                if i < cooldown_until:
                    continue

                # 현재 시점까지의 지표 계산
                try:
                    i15 = calc_indicators(df15.iloc[max(0, i-60):i].reset_index(drop=True))

                    # 1h: 15분봉 시간 기준으로 가장 가까운 1h 캔들 매칭
                    ts_15m = df15.iloc[i-1]['timestamp'] if 'timestamp' in df15.columns else df15.index[i-1]
                    # 1h 캔들 중 현재 15m 이전인 것만
                    mask_1h = df1h.index < i // 4 + len(df1h) - len(df15) // 4
                    i1h_end = min(i // 4 + 1, len(df1h))
                    i1h_start = max(0, i1h_end - 60)
                    if i1h_end - i1h_start < 20:
                        continue
                    i1h = calc_indicators(df1h.iloc[i1h_start:i1h_end].reset_index(drop=True))

                    # 4h
                    i4h_end = min(i // 16 + 1, len(df4h))
                    i4h_start = max(0, i4h_end - 60)
                    if i4h_end - i4h_start < 15:
                        continue
                    i4h = calc_indicators(df4h.iloc[i4h_start:i4h_end].reset_index(drop=True))

                    # BTC 추세
                    btc_end = min(i // 4 + 1, len(df_btc))
                    btc_start = max(0, btc_end - 60)
                    if btc_end - btc_start < 20:
                        continue
                    btc_ind = calc_indicators(df_btc.iloc[btc_start:btc_end].reset_index(drop=True))
                    btc_up = (btc_ind.get('ema20', 0) or 0) > (btc_ind.get('ema50', 0) or 0)
                except Exception:
                    continue

                # 스코어링
                score, direction, rsi, adx = score_at(i15, i1h, i4h, btc_up)
                if direction == 'wait':
                    continue

                # 진입가 = 현재 캔들 종가
                entry = float(df15.iloc[i-1]['close'])
                atr = i1h.get('atr', 0) or 0
                is_eth = sym == 'ETHUSDT'
                sl, tp = calc_sl_tp(entry, direction, atr, is_eth)

                # 이후 캔들로 시뮬레이션 (최대 40캔들 = 10시간)
                result = 'open'
                exit_price = entry
                exit_idx = i + 40
                for j in range(i, min(i + 40, len(df15))):
                    h = float(df15.iloc[j]['high'])
                    l = float(df15.iloc[j]['low'])
                    if direction == 'long':
                        if l <= sl:
                            result, exit_price, exit_idx = 'sl', sl, j
                            break
                        if h >= tp:
                            result, exit_price, exit_idx = 'tp', tp, j
                            break
                    else:
                        if h >= sl:
                            result, exit_price, exit_idx = 'sl', sl, j
                            break
                        if l <= tp:
                            result, exit_price, exit_idx = 'tp', tp, j
                            break
                else:
                    exit_price = float(df15.iloc[min(i+39, len(df15)-1)]['close'])

                # 가격 변동 없으면 무의미한 거래 → 스킵
                if abs(exit_price - entry) / entry < 0.001:
                    cooldown_until = exit_idx + 8
                    continue

                if direction == 'long':
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_pct = (entry - exit_price) / entry * 100
                pnl_pct -= FEE_PCT

                trades.append({
                    'symbol': sym, 'direction': direction, 'score': score,
                    'entry': round(entry, 6), 'sl': round(sl, 6), 'tp': round(tp, 6),
                    'exit': round(exit_price, 6), 'result': result,
                    'pnl_pct': round(pnl_pct, 2), 'rsi': round(rsi, 0), 'adx': round(adx, 0),
                    'candle_idx': i,
                })
                sym_trades += 1
                # 시뮬레이션 종료 시점 + 8캔들(2시간) 쿨다운
                cooldown_until = exit_idx + 8

            print(f"{sym_trades}건")
        except Exception as e:
            print(f"에러: {e}")
            continue

    # ── 결과 출력 ──
    print(f"\n{'='*60}")
    print(f"백테스트 결과 ({days}일, {len(symbols)}종목)")
    print(f"{'='*60}")

    if not trades:
        print("거래 없음 — MIN_SCORE가 너무 높거나 데이터 부족")
        return

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    total_pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0

    print(f"\n총 거래: {len(trades)}건 | 승 {len(wins)} | 패 {len(losses)} | 승률 {len(wins)/len(trades)*100:.0f}%")
    print(f"총 PnL: {total_pnl:+.2f}% (수수료 차감 후)")
    print(f"평균 수익: {avg_win:+.2f}% | 평균 손실: {avg_loss:+.2f}%")

    # 방향별
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']
    if longs:
        l_w = len([t for t in longs if t['pnl_pct'] > 0])
        print(f"\n롱: {len(longs)}건 승률 {l_w/len(longs)*100:.0f}% PnL {sum(t['pnl_pct'] for t in longs):+.2f}%")
    if shorts:
        s_w = len([t for t in shorts if t['pnl_pct'] > 0])
        print(f"숏: {len(shorts)}건 승률 {s_w/len(shorts)*100:.0f}% PnL {sum(t['pnl_pct'] for t in shorts):+.2f}%")

    # 종목별
    print(f"\n종목별:")
    by_sym = {}
    for t in trades:
        by_sym.setdefault(t['symbol'], []).append(t)
    for s, ts in sorted(by_sym.items(), key=lambda x: sum(t['pnl_pct'] for t in x[1]), reverse=True):
        w = len([t for t in ts if t['pnl_pct'] > 0])
        pnl = sum(t['pnl_pct'] for t in ts)
        print(f"  {s:14s} {len(ts):2d}건 승률 {w/len(ts)*100:3.0f}% PnL {pnl:+6.2f}%")

    # 점수 구간별
    print(f"\n점수 구간별:")
    for lo, hi, label in [(5, 8, '+5~7'), (8, 99, '+8이상'), (-99, -10, '-10이하'), (-10, -7, '-7~-9')]:
        grp = [t for t in trades if lo <= t['score'] < hi]
        if grp:
            w = len([t for t in grp if t['pnl_pct'] > 0])
            pnl = sum(t['pnl_pct'] for t in grp)
            print(f"  {label:8s}: {len(grp):3d}건 승률 {w/len(grp)*100:3.0f}% PnL {pnl:+6.2f}%")

    # 거래 상세
    print(f"\n거래 상세:")
    for t in trades:
        f = 'W' if t['pnl_pct'] > 0 else 'L'
        print(f"  {t['symbol']:14s} {t['direction']:5s} sc={t['score']:+3d} ${t['entry']:.4f}→${t['exit']:.4f} {t['result']:4s} {t['pnl_pct']:+.2f}% {f}")

    # 결과 저장
    rf = RESULT_DIR / f"bt_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(rf, 'w') as f:
        json.dump({'config': {'min_score': MIN_SCORE, 'min_score_short': MIN_SCORE_SHORT,
                              'sl_mult': SL_MULT, 'tp_mult': TP_MULT, 'days': days},
                   'summary': {'total': len(trades), 'wins': len(wins), 'losses': len(losses),
                               'win_rate': round(len(wins)/len(trades)*100, 1),
                               'total_pnl_pct': round(total_pnl, 2)},
                   'trades': trades}, f, indent=2)
    print(f"\n결과 저장: {rf}")


def run_paper():
    """페이퍼 트레이딩 (실시간, 주문 없음)"""
    print("=== 페이퍼 트레이딩 시작 ===")
    print(f"롱≥{MIN_SCORE} 숏≤-{MIN_SCORE_SHORT} | 실제 주문 없이 가상 거래\n")

    positions = {}
    history = []
    cycle = 0

    while True:
        cycle += 1
        now = datetime.now().strftime('%H:%M:%S')
        print(f"\n--- #{cycle} ({now}) ---")
        try:
            btc_ind = calc_indicators(get_klines('BTCUSDT', '1h', 60))
            btc_up = (btc_ind.get('ema20', 0) or 0) > (btc_ind.get('ema50', 0) or 0)

            # 보유 포지션 SL/TP 체크
            for sym in list(positions.keys()):
                p = positions[sym]
                cur = get_price(sym)
                hit = None
                if p['direction'] == 'long':
                    if cur <= p['sl']: hit = ('sl', p['sl'])
                    elif cur >= p['tp']: hit = ('tp', p['tp'])
                else:
                    if cur >= p['sl']: hit = ('sl', p['sl'])
                    elif cur <= p['tp']: hit = ('tp', p['tp'])

                if hit:
                    res, ep = hit
                    pnl = ((ep - p['entry']) / p['entry'] * 100 if p['direction'] == 'long'
                           else (p['entry'] - ep) / p['entry'] * 100) - FEE_PCT
                    print(f"  {res.upper()} {sym} {p['direction']} {pnl:+.2f}%")
                    history.append({**p, 'exit': ep, 'result': res, 'pnl_pct': round(pnl, 2)})
                    del positions[sym]
                else:
                    upnl = ((cur - p['entry']) / p['entry'] * 100 if p['direction'] == 'long'
                            else (p['entry'] - cur) / p['entry'] * 100)
                    print(f"  보유 {sym} {p['direction']} uPnL={upnl:+.2f}%")

            # 새 종목 스캔
            alts = get_alt_futures_symbols(15)
            for sym in ['ETHUSDT'] + alts:
                if sym in positions or sym in BLACKLIST or sym == 'BTCUSDT':
                    continue
                if len(positions) >= 5:
                    break
                try:
                    i15 = calc_indicators(get_klines(sym, '15m', 60))
                    i1h = calc_indicators(get_klines(sym, '1h', 60))
                    i4h = calc_indicators(get_klines(sym, '4h', 60))
                    sc, d, rsi, adx = score_at(i15, i1h, i4h, btc_up)
                    if d == 'wait': continue
                    entry = get_price(sym)
                    atr = i1h.get('atr', 0) or 0
                    sl, tp = calc_sl_tp(entry, d, atr, sym == 'ETHUSDT')
                    positions[sym] = {'symbol': sym, 'direction': d, 'entry': entry,
                                      'sl': sl, 'tp': tp, 'score': sc,
                                      'time': datetime.now().strftime('%Y-%m-%d %H:%M')}
                    sl_p = abs(entry - sl) / entry * 100
                    tp_p = abs(tp - entry) / entry * 100
                    print(f"  진입 {sym} {d} sc={sc:+d} @${entry:.4f} SL({sl_p:.1f}%) TP({tp_p:.1f}%)")
                    time.sleep(0.3)
                except Exception:
                    continue

            if history:
                w = len([h for h in history if h['pnl_pct'] > 0])
                tp = sum(h['pnl_pct'] for h in history)
                print(f"  통계: {len(history)}건 승률 {w/len(history)*100:.0f}% PnL {tp:+.2f}%")
        except Exception as e:
            print(f"  오류: {e}")
        print(f"  다음: 300초 후")
        time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['backtest', 'paper'], default='backtest')
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()

    if args.mode == 'backtest':
        syms = ['ETHUSDT'] + get_alt_futures_symbols(15)
        run_backtest(syms, days=args.days)
    else:
        run_paper()

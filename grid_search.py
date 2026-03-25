#!/usr/bin/env python3
# 파라미터 그리드 서치 — 최적 MIN_SCORE / SL / TP 조합 탐색
# GPU 불필요, CPU만 사용

import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
import json
from datetime import datetime
from pathlib import Path
from binance_client import get_klines
from indicators import calc_indicators
from alt_scanner import get_alt_futures_symbols

FEE_PCT = 0.07
MAX_SL_PCT = 5.0
MIN_TP_PCT = 0.3
BLACKLIST = {'BRUSDT', 'SIRENUSDT', 'XAUUSDT', 'XAGUSDT', 'RIVERUSDT'}

# ── 그리드 파라미터 ──
GRID = {
    'min_score': [3, 4, 5],
    'short_extra': [0, 2, 99],  # 99 = 숏 비활성화
    'sl_mult': [1.5, 2.0, 2.5],
    'tp_mult': [3.0, 4.0, 5.0],
}

RESULT_DIR = Path('/home/hyeok/01.APCC/00.ai-lab/backtest_results')
RESULT_DIR.mkdir(exist_ok=True)


def load_data(symbols):
    """데이터 1회 로드 (전 조합 재사용)"""
    data = {}
    btc_1h = get_klines('BTCUSDT', '1h', 500)
    data['BTCUSDT_1h'] = btc_1h
    print(f"BTC 1h 로드 ({len(btc_1h)}캔들)")

    for sym in symbols:
        if sym in BLACKLIST or sym == 'BTCUSDT':
            continue
        try:
            df15 = get_klines(sym, '15m', 500)
            df1h = get_klines(sym, '1h', 500)
            df4h = get_klines(sym, '4h', 200)
            # 저유동성 필터
            recent = df15.tail(100)
            rng = (recent['high'].max() - recent['low'].min()) / recent['close'].mean() * 100
            if rng < 1.0:
                print(f"  {sym}: 저유동성 ({rng:.1f}%) 스킵")
                continue
            if len(df15) < 100 or len(df1h) < 60 or len(df4h) < 30:
                continue
            data[sym] = {'15m': df15, '1h': df1h, '4h': df4h}
            print(f"  {sym}: 15m={len(df15)} 1h={len(df1h)} 4h={len(df4h)} (변동 {rng:.1f}%)")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {sym}: 에러 {e}")
    return data


def score_at(i15, i1h, i4h, btc_up):
    """스코어링 (MIN_SCORE 무관하게 점수만 반환)"""
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
    return sc, rsi, adx


def precompute_signals(data):
    """모든 종목의 모든 시점 점수를 미리 계산 (1회만)"""
    signals = {}  # {sym: [(candle_idx, score, entry_price, atr, rsi, adx, btc_up), ...]}
    btc_1h = data['BTCUSDT_1h']

    for sym, d in data.items():
        if sym == 'BTCUSDT_1h':
            continue
        df15 = d['15m']
        df1h = d['1h']
        df4h = d['4h']
        sym_signals = []

        for i in range(60, len(df15) - 20, 4):
            try:
                i15 = calc_indicators(df15.iloc[max(0, i-60):i].reset_index(drop=True))
                i1h_end = min(i // 4 + 1, len(df1h))
                i1h_start = max(0, i1h_end - 60)
                if i1h_end - i1h_start < 20: continue
                i1h = calc_indicators(df1h.iloc[i1h_start:i1h_end].reset_index(drop=True))
                i4h_end = min(i // 16 + 1, len(df4h))
                i4h_start = max(0, i4h_end - 60)
                if i4h_end - i4h_start < 15: continue
                i4h = calc_indicators(df4h.iloc[i4h_start:i4h_end].reset_index(drop=True))
                btc_end = min(i // 4 + 1, len(btc_1h))
                btc_start = max(0, btc_end - 60)
                if btc_end - btc_start < 20: continue
                btc_ind = calc_indicators(btc_1h.iloc[btc_start:btc_end].reset_index(drop=True))
                btc_up = (btc_ind.get('ema20', 0) or 0) > (btc_ind.get('ema50', 0) or 0)

                sc, rsi, adx = score_at(i15, i1h, i4h, btc_up)
                entry = float(df15.iloc[i-1]['close'])
                atr = i1h.get('atr', 0) or 0

                sym_signals.append({
                    'idx': i, 'score': sc, 'entry': entry, 'atr': atr,
                    'rsi': rsi, 'adx': adx,
                })
            except Exception:
                continue

        signals[sym] = sym_signals
        print(f"  {sym}: {len(sym_signals)} 시그널")
    return signals


def simulate_combo(signals, data, min_score, short_extra, sl_mult, tp_mult):
    """주어진 파라미터 조합으로 시뮬레이션"""
    trades = []
    short_threshold = -(min_score + short_extra)

    for sym, sigs in signals.items():
        df15 = data[sym]['15m']
        is_eth = sym == 'ETHUSDT'
        sl_m = 1.5 if is_eth else sl_mult
        tp_m = 3.0 if is_eth else tp_mult
        cooldown_until = 0

        for sig in sigs:
            i = sig['idx']
            if i < cooldown_until:
                continue

            sc = sig['score']
            if sc >= min_score:
                direction = 'long'
            elif sc <= short_threshold:
                direction = 'short'
            else:
                continue

            entry = sig['entry']
            atr = sig['atr']
            if atr <= 0: atr = entry * 0.02
            if entry > 0 and atr / entry > 0.15: continue

            # SL/TP
            if direction == 'long':
                sl = entry - atr * sl_m
                tp = entry + atr * tp_m
            else:
                sl = entry + atr * sl_m
                tp = entry - atr * tp_m

            # SL 캡
            sl_dist = abs(entry - sl) / entry * 100
            if sl_dist > MAX_SL_PCT:
                sl = entry * (1 - MAX_SL_PCT / 100) if direction == 'long' else entry * (1 + MAX_SL_PCT / 100)
            # R:R 최소 1:2
            risk = abs(entry - sl)
            if risk > 0 and abs(tp - entry) / risk < 2.0:
                tp = entry + risk * 2.0 if direction == 'long' else entry - risk * 2.0
            # TP 최소 거리
            min_tp = entry * MIN_TP_PCT / 100
            if abs(tp - entry) < min_tp:
                tp = entry + min_tp if direction == 'long' else entry - min_tp

            # 시뮬레이션
            result = 'open'
            exit_price = entry
            exit_idx = i + 40
            for j in range(i, min(i + 40, len(df15))):
                h = float(df15.iloc[j]['high'])
                l = float(df15.iloc[j]['low'])
                if direction == 'long':
                    if l <= sl: result, exit_price, exit_idx = 'sl', sl, j; break
                    if h >= tp: result, exit_price, exit_idx = 'tp', tp, j; break
                else:
                    if h >= sl: result, exit_price, exit_idx = 'sl', sl, j; break
                    if l <= tp: result, exit_price, exit_idx = 'tp', tp, j; break
            else:
                exit_price = float(df15.iloc[min(i+39, len(df15)-1)]['close'])

            if abs(exit_price - entry) / entry < 0.001:
                cooldown_until = exit_idx + 8
                continue

            if direction == 'long':
                pnl_pct = (exit_price - entry) / entry * 100
            else:
                pnl_pct = (entry - exit_price) / entry * 100
            pnl_pct -= FEE_PCT

            trades.append({
                'symbol': sym, 'direction': direction, 'score': sc,
                'result': result, 'pnl_pct': round(pnl_pct, 2),
            })
            cooldown_until = exit_idx + 8

    return trades


def run_grid_search():
    print("=== 그리드 서치 시작 ===")
    total = (len(GRID['min_score']) * len(GRID['short_extra']) *
             len(GRID['sl_mult']) * len(GRID['tp_mult']))
    print(f"조합 수: {total}개\n")

    # 1. 데이터 로드 (1회)
    print("데이터 로딩...")
    syms = ['ETHUSDT'] + get_alt_futures_symbols(15)
    data = load_data(syms)
    print(f"\n유효 종목: {len(data) - 1}개")

    # 2. 시그널 미리 계산 (1회)
    print("\n시그널 계산...")
    signals = precompute_signals(data)
    total_sigs = sum(len(v) for v in signals.values())
    print(f"총 시그널: {total_sigs}개\n")

    # 3. 조합별 시뮬레이션
    results = []
    done = 0
    for ms in GRID['min_score']:
        for se in GRID['short_extra']:
            for sl in GRID['sl_mult']:
                for tp in GRID['tp_mult']:
                    trades = simulate_combo(signals, data, ms, se, sl, tp)
                    done += 1

                    if not trades:
                        results.append({
                            'min_score': ms, 'short_extra': se, 'sl_mult': sl, 'tp_mult': tp,
                            'trades': 0, 'wins': 0, 'win_rate': 0, 'pnl': 0,
                            'long_trades': 0, 'short_trades': 0,
                        })
                        continue

                    wins = len([t for t in trades if t['pnl_pct'] > 0])
                    pnl = sum(t['pnl_pct'] for t in trades)
                    longs = [t for t in trades if t['direction'] == 'long']
                    shorts = [t for t in trades if t['direction'] == 'short']

                    results.append({
                        'min_score': ms, 'short_extra': se, 'sl_mult': sl, 'tp_mult': tp,
                        'trades': len(trades), 'wins': wins,
                        'win_rate': round(wins / len(trades) * 100, 1),
                        'pnl': round(pnl, 2),
                        'avg_pnl': round(pnl / len(trades), 2),
                        'long_trades': len(longs),
                        'long_wr': round(len([t for t in longs if t['pnl_pct'] > 0]) / max(len(longs), 1) * 100, 0),
                        'short_trades': len(shorts),
                        'short_wr': round(len([t for t in shorts if t['pnl_pct'] > 0]) / max(len(shorts), 1) * 100, 0),
                    })

                    if done % 9 == 0:
                        print(f"  진행: {done}/{total} ({done/total*100:.0f}%)")

    # 4. 결과 정렬 (PnL 기준)
    results.sort(key=lambda x: x['pnl'], reverse=True)

    print(f"\n{'='*80}")
    print(f"그리드 서치 결과 — TOP 10 (PnL 기준)")
    print(f"{'='*80}")
    print(f"{'#':>2} {'MS':>2} {'숏+':>3} {'SL':>4} {'TP':>4} | {'건수':>4} {'승률':>5} {'PnL':>8} {'평균':>6} | {'롱':>3}{'롱WR':>5} {'숏':>3}{'숏WR':>5}")
    print("-" * 80)
    for i, r in enumerate(results[:10], 1):
        se_label = 'OFF' if r['short_extra'] >= 99 else f"+{r['short_extra']}"
        print(f"{i:2d} {r['min_score']:2d} {se_label:>3} {r['sl_mult']:4.1f} {r['tp_mult']:4.1f} | "
              f"{r['trades']:4d} {r['win_rate']:4.1f}% {r['pnl']:+7.2f}% {r.get('avg_pnl',0):+5.2f} | "
              f"{r['long_trades']:3d} {r.get('long_wr',0):4.0f}% {r['short_trades']:3d} {r.get('short_wr',0):4.0f}%")

    print(f"\n{'='*80}")
    print(f"WORST 5 (참고)")
    print(f"{'='*80}")
    for i, r in enumerate(results[-5:], 1):
        se_label = 'OFF' if r['short_extra'] >= 99 else f"+{r['short_extra']}"
        print(f"{i:2d} {r['min_score']:2d} {se_label:>3} {r['sl_mult']:4.1f} {r['tp_mult']:4.1f} | "
              f"{r['trades']:4d} {r['win_rate']:4.1f}% {r['pnl']:+7.2f}% | "
              f"롱 {r['long_trades']} 숏 {r['short_trades']}")

    # 숏 ON vs OFF 비교
    print(f"\n{'='*80}")
    print(f"숏 ON vs OFF 비교 (같은 MS/SL/TP끼리)")
    print(f"{'='*80}")
    for ms in GRID['min_score']:
        on = [r for r in results if r['min_score'] == ms and r['short_extra'] < 99]
        off = [r for r in results if r['min_score'] == ms and r['short_extra'] >= 99]
        best_on = max(on, key=lambda x: x['pnl']) if on else None
        best_off = max(off, key=lambda x: x['pnl']) if off else None
        if best_on and best_off:
            print(f"  MS={ms}: 숏ON 최고 {best_on['pnl']:+.2f}% (SL{best_on['sl_mult']}/TP{best_on['tp_mult']}) | "
                  f"숏OFF 최고 {best_off['pnl']:+.2f}% (SL{best_off['sl_mult']}/TP{best_off['tp_mult']})")

    # 저장
    rf = RESULT_DIR / f"grid_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(rf, 'w') as f:
        json.dump({'grid': GRID, 'results': results[:20]}, f, indent=2)
    print(f"\n결과 저장: {rf}")


if __name__ == '__main__':
    run_grid_search()

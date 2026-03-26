#!/usr/bin/env python3
# 백테스트 엔진 v2 — updater 최신 로직 동기화 + 트레일링 + 그리드서치
# Phase 1~3: 현재 파라미터 검증 → 243조합 그리드서치 → 최적 파라미터 도출

import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
import json
import itertools
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from binance_client import get_klines, get_client
from indicators import calc_indicators
from alt_scanner import get_alt_futures_symbols

# ── 기본 설정 (updater 동기화) ──
DEFAULTS = {
    'min_score': 5,
    'sl_mult': 1.5,       # ATR × 1.5
    'tp_mult': 3.5,       # ATR × 3.5 (#128)
    'trail_start': 0.7,   # 0.7% 수익부터 트레일링 (#127)
    'noon_bonus': 2,      # 12-15시 MIN_SCORE +2 (#126)
}
MAX_SL_PCT = 5.0          # 실제 기준 최대 SL (레버리지 2x → 1x 2.5%)
MIN_SL_PCT = 2.0          # 실제 기준 최소 SL (#130, 1x 1.0%)
LEV = 2                   # 알트 레버리지
FEE_PCT = 0.07            # 수수료 0.07% (편도 0.04% × 양쪽 - 메이커 리베이트)
BLACKLIST = {'BRUSDT', 'SIRENUSDT', 'XAUUSDT', 'XAGUSDT', 'RIVERUSDT'}
RESULT_DIR = Path('/home/hyeok/01.APCC/00.ai-lab/backtest_results')
RESULT_DIR.mkdir(exist_ok=True)


def score_at(i15, i1h, i4h, btc_up, hour=12, params=None):
    """스코어링 — updater score_symbol() 동기화 (2026-03-26)"""
    p = params or DEFAULTS
    sc = 0

    # 1. EMA 배열 (4h 가중치 강화)
    for ind, w in [(i15, 2), (i1h, 2), (i4h, 5)]:
        e20 = ind.get('ema20', 0) or 0
        e50 = ind.get('ema50', 0) or 0
        if e20 and e50:
            sc += w if e20 > e50 else -w

    # 2. RSI
    rsi = i15.get('rsi', 50) or 50
    if rsi < 35: sc += 3
    elif rsi < 45: sc += 1
    elif rsi > 65: sc -= 3
    elif rsi > 55: sc -= 1
    rsi1h = i1h.get('rsi', 50) or 50
    if rsi1h < 40: sc += 2
    elif rsi1h > 60: sc -= 2

    # 2-1. 눌림 보너스
    if 40 <= rsi <= 60:
        sc += 2

    # 3. MACD
    if (i15.get('macd_hist', 0) or 0) > 0: sc += 1
    else: sc -= 1
    if (i1h.get('macd_hist', 0) or 0) > 0: sc += 2
    else: sc -= 2

    # 4. ADX
    adx = i15.get('adx', 0) or 0
    adx_1h = i1h.get('adx', 0) or 0
    if adx < 15:
        sc = int(sc * 0.5)
    if adx_1h >= 25:
        sc += 2

    # 5. BTC 동조
    if btc_up: sc += 2
    else: sc -= 2

    # 7. BB 지지/저항
    px = i15.get('close', 0) or (i1h.get('close', 0) or 0)
    bb_upper = i1h.get('bb_upper', 0) or 0
    bb_lower = i1h.get('bb_lower', 0) or 0
    if bb_upper and bb_lower and bb_upper > bb_lower and px > 0:
        bb_range = bb_upper - bb_lower
        bb_pct = (px - bb_lower) / bb_range * 100
        if bb_pct < 30: sc += 2
        elif bb_pct > 70: sc -= 2
        elif 40 < bb_pct < 60: sc = int(sc * 0.7)
        bb_width_pct = bb_range / px * 100
        if bb_width_pct < 1.5:
            sc = int(sc * 1.3)

    # 9. 거래량
    vol = i15.get('volume', 0) or 0
    vol_avg = i15.get('vol_ma20', 0) or 0
    if vol_avg and vol_avg > 0:
        vol_ratio = vol / vol_avg
        if vol_ratio > 2.0:
            sc += 2 if sc > 0 else -2
        elif vol_ratio < 0.5:
            sc = int(sc * 0.7)

    # 하락추세 필터 (#124)
    e20_4h = i4h.get('ema20', 0) or 0
    e50_4h = i4h.get('ema50', 0) or 0
    if e20_4h and e50_4h and e20_4h < e50_4h and rsi1h < 40:
        sc = min(sc, p['min_score'] - 1)

    # 방향 — 롱 전용 + 12-15시 보너스 (#126)
    min_sc = p['min_score'] + p['noon_bonus'] if 12 <= hour < 15 else p['min_score']
    direction = 'long' if sc >= min_sc else 'wait'

    # RSI 극단값 차단
    if direction == 'long' and rsi > 85:
        direction = 'wait'

    return sc, direction, rsi, adx, adx_1h


def calc_sl_tp(entry, atr, params=None):
    """SL/TP 계산 — 롱 전용, 최신 로직"""
    p = params or DEFAULTS
    if atr <= 0: atr = entry * 0.02

    sl = entry - atr * p['sl_mult']
    tp = entry + atr * p['tp_mult']

    # SL 최소 거리 (#130)
    min_sl_dist = entry * (MIN_SL_PCT / LEV / 100)
    if abs(entry - sl) < min_sl_dist:
        sl = entry - min_sl_dist

    # SL 최대 거리
    max_sl_dist = entry * (MAX_SL_PCT / LEV / 100)
    if abs(entry - sl) > max_sl_dist:
        sl = entry - max_sl_dist

    # R:R 최소 1.5
    risk = abs(entry - sl)
    if risk > 0 and abs(tp - entry) / risk < 1.5:
        tp = entry + risk * 1.5

    # TP 최소 4% real
    min_tp = entry * (0.04 / LEV)
    if abs(tp - entry) < min_tp:
        tp = entry + min_tp

    return sl, tp


def simulate_trade(df15, i, entry, sl, tp, params=None):
    """거래 시뮬레이션 — SL/TP/트레일링"""
    p = params or DEFAULTS
    trail_start = p['trail_start'] / 100  # % → 비율

    peak = entry
    trail_active = False

    for j in range(i, min(i + 40, len(df15))):
        h = float(df15.iloc[j]['high'])
        l = float(df15.iloc[j]['low'])

        # SL 히트
        if l <= sl:
            pnl = (sl - entry) / entry * 100
            return 'sl', sl, j, pnl

        # TP 히트
        if h >= tp:
            pnl = (tp - entry) / entry * 100
            return 'tp', tp, j, pnl

        # 피크 업데이트
        if h > peak:
            peak = h

        # 트레일링 체크
        pnl_from_entry = (peak - entry) / entry
        if pnl_from_entry >= trail_start:
            trail_active = True
            # ATR 비례 트레일 거리 (간이: 진입가 대비 수익의 40%를 허용)
            trail_dist = max(pnl_from_entry * 0.4, 0.003)  # 최소 0.3%
            drop_from_peak = (peak - l) / peak
            if drop_from_peak >= trail_dist:
                exit_px = peak * (1 - trail_dist)
                pnl = (exit_px - entry) / entry * 100
                return 'trail', exit_px, j, pnl

    # 타임아웃 (40캔들 = 10시간)
    exit_px = float(df15.iloc[min(i + 39, len(df15) - 1)]['close'])
    pnl = (exit_px - entry) / entry * 100
    return 'timeout', exit_px, i + 39, pnl


def run_backtest(symbols, params=None, verbose=True):
    """백테스트 실행 — 반환: trades 리스트"""
    p = params or DEFAULTS
    trades = []

    for sym in symbols:
        if sym in BLACKLIST or sym == 'BTCUSDT':
            continue
        try:
            df15 = get_klines(sym, '15m', 500)
            df1h = get_klines(sym, '1h', 500)
            df4h = get_klines(sym, '4h', 200)
            df_btc = get_klines('BTCUSDT', '1h', 500)
            time.sleep(0.3)

            if len(df15) < 100 or len(df1h) < 60 or len(df4h) < 30:
                continue

            # 저유동성 필터
            recent = df15.tail(100)
            pr = (recent['high'].max() - recent['low'].min()) / recent['close'].mean() * 100
            if pr < 1.0:
                continue

            sym_trades = 0
            cooldown_until = 0

            for i in range(60, len(df15) - 20, 4):
                if i < cooldown_until:
                    continue

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

                    btc_end = min(i // 4 + 1, len(df_btc))
                    btc_start = max(0, btc_end - 60)
                    if btc_end - btc_start < 20: continue
                    btc_ind = calc_indicators(df_btc.iloc[btc_start:btc_end].reset_index(drop=True))
                    btc_up = (btc_ind.get('ema20', 0) or 0) > (btc_ind.get('ema50', 0) or 0)
                except Exception:
                    continue

                # 시간대 추정 (15분봉 인덱스 기반)
                try:
                    hour = df15.iloc[i-1].get('time', datetime.now()).hour if hasattr(df15.iloc[i-1].get('time', 0), 'hour') else 12
                except:
                    hour = 12

                score, direction, rsi, adx, adx_1h = score_at(i15, i1h, i4h, btc_up, hour, p)
                if direction == 'wait':
                    continue

                entry = float(df15.iloc[i-1]['close'])
                atr = i1h.get('atr', 0) or 0
                sl, tp = calc_sl_tp(entry, atr, p)

                result, exit_px, exit_idx, pnl_pct = simulate_trade(df15, i, entry, sl, tp, p)
                pnl_pct -= FEE_PCT

                if abs(exit_px - entry) / entry < 0.001:
                    cooldown_until = exit_idx + 8
                    continue

                trades.append({
                    'symbol': sym, 'score': score, 'entry': round(entry, 6),
                    'sl': round(sl, 6), 'tp': round(tp, 6),
                    'exit': round(exit_px, 6), 'result': result,
                    'pnl_pct': round(pnl_pct, 2), 'rsi': round(rsi, 0),
                    'adx': round(adx, 0), 'adx_1h': round(adx_1h, 0),
                    'candle_idx': i, 'hour': hour,
                })
                sym_trades += 1
                cooldown_until = exit_idx + 8

            if verbose:
                print(f"  {sym:14s} {sym_trades}건")
        except Exception as e:
            if verbose:
                print(f"  {sym:14s} 에러: {e}")
            continue

    return trades


def print_result(trades, label=""):
    """결과 출력"""
    if not trades:
        print(f"{label} 거래 없음")
        return

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    total_pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
    wr = len(wins) / len(trades) * 100

    print(f"\n{'='*70}")
    print(f"  {label or '백테스트 결과'}")
    print(f"{'='*70}")
    print(f"총 거래: {len(trades)}건 | 승 {len(wins)} 패 {len(losses)} | 승률 {wr:.0f}%")
    print(f"총 PnL: {total_pnl:+.2f}% | 평균 수익: {avg_win:+.2f}% | 평균 손실: {avg_loss:+.2f}%")
    if avg_loss:
        print(f"손익비: {abs(avg_win/avg_loss):.2f}:1")
        gw = sum(t['pnl_pct'] for t in wins)
        gl = abs(sum(t['pnl_pct'] for t in losses))
        if gl > 0: print(f"Profit Factor: {gw/gl:.2f}")

    # 청산 방식별
    print(f"\n청산 방식:")
    for r in ['sl', 'tp', 'trail', 'timeout']:
        sub = [t for t in trades if t['result'] == r]
        if sub:
            w = sum(1 for t in sub if t['pnl_pct'] > 0)
            p = sum(t['pnl_pct'] for t in sub)
            print(f"  {r:8s}: {len(sub):3d}건 승률{w/len(sub)*100:4.0f}% PnL {p:+.2f}%")

    # 종목별 상위 5
    print(f"\n종목별 (상위5):")
    by_sym = {}
    for t in trades:
        by_sym.setdefault(t['symbol'], []).append(t)
    for s, ts in sorted(by_sym.items(), key=lambda x: sum(t['pnl_pct'] for t in x[1]), reverse=True)[:5]:
        w = len([t for t in ts if t['pnl_pct'] > 0])
        pnl = sum(t['pnl_pct'] for t in ts)
        print(f"  {s:14s} {len(ts):2d}건 승률{w/len(ts)*100:3.0f}% PnL {pnl:+6.2f}%")

    # 점수별
    print(f"\n점수별:")
    for lo, hi, label in [(5, 8, 'sc5-7'), (8, 11, 'sc8-10'), (11, 99, 'sc11+')]:
        sub = [t for t in trades if lo <= t['score'] < hi]
        if sub:
            w = sum(1 for t in sub if t['pnl_pct'] > 0)
            p = sum(t['pnl_pct'] for t in sub)
            print(f"  {label:8s}: {len(sub):3d}건 승률{w/len(sub)*100:4.0f}% PnL {p:+.2f}%")


def run_grid_search(symbols):
    """243조합 그리드서치"""
    grid = {
        'min_score': [5, 6, 7],
        'sl_mult': [1.0, 1.5, 2.0],
        'tp_mult': [2.5, 3.5, 5.0],
        'trail_start': [0.5, 0.7, 1.0],
        'noon_bonus': [0, 2, 3],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n{'='*70}")
    print(f"  그리드서치: {len(combos)}조합")
    print(f"{'='*70}")

    # 데이터 한 번만 로드 (14일 = 1000캔들)
    print("14일 데이터 로드 중...")
    data_cache = {}
    btc_data = None
    for sym in symbols:
        if sym in BLACKLIST or sym == 'BTCUSDT':
            continue
        try:
            df15 = get_klines(sym, '15m', 1000)
            df1h = get_klines(sym, '1h', 500)
            df4h = get_klines(sym, '4h', 200)
            if len(df15) < 100 or len(df1h) < 60 or len(df4h) < 30:
                continue
            data_cache[sym] = (df15, df1h, df4h)
            time.sleep(0.3)
        except:
            continue
        if btc_data is None:
            try:
                btc_data = get_klines('BTCUSDT', '1h', 500)
            except:
                pass
    print(f"  {len(data_cache)}종목 로드 완료\n")

    # 멀티프로세싱 (CPU 코어 10개, 서버 50% 이하)
    n_workers = min(10, mp.cpu_count() // 2, len(combos))
    print(f"  {n_workers}코어 병렬 처리...")

    args_list = [(dict(zip(keys, combo)), data_cache, btc_data) for combo in combos]

    with mp.Pool(n_workers) as pool:
        raw_results = pool.map(_grid_worker, args_list)

    results = [r for r in raw_results if r is not None]
    print(f"  {len(results)}/{len(combos)} 유효 조합")

    # 상위 10
    results.sort(key=lambda x: x['pnl'], reverse=True)
    print(f"\n{'='*70}")
    print(f"  상위 10 조합 (PnL 기준)")
    print(f"{'='*70}")
    print(f"{'#':>3} {'sc':>3} {'SL':>4} {'TP':>4} {'TR':>4} {'noon':>4} {'건수':>4} {'승률':>5} {'PnL%':>7} {'avg_W':>6} {'avg_L':>6}")
    print("-" * 60)
    for i, r in enumerate(results[:10], 1):
        p = r['params']
        print(f"{i:3d} {p['min_score']:3d} {p['sl_mult']:4.1f} {p['tp_mult']:4.1f} {p['trail_start']:4.1f} {p['noon_bonus']:4d} "
              f"{r['total']:4d} {r['wr']:4.0f}% {r['pnl']:+7.2f} {r['avg_win']:+6.2f} {r['avg_loss']:+6.2f}")

    # 현재 설정과 비교
    cur = next((r for r in results if r['params'] == DEFAULTS), None)
    if cur:
        rank = results.index(cur) + 1
        print(f"\n현재 설정 순위: {rank}/{len(results)} | PnL {cur['pnl']:+.2f}% 승률 {cur['wr']:.0f}%")

    # 결과 저장
    rf = RESULT_DIR / f"grid_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    save_results = [{k: v for k, v in r.items() if k != 'trades'} for r in results[:20]]
    with open(rf, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n결과 저장: {rf}")

    return results


def _grid_worker(args):
    """그리드서치 워커 (멀티프로세싱용)"""
    params, data_cache, btc_data = args
    trades = _run_bt_cached(data_cache, btc_data, params)
    if not trades:
        return None
    wins = [t for t in trades if t['pnl_pct'] > 0]
    total_pnl = sum(t['pnl_pct'] for t in trades)
    wr = len(wins) / len(trades) * 100
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss_list = [t for t in trades if t['pnl_pct'] <= 0]
    avg_loss = sum(t['pnl_pct'] for t in avg_loss_list) / len(avg_loss_list) if avg_loss_list else 0
    return {
        'params': params, 'total': len(trades), 'wins': len(wins),
        'wr': round(wr, 1), 'pnl': round(total_pnl, 2),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
    }


def _run_bt_cached(data_cache, btc_data, params):
    """캐시된 데이터로 백테스트 (API 호출 없음)"""
    trades = []

    for sym, (df15, df1h, df4h) in data_cache.items():
        cooldown_until = 0

        for i in range(60, len(df15) - 20, 4):
            if i < cooldown_until:
                continue

            try:
                i15 = calc_indicators(df15.iloc[max(0, i-60):i].reset_index(drop=True))
                i1h_end = min(i // 4 + 1, len(df1h))
                i1h_start = max(0, i1h_end - 60)
                if i1h_end - i1h_start < 20: continue
                ind_1h = calc_indicators(df1h.iloc[i1h_start:i1h_end].reset_index(drop=True))

                i4h_end = min(i // 16 + 1, len(df4h))
                i4h_start = max(0, i4h_end - 60)
                if i4h_end - i4h_start < 15: continue
                ind_4h = calc_indicators(df4h.iloc[i4h_start:i4h_end].reset_index(drop=True))

                btc_end = min(i // 4 + 1, len(btc_data))
                btc_start = max(0, btc_end - 60)
                if btc_end - btc_start < 20: continue
                btc_ind = calc_indicators(btc_data.iloc[btc_start:btc_end].reset_index(drop=True))
                btc_up = (btc_ind.get('ema20', 0) or 0) > (btc_ind.get('ema50', 0) or 0)
            except:
                continue

            try:
                hour = df15.iloc[i-1]['time'].hour if hasattr(df15.iloc[i-1].get('time', 0), 'hour') else 12
            except:
                hour = 12

            score, direction, rsi, adx, adx_1h = score_at(i15, ind_1h, ind_4h, btc_up, hour, params)
            if direction == 'wait':
                continue

            entry = float(df15.iloc[i-1]['close'])
            atr = ind_1h.get('atr', 0) or 0
            sl, tp = calc_sl_tp(entry, atr, params)

            result, exit_px, exit_idx, pnl_pct = simulate_trade(df15, i, entry, sl, tp, params)
            pnl_pct -= FEE_PCT

            if abs(exit_px - entry) / entry < 0.001:
                cooldown_until = exit_idx + 8
                continue

            trades.append({
                'symbol': sym, 'score': score, 'entry': round(entry, 6),
                'sl': round(sl, 6), 'tp': round(tp, 6),
                'exit': round(exit_px, 6), 'result': result,
                'pnl_pct': round(pnl_pct, 2), 'rsi': round(rsi, 0),
                'adx': round(adx, 0), 'adx_1h': round(adx_1h, 0),
                'candle_idx': i, 'hour': hour,
            })
            cooldown_until = exit_idx + 8

    return trades


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['backtest', 'grid'], default='backtest')
    args = parser.parse_args()

    syms = get_alt_futures_symbols(25)
    print(f"대상: {len(syms)}종목 (BLACKLIST 제외)")

    if args.mode == 'backtest':
        print("\n--- 현재 파라미터로 백테스트 ---")
        print(f"설정: {DEFAULTS}")
        trades = run_backtest(syms, DEFAULTS, verbose=True)
        print_result(trades, "현재 파라미터 백테스트")

        rf = RESULT_DIR / f"bt_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(rf, 'w') as f:
            json.dump({'config': DEFAULTS, 'trades': trades}, f, indent=2)
        print(f"\n결과 저장: {rf}")

    elif args.mode == 'grid':
        results = run_grid_search(syms)
        if results:
            # 텔레그램 결과 알림
            try:
                from telegram_notifier import send_message
                from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
                top = results[0]
                cur = next((r for r in results if r['params'] == DEFAULTS), None)
                cur_rank = results.index(cur) + 1 if cur else '?'
                msg = (
                    f"<b>📊 주간 그리드서치 결과</b>\n\n"
                    f"<b>1위:</b> PnL {top['pnl']:+.2f}% 승률{top['wr']:.0f}% ({top['total']}건)\n"
                    f"  sc={top['params']['min_score']} SL={top['params']['sl_mult']} "
                    f"TP={top['params']['tp_mult']} TR={top['params']['trail_start']}\n\n"
                    f"<b>현재:</b> {cur_rank}위/{len(results)} | PnL {cur['pnl']:+.2f}%\n" if cur else ""
                    f"\n상위 5:\n"
                )
                for i, r in enumerate(results[:5], 1):
                    p = r['params']
                    msg += f"{i}. sc{p['min_score']} SL{p['sl_mult']} TP{p['tp_mult']} TR{p['trail_start']} → {r['pnl']:+.1f}%\n"
                send_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
                print("텔레그램 알림 전송 완료")
            except Exception as e:
                print(f"텔레그램 실패: {e}")

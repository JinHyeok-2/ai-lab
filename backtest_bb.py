#!/usr/bin/env python3
# BB 박스 전략 백테스트 + 그리드서치
# 과거 7일 데이터로 BB 롱/숏 파라미터 최적화
import sys; sys.stdout.reconfigure(line_buffering=True)
import os; os.chdir('/home/hyeok/01.APCC/00.ai-lab')

import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
from binance_client import get_client
from indicators import calc_indicators

# ── 설정 ──
SYMBOLS = [
    'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'WLDUSDT',
    'HYPEUSDT', 'ZECUSDT', 'TAOUSDT', 'ONUSDT', 'ONTUSDT',
    'PLAYUSDT', 'STOUSDT',
]
DAYS = 7
USDT = 25
LEV = 2
FEE = 0.0008  # 0.04% × 2 (진입+청산)

# ── 그리드서치 파라미터 ──
GRID = {
    'bb_width_min': [1.0, 1.5],
    'bb_width_max': [4.0, 5.0, 5.5],
    'rsi_long_max': [45, 50, 55],
    'rsi_short_min': [50, 55],
    'rsi_short_max': [70, 75],
    'tp_mode': ['mid', 'upper70', 'upper'],  # mid=중간선, upper70=mid+상단30%, upper=상단
    'sl_pct': [0.3, 0.5, 0.8],  # BB 하단/상단 대비 %
    'mtf_min': [2, 3],
    'max_hold_h': [4, 6, 8],
}


def download_klines(sym, interval='1h', days=7):
    """바이낸스에서 과거 캔들 다운로드"""
    client = get_client()
    limit = days * 24 if interval == '1h' else days * 24 * 4
    klines = client.futures_klines(symbol=sym, interval=interval, limit=min(limit, 1000))
    df = pd.DataFrame(klines, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df


def calc_bb(df, period=20, std=2):
    """볼린저 밴드 계산"""
    mid = df['close'].rolling(period).mean()
    std_val = df['close'].rolling(period).std()
    upper = mid + std * std_val
    lower = mid - std * std_val
    return lower, mid, upper


def calc_rsi(df, period=14):
    """RSI 계산"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def simulate_bb(sym, df_1h, df_15m, df_30m, params):
    """단일 파라미터 조합으로 BB 롱+숏 시뮬레이션"""
    trades = []

    # 1h BB + RSI 계산
    bb_lower, bb_mid, bb_upper = calc_bb(df_1h)
    rsi_1h = calc_rsi(df_1h)
    bb_width = (bb_upper - bb_lower) / bb_mid * 100

    # 15m, 30m BB (MTF용)
    bb_lower_15m, _, bb_upper_15m = calc_bb(df_15m)
    bb_lower_30m, _, bb_upper_30m = calc_bb(df_30m)

    # 15m/30m → 1h 시간 매핑 (각 1h 캔들 시점의 15m/30m 위치)
    for i in range(25, len(df_1h)):
        px = df_1h['close'].iloc[i]
        low_1h = bb_lower.iloc[i]
        mid_1h = bb_mid.iloc[i]
        up_1h = bb_upper.iloc[i]
        w = bb_width.iloc[i]
        rsi = rsi_1h.iloc[i]

        if pd.isna(w) or pd.isna(rsi) or pd.isna(low_1h):
            continue
        if not (up_1h > low_1h > 0):
            continue

        # BB 폭 조건
        if not (params['bb_width_min'] < w < params['bb_width_max']):
            continue

        bb_pos = (px - low_1h) / (up_1h - low_1h) * 100

        # MTF 하단/상단 합의 (1h 타임스탬프에 해당하는 15m/30m 위치)
        ts_1h = df_1h['time'].iloc[i]
        # 15m: 해당 시간 근처 마지막 캔들
        idx_15m = df_15m['time'].searchsorted(ts_1h, side='right') - 1
        idx_30m = df_30m['time'].searchsorted(ts_1h, side='right') - 1
        if idx_15m < 20 or idx_30m < 20:
            continue

        px_15m = df_15m['close'].iloc[idx_15m]
        px_30m = df_30m['close'].iloc[idx_30m]

        # 하단 합의 (롱)
        bottom_count = 0
        if not pd.isna(bb_lower_15m.iloc[idx_15m]) and not pd.isna(bb_upper_15m.iloc[idx_15m]):
            bw_15 = bb_upper_15m.iloc[idx_15m] - bb_lower_15m.iloc[idx_15m]
            if bw_15 > 0:
                pos_15 = (px_15m - bb_lower_15m.iloc[idx_15m]) / bw_15 * 100
                if -5 < pos_15 < 20:
                    bottom_count += 1
        if not pd.isna(bb_lower_30m.iloc[idx_30m]) and not pd.isna(bb_upper_30m.iloc[idx_30m]):
            bw_30 = bb_upper_30m.iloc[idx_30m] - bb_lower_30m.iloc[idx_30m]
            if bw_30 > 0:
                pos_30 = (px_30m - bb_lower_30m.iloc[idx_30m]) / bw_30 * 100
                if -5 < pos_30 < 20:
                    bottom_count += 1
        if -5 < bb_pos < 20:
            bottom_count += 1

        # 상단 합의 (숏)
        top_count = 0
        if not pd.isna(bb_lower_15m.iloc[idx_15m]) and not pd.isna(bb_upper_15m.iloc[idx_15m]):
            bw_15 = bb_upper_15m.iloc[idx_15m] - bb_lower_15m.iloc[idx_15m]
            if bw_15 > 0:
                pos_15 = (px_15m - bb_lower_15m.iloc[idx_15m]) / bw_15 * 100
                if 80 < pos_15 < 105:
                    top_count += 1
        if not pd.isna(bb_lower_30m.iloc[idx_30m]) and not pd.isna(bb_upper_30m.iloc[idx_30m]):
            bw_30 = bb_upper_30m.iloc[idx_30m] - bb_lower_30m.iloc[idx_30m]
            if bw_30 > 0:
                pos_30 = (px_30m - bb_lower_30m.iloc[idx_30m]) / bw_30 * 100
                if 80 < pos_30 < 105:
                    top_count += 1
        if 80 < bb_pos < 105:
            top_count += 1

        # TP 위치 계산
        if params['tp_mode'] == 'mid':
            tp_long = mid_1h
            tp_short = mid_1h
        elif params['tp_mode'] == 'upper70':
            tp_long = mid_1h + (up_1h - mid_1h) * 0.7
            tp_short = mid_1h - (mid_1h - low_1h) * 0.7
        else:  # upper
            tp_long = up_1h
            tp_short = low_1h

        # === BB 롱 시그널 ===
        if (bottom_count >= params['mtf_min'] and
                30 < rsi < params['rsi_long_max'] and
                bb_pos < 20):
            entry = px
            sl = low_1h * (1 - params['sl_pct'] / 100)
            tp = tp_long
            if tp <= entry * 1.005:
                continue
            # 시뮬레이션: 이후 캔들에서 SL/TP 히트 체크
            trade = _simulate_trade(df_1h, i, entry, sl, tp, 'long', params['max_hold_h'])
            if trade:
                trade['symbol'] = sym
                trades.append(trade)

        # === BB 숏 시그널 ===
        if (top_count >= params['mtf_min'] and
                params['rsi_short_min'] < rsi < params['rsi_short_max'] and
                bb_pos > 80):
            entry = px
            sl = up_1h * (1 + params['sl_pct'] / 100)
            tp = tp_short
            if tp >= entry * 0.995:
                continue
            trade = _simulate_trade(df_1h, i, entry, sl, tp, 'short', params['max_hold_h'])
            if trade:
                trade['symbol'] = sym
                trades.append(trade)

    return trades


def _simulate_trade(df, entry_idx, entry, sl, tp, direction, max_hold_h):
    """진입 후 SL/TP/시간초과 시뮬레이션"""
    for j in range(entry_idx + 1, min(entry_idx + max_hold_h + 1, len(df))):
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]

        if direction == 'long':
            if low <= sl:
                pnl_pct = (sl - entry) / entry * 100
                return {'dir': 'long', 'entry': entry, 'exit': sl, 'pnl_pct': pnl_pct, 'reason': 'SL', 'hold_h': j - entry_idx}
            if high >= tp:
                pnl_pct = (tp - entry) / entry * 100
                return {'dir': 'long', 'entry': entry, 'exit': tp, 'pnl_pct': pnl_pct, 'reason': 'TP', 'hold_h': j - entry_idx}
        else:  # short
            if high >= sl:
                pnl_pct = (entry - sl) / entry * 100
                return {'dir': 'short', 'entry': entry, 'exit': sl, 'pnl_pct': pnl_pct, 'reason': 'SL', 'hold_h': j - entry_idx}
            if low <= tp:
                pnl_pct = (entry - tp) / entry * 100
                return {'dir': 'short', 'entry': entry, 'exit': tp, 'pnl_pct': pnl_pct, 'reason': 'TP', 'hold_h': j - entry_idx}

    # 시간초과 — 마지막 캔들 종가로 청산
    last_idx = min(entry_idx + max_hold_h, len(df) - 1)
    exit_px = df['close'].iloc[last_idx]
    if direction == 'long':
        pnl_pct = (exit_px - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_px) / entry * 100
    return {'dir': direction, 'entry': entry, 'exit': exit_px, 'pnl_pct': pnl_pct, 'reason': 'TIMEOUT', 'hold_h': last_idx - entry_idx}


def main():
    print(f"BB 백테스트 시작 — {len(SYMBOLS)}종목, {DAYS}일, 레버리지 {LEV}x")
    print(f"그리드: {len(list(product(*GRID.values())))}개 조합")
    print()

    # 1. 데이터 다운로드
    print("📥 데이터 다운로드 중...")
    data = {}
    for sym in SYMBOLS:
        try:
            data[sym] = {
                '1h': download_klines(sym, '1h', DAYS),
                '15m': download_klines(sym, '15m', DAYS),
                '30m': download_klines(sym, '30m', DAYS),
            }
            print(f"  {sym}: 1h={len(data[sym]['1h'])}캔들")
        except Exception as e:
            print(f"  {sym}: 오류 {e}")
    print()

    # 2. 그리드서치
    print("🔍 그리드서치 시작...")
    keys = list(GRID.keys())
    values = list(GRID.values())
    results = []

    for combo_idx, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))

        all_trades = []
        for sym, sym_data in data.items():
            trades = simulate_bb(sym, sym_data['1h'], sym_data['15m'], sym_data['30m'], params)
            all_trades.extend(trades)

        if not all_trades:
            continue

        # 성과 계산
        n = len(all_trades)
        wins = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        # PnL: 수익률 × 노셔널 - 수수료
        notional = USDT * LEV
        total_pnl = sum(t['pnl_pct'] / 100 * notional - notional * FEE for t in all_trades)
        avg_pnl = total_pnl / n
        win_rate = wins / n * 100
        tp_hits = sum(1 for t in all_trades if t['reason'] == 'TP')
        sl_hits = sum(1 for t in all_trades if t['reason'] == 'SL')
        timeouts = sum(1 for t in all_trades if t['reason'] == 'TIMEOUT')
        avg_hold = np.mean([t['hold_h'] for t in all_trades])

        # 롱/숏 분리
        longs = [t for t in all_trades if t['dir'] == 'long']
        shorts = [t for t in all_trades if t['dir'] == 'short']
        long_pnl = sum(t['pnl_pct'] / 100 * notional - notional * FEE for t in longs)
        short_pnl = sum(t['pnl_pct'] / 100 * notional - notional * FEE for t in shorts)

        results.append({
            'params': params,
            'trades': n,
            'wins': wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'timeouts': timeouts,
            'avg_hold': avg_hold,
            'longs': len(longs),
            'shorts': len(shorts),
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
        })

        if (combo_idx + 1) % 50 == 0:
            print(f"  {combo_idx + 1}/{len(list(product(*values)))} 완료...")

    # 3. 결과 정렬 (PnL 기준)
    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"\n{'='*90}")
    print(f"총 {len(results)}개 유효 조합 (거래 있는 것만)")
    print(f"{'='*90}")

    # 상위 10개
    print(f"\n🏆 상위 10개 조합")
    print(f"{'순위':>4} {'건수':>4} {'승률':>5} {'PnL':>8} {'TP':>3} {'SL':>3} {'TO':>3} {'롱':>3} {'숏':>3} {'보유h':>5} | 파라미터")
    print("-" * 90)
    for rank, r in enumerate(results[:10], 1):
        p = r['params']
        param_str = f"w={p['bb_width_min']}-{p['bb_width_max']} rsi_l<{p['rsi_long_max']} rsi_s={p['rsi_short_min']}-{p['rsi_short_max']} tp={p['tp_mode']} sl={p['sl_pct']}% mtf≥{p['mtf_min']} hold≤{p['max_hold_h']}h"
        print(f"#{rank:>3} {r['trades']:>4} {r['win_rate']:>4.0f}% ${r['total_pnl']:>7.2f} {r['tp_hits']:>3} {r['sl_hits']:>3} {r['timeouts']:>3} {r['longs']:>3} {r['shorts']:>3} {r['avg_hold']:>4.1f}h | {param_str}")

    # 하위 5개 (회피할 조합)
    print(f"\n❌ 하위 5개 조합")
    for rank, r in enumerate(results[-5:], len(results) - 4):
        p = r['params']
        param_str = f"w={p['bb_width_min']}-{p['bb_width_max']} rsi_l<{p['rsi_long_max']} rsi_s={p['rsi_short_min']}-{p['rsi_short_max']} tp={p['tp_mode']} sl={p['sl_pct']}%"
        print(f"#{rank:>3} {r['trades']:>4} {r['win_rate']:>4.0f}% ${r['total_pnl']:>7.2f} | {param_str}")

    # 현재 설정과 비교
    current = {
        'bb_width_min': 1.0, 'bb_width_max': 5.5,
        'rsi_long_max': 50, 'rsi_short_min': 50, 'rsi_short_max': 70,
        'tp_mode': 'mid', 'sl_pct': 0.5, 'mtf_min': 2, 'max_hold_h': 6,
    }
    current_result = None
    for r in results:
        if r['params'] == current:
            current_result = r
            break
    if current_result:
        cur_rank = results.index(current_result) + 1
        print(f"\n📊 현재 설정: #{cur_rank}/{len(results)} — {current_result['trades']}건 승률{current_result['win_rate']:.0f}% PnL=${current_result['total_pnl']:.2f}")
    else:
        print(f"\n📊 현재 설정: 결과에 정확히 매칭되는 조합 없음")

    # 1위 vs 현재 비교
    if results:
        best = results[0]
        print(f"\n🥇 1위 추천 파라미터:")
        for k, v in best['params'].items():
            cur_v = current.get(k, '?')
            changed = ' ← 변경' if v != cur_v else ''
            print(f"  {k}: {cur_v} → {v}{changed}")
        print(f"  예상: {best['trades']}건 승률{best['win_rate']:.0f}% PnL=${best['total_pnl']:.2f} (7일)")

    # 파라미터별 최적값 분석
    print(f"\n📈 파라미터별 최적값 (상위 20개 평균)")
    top20 = results[:20]
    for key in GRID.keys():
        val_stats = {}
        for r in top20:
            v = r['params'][key]
            if v not in val_stats:
                val_stats[v] = []
            val_stats[v].append(r['total_pnl'])
        print(f"  {key}:")
        for v, pnls in sorted(val_stats.items(), key=lambda x: -np.mean(x[1])):
            print(f"    {v}: 평균 PnL=${np.mean(pnls):.2f} ({len(pnls)}건)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# 고급 백테스트: 레짐 감지 + Monte Carlo + Walk-Forward
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from binance_client import get_klines
from indicators import calc_indicators
from alt_scanner import get_alt_futures_symbols

FEE_PCT = 0.07
MAX_SL_PCT = 5.0
MIN_TP_PCT = 0.3
BLACKLIST = {'BRUSDT', 'SIRENUSDT', 'XAUUSDT', 'XAGUSDT', 'RIVERUSDT'}
RESULT_DIR = Path('/home/hyeok/01.APCC/00.ai-lab/backtest_results')
RESULT_DIR.mkdir(exist_ok=True)

# 그리드서치 최적 파라미터
BEST = {'min_score': 3, 'sl_mult': 1.5, 'tp_mult': 5.0}


# ═══════════════════════════════════════════════════════════
# 공통 함수
# ═══════════════════════════════════════════════════════════

def load_data(symbols):
    """데이터 1회 로드"""
    data = {}
    data['BTCUSDT_1h'] = get_klines('BTCUSDT', '1h', 500)
    print(f"BTC 1h 로드")
    for sym in symbols:
        if sym in BLACKLIST or sym == 'BTCUSDT':
            continue
        try:
            df15 = get_klines(sym, '15m', 500)
            df1h = get_klines(sym, '1h', 500)
            df4h = get_klines(sym, '4h', 200)
            recent = df15.tail(100)
            rng = (recent['high'].max() - recent['low'].min()) / recent['close'].mean() * 100
            if rng < 1.0:
                continue
            if len(df15) < 100 or len(df1h) < 60 or len(df4h) < 30:
                continue
            data[sym] = {'15m': df15, '1h': df1h, '4h': df4h}
            print(f"  {sym} OK (변동 {rng:.1f}%)")
            time.sleep(0.3)
        except:
            continue
    return data


def score_at(i15, i1h, i4h, btc_up):
    """스코어링"""
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


def detect_regime(i1h):
    """시장 레짐 감지: trend(추세) / range(횡보) / volatile(급변)"""
    adx = i1h.get('adx', 0) or 0
    atr = i1h.get('atr', 0) or 0
    price = i1h.get('price', 0) or i1h.get('close', 0) or 0
    bb_upper = i1h.get('bb_upper', 0) or 0
    bb_lower = i1h.get('bb_lower', 0) or 0

    # BB 폭으로 변동성 판단
    bb_width = (bb_upper - bb_lower) / price * 100 if price > 0 else 0

    if adx >= 30:
        return 'trend'      # 강한 추세
    elif adx < 20 and bb_width < 3:
        return 'range'      # 횡보 (좁은 BB + 낮은 ADX)
    elif bb_width > 8:
        return 'volatile'   # 급변동
    else:
        return 'trend'      # 약한 추세도 추세로


def precompute_all(data):
    """모든 시점의 점수 + 레짐 미리 계산"""
    signals = {}
    btc_1h = data['BTCUSDT_1h']

    for sym, d in data.items():
        if sym == 'BTCUSDT_1h':
            continue
        df15, df1h, df4h = d['15m'], d['1h'], d['4h']
        sigs = []
        for i in range(60, len(df15) - 20, 4):
            try:
                i15 = calc_indicators(df15.iloc[max(0,i-60):i].reset_index(drop=True))
                i1h_end = min(i//4+1, len(df1h))
                i1h_start = max(0, i1h_end-60)
                if i1h_end - i1h_start < 20: continue
                i1h_data = calc_indicators(df1h.iloc[i1h_start:i1h_end].reset_index(drop=True))
                i4h_end = min(i//16+1, len(df4h))
                i4h_start = max(0, i4h_end-60)
                if i4h_end - i4h_start < 15: continue
                i4h_data = calc_indicators(df4h.iloc[i4h_start:i4h_end].reset_index(drop=True))
                btc_end = min(i//4+1, len(btc_1h))
                btc_start = max(0, btc_end-60)
                if btc_end - btc_start < 20: continue
                btc_ind = calc_indicators(btc_1h.iloc[btc_start:btc_end].reset_index(drop=True))
                btc_up = (btc_ind.get('ema20',0) or 0) > (btc_ind.get('ema50',0) or 0)

                sc, rsi, adx = score_at(i15, i1h_data, i4h_data, btc_up)
                entry = float(df15.iloc[i-1]['close'])
                atr = i1h_data.get('atr', 0) or 0
                regime = detect_regime(i1h_data)

                sigs.append({
                    'idx': i, 'score': sc, 'entry': entry, 'atr': atr,
                    'rsi': rsi, 'adx': adx, 'regime': regime,
                })
            except:
                continue
        signals[sym] = sigs
    return signals


def simulate_one(df15, entry, sl, tp, direction, start_idx, max_candles=40):
    """단일 거래 시뮬레이션"""
    exit_idx = start_idx + max_candles
    for j in range(start_idx, min(start_idx + max_candles, len(df15))):
        h = float(df15.iloc[j]['high'])
        l = float(df15.iloc[j]['low'])
        if direction == 'long':
            if l <= sl: return 'sl', sl, j
            if h >= tp: return 'tp', tp, j
        else:
            if h >= sl: return 'sl', sl, j
            if l <= tp: return 'tp', tp, j
    ep = float(df15.iloc[min(start_idx+max_candles-1, len(df15)-1)]['close'])
    return 'open', ep, exit_idx


def run_strategy(signals, data, min_score=3, sl_mult=1.5, tp_mult=5.0, use_regime=False):
    """전략 시뮬레이션 (레짐 옵션 포함)"""
    trades = []
    for sym, sigs in signals.items():
        df15 = data[sym]['15m']
        cooldown = 0
        for sig in sigs:
            i = sig['idx']
            if i < cooldown: continue
            sc = sig['score']

            # 방향
            if sc >= min_score: direction = 'long'
            elif sc <= -min_score: direction = 'short'
            else: continue

            # 레짐 필터
            if use_regime:
                regime = sig['regime']
                if regime == 'range':
                    continue  # 횡보장 진입 안 함
                if regime == 'volatile':
                    # 급변동: SL 넓히고 TP도 넓힘
                    sl_m = sl_mult * 1.5
                    tp_m = tp_mult * 1.5
                else:
                    sl_m = sl_mult
                    tp_m = tp_mult
            else:
                sl_m = sl_mult
                tp_m = tp_mult

            entry = sig['entry']
            atr = sig['atr']
            if atr <= 0: atr = entry * 0.02
            if entry > 0 and atr / entry > 0.15: continue

            # SL/TP 계산
            if direction == 'long':
                sl = entry - atr * sl_m
                tp = entry + atr * tp_m
            else:
                sl = entry + atr * sl_m
                tp = entry - atr * tp_m

            sl_dist = abs(entry - sl) / entry * 100
            if sl_dist > MAX_SL_PCT:
                sl = entry * (1 - MAX_SL_PCT/100) if direction == 'long' else entry * (1 + MAX_SL_PCT/100)
            risk = abs(entry - sl)
            if risk > 0 and abs(tp - entry) / risk < 2.0:
                tp = entry + risk * 2.0 if direction == 'long' else entry - risk * 2.0
            min_tp = entry * MIN_TP_PCT / 100
            if abs(tp - entry) < min_tp:
                tp = entry + min_tp if direction == 'long' else entry - min_tp

            result, exit_price, exit_idx = simulate_one(df15, entry, sl, tp, direction, i)

            if abs(exit_price - entry) / entry < 0.001:
                cooldown = exit_idx + 8
                continue

            pnl_pct = ((exit_price - entry) / entry * 100 if direction == 'long'
                       else (entry - exit_price) / entry * 100) - FEE_PCT

            trades.append({
                'symbol': sym, 'direction': direction, 'score': sc,
                'entry': entry, 'exit': exit_price, 'result': result,
                'pnl_pct': round(pnl_pct, 2), 'regime': sig.get('regime', ''),
                'idx': i,
            })
            cooldown = exit_idx + 8
    return trades


def summarize(trades, label=""):
    """거래 결과 요약"""
    if not trades:
        print(f"  {label}: 거래 없음")
        return 0, 0, 0
    wins = len([t for t in trades if t['pnl_pct'] > 0])
    pnl = sum(t['pnl_pct'] for t in trades)
    wr = wins / len(trades) * 100
    avg = pnl / len(trades)
    print(f"  {label}: {len(trades)}건 승률 {wr:.0f}% PnL {pnl:+.2f}% (건당 {avg:+.2f}%)")
    return len(trades), wr, pnl


# ═══════════════════════════════════════════════════════════
# 1. 레짐 감지 백테스트
# ═══════════════════════════════════════════════════════════

def test_regime(signals, data):
    print(f"\n{'='*60}")
    print("1. 레짐 감지 (Regime Detection) 백테스트")
    print(f"{'='*60}")

    # 레짐 분포 확인
    all_regimes = []
    for sigs in signals.values():
        for s in sigs:
            all_regimes.append(s['regime'])
    from collections import Counter
    rc = Counter(all_regimes)
    print(f"\n레짐 분포: {dict(rc)}")

    # A. 레짐 없이 (기존)
    trades_no = run_strategy(signals, data, **BEST, use_regime=False)
    summarize(trades_no, "레짐 OFF (기존)")

    # B. 레짐 적용
    trades_yes = run_strategy(signals, data, **BEST, use_regime=True)
    summarize(trades_yes, "레짐 ON (횡보 스킵)")

    # 레짐별 상세
    print(f"\n  레짐별 성과 (레짐 OFF 기준):")
    for regime in ['trend', 'range', 'volatile']:
        grp = [t for t in trades_no if t['regime'] == regime]
        if grp:
            w = len([t for t in grp if t['pnl_pct'] > 0])
            p = sum(t['pnl_pct'] for t in grp)
            print(f"    {regime:10s}: {len(grp):2d}건 승률 {w/len(grp)*100:3.0f}% PnL {p:+.2f}%")

    return trades_no, trades_yes


# ═══════════════════════════════════════════════════════════
# 2. Monte Carlo 시뮬레이션
# ═══════════════════════════════════════════════════════════

def test_montecarlo(trades, n_sim=1000):
    print(f"\n{'='*60}")
    print(f"2. Monte Carlo 시뮬레이션 ({n_sim}회)")
    print(f"{'='*60}")

    if not trades:
        print("  거래 없음")
        return

    pnls = [t['pnl_pct'] for t in trades]
    original_total = sum(pnls)
    print(f"\n  원본: {len(pnls)}건, 총 PnL {original_total:+.2f}%")

    sim_totals = []
    sim_max_dd = []
    for _ in range(n_sim):
        shuffled = random.sample(pnls, len(pnls))
        total = sum(shuffled)
        sim_totals.append(total)

        # 최대 드로다운 계산
        cumsum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cumsum)
        dd = peak - cumsum
        sim_max_dd.append(np.max(dd) if len(dd) > 0 else 0)

    sim_totals = np.array(sim_totals)
    sim_max_dd = np.array(sim_max_dd)

    # 통계
    print(f"\n  Monte Carlo 결과 ({n_sim}회 시뮬레이션):")
    print(f"    PnL 평균: {np.mean(sim_totals):+.2f}%")
    print(f"    PnL 중앙값: {np.median(sim_totals):+.2f}%")
    print(f"    PnL 5th pct: {np.percentile(sim_totals, 5):+.2f}% (최악)")
    print(f"    PnL 95th pct: {np.percentile(sim_totals, 95):+.2f}% (최고)")
    print(f"    PnL 표준편차: {np.std(sim_totals):.2f}%")
    print(f"    수익 확률: {(sim_totals > 0).sum() / n_sim * 100:.1f}%")
    print(f"    최대 드로다운 평균: {np.mean(sim_max_dd):.2f}%")
    print(f"    최대 드로다운 95th: {np.percentile(sim_max_dd, 95):.2f}%")

    # 신뢰도 판단
    profit_prob = (sim_totals > 0).sum() / n_sim * 100
    if profit_prob >= 70:
        verdict = "전략 유효 (70%+ 수익 확률)"
    elif profit_prob >= 50:
        verdict = "약간 유효 (50-70% 수익 확률)"
    else:
        verdict = "전략 불안정 (50% 미만 수익 확률)"
    print(f"\n    판정: {verdict}")

    return sim_totals


# ═══════════════════════════════════════════════════════════
# 3. Walk-Forward 최적화
# ═══════════════════════════════════════════════════════════

def test_walkforward(signals, data):
    print(f"\n{'='*60}")
    print("3. Walk-Forward 최적화")
    print(f"{'='*60}")

    # 시간순 정렬된 전체 거래 (MIN_SCORE=1로 모든 시그널 포함)
    # 각 윈도우에서 최적 파라미터를 찾고, 다음 윈도우에서 테스트

    # 캔들 인덱스 범위 파악
    all_indices = []
    for sigs in signals.values():
        for s in sigs:
            all_indices.append(s['idx'])
    if not all_indices:
        print("  시그널 없음")
        return

    min_idx = min(all_indices)
    max_idx = max(all_indices)
    total_range = max_idx - min_idx

    # 4등분 (3구간 학습 + 1구간 테스트를 롤링)
    quarter = total_range // 4
    windows = []
    for w in range(3):
        train_start = min_idx + w * quarter
        train_end = train_start + quarter * 2
        test_start = train_end
        test_end = test_start + quarter
        windows.append((train_start, train_end, test_start, test_end))

    print(f"\n  전체 범위: idx {min_idx}~{max_idx} ({total_range} 캔들)")
    print(f"  윈도우 크기: 학습 {quarter*2} / 테스트 {quarter} 캔들")

    # 파라미터 후보
    param_grid = [
        {'min_score': ms, 'sl_mult': sl, 'tp_mult': tp}
        for ms in [3, 4, 5]
        for sl in [1.5, 2.0, 2.5]
        for tp in [3.0, 4.0, 5.0]
    ]

    wf_trades = []  # Walk-Forward 결과

    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"\n  윈도우 {wi+1}/3: 학습 [{tr_s}~{tr_e}] → 테스트 [{te_s}~{te_e}]")

        # 학습 구간에서 최적 파라미터 찾기
        best_pnl = -999
        best_params = BEST.copy()

        for params in param_grid:
            # 학습 구간 시그널만 필터
            train_sigs = {}
            for sym, sigs in signals.items():
                filtered = [s for s in sigs if tr_s <= s['idx'] < tr_e]
                if filtered:
                    train_sigs[sym] = filtered

            trades = run_strategy(train_sigs, data, **params, use_regime=False)
            pnl = sum(t['pnl_pct'] for t in trades) if trades else -999

            if pnl > best_pnl:
                best_pnl = pnl
                best_params = params.copy()

        print(f"    학습 최적: MS={best_params['min_score']} SL={best_params['sl_mult']} TP={best_params['tp_mult']} (학습 PnL {best_pnl:+.2f}%)")

        # 테스트 구간에 적용
        test_sigs = {}
        for sym, sigs in signals.items():
            filtered = [s for s in sigs if te_s <= s['idx'] < te_e]
            if filtered:
                test_sigs[sym] = filtered

        test_trades = run_strategy(test_sigs, data, **best_params, use_regime=False)
        test_pnl = sum(t['pnl_pct'] for t in test_trades) if test_trades else 0
        test_cnt = len(test_trades)
        test_wr = (len([t for t in test_trades if t['pnl_pct'] > 0]) / test_cnt * 100) if test_cnt > 0 else 0

        print(f"    테스트: {test_cnt}건 승률 {test_wr:.0f}% PnL {test_pnl:+.2f}%")
        wf_trades.extend(test_trades)

    # Walk-Forward 전체 결과
    print(f"\n  {'─'*40}")
    if wf_trades:
        wf_wins = len([t for t in wf_trades if t['pnl_pct'] > 0])
        wf_pnl = sum(t['pnl_pct'] for t in wf_trades)
        print(f"  Walk-Forward 종합: {len(wf_trades)}건 승률 {wf_wins/len(wf_trades)*100:.0f}% PnL {wf_pnl:+.2f}%")

        # 고정 파라미터와 비교
        fixed_trades = run_strategy(signals, data, **BEST, use_regime=False)
        fixed_pnl = sum(t['pnl_pct'] for t in fixed_trades)
        print(f"  고정 파라미터 비교: {len(fixed_trades)}건 PnL {fixed_pnl:+.2f}%")

        if wf_pnl > fixed_pnl:
            print(f"  → Walk-Forward가 {wf_pnl - fixed_pnl:+.2f}% 더 좋음")
        else:
            print(f"  → 고정 파라미터가 {fixed_pnl - wf_pnl:+.2f}% 더 좋음 (과적합 아닐 가능성)")
    else:
        print(f"  Walk-Forward 거래 없음")

    return wf_trades


# ═══════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("고급 백테스트: 레짐 감지 + Monte Carlo + Walk-Forward")
    print("=" * 60)

    # 데이터 로드
    print("\n데이터 로딩...")
    syms = ['ETHUSDT'] + get_alt_futures_symbols(15)
    data = load_data(syms)
    print(f"유효 종목: {len(data)-1}개")

    # 시그널 계산
    print("\n시그널 + 레짐 계산...")
    signals = precompute_all(data)
    total = sum(len(v) for v in signals.values())
    print(f"총 시그널: {total}개")

    # 1. 레짐 감지
    trades_no, trades_yes = test_regime(signals, data)

    # 2. Monte Carlo (레짐 OFF 기준)
    test_montecarlo(trades_no, n_sim=1000)

    # 3. Walk-Forward
    test_walkforward(signals, data)

    # 종합 요약
    print(f"\n{'='*60}")
    print("종합 요약")
    print(f"{'='*60}")
    print(f"  그리드서치 최적: MS=3 SL=1.5 TP=5.0")
    if trades_no:
        pnl_no = sum(t['pnl_pct'] for t in trades_no)
        print(f"  레짐 OFF: PnL {pnl_no:+.2f}%")
    if trades_yes:
        pnl_yes = sum(t['pnl_pct'] for t in trades_yes)
        print(f"  레짐 ON:  PnL {pnl_yes:+.2f}%")

    # 결과 저장
    rf = RESULT_DIR / f"advanced_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(rf, 'w') as f:
        json.dump({
            'best_params': BEST,
            'regime_off_trades': len(trades_no),
            'regime_off_pnl': round(sum(t['pnl_pct'] for t in trades_no), 2) if trades_no else 0,
            'regime_on_trades': len(trades_yes),
            'regime_on_pnl': round(sum(t['pnl_pct'] for t in trades_yes), 2) if trades_yes else 0,
        }, f, indent=2)
    print(f"\n결과 저장: {rf}")


if __name__ == '__main__':
    main()

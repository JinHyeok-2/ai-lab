#!/usr/bin/env python3
# 펀딩비 + 롱숏비율 + OI 시그널 백테스트
# 기존 스코어링 vs 파생지표 추가 스코어링 비교

import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from binance_client import get_klines, get_client
from indicators import calc_indicators
from alt_scanner import get_alt_futures_symbols

FEE_PCT = 0.07
MAX_SL_PCT = 5.0
MIN_TP_PCT = 0.3
MIN_SCORE = 3
SL_MULT = 1.5
TP_MULT = 5.0
BLACKLIST = {'BRUSDT', 'SIRENUSDT', 'XAUUSDT', 'XAGUSDT', 'RIVERUSDT'}
RESULT_DIR = Path('/home/hyeok/01.APCC/00.ai-lab/backtest_results')
RESULT_DIR.mkdir(exist_ok=True)


def fetch_sentiment(symbols):
    """펀딩비, 롱숏비율, OI 변화율 수집"""
    c = get_client()
    data = {}

    for sym in symbols:
        if sym in BLACKLIST:
            continue
        try:
            # 펀딩비 (최근 1건)
            fr = c.futures_funding_rate(symbol=sym, limit=8)
            funding_rate = float(fr[-1]['fundingRate']) if fr else 0
            # 최근 8회 평균 대비 현재
            fr_avg = np.mean([float(f['fundingRate']) for f in fr]) if len(fr) > 1 else funding_rate

            # 롱숏비율
            try:
                ls = c.futures_top_longshort_account_ratio(symbol=sym, period='1h', limit=5)
                if ls:
                    long_pct = float(ls[-1]['longAccount'])
                    ls_ratio = float(ls[-1]['longShortRatio'])
                else:
                    long_pct, ls_ratio = 0.5, 1.0
            except:
                long_pct, ls_ratio = 0.5, 1.0

            # OI 변화율 (1h)
            try:
                oih = c.futures_open_interest_hist(symbol=sym, period='1h', limit=5)
                if len(oih) >= 2:
                    oi_now = float(oih[-1]['sumOpenInterestValue'])
                    oi_prev = float(oih[-2]['sumOpenInterestValue'])
                    oi_change = (oi_now - oi_prev) / oi_prev * 100 if oi_prev > 0 else 0
                    # 5시간 평균 대비
                    oi_vals = [float(o['sumOpenInterestValue']) for o in oih]
                    oi_avg_change = (oi_vals[-1] - oi_vals[0]) / oi_vals[0] * 100 if oi_vals[0] > 0 else 0
                else:
                    oi_change, oi_avg_change = 0, 0
            except:
                oi_change, oi_avg_change = 0, 0

            data[sym] = {
                'funding_rate': funding_rate,
                'funding_rate_avg': fr_avg,
                'long_pct': long_pct,
                'ls_ratio': ls_ratio,
                'oi_change_1h': oi_change,
                'oi_change_5h': oi_avg_change,
            }
            time.sleep(0.3)
        except Exception as e:
            data[sym] = {'funding_rate': 0, 'funding_rate_avg': 0,
                         'long_pct': 0.5, 'ls_ratio': 1.0,
                         'oi_change_1h': 0, 'oi_change_5h': 0}
    return data


def score_base(i15, i1h, i4h, btc_up):
    """기존 스코어링"""
    sc = 0
    for ind, w in [(i15, 1), (i1h, 2), (i4h, 3)]:
        e20 = ind.get('ema20', 0) or 0
        e50 = ind.get('ema50', 0) or 0
        if e20 and e50: sc += w if e20 > e50 else -w
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


def score_with_sentiment(base_score, sent):
    """기존 점수 + 파생지표 보정"""
    bonus = 0
    fr = sent['funding_rate']
    long_pct = sent['long_pct']
    oi_change = sent['oi_change_1h']

    # 1. 펀딩비 역발상 (±2점)
    # 펀딩 양수(롱 과열) → 숏 보너스, 음수(숏 과열) → 롱 보너스
    if fr > 0.0005:       # 0.05%+ 강한 롱 과열
        bonus -= 2
    elif fr > 0.0001:     # 약한 롱 과열
        bonus -= 1
    elif fr < -0.0005:    # 강한 숏 과열
        bonus += 2
    elif fr < -0.0001:    # 약한 숏 과열
        bonus += 1

    # 2. 롱숏비율 역발상 (±2점)
    # 개인 65%+ 롱 → 숏 유리, 65%+ 숏 → 롱 유리
    if long_pct > 0.65:
        bonus -= 2  # 개인 과도 롱 → 숏 시그널
    elif long_pct > 0.58:
        bonus -= 1
    elif long_pct < 0.35:
        bonus += 2  # 개인 과도 숏 → 롱 시그널
    elif long_pct < 0.42:
        bonus += 1

    # 3. OI 변화율 (±1점)
    # OI 급증 + 기존 방향 → 추세 강화 확인
    if abs(oi_change) > 3:  # 1시간에 3%+ OI 변화 = 큰 움직임
        if oi_change > 0:
            bonus += 1  # OI 증가 = 새 자금 유입 → 현재 추세 강화
        else:
            bonus -= 1  # OI 감소 = 자금 이탈

    return base_score + bonus


def simulate_trade(df15, entry, sl, tp, direction, start_idx):
    """단일 거래 시뮬레이션"""
    for j in range(start_idx, min(start_idx + 40, len(df15))):
        h = float(df15.iloc[j]['high'])
        l = float(df15.iloc[j]['low'])
        if direction == 'long':
            if l <= sl: return 'sl', sl, j
            if h >= tp: return 'tp', tp, j
        else:
            if h >= sl: return 'sl', sl, j
            if l <= tp: return 'tp', tp, j
    ep = float(df15.iloc[min(start_idx+39, len(df15)-1)]['close'])
    return 'open', ep, start_idx + 40


def run_comparison():
    print("=" * 60)
    print("파생지표 시그널 효과 비교 백테스트")
    print("=" * 60)

    # 데이터 로드
    print("\n데이터 로딩...")
    syms_raw = ['ETHUSDT'] + get_alt_futures_symbols(15)
    syms = [s for s in syms_raw if s not in BLACKLIST and s != 'BTCUSDT']

    # 파생지표 수집 (현재 스냅샷)
    print("\n파생지표 수집...")
    sentiment = fetch_sentiment(syms)
    for sym, s in sentiment.items():
        fr_pct = s['funding_rate'] * 100
        print(f"  {sym:14s} 펀딩={fr_pct:+.4f}% 롱비={s['long_pct']:.0%} OI변화={s['oi_change_1h']:+.2f}%")

    # 캔들 데이터
    print("\n캔들 데이터 로딩...")
    data = {}
    btc_1h = get_klines('BTCUSDT', '1h', 500)
    for sym in syms:
        try:
            df15 = get_klines(sym, '15m', 500)
            df1h = get_klines(sym, '1h', 500)
            df4h = get_klines(sym, '4h', 200)
            recent = df15.tail(100)
            rng = (recent['high'].max() - recent['low'].min()) / recent['close'].mean() * 100
            if rng < 1.0 or len(df15) < 100 or len(df1h) < 60 or len(df4h) < 30:
                continue
            data[sym] = {'15m': df15, '1h': df1h, '4h': df4h}
            time.sleep(0.3)
        except:
            continue
    print(f"유효 종목: {len(data)}개")

    # 시뮬레이션 (두 버전 비교)
    results_base = []
    results_sent = []

    for sym, d in data.items():
        df15, df1h, df4h = d['15m'], d['1h'], d['4h']
        sent = sentiment.get(sym, {'funding_rate': 0, 'long_pct': 0.5, 'oi_change_1h': 0})
        cooldown_b = 0
        cooldown_s = 0

        for i in range(60, len(df15) - 20, 4):
            try:
                i15 = calc_indicators(df15.iloc[max(0,i-60):i].reset_index(drop=True))
                i1h_end = min(i//4+1, len(df1h))
                i1h_s = max(0, i1h_end-60)
                if i1h_end - i1h_s < 20: continue
                i1h_data = calc_indicators(df1h.iloc[i1h_s:i1h_end].reset_index(drop=True))
                i4h_end = min(i//16+1, len(df4h))
                i4h_s = max(0, i4h_end-60)
                if i4h_end - i4h_s < 15: continue
                i4h_data = calc_indicators(df4h.iloc[i4h_s:i4h_end].reset_index(drop=True))
                btc_end = min(i//4+1, len(btc_1h))
                btc_s = max(0, btc_end-60)
                if btc_end - btc_s < 20: continue
                btc_ind = calc_indicators(btc_1h.iloc[btc_s:btc_end].reset_index(drop=True))
                btc_up = (btc_ind.get('ema20',0) or 0) > (btc_ind.get('ema50',0) or 0)
            except:
                continue

            base_sc, rsi, adx = score_base(i15, i1h_data, i4h_data, btc_up)
            sent_sc = score_with_sentiment(base_sc, sent)
            entry = float(df15.iloc[i-1]['close'])
            atr = i1h_data.get('atr', 0) or 0
            if atr <= 0: atr = entry * 0.02
            if entry > 0 and atr / entry > 0.15: continue

            # 기존 스코어링
            if i >= cooldown_b:
                if base_sc >= MIN_SCORE: d_b = 'long'
                elif base_sc <= -MIN_SCORE: d_b = 'short'
                else: d_b = None
                if d_b:
                    sl = entry - atr*SL_MULT if d_b=='long' else entry + atr*SL_MULT
                    tp = entry + atr*TP_MULT if d_b=='long' else entry - atr*TP_MULT
                    sl_dist = abs(entry-sl)/entry*100
                    if sl_dist > MAX_SL_PCT:
                        sl = entry*(1-MAX_SL_PCT/100) if d_b=='long' else entry*(1+MAX_SL_PCT/100)
                    risk = abs(entry-sl)
                    if risk > 0 and abs(tp-entry)/risk < 2: tp = entry+risk*2 if d_b=='long' else entry-risk*2
                    res, ep, eidx = simulate_trade(df15, entry, sl, tp, d_b, i)
                    if abs(ep-entry)/entry >= 0.001:
                        pnl = ((ep-entry)/entry*100 if d_b=='long' else (entry-ep)/entry*100) - FEE_PCT
                        results_base.append({'sym': sym, 'dir': d_b, 'sc': base_sc, 'pnl': round(pnl,2), 'res': res})
                    cooldown_b = eidx + 8

            # 파생지표 보정 스코어링
            if i >= cooldown_s:
                if sent_sc >= MIN_SCORE: d_s = 'long'
                elif sent_sc <= -MIN_SCORE: d_s = 'short'
                else: d_s = None
                if d_s:
                    sl = entry - atr*SL_MULT if d_s=='long' else entry + atr*SL_MULT
                    tp = entry + atr*TP_MULT if d_s=='long' else entry - atr*TP_MULT
                    sl_dist = abs(entry-sl)/entry*100
                    if sl_dist > MAX_SL_PCT:
                        sl = entry*(1-MAX_SL_PCT/100) if d_s=='long' else entry*(1+MAX_SL_PCT/100)
                    risk = abs(entry-sl)
                    if risk > 0 and abs(tp-entry)/risk < 2: tp = entry+risk*2 if d_s=='long' else entry-risk*2
                    res, ep, eidx = simulate_trade(df15, entry, sl, tp, d_s, i)
                    if abs(ep-entry)/entry >= 0.001:
                        pnl = ((ep-entry)/entry*100 if d_s=='long' else (entry-ep)/entry*100) - FEE_PCT
                        results_sent.append({'sym': sym, 'dir': d_s, 'sc': sent_sc, 'sc_base': base_sc, 'pnl': round(pnl,2), 'res': res})
                    cooldown_s = eidx + 8

    # ── 결과 비교 ──
    print(f"\n{'='*60}")
    print("결과 비교")
    print(f"{'='*60}")

    def summary(trades, label):
        if not trades:
            print(f"\n{label}: 거래 없음")
            return
        w = len([t for t in trades if t['pnl'] > 0])
        pnl = sum(t['pnl'] for t in trades)
        longs = [t for t in trades if t['dir'] == 'long']
        shorts = [t for t in trades if t['dir'] == 'short']
        l_w = len([t for t in longs if t['pnl'] > 0])
        s_w = len([t for t in shorts if t['pnl'] > 0])
        print(f"\n{label}:")
        print(f"  총 {len(trades)}건 | 승 {w} | 승률 {w/len(trades)*100:.0f}% | PnL {pnl:+.2f}%")
        if longs:
            print(f"  롱 {len(longs)}건 승률 {l_w/len(longs)*100:.0f}% PnL {sum(t['pnl'] for t in longs):+.2f}%")
        if shorts:
            print(f"  숏 {len(shorts)}건 승률 {s_w/len(shorts)*100:.0f}% PnL {sum(t['pnl'] for t in shorts):+.2f}%")

    summary(results_base, "A. 기존 스코어링 (기술적 지표만)")
    summary(results_sent, "B. 파생지표 보정 (펀딩비+롱숏+OI)")

    # 차이 분석
    if results_base and results_sent:
        pnl_a = sum(t['pnl'] for t in results_base)
        pnl_b = sum(t['pnl'] for t in results_sent)
        diff = pnl_b - pnl_a
        print(f"\n{'─'*40}")
        if diff > 0:
            print(f"  파생지표 보정이 {diff:+.2f}% 더 좋음!")
        else:
            print(f"  기존이 {-diff:+.2f}% 더 좋음")

    # 파생지표 상세
    print(f"\n{'='*60}")
    print("현재 파생지표 스냅샷 → 시그널 해석")
    print(f"{'='*60}")
    for sym, s in sentiment.items():
        if sym not in data:
            continue
        signals = []
        fr = s['funding_rate']
        if fr > 0.0005: signals.append("펀딩↑(숏유리)")
        elif fr < -0.0005: signals.append("펀딩↓(롱유리)")
        if s['long_pct'] > 0.65: signals.append(f"개인롱{s['long_pct']:.0%}(숏유리)")
        elif s['long_pct'] < 0.35: signals.append(f"개인숏{1-s['long_pct']:.0%}(롱유리)")
        if abs(s['oi_change_1h']) > 3: signals.append(f"OI{s['oi_change_1h']:+.1f}%")
        sig_str = " | ".join(signals) if signals else "중립"
        print(f"  {sym:14s} → {sig_str}")

    # 거래 상세
    print(f"\n파생지표 보정 거래 상세:")
    for t in results_sent:
        f = 'W' if t['pnl'] > 0 else 'L'
        bonus = t['sc'] - t['sc_base']
        print(f"  {t['sym']:14s} {t['dir']:5s} base={t['sc_base']:+3d} bonus={bonus:+2d} → {t['sc']:+3d} {t['res']:4s} {t['pnl']:+.2f}% {f}")

    # 저장
    rf = RESULT_DIR / f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(rf, 'w') as f:
        json.dump({
            'sentiment': {k: v for k, v in sentiment.items() if k in data},
            'base_pnl': round(sum(t['pnl'] for t in results_base), 2) if results_base else 0,
            'sent_pnl': round(sum(t['pnl'] for t in results_sent), 2) if results_sent else 0,
            'base_trades': len(results_base),
            'sent_trades': len(results_sent),
        }, f, indent=2)
    print(f"\n결과 저장: {rf}")


if __name__ == '__main__':
    run_comparison()

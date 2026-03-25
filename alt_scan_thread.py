#!/usr/bin/env python3
# 알트코인 자동 스캔 백그라운드 스레드 — 5분마다 독립 실행
# Streamlit 의존성 없이 동작

import sys
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
import threading
import traceback
from datetime import datetime
from config import ALT_SCAN_LIMIT, ALT_MIN_SCORE, ALT_AUTO_CONFIDENCE, ALT_LEVERAGE, MAX_USDT_ALT, POSITION_PCT_ALT
from alt_scanner import get_alt_futures_symbols, screen_altcoins, run_alt_analysis
from binance_client import (get_positions, get_balance, get_klines, get_price,
                            place_limit_order, cancel_open_orders, get_client)
from indicators import calc_indicators
import trade_db
import logging

_logger = logging.getLogger("alt_scan_thread")
_ALT_SCAN_INTERVAL = 300  # 5분

# 스레드 안전 로그 (trading.log에 직접 기록)
_log_lock = threading.Lock()
_LOG_PATH = '/home/hyeok/01.APCC/00.ai-lab/trading.log'

def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}\n"
    with _log_lock:
        try:
            with open(_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            pass


def _calc_confidence_simple(cand, tj, btc_ind=None):
    """알트 신뢰도 계산 (app.py calc_confidence_alt 기반 강화 버전)"""
    direction = tj.get("signal", "wait")
    if direction == "wait":
        return 25, "wait"
    score = 20
    ind = cand.get("indicators", {})
    # 1. 스크리너 점수 (최대 20점)
    sc = cand.get("score", 0)
    if sc >= 60: score += 20
    elif sc >= 40: score += 12
    elif sc >= 25: score += 6
    # 2. 거래량 (최대 15점)
    vr = cand.get("vol_ratio", 1)
    if vr >= 5: score += 15
    elif vr >= 3: score += 10
    elif vr >= 2: score += 5
    # 3. RSI (최대 12점)
    rsi = ind.get("rsi", 50) or 50
    if direction == "long" and rsi < 30: score += 12
    elif direction == "long" and rsi < 40: score += 8
    elif direction == "short" and rsi > 70: score += 12
    elif direction == "short" and rsi > 60: score += 8
    elif direction == "long" and rsi > 70: score -= 8
    elif direction == "short" and rsi < 30: score -= 8
    # 4. MACD 방향 일치 (8점)
    macd_hist = ind.get("macd_hist")
    if macd_hist is not None:
        if direction == "long" and macd_hist > 0: score += 8
        elif direction == "short" and macd_hist < 0: score += 8
        else: score -= 4
    # 5. EMA 배열 (8점)
    ema20 = ind.get("ema20")
    ema50 = ind.get("ema50")
    price = ind.get("price", 0)
    if ema20 and ema50 and price:
        if direction == "long" and ema20 > ema50 and price > ema20: score += 8
        elif direction == "short" and ema20 < ema50 and price < ema20: score += 8
        elif direction == "long" and ema20 < ema50: score -= 4
        elif direction == "short" and ema20 > ema50: score -= 4
    # 6. ADX 추세 강도 (8점)
    adx = ind.get("adx", 0) or 0
    if adx >= 25: score += 8
    elif adx < 15: score -= 3
    # 7. BTC 동조 (±8점)
    if btc_ind:
        be20 = btc_ind.get("ema20", 0) or 0
        be50 = btc_ind.get("ema50", 0) or 0
        if be20 and be50:
            if direction == "long" and be20 > be50: score += 8
            elif direction == "short" and be20 < be50: score += 8
            elif direction == "long" and be20 < be50: score -= 6
            elif direction == "short" and be20 > be50: score -= 6
    return min(max(score, 0), 100), direction


def _alt_place_order_thread(sym, sig, tj, cand):
    """스레드 전용 알트 지정가 주문"""
    try:
        px = cand.get("price", 0)
        atr = cand.get("atr", 0) or 0.001
        side = "BUY" if sig == "long" else "SELL"

        # 4H 추세 필터
        try:
            kl4h = get_klines(sym, "4h", 60)
            ind4h = calc_indicators(kl4h)
            e20 = ind4h.get("ema20", 0) or 0
            e50 = ind4h.get("ema50", 0) or 0
            if e20 and e50:
                if sig == "long" and e20 < e50:
                    _log(f"🚫 알트 {sym} 롱 차단: 4H 하락추세 (스레드)")
                    return
                if sig == "short" and e20 > e50:
                    _log(f"🚫 알트 {sym} 숏 차단: 4H 상승추세 (스레드)")
                    return
        except Exception:
            pass

        # BTC 동조
        try:
            btc_kl = get_klines("BTCUSDT", "1h", 60)
            btc_ind = calc_indicators(btc_kl)
            btc_e20 = btc_ind.get("ema20", 0) or 0
            btc_e50 = btc_ind.get("ema50", 0) or 0
            btc_chg = btc_ind.get("change_pct", 0) or 0
            if sig == "long" and btc_e20 < btc_e50 and btc_chg < -1.0:
                _log(f"🚫 알트 {sym} 롱 차단: BTC 하락 (스레드)")
                return
            if sig == "short" and btc_e20 > btc_e50 and btc_chg > 1.0:
                _log(f"🚫 알트 {sym} 숏 차단: BTC 상승 (스레드)")
                return
        except Exception:
            pass

        # 최적 진입가 (현재가에 가깝게 — 체결률 우선)
        entry = tj.get("entry")
        if entry:
            entry = float(entry)
            if abs(entry - px) / px * 100 > 3.0:
                entry = None
        if not entry:
            entry = round(px - atr * 0.3, 6) if sig == "long" else round(px + atr * 0.3, 6)

        sl = tj.get("sl")
        tp = tj.get("tp")
        if not sl:
            sl = round(px - atr * 2, 6) if sig == "long" else round(px + atr * 2, 6)
        if not tp:
            tp = round(px + atr * 4, 6) if sig == "long" else round(px - atr * 4, 6)

        bal = get_balance()
        usdt = min(bal["available"] * POSITION_PCT_ALT / 100, MAX_USDT_ALT)

        cancel_open_orders(sym)
        time.sleep(0.3)
        r = place_limit_order(sym, side, usdt, entry, ALT_LEVERAGE,
                              sl_price=float(sl), tp_price=float(tp))
        if r["success"]:
            _log(f"📋 알트 지정가(스레드): {sym} {'🟢 롱' if sig=='long' else '🔴 숏'} @ ${entry} (SL ${sl}, TP ${tp})")
        else:
            _log(f"❌ 알트 주문 실패(스레드): {sym} {r.get('error','')[:60]}")
    except Exception as e:
        _log(f"알트 주문 오류(스레드): {sym} {e}")


def _scan_cycle():
    """알트 스캔 1사이클"""
    try:
        syms = get_alt_futures_symbols(ALT_SCAN_LIMIT)
        results = screen_altcoins(syms, top_n=10)
        if results:
            _log(f"🔥 알트 스캔(스레드): {results[0]['symbol']} 1위 (점수 {results[0]['score']})")

        # BTC 지표 캐시
        btc_ind = {}
        try:
            btc_kl = get_klines("BTCUSDT", "1h", 60)
            btc_ind = calc_indicators(btc_kl)
        except Exception:
            pass

        # 포지션 한도 확인
        try:
            from config import MAX_ALT_POSITIONS, SYMBOLS
            cur_pos = get_positions()
            alt_pos_cnt = len([p for p in cur_pos if p["symbol"] not in SYMBOLS])
            if alt_pos_cnt >= MAX_ALT_POSITIONS:
                _log(f"⏭️ 알트 스캔 스킵(스레드): 포지션 {alt_pos_cnt}/{MAX_ALT_POSITIONS}")
                return
        except Exception:
            pass

        # 상위 5개 분석
        from concurrent.futures import ThreadPoolExecutor
        to_analyze = [c for c in (results or [])[:5] if c["score"] >= ALT_MIN_SCORE]

        if to_analyze:
            def _run(c):
                try:
                    return c["symbol"], run_alt_analysis(c)
                except Exception:
                    return c["symbol"], None

            analysis = {}
            with ThreadPoolExecutor(max_workers=min(3, len(to_analyze))) as ex:
                for s, r in [f.result() for f in [ex.submit(_run, c) for c in to_analyze]]:
                    if r:
                        analysis[s] = r

            for cand in to_analyze:
                sym = cand["symbol"]
                res = analysis.get(sym)
                if not res:
                    continue
                tj = res.get("trader_json", {})
                conf, direction = _calc_confidence_simple(cand, tj, btc_ind)

                # 숏은 +5pt 높은 컷오프 (과거 숏 전패 교훈)
                _min_conf = ALT_AUTO_CONFIDENCE + (5 if direction == "short" else 0)
                if direction != "wait" and conf >= _min_conf:
                    _alt_place_order_thread(sym, direction, tj, cand)
                else:
                    _log(f"⏸ 알트 보류(스레드): {sym} 신뢰도 {conf}% dir={direction}")

                # 분석 이력 저장
                try:
                    trade_db.save_analysis({
                        "symbol": sym, "decision": direction, "confidence": conf,
                        "entry_price": tj.get("entry"), "sl": tj.get("sl"), "tp": tj.get("tp"),
                        "rsi_15m": cand.get("indicators", {}).get("rsi"),
                        "adx_15m": cand.get("indicators", {}).get("adx"),
                        "atr": cand.get("atr"),
                        "btc_trend": "up" if (btc_ind.get("ema20", 0) or 0) > (btc_ind.get("ema50", 0) or 0) else "down",
                        "llm_reason": tj.get("reason", "")[:500],
                        "source": "alt_thread",
                    })
                except Exception:
                    pass

    except Exception as e:
        _log(f"알트 스캔 스레드 오류: {e}")
        traceback.print_exc()


def start_alt_scan_thread():
    """알트 스캔 스레드 시작 (5분마다)"""
    def _run():
        _log("🔄 알트 스캔 백그라운드 스레드 시작 (5분 주기)")
        while True:
            _scan_cycle()
            time.sleep(_ALT_SCAN_INTERVAL)

    t = threading.Thread(target=_run, daemon=True, name="alt-scan-5min")
    t.start()
    return t

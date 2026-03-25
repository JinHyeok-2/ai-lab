#!/usr/bin/env python3
# 5분마다 자동 종목 스캔 → 기술적 분석 → 지정가 예약 업데이트
# LIMIT만 배치, 체결 시 SL/TP 자동 배치 (conditional 잔존 방지)

import sys
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
from datetime import datetime, timezone
from binance_client import (get_klines, get_price, get_balance, get_positions,
                            get_client, place_sl_tp, _get_symbol_filters, _round_price,
                            set_margin_type, set_leverage)
from indicators import calc_indicators
from telegram_notifier import send_message
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from alt_scanner import get_alt_futures_symbols
from signal_queue import pop_signals
import trade_db

# ── 설정 ──
INTERVAL = 300          # 5분
MAX_ORDERS = 4          # 동시 최대 주문 수 (3→4, 분산 효과)
ETH_USDT = 10           # ETH 진입금
ALT_USDT = 8            # 알트 진입금
ALT_USDT_MAX = 20       # 알트 노셔널 상한
ETH_LEV = 3             # ETH 레버리지
ALT_LEV = 2             # 알트 레버리지
MIN_SCORE = 3           # 최소 진입 점수 (그리드서치 최적: 3)
MAX_SL_PCT = 5.0        # SL 최대 거리 (%)
MAX_SAME_DIR = 4        # 동일 방향 최대 (2→4, 롱전용이므로 MAX_ORDERS와 동일)
MAX_DAILY_TRADES = 8    # 하루 최대 거래 횟수 (5→8, 감지기 추가로 기회 증가)
NIGHT_HOURS = {0, 1, 2, 3, 4}  # 야간 진입 차단 (KST)
BLACKLIST = {'BRUSDT', 'SIRENUSDT', 'XAUUSDT', 'XAGUSDT', 'RIVERUSDT'}  # 반복 손실/극단 변동성/TradFi 미서명/지표역행
TG_TOKEN = TELEGRAM_TOKEN
TG_CHAT  = TELEGRAM_CHAT_ID

# 스캔 대상 (ETH 고정 + 거래량 상위 동적 선택)
SCAN_FIXED = ['ETHUSDT']  # 항상 포함
SCAN_ALT_COUNT = 25       # 알트 동적 선택 수 (15→25, 더 많은 기회 포착)
_scan_cache = {'ts': 0, 'symbols': []}

LOG_PATH = '/home/hyeok/01.APCC/00.ai-lab/position_updater.log'
LOCK_PATH = '/home/hyeok/01.APCC/00.ai-lab/updater_managed.json'

# 체결 대기 추적
_pending_fills = {}
# SL/TP 배치 완료 종목 (재시작 시 1회만 보완)
_sltp_done = set()
# 포지션별 TP 캐시 (부분 익절에서 DB 조회 대신 사용)
_tp_cache = {}  # {symbol: tp_price}
# 트레일링 스탑: 최고/최저 가격 추적
_trail_peak = {}  # {symbol: best_price_so_far}
_trail_atr = {}   # {symbol: ATR(1h)} — 트레일링 활성화 시 캐시
TRAIL_ACTIVATE_PCT = 1.0  # 수익 1%부터 트레일링 활성화 (1x 기준)
# ATR 비례 트레일링 (수익 단계별 ATR 배수)
TRAIL_ATR_TIERS = [
    (5.0, 0.2),   # 수익 5%+ → ATR × 0.2 (타이트)
    (2.0, 0.3),   # 수익 2~5% → ATR × 0.3
    (1.0, 0.4),   # 수익 1~2% → ATR × 0.4 (기본)
]
TRAIL_MIN_PCT = 0.3   # 최소 트레일 거리 0.3% (ETH 등 저변동성 보호)
TRAIL_MAX_PCT = 3.0   # 최대 트레일 거리 3.0% (극변동 종목 캡)
# 부분 익절 완료 종목 (사이클마다 중복 방지)
_partial_done = set()
# 종목별 쿨다운 (청산 후 30분간 재진입 방지)
COOLDOWN_SEC = 1800  # 30분
_cooldown = {}  # {symbol: expire_time}
# BTC 지표 캐시 (사이클당 1회만 조회)
_btc_cache = {'ts': 0, 'up': True, 'rsi': 50}
# 텔레그램 전송 주기 (분석은 5분, 알림은 30분)
TG_INTERVAL = 1800  # 30분
_last_tg_time = 0
# 일일 거래 횟수 + 연속 손실 추적
_daily_trades = {'date': '', 'count': 0}
_consecutive_losses = 0


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def get_scan_universe():
    """ETH 고정 + 거래량 상위 알트 동적 선택 (10분 캐시)"""
    now = time.time()
    if now - _scan_cache['ts'] < 600 and _scan_cache['symbols']:
        return _scan_cache['symbols']
    try:
        alts = get_alt_futures_symbols(SCAN_ALT_COUNT)
        alt_syms = [s for s in alts if s not in SCAN_FIXED][:SCAN_ALT_COUNT]
        result = SCAN_FIXED + alt_syms
        _scan_cache.update({'ts': now, 'symbols': result})
        return result
    except Exception:
        return _scan_cache.get('symbols', SCAN_FIXED)


def get_btc_trend():
    """BTC 1H 추세 (사이클당 1회 캐시)"""
    now = time.time()
    if now - _btc_cache['ts'] < 250:  # 4분 캐시
        return _btc_cache['up'], _btc_cache['rsi']
    try:
        btc = calc_indicators(get_klines('BTCUSDT', '1h', 60))
        up = (btc.get('ema20', 0) or 0) > (btc.get('ema50', 0) or 0)
        rsi = btc.get('rsi', 50) or 50
        _btc_cache.update({'ts': now, 'up': up, 'rsi': rsi})
        return up, rsi
    except Exception:
        return _btc_cache['up'], _btc_cache['rsi']


# 파생지표 캐시 (10분 TTL)
_sentiment_cache = {'ts': 0, 'data': {}}

def _get_sentiment_bonus(symbol):
    """펀딩비 + 롱숏비율 + OI 역발상 보너스 (±5점, 10분 캐시)"""
    now = time.time()
    if now - _sentiment_cache['ts'] > 600:
        # 캐시 만료 → 갱신
        _sentiment_cache['ts'] = now
        _sentiment_cache['data'] = {}

    if symbol in _sentiment_cache['data']:
        return _sentiment_cache['data'][symbol]

    bonus = 0
    try:
        client = get_client()

        # 1. 펀딩비 역발상 (±2점)
        fr = client.futures_funding_rate(symbol=symbol, limit=1)
        if fr:
            rate = float(fr[-1]['fundingRate'])
            if rate > 0.0005: bonus -= 2      # 롱 과열 → 숏 유리
            elif rate > 0.0001: bonus -= 1
            elif rate < -0.0005: bonus += 2    # 숏 과열 → 롱 유리
            elif rate < -0.0001: bonus += 1

        # 2. 롱숏비율 역발상 (±2점)
        try:
            ls = client.futures_top_longshort_account_ratio(symbol=symbol, period='1h', limit=1)
            if ls:
                long_pct = float(ls[-1]['longAccount'])
                if long_pct > 0.65: bonus -= 2    # 개인 과도 롱 → 숏 유리
                elif long_pct > 0.58: bonus -= 1
                elif long_pct < 0.35: bonus += 2   # 개인 과도 숏 → 롱 유리
                elif long_pct < 0.42: bonus += 1
        except Exception:
            pass

        # 3. OI 변화율 (±1점)
        try:
            oih = client.futures_open_interest_hist(symbol=symbol, period='1h', limit=2)
            if len(oih) >= 2:
                oi_now = float(oih[-1]['sumOpenInterestValue'])
                oi_prev = float(oih[-2]['sumOpenInterestValue'])
                oi_change = (oi_now - oi_prev) / oi_prev * 100 if oi_prev > 0 else 0
                if oi_change > 3: bonus += 1      # OI 급증 → 추세 강화
                elif oi_change < -3: bonus -= 1    # OI 급감 → 자금 이탈
        except Exception:
            pass
    except Exception:
        pass

    _sentiment_cache['data'][symbol] = bonus
    return bonus


def score_symbol(symbol):
    """멀티TF 기술적 분석 + 파생지표 → 점수 + 메타데이터"""
    try:
        px = get_price(symbol)
        i15 = calc_indicators(get_klines(symbol, '15m', 60))
        i1h = calc_indicators(get_klines(symbol, '1h', 60))
        i4h = calc_indicators(get_klines(symbol, '4h', 60))
        btc_up, _ = get_btc_trend()

        # 데이터 부족 종목 필터 (RSI/ADX가 nan이면 스킵)
        import math
        _rsi_check = i15.get('rsi', 0) or 0
        _adx_check = i15.get('adx', 0) or 0
        if (isinstance(_rsi_check, float) and math.isnan(_rsi_check)) or \
           (isinstance(_adx_check, float) and math.isnan(_adx_check)):
            return {'symbol': symbol, 'score': 0, 'direction': 'wait', 'error': 'nan_data'}

        score = 0

        # 1. EMA 배열 (15m/1h/4h — 4h 가중치 강화)
        for lb, ind in [('15m', i15), ('1h', i1h), ('4h', i4h)]:
            e20 = ind.get('ema20', 0) or 0
            e50 = ind.get('ema50', 0) or 0
            if e20 and e50:
                w = 5 if lb == '4h' else 2  # 4h: 3→5 (추세 종목 캐치)
                score += w if e20 > e50 else -w

        # 2. RSI
        rsi = i15.get('rsi', 50) or 50
        if rsi < 35: score += 3
        elif rsi < 45: score += 1
        elif rsi > 65: score -= 3
        elif rsi > 55: score -= 1
        rsi1h = i1h.get('rsi', 50) or 50
        if rsi1h < 40: score += 2
        elif rsi1h > 60: score -= 2

        # 2-1. 눌림 구간 보너스 (15m RSI 40~60 = 과매수 아닌 진입 적기)
        if 40 <= rsi <= 60:
            score += 2  # 눌림에서 진입 우대

        # 3. MACD
        if (i15.get('macd_hist', 0) or 0) > 0: score += 1
        else: score -= 1
        if (i1h.get('macd_hist', 0) or 0) > 0: score += 2
        else: score -= 2

        # 4. ADX (추세 강도)
        adx = i15.get('adx', 0) or 0
        adx_1h = i1h.get('adx', 0) or 0
        if adx < 15:
            score = int(score * 0.5)  # 횡보 감점
        if adx_1h >= 25:
            score += 2  # 1h ADX 25+ = 추세 확인 보너스

        # 5. BTC 동조
        if btc_up: score += 2
        else: score -= 2

        # 6. 파생지표 역발상 (펀딩비 + 롱숏비율 + OI)
        sentiment_bonus = _get_sentiment_bonus(symbol)
        score += sentiment_bonus

        # 7. 지지/저항 보너스 (±2점)
        bb_upper = i1h.get('bb_upper', 0) or 0
        bb_lower = i1h.get('bb_lower', 0) or 0
        bb_pct = 50  # 기본값
        bb_squeeze = False

        if bb_upper and bb_lower and bb_upper > bb_lower:
            bb_range = bb_upper - bb_lower
            bb_pct = (px - bb_lower) / bb_range * 100
            # 롱: BB 하단 30% 이내 → 지지선 근처 (+2)
            if bb_pct < 30:
                score += 2
            # 숏: BB 상단 30% 이내 → 저항선 근처 (-2)
            elif bb_pct > 70:
                score -= 2
            # 박스 중간 (40~60%) → 방향성 약함, 감점
            elif 40 < bb_pct < 60:
                score = int(score * 0.7)

            # 8. 볼린저 스퀴즈 감지 (BB 폭이 가격의 1.5% 이하 = 곧 큰 움직임)
            bb_width_pct = bb_range / px * 100
            if bb_width_pct < 1.5:
                bb_squeeze = True
                score = int(score * 1.3)  # 스퀴즈 시 확신도 30% 증폭

        # 9. 거래량 확인 (±2점)
        vol = i15.get('volume', 0) or 0
        vol_avg = i15.get('vol_ma20', 0) or 0
        if vol_avg and vol_avg > 0:
            vol_ratio = vol / vol_avg
            if vol_ratio > 2.0:
                # 거래량 2배+ → 방향 강화
                score += 2 if score > 0 else -2
            elif vol_ratio < 0.5:
                # 거래량 절반 이하 → 가짜 움직임, 감점
                score = int(score * 0.7)

        # 방향 (그리드서치: 숏 엄격도 +0이 최적)
        if score >= MIN_SCORE: direction = 'long'
        # elif score <= -MIN_SCORE: direction = 'short'  # 숏 비활성화 (15건 40% -$9.50)
        else: direction = 'wait'

        # 10. RSI 극단값 진입 차단 (급등 되돌림 방지)
        if direction == 'long' and rsi > 85:
            direction = 'wait'
        elif direction == 'short' and rsi < 15:
            direction = 'wait'

        atr = i1h.get('atr', 0) or 0
        return {
            'symbol': symbol, 'price': px, 'score': score, 'direction': direction,
            'rsi': rsi, 'adx': adx, 'atr_1h': atr,
            'ema20_1h': i1h.get('ema20', 0) or 0,
            'bb_lower': i1h.get('bb_lower', 0) or 0,
            'bb_upper': i1h.get('bb_upper', 0) or 0,
            'btc_up': btc_up, 'sentiment_bonus': sentiment_bonus,
            'bb_pct': round(bb_pct, 0), 'bb_squeeze': bb_squeeze,
        }
    except Exception as e:
        return {'symbol': symbol, 'score': 0, 'direction': 'wait', 'error': str(e)[:40]}


def calc_entry(a):
    """분석 결과 → 진입가/SL/TP"""
    px = a['price']
    d = a['direction']
    atr = a['atr_1h']
    ema20 = a['ema20_1h']

    if d == 'wait':
        return None, None, None

    # ATR=0 폴백: 가격의 2%를 ATR로 사용
    if atr <= 0:
        atr = px * 0.02

    # 극변동 종목 스킵 (ATR > 가격의 15%)
    if px > 0 and atr / px > 0.15:
        return None, None, None

    is_major = a['symbol'] in ('ETHUSDT', 'BTCUSDT')
    lev = 3 if is_major else 2  # ETH/BTC 3x, 알트 2x
    sl_m = 1.5  # SL: ATR×1.5
    tp_m = 3.0 if is_major else 2.0  # TP: ETH/BTC ATR×3, 알트 ATR×2 (수시간 내 도달 가능)

    if d == 'long':
        entry = px - atr * 0.1  # 현재가 가까이 (체결률 ↑)
        sl = entry - atr * sl_m
        tp = entry + atr * tp_m
    else:
        entry = px + atr * 0.1
        sl = entry + atr * sl_m
        tp = entry - atr * tp_m

    # ── 고점/저점 기반 TP (실제 저항/지지선 활용) ──
    # 단, 4h ADX 25+ (강한 추세) → 고점 캡 해제 (추세 종목은 고점 갱신 가능)
    _4h_adx = a.get('adx', 0) or 0  # score_symbol에서 이미 계산된 4h ADX 사용 불가 → 재조회
    _skip_resistance_cap = False
    try:
        client = get_client()
        _kl_4h = client.futures_klines(symbol=a['symbol'], interval='4h', limit=30)
        # 4h ADX 간이 계산 (DX 평균)
        _closes_4h = [float(k[4]) for k in _kl_4h]
        _highs_4h = [float(k[2]) for k in _kl_4h]
        _lows_4h = [float(k[3]) for k in _kl_4h]
        if len(_closes_4h) >= 15:
            _dm_plus, _dm_minus, _tr_list = [], [], []
            for _i in range(1, len(_closes_4h)):
                _h_diff = _highs_4h[_i] - _highs_4h[_i-1]
                _l_diff = _lows_4h[_i-1] - _lows_4h[_i]
                _dm_plus.append(_h_diff if _h_diff > _l_diff and _h_diff > 0 else 0)
                _dm_minus.append(_l_diff if _l_diff > _h_diff and _l_diff > 0 else 0)
                _tr_list.append(max(_highs_4h[_i]-_lows_4h[_i],
                                    abs(_highs_4h[_i]-_closes_4h[_i-1]),
                                    abs(_lows_4h[_i]-_closes_4h[_i-1])))
            _n = 14
            if len(_tr_list) >= _n:
                _atr14 = sum(_tr_list[-_n:]) / _n
                _dp14 = sum(_dm_plus[-_n:]) / _n
                _dn14 = sum(_dm_minus[-_n:]) / _n
                _di_p = (_dp14 / _atr14 * 100) if _atr14 > 0 else 0
                _di_n = (_dn14 / _atr14 * 100) if _atr14 > 0 else 0
                _dx = abs(_di_p - _di_n) / (_di_p + _di_n) * 100 if (_di_p + _di_n) > 0 else 0
                _4h_adx = _dx
                if _4h_adx >= 25:
                    _skip_resistance_cap = True
    except Exception:
        pass

    if not _skip_resistance_cap:
        try:
            kl_1h = client.futures_klines(symbol=a['symbol'], interval='1h', limit=24)
            if d == 'long':
                recent_high = max(float(k[2]) for k in kl_1h)
                tp_resistance = recent_high * 0.998
                if tp_resistance > entry:
                    tp = min(tp, tp_resistance)
            else:
                recent_low = min(float(k[3]) for k in kl_1h)
                tp_support = recent_low * 1.002
                if tp_support < entry:
                    tp = max(tp, tp_support)
        except Exception:
            pass
    # 강한 추세 → ATR 기반 TP 유지 (고점 캡 해제)

    # SL 최대 거리 캡 — 레버리지 기준 실제 손실 5% 이내
    max_sl_1x = MAX_SL_PCT / lev
    sl_dist_pct = abs(entry - sl) / entry * 100
    if sl_dist_pct > max_sl_1x:
        sl = entry * (1 - max_sl_1x / 100) if d == 'long' else entry * (1 + max_sl_1x / 100)

    # TP 최소 보장 — 레버리지 기준 실제 수익 최소 4% (고점 기반은 가까울 수 있음)
    min_tp_real = 0.04  # 실제 4% (수수료 0.14% 차감 후 3.86%)
    min_tp_1x = entry * (min_tp_real / lev)
    if abs(tp - entry) < min_tp_1x:
        tp = entry + min_tp_1x if d == 'long' else entry - min_tp_1x

    # R:R 최소 1:1.5 보장
    risk = abs(entry - sl)
    if risk > 0 and abs(tp - entry) / risk < 1.5:
        tp = entry + risk * 1.5 if d == 'long' else entry - risk * 1.5

    # 소수점 정리
    if px > 100:
        entry, sl, tp = round(entry, 2), round(sl, 2), round(tp, 2)
    elif px > 1:
        entry, sl, tp = round(entry, 4), round(sl, 4), round(tp, 4)
    else:
        entry, sl, tp = round(entry, 6), round(sl, 6), round(tp, 6)

    return entry, sl, tp


def place_limit_only(symbol, side, usdt, entry_price, leverage):
    """LIMIT만 배치 (SL/TP 없이)"""
    client = get_client()
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
        time.sleep(0.3)
    except Exception:
        pass

    set_margin_type(symbol, "ISOLATED")
    set_leverage(symbol, leverage)

    step, tick = _get_symbol_filters(symbol)
    qty_raw = (usdt * leverage) / entry_price
    dec = 0 if step >= 1 else len(str(step).rstrip("0").split(".")[-1])
    qty = round(qty_raw - (qty_raw % step), dec)
    if qty <= 0:
        qty = step
    # 노셔널 $20 미달 시 자동 상향 (상한 $25)
    if qty * entry_price < 20:
        min_qty = 20.0 / entry_price
        qty = round(min_qty + step - (min_qty % step), dec)
        if qty <= 0:
            qty = step
    # 노셔널 상한 캡
    max_usdt = ALT_USDT_MAX if symbol != 'ETHUSDT' else ETH_USDT * ETH_LEV
    if qty * entry_price > max_usdt:
        qty = round(max_usdt / entry_price - (max_usdt / entry_price % step), dec)
        if qty <= 0:
            qty = step

    ep = _round_price(entry_price, tick)
    try:
        order = client.futures_create_order(
            symbol=symbol, side=side, type="LIMIT",
            price=ep, quantity=qty, timeInForce="GTC",
        )
        return {"success": True, "order_id": order.get("orderId"), "qty": qty, "price": ep}
    except Exception as e:
        return {"success": False, "error": str(e)[:80]}


def has_position(symbol):
    return any(p['symbol'] == symbol for p in _get_positions_cached())


_positions_cache = {'ts': 0, 'data': []}

def _get_positions_cached():
    """get_positions() 캐시 (5초 TTL) — 사이클당 API 호출 1회로 축소"""
    now = time.time()
    if now - _positions_cache['ts'] < 5:
        return _positions_cache['data']
    try:
        data = get_positions()
        _positions_cache.update({'ts': now, 'data': data})
        return data
    except Exception:
        return _positions_cache['data']


def get_held_symbols():
    """현재 포지션 심볼 목록"""
    return {p['symbol'] for p in _get_positions_cached()}


def _get_held_direction(symbol):
    """포지션 방향 조회 (long/short/None)"""
    for p in _get_positions_cached():
        if p['symbol'] == symbol:
            return 'long' if p['side'] == 'LONG' else 'short'
    return None


def check_fills():
    """체결 감지 → SL/TP 배치 + 기존 포지션 SL/TP 누락 보완"""
    # 1. pending에서 체결 감지
    filled = []
    for sym, info in list(_pending_fills.items()):
        if has_position(sym):
            try:
                pos = [p for p in _get_positions_cached() if p['symbol'] == sym][0]
                qty = float(pos.get('size', 0))
                if qty > 0:
                    place_sl_tp(sym, info['side'], qty,
                                sl_price=info['sl'], tp_price=info['tp'])
                    _sltp_done.add(sym)
                    _tp_cache[sym] = info['tp']
                    _daily_trades['count'] += 1
                    log(f"  ✅ {sym} 체결! SL/TP 배치 (SL ${info['sl']}, TP ${info['tp']}) [오늘 {_daily_trades['count']}/{MAX_DAILY_TRADES}건]")
                    # DB 기록
                    try:
                        _side_label = "🟢 롱" if info['side'] == 'BUY' else "🔴 숏"
                        trade_db.add_trade({
                            "symbol": sym, "side": _side_label, "action": "진입",
                            "qty": qty, "price": info['entry'],
                            "sl": info['sl'], "tp": info['tp'],
                            "atr": info.get('atr', 0),
                            "confidence": abs(info.get('score', 0)),
                            "source": "updater",
                            "extra": f"score={info.get('score',0)}",
                        })
                    except Exception:
                        pass
                    try:
                        send_message(TG_TOKEN, TG_CHAT,
                            f"🎯 <b>{sym} 체결!</b>\n"
                            f"   진입 ${info['entry']}\n"
                            f"   SL ${info['sl']} / TP ${info['tp']}")
                    except Exception:
                        pass
            except Exception as e:
                log(f"  ⚠️ {sym} SL/TP 실패: {e}")
            filled.append(sym)
    for sym in filled:
        del _pending_fills[sym]

    # 2. 포지션 보유 중인데 SL/TP 추적 안 된 종목 보완 (재시작 시 1회만)
    try:
        for pos in _get_positions_cached():
            sym = pos['symbol']
            if sym in _pending_fills or sym in _sltp_done:
                continue
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            side_str = pos.get('side', 'LONG')
            if entry > 0 and qty > 0:
                i1h = calc_indicators(get_klines(sym, '1h', 60))
                atr = i1h.get('atr', 0) or 0
                if atr > 0:
                    is_long = 'LONG' in side_str.upper()
                    sl_m, tp_m = (1.5, 5.0) if sym == 'ETHUSDT' else (1.5, 5.0)
                    sl = round(entry - atr * sl_m, 6) if is_long else round(entry + atr * sl_m, 6)
                    tp = round(entry + atr * tp_m, 6) if is_long else round(entry - atr * tp_m, 6)
                    buy_side = 'BUY' if is_long else 'SELL'
                    place_sl_tp(sym, buy_side, qty, sl_price=sl, tp_price=tp)
                    _sltp_done.add(sym)
                    _tp_cache[sym] = tp
                    log(f"  🔧 {sym} SL/TP 보완: SL ${sl} TP ${tp}")
    except Exception as e:
        log(f"  SL/TP 보완 오류: {e}")


def check_partial_tp():
    """수익 50% 도달 시 절반 청산 + SL 본절 이동"""
    try:
        client = get_client()
        for pos in _get_positions_cached():
            sym = pos['symbol']
            if sym in _partial_done:
                continue
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            side = pos.get('side', 'LONG')
            if entry <= 0 or qty <= 0:
                continue

            cur = get_price(sym)
            is_long = 'LONG' in side.upper()

            # TP 캐시에서 가져오기 (DB 조회 불필요)
            if sym not in _tp_cache:
                continue
            tp = float(_tp_cache[sym])

            # TP까지의 거리 대비 현재 수익률
            tp_dist = abs(tp - entry)
            if tp_dist <= 0:
                continue
            if is_long:
                progress = (cur - entry) / tp_dist
            else:
                progress = (entry - cur) / tp_dist

            if progress >= 0.5:
                # 절반 청산
                half_qty = _round_qty(sym, qty / 2)
                if half_qty <= 0:
                    continue
                close_side = 'SELL' if is_long else 'BUY'
                try:
                    client.futures_create_order(
                        symbol=sym, side=close_side, type='MARKET',
                        quantity=half_qty, reduceOnly=True)
                    _positions_cache['ts'] = 0  # 부분 청산 후 캐시 즉시 무효화
                    log(f"  💰 {sym} 부분 익절 {half_qty} (진행률 {progress*100:.0f}%)")

                    # SL을 본절(진입가)로 이동
                    remain = _round_qty(sym, qty - half_qty)
                    if remain > 0:
                        entry_side = 'BUY' if is_long else 'SELL'
                        # 기존 SL/TP 취소 후 재배치
                        try:
                            client.futures_cancel_all_open_orders(symbol=sym)
                            time.sleep(0.3)
                        except Exception:
                            pass
                        # 본절 SL + 기존 TP 유지
                        place_sl_tp(sym, entry_side, remain, sl_price=entry, tp_price=tp)
                        log(f"  🔒 {sym} SL → 본절 ${entry}, 잔여 {remain}")

                    _partial_done.add(sym)

                    try:
                        pnl_pct = progress * abs(tp - entry) / entry * 100
                        send_message(TG_TOKEN, TG_CHAT,
                            f"💰 <b>{sym} 부분 익절!</b>\n"
                            f"   절반 청산 {half_qty} (수익 ~{pnl_pct:.1f}%)\n"
                            f"   SL → 본절 ${entry}")
                    except Exception:
                        pass
                except Exception as e:
                    log(f"  ⚠️ {sym} 부분 익절 실패: {e}")
    except Exception as e:
        log(f"  부분 익절 체크 오류: {e}")


def _get_trail_distance(pnl_pct_1x, atr_pct):
    """ATR 비례 + 수익 단계별 트레일 거리 (1x % 반환)
    atr_pct: ATR / 진입가 × 100 (1x 기준 %)"""
    for threshold, atr_mult in TRAIL_ATR_TIERS:
        if pnl_pct_1x >= threshold:
            dist = atr_pct * atr_mult
            # 최소/최대 캡
            return max(TRAIL_MIN_PCT, min(TRAIL_MAX_PCT, dist))
    # 기본
    dist = atr_pct * TRAIL_ATR_TIERS[-1][1]
    return max(TRAIL_MIN_PCT, min(TRAIL_MAX_PCT, dist))


def check_trailing_stop():
    """트레일링 스탑: 수익 단계별 + 레버리지 반영, 30초 빠른 체크 지원"""
    try:
        client = get_client()
        for pos in _get_positions_cached():
            sym = pos['symbol']
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            side = pos.get('side', 'LONG')
            if entry <= 0 or qty <= 0:
                continue

            cur = get_price(sym)
            is_long = 'LONG' in side.upper()
            is_major = sym in ('ETHUSDT', 'BTCUSDT')
            lev = 3 if is_major else 2

            # 수익률 계산 (1x 기준)
            if is_long:
                pnl_pct = (cur - entry) / entry * 100
            else:
                pnl_pct = (entry - cur) / entry * 100

            # 트레일링 활성화 전이면 피크 업데이트만
            if pnl_pct < TRAIL_ACTIVATE_PCT:
                _trail_peak.pop(sym, None)
                continue

            # ATR 캐시 (활성화 시 1회 조회, 이후 재사용)
            if sym not in _trail_atr:
                try:
                    _kl = client.futures_klines(symbol=sym, interval='1h', limit=20)
                    _trs = []
                    for _i in range(1, len(_kl)):
                        _h, _l, _pc = float(_kl[_i][2]), float(_kl[_i][3]), float(_kl[_i-1][4])
                        _trs.append(max(_h-_l, abs(_h-_pc), abs(_l-_pc)))
                    _trail_atr[sym] = sum(_trs[-14:]) / 14 if _trs else entry * 0.02
                except Exception:
                    _trail_atr[sym] = entry * 0.02  # 폴백: 가격의 2%

            # 피크 가격 업데이트
            if sym not in _trail_peak:
                _trail_peak[sym] = cur
                atr_val = _trail_atr[sym]
                atr_pct = atr_val / entry * 100
                real_pnl = pnl_pct * lev
                _dist = _get_trail_distance(pnl_pct, atr_pct)
                log(f"  📈 {sym} 트레일링 활성화 (수익 {pnl_pct:.1f}%×{lev}={real_pnl:.1f}% | ATR={atr_pct:.1f}% 거리={_dist:.2f}%)")
            else:
                if is_long:
                    _trail_peak[sym] = max(_trail_peak[sym], cur)
                else:
                    _trail_peak[sym] = min(_trail_peak[sym], cur)

            # 고점/저점 대비 하락/상승 체크
            peak = _trail_peak[sym]
            if is_long:
                drop_pct = (peak - cur) / peak * 100
            else:
                drop_pct = (cur - peak) / peak * 100

            # ATR 비례 + 수익 단계별 트레일 거리
            atr_pct = _trail_atr.get(sym, entry * 0.02) / entry * 100
            trail_dist = _get_trail_distance(pnl_pct, atr_pct)

            if drop_pct >= trail_dist:
                # 트레일링 스탑 발동 → 전량 마켓 청산
                close_side = 'SELL' if is_long else 'BUY'
                try:
                    client.futures_create_order(
                        symbol=sym, side=close_side, type='MARKET',
                        quantity=qty, reduceOnly=True)
                    _positions_cache['ts'] = 0
                    real_pnl = pnl_pct * lev
                    log(f"  🔔 {sym} 트레일링 스탑! 고점 대비 {drop_pct:.1f}% 하락 → 청산 (수익 {pnl_pct:.1f}% 1x = {real_pnl:.1f}% 실제, 거리={trail_dist:.1f}%)")
                    _trail_peak.pop(sym, None)
                    _trail_atr.pop(sym, None)
                    try:
                        send_message(TG_TOKEN, TG_CHAT,
                            f"🔔 <b>{sym} 트레일링 스탑</b>\n"
                            f"   고점 대비 {drop_pct:.1f}% 하락 (거리={trail_dist:.2f}%) → 청산\n"
                            f"   수익 ~{real_pnl:.1f}% ({lev}x)")
                    except:
                        pass
                except Exception as e:
                    log(f"  ⚠️ {sym} 트레일링 청산 실패: {e}")
    except Exception as e:
        log(f"  트레일링 체크 오류: {e}")


def _round_qty(symbol, qty):
    """수량을 심볼 규격에 맞게 반올림"""
    try:
        from math import floor
        step, _ = _get_symbol_filters(symbol)
        step = float(step)
        return floor(qty / step) * step
    except Exception:
        return round(qty, 3)


def update_cycle():
    """자동 스캔 → 상위 종목 선택 → 주문 업데이트"""
    bal = get_balance()
    log(f"--- 사이클 | 잔고 ${bal['total']:.2f} (가용 ${bal['available']:.2f}) ---")

    held = get_held_symbols()
    held_count = len(held)

    # 시그널 큐 우선 처리 (공지/급등/펀딩비 등 외부 감지기 시그널)
    _signals = pop_signals()
    _signal_syms = set()
    if _signals:
        for sig in _signals:
            _signal_syms.add(sig['symbol'])
            log(f"  [시그널큐] {sig['symbol']} ← {sig['source']} (우선순위={sig['priority']})")

    # 전체 스캔
    btc_up, btc_rsi = get_btc_trend()
    log(f"  BTC: {'UP' if btc_up else 'DOWN'} RSI={btc_rsi:.0f}")

    scan_list = get_scan_universe()
    # 시그널 큐 종목을 스캔 리스트 맨 앞에 추가 (중복 제거)
    if _signal_syms:
        scan_list = list(_signal_syms - set(scan_list)) + scan_list
    log(f"  스캔 대상: {len(scan_list)}종목" + (f" (시그널 우선: {list(_signal_syms)})" if _signal_syms else ""))

    # 야간 진입 차단 (KST 00~04시)
    kst_hour = (datetime.now(tz=timezone.utc).hour + 9) % 24
    is_night = kst_hour in NIGHT_HOURS

    # 경제 이벤트 고변동 시간대 주의 (UTC 18:00=KST 03:00 FOMC, UTC 12:30=KST 21:30 CPI 등)
    # 정각 전후 5분은 변동성 급증 가능 → 신규 진입 억제 (기존 포지션은 유지)
    utc_min = datetime.now(tz=timezone.utc).minute
    is_event_risk = utc_min <= 5 or utc_min >= 55  # 정각 ±5분

    scored = []
    for sym in scan_list:
        if sym in held:
            log(f"  {sym}: 포지션 보유 → 스킵")
            continue
        if sym in BLACKLIST:
            continue
        if sym in _cooldown:
            if time.time() < _cooldown[sym]:
                continue
            else:
                del _cooldown[sym]  # 만료된 쿨다운 정리
        a = score_symbol(sym)
        if a.get('error'):
            continue
        scored.append(a)
        time.sleep(0.2)  # API 레이트 리밋

    # 점수 절대값 기준 내림차순 (방향 있는 것만)
    candidates = [a for a in scored if a['direction'] != 'wait']
    candidates.sort(key=lambda x: abs(x['score']), reverse=True)

    # 동일 방향 집중 제한 — 현재 포지션 + 대기 주문의 방향 카운트
    long_count = sum(1 for s in held if _get_held_direction(s) == 'long')
    short_count = sum(1 for s in held if _get_held_direction(s) == 'short')
    # 대기 주문도 카운트
    for pf in _pending_fills.values():
        if pf.get('side') == 'BUY': long_count += 1
        elif pf.get('side') == 'SELL': short_count += 1
    if long_count >= MAX_SAME_DIR:
        candidates = [c for c in candidates if c['direction'] != 'long']
        log(f"  ⚖️ 롱 {long_count}개 → 추가 롱 차단")
    if short_count >= MAX_SAME_DIR:
        candidates = [c for c in candidates if c['direction'] != 'short']
        log(f"  ⚖️ 숏 {short_count}개 → 추가 숏 차단")

    # 일일 거래 횟수 제한
    today = datetime.now().strftime('%Y-%m-%d')
    if _daily_trades['date'] != today:
        _daily_trades['date'] = today
        _daily_trades['count'] = 0
    daily_limit_hit = _daily_trades['count'] >= MAX_DAILY_TRADES

    # 연속 손실 시 진입금 축소 (2연패 50%, 3연패+ 진입 중단)
    global _consecutive_losses
    if _consecutive_losses >= 3:
        loss_block = True
    else:
        loss_block = False

    if is_night:
        log(f"  🌙 야간 {kst_hour}시 → 신규 진입 차단")
        candidates = []
    elif is_event_risk:
        log(f"  ⏰ 정각 전후 5분 → 진입 보류")
        candidates = []
    elif daily_limit_hit:
        log(f"  🛑 일일 거래 한도 {MAX_DAILY_TRADES}건 도달 → 진입 중단")
        candidates = []
    elif loss_block:
        log(f"  🛑 연속 {_consecutive_losses}패 → 진입 중단 (다음 사이클까지 대기)")
        _consecutive_losses = 0  # 리셋 (1사이클 쉬고 재개)
        candidates = []

    # 전체 스캔 결과 로그
    for a in scored:
        flag = '🟢' if a['direction'] == 'long' else ('🔴' if a['direction'] == 'short' else '⚪')
        sb = a.get('sentiment_bonus', 0)
        sb_str = f" S{sb:+d}" if sb != 0 else ""
        log(f"  {flag} {a['symbol']:12s} ${a.get('price',0):.4f} 점수={a['score']:+3d}{sb_str} RSI={a.get('rsi',0):.0f} ADX={a.get('adx',0):.0f}")

    # 알트 상관관계 필터: 같은 방향 후보가 3개+ 이면 점수 상위 2개만
    long_cands = [c for c in candidates if c['direction'] == 'long']
    short_cands = [c for c in candidates if c['direction'] == 'short']
    if len(long_cands) > 2:
        long_cands = long_cands[:2]
        log(f"  🔗 롱 후보 {len([c for c in candidates if c['direction']=='long'])}개 → 상관관계 필터 → 상위 2개만")
    if len(short_cands) > 2:
        short_cands = short_cands[:2]
        log(f"  🔗 숏 후보 {len([c for c in candidates if c['direction']=='short'])}개 → 상관관계 필터 → 상위 2개만")
    candidates = sorted(long_cands + short_cands, key=lambda x: abs(x['score']), reverse=True)

    # 상위 N개 선택 (점수 절대값 내림차순)
    client = get_client()
    top_n = candidates[:MAX_ORDERS - held_count] if candidates else []
    top_syms = {c['symbol'] for c in top_n}

    # 기존 LIMIT 주문 중 상위 N에 없는 것 정리 (SL/TP는 절대 건드리지 않음)
    try:
        open_orders = client.futures_get_open_orders()
        for o in open_orders:
            osym = o['symbol']
            otype = o.get('type', '')
            # SL/TP 주문은 보호 (STOP_MARKET, TAKE_PROFIT_MARKET)
            if otype in ('STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP', 'TAKE_PROFIT'):
                continue
            if osym not in top_syms and osym not in held:
                try:
                    client.futures_cancel_order(symbol=osym, orderId=o['orderId'])
                    log(f"  {osym}: 상위 탈락 → LIMIT 주문 취소")
                    _pending_fills.pop(osym, None)
                except Exception:
                    pass
    except Exception:
        pass

    # 슬롯 계산
    slots = MAX_ORDERS - held_count
    log(f"  슬롯: {slots}개 (보유 {held_count}/{MAX_ORDERS}) | 후보: {[c['symbol']+'('+str(c['score'])+')' for c in top_n]}")
    if slots <= 0:
        log(f"  슬롯 없음")
    else:
        for a in top_n:
            sym = a['symbol']
            try:
                entry, sl, tp = calc_entry(a)
                if entry is None:
                    log(f"  ⚠️ {sym}: 진입가 계산 실패 (ATR=0?)")
                    continue

                is_eth = sym == 'ETHUSDT'
                base_usdt = ETH_USDT if is_eth else ALT_USDT
                lev = ETH_LEV if is_eth else ALT_LEV
                # 점수 비례 사이징: |점수| 5+ → 1.5배, 8+ → 2배
                # 점수 비례 사이징
                abs_sc = abs(a['score'])
                if abs_sc >= 8: size_mult = 2.0
                elif abs_sc >= 5: size_mult = 1.5
                else: size_mult = 1.0
                # 연속 손실 시 진입금 축소 (2연패 → 50%)
                if _consecutive_losses >= 2:
                    size_mult *= 0.5
                    log(f"  ⚠️ 연속 {_consecutive_losses}패 → 진입금 50% 축소")
                usdt = min(base_usdt * size_mult, ALT_USDT_MAX / lev)
                side = 'BUY' if a['direction'] == 'long' else 'SELL'

                r = place_limit_only(sym, side, usdt, entry, lev)
                if r['success']:
                    _pending_fills[sym] = {
                        'order_id': r['order_id'], 'sl': sl, 'tp': tp,
                        'side': side, 'qty': r['qty'], 'entry': entry,
                        'score': a.get('score', 0), 'atr': a.get('atr_1h', 0),
                    }
                    dist = abs(entry - a['price']) / a['price'] * 100
                    sl_pct = abs(entry - sl) / entry * 100
                    tp_pct = abs(tp - entry) / entry * 100
                    log(f"  📋 {sym} {'LONG' if a['direction']=='long' else 'SHORT'} @ ${entry} ({dist:.1f}%) SL({sl_pct:.1f}%) TP({tp_pct:.1f}%) R:R 1:{tp_pct/sl_pct:.1f}")
                else:
                    log(f"  ❌ {sym}: {r.get('error','')[:60]}")
            except Exception as e:
                log(f"  ❌ {sym} 주문 예외: {str(e)[:80]}")
            time.sleep(0.5)

    # 텔레그램 보고 (30분마다만 전송, 체결 알림은 별도)
    global _last_tg_time
    now = time.time()
    if now - _last_tg_time >= TG_INTERVAL:
        _last_tg_time = now
        try:
            ts = datetime.now().strftime("%H:%M")
            lines = [f"<b>📊 자동 스캔</b> ({ts})", f"💰 ${bal['total']:.2f} | BTC {'↑' if btc_up else '↓'}", ""]

            for a in scored:
                if a['direction'] != 'wait':
                    e = '🟢' if a['direction'] == 'long' else '🔴'
                    d = '롱' if a['direction'] == 'long' else '숏'
                    lines.append(f"{e} <b>{a['symbol']}</b> ${a.get('price',0):.4f} {d}({a['score']:+d}) RSI {a.get('rsi',0):.0f}")

            pending_syms = list(_pending_fills.keys())
            if pending_syms:
                lines.append("")
                lines.append(f"📋 예약: {', '.join(pending_syms)}")

            if held:
                lines.append(f"✅ 보유: {', '.join(held)}")

            waits = [a for a in scored if a['direction'] == 'wait']
            if waits:
                names = ', '.join(a['symbol'].replace('USDT','') for a in waits[:6])
                lines.append(f"⏸ 관망: {names}")

            send_message(TG_TOKEN, TG_CHAT, "\n".join(lines))
        except Exception as e:
            log(f"  TG 전송 실패: {e}")

    # 관리 종목 파일 갱신 (app.py 충돌 방지)
    import json
    managed = list(held | set(_pending_fills.keys()) | top_syms)
    try:
        with open(LOCK_PATH, 'w') as f:
            json.dump({"symbols": managed, "ts": datetime.now().isoformat()}, f)
    except Exception:
        pass


def check_extreme_funding():
    """펀딩비 극단값 감지 → 역발상 시그널 큐 전달 (사이클당 1회)"""
    try:
        client = get_client()
        # 거래량 상위 10개만 빠르게 체크
        tickers = client.futures_ticker()
        tickers = [t for t in tickers if t['symbol'].endswith('USDT') and t['symbol'] not in BLACKLIST]
        tickers.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        top_syms = [t['symbol'] for t in tickers[:10]]

        held = get_held_symbols()
        from signal_queue import push_signal

        for sym in top_syms:
            if sym in held:
                continue
            try:
                fr = client.futures_funding_rate(symbol=sym, limit=1)
                if not fr:
                    continue
                rate = float(fr[-1]['fundingRate'])
                # 극단 펀딩비: ±0.1% 이상 → 역발상 시그널
                if rate >= 0.001:  # +0.1% 이상 → 롱 과열 → 숏 (현재 롱전용이므로 스킵)
                    log(f"  [펀딩] {sym} 펀딩={rate*100:+.3f}% (롱과열 → 스킵, 롱전용)")
                elif rate <= -0.001:  # -0.1% 이상 → 숏 과열 → 롱 역발상
                    push_signal(sym, 'funding', direction='long', priority=3,
                                meta={'funding_rate': round(rate * 100, 4)})
                    log(f"  [펀딩] {sym} 펀딩={rate*100:+.3f}% (숏과열 → 롱 역발상 시그널)")
            except Exception:
                pass
            time.sleep(0.1)
    except Exception as e:
        log(f"  펀딩비 감지 오류: {e}")


_last_daily_report = ""

def daily_summary():
    """매일 23시 일일 성과 텔레그램 발송"""
    global _last_daily_report
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.hour != 23 or _last_daily_report == today:
        return
    _last_daily_report = today
    try:
        bal = get_balance()
        trades = trade_db.get_all_trades(limit=50)
        today_trades = [t for t in trades if (t.get("time") or "").startswith(today)]
        today_pnl = sum(t.get("pnl") or 0 for t in today_trades if t.get("pnl") is not None)
        wins = len([t for t in today_trades if (t.get("pnl") or 0) > 0])
        losses = len([t for t in today_trades if (t.get("pnl") or 0) < 0])
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        pos = _get_positions_cached()
        pos_lines = []
        for p in pos:
            upnl = float(p.get('unrealized_pnl', 0))
            pos_lines.append(f"  {p['symbol']} {p['side']} uPnL ${upnl:.2f}")

        lines = [
            f"<b>🌙 일일 마감 리포트</b> ({today})",
            f"💰 잔고: ${bal['total']:.2f}",
            f"📊 오늘: {len(today_trades)}건 | 승 {wins} 패 {losses} | 승률 {wr:.0f}%",
            f"💵 실현 PnL: ${today_pnl:+.2f}",
        ]
        if pos_lines:
            lines.append(f"📌 보유 포지션:")
            lines.extend(pos_lines)
        lines.append(f"\n전체 누적: {len(trades)}건 | PnL ${sum(t.get('pnl',0) or 0 for t in trades if t.get('pnl')):.2f}")

        send_message(TG_TOKEN, TG_CHAT, "\n".join(lines))
        log(f"  🌙 일일 마감 리포트 발송")
    except Exception as e:
        log(f"  일일 리포트 오류: {e}")


_funding_cycle = 0  # 펀딩비 체크 사이클 카운터

def main():
    global _funding_cycle
    log("=" * 60)
    log(f"포지션 업데이터 시작 (5분 스캔, 최대 {MAX_ORDERS}슬롯, 롱≥{MIN_SCORE} 숏≤{-MIN_SCORE} SL×1.5 TP×5.0)")
    log(f"  시그널 큐 활성 | 펀딩비 감지 30분 주기 | 스캔 {SCAN_ALT_COUNT}종목 | 일일 {MAX_DAILY_TRADES}건")
    log("=" * 60)

    while True:
        try:
            _positions_cache['ts'] = 0  # 사이클 시작 시 캐시 리셋
            check_fills()
            check_trailing_stop()
            check_partial_tp()
            # 청산된 종목 → 쿨다운 등록 + 연속 손실 추적
            _held = get_held_symbols()
            for sym in (_sltp_done - _held):
                _cooldown[sym] = time.time() + COOLDOWN_SEC
                # DB에서 마지막 거래 PnL 확인 → 연속 손실 추적
                try:
                    import sqlite3 as _sq
                    _db = _sq.connect('/home/hyeok/01.APCC/00.ai-lab/trades.db')
                    _row = _db.execute(
                        "SELECT pnl FROM trades WHERE symbol=? AND pnl IS NOT NULL ORDER BY id DESC LIMIT 1",
                        (sym,)).fetchone()
                    _db.close()
                    if _row and _row[0] is not None:
                        if _row[0] <= 0:
                            _consecutive_losses += 1
                            log(f"  ⏳ {sym} 손절 → 연속 {_consecutive_losses}패 | {COOLDOWN_SEC//60}분 쿨다운")
                        else:
                            _consecutive_losses = 0
                            log(f"  ⏳ {sym} 익절 → 연패 리셋 | {COOLDOWN_SEC//60}분 쿨다운")
                    else:
                        log(f"  ⏳ {sym} 청산 → {COOLDOWN_SEC//60}분 쿨다운")
                except Exception:
                    log(f"  ⏳ {sym} 청산 → {COOLDOWN_SEC//60}분 쿨다운")
            _partial_done.difference_update(_partial_done - _held)
            _sltp_done.difference_update(_sltp_done - _held)
            for sym in list(_tp_cache):
                if sym not in _held:
                    del _tp_cache[sym]
            for sym in list(_trail_peak):
                if sym not in _held:
                    del _trail_peak[sym]
            for sym in list(_trail_atr):
                if sym not in _held:
                    del _trail_atr[sym]
            update_cycle()
            # 펀딩비 극단값 체크 (6사이클=30분마다)
            _funding_cycle += 1
            if _funding_cycle % 6 == 0:
                check_extreme_funding()
            daily_summary()
        except Exception as e:
            log(f"사이클 오류: {e}")
        log(f"--- 다음: {INTERVAL}초 후 (트레일링 30초 체크) ---")
        # 5분 대기 중 30초마다 트레일링 스탑 빠른 체크
        _wait_start = time.time()
        while time.time() - _wait_start < INTERVAL:
            time.sleep(30)
            try:
                _positions_cache['ts'] = 0  # 캐시 리셋
                check_trailing_stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()

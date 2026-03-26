#!/usr/bin/env python3
# 밑꼬리 반등 감지기 — 1분봉 실시간 감시 → 스탑헌팅 V자 반등 시 즉시 시장가 롱 진입
# 대상: ETHUSDT, BTCUSDT
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import time
import json
from datetime import datetime
from binance_client import get_client, get_klines, get_price, get_balance, _get_symbol_filters, _round_price
from indicators import calc_indicators
from telegram_notifier import send_message
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

# ── 설정 ──
SYMBOLS = ['ETHUSDT', 'BTCUSDT']
CHECK_INTERVAL = 10       # 10초마다 체크
WICK_RATIO = 0.50         # 밑꼬리가 전체 레인지의 50% 이상 (#6: 60→50, 감지 0건→완화)
VOL_MULT = 3.0            # 거래량이 평균 대비 3배 이상 (#6: 5→3)
MIN_DROP_PCT = 0.2        # 최소 낙폭 0.2% (#6: 0.3→0.2)
MAX_CLOSE_DROP_PCT = 0.2  # 종가가 시가 대비 -0.2% 이내 (회복 확인)
RSI_MAX = 80              # 15분 RSI 80 이상이면 과매수 → 스킵
COOLDOWN_SEC = 1800       # 동일 종목 30분 쿨다운
MAX_POSITIONS = 2         # 이 전략으로 최대 동시 보유
SL_BUFFER_PCT = 0.1       # SL = 꼬리 저점 - 0.1%
TP_ATR_MULT = 3.0         # TP = ATR × 3 (빠른 익절)
TP_MIN_PCT = 2.0          # TP 최소 2%
LIMIT_OFFSET_PCT = 0.05   # 꼬리 저점 대비 0.05% 아래에 지정가
LIMIT_EXPIRE_SEC = 300    # 미체결 5분 후 자동 취소

# 진입금
USDT_MAP = {'ETHUSDT': 20, 'BTCUSDT': 20}
LEV_MAP = {'ETHUSDT': 3, 'BTCUSDT': 3}

TG_TOKEN = TELEGRAM_TOKEN
TG_CHAT = TELEGRAM_CHAT_ID
LOG_PATH = '/home/hyeok/01.APCC/00.ai-lab/wick_detector.log'

# 상태
_cooldown = {}        # {symbol: expire_timestamp}
_positions = set()    # 이 전략으로 진입한 종목
_last_candle = {}     # {symbol: last_processed_open_time}
_vol_avg = {}         # {symbol: 20봉 평균 거래량 캐시}
_vol_avg_ts = {}      # {symbol: 캐시 시각}
_exchange_info_cache = {'ts': 0, 'data': None}  # exchange_info 캐시 (1시간)
_pending_limit = {}   # {symbol: {order_id, qty, limit_px, sl, tp, placed_at, price_prec}}

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, 'a') as f:
            f.write(line + '\n')
    except:
        pass

def tg(msg):
    try:
        send_message(TG_TOKEN, TG_CHAT, msg)
    except:
        pass

def get_vol_avg(client, symbol, period=20):
    """1분봉 20개 평균 거래량 (60초 캐시)"""
    now = time.time()
    if symbol in _vol_avg and now - _vol_avg_ts.get(symbol, 0) < 60:
        return _vol_avg[symbol]
    klines = client.futures_klines(symbol=symbol, interval='1m', limit=period + 2)
    # 마지막 봉은 미완성이므로 제외
    vols = [float(k[5]) for k in klines[:-1]][-period:]
    avg = sum(vols) / len(vols) if vols else 1
    _vol_avg[symbol] = avg
    _vol_avg_ts[symbol] = now
    return avg

def get_rsi_15m(symbol):
    """15분 RSI (과매수 필터용)"""
    try:
        klines = get_klines(symbol, '15m', 30)
        ind = calc_indicators(klines)
        return ind.get('rsi', 50) or 50
    except:
        return 50

def get_atr_1h(client, symbol):
    """1시간 ATR(14)"""
    try:
        klines = client.futures_klines(symbol=symbol, interval='1h', limit=20)
        trs = []
        for i in range(1, len(klines)):
            h, l, pc = float(klines[i][2]), float(klines[i][3]), float(klines[i - 1][4])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs[-14:]) / 14
    except:
        return 0

def check_wick(client, symbol):
    """최근 완성된 1분봉에서 밑꼬리 패턴 감지"""
    klines = client.futures_klines(symbol=symbol, interval='1m', limit=3)
    if len(klines) < 2:
        return None

    # 마지막 완성 봉 (현재 봉은 미완성)
    candle = klines[-2]
    open_time = candle[0]

    # 이미 처리한 캔들이면 스킵
    if _last_candle.get(symbol) == open_time:
        return None
    _last_candle[symbol] = open_time

    o, h, l, c, vol = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])
    total_range = h - l
    if total_range <= 0:
        return None

    # 밑꼬리 계산
    body_low = min(o, c)
    lower_wick = body_low - l
    wick_ratio = lower_wick / total_range

    # 낙폭 확인 (고가 기준)
    drop_pct = (h - l) / h * 100

    # 종가 회복 확인 (시가 대비)
    close_drop_pct = (o - c) / o * 100 if o > 0 else 0

    # 거래량 비교
    vol_avg = get_vol_avg(client, symbol)
    vol_ratio = vol / vol_avg if vol_avg > 0 else 0

    # 모든 조건 체크
    if (wick_ratio >= WICK_RATIO and
        drop_pct >= MIN_DROP_PCT and
        close_drop_pct <= MAX_CLOSE_DROP_PCT and
        vol_ratio >= VOL_MULT):

        ts = datetime.fromtimestamp(open_time / 1000).strftime('%H:%M')
        return {
            'symbol': symbol,
            'time': ts,
            'open': o, 'high': h, 'low': l, 'close': c,
            'wick_ratio': wick_ratio,
            'drop_pct': drop_pct,
            'vol_ratio': vol_ratio,
            'volume': vol,
        }
    return None

def enter_position(client, signal):
    """시장가 롱 진입 + SL/TP"""
    sym = signal['symbol']

    # 쿨다운 체크
    if sym in _cooldown and time.time() < _cooldown[sym]:
        remain = int(_cooldown[sym] - time.time())
        log(f"  {sym}: 쿨다운 {remain}초 남음 → 스킵")
        return False

    # 전체 포지션 수 체크 (position_updater와 합산)
    all_positions = client.futures_position_information()
    open_count = sum(1 for p in all_positions if float(p['positionAmt']) != 0)
    if open_count >= 5:  # 전체 계좌 최대 5포지션 (updater 4 + wick 1 여유)
        log(f"  전체 포지션 {open_count}개 → 슬롯 없음")
        return False

    # 이미 해당 종목 포지션 보유 체크
    for p in all_positions:
        if p['symbol'] == sym and float(p['positionAmt']) != 0:
            log(f"  {sym}: 이미 포지션 보유 → 스킵")
            return False

    # 15분 RSI 과매수 필터
    rsi = get_rsi_15m(sym)
    if rsi >= RSI_MAX:
        log(f"  {sym}: RSI {rsi:.0f} >= {RSI_MAX} 과매수 → 스킵")
        return False

    # 레버리지 설정
    lev = LEV_MAP.get(sym, 2)
    try:
        client.futures_change_leverage(symbol=sym, leverage=lev)
    except:
        pass

    # 수량 계산
    px = float(client.futures_symbol_ticker(symbol=sym)['price'])
    usdt = USDT_MAP.get(sym, 8)
    notional = usdt * lev

    # exchange_info 캐시 (1시간)
    now_t = time.time()
    if now_t - _exchange_info_cache['ts'] > 3600 or _exchange_info_cache['data'] is None:
        _exchange_info_cache['data'] = client.futures_exchange_info()
        _exchange_info_cache['ts'] = now_t
    info = _exchange_info_cache['data']
    sym_info = [s for s in info['symbols'] if s['symbol'] == sym][0]
    price_prec = int(sym_info['pricePrecision'])
    qty_prec = int(sym_info['quantityPrecision'])
    qty = round(notional / px, qty_prec)

    if qty <= 0:
        log(f"  {sym}: 수량 0 → 스킵")
        return False

    # 꼬리 저점 아래에 지정가 배치 (더 보수적 진입)
    limit_px = round(signal['low'] * (1 - LIMIT_OFFSET_PCT / 100), price_prec)

    # 현재가보다 높으면 의미 없음 (이미 반등 완료) → 현재가 기준 재계산
    if limit_px >= px:
        limit_px = round(px * 0.998, price_prec)  # 현재가 -0.2%

    # SL/TP 미리 계산 (체결 시 배치)
    sl_wick = signal['low'] * (1 - SL_BUFFER_PCT / 100)
    max_sl_1x = 5.0 / lev
    sl_cap = limit_px * (1 - max_sl_1x / 100)
    sl = round(max(sl_wick, sl_cap), price_prec)

    atr = get_atr_1h(client, sym)
    tp_atr = limit_px + atr * TP_ATR_MULT
    tp_min = limit_px * (1 + TP_MIN_PCT / 100)
    tp = round(max(tp_atr, tp_min), price_prec)

    # 지정가 주문
    try:
        order = client.futures_create_order(
            symbol=sym, side='BUY', type='LIMIT',
            price=str(limit_px), quantity=str(qty),
            timeInForce='GTC'
        )
        order_id = order['orderId']
        log(f"  {sym}: LIMIT BUY @ {limit_px} (꼬리저점 {signal['low']} 아래) ID={order_id}")
    except Exception as e:
        log(f"  {sym}: 지정가 실패 — {e}")
        return False

    # 체결 대기 큐에 등록 (SL/TP는 체결 후 배치)
    _pending_limit[sym] = {
        'order_id': order_id,
        'qty': qty,
        'limit_px': limit_px,
        'sl': sl,
        'tp': tp,
        'placed_at': time.time(),
        'price_prec': price_prec,
        'signal': signal,
        'rsi': rsi,
        'atr': atr,
        'lev': lev,
    }

    _cooldown[sym] = time.time() + COOLDOWN_SEC

    sl_pct = (sl - limit_px) / limit_px * 100
    tp_pct = (tp - limit_px) / limit_px * 100

    tg(
        f"🕯️ <b>WICK LIMIT</b>\n"
        f"{sym} LIMIT BUY @ ${limit_px:.2f} (꼬리저점 {signal['low']:.2f} 아래)\n"
        f"밑꼬리 {signal['wick_ratio']:.0%} | 낙폭 {signal['drop_pct']:.2f}% | 거래량 {signal['vol_ratio']:.1f}x\n"
        f"체결 시 → SL ${sl:.2f} ({sl_pct:+.2f}%) TP ${tp:.2f} ({tp_pct:+.2f}%)\n"
        f"RSI(15m): {rsi:.0f} | 5분 미체결 시 자동 취소"
    )
    log(f"  대기: {sym} LIMIT @ {limit_px} | SL {sl} | TP {tp} | 5분 후 취소")
    return True


def check_pending_limits(client):
    """지정가 체결 확인 → SL/TP 배치 / 5분 미체결 → 취소"""
    now = time.time()
    done = []

    for sym, info in list(_pending_limit.items()):
        order_id = info['order_id']
        elapsed = now - info['placed_at']

        try:
            order = client.futures_get_order(symbol=sym, orderId=order_id)
            status = order['status']
        except Exception as e:
            log(f"  {sym}: 주문 조회 실패 — {e}")
            if elapsed > LIMIT_EXPIRE_SEC:
                done.append(sym)
            continue

        if status == 'FILLED':
            # 체결! SL/TP 배치
            ep = float(order['avgPrice']) or info['limit_px']
            sl = info['sl']
            tp = info['tp']
            qty = info['qty']
            price_prec = info['price_prec']

            # 체결가 기준 SL/TP 재계산
            sl_wick = info['signal']['low'] * (1 - SL_BUFFER_PCT / 100)
            max_sl_1x = 5.0 / info['lev']
            sl_cap = ep * (1 - max_sl_1x / 100)
            sl = round(max(sl_wick, sl_cap), price_prec)
            tp_atr = ep + info['atr'] * TP_ATR_MULT
            tp_min = ep * (1 + TP_MIN_PCT / 100)
            tp = round(max(tp_atr, tp_min), price_prec)

            try:
                client.futures_create_order(
                    symbol=sym, side='SELL', type='STOP_MARKET',
                    stopPrice=str(sl), closePosition='true')
                log(f"  {sym}: 체결 @ {ep} → SL {sl}")
            except Exception as e:
                log(f"  {sym}: SL 배치 실패 — {e}")

            try:
                client.futures_create_order(
                    symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                    stopPrice=str(tp), closePosition='true')
                log(f"  {sym}: TP {tp}")
            except Exception as e:
                log(f"  {sym}: TP 배치 실패 — {e}")

            _positions.add(sym)
            sl_pct = (sl - ep) / ep * 100
            tp_pct = (tp - ep) / ep * 100
            tg(
                f"✅ <b>WICK 체결!</b>\n"
                f"{sym} LONG @ ${ep:.2f}\n"
                f"SL ${sl:.2f} ({sl_pct:+.2f}%) | TP ${tp:.2f} ({tp_pct:+.2f}%)"
            )
            done.append(sym)

        elif status in ('CANCELED', 'EXPIRED', 'REJECTED'):
            log(f"  {sym}: 주문 {status}")
            done.append(sym)

        elif elapsed > LIMIT_EXPIRE_SEC:
            # 5분 초과 미체결 → 취소
            try:
                client.futures_cancel_order(symbol=sym, orderId=order_id)
                log(f"  {sym}: {elapsed:.0f}초 미체결 → 취소")
            except Exception as e:
                log(f"  {sym}: 취소 실패 — {e}")
            done.append(sym)

    for sym in done:
        _pending_limit.pop(sym, None)

def check_exits(client):
    """포지션 청산 감지 → _positions 업데이트"""
    closed = set()
    for sym in list(_positions):
        try:
            pos = client.futures_position_information(symbol=sym)
            has_pos = any(float(p['positionAmt']) != 0 for p in pos)
            if not has_pos:
                closed.add(sym)
        except:
            pass
    for sym in closed:
        _positions.discard(sym)
        log(f"  {sym}: 청산 감지 → 슬롯 반환")

def main():
    log("=" * 50)
    log("밑꼬리 반등 감지기 시작")
    log(f"  대상: {SYMBOLS}")
    log(f"  밑꼬리 {WICK_RATIO:.0%}+ | 거래량 {VOL_MULT}x+ | 낙폭 {MIN_DROP_PCT}%+")
    log(f"  쿨다운 {COOLDOWN_SEC}초 | 최대 {MAX_POSITIONS}포지션")
    log("=" * 50)

    client = get_client()
    cycle = 0

    while True:
        try:
            cycle += 1

            # 10사이클마다 청산 감지
            if cycle % 6 == 0:
                check_exits(client)

            # 지정가 체결 확인 (매 사이클)
            if _pending_limit:
                check_pending_limits(client)

            for sym in SYMBOLS:
                # 이미 지정가 대기 중이면 스킵
                if sym in _pending_limit:
                    continue
                signal = check_wick(client, sym)
                if signal:
                    log(f"밑꼬리 감지! {sym} {signal['time']} | "
                        f"꼬리={signal['wick_ratio']:.0%} 낙폭={signal['drop_pct']:.2f}% "
                        f"거래량={signal['vol_ratio']:.1f}x")
                    enter_position(client, signal)

        except Exception as e:
            err_msg = str(e)
            if '-1003' in err_msg or 'Too many' in err_msg:
                log(f"API 레이트 리밋 → 60초 대기")
                time.sleep(60)
                continue
            log(f"에러: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()

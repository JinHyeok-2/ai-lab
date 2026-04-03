#!/usr/bin/env python3
"""CVD + BB롱 독립 봇 — app.py/Streamlit 의존 없음, 단독 실행"""
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')
import time, json, sqlite3, uuid
from datetime import datetime, timezone
from binance_client import get_client, get_klines, get_price, get_balance
from indicators import calc_indicators
from telegram_notifier import send_message
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
TG_T, TG_C = TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

def tg(msg):
    """텔레그램 알림 (실패해도 무시)"""
    try: send_message(TG_T, TG_C, msg)
    except: pass

# ── 설정 ──
MAX_ORDERS = 1           # 동시 1건
MAX_DAILY = 4            # 일일 4건
COOLDOWN_WIN = 1800      # 익절 후 30분
COOLDOWN_LOSS = 7200     # 손절 후 2시간
INTERVAL = 300           # 5분 주기
BTC_RSI_MIN = 45         # BTC RSI 최소
NIGHT_HOURS = {5}        # KST 05시 차단
CVD_BLOCK_HOURS = set(range(17, 24))  # CVD 17~23시 차단

BLACKLIST = {'STOUSDT','ONTUSDT','BRUSDT','SIRENUSDT','XAUUSDT','XAGUSDT',
             'PAXGUSDT','BSBUSDT','ALPACAUSDT','BNXUSDT','ALPHAUSDT'}
WEAK = {'ETHUSDT','DOGEUSDT','TAOUSDT','HYPEUSDT'}

# ── 상태 ──
_cooldown = {}           # {symbol: expire_time}
_daily = {'date': '', 'count': 0}
_consecutive_losses = 0
_sym_loss_count = {}     # {symbol: 연패}
_sym_banned = set()      # 당일 자동 블랙리스트
_positions = {}          # {symbol: {entry, qty, side, sl, tp, source, time}}

client = get_client()

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_btc_rsi():
    ind = calc_indicators(get_klines('BTCUSDT', '1h', 50))
    return ind.get('rsi', 50) or 50

def get_universe():
    """거래량 상위 30종목 (BLACKLIST/WEAK 제외)"""
    tickers = client.futures_ticker()
    cands = [(t['symbol'], float(t.get('quoteVolume', 0))) for t in tickers
             if t['symbol'].endswith('USDT')
             and t['symbol'] not in BLACKLIST
             and t['symbol'] not in WEAK]
    cands.sort(key=lambda x: -x[1])
    return [s for s, v in cands[:30]]

def has_position(sym):
    return sym in _positions

def can_enter(sym):
    if sym in BLACKLIST or sym in WEAK or sym in _sym_banned:
        return False
    if has_position(sym):
        return False
    if len(_positions) >= MAX_ORDERS:
        return False
    if sym in _cooldown and time.time() < _cooldown[sym]:
        return False
    today = datetime.now().strftime('%Y-%m-%d')
    if _daily.get('date') == today and _daily.get('count', 0) >= MAX_DAILY:
        return False
    kst = (datetime.now(timezone.utc).hour + 9) % 24
    if kst in NIGHT_HOURS:
        return False
    return True

def enter_long(sym, sl, tp, source, usdt=None):
    """롱 시장가 진입 + SL/TP 배치"""
    if not can_enter(sym):
        return False
    try:
        px = float(get_price(sym))
        _b = get_balance()
        bal = float(_b.get('available', _b.get('total', 50))) if isinstance(_b, dict) else float(_b or 50)
        if usdt is None:
            usdt = min(bal * 0.40, 40)  # 잔고 40%, 최대 $40
        lev = 2
        try: client.futures_change_margin_type(symbol=sym, marginType='ISOLATED')
        except: pass
        client.futures_change_leverage(symbol=sym, leverage=lev)

        qty_raw = usdt * lev / px
        # 수량 조정
        info = [s for s in client.futures_exchange_info()['symbols'] if s['symbol']==sym]
        if info:
            step = float([f for f in info[0]['filters'] if f['filterType']=='LOT_SIZE'][0]['stepSize'])
            qty = round(qty_raw / step) * step if step < 1 else int(qty_raw)
        else:
            qty = round(qty_raw, 2)
        if qty * px < 20:
            qty = round(21 / px, 4)
        if qty <= 0:
            return False

        # 시장가 매수
        order = client.futures_create_order(symbol=sym, side='BUY', type='MARKET',
            quantity=str(qty), newClientOrderId=f'nb_{uuid.uuid4().hex[:12]}')
        time.sleep(0.3)

        # SL/TP
        try:
            client.futures_create_order(symbol=sym, side='SELL', type='STOP_MARKET',
                stopPrice=str(round(sl, 6)), quantity=str(qty), reduceOnly='true')
        except: pass
        try:
            client.futures_create_order(symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                stopPrice=str(round(tp, 6)), quantity=str(qty), reduceOnly='true')
        except: pass

        _positions[sym] = {'entry': px, 'qty': qty, 'side': 'LONG', 'sl': sl, 'tp': tp,
                           'source': source, 'time': time.time()}
        today = datetime.now().strftime('%Y-%m-%d')
        if _daily.get('date') != today:
            _daily['date'] = today; _daily['count'] = 0
        _daily['count'] += 1
        _cooldown[sym] = time.time() + COOLDOWN_WIN

        log(f"🟢 {source} 롱 진입: {sym} @ ${px:.4f} qty={qty} SL=${sl:.4f} TP=${tp:.4f}")
        tg(f"🟢 <b>{source} 롱 진입</b>\n{sym} @ ${px:.4f}\nSL ${sl:.4f} → TP ${tp:.4f}\n잔고 ${bal:.2f}")
        return True
    except Exception as e:
        log(f"❌ {sym} 진입 실패: {e}")
        return False

def enter_short(sym, sl, tp, source, usdt=None):
    """숏 시장가 진입 + SL/TP 배치"""
    if not can_enter(sym):
        return False
    try:
        px = float(get_price(sym))
        _b = get_balance()
        bal = float(_b.get('available', _b.get('total', 50))) if isinstance(_b, dict) else float(_b or 50)
        if usdt is None:
            usdt = min(bal * 0.40, 40)
        lev = 2
        try: client.futures_change_margin_type(symbol=sym, marginType='ISOLATED')
        except: pass
        client.futures_change_leverage(symbol=sym, leverage=lev)

        qty_raw = usdt * lev / px
        info = [s for s in client.futures_exchange_info()['symbols'] if s['symbol']==sym]
        if info:
            step = float([f for f in info[0]['filters'] if f['filterType']=='LOT_SIZE'][0]['stepSize'])
            qty = round(qty_raw / step) * step if step < 1 else int(qty_raw)
        else:
            qty = round(qty_raw, 2)
        if qty * px < 20:
            qty = round(21 / px, 4)
        if qty <= 0:
            return False

        order = client.futures_create_order(symbol=sym, side='SELL', type='MARKET',
            quantity=str(qty), newClientOrderId=f'ns_{uuid.uuid4().hex[:12]}')
        time.sleep(0.3)

        try:
            client.futures_create_order(symbol=sym, side='BUY', type='STOP_MARKET',
                stopPrice=str(round(sl, 6)), quantity=str(qty), reduceOnly='true')
        except: pass
        try:
            client.futures_create_order(symbol=sym, side='BUY', type='TAKE_PROFIT_MARKET',
                stopPrice=str(round(tp, 6)), quantity=str(qty), reduceOnly='true')
        except: pass

        _positions[sym] = {'entry': px, 'qty': qty, 'side': 'SHORT', 'sl': sl, 'tp': tp,
                           'source': source, 'time': time.time()}
        today = datetime.now().strftime('%Y-%m-%d')
        if _daily.get('date') != today:
            _daily['date'] = today; _daily['count'] = 0
        _daily['count'] += 1
        _cooldown[sym] = time.time() + COOLDOWN_WIN

        log(f"🔴 {source} 숏 진입: {sym} @ ${px:.4f} qty={qty} SL=${sl:.4f} TP=${tp:.4f}")
        tg(f"🔴 <b>{source} 숏 진입</b>\n{sym} @ ${px:.4f}\nSL ${sl:.4f} → TP ${tp:.4f}\n잔고 ${bal:.2f}")
        return True
    except Exception as e:
        log(f"❌ {sym} 숏 진입 실패: {e}")
        return False

# ── CVD 전략 ──
def check_cvd(btc_rsi, universe):
    if btc_rsi < BTC_RSI_MIN:
        return
    kst = (datetime.now(timezone.utc).hour + 9) % 24
    if kst in CVD_BLOCK_HOURS:
        return

    for sym in universe:
        if not can_enter(sym):
            continue
        try:
            k = client.futures_klines(symbol=sym, interval='1h', limit=48)
            if len(k) < 30:
                continue
            closes = [float(x[4]) for x in k]
            lows = [float(x[3]) for x in k]
            tbv = [float(x[9]) for x in k]  # taker_buy_base

            px = closes[-1]
            # CVD 계산
            cvd = []
            _c = 0
            for j in range(len(tbv)):
                _c += tbv[j] - (float(k[j][5]) - tbv[j])
                cvd.append(_c)

            recent_cvd = cvd[-24:]
            recent_closes = closes[-24:]
            cvd_now = cvd[-1]
            cvd_12h_low = min(recent_cvd)
            price_12h_low = min(lows[-24:])

            # 조건
            _low_dist = (px - price_12h_low) / price_12h_low if price_12h_low > 0 else 999
            near_low = -0.02 < _low_dist < 0.01

            cvd_prev = sum(tbv[-24:-12])
            cvd_recent = sum(tbv[-12:])
            cvd_rising = cvd_recent > cvd_prev * 1.05 if cvd_prev else False

            ind = calc_indicators(get_klines(sym, '1h', 50))
            rsi = ind.get('rsi', 50) or 50
            atr = ind.get('atr', 0) or 0

            if rsi < 35 and near_low and cvd_rising and atr > 0:
                # SL/TP (그리드서치 최적: SL×1.0, TP×4.0)
                atr_pct = atr / px * 100
                sl_pct = max(atr_pct * 1.0, 1.5)
                tp_pct = max(atr_pct * 4.0, sl_pct * 3.0)
                sl = px * (1 - sl_pct / 100)
                tp = px * (1 + tp_pct / 100)

                if enter_long(sym, sl, tp, 'cvd'):
                    return  # 1건만
        except:
            continue

# ── BB 롱 전략 ──
def check_bb(btc_rsi, universe):
    if btc_rsi < 55:  # BB롱은 BTC>55에서만 (그리드서치: PF 0.7→1.0)
        return

    for sym in universe:
        if not can_enter(sym):
            continue
        try:
            ind = calc_indicators(get_klines(sym, '5m', 50))
            px = float(get_price(sym))
            rsi = ind.get('rsi', 50) or 50
            bbu = ind.get('bb_upper', 0) or 0
            bbl = ind.get('bb_lower', 0) or 0
            bbm = ind.get('bb_mid', 0) or 0

            if not (bbu > bbl > 0):
                continue
            bb_pos = (px - bbl) / (bbu - bbl) * 100
            bb_width = (bbu - bbl) / bbm * 100 if bbm else 0

            if bb_pos < 30 and 2.5 < bb_width < 6.0 and 30 < rsi < 55:
                sl = bbl * 0.997
                tp = bbu  # bb_upper
                if tp <= px * 1.005 or sl >= px * 0.995:
                    continue
                if enter_long(sym, sl, tp, 'bb_long'):
                    return
        except:
            continue

# ── 추세 숏 전략 (하락장) ──
def check_trend_short(btc_rsi, universe):
    if btc_rsi > 35:  # BTC < 35에서만 (백테스트 PF 5.48, 837건)
        return

    for sym in universe:
        if not can_enter(sym):
            continue
        try:
            ind = calc_indicators(get_klines(sym, '5m', 50))
            px = float(get_price(sym))
            rsi = ind.get('rsi', 50) or 50
            adx = ind.get('adx', 0) or 0
            atr = ind.get('atr', 0) or 0
            bbu = ind.get('bb_upper', 0) or 0
            bbl = ind.get('bb_lower', 0) or 0
            bbm = ind.get('bb_mid', 0) or 0

            if not (bbu > bbl > 0) or atr <= 0:
                continue
            bb_pos = (px - bbl) / (bbu - bbl) * 100

            if rsi < 40 and bb_pos < 25 and adx > 25:
                # SL: min(BB mid, 진입가 + ATR×1.5)
                sl = min(bbm, px + atr * 1.5)
                # SL 최대 3% 캡
                if (sl - px) / px > 0.03:
                    sl = px * 1.03
                # TP: 진입가 - ATR×2.0
                tp = px - atr * 2.0
                if tp >= px * 0.995 or sl <= px * 1.001:
                    continue

                if enter_short(sym, sl, tp, 'trend_short'):
                    return  # 1건만
        except:
            continue

# ── 포지션 관리 ──
def check_positions():
    """포지션 청산 감지 + 강제 청산"""
    global _consecutive_losses
    closed = []
    for sym, pos in list(_positions.items()):
        try:
            api_pos = [p for p in client.futures_account()['positions']
                       if p['symbol'] == sym and float(p['positionAmt']) != 0]
            if not api_pos:
                # 청산됨 (SL/TP 히트)
                # PnL 확인
                trades = client.futures_account_trades(symbol=sym, limit=3)
                pnl = sum(float(t['realizedPnl']) for t in trades[-3:] if abs(float(t['realizedPnl'])) > 0.001)

                if pnl > 0:
                    _consecutive_losses = 0
                    _sym_loss_count[sym] = 0
                    _cooldown[sym] = time.time() + COOLDOWN_WIN
                    log(f"✅ {sym} 익절 ${pnl:+.2f} | 연패 리셋")
                    tg(f"✅ <b>{sym} 익절</b> ${pnl:+.2f}")
                else:
                    _consecutive_losses += 1
                    _sym_loss_count[sym] = _sym_loss_count.get(sym, 0) + 1
                    _cooldown[sym] = time.time() + COOLDOWN_LOSS
                    log(f"❌ {sym} 손절 ${pnl:+.2f} | 연속 {_consecutive_losses}패")
                    tg(f"❌ <b>{sym} 손절</b> ${pnl:+.2f} | 연속 {_consecutive_losses}패")
                    if _sym_loss_count[sym] >= 3:
                        _sym_banned.add(sym)
                        log(f"🚫 {sym} 3연패 → 당일 블랙리스트")
                        tg(f"🚫 <b>{sym} 3연패</b> → 당일 블랙리스트")

                closed.append(sym)
                continue

            # 강제 청산 (CVD 2시간)
            hold = time.time() - pos['time']
            if pos['source'] == 'cvd' and hold > 2 * 3600:
                try:
                    for a in client.futures_get_open_algo_orders():
                        if a.get('symbol') == sym:
                            try: client.futures_cancel_algo_order(symbol=sym, algoId=a['algoId'])
                            except: pass
                    time.sleep(0.3)
                    qty = abs(float(api_pos[0]['positionAmt']))
                    client.futures_create_order(symbol=sym, side='SELL', type='MARKET',
                        quantity=str(qty), reduceOnly=True)
                    log(f"⏰ {sym} CVD 2시간 강제 청산")
                except: pass
                closed.append(sym)

        except Exception as e:
            log(f"⚠️ {sym} 체크 오류: {e}")

    for sym in closed:
        _positions.pop(sym, None)

# ── 메인 루프 ──
def main():
    global _sym_banned, _sym_loss_count

    log("=" * 50)
    log("새 봇 시작 — CVD + BB롱, 독립 실행")
    _b = get_balance()
    _bal = float(_b.get('total', _b.get('available', 0))) if isinstance(_b, dict) else float(_b or 0)
    log(f"잔고: ${_bal:.2f}")
    log("=" * 50)

    while True:
        try:
            # 날짜 리셋
            today = datetime.now().strftime('%Y-%m-%d')
            if _daily.get('date') != today:
                _daily['date'] = today; _daily['count'] = 0
                _sym_banned.clear(); _sym_loss_count.clear()
                _consecutive_losses = 0

            # 5연패 정지
            if _consecutive_losses >= 5:
                log(f"🛑 {_consecutive_losses}연패 → 당일 정지")
                time.sleep(INTERVAL)
                continue

            btc_rsi = get_btc_rsi()
            universe = get_universe()
            _b2 = get_balance()
            bal = float(_b2.get('available', _b2.get('total', 0))) if isinstance(_b2, dict) else float(_b2 or 0)

            log(f"스캔: BTC RSI={btc_rsi:.0f} | 잔고=${bal:.2f} | 포지션={len(_positions)} | 종목={len(universe)}")

            # 30분마다 텔레그램 상태 보고
            if not hasattr(main, '_tg_cycle'):
                main._tg_cycle = 0
            main._tg_cycle += 1
            if main._tg_cycle % 6 == 0:  # 5분×6=30분
                _cvd_s = '🟢' if btc_rsi > 45 else '🔴'
                _bb_s = '🟢' if btc_rsi > 55 else '🔴'
                _ts_s = '🟢' if btc_rsi < 35 else '🔴'
                _pos_str = ', '.join(f'{s} ${float(p.get("unrealizedProfit",0)):+.2f}' for s,p in
                    [(p['symbol'],p) for p in client.futures_account()['positions'] if float(p['positionAmt'])!=0]) or '없음'
                tg(f"📊 ${bal:.2f} | BTC {btc_rsi:.0f} | CVD{_cvd_s} BB{_bb_s} 숏{_ts_s} | {_pos_str}")

            # 전략 실행
            check_cvd(btc_rsi, universe)
            check_bb(btc_rsi, universe)
            check_trend_short(btc_rsi, universe)

            # 포지션 관리
            check_positions()

        except Exception as e:
            log(f"사이클 오류: {e}")

        # 5분 대기 (30초마다 포지션 체크)
        for _ in range(INTERVAL // 30):
            time.sleep(30)
            try:
                check_positions()
            except:
                pass

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# 바이낸스 선물 API 래퍼 — 싱글턴 Client + exchange_info 캐싱 + API 재시도

import time
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import API_KEY, API_SECRET, TESTNET

# ── 싱글턴 클라이언트 ──────────────────────────────────────────────────
_client: Client | None = None

def get_client() -> Client:
    """Client 인스턴스 재사용 (싱글턴)"""
    global _client
    if _client is None:
        _client = Client(API_KEY, API_SECRET, testnet=TESTNET)
    return _client

def reset_client():
    """연결 문제 시 Client 재생성"""
    global _client
    _client = None

# ── exchange_info 캐싱 (5분 TTL) ──────────────────────────────────────
_exchange_info_cache = {"data": None, "ts": 0}
_EXCHANGE_INFO_TTL = 300  # 5분

def _get_exchange_info():
    """exchange_info 캐싱 — 매 주문마다 호출하지 않음"""
    now = time.time()
    if _exchange_info_cache["data"] and now - _exchange_info_cache["ts"] < _EXCHANGE_INFO_TTL:
        return _exchange_info_cache["data"]
    client = get_client()
    info = client.futures_exchange_info()
    _exchange_info_cache["data"] = info
    _exchange_info_cache["ts"] = now
    return info

def _get_symbol_filters(symbol: str) -> tuple:
    """심볼의 step_size, tick_size 반환 (캐싱된 exchange_info 사용)"""
    info = _get_exchange_info()
    sym_info = next(s for s in info["symbols"] if s["symbol"] == symbol)
    step_size = float(next(
        f["stepSize"] for f in sym_info["filters"] if f["filterType"] == "LOT_SIZE"
    ))
    tick_size = float(next(
        f["tickSize"] for f in sym_info["filters"] if f["filterType"] == "PRICE_FILTER"
    ))
    return step_size, tick_size

# ── API 속도 제한 보호 ─────────────────────────────────────────────────
_last_api_call_ts = 0.0
_API_MIN_INTERVAL = 0.12  # 초 (분당 ~500회 이하 유지)

# ── API 재시도 래퍼 ────────────────────────────────────────────────────
def _api_call(func, *args, max_retries=3, **kwargs):
    """API 호출 재시도 — 속도 제한 보호 + 네트워크 오류 시 최대 3회"""
    global _last_api_call_ts
    # 요청 간 최소 간격 유지 (IP 밴 방지)
    _elapsed = time.time() - _last_api_call_ts
    if _elapsed < _API_MIN_INTERVAL:
        time.sleep(_API_MIN_INTERVAL - _elapsed)
    for attempt in range(max_retries):
        try:
            _last_api_call_ts = time.time()
            result = func(*args, **kwargs)
            return result
        except BinanceAPIException as e:
            # IP 밴 감지 시 30초 대기 후 재시도
            if "banned" in str(e).lower() or "too many" in str(e).lower():
                time.sleep(30)
                if attempt < max_retries - 1:
                    continue
            raise  # 바이낸스 비즈니스 에러는 즉시 raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # 1초, 2초 대기
                reset_client()  # 연결 재생성
            else:
                raise ConnectionError(f"API 호출 {max_retries}회 실패: {e}") from e

# ── 캔들 데이터 가져오기 ─────────────────────────────────────────────
def get_klines(symbol: str, interval: str = "15m", limit: int = 100) -> pd.DataFrame:
    client = get_client()
    raw = _api_call(client.futures_klines, symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["time"]  = pd.to_datetime(df["time"], unit="ms")
    df["open"]  = df["open"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"]= df["volume"].astype(float)
    df["taker_buy_base"] = df["taker_buy_base"].astype(float)
    return df[["time", "open", "high", "low", "close", "volume", "taker_buy_base"]]

# ── 현재가 ────────────────────────────────────────────────────────────
def get_price(symbol: str) -> float:
    client = get_client()
    ticker = _api_call(client.futures_symbol_ticker, symbol=symbol)
    return float(ticker["price"])

# ── 잔고 조회 (10초 TTL 캐싱) ──────────────────────────────────────────
_balance_cache = {"data": None, "ts": 0}
_BALANCE_TTL = 10  # 초

def get_balance(force: bool = False) -> dict:
    now = time.time()
    if not force and _balance_cache["data"] and now - _balance_cache["ts"] < _BALANCE_TTL:
        return _balance_cache["data"]
    client = get_client()
    account = _api_call(client.futures_account)
    usdt = next((a for a in account["assets"] if a["asset"] == "USDT"), None)
    if usdt:
        result = {
            "total": float(usdt["walletBalance"]),
            "available": float(usdt["availableBalance"]),
            "unrealized_pnl": float(usdt["unrealizedProfit"])
        }
    else:
        result = {"total": 0, "available": 0, "unrealized_pnl": 0}
    _balance_cache["data"] = result
    _balance_cache["ts"] = now
    return result

# ── 포지션 조회 ──────────────────────────────────────────────────────
def get_positions() -> list:
    client = get_client()
    positions = _api_call(client.futures_position_information)
    active = [
        {
            "symbol": p["symbol"],
            "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
            "size": abs(float(p["positionAmt"])),
            "entry_price": float(p["entryPrice"]),
            "unrealized_pnl": float(p["unRealizedProfit"]),
            "leverage": int(p.get("leverage", 1)),
        }
        for p in positions if float(p["positionAmt"]) != 0
    ]
    return active

# ── 마진 타입 설정 ───────────────────────────────────────────────────
def set_margin_type(symbol: str, margin_type: str = "ISOLATED"):
    client = get_client()
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
    except BinanceAPIException as e:
        err = str(e)
        if "No need to change margin type" in err:
            pass  # 이미 동일 타입
        elif "-4067" in err:
            pass  # algo 주문 잔존으로 변경 불가 — 이미 ISOLATED이면 무시
        else:
            raise

# ── 레버리지 설정 ────────────────────────────────────────────────────
def set_leverage(symbol: str, leverage: int):
    client = get_client()
    client.futures_change_leverage(symbol=symbol, leverage=leverage)

# ── 주문 실행 ─────────────────────────────────────────────────────────
def _round_price(price: float, tick_size: float) -> float:
    """tick_size에 맞게 가격 반올림"""
    if tick_size >= 1:
        return round(price)
    decimals = len(str(tick_size).rstrip("0").split(".")[-1])
    return round(round(price / tick_size) * tick_size, decimals)


def place_order(symbol: str, side: str, usdt_amount: float,
                leverage: int = 3, sl_price: float = None, tp_price: float = None) -> dict:
    """
    side: 'BUY' (롱) or 'SELL' (숏)
    usdt_amount: 진입 금액 (USDT 기준)
    sl_price: 손절 가격 (ATR 기반 계산값)
    tp_price: 익절 가격 (ATR 기반 계산값)
    """
    try:
        client = get_client()

        # 기존 주문 전체 정리 — margin type 변경 전에 해야 -4067 방지
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
            time.sleep(0.5)
        except Exception:
            pass

        set_margin_type(symbol, "ISOLATED")
        set_leverage(symbol, leverage)

        price = get_price(symbol)

        # 수량 계산 (캐싱된 exchange_info 사용)
        step_size, tick_size = _get_symbol_filters(symbol)
        qty_raw = (usdt_amount * leverage) / price
        if step_size >= 1:
            decimals = 0
        else:
            decimals = len(str(step_size).rstrip("0").split(".")[-1])
        qty = round(qty_raw - (qty_raw % step_size), decimals)

        # 최소 수량 보장: 0이면 step_size로 올림 (notional 부족은 거래소가 검증)
        if qty <= 0:
            qty = step_size

        # 최소 notional $20 미달 시 진입 차단 (강제 올림 제거 — 리스크 관리 우회 방지)
        _notional = qty * price
        if _notional < 20:
            return {"success": False, "error": f"노셔널 ${_notional:.1f} < $20 미달, 진입 차단 (진입금 부족)"}

        # 시장가 진입
        order = client.futures_create_order(
            symbol=symbol, side=side, type="MARKET", quantity=qty,
        )

        # SL 주문 (STOP_MARKET) — quantity 지정 (Algo API 호환)
        sl_placed = False
        if sl_price:
            sl_side = "SELL" if side == "BUY" else "BUY"
            sl_rounded = _round_price(sl_price, tick_size)
            for _try in range(2):
                try:
                    _sl_resp = client.futures_create_order(
                        symbol=symbol, side=sl_side,
                        type="STOP_MARKET",
                        stopPrice=sl_rounded,
                        quantity=qty,
                        reduceOnly=True,
                    )
                    # Algo 주문은 algoId, 일반 주문은 orderId 반환
                    sl_placed = bool(_sl_resp.get("orderId") or _sl_resp.get("algoId"))
                    break
                except Exception as e:
                    if _try == 0:
                        time.sleep(0.5)
                    else:
                        print(f"⚠️ SL 주문 실패 ({symbol}): {e}")

        # TP 주문 (TAKE_PROFIT_MARKET) — quantity 지정 (Algo API 호환)
        tp_placed = False
        if tp_price:
            tp_side = "SELL" if side == "BUY" else "BUY"
            tp_rounded = _round_price(tp_price, tick_size)
            for _try in range(2):
                try:
                    _tp_resp = client.futures_create_order(
                        symbol=symbol, side=tp_side,
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_rounded,
                        quantity=qty,
                        reduceOnly=True,
                    )
                    tp_placed = bool(_tp_resp.get("orderId") or _tp_resp.get("algoId"))
                    break
                except Exception as e:
                    if _try == 0:
                        time.sleep(0.5)
                    else:
                        print(f"⚠️ TP 주문 실패 ({symbol}): {e}")

        # 실제 체결 평균가 추출 (슬리피지 추적용)
        fill_price = price
        try:
            if float(order.get("avgPrice", 0)) > 0:
                fill_price = float(order["avgPrice"])
            elif order.get("fills"):
                _total_qty = sum(float(f["qty"]) for f in order["fills"])
                _total_cost = sum(float(f["qty"]) * float(f["price"]) for f in order["fills"])
                if _total_qty > 0:
                    fill_price = _total_cost / _total_qty
        except Exception:
            pass

        return {"success": True, "order": order, "qty": qty, "price": price,
                "fill_price": round(fill_price, 8),
                "sl_price": sl_price, "tp_price": tp_price,
                "sl_placed": sl_placed, "tp_placed": tp_placed}
    except BinanceAPIException as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def place_limit_order(symbol: str, side: str, usdt_amount: float,
                      entry_price: float, leverage: int = 3,
                      sl_price: float = None, tp_price: float = None) -> dict:
    """지정가(LIMIT) 주문 + SL/TP 동시 배치
    바이낸스 algo API는 포지션 없어도 SL/TP 배치 가능 → 체결 즉시 보호
    """
    try:
        client = get_client()

        # 기존 주문 전체 정리 (일반 + algo) — margin type 변경 전에 해야 -4067 방지
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
            time.sleep(0.5)
        except Exception:
            pass

        set_margin_type(symbol, "ISOLATED")
        set_leverage(symbol, leverage)

        # 수량 계산
        step_size, tick_size = _get_symbol_filters(symbol)
        qty_raw = (usdt_amount * leverage) / entry_price
        if step_size >= 1:
            decimals = 0
        else:
            decimals = len(str(step_size).rstrip("0").split(".")[-1])
        qty = round(qty_raw - (qty_raw % step_size), decimals)
        if qty <= 0:
            qty = step_size

        _notional = qty * entry_price
        if _notional < 20:
            return {"success": False, "error": f"노셔널 ${_notional:.1f} < $20 미달"}

        # 지정가 진입 (LIMIT + GTC)
        entry_rounded = _round_price(entry_price, tick_size)
        order = client.futures_create_order(
            symbol=symbol, side=side, type="LIMIT",
            price=entry_rounded, quantity=qty, timeInForce="GTC",
        )
        order_id = order.get("orderId")

        # SL/TP 동시 배치 (algo API — 포지션 없어도 가능, 체결 즉시 보호)
        sl_placed, tp_placed = False, False
        sl_side = "SELL" if side == "BUY" else "BUY"
        sl_algo_id, tp_algo_id = None, None
        if sl_price:
            sl_rounded = _round_price(sl_price, tick_size)
            for _try in range(2):
                try:
                    _sl_r = client.futures_create_order(
                        symbol=symbol, side=sl_side, type="STOP_MARKET",
                        stopPrice=sl_rounded, quantity=qty, reduceOnly=True,
                    )
                    sl_placed = bool(_sl_r.get("orderId") or _sl_r.get("algoId"))
                    sl_algo_id = _sl_r.get("algoId") or _sl_r.get("orderId")
                    break
                except Exception:
                    if _try == 0: time.sleep(0.5)
        if tp_price:
            tp_rounded = _round_price(tp_price, tick_size)
            for _try in range(2):
                try:
                    _tp_r = client.futures_create_order(
                        symbol=symbol, side=sl_side, type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_rounded, quantity=qty, reduceOnly=True,
                    )
                    tp_placed = bool(_tp_r.get("orderId") or _tp_r.get("algoId"))
                    tp_algo_id = _tp_r.get("algoId") or _tp_r.get("orderId")
                    break
                except Exception:
                    if _try == 0: time.sleep(0.5)

        return {"success": True, "order": order, "qty": qty,
                "price": entry_price, "fill_price": entry_price,
                "order_id": order_id, "order_type": "LIMIT",
                "sl_price": sl_price, "tp_price": tp_price,
                "sl_placed": sl_placed, "tp_placed": tp_placed,
                "sl_algo_id": sl_algo_id, "tp_algo_id": tp_algo_id}
    except BinanceAPIException as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def place_sl_tp(symbol: str, side: str, qty: float,
                sl_price: float = None, tp_price: float = None) -> dict:
    """체결된 포지션에 SL/TP 배치 (STOP/TAKE_PROFIT LIMIT — 일반 주문으로 관리)"""
    client = get_client()
    _, tick_size = _get_symbol_filters(symbol)
    sl_side = "SELL" if side == "BUY" else "BUY"

    sl_placed, tp_placed = False, False
    if sl_price:
        sl_rounded = _round_price(sl_price, tick_size)
        for _try in range(2):
            try:
                _sl_resp = client.futures_create_order(
                    symbol=symbol, side=sl_side, type="STOP",
                    stopPrice=sl_rounded, price=sl_rounded,
                    quantity=qty, reduceOnly=True, timeInForce="GTC",
                )
                sl_placed = bool(_sl_resp.get("orderId") or _sl_resp.get("algoId"))
                break
            except Exception:
                if _try == 0:
                    time.sleep(0.5)

    if tp_price:
        tp_rounded = _round_price(tp_price, tick_size)
        for _try in range(2):
            try:
                _tp_resp = client.futures_create_order(
                    symbol=symbol, side=sl_side, type="TAKE_PROFIT",
                    stopPrice=tp_rounded, price=tp_rounded,
                    quantity=qty, reduceOnly=True, timeInForce="GTC",
                )
                tp_placed = bool(_tp_resp.get("orderId") or _tp_resp.get("algoId"))
                break
            except Exception:
                if _try == 0:
                    time.sleep(0.5)

    return {"sl_placed": sl_placed, "tp_placed": tp_placed}


def cancel_all_orders(symbol: str):
    """일반 주문 취소 (algo/conditional은 API 미지원 → 앱에서 수동 취소)"""
    client = get_client()
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
    except Exception:
        pass


# ── 펀딩비 조회 ──────────────────────────────────────────────────────
def get_funding_rate(symbol: str) -> dict:
    """현재 펀딩비 반환 — 양수: 롱 과열, 음수: 숏 과열"""
    try:
        client = get_client()
        data = _api_call(client.futures_mark_price, symbol=symbol)
        rate = float(data["lastFundingRate"])
        return {"rate": round(rate * 100, 4), "available": True}  # % 단위
    except Exception:
        return {"rate": 0, "available": False}

# ── 펀딩비 추이 (8회) ─────────────────────────────────────────────────
def get_funding_rate_history(symbol: str, limit: int = 8) -> list:
    """최근 N회 펀딩비 반환 — 연속 방향성 감지용 (oldest→newest)"""
    try:
        client = get_client()
        rows = _api_call(client.futures_funding_rate, symbol=symbol, limit=limit)
        return [round(float(r["fundingRate"]) * 100, 4) for r in rows]
    except Exception:
        return []

# ── 오픈 인터레스트 조회 ──────────────────────────────────────────────
def get_open_interest(symbol: str) -> dict:
    """현재 미결제약정(OI) 반환"""
    try:
        client = get_client()
        data = _api_call(client.futures_open_interest, symbol=symbol)
        return {"oi": float(data["openInterest"]), "available": True}
    except Exception:
        return {"oi": 0, "available": False}


def get_long_short_ratio(symbol: str, period: str = "15m", limit: int = 5) -> dict:
    """탑 트레이더 롱/숏 비율 — 청산 압력 방향 추정"""
    try:
        client = get_client()
        data = _api_call(client.futures_top_longshort_account_ratio, symbol=symbol, period=period, limit=limit)
        if data:
            latest = data[-1]
            ratio = float(latest.get("longShortRatio", 1.0))
            long_pct = float(latest.get("longAccount", 0.5))
            return {"ratio": round(ratio, 3), "long_pct": round(long_pct, 3), "available": True}
        return {"ratio": 1.0, "long_pct": 0.5, "available": False}
    except Exception:
        return {"ratio": 1.0, "long_pct": 0.5, "available": False}


def get_oi_change(symbol: str, period: str = "15m", limit: int = 5) -> dict:
    """OI 히스토리 기반 변화율 계산 (%)"""
    try:
        client = get_client()
        data = _api_call(client.futures_open_interest_hist, symbol=symbol, period=period, limit=limit)
        if len(data) >= 2:
            oldest = float(data[0]["sumOpenInterest"])
            newest = float(data[-1]["sumOpenInterest"])
            if oldest > 0:
                change_pct = (newest - oldest) / oldest * 100
                return {"change_pct": round(change_pct, 2), "available": True}
        return {"change_pct": 0, "available": False}
    except Exception:
        return {"change_pct": 0, "available": False}


# ── 최근 체결 내역 (실현 손익 포함) ──────────────────────────────────
def get_recent_trades(symbol: str, limit: int = 20) -> list:
    """최근 체결 내역 조회 — SL/TP 자동 청산 감지용"""
    try:
        client = get_client()
        trades = _api_call(client.futures_account_trades, symbol=symbol, limit=limit)
        return [
            {
                "time": pd.to_datetime(t["time"], unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
                "realized_pnl": float(t["realizedPnl"]),
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "side": t["side"],
            }
            for t in trades
        ]
    except Exception:
        return []

# ── SL 업데이트 (트레일링 스탑용) ────────────────────────────────────
def update_stop_loss(symbol: str, new_sl_price: float, position_side: str) -> dict:
    """기존 SL 주문 전체 취소 후 새 SL 가격으로 재배치 (GTE closePosition 충돌 방지)"""
    try:
        client = get_client()
        # 해당 심볼의 STOP_MARKET + STOP 주문 모두 취소 (GTE closePosition 포함)
        open_orders = _api_call(client.futures_get_open_orders, symbol=symbol)
        for o in open_orders:
            if o["type"] in ("STOP_MARKET", "STOP"):
                try:
                    _api_call(client.futures_cancel_order, symbol=symbol, orderId=o["orderId"])
                except Exception:
                    pass
        time.sleep(0.3)  # 취소 처리 대기
        # 새 SL 배치 (캐싱된 exchange_info 사용)
        step_size, tick_size = _get_symbol_filters(symbol)
        sl_rounded = _round_price(new_sl_price, tick_size)
        sl_side = "SELL" if position_side == "LONG" else "BUY"
        # 포지션 수량 조회
        _pos_qty = 0
        try:
            _positions = client.futures_position_information(symbol=symbol)
            for _p in _positions:
                _amt = abs(float(_p.get("positionAmt", 0)))
                if _amt > 0:
                    _pos_qty = _amt
                    break
        except Exception:
            pass
        if _pos_qty <= 0:
            _pos_qty = step_size  # 폴백
        _api_call(client.futures_create_order,
            symbol=symbol, side=sl_side,
            type="STOP_MARKET",
            stopPrice=sl_rounded,
            quantity=_pos_qty,
            reduceOnly=True,
        )
        return {"success": True, "new_sl": sl_rounded}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 부분 청산 (TP 도달 시 일부 청산) ────────────────────────────────
def partial_close_position(symbol: str, close_pct: float = 0.5) -> dict:
    """포지션의 일부(기본 50%)를 시장가로 청산"""
    positions = get_positions()
    pos = next((p for p in positions if p["symbol"] == symbol), None)
    if not pos:
        return {"success": False, "error": "포지션 없음"}
    total_qty = pos["size"]
    close_qty_raw = total_qty * close_pct
    if close_qty_raw <= 0:
        return {"success": False, "error": "청산 수량 0"}
    close_side = "SELL" if pos["side"] == "LONG" else "BUY"
    try:
        client = get_client()
        # 캐싱된 exchange_info 사용
        step_size, _ = _get_symbol_filters(symbol)
        if step_size >= 1:
            decimals = 0
        else:
            decimals = len(str(step_size).rstrip("0").split(".")[-1])
        close_qty = round(close_qty_raw - (close_qty_raw % step_size), decimals)
        if close_qty <= 0:
            return {"success": False, "error": "최소 수량 미달"}
        order = client.futures_create_order(
            symbol=symbol, side=close_side, type="MARKET",
            quantity=close_qty, reduceOnly=True,
        )
        return {"success": True, "order": order, "qty": close_qty, "close_pct": close_pct}
    except BinanceAPIException as e:
        return {"success": False, "error": str(e)}

# ── 포지션 청산 ──────────────────────────────────────────────────────
def close_position(symbol: str) -> dict:
    positions = get_positions()
    pos = next((p for p in positions if p["symbol"] == symbol), None)
    if not pos:
        return {"success": False, "error": "포지션 없음"}

    close_side = "SELL" if pos["side"] == "LONG" else "BUY"
    try:
        client = get_client()
        order = client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type="MARKET",
            quantity=pos["size"],
            reduceOnly=True,
        )
        # SL/TP 오펀 주문 취소 (일괄 + 개별)
        cancel_open_orders(symbol)
        return {"success": True, "order": order}
    except BinanceAPIException as e:
        return {"success": False, "error": str(e)}

# ── 잔존 주문 정리 ────────────────────────────────────────────────────
def cancel_open_orders(symbol: str, algo_ids: list = None) -> dict:
    """해당 심볼의 미체결 주문 전체 취소 (일반 + algo 주문 모두)
    algo_ids: 추가로 취소할 algoId 리스트 (선택)
    """
    try:
        client = get_client()
        # 1차: 일괄 취소 (일반 + algo 모두 취소됨)
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
        except Exception:
            pass
        # 2차: 개별 주문 ID로 확실히 정리
        try:
            remaining = client.futures_get_open_orders(symbol=symbol)
            for o in remaining:
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
                except Exception:
                    pass
        except Exception:
            pass
        # 3차: 명시적 algo 주문 취소 (algoId 지정 시)
        if algo_ids:
            for aid in algo_ids:
                if aid:
                    try:
                        client._request_futures_api('delete', 'algoOrder', True, data={'algoId': aid})
                    except Exception:
                        pass
        return {"success": True}
    except BinanceAPIException as e:
        return {"success": False, "error": str(e)}

def get_open_orders(symbol: str) -> list:
    """해당 심볼의 미체결 주문 목록 조회"""
    try:
        client = get_client()
        return client.futures_get_open_orders(symbol=symbol)
    except Exception:
        return []


# ── 비상 전량 청산 ────────────────────────────────────────────────────
def emergency_close_all() -> dict:
    """모든 포지션 청산 + 모든 미체결 주문 취소 (API 장애/비상 시)"""
    results = {"closed": [], "errors": [], "orders_cancelled": False}
    try:
        client = get_client()
        # 1) 모든 미체결 주문 취소
        try:
            positions = client.futures_position_information()
            symbols_with_orders = set()
            for p in positions:
                if float(p["positionAmt"]) != 0:
                    symbols_with_orders.add(p["symbol"])
            for sym in symbols_with_orders:
                try:
                    client.futures_cancel_all_open_orders(symbol=sym)
                except Exception:
                    pass
            results["orders_cancelled"] = True
        except Exception as e:
            results["errors"].append(f"주문 취소 실패: {e}")

        # 2) 모든 포지션 시장가 청산
        try:
            positions = client.futures_position_information()
            for p in positions:
                amt = float(p["positionAmt"])
                if amt == 0:
                    continue
                sym = p["symbol"]
                close_side = "SELL" if amt > 0 else "BUY"
                try:
                    client.futures_create_order(
                        symbol=sym, side=close_side, type="MARKET",
                        quantity=abs(amt), reduceOnly=True,
                    )
                    results["closed"].append(sym)
                except Exception as e:
                    results["errors"].append(f"{sym} 청산 실패: {e}")
        except Exception as e:
            results["errors"].append(f"포지션 조회 실패: {e}")
    except Exception as e:
        # Client 자체 생성 실패
        results["errors"].append(f"Client 연결 실패: {e}")
    return results

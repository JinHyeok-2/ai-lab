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

# ── API 재시도 래퍼 ────────────────────────────────────────────────────
def _api_call(func, *args, max_retries=3, **kwargs):
    """API 호출 재시도 — 네트워크 오류 시 최대 3회, BinanceAPIException은 즉시 raise"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except BinanceAPIException:
            raise  # 바이낸스 비즈니스 에러는 재시도 안 함
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

# ── 잔고 조회 ─────────────────────────────────────────────────────────
def get_balance() -> dict:
    client = get_client()
    account = _api_call(client.futures_account)
    usdt = next((a for a in account["assets"] if a["asset"] == "USDT"), None)
    if usdt:
        return {
            "total": float(usdt["walletBalance"]),
            "available": float(usdt["availableBalance"]),
            "unrealized_pnl": float(usdt["unrealizedProfit"])
        }
    return {"total": 0, "available": 0, "unrealized_pnl": 0}

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
        if "No need to change margin type" in str(e):
            pass  # 이미 동일 타입이면 무시
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
                leverage: int = 3, sl_price: float = None, tp_price: float = None,
                use_limit: bool = False) -> dict:
    """
    side: 'BUY' (롱) or 'SELL' (숏)
    usdt_amount: 진입 금액 (USDT 기준)
    sl_price: 손절 가격 (ATR 기반 계산값)
    tp_price: 익절 가격 (ATR 기반 계산값)
    use_limit: True → 지정가 IOC (±0.03%) 먼저 시도, 미체결 시 시장가 폴백
    """
    try:
        client = get_client()
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

        # 지정가 IOC 시도 (use_limit=True) — 메이커 수수료 절감 목적 (±0.03%)
        order = None
        filled_qty = 0.0
        if use_limit:
            offset = 0.0003
            limit_px = price * (1 - offset) if side == "BUY" else price * (1 + offset)
            limit_px = _round_price(limit_px, tick_size)
            try:
                _lo = client.futures_create_order(
                    symbol=symbol, side=side, type="LIMIT",
                    quantity=qty, price=limit_px, timeInForce="IOC",
                )
                filled_qty = float(_lo.get("executedQty", 0))
                if filled_qty >= qty * 0.95:
                    # 95% 이상 체결 → 완료
                    order = _lo
                    price = float(_lo.get("avgPrice", price)) or price
                elif filled_qty > 0:
                    # 부분 체결 → 나머지 시장가 폴백
                    remain_qty = round(qty - filled_qty, decimals)
                    if remain_qty > 0:
                        _mo = client.futures_create_order(
                            symbol=symbol, side=side, type="MARKET",
                            quantity=remain_qty,
                        )
                        # 두 주문 합산 (qty=전체, price=가중평균)
                        mo_filled = float(_mo.get("executedQty", 0))
                        mo_price  = float(_mo.get("avgPrice", 0)) or price
                        ioc_price = float(_lo.get("avgPrice", 0)) or price
                        total_filled = filled_qty + mo_filled
                        if total_filled > 0:
                            price = round((ioc_price * filled_qty + mo_price * mo_filled) / total_filled, 8)
                        order = _mo
                        order["executedQty"] = str(total_filled)
                    else:
                        order = _lo
                        price = float(_lo.get("avgPrice", price)) or price
            except Exception:
                pass  # 지정가 실패 시 시장가 폴백

        # 시장가 진입 (지정가 미체결 시 폴백)
        if order is None:
            order = client.futures_create_order(
                symbol=symbol, side=side, type="MARKET", quantity=qty,
            )

        # SL 주문 (STOP_MARKET) — 실패 시 1회 재시도
        sl_placed = False
        if sl_price:
            sl_side = "SELL" if side == "BUY" else "BUY"
            sl_rounded = _round_price(sl_price, tick_size)
            for _try in range(2):
                try:
                    client.futures_create_order(
                        symbol=symbol, side=sl_side,
                        type="STOP_MARKET",
                        stopPrice=sl_rounded,
                        closePosition=True,
                    )
                    sl_placed = True
                    break
                except Exception as e:
                    if _try == 0:
                        time.sleep(0.5)
                    else:
                        print(f"⚠️ SL 주문 실패 ({symbol}): {e}")

        # TP 주문 (TAKE_PROFIT_MARKET) — 실패 시 1회 재시도
        tp_placed = False
        if tp_price:
            tp_side = "SELL" if side == "BUY" else "BUY"
            tp_rounded = _round_price(tp_price, tick_size)
            for _try in range(2):
                try:
                    client.futures_create_order(
                        symbol=symbol, side=tp_side,
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_rounded,
                        closePosition=True,
                    )
                    tp_placed = True
                    break
                except Exception as e:
                    if _try == 0:
                        time.sleep(0.5)
                    else:
                        print(f"⚠️ TP 주문 실패 ({symbol}): {e}")

        return {"success": True, "order": order, "qty": qty, "price": price,
                "sl_price": sl_price, "tp_price": tp_price,
                "sl_placed": sl_placed, "tp_placed": tp_placed}
    except BinanceAPIException as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}

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
    """기존 STOP_MARKET 주문만 취소 후 새 SL 가격으로 재배치"""
    try:
        client = get_client()
        # STOP_MARKET 주문만 찾아서 취소
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for o in open_orders:
            if o["type"] == "STOP_MARKET":
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
                except Exception:
                    pass
        # 새 SL 배치 (캐싱된 exchange_info 사용)
        _, tick_size = _get_symbol_filters(symbol)
        sl_rounded = _round_price(new_sl_price, tick_size)
        sl_side = "SELL" if position_side == "LONG" else "BUY"
        client.futures_create_order(
            symbol=symbol, side=sl_side,
            type="STOP_MARKET",
            stopPrice=sl_rounded,
            closePosition=True,
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
        # SL/TP 오펀 주문 취소
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
        except Exception:
            pass
        return {"success": True, "order": order}
    except BinanceAPIException as e:
        return {"success": False, "error": str(e)}

# ── 잔존 주문 정리 ────────────────────────────────────────────────────
def cancel_open_orders(symbol: str) -> dict:
    """해당 심볼의 미체결 주문 전체 취소 (SL/TP 잔존 주문 정리용)"""
    try:
        client = get_client()
        client.futures_cancel_all_open_orders(symbol=symbol)
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

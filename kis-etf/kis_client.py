#!/usr/bin/env python3
# 한국투자증권 OpenAPI 클라이언트 래퍼 (python-kis 기반)
import sys; sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from datetime import date, timedelta
from pykis import PyKis

from config import (
    KIS_APP_KEY, KIS_APP_SECRET,
    KIS_VIRTUAL_APP_KEY, KIS_VIRTUAL_APP_SECRET,
    KIS_ACCOUNT_NO, KIS_HTS_ID, IS_VIRTUAL, TOKEN_CACHE, ETF_UNIVERSE,
)

_kis: PyKis = None


def connect() -> PyKis:
    """KIS API 연결 (싱글턴)"""
    global _kis
    if _kis is not None:
        return _kis

    if not KIS_APP_KEY or not KIS_APP_SECRET:
        raise ValueError("KIS_APP_KEY / KIS_APP_SECRET 미설정. .env 파일 확인")
    if not KIS_ACCOUNT_NO:
        raise ValueError("KIS_ACCOUNT_NO 미설정. .env 파일 확인")

    # 모의투자 / 실거래 분기
    kwargs = {
        "account": KIS_ACCOUNT_NO,
        "keep_token": str(TOKEN_CACHE),
        "use_websocket": False,  # 데일리 봇이므로 웹소켓 불필요
    }

    # 실전 키 (시세 조회용) + 모의 키 (주문용)
    kwargs["id"]        = KIS_HTS_ID
    kwargs["appkey"]    = KIS_APP_KEY
    kwargs["secretkey"] = KIS_APP_SECRET

    if IS_VIRTUAL and KIS_VIRTUAL_APP_KEY:
        kwargs["virtual_id"]        = KIS_HTS_ID
        kwargs["virtual_appkey"]    = KIS_VIRTUAL_APP_KEY
        kwargs["virtual_secretkey"] = KIS_VIRTUAL_APP_SECRET

    _kis = PyKis(**kwargs)
    mode = "모의투자" if IS_VIRTUAL else "실거래"
    print(f"[KIS] 연결 성공 ({mode}, 계좌: {KIS_ACCOUNT_NO})")
    return _kis


def get_account():
    """계좌 객체 반환"""
    kis = connect()
    return kis.account()


def get_balance() -> dict:
    """잔고 조회 — 예수금 + 보유 종목"""
    acct = get_account()
    bal  = acct.balance()

    # deposit은 메서드 (통화별 예수금 중 KRW)
    try:
        krw_deposit = int(bal.deposit("KRW").amount)
    except Exception:
        krw_deposit = int(bal.total) - int(bal.purchase_amount)

    result = {
        "deposit":        krw_deposit,                # 예수금 (KRW)
        "total":          int(bal.total),              # 총 평가금액
        "holdings_value": int(bal.purchase_amount),    # 매입금액 합계
        "profit":         int(bal.profit),             # 평가손익
        "profit_rate":    float(bal.profit_rate) if bal.profit_rate else 0.0,
        "stocks":         [],
    }

    for s in bal.stocks:
        qty = int(s.qty) if hasattr(s, 'qty') else 0
        result["stocks"].append({
            "ticker":    s.symbol,
            "name":      s.name,
            "qty":       qty,
            "avg_price": int(s.purchase_amount / qty) if qty else 0,
            "cur_price": int(s.current_amount / qty) if qty else 0,
            "pnl":       int(s.profit),
            "pnl_pct":   float(s.profit_rate) if s.profit_rate else 0.0,
        })

    return result


def get_price(ticker: str) -> dict:
    """현재가 조회"""
    kis   = connect()
    stock = kis.stock(ticker)
    q     = stock.quote()
    return {
        "price":  int(q.price),
        "open":   int(q.open),
        "high":   int(q.high),
        "low":    int(q.low),
        "volume": int(q.volume),
        "change": float(q.rate) if q.rate else 0.0,
        "name":   q.name,
    }


def get_ohlcv(ticker: str, days: int = 100) -> pd.DataFrame:
    """일봉 OHLCV 조회 (최근 N일)"""
    kis   = connect()
    stock = kis.stock(ticker)
    chart = stock.chart(
        start=date.today() - timedelta(days=int(days * 1.5)),  # 영업일 보정
        end=date.today(),
        period="day",
    )
    df = chart.df()
    # 컬럼 정리
    if df is not None and len(df) > 0:
        df = df.tail(days)
    return df


def buy_market(ticker: str, qty: int) -> dict:
    """시장가 매수"""
    kis   = connect()
    stock = kis.stock(ticker, account=get_account().account_number)
    order = stock.order("buy", qty=qty)
    print(f"[KIS] 매수 주문: {ticker} {qty}주, 주문번호: {order.number}")
    return {
        "number":  order.number,
        "ticker":  ticker,
        "side":    "매수",
        "qty":     qty,
    }


def sell_market(ticker: str, qty: int) -> dict:
    """시장가 매도"""
    kis   = connect()
    stock = kis.stock(ticker, account=get_account().account_number)
    order = stock.order("sell", qty=qty)
    print(f"[KIS] 매도 주문: {ticker} {qty}주, 주문번호: {order.number}")
    return {
        "number":  order.number,
        "ticker":  ticker,
        "side":    "매도",
        "qty":     qty,
    }


def get_etf_name(ticker: str) -> str:
    """ETF 이름 조회"""
    if ticker in ETF_UNIVERSE:
        return ETF_UNIVERSE[ticker]
    try:
        kis   = connect()
        stock = kis.stock(ticker)
        info  = stock.info()
        return info.name_kor
    except Exception:
        return ticker

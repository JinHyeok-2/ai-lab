#!/usr/bin/env python3
# KIS-ETF 텔레그램 알림 모듈 (경량)
import sys; sys.stdout.reconfigure(line_buffering=True)

import requests
from datetime import datetime
from config import ETF_TG_TOKEN, ETF_TG_CHAT_ID


def _send(text: str, parse_mode: str = "HTML") -> bool:
    """텔레그램 메시지 전송 (내부용)"""
    if not ETF_TG_TOKEN or not ETF_TG_CHAT_ID:
        print("[TG] 토큰/채팅ID 미설정, 스킵")
        return False
    try:
        url  = f"https://api.telegram.org/bot{ETF_TG_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    ETF_TG_CHAT_ID,
            "text":       text,
            "parse_mode": parse_mode,
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[TG] 전송 실패: {e}")
        return False


def send_message(text: str) -> bool:
    """범용 메시지 전송"""
    return _send(text)


def send_order(ticker: str, name: str, side: str, qty: int, price: int, reason: str = "") -> bool:
    """주문 실행 알림"""
    emoji = "🟢" if side == "매수" else "🔴"
    now = datetime.now().strftime("%m/%d %H:%M")
    amount = qty * price
    text = (
        f"{emoji} <b>[ETF {side}] {name}</b>\n"
        f"📌 종목: {ticker}\n"
        f"💰 가격: {price:,}원 × {qty}주 = {amount:,}원\n"
        f"🕐 {now}"
    )
    if reason:
        text += f"\n📋 사유: {reason}"
    return _send(text)


def send_close(ticker: str, name: str, qty: int, price: int, pnl: int, pnl_pct: float) -> bool:
    """매도 완료 알림"""
    emoji = "✅" if pnl >= 0 else "❌"
    now = datetime.now().strftime("%m/%d %H:%M")
    text = (
        f"{emoji} <b>[ETF 매도] {name}</b>\n"
        f"📌 종목: {ticker}\n"
        f"💰 매도가: {price:,}원 × {qty}주\n"
        f"📊 손익: {pnl:+,}원 ({pnl_pct:+.2f}%)\n"
        f"🕐 {now}"
    )
    return _send(text)


def send_error(msg: str) -> bool:
    """오류 알림"""
    now = datetime.now().strftime("%m/%d %H:%M")
    return _send(f"🚨 <b>[ETF 오류]</b>\n{msg}\n🕐 {now}")


def send_status(balance: int, holdings: list, unrealized_pnl: int = 0) -> bool:
    """상태 보고"""
    now = datetime.now().strftime("%m/%d %H:%M")
    text = (
        f"📊 <b>[ETF 상태]</b>\n"
        f"💰 예수금: {balance:,}원\n"
        f"📈 미실현손익: {unrealized_pnl:+,}원\n"
        f"📦 보유 {len(holdings)}종목\n"
    )
    for h in holdings[:5]:
        text += f"  • {h['name']} {h['qty']}주 ({h['pnl_pct']:+.1f}%)\n"
    text += f"🕐 {now}"
    return _send(text)


def send_daily_report(date_str: str, trades: int, pnl: int, balance: int) -> bool:
    """일일 리포트"""
    emoji = "📈" if pnl >= 0 else "📉"
    text = (
        f"{emoji} <b>[ETF 일일 리포트] {date_str}</b>\n"
        f"거래: {trades}건\n"
        f"손익: {pnl:+,}원\n"
        f"잔고: {balance:,}원"
    )
    return _send(text)


def send_preview(signals: list, date_str: str) -> bool:
    """장전 신호 미리보기 (08:55)"""
    if not signals:
        return _send(
            f"📋 <b>[ETF 장전 미리보기]</b> {date_str}\n"
            f"오늘 매매 신호 없음 (모두 HOLD)"
        )
    text = f"📋 <b>[ETF 장전 미리보기]</b> {date_str}\n"
    text += f"09:05 주문 예정 {len(signals)}건:\n\n"
    for s in signals:
        emoji = "🟢" if s["action"] == "BUY" else "🔴"
        amount = s["qty"] * s["price"]
        text += f"{emoji} <b>{s['name']}</b> {s['action']} {s['qty']}주\n"
        text += f"   예상가 {s['price']:,}원 ({amount:,}원)\n"
        text += f"   📋 {s.get('reason', '')}\n\n"
    return _send(text)


def send_fill_report(fills: list, date_str: str) -> bool:
    """체결 확인 리포트 (09:10)"""
    if not fills:
        return _send(
            f"📦 <b>[ETF 체결 확인]</b> {date_str}\n"
            f"오늘 체결 내역 없음"
        )
    text = f"📦 <b>[ETF 체결 확인]</b> {date_str}\n\n"
    for f in fills:
        emoji = "🟢" if f["side"] == "매수" else "🔴"
        status = "✅ 체결" if f["filled"] else "⏳ 미체결"
        text += f"{emoji} {f['name']} {f['side']} {f['qty']}주\n"
        text += f"   {status}"
        if f.get("fill_price"):
            text += f" @ {f['fill_price']:,}원"
        text += "\n\n"
    return _send(text)


def test_connection() -> bool:
    """연결 테스트"""
    now = datetime.now().strftime("%H:%M:%S")
    from config import IS_VIRTUAL
    mode = "모의투자" if IS_VIRTUAL else "실거래"
    return _send(
        f"✅ <b>KIS-ETF 봇 연결 성공</b>\n"
        f"🕐 {now}\n"
        f"모드: {mode}"
    )

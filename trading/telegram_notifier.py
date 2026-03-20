#!/usr/bin/env python3
# 텔레그램 알림 + 명령어 처리 모듈 — AI 트레이딩 봇

import requests
import json
import threading
import time
from datetime import datetime
from pathlib import Path

_CMD_FILE = Path(__file__).parent / "tg_commands.json"
_cmd_lock = threading.Lock()


def _send(token: str, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
    """텔레그램 메시지 전송 (내부용)"""
    if not token or not chat_id:
        return False
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": parse_mode,
        }, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def test_connection(token: str, chat_id: str) -> bool:
    """연결 테스트 메시지 전송"""
    now = datetime.now().strftime("%H:%M:%S")
    return _send(token, chat_id,
        f"✅ <b>AI 트레이딩 봇 연결 성공</b>\n"
        f"🕐 {now}\n"
        f"텔레그램 알림이 활성화되었습니다.\n\n"
        f"📋 <b>사용 가능한 명령어:</b>\n"
        f"/status — 현재 상태 조회\n"
        f"/pause — 자동 거래 일시 중지\n"
        f"/resume — 자동 거래 재개\n"
        f"/close ETHUSDT — 포지션 청산")


def send_signal(token: str, chat_id: str, symbol: str, decision: str,
                confluence: str = "", rl_label: str = "", price: float = 0) -> bool:
    """매매 신호 알림"""
    if "롱 진입" in decision:
        emoji, side = "🟢", "롱 진입"
    elif "숏 진입" in decision:
        emoji, side = "🔴", "숏 진입"
    else:
        emoji, side = "⚪", "관망"
    now  = datetime.now().strftime("%m/%d %H:%M")
    text = f"{emoji} <b>[{symbol}] {side}</b>\n🕐 {now}"
    if price:
        text += f"\n💰 현재가: <code>${price:,.2f}</code>"
    if confluence:
        text += f"\n📡 컨플루언스: {confluence}"
    if rl_label:
        text += f"\n🤖 PPO 신호: {rl_label}"
    return _send(token, chat_id, text)


def send_order(token: str, chat_id: str, symbol: str,
               side: str, qty: float, price: float,
               sl: float = None, tp: float = None) -> bool:
    """자동 주문 실행 알림"""
    emoji   = "🟢" if side == "BUY" else "🔴"
    side_kr = "롱" if side == "BUY" else "숏"
    now  = datetime.now().strftime("%m/%d %H:%M")
    text = (
        f"{emoji} <b>[주문 실행] {symbol} {side_kr}</b>\n"
        f"🕐 {now}\n"
        f"📦 수량: <code>{qty}</code>\n"
        f"💰 체결가: <code>${price:,.2f}</code>"
    )
    if sl:
        text += f"\n🛑 SL: <code>${sl:,.2f}</code>"
    if tp:
        text += f"\n🎯 TP: <code>${tp:,.2f}</code>"
    return _send(token, chat_id, text)


def send_close(token: str, chat_id: str, symbol: str, pnl: float = None) -> bool:
    """포지션 청산 알림"""
    now  = datetime.now().strftime("%m/%d %H:%M")
    text = f"🔵 <b>[청산] {symbol} 포지션 청산 완료</b>\n🕐 {now}"
    if pnl is not None:
        color = "🟢" if pnl >= 0 else "🔴"
        text += f"\n{color} 손익: <code>${pnl:+.2f}</code>"
    return _send(token, chat_id, text)


def send_progress(token: str, chat_id: str, symbol: str,
                  stage: str, detail: str = "", is_start: bool = False) -> bool:
    """분석 진행 상황 알림"""
    sep = "─" * 20
    if is_start:
        text = f"{sep}\n{stage} <b>[{symbol}]</b>"
    else:
        text = f"{stage} <b>[{symbol}]</b>"
    if detail:
        text += f"\n<i>{detail}</i>"
    if is_start:
        text += f"\n{sep}"
    return _send(token, chat_id, text)


def send_error(token: str, chat_id: str, msg: str) -> bool:
    """오류 알림"""
    now  = datetime.now().strftime("%m/%d %H:%M")
    text = f"⚠️ <b>[오류]</b> {now}\n<code>{msg[:300]}</code>"
    return _send(token, chat_id, text)


def send_analysis_summary(token: str, chat_id: str, symbol: str,
                           decision: str, price: float = 0,
                           confluence: str = "", rl_label: str = "",
                           rsi: float = None, macd_hist: float = None,
                           analyst_summary: str = "") -> bool:
    """분석 완료 요약 알림"""
    if "롱 진입" in decision:
        emoji, side = "🟢", "롱 진입"
    elif "숏 진입" in decision:
        emoji, side = "🔴", "숏 진입"
    else:
        emoji, side = "⚪", "관망"
    now  = datetime.now().strftime("%m/%d %H:%M")
    text = (
        f"{emoji} <b>[{symbol}] AI 분석 완료</b>\n"
        f"🕐 {now}\n"
        f"📋 결정: <b>{side}</b>"
    )
    if price:
        text += f"\n💰 현재가: <code>${price:,.2f}</code>"
    if confluence:
        text += f"\n📡 컨플루언스: {confluence}"
    if rl_label:
        text += f"\n🤖 PPO: {rl_label}"
    if rsi is not None:
        text += f"\n📊 RSI: {rsi}"
        if macd_hist is not None:
            text += f" | MACD Hist: {macd_hist:+.4f}"
    if analyst_summary:
        snippet = analyst_summary[:200].replace("\n", " ").strip()
        text += f"\n\n💬 <i>{snippet}...</i>"
    return _send(token, chat_id, text)


def send_status(token: str, chat_id: str, balance: dict, positions: list,
                paused: bool, daily_loss: float, daily_limit: float) -> bool:
    """현재 봇 상태 알림 (/status 명령어 응답)"""
    now    = datetime.now().strftime("%m/%d %H:%M")
    status = "⏸ 일시 중지" if paused else "🟢 실행 중"
    text   = (
        f"📊 <b>봇 상태</b> — {now}\n"
        f"상태: {status}\n"
        f"💰 잔고: <code>${balance.get('total', 0):,.2f}</code> "
        f"(가용: <code>${balance.get('available', 0):,.2f}</code>)\n"
        f"📉 오늘 손실: <code>${daily_loss:,.2f}</code> / 한도 ${daily_limit:,.0f}\n"
        f"📌 포지션: {len(positions)}개"
    )
    for p in positions:
        pnl = p.get("unrealized_pnl", 0)
        c   = "🟢" if pnl >= 0 else "🔴"
        text += (
            f"\n  {c} {p['symbol']} {p['side']} "
            f"{p['size']} @ ${p['entry_price']:,.2f} "
            f"(PnL: ${pnl:+.2f})"
        )
    return _send(token, chat_id, text)


def send_trader_decision(token: str, chat_id: str, symbol: str, tj: dict,
                          confidence: int = 0, rl_label: str = "") -> bool:
    """트레이더 결정 카드 알림 (분석 완료 시)"""
    signal = tj.get("signal", "wait")
    if signal == "long":
        emoji, side = "🟢", "롱 진입"
        sig_bar = "🟩🟩🟩🟩🟩"
    elif signal == "short":
        emoji, side = "🔴", "숏 진입"
        sig_bar = "🟥🟥🟥🟥🟥"
    else:
        emoji, side = "⚪", "관망"
        sig_bar = "⬜⬜⬜⬜⬜"

    conf = tj.get("confidence", confidence)
    conf_bar = "🟦" * (conf // 20) + "⬜" * (5 - conf // 20)
    now = datetime.now().strftime("%m/%d %H:%M")

    text = (
        f"{emoji} <b>[{symbol}] {side}</b>\n"
        f"🕐 {now}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📊 신뢰도: {conf_bar} <b>{conf}%</b>\n"
    )
    if tj.get("entry"):
        text += f"💰 진입가: <code>${tj['entry']:,.2f}</code>\n"
    if tj.get("sl"):
        text += f"🛑 SL: <code>${tj['sl']:,.2f}</code>\n"
    if tj.get("tp"):
        text += f"🎯 TP: <code>${tj['tp']:,.2f}</code>\n"
    if rl_label:
        text += f"🤖 PPO: {rl_label}\n"
    if tj.get("reason"):
        snippet = tj["reason"][:200].strip()
        text += f"━━━━━━━━━━━━━━\n💬 <i>{snippet}</i>\n"
    if tj.get("condition"):
        cond = tj["condition"][:150].strip()
        text += f"⚠️ <i>{cond}</i>"
    return _send(token, chat_id, text)


def send_daily_limit_alert(token: str, chat_id: str, loss: float, limit: float) -> bool:
    """일일 손실 한도 초과 알림"""
    now  = datetime.now().strftime("%m/%d %H:%M")
    text = (
        f"🚨 <b>[경고] 일일 손실 한도 초과</b>\n"
        f"🕐 {now}\n"
        f"손실: <code>${loss:,.2f}</code> / 한도: <code>${limit:,.0f}</code>\n"
        f"⏸ 자동 거래가 중단되었습니다."
    )
    return _send(token, chat_id, text)


# ── 텔레그램 명령어 폴링 ─────────────────────────────────────────────
def _write_command(cmd: dict):
    """명령어를 파일에 큐잉 (스레드 안전)"""
    with _cmd_lock:
        try:
            cmds = []
            if _CMD_FILE.exists():
                cmds = json.loads(_CMD_FILE.read_text())
            cmds.append(cmd)
            _CMD_FILE.write_text(json.dumps(cmds))
        except Exception:
            pass


def poll_commands(token: str, chat_id: str):
    """
    텔레그램 메시지 폴링 — 백그라운드 스레드에서 실행.
    허가된 chat_id의 메시지만 처리.
    지원 명령어: /status /pause /resume /close <SYMBOL>
    """
    if not token or not chat_id:
        return

    offset = 0
    while True:
        try:
            url  = f"https://api.telegram.org/bot{token}/getUpdates"
            resp = requests.get(url, params={"offset": offset, "timeout": 20}, timeout=25)
            if resp.status_code != 200:
                time.sleep(5)
                continue
            updates = resp.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})
                # 허가된 chat_id만 처리
                if str(msg.get("chat", {}).get("id", "")) != str(chat_id):
                    continue
                text = msg.get("text", "").strip().lower()
                if text == "/status":
                    _write_command({"cmd": "status"})
                elif text == "/pause":
                    _write_command({"cmd": "pause"})
                    _send(token, chat_id, "⏸ <b>자동 거래 일시 중지됨</b>")
                elif text == "/resume":
                    _write_command({"cmd": "resume"})
                    _send(token, chat_id, "▶️ <b>자동 거래 재개됨</b>")
                elif text.startswith("/close"):
                    parts = text.split()
                    if len(parts) < 2:
                        _send(token, chat_id, "⚠️ 심볼을 지정하세요: /close ETHUSDT")
                        continue
                    sym = parts[1].upper()
                    _write_command({"cmd": "close", "symbol": sym})
                    _send(token, chat_id, f"🔵 <b>{sym} 청산 명령 접수됨</b>")
                elif text == "/help":
                    _send(token, chat_id,
                        "📋 <b>명령어 목록</b>\n"
                        "/status — 현재 상태\n"
                        "/pause — 자동 거래 중지\n"
                        "/resume — 자동 거래 재개\n"
                        "/close ETHUSDT — 포지션 청산"
                    )
        except Exception as e:
            print(f"⚠️ 텔레그램 폴링 오류: {e}")
        time.sleep(3)


def start_polling_thread(token: str, chat_id: str) -> threading.Thread:
    """폴링 스레드 시작 (데몬 스레드)"""
    t = threading.Thread(target=poll_commands, args=(token, chat_id), daemon=True)
    t.start()
    return t


def read_and_clear_commands() -> list:
    """명령어 파일 읽고 비우기 (스레드 안전)"""
    with _cmd_lock:
        try:
            if not _CMD_FILE.exists():
                return []
            cmds = json.loads(_CMD_FILE.read_text())
            _CMD_FILE.write_text("[]")
            return cmds
        except Exception:
            return []

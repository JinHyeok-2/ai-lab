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


def send_message(token: str, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
    """텔레그램 메시지 전송 (공개 API)"""
    return _send(token, chat_id, text, parse_mode)


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
               sl: float = None, tp: float = None,
               balance: float = None, used_usdt: float = None,
               leverage: int = None, reason: str = None) -> bool:
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
    if used_usdt is not None:
        text += f"\n💵 진입금: <code>${used_usdt:,.1f}</code>"
        if leverage:
            text += f" (x{leverage})"
    if balance is not None:
        _remain = balance - (used_usdt or 0)
        text += f"\n🏦 잔고: <code>${balance:,.2f}</code> → 잔여 <code>${_remain:,.2f}</code>"
    if reason:
        text += f"\n📝 <b>근거:</b> {reason}"
    return _send(token, chat_id, text)


def send_close(token: str, chat_id: str, symbol: str, pnl: float = None,
               balance: float = None, daily_pnl: float = None) -> bool:
    """포지션 청산 알림"""
    now  = datetime.now().strftime("%m/%d %H:%M")
    if pnl is not None and pnl >= 0:
        label = "익절"
        icon = "🟢"
    else:
        label = "손절"
        icon = "🔴"
    text = f"{icon} <b>[{label}] {symbol}</b>\n🕐 {now}"
    if pnl is not None:
        text += f"\n{icon} 손익: <code>${pnl:+.2f}</code>"
    if balance is not None:
        text += f"\n🏦 잔고: <code>${balance:,.2f}</code>"
    if daily_pnl is not None:
        text += f"\n📊 오늘 누적: <code>${daily_pnl:+.2f}</code>"
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
                           analyst_summary: str = "",
                           gate_passed: bool = None, gate_reason: str = "",
                           confidence: int = 0) -> bool:
    """분석 완료 요약 알림 (gate 결과 포함)"""
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
    if confidence:
        text += f" (신뢰도: {confidence}%)"
    if price:
        text += f"\n💰 현재가: <code>${price:,.2f}</code>"
    if confluence:
        text += f"\n📡 컨플루언스: {confluence}"
    if gate_passed is not None:
        gate_icon = "✅ 통과" if gate_passed else "🚫 차단"
        text += f"\n🚦 게이트: <b>{gate_icon}</b>"
        if gate_reason:
            text += f"\n   └ {gate_reason[:100]}"
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
        text += f"🤖 RL앙상블: {rl_label}\n"
    if tj.get("reason"):
        snippet = tj["reason"][:200].strip()
        text += f"━━━━━━━━━━━━━━\n💬 <i>{snippet}</i>\n"
    if tj.get("condition"):
        cond = tj["condition"][:150].strip()
        text += f"⚠️ <i>{cond}</i>"
    return _send(token, chat_id, text)


def send_hourly_briefing(token: str, chat_id: str, balance: dict,
                          positions: list, closed_today: list = None,
                          daily_pnl: float = 0, gate_pass: int = 0,
                          gate_block: int = 0, trades_today: int = 0) -> bool:
    """매 정시 포지션 브리핑"""
    now = datetime.now().strftime("%m/%d %H:%M")
    total = balance.get("total", 0)
    avail = balance.get("available", 0)
    upnl = balance.get("unrealized_pnl", 0)

    text = (
        f"📋 <b>[정시 브리핑]</b> {now}\n"
        f"━━━━━━━━━━━━━━\n"
        f"💰 잔고: <code>${total:,.1f}</code> (가용 <code>${avail:,.1f}</code>)\n"
        f"📈 미실현 PnL: <code>${upnl:+,.2f}</code>\n"
        f"📊 오늘 실현 PnL: <code>${daily_pnl:+,.2f}</code>\n"
    )

    # 포지션 상세
    if positions:
        text += f"\n📌 <b>보유 포지션 ({len(positions)}개)</b>\n"
        for p in positions:
            pnl = p.get("unrealized_pnl", 0)
            icon = "🟢" if pnl >= 0 else "🔴"
            side_kr = "롱" if p["side"] == "LONG" else "숏"
            pnl_pct = (pnl / (p["entry_price"] * p["size"] / p.get("leverage", 1))) * 100 if p["entry_price"] * p["size"] > 0 else 0
            text += (
                f"  {icon} <b>{p['symbol']}</b> {side_kr}\n"
                f"     진입 <code>${p['entry_price']:,.2f}</code> | "
                f"PnL <code>${pnl:+.2f}</code> ({pnl_pct:+.1f}%)\n"
            )
    else:
        text += f"\n📌 보유 포지션 없음\n"

    # 오늘 청산된 거래
    if closed_today:
        _wins = sum(1 for c in closed_today if (c.get("pnl") or 0) > 0)
        _total = len(closed_today)
        _sum_pnl = sum(c.get("pnl") or 0 for c in closed_today)
        text += (
            f"\n📊 <b>오늘 마감</b>: {_total}건 "
            f"(익절 {_wins} / 손절 {_total - _wins}) "
            f"합계 <code>${_sum_pnl:+,.2f}</code>\n"
        )

    # 활동 요약
    _total_analysis = gate_pass + gate_block
    text += (
        f"\n🔄 분석 {_total_analysis}회 (진입 판단 {gate_pass} / 관망 판단 {gate_block})\n"
        f"📝 오늘 거래: {trades_today}건\n"
        f"━━━━━━━━━━━━━━"
    )
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


def send_weekly_report(token: str, chat_id: str, stats: dict,
                       balance_hist: list = None, start_bal: float = 0) -> bool:
    """주간 성과 리포트 (매주 일요일 자정 발송)"""
    now = datetime.now().strftime("%m/%d %H:%M")
    total = stats.get("total", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    wr = stats.get("win_rate", 0)
    pf = stats.get("profit_factor", 0)
    total_pnl = stats.get("total_pnl", 0)
    mdd = stats.get("mdd", 0)
    avg_win = stats.get("avg_win", 0)
    avg_loss = stats.get("avg_loss", 0)

    # 주간 잔고 변화
    _week_pnl = 0
    _week_start = start_bal
    _week_end = start_bal
    if balance_hist and len(balance_hist) >= 1:
        _week_end = balance_hist[0].get("close_bal") or balance_hist[0].get("open_bal", 0)
        _oldest = balance_hist[-1]
        _week_start = _oldest.get("open_bal", _week_end)
        _week_pnl = round(_week_end - _week_start, 2)

    _pnl_icon = "📈" if _week_pnl >= 0 else "📉"
    _pnl_pct = (_week_pnl / _week_start * 100) if _week_start > 0 else 0

    text = (
        f"📊 <b>[주간 리포트]</b> {now}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"\n💰 <b>자금 현황</b>\n"
        f"  시작: <code>${_week_start:,.2f}</code>\n"
        f"  현재: <code>${_week_end:,.2f}</code>\n"
        f"  {_pnl_icon} 주간 PnL: <code>${_week_pnl:+,.2f}</code> ({_pnl_pct:+.1f}%)\n"
        f"\n📋 <b>거래 통계 (전체)</b>\n"
        f"  거래: {total}건 (익절 {wins} / 손절 {losses})\n"
        f"  승률: <code>{wr:.1f}%</code>\n"
        f"  PF: <code>{pf:.2f}</code>\n"
        f"  평균 수익: <code>${avg_win:+.2f}</code>\n"
        f"  평균 손실: <code>${avg_loss:+.2f}</code>\n"
        f"  MDD: <code>${mdd:.2f}</code>\n"
    )

    # 일별 PnL 막대
    if balance_hist:
        text += f"\n📅 <b>일별 PnL</b>\n"
        for day in reversed(balance_hist[:7]):
            d = day.get("date", "")[-5:]  # MM-DD
            p = day.get("pnl", 0) or 0
            t = day.get("trades", 0) or 0
            icon = "🟢" if p >= 0 else "🔴"
            bar = "█" * min(int(abs(p) / 0.5 + 0.5), 10)
            text += f"  {d} {icon} <code>${p:+6.2f}</code> {bar} ({t}건)\n"

    text += f"\n━━━━━━━━━━━━━━━━━━"
    return _send(token, chat_id, text)


def send_pnl_report(token: str, chat_id: str, today_trades: list,
                    total_pnl: float, daily_pnl: float, stats: dict) -> bool:
    """오늘 거래 내역 + 누적 손익 (/pnl 명령어 응답)"""
    now = datetime.now().strftime("%m/%d %H:%M")
    text = (
        f"💹 <b>[PnL 리포트]</b> {now}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📊 오늘 PnL: <code>${daily_pnl:+.2f}</code>\n"
        f"📈 누적 PnL: <code>${total_pnl:+.2f}</code>\n"
        f"🏆 승률: <code>{stats.get('win_rate', 0):.1f}%</code> | "
        f"PF: <code>{stats.get('profit_factor', 0):.2f}</code>\n"
    )
    if today_trades:
        text += f"\n📝 <b>오늘 거래 ({len(today_trades)}건)</b>\n"
        for t in today_trades[:10]:
            pnl = t.get("pnl") or 0
            icon = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⏳"
            sym = t.get("symbol", "?")
            side = t.get("side", "?")
            text += f"  {icon} {sym} {side} <code>${pnl:+.2f}</code>\n"
    else:
        text += "\n📝 오늘 거래 없음\n"
    text += f"━━━━━━━━━━━━━━"
    return _send(token, chat_id, text)


def send_balance_report(token: str, chat_id: str, balance: dict,
                        start_bal: float, balance_hist: list = None) -> bool:
    """잔고 추이 리포트 (/balance 명령어 응답)"""
    now = datetime.now().strftime("%m/%d %H:%M")
    total = balance.get("total", 0)
    avail = balance.get("available", 0)
    change = total - start_bal
    pct = (change / start_bal * 100) if start_bal > 0 else 0
    _icon = "📈" if change >= 0 else "📉"

    text = (
        f"🏦 <b>[잔고 리포트]</b> {now}\n"
        f"━━━━━━━━━━━━━━\n"
        f"💰 현재: <code>${total:,.2f}</code> (가용: <code>${avail:,.2f}</code>)\n"
        f"🏁 시작: <code>${start_bal:,.2f}</code>\n"
        f"{_icon} 변동: <code>${change:+,.2f}</code> ({pct:+.1f}%)\n"
    )
    if balance_hist:
        text += f"\n📅 <b>최근 잔고</b>\n"
        for day in balance_hist[:7]:
            d = day.get("date", "")[-5:]
            cb = day.get("close_bal") or day.get("open_bal", 0)
            p = day.get("pnl", 0) or 0
            icon = "🟢" if p >= 0 else "🔴"
            text += f"  {d} {icon} <code>${cb:,.2f}</code> ({p:+.2f})\n"
    text += f"━━━━━━━━━━━━━━"
    return _send(token, chat_id, text)


def send_daily_close_report(token: str, chat_id: str, balance: float,
                            start_bal: float, trades_today: list,
                            daily_pnl: float, stats: dict) -> bool:
    """일일 마감 리포트 (23:59 자동 발송)"""
    now = datetime.now().strftime("%m/%d")
    change = balance - start_bal
    pct = (change / start_bal * 100) if start_bal > 0 else 0
    _icon = "📈" if change >= 0 else "📉"
    wins = sum(1 for t in trades_today if (t.get("pnl") or 0) > 0)
    losses = len(trades_today) - wins

    text = (
        f"🌙 <b>[일일 마감]</b> {now}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{_icon} 오늘 PnL: <code>${daily_pnl:+.2f}</code>\n"
        f"💰 잔고: <code>${balance:,.2f}</code> ({pct:+.1f}%)\n"
        f"📝 거래: {len(trades_today)}건 (익절 {wins} / 손절 {losses})\n"
    )
    if trades_today:
        text += f"\n📋 <b>거래 내역</b>\n"
        for t in trades_today[:15]:
            pnl = t.get("pnl") or 0
            icon = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⏳"
            text += f"  {icon} {t.get('symbol','?')} {t.get('side','?')} <code>${pnl:+.2f}</code>\n"
    text += (
        f"\n📊 <b>전체 통계</b>\n"
        f"  승률: {stats.get('win_rate',0):.1f}% | PF: {stats.get('profit_factor',0):.2f}\n"
        f"  MDD: ${stats.get('mdd',0):.2f}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )
    return _send(token, chat_id, text)


def send_consec_loss_alert(token: str, chat_id: str, streak: int,
                           new_usdt: float, original_usdt: float) -> bool:
    """연속 손실 경고 + 진입금 축소 알림"""
    text = (
        f"🚨 <b>[연속 손실 경고]</b>\n"
        f"━━━━━━━━━━━━━━\n"
        f"❌ {streak}회 연속 손실 발생!\n"
        f"💵 진입금 자동 축소: <code>${original_usdt:.0f}</code> → <code>${new_usdt:.0f}</code>\n"
        f"⚠️ 연승 2회 달성 시 원래 금액으로 복원됩니다."
    )
    return _send(token, chat_id, text)


def send_image(token: str, chat_id: str, image_path: str, caption: str = "") -> bool:
    """텔레그램 이미지 전송"""
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(image_path, "rb") as f:
            resp = requests.post(url, data={
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML",
            }, files={"photo": f}, timeout=15)
        return resp.status_code == 200
    except Exception:
        return False


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
                elif text == "/pnl":
                    _write_command({"cmd": "pnl"})
                elif text == "/balance":
                    _write_command({"cmd": "balance"})
                elif text == "/help":
                    _send(token, chat_id,
                        "📋 <b>명령어 목록</b>\n"
                        "/status — 현재 상태\n"
                        "/pnl — 오늘 거래 + 손익\n"
                        "/balance — 잔고 추이\n"
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

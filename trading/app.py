#!/usr/bin/env python3
# 멀티에이전트 선물거래 봇 — Streamlit GUI
# 실행: streamlit run trading/app.py

import streamlit as st
import pandas as pd
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go

# ── 페이지 설정 ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 트레이딩 봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #111; }
[data-testid="stSidebar"] { background-color: #f5f5f5; }
[data-testid="stSidebar"] * { color: #111 !important; }
p, span, div, label, h1, h2, h3, h4, h5 { color: #111; }

.header-box {
    background: linear-gradient(135deg, #0d1b2a, #1b3a5c);
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 20px;
}
.header-box h2 { color: #fff !important; margin: 0; }
.header-box p  { color: #aac4e0 !important; margin: 4px 0 0 0; font-size: 14px; }

.agent-card {
    background: #f8f9ff;
    border-left: 4px solid #ccc;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 13px;
}
.agent-card.working {
    border-left-color: #FF9800;
    background: #fff8ee;
    animation: pulse 1.5s infinite;
}
.agent-card.done { border-left-color: #4CAF50; background: #f0fff4; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

.metric-box {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    border: 1px solid #d0d7f0;
}
.metric-box .value { font-size: 22px; font-weight: 700; color: #1a237e; }
.metric-box .label { font-size: 12px; color: #888; margin-top: 2px; }

.decision-long  { background:#e8f5e9; border:2px solid #4CAF50; border-radius:12px; padding:16px; }
.decision-short { background:#fce4ec; border:2px solid #e53935; border-radius:12px; padding:16px; }
.decision-wait  { background:#f5f5f5; border:2px solid #9e9e9e; border-radius:12px; padding:16px; }

.log-box {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px;
    font-size: 13px;
    max-height: 260px;
    overflow-y: auto;
}
.stButton > button {
    background: #fff; color: #1a237e;
    border: 1.5px solid #1a237e;
    border-radius: 8px; font-weight: 600;
}
.stButton > button:hover { background: #e8ecff; }
</style>
""", unsafe_allow_html=True)

# ── import ──────────────────────────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent))
    from binance_client import get_klines, get_price, get_balance, get_positions, place_order, close_position, get_funding_rate, get_recent_trades, update_stop_loss, get_funding_rate_history, get_open_interest, partial_close_position
    from indicators import calc_indicators, format_for_agent
    from agents import run_agent
    from config import SYMBOLS, LEVERAGE, MAX_USDT, INTERVAL, CANDLE_CNT, TESTNET
    from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ATR_SL_MULT, ATR_TP_MULT
    from telegram_notifier import (
        test_connection, send_signal, send_order, send_close, send_error,
        send_progress, send_analysis_summary, send_status, send_daily_limit_alert,
        send_trader_decision, start_polling_thread, read_and_clear_commands
    )
    from config import MAX_DAILY_LOSS
    from config import MAX_USDT_ALT, MAX_ALT_POSITIONS, ALT_LEVERAGE, ALT_ATR_SL_MULT, ALT_ATR_TP_MULT, ALT_SCAN_LIMIT, ALT_MIN_SCORE, ALT_AUTO_CONFIDENCE, ALT_MANUAL_MIN_CONF
    from alt_scanner import get_alt_futures_symbols, screen_altcoins, check_binance_announcements, run_alt_analysis
    BINANCE_READY = True
except Exception as e:
    BINANCE_READY = False
    BINANCE_ERR = str(e)
    SYMBOLS      = ["ETHUSDT", "BTCUSDT"]
    LEVERAGE     = 3
    MAX_USDT     = 100
    INTERVAL     = "15m"
    CANDLE_CNT   = 100
    TESTNET      = True
    TELEGRAM_TOKEN   = ""
    TELEGRAM_CHAT_ID = ""
    ATR_SL_MULT    = 1.5
    ATR_TP_MULT    = 3.0
    MAX_DAILY_LOSS = 200
    MAX_USDT_ALT      = 50
    MAX_ALT_POSITIONS = 2
    ALT_LEVERAGE      = 2
    ALT_ATR_SL_MULT   = 1.0
    ALT_ATR_TP_MULT   = 2.0
    ALT_SCAN_LIMIT    = 50
    ALT_MIN_SCORE          = 30
    ALT_AUTO_CONFIDENCE    = 75
    ALT_MANUAL_MIN_CONF    = 60

# ── 세션 초기화 ──────────────────────────────────────────────────────
# ── 파일 기반 캐시 (새로고침 후에도 분석 결과 유지) ─────────────────
_CACHE_PATH   = Path(__file__).parent / "last_analysis.json"
_HISTORY_PATH = Path(__file__).parent / "analysis_history.json"
_LOG_PATH     = Path(__file__).parent / "trading.log"

# ── 파일 로거 설정 ────────────────────────────────────────────────────
_file_logger = logging.getLogger("trading_bot")
if not _file_logger.handlers:
    _file_logger.setLevel(logging.INFO)
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _file_logger.addHandler(_fh)

def _log(msg: str, level: str = "info"):
    """UI 로그 + 파일 로그 동시 기록"""
    if level == "error":
        _file_logger.error(msg)
    else:
        _file_logger.info(msg)

def _save_analysis(data: dict):
    """last_analysis를 JSON 파일로 저장 (df 제외)"""
    try:
        save = {}
        for sym, res in data.items():
            save[sym] = {k: v for k, v in res.items()
                         if k != "df" and not isinstance(v, pd.DataFrame)}
        _CACHE_PATH.write_text(json.dumps(save, ensure_ascii=False, indent=2))
    except Exception:
        pass

def _load_analysis() -> dict:
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except Exception:
        pass
    return {}

def _save_history(history: list):
    try:
        _HISTORY_PATH.write_text(json.dumps(history[-100:], ensure_ascii=False, indent=2))
    except Exception:
        pass

def _load_history() -> list:
    try:
        if _HISTORY_PATH.exists():
            return json.loads(_HISTORY_PATH.read_text())
    except Exception:
        pass
    return []

_JOURNAL_PATH = Path(__file__).parent / "trade_journal.json"

def _load_journal() -> list:
    try:
        if _JOURNAL_PATH.exists():
            return json.loads(_JOURNAL_PATH.read_text())
    except Exception:
        pass
    return []

def _save_journal(journal: list):
    try:
        _JOURNAL_PATH.write_text(json.dumps(journal[-200:], ensure_ascii=False, indent=2))
    except Exception:
        pass

def _add_journal_entry(entry: dict):
    j = _load_journal()
    j.insert(0, entry)
    _save_journal(j)

def _update_journal_pnl(symbol: str, pnl: float, close_price: float = None):
    """심볼의 가장 최근 미청산(pnl=None) journal 항목에 PnL 기록"""
    j = _load_journal()
    for entry in j:
        if entry.get("symbol") == symbol and entry.get("pnl") is None:
            entry["pnl"]        = round(pnl, 4)
            entry["action"]     = "청산"
            entry["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if close_price is not None:
                entry["close_price"] = close_price
            break
    _save_journal(j)

def _check_sl_tp_closed():
    """SL/TP 자동 청산 감지 — pnl=None 항목 중 포지션이 사라진 심볼 업데이트"""
    j = _load_journal()
    open_entries = [e for e in j if e.get("pnl") is None and e.get("symbol")]
    if not open_entries:
        return
    try:
        active_syms = {p["symbol"] for p in get_positions()}
        updated = False
        for entry in open_entries:
            sym = entry["symbol"]
            if sym in active_syms:
                continue  # 아직 포지션 열려있음
            # 포지션 사라짐 → SL/TP 또는 다른 방법으로 청산됨
            trades = get_recent_trades(sym, limit=30)
            entry_time = entry.get("time", "")
            # entry_time 이후 체결된 trade들만 필터링 (심볼별 독립 조회)
            recent_trades = [t for t in trades if t["time"] >= entry_time]
            total_rpnl = sum(
                t["realized_pnl"] for t in recent_trades
                if t["realized_pnl"] != 0
            )
            if total_rpnl == 0 and trades:
                # realized_pnl이 0이면 가장 최신 체결가 기반으로 추정
                # futures_account_trades는 oldest first → trades[-1]이 최신
                entry_price = entry.get("price", 0)
                qty         = entry.get("qty", 0)
                side        = entry.get("side", "")
                last_trade  = recent_trades[-1] if recent_trades else trades[-1]
                close_price = last_trade["price"]
                if "롱" in side:
                    total_rpnl = round((close_price - entry_price) * qty, 4)
                elif "숏" in side:
                    total_rpnl = round((entry_price - close_price) * qty, 4)
            close_px = (recent_trades[-1]["price"] if recent_trades
                        else trades[-1]["price"] if trades else None)
            _update_journal_pnl(sym, total_rpnl, close_px)
            _log(f"📋 {sym} SL/TP 자동 청산 감지: PnL ${total_rpnl:+.4f}")
            # 즉시 재분석 큐에 추가
            _q = st.session_state.get("immediate_reanalyze", [])
            if sym not in _q:
                _q.append(sym)
            st.session_state.immediate_reanalyze = _q
            # 부분 청산 플래그 초기화 (다음 포지션에 이월 방지)
            _done = st.session_state.get("partial_tp_done", {})
            _done.pop(sym, None)
            st.session_state.partial_tp_done = _done
            updated = True
        if updated:
            pass
    except Exception as e:
        _log(f"SL/TP 감지 오류: {e}", "error")


def _update_trailing_stop():
    """트레일링 스탑: 미실현 수익이 ATR×배수 이상이면 SL을 break-even으로 이동"""
    if not st.session_state.get("trailing_stop_on", True):
        return
    try:
        positions = get_positions()
        if not positions:
            return
        j = _load_journal()
        for pos in positions:
            sym   = pos["symbol"]
            upnl  = pos["unrealized_pnl"]
            ep    = pos["entry_price"]
            side  = pos["side"]
            psize = pos["size"]
            if upnl <= 0:
                continue  # 손실 중이면 스킵
            # 거래 일지에서 해당 포지션의 ATR 조회
            open_entry = next(
                (e for e in j if e.get("symbol") == sym and e.get("pnl") is None),
                None
            )
            if not open_entry:
                continue
            atr = open_entry.get("atr", 0) or 0
            if atr <= 0:
                continue
            _mult = st.session_state.get("trailing_atr_mult", 1.0)
            # 수익이 ATR × 배수 × 수량 이상이면 break-even SL 이동
            if upnl >= atr * _mult * psize:
                # break-even: 진입가 ± 0.1% 여유 (수수료 고려)
                if side == "LONG":
                    new_sl = round(ep * 1.001, 2)
                else:
                    new_sl = round(ep * 0.999, 2)
                # 이미 break-even 이상으로 SL이 설정된 경우 스킵
                current_sl = open_entry.get("sl")
                if current_sl:
                    if side == "LONG" and float(current_sl) >= ep:
                        continue
                    if side == "SHORT" and float(current_sl) <= ep:
                        continue
                r = update_stop_loss(sym, new_sl, side)
                if r["success"]:
                    _log(f"🔄 {sym} 트레일링: SL → break-even ${new_sl:.2f} (수익 ${upnl:+.2f})")
                    add_log(f"🔄 {sym} SL → break-even ${new_sl:.2f}")
                else:
                    _log(f"트레일링 SL 업데이트 실패 {sym}: {r['error']}", "error")
    except Exception as e:
        _log(f"트레일링 스탑 오류: {e}", "error")


def _check_partial_tp():
    """TP 도달 시 50% 부분 청산 — 나머지는 break-even SL로 유지"""
    if not st.session_state.get("partial_tp_on", True):
        return
    if not BINANCE_READY:
        return
    try:
        positions = get_positions()
        if not positions:
            return
        j = _load_journal()
        for pos in positions:
            sym  = pos["symbol"]
            side = pos["side"]
            ep   = pos["entry_price"]
            upnl = pos["unrealized_pnl"]
            psize = pos["size"]
            cur_price = ep + (upnl / psize) if psize > 0 else ep

            # 이미 부분 청산한 포지션은 스킵
            if st.session_state.get("partial_tp_done", {}).get(sym):
                continue

            open_entry = next(
                (e for e in j if e.get("symbol") == sym and e.get("pnl") is None),
                None
            )
            if not open_entry:
                continue
            tp = open_entry.get("tp")
            if not tp:
                continue
            tp = float(tp)

            # TP 도달 여부 확인
            tp_reached = (side == "LONG"  and cur_price >= tp) or \
                         (side == "SHORT" and cur_price <= tp)
            if not tp_reached:
                continue

            # 50% 부분 청산 실행
            r = partial_close_position(sym, close_pct=0.5)
            if r["success"]:
                _log(f"🎯 {sym} TP 50% 부분 청산 완료 (qty: {r['qty']})")
                add_log(f"🎯 {sym} TP 도달 → 50% 부분 청산 | 수익 ${upnl:+.2f}")
                # 부분 청산 완료 플래그
                _done = st.session_state.get("partial_tp_done", {})
                _done[sym] = True
                st.session_state.partial_tp_done = _done
                # SL → break-even 이동
                new_sl = round(ep * 1.001, 2) if side == "LONG" else round(ep * 0.999, 2)
                update_stop_loss(sym, new_sl, side)
                add_log(f"🔄 {sym} SL → break-even ${new_sl:.2f}")
                if st.session_state.get("tg_notify"):
                    send_error(st.session_state.tg_token, st.session_state.tg_chat_id,
                               f"🎯 {sym} TP 50% 부분 청산 | ${upnl:+.2f}")
            else:
                _log(f"부분 청산 실패 {sym}: {r.get('error','')}", "error")
    except Exception as e:
        _log(f"부분 청산 체크 오류: {e}", "error")


def _get_consec_losses() -> int:
    """최근 연속 손실 횟수 반환 (청산 완료된 항목 기준)"""
    j = _load_journal()
    closed = [e for e in j if e.get("pnl") is not None]
    count = 0
    for entry in closed:
        if entry["pnl"] < 0:
            count += 1
        else:
            break
    return count


def _calc_dynamic_params() -> dict:
    """거래 일지 + 분석 히스토리 기반 리스크 파라미터 자동 계산 (Kelly/확률론)"""
    import math, statistics as _st

    j      = _load_journal()
    closed = [e for e in j if e.get("pnl") is not None]
    hist   = _load_history()

    # ── 기본값 (데이터 3건 미만 시 사용) ──
    result = {
        "position_pct":       15,
        "consec_loss_count":   3,
        "consec_loss_hours":   2,
        "atr_volatility_mult": 2.0,
        "trailing_atr_mult":   1.0,
        "early_exit_conf":     70,
        "kelly":     0.0,
        "win_rate":  0.5,
        "rr_ratio":  2.0,
        "basis": "기본값 (거래 데이터 부족)",
    }
    if len(closed) < 3:
        return result

    wins   = [e for e in closed if e["pnl"] > 0]
    losses = [e for e in closed if e["pnl"] < 0]
    win_rate = len(wins) / len(closed)
    avg_win  = (_st.mean(e["pnl"] for e in wins)      if wins   else 1.0)
    avg_loss = abs(_st.mean(e["pnl"] for e in losses) if losses else 0.5)
    rr_ratio = avg_win / max(avg_loss, 0.001)

    # ── Kelly Criterion: (p×b − q) / b ──
    q     = 1 - win_rate
    kelly = max(0.0, (win_rate * rr_ratio - q) / rr_ratio)
    kelly = min(kelly, 0.5)   # 50% 상한 (안전 캡)

    # 1. 포지션 비율 — Half-Kelly × 100 (5~40% 범위)
    result["position_pct"] = max(5, min(40, round(kelly * 50)))

    # 2. 연속 손실 횟수 — P(n연속 손실) < 10% 되는 최소 n
    #    (1−p)^n < 0.1  →  n = ceil(log(0.1) / log(1−p))
    if 0 < win_rate < 1:
        n_raw = math.log(0.10) / math.log(max(1 - win_rate, 1e-6))
        result["consec_loss_count"] = max(3, min(6, math.ceil(n_raw)))

    # 3. 쿨다운 시간 — 승률 역비례 (낮을수록 길게, 1~8h)
    #    hours = 2 + (0.5 − win_rate) × 8
    result["consec_loss_hours"] = max(1, min(8, round(2 + (0.5 - win_rate) * 8)))

    # 4. 트레일링 스탑 발동 — TP 경로의 (1−Kelly) 비율 지점
    #    Kelly 높을수록 수익 확신 → 더 오래 기다렸다가 발동
    result["trailing_atr_mult"] = round(
        max(0.5, min(2.5, ATR_TP_MULT * max(0.2, 1 - kelly))), 1
    )

    # 5. 조기 청산 신뢰도 임계값
    #    신뢰도 기록 있으면: 진입 confidence P80 사용
    #    없으면: 분석 히스토리 trader_conf P75 + 5pt
    conf_entries = sorted(e["confidence"] for e in closed if e.get("confidence") is not None)
    if len(conf_entries) >= 3:
        idx = int(len(conf_entries) * 0.80)
        result["early_exit_conf"] = max(60, min(85, conf_entries[min(idx, len(conf_entries)-1)]))
    else:
        hconfs = sorted(h.get("trader_conf", 0) for h in hist if h.get("trader_conf"))
        if len(hconfs) >= 10:
            idx = int(len(hconfs) * 0.75)
            result["early_exit_conf"] = max(60, min(85, hconfs[idx] + 5))

    result["kelly"]    = kelly
    result["win_rate"] = win_rate
    result["rr_ratio"] = rr_ratio
    result["basis"]    = (
        f"{len(closed)}건 | 승률 {win_rate*100:.0f}% | "
        f"R:R 1:{rr_ratio:.1f} | Kelly {kelly*100:.0f}%"
    )
    return result


def calc_trade_stats(journal: list) -> dict:
    """거래 일지 기반 승률/수익 통계"""
    closed = [j for j in journal if j.get("pnl") is not None]
    if not closed:
        return {}
    wins   = [j for j in closed if j["pnl"] > 0]
    losses = [j for j in closed if j["pnl"] <= 0]
    total_pnl = sum(j["pnl"] for j in closed)
    win_rate  = len(wins) / len(closed) * 100 if closed else 0
    avg_win   = sum(j["pnl"] for j in wins)   / len(wins)   if wins   else 0
    avg_loss  = sum(j["pnl"] for j in losses) / len(losses) if losses else 0
    profit_factor = abs(sum(j["pnl"] for j in wins) / sum(j["pnl"] for j in losses)) if losses and sum(j["pnl"] for j in losses) != 0 else 0
    # MDD 계산
    running, peak, mdd = 0, 0, 0
    for j in reversed(closed):
        running += j["pnl"]
        if running > peak:
            peak = running
        dd = peak - running
        if dd > mdd:
            mdd = dd
    return {
        "total": len(closed), "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate, "total_pnl": total_pnl,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor, "mdd": mdd,
    }


def parse_trader_json(raw: str) -> dict:
    """trader 에이전트 JSON 출력 파싱. 실패 시 텍스트 기반 폴백."""
    import re, json as _json
    # JSON 블록 추출
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = _json.loads(match.group())
            signal = data.get("signal", "wait").lower()
            if signal not in ("long", "short", "wait"):
                signal = "wait"
            return {
                "signal":     signal,
                "entry":      data.get("entry"),
                "sl":         data.get("sl"),
                "tp":         data.get("tp"),
                "confidence": int(data.get("confidence", 50)),
                "reason":     data.get("reason", ""),
                "condition":  data.get("condition", ""),
                "raw":        raw,
            }
        except Exception:
            pass
    # 폴백: 텍스트에서 신호 추출
    if "🟢 롱 진입" in raw or '"signal": "long"' in raw:
        signal = "long"
    elif "🔴 숏 진입" in raw or '"signal": "short"' in raw:
        signal = "short"
    else:
        signal = "wait"
    return {"signal": signal, "entry": None, "sl": None, "tp": None,
            "confidence": 50, "reason": raw[:300], "condition": "", "raw": raw}


def trader_signal_text(signal: str) -> str:
    """signal → 표시용 텍스트"""
    return {"long": "🟢 롱 진입", "short": "🔴 숏 진입", "wait": "⚪ 관망"}.get(signal, "⚪ 관망")


def calc_confidence(result: dict) -> tuple:
    """분석 결과에서 신뢰도 점수(0-100)와 방향 반환"""
    score = 10  # 베이스 점수

    tj = result.get("trader_json", {})
    direction = tj.get("signal", "wait")
    if direction == "wait":
        return 30, "wait"

    # 트레이더 JSON confidence를 베이스로 활용 (50% 반영)
    trader_conf = tj.get("confidence", 50)
    score = int(trader_conf * 0.4)  # 트레이더 자체 신뢰도 40% 반영 (0.3→0.4 상향)

    ind    = result.get("indicators", {})    # 15m 지표
    ind_1h = result.get("indicators_1h", {}) # 1h 지표

    # 1. 컨플루언스 (25점) — 15m+1h 방향 일치
    cf_type = result.get("confluence_type", "mixed")
    if cf_type == direction:
        score += 25
    elif cf_type == "mixed":
        score += 5
    else:
        score -= 10

    # 2. RL 신호 (15점) — 충돌 시 중립 처리 (패널티 없음)
    rl = result.get("rl", {})
    if rl.get("available"):
        rl_type = rl.get("type", "wait")
        if rl_type == direction:
            score += 15
        elif rl_type == "wait":
            score += 3
        else:
            pass  # 에이전트 방향과 충돌 시 중립 (0점, 패널티 제거)
    else:
        score += 5

    # 3. 에이전트 텍스트 동의 (최대 18점, 에이전트당 6점)
    keywords_long  = ["상승", "매수", "롱", "bullish", "buy", "상방"]
    keywords_short = ["하락", "매도", "숏", "bearish", "sell", "하방"]
    kw = keywords_long if direction == "long" else keywords_short
    for txt in [result.get("analyst",""), result.get("news",""), result.get("risk","")]:
        if any(k in txt for k in kw):
            score += 6

    # 4. RSI 방향 일치 (최대 12점) — 15m 기준
    rsi = ind.get("rsi")
    if rsi:
        if direction == "long" and rsi < 25:
            score += 12  # 극단 과매도(RSI<25) → 강한 반등 기대 (+12)
        elif direction == "long" and rsi < 40:
            score += 8   # 과매도 → 반등 기대
        elif direction == "long" and 40 <= rsi <= 60:
            score += 4
        elif direction == "long" and rsi > 70:
            score -= 5   # 과매수에서 롱 → 페널티
        elif direction == "short" and rsi > 60:
            score += 8   # 과매수 → 하락 기대
        elif direction == "short" and 40 <= rsi <= 60:
            score += 4
        elif direction == "short" and rsi < 25:
            score -= 8   # 극단 과매도(RSI<25)에서 숏 → 강한 페널티
        elif direction == "short" and rsi < 30:
            score -= 5   # 과매도에서 숏 → 페널티

    # 5. MACD 히스토그램 방향 (10점) — 15m
    macd_hist = ind.get("macd_hist")
    if macd_hist is not None:
        if direction == "long" and macd_hist > 0:
            score += 10
        elif direction == "short" and macd_hist < 0:
            score += 10
        else:
            score -= 3

    # 6. EMA 배열 (추세) (8점) — 15m
    ema20 = ind.get("ema20")
    ema50 = ind.get("ema50")
    price = ind.get("price")
    if ema20 and ema50 and price:
        if direction == "long" and ema20 > ema50 and price > ema20:
            score += 8   # 골든크로스 + 가격 위
        elif direction == "short" and ema20 < ema50 and price < ema20:
            score += 8   # 데드크로스 + 가격 아래
        elif direction == "long" and ema20 > ema50:
            score += 4
        elif direction == "short" and ema20 < ema50:
            score += 4

    # 7. 볼린저밴드 위치 (6점) — 15m
    bb_upper = ind.get("bb_upper")
    bb_lower = ind.get("bb_lower")
    if bb_upper and bb_lower and price:
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            pct = (price - bb_lower) / bb_range  # 0=하단, 1=상단
            if direction == "long" and pct < 0.25:
                score += 6   # 하단 근접 → 반등 기대
            elif direction == "short" and pct > 0.75:
                score += 6   # 상단 근접 → 하락 기대
            elif 0.25 <= pct <= 0.75:
                score += 2   # 중간

    # 8. 1H MACD 방향 일치 (6점)
    macd_1h = ind_1h.get("macd_hist")
    if macd_1h is not None:
        if direction == "long" and macd_1h > 0:
            score += 6
        elif direction == "short" and macd_1h < 0:
            score += 6
        else:
            score -= 3

    # 9. ADX 추세 강도 (6점) — 강한 추세일수록 신호 신뢰도 상승
    adx = ind.get("adx")
    dmp = ind.get("adx_dmp")
    dmn = ind.get("adx_dmn")
    if adx and dmp and dmn:
        if adx >= 25:
            if direction == "long" and dmp > dmn:
                score += 6   # 강한 상승 추세
            elif direction == "short" and dmn > dmp:
                score += 6   # 강한 하락 추세
            else:
                score -= 3   # 추세 방향 반대
        # ADX < 25: 추세 없음 → 점수 변화 없음

    # 10. Stochastic RSI (6점) — 극단 과매수/과매도 포착
    stoch_k = ind.get("stoch_k")
    if stoch_k is not None:
        if direction == "long" and stoch_k < 20:
            score += 6   # 극단 과매도 → 반등 기대
        elif direction == "short" and stoch_k > 80:
            score += 6   # 극단 과매수 → 하락 기대
        elif direction == "long" and stoch_k > 80:
            score -= 4   # 과매수에서 롱 → 페널티
        elif direction == "short" and stoch_k < 20:
            score -= 4   # 과매도에서 숏 → 페널티

    # 11. OBV 거래량 방향 일치 (5점)
    obv_trend = ind.get("obv_trend")
    if obv_trend:
        if direction == "long" and obv_trend == "up":
            score += 5   # 거래량이 상승 지지
        elif direction == "short" and obv_trend == "down":
            score += 5   # 거래량이 하락 지지
        else:
            score -= 2   # 거래량 방향 불일치

    # 12. VWAP 위치 (5점) — 기관 기준선
    vwap = ind.get("vwap")
    if vwap and price:
        if direction == "long" and price > vwap:
            score += 5   # 가격이 VWAP 위 → 강세
        elif direction == "short" and price < vwap:
            score += 5   # 가격이 VWAP 아래 → 약세
        else:
            score -= 2

    # 13. 펀딩비 역추세 신호 (5점)
    fr = result.get("funding_rate", {})
    if fr.get("available"):
        rate = fr.get("rate", 0)
        if direction == "long" and rate < -0.02:
            score += 5   # 숏 과열 → 롱 유리
        elif direction == "short" and rate > 0.02:
            score += 5   # 롱 과열 → 숏 유리
        elif direction == "long" and rate > 0.05:
            score -= 4   # 롱 과열인데 롱 진입 → 페널티
        elif direction == "short" and rate < -0.05:
            score -= 4   # 숏 과열인데 숏 진입 → 페널티

    # 14. 4h 추세 방향 일치 (8점) — 큰 추세 필터
    ind_4h = result.get("indicators_4h", {})
    if ind_4h:
        rsi_4h   = ind_4h.get("rsi")
        hist_4h  = ind_4h.get("macd_hist")
        price_4h = ind_4h.get("price")
        ema20_4h = ind_4h.get("ema20")
        _4h_score = 0
        if rsi_4h:
            if direction == "long"  and rsi_4h < 50: _4h_score += 1
            if direction == "short" and rsi_4h > 50: _4h_score += 1
        if hist_4h is not None:
            if direction == "long"  and hist_4h > 0: _4h_score += 1
            if direction == "short" and hist_4h < 0: _4h_score += 1
        if price_4h and ema20_4h:
            if direction == "long"  and price_4h > ema20_4h: _4h_score += 1
            if direction == "short" and price_4h < ema20_4h: _4h_score += 1
        # 3점 만점 → 8점 환산
        score += round(_4h_score / 3 * 8)

    # 15. 세션 시간 가중치 (±5점) — 고유동성 시간대 우대
    # UTC 기준: 런던 사전장 06~10시, 뉴욕장 13~22시
    _utc_hour = datetime.utcnow().hour
    if 6 <= _utc_hour < 10 or 13 <= _utc_hour < 22:
        score += 5   # 고유동성 (런던/뉴욕) — 신호 신뢰도 높음
    elif 0 <= _utc_hour < 4:
        score -= 5   # 저유동성 데드존 — 진입 기준 강화

    # 16. Fear & Greed 지수 (±8점)
    _fg = result.get("fear_greed", {})
    if _fg.get("available"):
        _fgv = _fg["value"]
        if direction == "long":
            if _fgv <= 20:
                score += 8   # 극도공포 → 반등 강한 근거
            elif _fgv <= 35:
                score += 4   # 공포 → 매수 우호
            elif _fgv >= 80:
                score -= 8   # 극도탐욕에서 롱 → 역추세 위험
            elif _fgv >= 65:
                score -= 3   # 탐욕 → 추가 상승 여력 제한
        elif direction == "short":
            if _fgv >= 80:
                score += 8   # 극도탐욕 → 반락 강한 근거
            elif _fgv >= 65:
                score += 4   # 탐욕 → 매도 우호
            elif _fgv <= 20:
                score -= 8   # 극도공포에서 숏 → 역추세 위험
            elif _fgv <= 35:
                score -= 3   # 공포 → 추가 하락 여력 제한

    # 17. OI delta (±5점) — 포지션 축적 방향 확인
    _oi_delta = result.get("oi_delta_pct", 0)
    if abs(_oi_delta) >= 0.5:  # 0.5% 이상 변화만 유의미
        if direction == "long" and _oi_delta > 0:
            score += 5   # OI 증가 + 롱 → 새 롱 포지션 유입
        elif direction == "short" and _oi_delta < 0:
            score += 5   # OI 감소 + 숏 → 롱 청산 가속
        elif direction == "long" and _oi_delta < -1.0:
            score -= 3   # OI 급감 + 롱 → 포지션 청산 국면
        elif direction == "short" and _oi_delta > 1.0:
            score -= 3   # OI 급증 + 숏 → 쇼트 커버링 위험

    # 18. CVD (Cumulative Volume Delta, ±5점) — 실제 매수/매도 압력
    _cvd_trend = ind.get("cvd_trend")
    if _cvd_trend:
        if direction == "long" and _cvd_trend == "up":
            score += 5   # 매수 압력 우위
        elif direction == "short" and _cvd_trend == "down":
            score += 5   # 매도 압력 우위
        elif direction == "long" and _cvd_trend == "down":
            score -= 3   # 가격 오르는데 매도 압력 — 다이버전스
        elif direction == "short" and _cvd_trend == "up":
            score -= 3   # 가격 내리는데 매수 압력 — 다이버전스

    # 19. 펀딩비 추이 (8회 연속성, ±4점) — 과열 지속 감지
    _fr_hist = result.get("funding_rate_history", [])
    if len(_fr_hist) >= 6:
        _pos_cnt = sum(1 for x in _fr_hist[-6:] if x > 0)
        _neg_cnt = 6 - _pos_cnt
        if direction == "long" and _pos_cnt >= 5:
            score -= 4   # 6회 중 5회+ 양수 → 롱 과열 누적
        elif direction == "short" and _neg_cnt >= 5:
            score -= 4   # 6회 중 5회+ 음수 → 숏 과열 누적
        elif direction == "long" and _neg_cnt >= 5:
            score += 4   # 숏 과열 지속 → 롱 유리
        elif direction == "short" and _pos_cnt >= 5:
            score += 4   # 롱 과열 지속 → 숏 유리

    return max(0, min(100, score)), direction


defaults = {
    "logs": [],
    "agent_status": {"analyst": "대기", "news": "대기", "risk": "대기", "trader": "대기"},
    "last_analysis": _load_analysis(),
    "analysis_history": _load_history(),
    "auto_run": True,
    "selected_symbol": "ETHUSDT",
    "last_auto_time": 0,
    "is_analyzing": False,
    "analyzing_symbol": "",
    "tg_token":       TELEGRAM_TOKEN   if BINANCE_READY else "",
    "tg_chat_id":     TELEGRAM_CHAT_ID if BINANCE_READY else "",
    "tg_notify":      False,
    "tg_signal_only": True,
    "pnl_baseline":        None,   # 1시간 PNL 기준 잔고
    "pnl_baseline_time":   0,      # 기준 시각 (time.time())
    "trading_paused":      False,  # 일시 중지 플래그
    "daily_start_balance": None,   # 당일 시작 잔고
    "daily_start_date":    "",     # 당일 날짜 문자열
    "tg_polling_started":  False,  # 텔레그램 폴링 스레드 시작 여부
    "last_tg_signal":      {},     # 심볼별 마지막 전송 신호 {symbol: signal}
    "position_pct":        20,     # 잔고 대비 진입 비율 (%)
    "bt_last_output":      "",     # 백테스팅 마지막 출력
    "bt_last_version":     "v4",  # 백테스팅 마지막 버전
    "bt_last_interval":    "30m", # 백테스팅 마지막 인터벌
    # 알트코인 스캐너
    "alt_scan_results":          [],   # 스크리너 결과
    "alt_last_scan_time":        0,    # 마지막 스크리너 실행 시각
    "alt_announcements":         {"found": False, "announcements": [], "summary": ""},
    "alt_last_announcement_time": 0,   # 마지막 공지 스캔 시각
    "alt_analysis":              {},   # 선택 종목 분석 결과 {symbol: result}
    "alt_auto_scan":             True, # 알트 자동 스캔 ON/OFF
    "alt_auto_trade":            True, # 알트 자동 주문 ON/OFF
    "cooldown_until":            0,    # 연속 손실 쿨다운 종료 timestamp
    "consec_loss_count":   3,     # 쿨다운 발동 연속 손실 횟수 (기본 3)
    "consec_loss_hours":   2,     # 쿨다운 지속 시간 (기본 2시간)
    "atr_volatility_mult": 2.0,   # 변동성 필터 배수 (기본 2배)
    "volatility_filter_on": True, # 변동성 필터 ON/OFF
    "trailing_stop_on":    True,  # 트레일링 스탑 ON/OFF
    "trailing_atr_mult":   1.0,   # SL 이동 발동 기준 (수익 ≥ ATR × 배수)
    "early_exit_on":       True,  # 조기 청산 ON/OFF
    "early_exit_conf":     70,    # 조기 청산 발동 최소 신뢰도
    "immediate_reanalyze": [],    # TP/SL 청산 후 즉시 재분석 대기 심볼 목록
    "oi_prev": {},               # 직전 OI 값 {symbol: oi_float} — OI delta 계산용
    "limit_order_on":   False,   # 지정가 IOC 진입 시도 ON/OFF
    "candle_confirm_on": True,   # 캔들 종가 확인 후 진입 ON/OFF
    "partial_tp_on":    True,    # 부분 청산 (TP 50%) ON/OFF
    "partial_tp_done":  {},      # 이미 부분 청산한 심볼 set {symbol: True}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

AGENTS = {
    "analyst": {"emoji": "[분석]",  "name": "분석가",  "desc": "기술적 분석"},
    "news":    {"emoji": "[뉴스]",   "name": "뉴스",    "desc": "시장 심리"},
    "risk":    {"emoji": "[리스크]", "name": "리스크",  "desc": "포지션 관리"},
    "trader":  {"emoji": "[결정]",  "name": "트레이더", "desc": "최종 결정"},
}

def add_log(msg: str, level: str = "info"):
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{now}] {msg}")
    if len(st.session_state.logs) > 50:
        st.session_state.logs = st.session_state.logs[:50]
    _log(msg, level)  # 파일 로그 동시 기록


# ── 차트 생성 ─────────────────────────────────────────────────────
def create_chart(df: pd.DataFrame) -> "go.Figure":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas_ta as ta

    close = df["close"]
    ema20 = ta.ema(close, length=20)
    ema50 = ta.ema(close, length=50)
    bb    = ta.bbands(close, length=20, std=2)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
    )

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="가격",
        increasing_line_color="#4CAF50", decreasing_line_color="#e53935",
        increasing_fillcolor="#4CAF50", decreasing_fillcolor="#e53935",
        line_width=1,
    ), row=1, col=1)

    # EMA
    fig.add_trace(go.Scatter(
        x=df["time"], y=ema20, name="EMA20",
        line=dict(color="#26a69a", width=1.5), opacity=0.9
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=ema50, name="EMA50",
        line=dict(color="#FF9800", width=1.5), opacity=0.9
    ), row=1, col=1)

    # 볼린저밴드 (컬럼명 동적 탐색)
    if bb is not None:
        col_upper = next((c for c in bb.columns if c.startswith("BBU")), None)
        col_lower = next((c for c in bb.columns if c.startswith("BBL")), None)
        if col_upper and col_lower:
            fig.add_trace(go.Scatter(
                x=df["time"], y=bb[col_upper], name="BB 상단",
                line=dict(color="#2196F3", width=1, dash="dot"), opacity=0.7
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df["time"], y=bb[col_lower], name="BB 하단",
                line=dict(color="#2196F3", width=1, dash="dot"), opacity=0.7,
                fill="tonexty", fillcolor="rgba(33,150,243,0.06)"
            ), row=1, col=1)

    # 거래량
    vol_colors = ["#4CAF50" if c >= o else "#e53935"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["time"], y=df["volume"],
        marker_color=vol_colors, name="거래량", opacity=0.7
    ), row=2, col=1)

    fig.update_layout(
        height=420,
        xaxis_rangeslider_visible=False,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafafa",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        font=dict(size=11),
    )
    fig.update_xaxes(gridcolor="#eeeeee", showgrid=True)
    fig.update_yaxes(gridcolor="#eeeeee", showgrid=True)

    return fig


# ── PPO 모델 로드 (캐시) ──────────────────────────────────────────
@st.cache_resource
def _load_ppo():
    try:
        from stable_baselines3 import PPO
        model_path = Path(__file__).parent / "rl/models/v4/ppo_eth_30m.zip"
        if not model_path.exists():
            return None
        return PPO.load(str(model_path))
    except Exception:
        return None


def get_rl_signal(symbol: str) -> dict:
    """ETHUSDT 전용 PPO 모델 신호 반환 (0=관망 1=롱 2=숏 3=청산)"""
    if symbol != "ETHUSDT":
        return {"available": False, "reason": "ETH 전용 모델"}
    try:
        import numpy as np
        import pandas_ta as ta

        model = _load_ppo()
        if model is None:
            return {"available": False, "reason": "모델 파일 없음"}

        # 최근 30m 데이터 70캔들 (v4 모델 학습 타임프레임)
        df = get_klines("ETHUSDT", "30m", 70)

        # 피처 계산 (env._preprocess + data.py 동일 방식)
        df["price_chg"] = df["close"].pct_change().fillna(0)
        df["rsi_norm"]  = ta.rsi(df["close"], length=14).fillna(50) / 100.0
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd_norm"] = (macd_df["MACD_12_26_9"] / df["close"]).fillna(0)
        bb = ta.bbands(df["close"], length=20, std=2)
        col_u = next(c for c in bb.columns if c.startswith("BBU"))
        col_l = next(c for c in bb.columns if c.startswith("BBL"))
        df["bb_pct"]    = ((df["close"] - bb[col_l]) / (bb[col_u] - bb[col_l])).fillna(0.5).clip(0, 1)
        ema20 = ta.ema(df["close"], length=20)
        ema50 = ta.ema(df["close"], length=50)
        df["ema_ratio"] = (ema20 / ema50 - 1).fillna(0)
        atr = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_norm"]  = (atr / df["close"]).fillna(0)
        df["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()).fillna(1.0).clip(0, 5) / 5.0

        df = df.dropna().reset_index(drop=True)
        if len(df) < 20:
            return {"available": False, "reason": "데이터 부족"}

        # obs 구성: 마지막 20캔들, 포지션 없음 상태
        feat_cols = ["price_chg", "rsi_norm", "macd_norm", "bb_pct", "ema_ratio", "atr_norm", "vol_ratio"]
        rows = []
        for i in range(len(df) - 20, len(df)):
            row = [float(df[c].iloc[i]) for c in feat_cols]
            row += [0.0, 0.0, 0.0, 0.0]  # position, upnl, hold_steps, cooldown
            rows.append(row)

        obs = np.array(rows, dtype=np.float32).flatten()
        action, _ = model.predict(obs, deterministic=True)

        action_map = {0: ("⚪ 관망", "wait"), 1: ("🟢 롱", "long"),
                      2: ("🔴 숏", "short"), 3: ("🔵 청산", "close")}
        label, atype = action_map.get(int(action), ("?", "wait"))
        return {"available": True, "action": int(action), "label": label, "type": atype}

    except Exception as e:
        return {"available": False, "reason": str(e)}


# ── 컨플루언스 계산 (15m + 1h 방향 일치 여부) ─────────────────────
def get_confluence(ind_15m: dict, ind_1h: dict) -> tuple[str, str]:
    """두 타임프레임 지표에서 방향 점수를 계산해 컨플루언스 판단"""
    def score(ind):
        s = 0
        rsi = ind.get("rsi")
        if rsi:
            if rsi > 55:   s += 1
            elif rsi < 45: s -= 1
        hist = ind.get("macd_hist")
        if hist:
            s += 1 if hist > 0 else -1
        price, ema20 = ind.get("price"), ind.get("ema20")
        if price and ema20:
            s += 1 if price > ema20 else -1
        return s

    s15 = score(ind_15m)
    s1h = score(ind_1h)

    if s15 > 0 and s1h > 0:
        return "🟢 상승 컨플루언스", "long"
    elif s15 < 0 and s1h < 0:
        return "🔴 하락 컨플루언스", "short"
    else:
        return "⚪ 혼조 (신호 불일치)", "mixed"


# ── Fear & Greed 지수 (alternative.me 무료 API) ──────────────────────
def _fetch_fear_greed() -> dict:
    """공포탐욕지수 실시간 조회 — 0(극도공포) ~ 100(극도탐욕)"""
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://api.alternative.me/fng/?limit=1",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json as _j
            data = _j.loads(resp.read())
        item = data["data"][0]
        value = int(item["value"])
        label = item["value_classification"]
        return {"value": value, "label": label, "available": True}
    except Exception:
        return {"value": 50, "label": "Neutral", "available": False}


# ── 실시간 뉴스 수집 (CryptoPanic RSS + Cointelegraph RSS) ──────────────
def _fetch_live_news(symbol: str) -> str:
    """RSS 피드에서 최신 헤드라인 수집 — Claude 지식 컷오프 극복"""
    import urllib.request
    import xml.etree.ElementTree as ET

    # 심볼별 필터 키워드
    kw_map = {
        "ETHUSDT": ["ethereum", "eth", "이더리움"],
        "BTCUSDT": ["bitcoin", "btc", "비트코인"],
    }
    coin_kw = kw_map.get(symbol, ["crypto", "bitcoin", "ethereum"])
    always_kw = ["fed", "cpi", "fomc", "macro", "inflation", "rate", "sec", "regulation",
                 "연준", "금리", "인플레", "규제"]

    feeds = [
        "https://cointelegraph.com/rss",       # 안정적 RSS (우선)
        "https://cryptopanic.com/news/rss/",   # 폴백 (XML 이슈 있을 수 있음)
    ]

    headlines = []
    for url in feeds:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            root = ET.fromstring(content)
            for item in root.findall(".//item")[:30]:
                title = (item.findtext("title") or "").strip()
                desc  = (item.findtext("description") or "").strip()
                pub   = (item.findtext("pubDate") or "")[:16]
                combined = (title + " " + desc).lower()
                # 코인 관련 or 매크로 이슈
                if any(k in combined for k in coin_kw + always_kw):
                    headlines.append(f"• {title} ({pub})")
                if len(headlines) >= 8:
                    break
        except Exception:
            continue
        if len(headlines) >= 8:
            break

    return "\n".join(headlines[:6]) if headlines else ""


# ── 분석 실행 (멀티 타임프레임 + analyst/news 병렬) ──────────────────
def _should_trade(auto_run: bool, confidence: int = 0) -> bool:
    """진입 조건: 자동 주문 ON + (정각/:30 OR 신뢰도 65%+) + 캔들 종가 확인(ON 시)"""
    if not auto_run:
        return False
    now_min = datetime.now().minute
    now_sec = datetime.now().second
    is_scheduled     = now_min in (0, 30)    # 정각 or 30분
    is_strong_signal = confidence >= 65      # 강한 신호
    if not (is_scheduled or is_strong_signal):
        return False
    # 캔들 종가 확인 모드: 15m 캔들 경계(0,15,30,45분) 후 3분 이내만 진입
    if st.session_state.get("candle_confirm_on", True):
        _boundaries = [0, 15, 30, 45]
        _dist = min(abs(now_min - b) for b in _boundaries)
        # 경계에서 3분 초과 시 진입 보류 (가짜 돌파 방지)
        if _dist > 3:
            return False
    return True


def run_analysis(symbol: str, execute_trade: bool = False):
    add_log(f"🔍 {symbol} 분석 시작 (15m + 1h)...")
    if st.session_state.get("tg_notify"):
        _start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        send_progress(st.session_state.tg_token, st.session_state.tg_chat_id,
                      symbol, "🔍 분석 시작", f"📅 {_start_time}",
                      is_start=True)

    # 에이전트 상태 초기화
    for k in st.session_state.agent_status:
        st.session_state.agent_status[k] = "대기"

    result = {}

    # ── 데이터 수집: 15m + 1h + 4h + 실시간 뉴스 병렬 ──
    try:
        def _fetch_15m(): return get_klines(symbol, "15m", CANDLE_CNT)
        def _fetch_1h():  return get_klines(symbol, "1h",  50)
        def _fetch_4h():  return get_klines(symbol, "4h",  30)
        def _fetch_news(): return _fetch_live_news(symbol)
        def _fetch_fg():   return _fetch_fear_greed()
        def _fetch_fr_hist(): return get_funding_rate_history(symbol, limit=8)
        def _fetch_oi():   return get_open_interest(symbol)

        with ThreadPoolExecutor(max_workers=7) as ex:
            fut_15m    = ex.submit(_fetch_15m)
            fut_1h     = ex.submit(_fetch_1h)
            fut_4h     = ex.submit(_fetch_4h)
            fut_news   = ex.submit(_fetch_news)
            fut_fg     = ex.submit(_fetch_fg)
            fut_fr     = ex.submit(_fetch_fr_hist)
            fut_oi     = ex.submit(_fetch_oi)
            df_15m      = fut_15m.result()
            df_1h       = fut_1h.result()
            df_4h       = fut_4h.result()
            live_news   = fut_news.result()
            fear_greed  = fut_fg.result()
            fr_history  = fut_fr.result()
            oi_data     = fut_oi.result()

        ind_15m = calc_indicators(df_15m)
        ind_1h  = calc_indicators(df_1h)
        ind_4h  = calc_indicators(df_4h)

        text_15m = format_for_agent(symbol, ind_15m, label="15분봉")
        text_1h  = format_for_agent(symbol, ind_1h,  label="1시간봉")

        confluence_label, confluence_type = get_confluence(ind_15m, ind_1h)

        result["indicators"]      = ind_15m   # 기본 지표는 15m 기준
        result["indicators_1h"]   = ind_1h
        result["indicators_4h"]   = ind_4h
        result["confluence"]      = confluence_label
        result["confluence_type"] = confluence_type
        result["df"]              = df_15m

        # 변동성 필터용 ATR 평균 + 동적 배수 계산 (최근 30캔들 기준)
        try:
            import pandas_ta as _ta_vol
            _atr_series = _ta_vol.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
            _atr_vals = _atr_series.dropna().iloc[-30:] if _atr_series is not None else None
            if _atr_vals is not None and len(_atr_vals) >= 10:
                _atr_avg  = float(_atr_vals.mean())
                _atr_std  = float(_atr_vals.std())
                # 동적 배수: mean + 1.5σ 이상이면 비정상 → mult = 1 + 1.5×CV
                _cv = _atr_std / max(_atr_avg, 1e-6)
                _auto_mult = round(max(1.5, min(4.0, 1 + 1.5 * _cv)), 1)
                st.session_state.atr_volatility_mult = _auto_mult
            else:
                _atr_avg = 0
        except Exception:
            _atr_avg = 0
        result["atr_avg"] = _atr_avg

        # RL 신호 (ETHUSDT만)
        rl = get_rl_signal(symbol)
        result["rl"] = rl
        if rl.get("available"):
            add_log(f"🤖 PPO 신호: {rl['label']}")

        # 펀딩비 수집 (현재값)
        fr = get_funding_rate(symbol)
        result["funding_rate"] = fr
        if fr.get("available"):
            add_log(f"💰 펀딩비: {fr['rate']:+.4f}%")

        # 펀딩비 추이 (8회)
        result["funding_rate_history"] = fr_history
        if fr_history:
            _fr_pos = sum(1 for x in fr_history if x > 0)
            _fr_neg = len(fr_history) - _fr_pos
            add_log(f"📊 펀딩 추이: 양수 {_fr_pos}회 / 음수 {_fr_neg}회 (최근 8회)")

        # Fear & Greed 지수
        result["fear_greed"] = fear_greed
        if fear_greed.get("available"):
            add_log(f"😱 공포탐욕: {fear_greed['value']} ({fear_greed['label']})")

        # OI delta 계산
        oi_delta_pct = 0.0
        if oi_data.get("available"):
            oi_now  = oi_data["oi"]
            oi_prev = st.session_state.oi_prev.get(symbol, oi_now)
            if oi_prev > 0:
                oi_delta_pct = round((oi_now - oi_prev) / oi_prev * 100, 3)
            st.session_state.oi_prev[symbol] = oi_now
            add_log(f"📈 OI: {oi_now:,.0f} (Δ {oi_delta_pct:+.2f}%)")
        result["oi_data"]      = oi_data
        result["oi_delta_pct"] = oi_delta_pct

        add_log(f"📡 데이터 수집 완료 | 컨플루언스: {confluence_label}")
    except Exception as e:
        add_log(f"⚠️ 데이터 수집 실패: {e}")
        return result

    analyst_input = f"""{symbol} 멀티 타임프레임 기술적 지표를 분석해주세요.

{text_15m}

{text_1h}

두 타임프레임의 신호가 일치할 때(컨플루언스) 신뢰도를 높게 평가해주세요."""

    _news_section = f"\n\n[실시간 뉴스 헤드라인 (RSS)]\n{live_news}" if live_news else ""
    if live_news:
        add_log(f"📰 실시간 뉴스 {live_news.count('•')}건 수집")
    _fg = result.get("fear_greed", {})
    _fg_section = (
        f"\n\n[공포탐욕지수] {_fg['value']}/100 — {_fg['label']}"
        if _fg.get("available") else ""
    )
    news_input = (
        f"{symbol} 현재 시장 심리와 최근 암호화폐 뉴스를 분석해주세요. "
        f"현재가: ${ind_15m.get('price', 'N/A')}{_fg_section}{_news_section}"
    )

    # ind_text는 리스크 에이전트용 (15m 기준)
    ind_text = text_15m

    # ── 1단계: analyst + news 병렬 실행 (부분 실패 허용) ──
    st.session_state.agent_status["analyst"] = "작업 중"
    st.session_state.agent_status["news"]    = "작업 중"

    def _run_analyst():
        try:
            return run_agent("analyst", analyst_input)
        except Exception as e:
            return f"[분석가 오류: {e}]"

    def _run_news():
        try:
            return run_agent("news", news_input)
        except Exception as e:
            return f"[뉴스 오류: {e}]"

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_analyst = executor.submit(_run_analyst)
        fut_news    = executor.submit(_run_news)
        result["analyst"] = fut_analyst.result()
        result["news"]    = fut_news.result()

    st.session_state.agent_status["analyst"] = "완료"
    st.session_state.agent_status["news"]    = "완료"
    add_log("📊 분석가 완료")
    add_log("📰 뉴스 에이전트 완료")

    # ── 2단계: 리스크 평가 (부분 실패 허용) ──
    st.session_state.agent_status["risk"] = "작업 중"
    risk_input = f"""다음 분석 결과를 바탕으로 리스크를 평가해주세요.

심볼: {symbol}
{ind_text}

[기술적 분석 결과]
{result.get('analyst', '')}

[시장 심리 결과]
{result.get('news', '')}"""
    try:
        result["risk"] = run_agent("risk", risk_input)
    except Exception as e:
        result["risk"] = f"[리스크 오류: {e}]"
    st.session_state.agent_status["risk"] = "완료"
    add_log("⚖️ 리스크 에이전트 완료")

    # ── 3단계: 최종 결정 (부분 실패 허용) ──
    st.session_state.agent_status["trader"] = "작업 중"
    trader_input = f"""다음 분석을 종합해 최종 매매 결정을 내려주세요.

심볼: {symbol}

[기술적 분석]
{result.get('analyst', '')}

[시장 심리]
{result.get('news', '')}

[리스크 평가]
{result.get('risk', '')}"""
    try:
        trader_raw = run_agent("trader", trader_input)
        result["trader"] = trader_raw
        result["trader_json"] = parse_trader_json(trader_raw)
    except Exception as e:
        result["trader"] = f"[트레이더 오류: {e}]"
        result["trader_json"] = {"signal": "wait", "entry": None, "sl": None, "tp": None,
                                  "confidence": 0, "reason": str(e), "condition": "", "raw": ""}
    st.session_state.agent_status["trader"] = "완료"
    tj = result["trader_json"]
    add_log(f"🤖 트레이더 결정: {trader_signal_text(tj['signal'])} | 신뢰도: {tj['confidence']}%")

    # ── 신뢰도 점수 계산 ──
    confidence, conf_dir = calc_confidence(result)

    # ── BTC↔ETH 교차 필터 (±5점) ──
    _other_sym = "BTCUSDT" if symbol == "ETHUSDT" else "ETHUSDT"
    _other_res = st.session_state.last_analysis.get(_other_sym, {})
    _other_dir = _other_res.get("trader_json", {}).get("signal", "wait")
    if _other_dir != "wait" and conf_dir != "wait":
        if _other_dir == conf_dir:
            confidence = min(100, confidence + 5)
            add_log(f"🔗 교차 필터: {_other_sym} 동방향({_other_dir}) → +5pt → {confidence}%")
        else:
            confidence = max(0, confidence - 5)
            add_log(f"⚠️ 교차 필터: {_other_sym} 반방향({_other_dir}) → -5pt → {confidence}%")

    result["confidence"] = confidence
    result["confidence_dir"] = conf_dir

    result["time"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["symbol"] = symbol
    st.session_state.last_analysis[symbol] = result

    # ── 히스토리 추가 ──
    tj  = result.get("trader_json", {})
    ind = result.get("indicators", {})
    hist_decision = trader_signal_text(tj.get("signal", "wait"))

    history_entry = {
        "time":       result["time"],
        "symbol":     symbol,
        "decision":   hist_decision,
        "price":      ind.get("price", 0),
        "rsi":        ind.get("rsi"),
        "confluence": result.get("confluence", ""),
        "rl_label":   result.get("rl", {}).get("label", "") if result.get("rl", {}).get("available") else "",
        "trader_conf": tj.get("confidence", 0),
    }
    st.session_state.analysis_history.insert(0, history_entry)
    if len(st.session_state.analysis_history) > 100:
        st.session_state.analysis_history = st.session_state.analysis_history[:100]
    _save_history(st.session_state.analysis_history)

    # ── 파일 캐시 저장 ──
    _save_analysis(st.session_state.last_analysis)

    # ── 텔레그램: 트레이더 결정 카드 전송 (신호 변경 시에만) ──
    if st.session_state.get("tg_notify"):
        _cur_signal  = tj.get("signal", "wait")
        _prev_signal = st.session_state.last_tg_signal.get(symbol, "")
        if _cur_signal != _prev_signal:
            _rl_label = result.get("rl", {}).get("label", "") if result.get("rl", {}).get("available") else ""
            send_trader_decision(
                st.session_state.tg_token, st.session_state.tg_chat_id,
                symbol, tj, confidence=confidence, rl_label=_rl_label
            )
            st.session_state.last_tg_signal[symbol] = _cur_signal
            add_log(f"📲 텔레그램 전송: {_cur_signal} (이전: {_prev_signal or '없음'})")
        else:
            add_log(f"📵 텔레그램 스킵: 신호 동일 ({_cur_signal})")

    # ── 조기 청산 체크 (반대 신호 + 손실 중) ──
    if BINANCE_READY and st.session_state.get("early_exit_on", True) and st.session_state.get("auto_run"):
        try:
            _positions = get_positions()
            _my_pos = next((p for p in _positions if p["symbol"] == symbol), None)
            if _my_pos:
                _pos_side = _my_pos["side"]   # "LONG" or "SHORT"
                _new_sig  = tj.get("signal", "wait")
                _upnl     = _my_pos.get("unrealized_pnl", 0)
                _exit_thr = st.session_state.get("early_exit_conf", 70)
                _is_opp   = (_pos_side == "LONG"  and _new_sig == "short") or \
                            (_pos_side == "SHORT" and _new_sig == "long")
                if _is_opp and confidence >= _exit_thr and _upnl < 0:
                    add_log(f"🚨 조기 청산: {symbol} {_pos_side} | 반대신호({_new_sig}, {confidence}%) | PnL ${_upnl:.2f}")
                    _cr = close_position(symbol)
                    if _cr["success"]:
                        _update_journal_pnl(symbol, _upnl, ind.get("price"))
                        add_log(f"✅ 조기 청산 완료: {symbol} PnL ${_upnl:.2f}")
                        if st.session_state.get("tg_notify"):
                            send_error(st.session_state.tg_token, st.session_state.tg_chat_id,
                                       f"🚨 {symbol} 조기 청산 | 반대신호 {confidence}% | PnL ${_upnl:.2f}")
                    else:
                        add_log(f"❌ 조기 청산 실패: {_cr.get('error', '')}", "error")
        except Exception as _e:
            add_log(f"조기 청산 체크 오류: {_e}", "error")

    # ── 자동 주문 실행 ──
    if execute_trade:
        # 일시 중지 체크
        if st.session_state.get("trading_paused"):
            add_log("⏸ 자동 거래 일시 중지 중 — 주문 건너뜀")
            execute_trade = False

    if execute_trade:
        # 일일 손실 한도 체크
        try:
            _bal_now  = get_balance()
            _today    = datetime.now().strftime("%Y-%m-%d")
            if st.session_state.daily_start_date != _today or st.session_state.daily_start_balance is None:
                st.session_state.daily_start_date    = _today
                st.session_state.daily_start_balance = _bal_now["total"]
            _daily_loss = st.session_state.daily_start_balance - _bal_now["total"]
            if _daily_loss >= MAX_DAILY_LOSS:
                add_log(f"🚨 일일 손실 한도 초과 (${_daily_loss:.2f}) — 거래 중단")
                st.session_state.trading_paused = True
                if st.session_state.get("tg_notify"):
                    send_daily_limit_alert(st.session_state.tg_token, st.session_state.tg_chat_id,
                                           _daily_loss, MAX_DAILY_LOSS)
                execute_trade = False
        except Exception:
            pass

    if execute_trade:
        # 연속 손실 쿨다운 체크
        _cooldown_until = st.session_state.get("cooldown_until", 0)
        if time.time() < _cooldown_until:
            _remaining_min = int((_cooldown_until - time.time()) / 60)
            add_log(f"❄️ 쿨다운 중 — {_remaining_min}분 후 거래 재개")
            execute_trade = False
        else:
            # 쿨다운 해제 후 연속 손실 재확인
            _max_consec = st.session_state.get("consec_loss_count", 3)
            _consec = _get_consec_losses()
            if _consec >= _max_consec:
                _hours = st.session_state.get("consec_loss_hours", 2)
                st.session_state.cooldown_until = time.time() + _hours * 3600
                add_log(f"🚨 연속 {_consec}회 손실 — {_hours}시간 거래 중단")
                if st.session_state.get("tg_notify"):
                    send_error(st.session_state.tg_token, st.session_state.tg_chat_id,
                               f"연속 {_consec}회 손실 감지 — {_hours}시간 쿨다운 시작")
                execute_trade = False

    if execute_trade:
        # 중복 진입 방지
        try:
            _existing = [p for p in get_positions() if p["symbol"] == symbol]
            if _existing:
                add_log(f"⚠️ 중복 진입 방지: {symbol} 이미 포지션 존재 ({_existing[0]['side']})")
                execute_trade = False
        except Exception:
            pass

    if execute_trade:
        # 변동성 필터 — ATR이 평균 대비 너무 크면 진입 차단
        if st.session_state.get("volatility_filter_on", True):
            _atr_avg = result.get("atr_avg", 0)
            _cur_atr = (result.get("indicators") or {}).get("atr", 0) or 0
            _v_mult  = st.session_state.get("atr_volatility_mult", 2.0)
            if _atr_avg > 0 and _cur_atr > _atr_avg * _v_mult:
                add_log(f"🌪️ 변동성 필터: ATR {_cur_atr:.2f} > 평균 {_atr_avg:.2f}×{_v_mult} — 진입 차단")
                execute_trade = False

    if execute_trade:
        execute_trade = _should_trade(True, confidence)
        if execute_trade:
            add_log(f"✅ 진입 조건 충족 (신뢰도:{confidence}% | 시각:{datetime.now().minute}분)")
        else:
            add_log(f"⏸ 진입 보류 (신뢰도:{confidence}% | 정각/:30 아님 또는 캔들 경계 외)")

    # 포트폴리오 총 익스포저 체크 (ETH+BTC 합산 $200 이하)
    if execute_trade and BINANCE_READY:
        try:
            _all_pos = get_positions()
            _total_exposure = sum(
                p["size"] * p["entry_price"] / p["leverage"]
                for p in _all_pos
            )
            _max_exposure = MAX_USDT * 2  # 최대 총 익스포저 (기본 $200)
            if _total_exposure >= _max_exposure:
                add_log(f"🚫 포트폴리오 한도: 총 익스포저 ${_total_exposure:.0f} ≥ ${_max_exposure:.0f} → 진입 차단")
                execute_trade = False
        except Exception:
            pass

    if execute_trade:
        _signal = tj.get("signal", "wait")
        _atr    = ind.get("atr", 0) or 0
        _price  = ind.get("price", 0) or 0

        # SL/TP: trader JSON 우선, 없으면 ATR 폴백 (ADX 동적 R:R)
        def _resolve_sl_tp(side: str):
            sl = tj.get("sl")
            tp = tj.get("tp")
            if sl and tp:
                add_log(f"📐 트레이더 JSON SL:${sl} / TP:${tp}")
                return sl, tp
            # ADX 기반 R:R 동적 조정
            _adx = ind.get("adx", 0) or 0
            if _adx >= 30:
                # 강한 추세: TP 확대 (×4.0), SL 유지
                _sl_mult = ATR_SL_MULT
                _tp_mult = ATR_TP_MULT * 1.33  # 3.0→4.0
                add_log(f"📈 강한 추세(ADX={_adx:.0f}): TP 확대 ×{_tp_mult:.1f}")
            elif _adx < 20:
                # 횡보: TP·SL 모두 축소
                _sl_mult = ATR_SL_MULT * 0.67  # 1.5→1.0
                _tp_mult = ATR_TP_MULT * 0.67  # 3.0→2.0
                add_log(f"↔️ 횡보(ADX={_adx:.0f}): SL/TP 축소 ×{_tp_mult:.1f}")
            else:
                _sl_mult = ATR_SL_MULT
                _tp_mult = ATR_TP_MULT
            dist_sl = _atr * _sl_mult
            dist_tp = _atr * _tp_mult
            if side == "BUY":
                sl = round(_price - dist_sl, 2) if dist_sl else None
                tp = round(_price + dist_tp, 2) if dist_tp else None
            else:
                sl = round(_price + dist_sl, 2) if dist_sl else None
                tp = round(_price - dist_tp, 2) if dist_tp else None
            add_log(f"📐 ATR폴백 SL:${sl} / TP:${tp} (R:R 1:{_tp_mult/_sl_mult:.1f})")
            return sl, tp

        # 진입 금액: 신뢰도 비례 배율 적용 후 MAX_USDT 상한
        # 65-69% → ×0.5 / 70-79% → ×0.75 / 80%+ → ×1.0
        try:
            _avail_bal  = get_balance()["available"]
            _base_pct   = st.session_state.get("position_pct", 20)
            if confidence >= 80:
                _conf_mult = 1.0
            elif confidence >= 70:
                _conf_mult = 0.75
            else:
                _conf_mult = 0.5
            _dyn_usdt   = round(_avail_bal * _base_pct / 100 * _conf_mult, 1)
            _order_usdt = min(_dyn_usdt, MAX_USDT)
        except Exception:
            _conf_mult  = 1.0
            _order_usdt = MAX_USDT
        add_log(f"💼 진입 금액: ${_order_usdt} (잔고 {st.session_state.get('position_pct', 20)}% × 신뢰도배율 {_conf_mult:.0%})")

        _use_lmt = st.session_state.get("limit_order_on", False)
        if _signal == "long":
            _sl, _tp = _resolve_sl_tp("BUY")
            r = place_order(symbol, "BUY", _order_usdt, LEVERAGE, sl_price=_sl, tp_price=_tp, use_limit=_use_lmt)
            if r["success"]:
                add_log(f"✅ 롱 주문 실행: {symbol} {r['qty']} @ ${r['price']}")
                if st.session_state.get("tg_notify"):
                    send_order(st.session_state.tg_token, st.session_state.tg_chat_id,
                               symbol, "BUY", r["qty"], r["price"])
                _add_journal_entry({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol, "side": "🟢 롱", "action": "진입",
                    "qty": r["qty"], "price": r["price"],
                    "sl": _sl, "tp": _tp, "atr": _atr, "pnl": None,
                    "confidence": confidence,
                })
            else:
                add_log(f"❌ 주문 실패: {r['error']}", "error")
        elif _signal == "short":
            _sl, _tp = _resolve_sl_tp("SELL")
            r = place_order(symbol, "SELL", _order_usdt, LEVERAGE, sl_price=_sl, tp_price=_tp, use_limit=_use_lmt)
            if r["success"]:
                add_log(f"✅ 숏 주문 실행: {symbol} {r['qty']} @ ${r['price']}")
                if st.session_state.get("tg_notify"):
                    send_order(st.session_state.tg_token, st.session_state.tg_chat_id,
                               symbol, "SELL", r["qty"], r["price"])
                _add_journal_entry({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol, "side": "🔴 숏", "action": "진입",
                    "qty": r["qty"], "price": r["price"],
                    "sl": _sl, "tp": _tp, "atr": _atr, "pnl": None,
                    "confidence": confidence,
                })
            else:
                add_log(f"❌ 주문 실패: {r['error']}", "error")
        else:
            add_log("⚪ 관망 — 주문 없음")

    return result


# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────

# ── 헤더 + 분석 버튼 ─────────────────────────────────────────────────
# ── 텔레그램 폴링 스레드 시작 (최초 1회) ────────────────────────────
if BINANCE_READY and not st.session_state.tg_polling_started:
    if st.session_state.get("tg_token") and st.session_state.get("tg_chat_id"):
        start_polling_thread(st.session_state.tg_token, st.session_state.tg_chat_id)
        st.session_state.tg_polling_started = True

# ── 텔레그램 명령어 처리 ─────────────────────────────────────────────
if BINANCE_READY:
    for _cmd in read_and_clear_commands():
        _c = _cmd.get("cmd")
        if _c == "pause":
            st.session_state.trading_paused = True
            add_log("⏸ 텔레그램 명령: 자동 거래 일시 중지")
        elif _c == "resume":
            st.session_state.trading_paused = False
            add_log("▶️ 텔레그램 명령: 자동 거래 재개")
        elif _c == "status":
            try:
                _b = get_balance()
                _p = get_positions()
                _today = datetime.now().strftime("%Y-%m-%d")
                if st.session_state.daily_start_date == _today and st.session_state.daily_start_balance:
                    _dloss = st.session_state.daily_start_balance - _b["total"]
                else:
                    _dloss = 0
                send_status(st.session_state.tg_token, st.session_state.tg_chat_id,
                            _b, _p, st.session_state.trading_paused, _dloss, MAX_DAILY_LOSS)
            except Exception as e:
                add_log(f"⚠️ status 명령 오류: {e}")
        elif _c == "close":
            _sym = _cmd.get("symbol", "ETHUSDT")
            try:
                # 청산 전 미실현 PnL 스냅샷
                _pre_pos = next((p for p in get_positions() if p["symbol"] == _sym), None)
                _tg_pnl  = _pre_pos["unrealized_pnl"] if _pre_pos else 0
                _r = close_position(_sym)
                if _r["success"]:
                    _update_journal_pnl(_sym, _tg_pnl)
                    add_log(f"✅ 텔레그램 명령: {_sym} 청산 완료 | PnL ${_tg_pnl:+.4f}")
                    send_close(st.session_state.tg_token, st.session_state.tg_chat_id, _sym, _tg_pnl)
                else:
                    add_log(f"❌ 텔레그램 청산 실패: {_r['error']}", "error")
            except Exception as e:
                add_log(f"⚠️ close 명령 오류: {e}")

hdr_col, btn_col = st.columns([5, 1])
with hdr_col:
    st.markdown("""
    <div class="header-box" style="background: #ffffff; border: 1.5px solid #e0e0e0;">
        <h2 style="color: #0d1b2a !important;">AI 멀티에이전트 트레이딩 봇</h2>
        <p style="color: #555 !important;">분석가 · 뉴스 · 리스크 · 트레이더 — 바이낸스 선물 자동매매 | analyst+news 병렬 실행</p>
    </div>
    """, unsafe_allow_html=True)
with btn_col:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    _sym_label = st.session_state.get("symbol_select", SYMBOLS[0])
    run_btn = st.button(f"🔍 {_sym_label}\n분석", type="primary", use_container_width=True)

# ── API 연결 확인 ─────────────────────────────────────────────────────
if not BINANCE_READY:
    st.error(f"⚠️ 바이낸스 연결 실패: {BINANCE_ERR}")
    st.info("👉 `trading/config.py`에서 API_KEY, API_SECRET을 입력하세요.")
    st.stop()

# ── 사이드바 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    st.divider()

    symbol = st.selectbox("💰 코인", SYMBOLS, key="symbol_select")
    st.session_state.selected_symbol = symbol

    mode_color = "#FF9800" if not TESTNET else "#4CAF50"
    mode_label = "🔴 실거래 모드" if not TESTNET else "🟢 테스트넷 모드"

    # API 상태 + 다크모드 한 줄
    try:
        _bal = get_balance()
        st.markdown("<span style='color:#4CAF50; font-weight:700'>🟢 API 연결됨</span>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<span style='color:#e53935; font-weight:700'>🔴 API 오류</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{mode_color}; font-weight:700'>{mode_label}</span>", unsafe_allow_html=True)

    st.divider()

    # 에이전트 상태 — 분석 완료 여부로 보정
    _has_result = bool(st.session_state.last_analysis.get(symbol))
    _running_now = st.session_state.get("is_analyzing", False)
    st.markdown("### 🤖 에이전트 상태")
    for key, info in AGENTS.items():
        status = st.session_state.agent_status[key]
        # 결과가 있고 분석 중이 아니면 모두 완료로 표시
        if _has_result and not _running_now and status == "대기":
            status = "완료"
        if status == "작업 중":
            cls, icon, badge_color = "working", "🟠", "#FF9800"
        elif status == "완료":
            cls, icon, badge_color = "done", "✅", "#4CAF50"
        else:
            cls, icon, badge_color = "", "⚪", "#aaa"
        st.markdown(f"""
        <div class="agent-card {cls}">
            <b>{info['emoji']} {info['name']}</b>
            <small style="color:#888"> — {info['desc']}</small><br>
            <small style="color:{badge_color}; font-weight:600">{icon} {status}</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 자동 주문 토글
    st.session_state.auto_run = st.toggle(
        "🔄 자동 주문 실행",
        value=st.session_state.auto_run,
        help="ON: 트레이더 결정에 따라 자동으로 주문 실행"
    )

    # 포지션 크기 — Half-Kelly 자동 계산
    _dp = _calc_dynamic_params()
    st.session_state.position_pct      = _dp["position_pct"]
    st.session_state.consec_loss_count = _dp["consec_loss_count"]
    st.session_state.consec_loss_hours = _dp["consec_loss_hours"]
    st.session_state.trailing_atr_mult = _dp["trailing_atr_mult"]
    st.session_state.early_exit_conf   = _dp["early_exit_conf"]
    try:
        _avail     = get_balance()["available"]
        _usdt_calc = round(_avail * _dp["position_pct"] / 100, 1)
        st.metric("💼 진입 비율 (Half-Kelly)", f"{_dp['position_pct']}%",
                  help="Kelly Criterion 기반 자동 계산")
        st.caption(f"≈ ${_usdt_calc} USDT | Kelly {_dp['kelly']*100:.0f}% | R:R 1:{_dp['rr_ratio']:.1f}")
    except Exception:
        st.metric("💼 진입 비율 (Half-Kelly)", f"{_dp['position_pct']}%")

    if st.session_state.auto_run:
        if st.session_state.get("trading_paused"):
            st.error("🚨 거래 일시 중지 (일일 손실 한도 초과 또는 /pause)")
            if st.button("▶️ 거래 재개", key="resume_btn"):
                st.session_state.trading_paused = False
                st.rerun()
        else:
            st.warning("⚠️ 자동 주문 활성화됨")

    # 일일 손실 현황
    _today = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.daily_start_date == _today and st.session_state.daily_start_balance:
        try:
            _bal_cur  = get_balance()
            _dloss    = st.session_state.daily_start_balance - _bal_cur["total"]
            _dloss_pct = (_dloss / MAX_DAILY_LOSS * 100) if MAX_DAILY_LOSS else 0
            _d_color   = "#e53935" if _dloss > MAX_DAILY_LOSS * 0.7 else "#555"
            st.markdown(
                f"<div style='font-size:12px; color:{_d_color}; margin-top:4px;'>"
                f"📉 오늘 손실: ${_dloss:+.2f} / ${MAX_DAILY_LOSS:.0f} "
                f"({_dloss_pct:.0f}%)</div>",
                unsafe_allow_html=True
            )
        except Exception:
            pass

    st.divider()

    # 리스크 필터 설정 (자동 계산)
    st.markdown("### 🛡️ 리스크 파라미터")
    st.caption(f"📐 {_dp['basis']}")

    # 쿨다운 상태
    _cd_until = st.session_state.get("cooldown_until", 0)
    if time.time() < _cd_until:
        _cd_remain = int((_cd_until - time.time()) / 60)
        st.error(f"❄️ 쿨다운 중 — {_cd_remain}분 남음")
        if st.button("쿨다운 해제", key="reset_cooldown"):
            st.session_state.cooldown_until = 0
            st.rerun()
    else:
        _consec_now = _get_consec_losses()
        if _consec_now > 0:
            st.caption(f"연속 손실: {_consec_now}/{_dp['consec_loss_count']}회")

    # 자동 계산값 표시 (읽기 전용)
    with st.expander("📊 자동 계산 파라미터", expanded=True):
        _c1, _c2 = st.columns(2)
        _c1.metric("연속 손실 횟수",  f"{_dp['consec_loss_count']}회",
                   help="P(n연속 손실)<10% 기준")
        _c2.metric("쿨다운 시간",     f"{_dp['consec_loss_hours']}h",
                   help="승률 역비례 자동 계산")
        _c1.metric("변동성 배수",     f"×{st.session_state.get('atr_volatility_mult', _dp['atr_volatility_mult']):.1f}",
                   help="ATR 1+1.5σ/μ 기반 (분석 시 갱신)")
        _c2.metric("트레일링 발동",   f"ATR×{_dp['trailing_atr_mult']:.1f}",
                   help="TP×(1-Kelly) 기반")
        _c1.metric("조기청산 신뢰도", f"{_dp['early_exit_conf']}%",
                   help="진입 신뢰도 P80 기반")
        _c2.metric("Kelly",           f"{_dp['kelly']*100:.0f}%",
                   help="Kelly Criterion")

    # ON/OFF 토글만 유지 (수치는 자동)
    st.session_state.volatility_filter_on = st.toggle(
        "변동성 필터",  value=st.session_state.volatility_filter_on,
        help="ATR > 평균×자동배수 이면 진입 차단"
    )
    st.session_state.trailing_stop_on = st.toggle(
        "트레일링 스탑", value=st.session_state.trailing_stop_on,
        help="수익 ≥ ATR×자동배수 이면 SL → break-even"
    )
    st.session_state.early_exit_on = st.toggle(
        "조기 청산", value=st.session_state.early_exit_on,
        help="반대신호 자동신뢰도% + 손실 중 → 즉시 청산"
    )
    st.session_state.partial_tp_on = st.toggle(
        "부분 청산 (TP 50%)", value=st.session_state.partial_tp_on,
        help="TP 도달 시 50% 청산 후 SL → break-even, 나머지 홀딩"
    )
    st.session_state.candle_confirm_on = st.toggle(
        "캔들 종가 확인", value=st.session_state.candle_confirm_on,
        help="15m 캔들 경계 후 3분 이내만 진입 (가짜 돌파 방지)"
    )
    st.session_state.limit_order_on = st.toggle(
        "지정가 진입 (IOC)", value=st.session_state.limit_order_on,
        help="±0.03% 지정가 우선 시도 → 미체결 시 시장가 폴백 (수수료 절감)"
    )

    st.divider()

    # 텔레그램 알림
    st.markdown("### 📲 텔레그램 알림")
    tg_token   = st.text_input("Bot Token",   value=st.session_state.tg_token,
                                type="password", placeholder="7123...:AAF...", key="tg_token_input")
    tg_chat_id = st.text_input("Chat ID",     value=st.session_state.tg_chat_id,
                                placeholder="123456789", key="tg_chat_input")
    st.session_state.tg_token   = tg_token
    st.session_state.tg_chat_id = tg_chat_id

    tg_ready = bool(tg_token and tg_chat_id)

    col_tg1, col_tg2 = st.columns(2)
    with col_tg1:
        st.session_state.tg_notify = st.toggle(
            "알림 ON",
            value=st.session_state.tg_notify and tg_ready,
            disabled=not tg_ready,
        )
    with col_tg2:
        if st.button("🔗 연결 테스트", disabled=not tg_ready, use_container_width=True):
            ok = test_connection(tg_token, tg_chat_id)
            if ok:
                st.success("✅ 연결 성공!")
            else:
                st.error("❌ 실패 — Token/Chat ID 확인")

    if st.session_state.tg_notify:
        st.session_state.tg_signal_only = st.toggle(
            "📶 신호 시만 알림",
            value=st.session_state.tg_signal_only,
            help="ON: 롱/숏 신호일 때만 요약 알림 전송 (관망 제외)\nOFF: 항상 전송"
        )
        st.success("📲 알림 활성화")
    elif tg_ready:
        st.caption("토글을 켜면 알림이 발송됩니다")
    else:
        st.caption("Token과 Chat ID를 입력하세요\n📌 Bot: @BotFather | ID: @userinfobot")

    st.divider()

    # 자동 분석 주기
    st.markdown("### ⏱️ 자동 분석")
    auto_interval = st.selectbox(
        "분석 간격",
        ["비활성", "10분", "30분", "1시간"],
        index=1,
        key="auto_interval_select"
    )

    st.divider()

    # 알트 자동 거래 설정
    st.markdown("### 🔥 알트 자동 거래")
    st.session_state.alt_auto_scan = st.toggle(
        "알트 자동 스캔 (3분)",
        value=st.session_state.alt_auto_scan,
        help="3분마다 알트 스크리너 + 바이낸스 공지 자동 실행"
    )
    st.session_state.alt_auto_trade = st.toggle(
        "알트 자동 주문",
        value=st.session_state.alt_auto_trade,
        help=f"신뢰도 {ALT_AUTO_CONFIDENCE}%+ → 자동 주문 / 공지 감지 → {ALT_MANUAL_MIN_CONF}%+ 자동 주문",
        disabled=not st.session_state.alt_auto_scan,
    )
    if st.session_state.alt_auto_trade:
        st.warning(f"⚡ 알트 자동 주문 활성 (≥{ALT_AUTO_CONFIDENCE}%)")
    st.caption(f"신뢰도 기준: 자동 {ALT_AUTO_CONFIDENCE}% | 수동 {ALT_MANUAL_MIN_CONF}%")

    st.divider()

    # 로그
    st.markdown("### 📋 실행 로그")
    log_html = "<br>".join(st.session_state.logs[:15]) if st.session_state.logs else "로그 없음"
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    st.divider()

    # 실거래 전환 체크리스트
    with st.expander("🚀 실거래 전환 체크리스트", expanded=False):
        _is_testnet = True
        try:
            from config import TESTNET
            _is_testnet = TESTNET
        except Exception:
            pass

        if _is_testnet:
            st.warning("⚠️ 현재 테스트넷 모드")
        else:
            st.success("✅ 실거래 모드 활성화됨")

        st.markdown("""
**전환 전 필수 확인 항목:**

- [ ] 실거래 API 키 발급 (바이낸스 → API Management)
- [ ] `config.py` `API_KEY` / `API_SECRET` 실거래 키로 교체
- [ ] `config.py` `TESTNET = False` 변경
- [ ] `MAX_USDT` 소액으로 시작 (예: $20~50)
- [ ] `LEVERAGE = 2` 이하로 낮추기 (실거래 초반)
- [ ] `MAX_DAILY_LOSS` 타이트하게 설정 (예: $50)
- [ ] 자동 주문 OFF 상태로 수동 분석 먼저 검증
- [ ] 텔레그램 알림 ON 확인
- [ ] 바이낸스 앱에서 API IP 화이트리스트 설정
- [ ] 소액 테스트 주문 1회 수동 실행 확인

**전환 순서:**
```
1. config.py 수정 (키 + TESTNET=False)
2. 봇 재시작 (launchctl unload → load)
3. 자동 주문 OFF로 분석만 모니터링 (3~5회)
4. 이상 없으면 자동 주문 ON
```
        """)
        st.caption("📁 config.py 경로: /Users/sunny/Desktop/ai-lab/trading/config.py")


# ── 실시간 메트릭 + 포지션 (1초 갱신) ────────────────────────────────
@st.fragment(run_every=1)
def live_panel(sym: str):
    try:
        price     = get_price(sym)
        balance   = get_balance()
        positions = get_positions()

        # 1시간 PNL 계산
        total_bal = balance['total']
        now_ts = time.time()
        if st.session_state.pnl_baseline is None or (now_ts - st.session_state.pnl_baseline_time) >= 3600:
            st.session_state.pnl_baseline      = total_bal
            st.session_state.pnl_baseline_time = now_ts
        hourly_pnl = total_bal - st.session_state.pnl_baseline
        elapsed_min = int((now_ts - st.session_state.pnl_baseline_time) / 60)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="value">${price:,.2f}</div>
                <div class="label">현재가</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="value">${balance['available']:,.1f}</div>
                <div class="label">가용 잔고 (USDT)</div></div>""", unsafe_allow_html=True)
        with m3:
            h_color = "#4CAF50" if hourly_pnl >= 0 else "#e53935"
            st.markdown(f"""<div class="metric-box">
                <div class="value" style="color:{h_color}">${hourly_pnl:+.2f}</div>
                <div class="label">시간당 PNL ({elapsed_min}분)</div></div>""", unsafe_allow_html=True)
        with m4:
            pnl = balance['unrealized_pnl']
            pnl_color = "#4CAF50" if pnl >= 0 else "#e53935"
            st.markdown(f"""<div class="metric-box">
                <div class="value" style="color:{pnl_color}">${pnl:+.2f}</div>
                <div class="label">미실현 손익</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-box">
                <div class="value">{len(positions)}</div>
                <div class="label">활성 포지션</div></div>""", unsafe_allow_html=True)

        # 포지션 카드
        if positions:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📌 현재 포지션")
            for pos in positions:
                side_color = "#4CAF50" if pos["side"] == "LONG" else "#e53935"
                pnl        = pos["unrealized_pnl"]
                pnl_color  = "#4CAF50" if pnl >= 0 else "#e53935"
                # PnL% 계산: (pnl / 진입금액) * leverage * 100
                entry_val  = float(pos.get("entry_price", 1)) * float(pos.get("size", 1))
                lev        = float(pos.get("leverage", 1))
                pnl_pct    = (pnl / entry_val * lev * 100) if entry_val else 0
                st.markdown(f"""
                <div style="background:#f8f9ff; border-radius:10px; padding:12px; margin-bottom:8px;
                            border-left:4px solid {side_color}">
                    <b>{pos['symbol']}</b> —
                    <span style="color:{side_color}"><b>{pos['side']}</b></span>
                    {pos['size']} 개 @ ${pos['entry_price']:,}<br>
                    <small>미실현 손익:
                        <span style="color:{pnl_color}"><b>${pnl:+.2f}
                        ({pnl_pct:+.2f}%)</b></span>
                        | 레버리지: {pos.get('leverage', '-')}x
                    </small>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"❌ {pos['symbol']} 청산", key=f"close_{pos['symbol']}"):
                    _pnl_snap = pos.get("unrealized_pnl", 0)
                    _px_snap  = get_price(pos["symbol"])
                    r = close_position(pos["symbol"])
                    if r["success"]:
                        _update_journal_pnl(pos["symbol"], _pnl_snap, _px_snap)
                        add_log(f"✅ {pos['symbol']} 포지션 청산 완료 | PnL ${_pnl_snap:+.4f}")
                        if st.session_state.get("tg_notify"):
                            send_close(st.session_state.tg_token,
                                       st.session_state.tg_chat_id, pos["symbol"], _pnl_snap)
                        st.success("청산 완료!")
                        st.rerun()
                    else:
                        st.error(f"청산 실패: {r['error']}")

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")

live_panel(symbol)
all_btn = False

# ── 거래 일지 + 분석 히스토리 (현재 포지션 바로 아래) ──
st.divider()
tab_pos, tab_j, tab_h, tab_c, tab_a0, tab_a1, tab_a2, tab_a3, tab_bt, tab_daily, tab_alt = st.tabs([
    "📌 현재 포지션", "📒 거래 일지", "📋 분석 히스토리", "📈 1분봉 차트",
    "🤖 트레이더 결정", "📊 기술적 분석", "📰 시장 심리", "⚖️ 리스크",
    "🔬 백테스팅", "📅 일별 성과", "🔥 알트 스캐너"
])
with tab_pos:
    try:
        _pos_list = get_positions()
        _balance  = get_balance()
        _price    = get_price(symbol)
        if not _pos_list:
            st.info("현재 활성 포지션이 없습니다.")
        else:
            for pos in _pos_list:
                side_color = "#4CAF50" if pos["side"] == "LONG" else "#e53935"
                pnl        = pos["unrealized_pnl"]
                pnl_color  = "#4CAF50" if pnl >= 0 else "#e53935"
                entry_val  = float(pos.get("entry_price", 1)) * float(pos.get("size", 1))
                lev        = float(pos.get("leverage", 1))
                pnl_pct    = (pnl / entry_val * lev * 100) if entry_val else 0
                st.markdown(f"""
                <div style="background:#f8f9ff; border-radius:12px; padding:16px 20px; margin-bottom:12px;
                            border-left:5px solid {side_color}">
                    <div style="font-size:18px; font-weight:700; margin-bottom:6px;">
                        {pos['symbol']} &nbsp;
                        <span style="color:{side_color}">{pos['side']}</span>
                    </div>
                    <div style="display:flex; gap:32px; font-size:14px; flex-wrap:wrap;">
                        <div><b>수량</b><br>{pos['size']}</div>
                        <div><b>진입가</b><br>${pos['entry_price']:,.4f}</div>
                        <div><b>현재가</b><br>${_price:,.4f}</div>
                        <div><b>레버리지</b><br>{pos.get('leverage','-')}x</div>
                        <div><b>미실현 손익</b><br>
                            <span style="color:{pnl_color}; font-weight:700;">
                                ${pnl:+.2f} ({pnl_pct:+.2f}%)
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"❌ {pos['symbol']} 청산", key=f"tabclose_{pos['symbol']}"):
                    _pnl_snap = pos.get("unrealized_pnl", 0)
                    _px_snap  = _price
                    r = close_position(pos["symbol"])
                    if r["success"]:
                        _update_journal_pnl(pos["symbol"], _pnl_snap, _px_snap)
                        add_log(f"✅ {pos['symbol']} 포지션 청산 완료 | PnL ${_pnl_snap:+.4f}")
                        if st.session_state.get("tg_notify"):
                            send_close(st.session_state.tg_token,
                                       st.session_state.tg_chat_id, pos["symbol"], _pnl_snap)
                        st.success("청산 완료!")
                        st.rerun()
                    else:
                        st.error(f"청산 실패: {r['error']}")
        # 잔고 요약
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("총 잔고 (USDT)", f"${_balance['total']:,.2f}")
        c2.metric("가용 잔고 (USDT)", f"${_balance['available']:,.2f}")
        c3.metric("미실현 손익", f"${_balance['unrealized_pnl']:+.2f}")
    except Exception as e:
        st.error(f"포지션 로드 실패: {e}")
with tab_j:
    journal = _load_journal()
    # ── 심볼 필터 ──
    _j_filter = st.radio("심볼 필터", ["전체"] + SYMBOLS, horizontal=True, key="j_sym_filter")
    _journal_filtered = journal if _j_filter == "전체" else [j for j in journal if j.get("symbol") == _j_filter]

    # ── 현재 포지션 (미실현 PnL 표시용) ──
    try:
        _cur_positions = {p["symbol"]: p for p in get_positions()} if BINANCE_READY else {}
    except Exception:
        _cur_positions = {}

    # ── 승률 통계 (필터 기준) ──
    stats = calc_trade_stats(_journal_filtered)
    if stats:
        s_color = "#4CAF50" if stats["total_pnl"] >= 0 else "#e53935"
        st.markdown(
            f"<div style='background:#f8f9ff; border-radius:10px; padding:12px 16px; margin-bottom:12px;'>"
            f"<b>📈 거래 통계</b> &nbsp; "
            f"총 {stats['total']}건 | "
            f"<span style='color:#4CAF50'>승 {stats['wins']}</span> / "
            f"<span style='color:#e53935'>패 {stats['losses']}</span> | "
            f"승률 <b>{stats['win_rate']:.1f}%</b><br>"
            f"총 손익: <b style='color:{s_color}'>${stats['total_pnl']:+.2f}</b> &nbsp;|&nbsp; "
            f"평균 수익: <span style='color:#4CAF50'>${stats['avg_win']:+.2f}</span> &nbsp;|&nbsp; "
            f"평균 손실: <span style='color:#e53935'>${stats['avg_loss']:+.2f}</span><br>"
            f"Profit Factor: <b>{stats['profit_factor']:.2f}</b> &nbsp;|&nbsp; "
            f"MDD: <span style='color:#e53935'>${stats['mdd']:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    if _journal_filtered:
        for j in _journal_filtered[:50]:
            side_color = "#4CAF50" if "롱" in j.get("side","") else "#e53935"
            _sym_j = j.get("symbol", "")
            if j.get("pnl") is not None:
                # 청산 완료 항목
                _pnl_val = j["pnl"]
                _pc = "#4CAF50" if _pnl_val >= 0 else "#e53935"
                pnl_str = f"손익: <b style='color:{_pc}'>${_pnl_val:+.2f}</b>"
                _close_str = f" | 청산가: <b>${j.get('close_price',0):,.2f}</b>" if j.get("close_price") else ""
            elif _sym_j in _cur_positions:
                # 아직 열린 포지션 — 미실현 PnL 실시간 표시
                _upnl = _cur_positions[_sym_j].get("unrealized_pnl", 0)
                _uc = "#4CAF50" if _upnl >= 0 else "#e53935"
                pnl_str = f"미실현: <b style='color:{_uc}'>${_upnl:+.2f}</b> <small style='color:#888'>(진행중)</small>"
                _close_str = ""
            else:
                pnl_str = "<span style='color:#aaa'>미기록</span>"
                _close_str = ""
            st.markdown(
                f"<div style='padding:10px 16px; border-left:4px solid {side_color}; "
                f"background:#fafafa; border-radius:8px; margin-bottom:8px;'>"
                f"<div style='font-size:16px; font-weight:700;'>{_sym_j} "
                f"<span style='color:{side_color}'>{j.get('side')}</span> {j.get('action','')}</div>"
                f"<div style='font-size:14px; color:#555; margin-top:4px;'>{j.get('time','')} | "
                f"진입가: <b>${j.get('price',0):,.2f}</b>{_close_str} | 수량: <b>{j.get('qty','')}</b> | {pnl_str}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.caption("자동 주문 실행 내역이 여기에 기록됩니다.")
with tab_h:
    history = st.session_state.get("analysis_history", [])
    # ── 심볼 필터 ──
    _h_filter = st.radio("심볼 필터", ["전체"] + SYMBOLS, horizontal=True, key="h_sym_filter")
    _history_filtered = history if _h_filter == "전체" else [h for h in history if h.get("symbol") == _h_filter]

    if _history_filtered:
        def _hist_card(h):
            dec = h.get("decision", "")
            if "롱" in dec:
                dec_color = "#4CAF50"
            elif "숏" in dec:
                dec_color = "#e53935"
            else:
                dec_color = "#888"
            rsi_val = f"RSI {h['rsi']}" if h.get("rsi") else ""
            rl_val  = h.get("rl_label", "")
            badge = f"<span style='font-size:12px; color:{dec_color}'>{dec}</span>"
            return (
                f"<div style='padding:8px 12px; border-left:3px solid {dec_color}; "
                f"background:#fafafa; border-radius:6px; margin-bottom:6px; font-size:13px;'>"
                f"<b>{h.get('symbol','')}</b> &nbsp;{badge}<br>"
                f"<small style='color:#888'>{h.get('time','')} | "
                f"${h.get('price',0):,.2f} | {h.get('confluence','')} "
                f"| {rsi_val} | PPO: {rl_val}</small>"
                f"</div>"
            )
        # 최근 5개는 고정, 나머지는 스크롤
        for h in _history_filtered[:5]:
            st.markdown(_hist_card(h), unsafe_allow_html=True)
        scroll_html = "".join(_hist_card(h) for h in _history_filtered[5:])
        st.markdown(
            f"<div style='max-height:300px; overflow-y:auto; padding-right:4px;'>"
            f"{scroll_html}</div>",
            unsafe_allow_html=True
        )
    else:
        st.caption("아직 분석 기록이 없습니다.")
with tab_c:
    try:
        df_1m = get_klines(symbol, "1m", 60)
        fig = go.Figure(data=[go.Candlestick(
            x=df_1m["time"],
            open=df_1m["open"], high=df_1m["high"],
            low=df_1m["low"],   close=df_1m["close"],
            increasing_line_color="#4CAF50",
            decreasing_line_color="#e53935",
        )])
        fig.update_layout(
            title=f"{symbol} 1분봉 (최근 60개)",
            xaxis_rangeslider_visible=False,
            height=340,
            margin=dict(l=0, r=0, t=36, b=0),
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"차트 로드 실패: {e}")

_last = st.session_state.last_analysis.get(symbol, {})
with tab_a0:
    _tj = _last.get("trader_json", {})
    if not _tj:
        st.info("분석 결과가 없습니다.")
    else:
        _sig = _tj.get("signal", "wait")
        _sig_txt, _sig_color = {"long": ("🟢 롱 진입", "#4CAF50"),
                                 "short": ("🔴 숏 진입", "#e53935"),
                                 "wait": ("⚪ 관망", "#888")}.get(_sig, ("⚪ 관망", "#888"))
        _conf = _tj.get("confidence", 0)
        _conf_color = "#4CAF50" if _conf >= 70 else "#FF9800" if _conf >= 50 else "#e53935"
        st.markdown(f"""
        <div style="background:#f8f9ff; border-radius:12px; padding:18px 22px; border-left:5px solid {_sig_color}; margin-bottom:12px;">
            <div style="font-size:22px; font-weight:800; color:{_sig_color}; margin-bottom:8px;">{_sig_txt}</div>
            <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:14px; margin-bottom:12px;">
                <div><b>신뢰도</b><br><span style="color:{_conf_color}; font-weight:700; font-size:18px;">{_conf}%</span></div>
                <div><b>진입가</b><br>${f"{_tj['entry']:,.2f}" if _tj.get('entry') else "—"}</div>
                <div><b>손절가 (SL)</b><br><span style="color:#e53935">${f"{_tj['sl']:,.2f}" if _tj.get('sl') else "—"}</span></div>
                <div><b>익절가 (TP)</b><br><span style="color:#4CAF50">${f"{_tj['tp']:,.2f}" if _tj.get('tp') else "—"}</span></div>
            </div>
            <div style="font-size:13px; color:#333; margin-bottom:8px;">
                <b>📋 결정 근거</b><br>{_tj.get('reason', '—')}
            </div>
            <div style="font-size:13px; color:#555;">
                <b>⚠️ 추가 조건</b><br>{_tj.get('condition', '—')}
            </div>
        </div>
        """, unsafe_allow_html=True)
with tab_a1:
    st.markdown(_last.get("analyst", "") or "분석 결과가 없습니다.")
with tab_a2:
    st.markdown(_last.get("news", "") or "분석 결과가 없습니다.")
with tab_a3:
    st.markdown(_last.get("risk", "") or "분석 결과가 없습니다.")
with tab_bt:
    st.subheader("🔬 RL 모델 백테스팅")
    # 버전별 사용 가능한 인터벌 매핑
    _BT_INTERVALS = {
        "v1": ["15m", "1h", "2h", "4h", "1d"],
        "v2": ["30m", "1h"],
        "v3": ["30m"],
        "v4": ["30m"],
        "v5": ["30m"],
    }
    _bt_col1, _bt_col2, _bt_col3 = st.columns([1, 1, 2])
    with _bt_col1:
        _bt_version = st.selectbox("모델 버전", list(_BT_INTERVALS.keys()), index=3, key="bt_version")
    with _bt_col2:
        _bt_interval = st.selectbox("인터벌", _BT_INTERVALS[_bt_version], key="bt_interval")
    with _bt_col3:
        _bt_test_start = st.text_input("테스트 시작 날짜 (선택)", placeholder="예: 2025-09-01", key="bt_test_start")
    _bt_run = st.button("▶️ 백테스트 실행", type="primary", key="bt_run_btn")
    if _bt_run:
        import subprocess
        _bt_cmd = [
            "python", str(Path(__file__).parent / "rl" / "backtest.py"),
            "--version", _bt_version,
            "--interval", _bt_interval,
        ]
        if _bt_test_start.strip():
            _bt_cmd += ["--test-start", _bt_test_start.strip()]
        with st.spinner(f"백테스트 실행 중... ({_bt_version} / {_bt_interval})"):
            try:
                _bt_proc = subprocess.run(
                    _bt_cmd,
                    capture_output=True, text=True,
                    cwd=str(Path(__file__).parent),
                    timeout=120,
                )
                _bt_out = _bt_proc.stdout + (_bt_proc.stderr or "")
                st.session_state["bt_last_output"]   = _bt_out
                st.session_state["bt_last_version"]  = _bt_version
                st.session_state["bt_last_interval"] = _bt_interval
            except subprocess.TimeoutExpired:
                st.error("백테스트 시간 초과 (120초)")
            except Exception as _bt_e:
                st.error(f"백테스트 오류: {_bt_e}")
    # 결과 출력
    if st.session_state.get("bt_last_output"):
        _bt_v = st.session_state.get("bt_last_version", "")
        _bt_i = st.session_state.get("bt_last_interval", "")
        st.caption(f"마지막 실행: {_bt_v} / {_bt_i}")
        st.code(st.session_state["bt_last_output"], language="text")
        # 생성된 PNG 이미지 표시
        _bt_img_paths = [
            Path(__file__).parent / "rl" / f"backtest_{_bt_i}_{_bt_v}.png",
            Path(__file__).parent / "rl" / f"backtest_{_bt_i}.png",
        ]
        for _bt_img in _bt_img_paths:
            if _bt_img.exists():
                st.image(str(_bt_img), use_container_width=True)
                break

with tab_daily:
    st.subheader("📅 일별 성과 대시보드")
    _dj = _load_journal()
    _closed = [j for j in _dj if j.get("pnl") is not None]
    if not _closed:
        st.info("청산된 거래가 없습니다. 거래가 청산되면 일별 성과가 표시됩니다.")
    else:
        # 날짜별 집계
        _day_map: dict = {}
        for j in _closed:
            _d = j.get("time", "")[:10]  # YYYY-MM-DD
            if not _d:
                continue
            if _d not in _day_map:
                _day_map[_d] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            _day_map[_d]["trades"] += 1
            _day_map[_d]["pnl"]    += j["pnl"]
            if j["pnl"] > 0:
                _day_map[_d]["wins"]   += 1
            else:
                _day_map[_d]["losses"] += 1
        _dates = sorted(_day_map.keys())
        _daily_pnl  = [_day_map[d]["pnl"]    for d in _dates]
        _cum_pnl    = []
        _cum = 0.0
        for p in _daily_pnl:
            _cum += p
            _cum_pnl.append(round(_cum, 4))

        # ── 요약 메트릭 ──────────────────────────────────────────────
        _total_days  = len(_dates)
        _profit_days = sum(1 for p in _daily_pnl if p > 0)
        _loss_days   = sum(1 for p in _daily_pnl if p <= 0)
        _best_day    = max(_daily_pnl)
        _worst_day   = min(_daily_pnl)
        _total_pnl   = _cum_pnl[-1] if _cum_pnl else 0

        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("총 손익", f"${_total_pnl:+.2f}", delta=None)
        _mc2.metric("수익일 / 손실일", f"{_profit_days}일 / {_loss_days}일")
        _mc3.metric("최고 수익일", f"${_best_day:+.2f}")
        _mc4.metric("최악 손실일", f"${_worst_day:+.2f}")

        # ── 일별 PnL 바 차트 ─────────────────────────────────────────
        _bar_colors = ["#4CAF50" if p >= 0 else "#e53935" for p in _daily_pnl]
        _fig_bar = go.Figure(go.Bar(
            x=_dates, y=_daily_pnl,
            marker_color=_bar_colors,
            text=[f"${p:+.2f}" for p in _daily_pnl],
            textposition="outside",
            name="일별 PnL",
        ))
        _fig_bar.update_layout(
            title="일별 손익 (Daily PnL)",
            xaxis_title="날짜", yaxis_title="PnL ($)",
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            height=320, margin=dict(t=40, b=40, l=40, r=20),
            showlegend=False,
        )
        _fig_bar.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
        st.plotly_chart(_fig_bar, use_container_width=True)

        # ── 누적 PnL 라인 차트 ───────────────────────────────────────
        _line_color = "#4CAF50" if _cum_pnl[-1] >= 0 else "#e53935"
        _fig_line = go.Figure(go.Scatter(
            x=_dates, y=_cum_pnl,
            mode="lines+markers",
            line=dict(color=_line_color, width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.08)" if _cum_pnl[-1] >= 0 else "rgba(229,57,53,0.08)",
            name="누적 PnL",
        ))
        _fig_line.update_layout(
            title="누적 손익 (Cumulative PnL)",
            xaxis_title="날짜", yaxis_title="누적 PnL ($)",
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            height=280, margin=dict(t=40, b=40, l=40, r=20),
            showlegend=False,
        )
        _fig_line.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
        st.plotly_chart(_fig_line, use_container_width=True)

        # ── 일별 상세 테이블 ─────────────────────────────────────────
        st.caption("일별 상세 내역")
        _df_daily = pd.DataFrame([
            {
                "날짜": d,
                "거래 수": _day_map[d]["trades"],
                "승": _day_map[d]["wins"],
                "패": _day_map[d]["losses"],
                "승률(%)": round(_day_map[d]["wins"] / _day_map[d]["trades"] * 100, 1) if _day_map[d]["trades"] else 0,
                "일별 PnL($)": round(_day_map[d]["pnl"], 4),
                "누적 PnL($)": _cum_pnl[i],
            }
            for i, d in enumerate(_dates)
        ]).sort_values("날짜", ascending=False)
        st.dataframe(
            _df_daily,
            use_container_width=True,
            hide_index=True,
        )

        # ── 신뢰도 캘리브레이션 ───────────────────────────────────────
        st.divider()
        st.subheader("🎯 신뢰도 캘리브레이션")
        st.caption("진입 시 신뢰도 구간별 실제 승률 — 임계값 최적화에 활용")
        _conf_trades = [j for j in _closed if j.get("confidence") is not None]
        if len(_conf_trades) < 2:
            st.info("신뢰도가 기록된 거래가 2건 이상이어야 표시됩니다. (오늘 이후 진입 거래부터 기록됩니다)")
        else:
            _buckets = [(0, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 101)]
            _cal_rows = []
            for _lo, _hi in _buckets:
                _b = [j for j in _conf_trades if _lo <= j["confidence"] < _hi]
                if not _b:
                    continue
                _bw = [j for j in _b if j["pnl"] > 0]
                _avg_pnl = sum(j["pnl"] for j in _b) / len(_b)
                _win_rt  = len(_bw) / len(_b) * 100
                _cal_rows.append({
                    "신뢰도 구간": f"{_lo}~{min(_hi,100)}%",
                    "거래 수": len(_b),
                    "승": len(_bw),
                    "패": len(_b) - len(_bw),
                    "승률(%)": f"{_win_rt:.0f}%",
                    "평균 PnL($)": f"{_avg_pnl:+.2f}",
                    "총 PnL($)": f"{sum(j['pnl'] for j in _b):+.2f}",
                })
            if _cal_rows:
                _df_cal = pd.DataFrame(_cal_rows)
                st.dataframe(_df_cal, use_container_width=True, hide_index=True)
                # 최적 구간 표시
                _best = max(_cal_rows, key=lambda r: float(r["평균 PnL($)"]))
                st.success(f"✅ 가장 수익률 높은 구간: **{_best['신뢰도 구간']}** (평균 {_best['평균 PnL($)']}$, 승률 {_best['승률(%)']})")

# ── 🔥 알트 스캐너 탭 ────────────────────────────────────────────────
with tab_alt:
    st.subheader("🔥 알트코인 스캐너")
    st.caption(f"바이낸스 USDT 선물 상위 {ALT_SCAN_LIMIT}개 알트 스크리닝 | 공지 최우선 | 동시 최대 {MAX_ALT_POSITIONS}포지션 | 1회 최대 ${MAX_USDT_ALT}")

    # ── 1. 바이낸스 공지 패널 (최우선) ───────────────────────────────
    st.markdown("### 📢 바이낸스 공지 (최우선 신호)")
    _ann = st.session_state.get("alt_announcements", {})
    _ann_time = st.session_state.get("alt_last_announcement_time", 0)
    _ann_age  = int((time.time() - _ann_time) / 60) if _ann_time else None

    col_ann1, col_ann2 = st.columns([3, 1])
    with col_ann1:
        if _ann.get("found") and _ann.get("announcements"):
            for _a in _ann["announcements"]:
                _urgency_color = "#e53935" if _a.get("urgency") == "high" else "#FF9800"
                _sig_color = "#4CAF50" if _a.get("signal") == "long" else "#e53935"
                st.markdown(
                    f"<div style='background:#fff3e0; border-left:4px solid {_urgency_color}; "
                    f"border-radius:8px; padding:10px 14px; margin-bottom:8px;'>"
                    f"<b style='color:{_urgency_color};'>🚨 [{_a.get('type','').upper()}]</b> "
                    f"<b>{_a.get('symbol','')}</b> — {_a.get('title','')}<br>"
                    f"<span style='color:{_sig_color}; font-weight:700;'>{'🟢 롱' if _a.get('signal')=='long' else '🔴 숏'}</span>"
                    f" {_a.get('reason','')}</div>",
                    unsafe_allow_html=True
                )
        else:
            _summary = _ann.get("summary", "아직 스캔 안 됨")
            st.info(f"📋 {_summary}" + (f" ({_ann_age}분 전)" if _ann_age else ""))
    with col_ann2:
        if st.button("🔍 공지 스캔", key="scan_announcement", use_container_width=True):
            with st.spinner("바이낸스 공지 확인 중..."):
                try:
                    _result_ann = check_binance_announcements()
                    st.session_state.alt_announcements         = _result_ann
                    st.session_state.alt_last_announcement_time = time.time()
                    add_log(f"📢 공지 스캔 완료: {_result_ann.get('summary','')}")
                    st.rerun()
                except Exception as _e:
                    st.error(f"공지 스캔 실패: {_e}")
        if _ann_age is not None:
            st.caption(f"마지막 스캔: {_ann_age}분 전")

    # 공지 기반 즉시 분석 버튼
    if _ann.get("found") and _ann.get("announcements"):
        for _ann_item in _ann["announcements"]:
            _ann_sym = _ann_item.get("symbol", "")
            if _ann_sym and st.button(f"⚡ {_ann_sym} 즉시 분석", key=f"ann_analyze_{_ann_sym}"):
                with st.spinner(f"{_ann_sym} 공지 기반 분석 중..."):
                    try:
                        from binance_client import get_klines as _gk
                        from indicators import calc_indicators as _ci
                        _kl = _gk(_ann_sym, "15m", 30)
                        _ind = _ci(_kl)
                        _cand = {
                            "symbol": _ann_sym,
                            "score": 90,
                            "signals": [f"바이낸스 공지: {_ann_item.get('type','')}"],
                            "direction": _ann_item.get("signal", "long"),
                            "price_change_1h": 0,
                            "vol_ratio": 1,
                            "rsi": _ind.get("rsi", 50),
                            "funding": 0,
                            "price": _ind.get("price", 0),
                            "atr": _ind.get("atr", 0),
                            "atr_pct": 0,
                            "indicators": _ind,
                            "indicators_1h": {},
                        }
                        _res = run_alt_analysis(_cand)
                        st.session_state.alt_analysis[_ann_sym] = _res
                        add_log(f"⚡ {_ann_sym} 공지 기반 분석 완료")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"분석 실패: {_e}")

    st.divider()

    # ── 2. 스크리너 패널 ─────────────────────────────────────────────
    st.markdown("### 🔍 알트 스크리너")
    _scan_results = st.session_state.get("alt_scan_results", [])
    _scan_time    = st.session_state.get("alt_last_scan_time", 0)
    _scan_age     = int((time.time() - _scan_time) / 60) if _scan_time else None

    sc_col1, sc_col2, sc_col3 = st.columns([2, 1, 1])
    with sc_col1:
        if _scan_age is not None:
            st.caption(f"마지막 스캔: {_scan_age}분 전 | 후보 {len(_scan_results)}개")
    with sc_col2:
        if st.button("🔥 전체 스캔", key="run_alt_scan", use_container_width=True):
            with st.spinner(f"알트 상위 {ALT_SCAN_LIMIT}개 스크리닝 중..."):
                try:
                    _syms = get_alt_futures_symbols(ALT_SCAN_LIMIT)
                    _results = screen_altcoins(_syms, top_n=5)
                    st.session_state.alt_scan_results  = _results
                    st.session_state.alt_last_scan_time = time.time()
                    add_log(f"🔥 알트 스캔 완료: {len(_results)}개 후보")
                    st.rerun()
                except Exception as _e:
                    st.error(f"스캔 실패: {_e}")
    with sc_col3:
        if _scan_results and st.button("📊 상위 종목 분석", key="analyze_top_alt", use_container_width=True):
            _top = _scan_results[0]
            with st.spinner(f"{_top['symbol']} 분석 중..."):
                try:
                    _res = run_alt_analysis(_top)
                    st.session_state.alt_analysis[_top["symbol"]] = _res
                    add_log(f"📊 {_top['symbol']} 알트 분석 완료")
                    st.rerun()
                except Exception as _e:
                    st.error(f"분석 실패: {_e}")

    # 스크리너 결과 카드
    if _scan_results:
        for _idx, _r in enumerate(_scan_results):
            _dir_color = "#4CAF50" if _r["direction"] == "long" else "#e53935" if _r["direction"] == "short" else "#888"
            _dir_label = {"long": "🟢 롱", "short": "🔴 숏", "wait": "⚪ 관망"}.get(_r["direction"], "⚪")
            with st.container():
                _rc1, _rc2, _rc3 = st.columns([4, 2, 1])
                with _rc1:
                    st.markdown(
                        f"<div style='background:#f8f9ff; border-left:4px solid {_dir_color}; "
                        f"border-radius:8px; padding:10px 14px;'>"
                        f"<b style='font-size:15px;'>#{_idx+1} {_r['symbol']}</b> "
                        f"<span style='color:{_dir_color}; font-weight:700;'>{_dir_label}</span> "
                        f"<span style='color:#888; font-size:12px;'>점수: {_r['score']}점</span><br>"
                        f"<span style='font-size:12px; color:#555;'>{' | '.join(_r['signals'])}</span><br>"
                        f"<span style='font-size:11px; color:#888;'>"
                        f"현재가: ${_r['price']:,} | RSI: {_r['rsi']} | "
                        f"거래량: {_r['vol_ratio']}배 | 1h: {_r['price_change_1h']:+.1f}% | "
                        f"펀딩비: {_r['funding']:+.3f}%</span></div>",
                        unsafe_allow_html=True
                    )
                with _rc2:
                    st.metric("ATR%", f"{_r['atr_pct']}%")
                with _rc3:
                    if st.button("분석", key=f"analyze_alt_{_r['symbol']}", use_container_width=True):
                        with st.spinner(f"{_r['symbol']} 분석 중..."):
                            try:
                                _res = run_alt_analysis(_r)
                                st.session_state.alt_analysis[_r["symbol"]] = _res
                                add_log(f"📊 {_r['symbol']} 알트 분석 완료")
                                st.rerun()
                            except Exception as _e:
                                st.error(f"분석 실패: {_e}")
    else:
        st.info("'전체 스캔' 버튼으로 알트코인 스크리닝을 시작하세요.")

    st.divider()

    # ── 3. 분석 결과 + 주문 ──────────────────────────────────────────
    _alt_analyses = st.session_state.get("alt_analysis", {})
    if _alt_analyses:
        st.markdown("### 📋 알트 분석 결과")
        for _asym, _ares in list(_alt_analyses.items()):
            _tj = _ares.get("trader_json", {})
            _sig = _tj.get("signal", "wait")
            _conf = _tj.get("confidence", 0)
            _sig_color = "#4CAF50" if _sig == "long" else "#e53935" if _sig == "short" else "#888"
            _sig_label = {"long": "🟢 롱 진입", "short": "🔴 숏 진입", "wait": "⚪ 관망"}.get(_sig, "⚪")

            with st.expander(f"**{_asym}** — {_sig_label} (신뢰도 {_conf}%) | {_ares.get('time','')}", expanded=True):
                _ac1, _ac2 = st.columns([3, 1])
                with _ac1:
                    st.markdown(f"**신호:** <span style='color:{_sig_color};font-weight:700'>{_sig_label}</span>  |  신뢰도: **{_conf}%**", unsafe_allow_html=True)
                    if _tj.get("entry"):
                        st.markdown(f"진입: `${_tj['entry']}` | SL: `${_tj.get('sl','N/A')}` | TP: `${_tj.get('tp','N/A')}`")
                    st.caption(_tj.get("reason", ""))

                    _tab_an, _tab_news = st.tabs(["📊 기술 분석", "📰 뉴스"])
                    with _tab_an:
                        st.text(_ares.get("analyst", ""))
                    with _tab_news:
                        st.text(_ares.get("news", ""))
                with _ac2:
                    # 수동 주문 버튼 (신뢰도 게이트)
                    if BINANCE_READY and _sig in ("long", "short"):
                        _cand_data = _ares.get("candidate", {})
                        _atr_val   = _cand_data.get("atr", 0) or 0
                        _entry_px  = _cand_data.get("price", 0)
                        # 신뢰도별 버튼 상태
                        if _conf >= ALT_AUTO_CONFIDENCE:
                            _btn_label = f"✅ {_sig_label}"
                            _btn_disabled = False
                            st.caption(f"🟢 신뢰도 {_conf}% — 자동진입 조건")
                        elif _conf >= ALT_MANUAL_MIN_CONF:
                            _btn_label = f"⚠️ {_sig_label}"
                            _btn_disabled = False
                            st.caption(f"🟡 신뢰도 {_conf}% — 수동 확인 권장")
                        else:
                            _btn_label = f"🚫 신뢰도 부족"
                            _btn_disabled = True
                            st.caption(f"🔴 신뢰도 {_conf}% < {ALT_MANUAL_MIN_CONF}% — 진입 불가")
                        if st.button(_btn_label, key=f"alt_order_{_asym}",
                                     use_container_width=True, disabled=_btn_disabled):
                            try:
                                _cur_pos = get_positions()
                                _alt_pos_cnt = len([p for p in _cur_pos if p["symbol"] not in SYMBOLS])
                                if _alt_pos_cnt >= MAX_ALT_POSITIONS:
                                    st.error(f"알트 최대 {MAX_ALT_POSITIONS}포지션 초과")
                                else:
                                    _side = "BUY" if _sig == "long" else "SELL"
                                    _sl = _tj.get("sl") or (round(_entry_px - _atr_val * ALT_ATR_SL_MULT, 4) if _sig == "long" else round(_entry_px + _atr_val * ALT_ATR_SL_MULT, 4))
                                    _tp = _tj.get("tp") or (round(_entry_px + _atr_val * ALT_ATR_TP_MULT, 4) if _sig == "long" else round(_entry_px - _atr_val * ALT_ATR_TP_MULT, 4))
                                    _r = place_order(_asym, _side, MAX_USDT_ALT, ALT_LEVERAGE, sl_price=_sl, tp_price=_tp)
                                    if _r["success"]:
                                        _add_journal_entry({
                                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "symbol": _asym, "side": "🟢 롱" if _sig == "long" else "🔴 숏",
                                            "action": "진입", "qty": _r["qty"], "price": _r["price"],
                                            "sl": _sl, "tp": _tp, "atr": _atr_val, "pnl": None,
                                        })
                                        st.success(f"✅ {_asym} {_sig_label} 실행 | ${_r['price']}")
                                        add_log(f"✅ 알트 {_asym} {_sig_label} @ ${_r['price']}")
                                    else:
                                        st.error(f"주문 실패: {_r['error']}")
                            except Exception as _e:
                                st.error(f"오류: {_e}")
                    if st.button("🗑️ 삭제", key=f"del_alt_{_asym}", use_container_width=True):
                        del st.session_state.alt_analysis[_asym]
                        st.rerun()

    # ── 4. 현재 알트 포지션 ─────────────────────────────────────────
    st.divider()
    st.markdown("### 📌 활성 알트 포지션")
    if BINANCE_READY:
        try:
            _all_pos = get_positions()
            _alt_pos = [p for p in _all_pos if p["symbol"] not in SYMBOLS]
            if _alt_pos:
                for _ap in _alt_pos:
                    _pc = "#4CAF50" if _ap["unrealized_pnl"] >= 0 else "#e53935"
                    _apc1, _apc2 = st.columns([4, 1])
                    with _apc1:
                        st.markdown(
                            f"**{_ap['symbol']}** {_ap['side']} | "
                            f"수량: {_ap['size']} | 진입가: ${_ap['entry_price']:,} | "
                            f"미실현 PnL: <span style='color:{_pc};font-weight:700'>${_ap['unrealized_pnl']:+.4f}</span>",
                            unsafe_allow_html=True
                        )
                    with _apc2:
                        if st.button(f"❌ 청산", key=f"alt_close_{_ap['symbol']}", use_container_width=True):
                            _px_snap  = _ap.get("unrealized_pnl", 0)
                            _close_r  = close_position(_ap["symbol"])
                            if _close_r["success"]:
                                _update_journal_pnl(_ap["symbol"], _px_snap)
                                st.success(f"✅ {_ap['symbol']} 청산 완료 | PnL ${_px_snap:+.4f}")
                                add_log(f"✅ 알트 {_ap['symbol']} 청산 | PnL ${_px_snap:+.4f}")
                                st.rerun()
            else:
                st.info("현재 활성 알트 포지션 없음")
        except Exception as _e:
            st.warning(f"포지션 조회 실패: {_e}")

# ── 에이전트 분석 결과 (전체 너비) ────────────────────────────────────
st.divider()

_is_running = (st.session_state.get("is_analyzing") and
               st.session_state.get("analyzing_symbol") == symbol)

# ── 분석 진행 중 UI ──
if _is_running:
    _AGENT_STEPS = [
        ("analyst", "📊 분석가",  "15m+1h 기술적 지표 분석"),
        ("news",    "📰 뉴스",    "시장 심리 및 뉴스 분석"),
        ("risk",    "⚖️ 리스크",  "포지션 리스크 평가"),
        ("trader",  "🤖 트레이더","최종 매매 결정"),
    ]
    _status = st.session_state.agent_status
    cards_html = ""
    for key, name, desc in _AGENT_STEPS:
        s = _status.get(key, "대기")
        if s == "작업 중":
            bg, border, icon, label_color = "#fff8ee", "#FF9800", "🟠", "#e65100"
        elif s == "완료":
            bg, border, icon, label_color = "#f0fff4", "#4CAF50", "✅", "#2e7d32"
        else:
            bg, border, icon, label_color = "#f5f5f5", "#ddd", "⏸", "#aaa"
        cards_html += (
            f"<div style='flex:1; background:{bg}; border:1px solid {border}; border-radius:8px; "
            f"padding:7px 10px; margin:0 4px; font-size:12px;'>"
            f"<span style='font-size:13px'>{icon}</span> <b>{name}</b>"
            f"<div style='font-size:10px; color:#999; margin-top:1px'>{desc}</div>"
            f"<div style='font-size:11px; color:{label_color}; font-weight:600'>{s}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div style='padding:10px 14px; background:#fff8ee; border:1.5px solid #FF9800; "
        f"border-radius:10px; margin-bottom:10px;'>"
        f"<b style='color:#e65100; font-size:14px;'>⏳ {symbol} 분석 진행 중</b>"
        f"<span style='color:#aaa; font-size:12px; margin-left:8px'>analyst+news 병렬 → 리스크 → 결정</span>"
        f"<div style='display:flex; margin-top:8px'>{cards_html}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ── 분석 결과 (진행 중에는 표시 안 함) ──
analysis = None if _is_running else st.session_state.last_analysis.get(symbol)

if analysis:
    # ── 오래된 분석 경고 ──
    _ana_time = analysis.get("time", "")
    _stale = False
    if _ana_time:
        try:
            _elapsed_min = (datetime.now() - datetime.strptime(_ana_time, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
            _stale = _elapsed_min > 60
        except Exception:
            pass
    _stale_badge = "  ⚠️ <span style='color:#FF9800;font-size:12px'>오래된 분석 (1시간+)</span>" if _stale else ""
    st.markdown(f"<small style='color:#aaa'>⏱️ 마지막 분석: {_ana_time}{_stale_badge}</small>",
                unsafe_allow_html=True)

    tj          = analysis.get("trader_json", {})
    _signal     = tj.get("signal", "wait")
    _signal_txt = trader_signal_text(_signal)

    # ── 신뢰도 점수 ──
    confidence   = analysis.get("confidence", 0)
    conf_dir     = analysis.get("confidence_dir", "wait")
    trader_conf  = tj.get("confidence", 0)
    if conf_dir == "long":
        conf_color, conf_bg = "#2e7d32", "#e8f5e9"
    elif conf_dir == "short":
        conf_color, conf_bg = "#c62828", "#fce4ec"
    else:
        conf_color, conf_bg = "#555", "#f5f5f5"
    st.markdown(
        f"<div style='background:{conf_bg}; border-radius:10px; padding:10px 16px; margin-bottom:10px;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<b style='color:{conf_color}'>🎯 종합 신뢰도</b>"
        f"<span><b style='font-size:22px; color:{conf_color}'>{confidence}%</b>"
        f"<span style='font-size:12px; color:#aaa; margin-left:8px'>트레이더: {trader_conf}%</span></span></div>"
        f"<div style='background:#ddd; border-radius:6px; height:8px; margin-top:6px;'>"
        f"<div style='background:{conf_color}; width:{confidence}%; height:8px; border-radius:6px;'></div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── 결정 카드 (JSON 기반) ──
    box_class = {"long": "decision-long", "short": "decision-short"}.get(_signal, "decision-wait")
    _reason   = tj.get("reason", "") or analysis.get("trader", "")[:200]
    _entry    = tj.get("entry")
    _sl       = tj.get("sl")
    _tp       = tj.get("tp")
    _cond     = tj.get("condition", "")

    sl_tp_html = ""
    if _entry:
        sl_tp_html += f"<br><small>진입: <b>${_entry:,.2f}</b>"
        if _sl: sl_tp_html += f" &nbsp;|&nbsp; SL: <b style='color:#e53935'>${_sl:,.2f}</b>"
        if _tp: sl_tp_html += f" &nbsp;|&nbsp; TP: <b style='color:#4CAF50'>${_tp:,.2f}</b>"
        sl_tp_html += "</small>"
    if _cond:
        sl_tp_html += f"<br><small style='color:#888'>💬 {_cond[:100]}</small>"

    st.markdown(
        f'<div class="{box_class}"><b>{_signal_txt}</b><br>{_reason[:200]}{"..." if len(_reason) > 200 else ""}'
        f'{sl_tp_html}</div>',
        unsafe_allow_html=True
    )

    # 컨플루언스 + RL 신호 배지
    cf_col, rl_col = st.columns(2)
    confluence = analysis.get("confluence")
    if confluence:
        cf_type   = analysis.get("confluence_type", "mixed")
        cf_bg     = {"long": "#e8f5e9", "short": "#fce4ec"}.get(cf_type, "#f5f5f5")
        cf_border = {"long": "#4CAF50", "short": "#e53935"}.get(cf_type, "#9e9e9e")
        with cf_col:
            st.markdown(
                f"<div style='margin-top:10px; padding:8px 14px; background:{cf_bg}; "
                f"border:1.5px solid {cf_border}; border-radius:8px; font-size:13px;'>"
                f"<b>📡 컨플루언스 (15m+1h)</b><br>{confluence}</div>",
                unsafe_allow_html=True
            )

    rl = analysis.get("rl", {})
    if rl.get("available"):
        rl_type   = rl.get("type", "wait")
        rl_bg     = {"long": "#e8f5e9", "short": "#fce4ec", "close": "#e3f2fd"}.get(rl_type, "#f5f5f5")
        rl_border = {"long": "#4CAF50", "short": "#e53935", "close": "#2196F3"}.get(rl_type, "#9e9e9e")
        with rl_col:
            st.markdown(
                f"<div style='margin-top:10px; padding:8px 14px; background:{rl_bg}; "
                f"border:1.5px solid {rl_border}; border-radius:8px; font-size:13px;'>"
                f"<b>🤖 PPO 모델 신호 (ETH)</b><br>{rl['label']}</div>",
                unsafe_allow_html=True
            )

elif not _is_running:
    st.markdown("""
    <div style='display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:200px; color:#aaa;'>
        <div style='font-size:48px; margin-bottom:16px;'>📈</div>
        <div style='font-size:16px; font-weight:600; color:#888;'>우측 상단 "분석" 버튼을 누르세요</div>
        <div style='font-size:13px; margin-top:8px; color:#bbb; text-align:center;'>
            4개 에이전트가 병렬+순차로 분석합니다
        </div>
    </div>
    """, unsafe_allow_html=True)



# ── 버튼 실행 처리 ────────────────────────────────────────────────────
if run_btn:
    st.session_state.is_analyzing    = True
    st.session_state.analyzing_symbol = symbol
    # 에이전트 상태 초기화 (UI 즉시 반영)
    for k in st.session_state.agent_status:
        st.session_state.agent_status[k] = "대기"
    st.rerun()

# ── is_analyzing 상태면 실제 분석 실행 ────────────────────────────────
if st.session_state.get("is_analyzing") and st.session_state.get("analyzing_symbol") == symbol:
    run_analysis(symbol, execute_trade=st.session_state.auto_run)
    st.session_state.last_auto_time = time.time()
    st.session_state.is_analyzing    = False
    st.session_state.analyzing_symbol = ""
    st.rerun()

if all_btn:
    for _sym in SYMBOLS:
        with st.spinner(f"🤖 {_sym} 분석 중..."):
            run_analysis(_sym, execute_trade=st.session_state.auto_run)
    st.session_state.last_auto_time = time.time()
    st.rerun()


# ── 즉시 재분석 (TP/SL 청산 감지 → 해당 심볼 즉시 재분석) ──────────────
_reanalyze_q = st.session_state.get("immediate_reanalyze", [])
if BINANCE_READY and _reanalyze_q:
    st.session_state.immediate_reanalyze = []  # 큐 비우기 (먼저 초기화)
    if st.session_state.auto_run and not st.session_state.get("trading_paused"):
        for _rsym in _reanalyze_q:
            if _rsym in SYMBOLS:
                add_log(f"⚡ {_rsym} TP/SL 청산 감지 → 즉시 재분석")
                with st.spinner(f"⚡ {_rsym} 즉시 재분석 중..."):
                    run_analysis(_rsym, execute_trade=True)
        st.session_state.last_auto_time = time.time()
        st.rerun()

# ── 자동 분석 처리 (메인 루프 체크) — 멀티심볼 병렬 실행 ────────────
if auto_interval != "비활성":
    interval_map = {"10분": 600, "30분": 1800, "1시간": 3600}
    secs    = interval_map[auto_interval]
    elapsed = time.time() - st.session_state.last_auto_time
    if elapsed >= secs:
        # SL/TP 자동 청산 감지 + 트레일링 스탑 + 부분 청산 체크 (분석 전 먼저)
        if BINANCE_READY:
            _check_sl_tp_closed()
            _update_trailing_stop()
            _check_partial_tp()
        for _sym in SYMBOLS:
            with st.spinner(f"⏰ 자동 분석: {_sym}..."):
                run_analysis(_sym, execute_trade=st.session_state.auto_run)
        st.session_state.last_auto_time = time.time()
        st.rerun()


# ── 자동 분석 카운트다운 표시 (30초마다 fragment 갱신) ─────────────────
@st.fragment(run_every=30)
def _auto_countdown():
    ai = st.session_state.get("auto_interval_select", "비활성")
    if ai == "비활성":
        return
    # ── 30초마다 SL/TP 청산 감지 + 부분 청산 체크 + 즉시 재분석 트리거 ──
    if BINANCE_READY:
        _check_sl_tp_closed()
        _check_partial_tp()
        if st.session_state.get("immediate_reanalyze"):
            st.rerun(scope="app")  # 즉시 앱 재실행 → 메인 루프에서 재분석
    interval_map = {"10분": 600, "30분": 1800, "1시간": 3600}
    secs      = interval_map[ai]
    elapsed   = time.time() - st.session_state.last_auto_time
    remaining = int(secs - elapsed)
    if remaining <= 0:
        st.rerun(scope="app")
    else:
        # sidebar 대신 메인 영역에 작게 표시
        st.caption(f"⏰ 다음 자동 분석: {remaining // 60}분 {remaining % 60}초 후")

_auto_countdown()


# ── 알트 자동 스캔 루프 (스크리너 3분 + 공지 3분 + 자동 주문) ──────────
if BINANCE_READY and st.session_state.get("alt_auto_scan", True):
    _ALT_SCREEN_INTERVAL   = 180  # 3분
    _ALT_ANNOUNCE_INTERVAL = 180  # 3분

    def _alt_place_order(sym, sig, tj, cand):
        """알트 자동 주문 실행 헬퍼"""
        try:
            _cur_pos = get_positions()
            _alt_cnt = len([p for p in _cur_pos if p["symbol"] not in SYMBOLS])
            if _alt_cnt >= MAX_ALT_POSITIONS:
                add_log(f"⚠️ 알트 자동 주문 스킵: 최대 {MAX_ALT_POSITIONS}포지션 초과")
                return
            # 일일 손실 한도 체크
            _today = datetime.now().strftime("%Y-%m-%d")
            if st.session_state.daily_start_date == _today and st.session_state.daily_start_balance:
                _dloss = st.session_state.daily_start_balance - get_balance()["total"]
                if _dloss >= MAX_DAILY_LOSS:
                    add_log(f"🚨 알트 자동 주문 스킵: 일일 손실 한도 초과")
                    return
            # 중복 포지션 체크
            if any(p["symbol"] == sym for p in _cur_pos):
                add_log(f"⚠️ 알트 자동 주문 스킵: {sym} 이미 포지션 존재")
                return
            _atr = cand.get("atr", 0) or 0
            _px  = cand.get("price", 0)
            _side = "BUY" if sig == "long" else "SELL"
            _sl = tj.get("sl") or (round(_px - _atr * ALT_ATR_SL_MULT, 4) if sig == "long" else round(_px + _atr * ALT_ATR_SL_MULT, 4))
            _tp = tj.get("tp") or (round(_px + _atr * ALT_ATR_TP_MULT, 4) if sig == "long" else round(_px - _atr * ALT_ATR_TP_MULT, 4))
            _r = place_order(sym, _side, MAX_USDT_ALT, ALT_LEVERAGE, sl_price=_sl, tp_price=_tp)
            if _r["success"]:
                _add_journal_entry({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym, "side": "🟢 롱" if sig == "long" else "🔴 숏",
                    "action": "진입", "qty": _r["qty"], "price": _r["price"],
                    "sl": _sl, "tp": _tp, "atr": _atr, "pnl": None,
                })
                _sig_label = "🟢 롱" if sig == "long" else "🔴 숏"
                add_log(f"🤖 알트 자동 진입: {sym} {_sig_label} @ ${_r['price']} (신뢰도 {tj.get('confidence')}%)")
                if st.session_state.get("tg_notify"):
                    send_order(st.session_state.tg_token, st.session_state.tg_chat_id,
                               sym, _side, _r["qty"], _r["price"])
            else:
                add_log(f"❌ 알트 자동 주문 실패: {sym} {_r.get('error','')}", "error")
        except Exception as _e:
            add_log(f"알트 자동 주문 오류: {_e}", "error")

    # 공지 스캔 (3분 주기)
    _ann_elapsed = time.time() - st.session_state.get("alt_last_announcement_time", 0)
    if _ann_elapsed >= _ALT_ANNOUNCE_INTERVAL:
        try:
            _ann_r = check_binance_announcements()
            st.session_state.alt_announcements          = _ann_r
            st.session_state.alt_last_announcement_time = time.time()
            if _ann_r.get("found"):
                add_log(f"📢 공지 감지: {_ann_r.get('summary','')}")
                # 공지 기반 자동 분석 + 자동 주문 (신뢰도 무관, 최우선 신호)
                if st.session_state.get("alt_auto_trade", True):
                    for _ann_item in _ann_r.get("announcements", []):
                        _ann_sym = _ann_item.get("symbol", "")
                        _ann_sig = _ann_item.get("signal", "wait")
                        if not _ann_sym or _ann_sig == "wait":
                            continue
                        if _ann_sym in st.session_state.alt_analysis:
                            continue  # 이미 분석한 종목 스킵
                        try:
                            from binance_client import get_klines as _gk2
                            from indicators import calc_indicators as _ci2
                            _kl2 = _gk2(_ann_sym, "15m", 30)
                            _ind2 = _ci2(_kl2)
                            _cand2 = {
                                "symbol": _ann_sym, "score": 90,
                                "signals": [f"바이낸스 공지: {_ann_item.get('type','')}"],
                                "direction": _ann_sig, "price_change_1h": 0,
                                "vol_ratio": 1, "rsi": _ind2.get("rsi", 50),
                                "funding": 0, "price": _ind2.get("price", 0),
                                "atr": _ind2.get("atr", 0), "atr_pct": 0,
                                "indicators": _ind2, "indicators_1h": {},
                            }
                            _res2 = run_alt_analysis(_cand2)
                            st.session_state.alt_analysis[_ann_sym] = _res2
                            _tj2 = _res2.get("trader_json", {})
                            # 공지는 신뢰도 60% 이상이면 자동 주문
                            if _tj2.get("confidence", 0) >= ALT_MANUAL_MIN_CONF and _tj2.get("signal") != "wait":
                                _alt_place_order(_ann_sym, _tj2["signal"], _tj2, _cand2)
                        except Exception as _e2:
                            add_log(f"공지 자동 분석 오류: {_ann_sym} {_e2}", "error")
        except Exception:
            pass

    # 스크리너 (3분 주기)
    _sc_elapsed = time.time() - st.session_state.get("alt_last_scan_time", 0)
    if _sc_elapsed >= _ALT_SCREEN_INTERVAL:
        try:
            _syms_auto    = get_alt_futures_symbols(ALT_SCAN_LIMIT)
            _results_auto = screen_altcoins(_syms_auto, top_n=5)
            st.session_state.alt_scan_results   = _results_auto
            st.session_state.alt_last_scan_time = time.time()
            if _results_auto:
                _top = _results_auto[0]
                add_log(f"🔥 알트 스캔: {_top['symbol']} 1위 (점수 {_top['score']})")
                # 상위 종목 자동 분석 + 자동 주문
                if st.session_state.get("alt_auto_trade", True) and _top["score"] >= ALT_MIN_SCORE:
                    if _top["symbol"] not in st.session_state.alt_analysis:
                        try:
                            _res_top = run_alt_analysis(_top)
                            st.session_state.alt_analysis[_top["symbol"]] = _res_top
                            _tj_top  = _res_top.get("trader_json", {})
                            _sig_top = _tj_top.get("signal", "wait")
                            _conf_top = _tj_top.get("confidence", 0)
                            # 신뢰도 75% 이상이면 자동 주문
                            if _sig_top != "wait" and _conf_top >= ALT_AUTO_CONFIDENCE:
                                _alt_place_order(_top["symbol"], _sig_top, _tj_top, _top)
                            else:
                                add_log(f"⏸ 알트 자동 주문 보류: {_top['symbol']} 신뢰도 {_conf_top}% (기준 {ALT_AUTO_CONFIDENCE}%)")
                        except Exception as _e3:
                            add_log(f"알트 자동 분석 오류: {_e3}", "error")
        except Exception:
            pass

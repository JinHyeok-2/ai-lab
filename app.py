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
import threading
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
    from binance_client import get_klines, get_price, get_balance, get_positions, place_order, place_limit_order, place_sl_tp, close_position, get_funding_rate, get_recent_trades, update_stop_loss, get_funding_rate_history, get_open_interest, partial_close_position, cancel_open_orders, get_open_orders, get_client, _get_symbol_filters, _round_price, emergency_close_all
    from indicators import calc_indicators, format_for_agent
    from agents import run_agent
    from config import SYMBOLS, LEVERAGE, MAX_USDT, INTERVAL, CANDLE_CNT, TESTNET, POSITION_PCT, SCALE_TABLE
    import trade_db
    from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ATR_SL_MULT, ATR_TP_MULT
    from telegram_notifier import (
        test_connection, send_signal, send_order, send_close, send_error,
        send_progress, send_analysis_summary, send_status, send_daily_limit_alert,
        send_trader_decision, send_hourly_briefing, start_polling_thread, read_and_clear_commands,
        send_message
    )
    from config import MAX_DAILY_LOSS
    from config import MAX_USDT_ALT, MAX_ALT_POSITIONS, ALT_LEVERAGE, ALT_ATR_SL_MULT, ALT_ATR_TP_MULT, ALT_SCAN_LIMIT, ALT_MIN_SCORE, ALT_AUTO_CONFIDENCE, ALT_MANUAL_MIN_CONF, POSITION_PCT_ALT
    from alt_scanner import get_alt_futures_symbols, screen_altcoins, check_binance_announcements, check_upbit_announcements, check_okx_announcements, check_coinbase_listings, run_alt_analysis
    from surge_detector import detect_surges, get_snapshot_count, get_snapshot_age_minutes
    from signal_queue import push_signal
    BINANCE_READY = True
except Exception as e:
    BINANCE_READY = False
    BINANCE_ERR = str(e)
    SYMBOLS      = ["ETHUSDT", "BTCUSDT"]
    LEVERAGE     = 3
    MAX_USDT     = 100
    POSITION_PCT = 12
    SCALE_TABLE  = [(100, 45, 3.0), (50, 40, 2.0), (20, 35, 1.5)]
    INTERVAL     = "15m"
    CANDLE_CNT   = 100
    TESTNET      = True
    TELEGRAM_TOKEN   = ""
    TELEGRAM_CHAT_ID = ""
    ATR_SL_MULT    = 3.0
    ATR_TP_MULT    = 6.0
    MAX_DAILY_LOSS = 200
    MAX_USDT_ALT      = 50
    POSITION_PCT_ALT  = 8
    MAX_ALT_POSITIONS = 2
    ALT_LEVERAGE      = 2
    ALT_ATR_SL_MULT   = 2.0
    ALT_ATR_TP_MULT   = 4.0
    ALT_SCAN_LIMIT    = 50
    ALT_MIN_SCORE          = 30
    ALT_AUTO_CONFIDENCE    = 75
    ALT_MANUAL_MIN_CONF    = 60

# ── SQLite 마이그레이션 (최초 1회) ────────────────────────────────────
try:
    import trade_db
    trade_db.migrate_from_json()
except Exception:
    pass

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
_journal_lock = threading.Lock()

# ── 모니터링 함수 debounce (중복 실행 방지) ────────────────────────────
_last_check_ts = {}
def _debounce(func_name: str, min_interval: float = 10.0) -> bool:
    """최소 간격 이내 재호출이면 True(스킵), 아니면 False(실행)"""
    now = time.time()
    if now - _last_check_ts.get(func_name, 0) < min_interval:
        return True
    _last_check_ts[func_name] = now
    return False

def _load_journal() -> list:
    """SQLite에서 전체 거래 조회 (최신순)"""
    try:
        return trade_db.get_all_trades(limit=500)
    except Exception:
        # 폴백: 기존 JSON
        with _journal_lock:
            try:
                if _JOURNAL_PATH.exists():
                    return json.loads(_JOURNAL_PATH.read_text())
            except Exception:
                pass
        return []

def _save_journal(journal: list):
    """호환용 — SQLite 사용 시 실질적으로 불필요하나 기존 호출 유지"""
    pass

def _add_journal_entry(entry: dict):
    """SQLite에 새 거래 추가"""
    try:
        trade_db.add_trade(entry)
    except Exception:
        # 폴백: 기존 JSON
        with _journal_lock:
            try:
                j = json.loads(_JOURNAL_PATH.read_text()) if _JOURNAL_PATH.exists() else []
            except Exception:
                j = []
            j.insert(0, entry)
            try:
                _JOURNAL_PATH.write_text(json.dumps(j[-200:], ensure_ascii=False, indent=2))
            except Exception:
                pass

def _update_journal_pnl(symbol: str, pnl: float, close_price: float = None):
    """심볼의 가장 최근 미청산(pnl=None) journal 항목에 PnL 기록"""
    try:
        trade_db.update_trade_pnl(symbol, pnl, close_price)
    except Exception:
        # 폴백: 기존 JSON
        j = _load_journal()
        for entry in j:
            if entry.get("symbol") == symbol and entry.get("pnl") is None:
                entry["pnl"]        = round(pnl, 4)
                entry["action"]     = "청산"
                entry["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if close_price is not None:
                    entry["close_price"] = close_price
                break

_MAX_HOLD_HOURS = 24  # 최대 보유 시간 (24h 초과 → 시장가 청산)

def _check_max_hold_time():
    """보유 시간 제한: 24시간 초과 포지션 자동 청산 (장기 보유 손실 방지)"""
    if _debounce("max_hold", 60):
        return
    if not BINANCE_READY:
        return
    try:
        j = _load_journal()
        open_entries = [e for e in j if e.get("pnl") is None and e.get("time")]
        if not open_entries:
            return
        positions = get_positions()
        pos_map = {p["symbol"]: p for p in positions}
        for entry in open_entries:
            sym = entry["symbol"]
            pos = pos_map.get(sym)
            if not pos:
                continue
            try:
                _entry_time = datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S")
                _hours = (datetime.now() - _entry_time).total_seconds() / 3600
            except Exception:
                continue
            if _hours >= _MAX_HOLD_HOURS:
                _upnl = pos.get("unrealized_pnl", 0)
                _pnl_label = "익절" if _upnl >= 0 else "손절"
                add_log(f"⏰ {sym} 보유 {_hours:.1f}h > {_MAX_HOLD_HOURS}h → 시간 초과 {_pnl_label}")
                r = close_position(sym)
                if r.get("success"):
                    _update_journal_pnl(sym, _upnl, pos["entry_price"])
                    add_log(f"✅ {sym} 시간 초과 청산 완료: PnL ${_upnl:+.2f}")
                    if st.session_state.get("tg_notify", True):
                        send_error(*_tg(),
                                   f"⏰ {sym} 보유 {_hours:.0f}h 초과 → {_pnl_label} | PnL ${_upnl:+.2f}")
                else:
                    add_log(f"❌ {sym} 시간 초과 청산 실패: {r.get('error','')}", "error")
    except Exception as e:
        _log(f"보유 시간 체크 오류: {e}", "error")


def _check_soft_sl():
    """소프트 SL 체크 — 박스권(ADX<20) 포지션의 캔들 종가 기준 청산
    STOP_MARKET(하드 SL)은 급락 안전망, 소프트 SL은 캔들 종가 확인 후 청산
    """
    if _debounce("soft_sl", 15):
        return
    if not BINANCE_READY:
        return
    try:
        j = _load_journal()
        # sl_mode="soft"인 열린 포지션만 대상
        soft_entries = [
            e for e in j
            if e.get("pnl") is None and e.get("sl_mode") == "soft" and e.get("sl")
        ]
        if not soft_entries:
            return
        positions = get_positions()
        pos_map = {p["symbol"]: p for p in positions}
        for entry in soft_entries:
            sym = entry["symbol"]
            pos = pos_map.get(sym)
            if not pos:
                continue  # 이미 청산됨 — _check_sl_tp_closed()에서 처리
            soft_sl = float(entry["sl"])
            side = pos["side"]
            # 최근 마감된 캔들의 종가 확인 (현재 캔들 제외 = limit 2의 첫번째)
            try:
                _df = get_klines(sym, interval=INTERVAL, limit=2)
                if _df.empty or len(_df) < 2:
                    continue
                last_close = float(_df.iloc[-2]["close"])  # 마지막 확정 캔들
            except Exception:
                continue
            # 캔들 종가가 소프트 SL을 돌파했는지 확인
            _breached = False
            if side == "LONG" and last_close <= soft_sl:
                _breached = True
            elif side == "SHORT" and last_close >= soft_sl:
                _breached = True
            if _breached:
                _log(f"🔔 {sym} 소프트SL 캔들종가 돌파 (종가:${last_close:.2f}, SL:${soft_sl:.2f}) → 청산")
                add_log(f"🔔 {sym} 소프트SL 발동: 캔들종가 ${last_close:.2f} → 시장가 청산")
                r = close_position(sym)
                if r.get("success"):
                    # PnL 계산 (side 판별: pos["side"]가 가장 신뢰 가능)
                    ep = pos["entry_price"]
                    qty = entry.get("qty", pos["size"])
                    if side == "LONG":
                        _pnl = round((last_close - ep) * qty, 4)
                    else:
                        _pnl = round((ep - last_close) * qty, 4)
                    _update_journal_pnl(sym, _pnl, last_close)
                    add_log(f"📋 {sym} 소프트SL 청산 완료: PnL ${_pnl:+.2f}")
                    if st.session_state.get("tg_notify", True):
                        send_error(*_tg(),
                                   f"🔔 {sym} 소프트SL 청산 | 종가 ${last_close:.2f} | PnL ${_pnl:+.2f}")
                else:
                    _log(f"소프트SL 청산 실패 {sym}: {r.get('error','')}", "error")
    except Exception as e:
        _log(f"소프트SL 체크 오류: {e}", "error")


def _smart_exit_check():
    """스마트 엑싯 — TP 근접 시 모멘텀 체크 후 TP 확장 (SL은 건드리지 않음)"""
    if _debounce("smart_exit", 15):
        return
    if not BINANCE_READY:
        return
    if not st.session_state.get("smart_exit_on", True):
        return
    try:
        j = _load_journal()
        open_entries = [e for e in j if e.get("pnl") is None and e.get("symbol")]
        if not open_entries:
            return
        positions = get_positions()
        pos_map = {p["symbol"]: p for p in positions}
        _updated = False
        for entry in open_entries:
            sym = entry["symbol"]
            pos = pos_map.get(sym)
            if not pos:
                continue
            tp = entry.get("tp")
            atr = entry.get("atr", 0) or 0
            if not tp or atr <= 0:
                continue
            tp = float(tp)
            side = pos["side"]
            ep = pos["entry_price"]
            cur_price_approx = ep + (pos["unrealized_pnl"] / pos["size"]) if side == "LONG" else ep - (pos["unrealized_pnl"] / pos["size"])

            # TP 근접도 계산 (ATR 기준)
            if side == "LONG":
                tp_dist = (tp - cur_price_approx) / atr if atr else 99
            else:
                tp_dist = (cur_price_approx - tp) / atr if atr else 99

            _NEAR_THRESHOLD = 0.5
            _max_adjust = int(entry.get("smart_adjust_count", 0))

            if tp_dist > _NEAR_THRESHOLD or _max_adjust >= 2:
                continue

            # 빠른 지표 조회
            try:
                _df = get_klines(sym, interval=INTERVAL, limit=50)
                if _df.empty or len(_df) < 20:
                    continue
                _ind = calc_indicators(_df)
            except Exception:
                continue

            _rsi = _ind.get("rsi", 50) or 50
            _adx = _ind.get("adx", 0) or 0
            _macd_h = _ind.get("macd_hist", 0) or 0
            _obv = _ind.get("obv_trend", "")

            # ── TP 근접 재평가 ──
            if tp_dist <= _NEAR_THRESHOLD and _max_adjust < 2:
                _extend_tp = False
                _reasons = []

                if side == "LONG":
                    # 롱 TP 근접: 추가 상승 가능성 체크
                    if _rsi < 75 and _rsi > 50:
                        _extend_tp = True
                        _reasons.append(f"RSI 여유({_rsi:.0f})")
                    if _macd_h > 0 and _adx > 25:
                        _extend_tp = True
                        _reasons.append(f"모멘텀 강함(ADX={_adx:.0f})")
                    if _obv == "상승":
                        _extend_tp = True
                        _reasons.append("OBV 상승")
                else:
                    # 숏 TP 근접: 추가 하락 가능성 체크
                    if _rsi > 25 and _rsi < 50:
                        _extend_tp = True
                        _reasons.append(f"RSI 여유({_rsi:.0f})")
                    if _macd_h < 0 and _adx > 25:
                        _extend_tp = True
                        _reasons.append(f"모멘텀 강함(ADX={_adx:.0f})")
                    if _obv == "하락":
                        _extend_tp = True
                        _reasons.append("OBV 하락")

                # 2개 이상 근거 시 TP 확장
                if _extend_tp and len(_reasons) >= 2:
                    # TP를 1 ATR 더 확장
                    if side == "LONG":
                        new_tp = round(tp + atr * 1.0, 2)
                    else:
                        new_tp = round(tp - atr * 1.0, 2)
                    # TP는 TAKE_PROFIT_MARKET 주문 교체
                    try:
                        client = get_client()
                        # 기존 TP 주문 전체 취소 (GTE closePosition 포함)
                        _orders = client.futures_get_open_orders(symbol=sym)
                        for _o in _orders:
                            if _o["type"] in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                                try:
                                    client.futures_cancel_order(symbol=sym, orderId=_o["orderId"])
                                except Exception:
                                    pass
                        time.sleep(0.3)  # 취소 처리 대기
                        # 새 TP 배치
                        _, tick_size = _get_symbol_filters(sym)
                        tp_rounded = _round_price(new_tp, tick_size)
                        tp_side = "SELL" if side == "LONG" else "BUY"
                        client.futures_create_order(
                            symbol=sym, side=tp_side,
                            type="TAKE_PROFIT_MARKET",
                            stopPrice=tp_rounded,
                            closePosition=True,
                        )
                        entry["tp"] = new_tp
                        entry["smart_adjust_count"] = _max_adjust + 1
                        if entry.get("id"):
                            trade_db.update_trade_field(entry["id"], tp=new_tp)
                        _updated = True
                        _reason_str = " + ".join(_reasons)
                        add_log(f"🧠 {sym} 스마트TP 확장: ${tp:.2f}→${new_tp:.2f} ({_reason_str})")
                        _log(f"🧠 스마트엑싯: {sym} TP 확장 {_max_adjust+1}/2회 | {_reason_str}")
                        if st.session_state.get("tg_notify", True):
                            send_error(*_tg(),
                                       f"🧠 {sym} TP 확장: ${tp:.2f}→${new_tp:.2f} | {_reason_str}")
                    except Exception as _e:
                        _log(f"스마트TP 확장 실패 {sym}: {_e}", "error")

        if _updated:
            _save_journal(j)
    except Exception as e:
        _log(f"스마트엑싯 체크 오류: {e}", "error")


def _check_sl_tp_closed():
    """SL/TP 자동 청산 감지 — pnl=None 항목 중 포지션이 사라진 심볼 업데이트"""
    if _debounce("sl_tp_closed", 10):
        return
    j = _load_journal()
    open_entries = [e for e in j if e.get("pnl") is None and e.get("symbol")]
    if not open_entries:
        return
    try:
        _positions = get_positions()
        active_syms = {p["symbol"] for p in _positions}
        # 방어: 열린 저널이 있는데 포지션 API가 빈 결과면 → 오판 방지
        # 체결 내역으로 실제 청산 여부 검증 후 처리
        if not active_syms and len(open_entries) > 0:
            # 2차 검증: 첫 번째 심볼의 최근 체결 확인
            _test_sym = open_entries[0]["symbol"]
            _test_trades = get_recent_trades(_test_sym, limit=5)
            _entry_time = open_entries[0].get("time", "")
            _has_close_trade = any(
                t["realized_pnl"] != 0 and t["time"] >= _entry_time
                for t in _test_trades
            )
            if not _has_close_trade:
                _log("⚠️ 포지션 API 빈 응답 — 실제 청산 미확인, 스킵", "error")
                return
        updated = False
        for entry in open_entries:
            sym = entry["symbol"]
            if sym in active_syms:
                continue  # 아직 포지션 열려있음
            # 포지션 사라짐 → 체결 내역으로 실제 청산 확인
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
                entry_price = entry.get("price", 0)
                # 부분 청산 반영: 잔여 수량으로 계산
                _partial_qty = entry.get("partial_close_qty", 0) or 0
                if not _partial_qty and entry.get("extra"):
                    try:
                        import json as _json
                        _extra = _json.loads(entry["extra"]) if isinstance(entry["extra"], str) else entry.get("extra", {})
                        _partial_qty = _extra.get("partial_close_qty", 0) or 0
                    except Exception:
                        pass
                qty = entry.get("qty", 0) - _partial_qty
                if qty <= 0:
                    qty = entry.get("qty", 0)  # 안전 폴백
                side        = entry.get("side", "")
                last_trade  = recent_trades[-1] if recent_trades else trades[-1]
                close_price = last_trade["price"]
                if "롱" in side:
                    total_rpnl = round((close_price - entry_price) * qty, 4)
                elif "숏" in side:
                    total_rpnl = round((entry_price - close_price) * qty, 4)
            close_px = (recent_trades[-1]["price"] if recent_trades
                        else trades[-1]["price"] if trades else None)
            # 부분 청산 PnL 합산 (50% 청산분 + 잔여분)
            _partial_pnl = 0
            if entry.get("partial_close_pnl"):
                _partial_pnl = entry["partial_close_pnl"]
            elif entry.get("extra"):
                try:
                    import json as _json
                    _ext = _json.loads(entry["extra"]) if isinstance(entry["extra"], str) else entry.get("extra", {})
                    _partial_pnl = _ext.get("partial_close_pnl", 0) or 0
                except Exception:
                    pass
            if _partial_pnl:
                total_rpnl = round(total_rpnl + _partial_pnl, 4)
                _log(f"📋 {sym} 부분 청산 PnL ${_partial_pnl:+.4f} + 잔여분 = 합산 PnL ${total_rpnl:+.4f}")
            _update_journal_pnl(sym, total_rpnl, close_px)
            _log(f"📋 {sym} SL/TP 자동 청산 감지: PnL ${total_rpnl:+.4f}")
            add_log(f"{'🟢' if total_rpnl >= 0 else '🔴'} {sym} SL/TP 자동 청산: PnL ${total_rpnl:+.2f}")
            # 텔레그램 청산 알림 (SL/TP 자동 체결)
            if st.session_state.get("tg_notify", True):
                try:
                    _sl_bal = get_balance().get("total", 0)
                    _sl_today = datetime.now().strftime("%Y-%m-%d")
                    _sl_dpnl = -((st.session_state.daily_start_balance or _sl_bal) - _sl_bal)
                    send_close(*_tg(),
                               sym, total_rpnl, balance=_sl_bal, daily_pnl=_sl_dpnl)
                except Exception:
                    pass
            # 잔존 SL/TP 주문 정리 — 무조건 취소 (get_open_orders가 조건부 주문 미반환 버그)
            cancel_open_orders(sym)
            _log(f"🧹 {sym} 잔존 주문 정리 완료")
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
    """트레일링 스탑: 수익 OR 시간 기반으로 SL을 단계적 타이트닝"""
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

            # ── 시간 기반 SL 타이트닝 (보유 시간이 길어질수록 SL 축소) ──
            _entry_time_str = open_entry.get("time", "")
            _hold_hours = 0
            if _entry_time_str:
                try:
                    _hold_hours = (datetime.now() - datetime.strptime(_entry_time_str, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600
                except Exception:
                    pass
            # 시간 기반 트레일링: 수익 중이면 시간에 따라 SL 이동
            if upnl > 0 and _hold_hours >= 8:
                # 8h+: break-even, 16h+: 수익 50% 확보
                if _hold_hours >= 16:
                    _time_offset = upnl * 0.5 / psize if psize > 0 else 0  # 수익 50% 확보
                else:
                    _time_offset = ep * 0.001  # break-even (수수료 여유)
                if side == "LONG":
                    _time_sl = round(ep + _time_offset, 2)
                else:
                    _time_sl = round(ep - _time_offset, 2)
                _cur_sl = open_entry.get("sl")
                if _cur_sl:
                    _cur_sl = float(_cur_sl)
                    _should_time_update = (side == "LONG" and _time_sl > _cur_sl) or \
                                          (side == "SHORT" and _time_sl < _cur_sl)
                    if _should_time_update:
                        _label = "수익50%확보" if _hold_hours >= 16 else "BE이동"
                        add_log(f"⏰ {sym} 보유 {_hold_hours:.0f}h → SL {_label}: ${_cur_sl} → ${_time_sl}")
                        r = update_stop_loss(sym, _time_sl, side)
                        if r.get("success"):
                            open_entry["sl"] = _time_sl
                            _save_journal(j)
                        continue  # 시간 기반 업데이트 했으면 수익 기반 스킵

            if upnl <= 0:
                continue  # 손실 중이면 수익 기반 트레일링 스킵

            _mult = st.session_state.get("trailing_atr_mult", 0.5)  # 1.0→0.5 (조기 발동)
            # 프로그레시브 트레일링: 수익 단계별 SL 상향
            # 1단계: 수익 ≥ ATR×0.5 → BE (진입가) — 조기 발동
            # 2단계: 수익 ≥ ATR×1.5 → 진입가 + ATR×0.5
            # 3단계: 수익 ≥ ATR×2.5 → 진입가 + ATR×1.0
            _profit_atr = upnl / (atr * psize) if atr * psize > 0 else 0
            if _profit_atr >= _mult:  # 최소 1단계 진입
                # 단계별 SL 계산 (조기 발동: 0.5ATR부터)
                if _profit_atr >= 2.5:
                    _sl_offset = atr * 1.0  # 진입가+1ATR 수익 확정
                elif _profit_atr >= 1.5:
                    _sl_offset = atr * 0.5  # 진입가+0.5ATR
                else:
                    _sl_offset = ep * 0.001  # BE (수수료 여유)

                if side == "LONG":
                    new_sl = round(ep + _sl_offset, 2)
                else:
                    new_sl = round(ep - _sl_offset, 2)

                # 현재 SL보다 더 유리할 때만 이동 (SL은 한 방향으로만)
                current_sl = open_entry.get("sl")
                _should_update = True
                if current_sl:
                    current_sl = float(current_sl)
                    if side == "LONG" and new_sl <= current_sl:
                        _should_update = False
                    elif side == "SHORT" and new_sl >= current_sl:
                        _should_update = False

                if _should_update:
                    # update_stop_loss 내부에서 STOP 계열만 취소+배치 (TP 보존)
                    r = update_stop_loss(sym, new_sl, side)
                    if r["success"]:
                        _stage = "3단계" if _profit_atr >= 2.5 else "2단계" if _profit_atr >= 1.5 else "BE"
                        _log(f"🔄 {sym} 트레일링 {_stage}: SL → ${new_sl:.2f} (수익 {_profit_atr:.1f}ATR)")
                        add_log(f"🔄 {sym} SL → ${new_sl:.2f} ({_stage}, {_profit_atr:.1f}ATR)")
                        open_entry["sl"] = new_sl
                        open_entry["sl_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if open_entry.get("id"):
                            trade_db.update_trade_field(open_entry["id"], sl=new_sl)
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
            # 현재가 역산: LONG → ep + (pnl/qty), SHORT → ep - (pnl/qty)
            if psize > 0:
                cur_price = ep + (upnl / psize) if side == "LONG" else ep - (upnl / psize)
            else:
                cur_price = ep

            open_entry = next(
                (e for e in j if e.get("symbol") == sym and e.get("pnl") is None),
                None
            )
            if not open_entry:
                continue
            # 이미 부분 청산한 포지션 → 저널 기반 확인 (세션 초기화 안전)
            if open_entry.get("partial_close_qty"):
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

            # 1) 기존 SL/TP 주문 전부 취소 (레이스 컨디션 방지)
            #    TP 주문(closePosition=True)이 남아있으면 잔여 포지션까지 전량 청산될 수 있음
            cancel_open_orders(sym)
            _log(f"🧹 {sym} 부분 청산 전 기존 SL/TP 주문 취소")

            # 2) 50% 부분 청산 실행
            r = partial_close_position(sym, close_pct=0.5)
            if r["success"]:
                _closed_qty = r["qty"]
                _remain_qty = round(psize - _closed_qty, 8)
                # 부분 청산분 실현 PnL 계산 (총 PnL의 약 50%)
                _partial_pnl = round(upnl * (_closed_qty / psize), 4) if psize > 0 else 0
                # 실제 체결 내역에서 정확한 PnL 가져오기
                try:
                    _pt = get_recent_trades(sym, limit=5)
                    _pt_rpnl = sum(t["realized_pnl"] for t in _pt if t["realized_pnl"] != 0)
                    if _pt_rpnl != 0:
                        _partial_pnl = round(_pt_rpnl, 4)
                except Exception:
                    pass
                _log(f"🎯 {sym} TP 50% 부분 청산 완료 (청산: {_closed_qty}, 잔여: {_remain_qty}, PnL: ${_partial_pnl:+.4f})")
                add_log(f"🎯 {sym} TP 도달 → 50% 부분 청산 | PnL ${_partial_pnl:+.4f}")
                # 저널에 잔여 수량 갱신 + 부분 청산 PnL 기록
                open_entry["qty"] = _remain_qty
                open_entry["partial_close_qty"] = _closed_qty
                open_entry["partial_close_pnl"] = _partial_pnl
                open_entry["partial_close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if open_entry.get("id"):
                    # extra에 부분 청산 PnL 저장 (최종 합산용)
                    import json as _json
                    _ext = {}
                    try:
                        _old_ext = open_entry.get("extra")
                        if _old_ext and isinstance(_old_ext, str):
                            _ext = _json.loads(_old_ext)
                        elif isinstance(_old_ext, dict):
                            _ext = _old_ext
                    except Exception:
                        pass
                    _ext["partial_close_pnl"] = _partial_pnl
                    _ext["partial_close_qty"] = _closed_qty
                    trade_db.update_trade_field(open_entry["id"], qty=_remain_qty,
                                                extra=_json.dumps(_ext, ensure_ascii=False))
                # 부분 청산 완료 플래그
                _done = st.session_state.get("partial_tp_done", {})
                _done[sym] = True
                st.session_state.partial_tp_done = _done
                # 3) SL → break-even 이동 (재시도 2회)
                new_sl = round(ep * 1.001, 2) if side == "LONG" else round(ep * 0.999, 2)
                _sl_ok = False
                for _sl_try in range(2):
                    _sl_r = update_stop_loss(sym, new_sl, side)
                    if _sl_r["success"]:
                        add_log(f"🔄 {sym} SL → break-even ${new_sl:.2f}")
                        open_entry["sl"] = new_sl
                        open_entry["sl_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if open_entry.get("id"):
                            trade_db.update_trade_field(open_entry["id"], sl=new_sl)
                        _sl_ok = True
                        break
                    time.sleep(0.5)
                if not _sl_ok:
                    # SL 배치 실패 → 안전을 위해 잔여 포지션 전량 청산
                    _log(f"🚨 {sym} SL 배치 2회 실패 → 잔여 포지션 전량 청산", "error")
                    add_log(f"🚨 {sym} SL 배치 실패 → 안전 청산 실행", "error")
                    _emergency = close_position(sym)
                    if _emergency["success"]:
                        add_log(f"✅ {sym} 안전 청산 완료")
                    else:
                        add_log(f"🚨 {sym} 안전 청산도 실패! 수동 확인 필요", "error")
                    if st.session_state.get("tg_notify", True):
                        send_error(*_tg(),
                                   f"🚨 {sym} 부분 청산 후 SL 배치 실패 → 안전 청산 실행")
                if st.session_state.get("tg_notify", True):
                    send_error(*_tg(),
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


def calc_trade_stats(journal: list = None, symbol: str = None) -> dict:
    """거래 통계 — SQLite 기반 (journal 인자는 호환용, 무시)"""
    try:
        return trade_db.get_trade_stats(symbol=symbol)
    except Exception:
        # 폴백: 리스트 기반
        if journal is None:
            journal = _load_journal()
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


def _parse_gate_json(raw: str) -> dict:
    """gate 에이전트 JSON 출력 파싱. 실패 시 폴백(통과)."""
    import re, json as _json
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = _json.loads(match.group())
            return {
                "should_trade": bool(data.get("should_trade", True)),
                "reason": data.get("reason", ""),
                "risk_level": data.get("risk_level", "medium"),
            }
        except Exception:
            pass
    # 폴백: 텍스트에 "false" 포함 시 차단
    if "false" in raw.lower() and "should_trade" in raw.lower():
        return {"should_trade": False, "reason": raw[:200], "risk_level": "high"}
    return {"should_trade": True, "reason": "파싱 실패 — 폴백 통과", "risk_level": "medium"}


def trader_signal_text(signal: str) -> str:
    """signal → 표시용 텍스트"""
    return {"long": "🟢 롱 진입", "short": "🔴 숏 진입", "wait": "⚪ 관망"}.get(signal, "⚪ 관망")


def generate_rule_signal(ind_15m: dict, ind_1h: dict, ind_4h: dict = None,
                          oi_delta_pct: float = 0, cvd_trend: str = None) -> dict:
    """규칙 기반 시그널 생성 — 통계적 에지가 검증된 조건만 사용.
    반환: {"signal": "long"|"short"|"wait", "score": 0~100, "reasons": [...]}
    """
    long_score = 0
    short_score = 0
    reasons = []

    price  = ind_15m.get("price", 0) or 0
    rsi    = ind_15m.get("rsi", 50) or 50
    stoch  = ind_15m.get("stoch_k", 50) or 50
    ema20  = ind_15m.get("ema20") or 0
    ema50  = ind_15m.get("ema50") or 0
    macd_h = ind_15m.get("macd_hist")
    adx    = ind_15m.get("adx", 0) or 0
    dmp    = ind_15m.get("adx_dmp", 0) or 0
    dmn    = ind_15m.get("adx_dmn", 0) or 0
    bb_up  = ind_15m.get("bb_upper", 0) or 0
    bb_lo  = ind_15m.get("bb_lower", 0) or 0
    obv_trend = ind_15m.get("obv_trend", "")

    rsi_1h   = (ind_1h.get("rsi", 50) or 50) if ind_1h else 50
    ema20_1h = (ind_1h.get("ema20") or 0) if ind_1h else 0
    ema50_1h = (ind_1h.get("ema50") or 0) if ind_1h else 0
    macd_1h  = ind_1h.get("macd_hist") if ind_1h else None

    # ── 규칙 1: EMA 크로스 + 추세 (가장 기본) ──
    if ema20 > ema50 and price > ema20:
        long_score += 15
        reasons.append("EMA20>50+가격위")
    elif ema20 < ema50 and price < ema20:
        short_score += 15
        reasons.append("EMA20<50+가격아래")

    # ── 규칙 2: RSI 극단 (과매도/과매수 반전) ──
    if rsi < 30:
        long_score += 12
        reasons.append(f"RSI과매도({rsi:.0f})")
    elif rsi > 70:
        short_score += 12
        reasons.append(f"RSI과매수({rsi:.0f})")

    # ── 규칙 3: Stoch RSI 극단 ──
    if stoch < 20:
        long_score += 10
        reasons.append(f"StochRSI과매도({stoch:.0f})")
    elif stoch > 80:
        short_score += 10
        reasons.append(f"StochRSI과매수({stoch:.0f})")

    # ── 규칙 4: ADX + DI 방향 (추세 존재 + 방향) ──
    if adx >= 25:
        if dmp > dmn:
            long_score += 10
            reasons.append(f"ADX강+DI+({adx:.0f})")
        else:
            short_score += 10
            reasons.append(f"ADX강+DI-({adx:.0f})")

    # ── 규칙 5: MACD 히스토그램 ──
    if macd_h is not None:
        if macd_h > 0:
            long_score += 8
        elif macd_h < 0:
            short_score += 8

    # ── 규칙 6: 1H TF 컨플루언스 (같은 방향이면 보너스) ──
    if ema20_1h > ema50_1h:
        long_score += 10
        reasons.append("1H상승추세")
    elif ema20_1h < ema50_1h:
        short_score += 10
        reasons.append("1H하락추세")

    if macd_1h is not None:
        if macd_1h > 0:
            long_score += 5
        elif macd_1h < 0:
            short_score += 5

    # ── 규칙 7: 볼린저밴드 위치 ──
    if bb_up > bb_lo > 0 and price:
        bb_pct = (price - bb_lo) / (bb_up - bb_lo)
        if bb_pct < 0.2:
            long_score += 8
            reasons.append("BB하단")
        elif bb_pct > 0.8:
            short_score += 8
            reasons.append("BB상단")

    # ── 규칙 7b: OBV 거래량 방향 ──
    if obv_trend == "up":
        long_score += 6
        reasons.append("OBV상승")
    elif obv_trend == "down":
        short_score += 6
        reasons.append("OBV하락")

    # ── 규칙 7c: 4H 타임프레임 컨플루언스 (대추세 확인) ──
    if ind_4h:
        ema20_4h = (ind_4h.get("ema20") or 0)
        ema50_4h = (ind_4h.get("ema50") or 0)
        rsi_4h   = (ind_4h.get("rsi", 50) or 50)
        macd_4h  = ind_4h.get("macd_hist")
        # EMA 추세
        if ema20_4h > ema50_4h:
            long_score += 8
            reasons.append("4H상승추세")
        elif ema20_4h < ema50_4h:
            short_score += 8
            reasons.append("4H하락추세")
        # RSI 극단
        if rsi_4h < 35:
            long_score += 5
            reasons.append(f"4H_RSI과매도({rsi_4h:.0f})")
        elif rsi_4h > 65:
            short_score += 5
            reasons.append(f"4H_RSI과매수({rsi_4h:.0f})")
        # MACD 방향
        if macd_4h is not None:
            if macd_4h > 0:
                long_score += 4
            elif macd_4h < 0:
                short_score += 4

    # ── 규칙 8: OI(미결제약정) 변화율 ──
    if abs(oi_delta_pct) >= 0.5:
        if oi_delta_pct > 0:
            long_score += 6  # OI 증가 → 새 포지션 유입 → 추세 지속
            reasons.append(f"OI증가({oi_delta_pct:+.1f}%)")
        else:
            short_score += 6  # OI 감소 → 포지션 청산 → 하락 압력
            reasons.append(f"OI감소({oi_delta_pct:+.1f}%)")

    # ── 규칙 9: CVD(매수/매도 누적 압력) ──
    if cvd_trend:
        if cvd_trend == "up":
            long_score += 6
            reasons.append("CVD매수압력")
        elif cvd_trend == "down":
            short_score += 6
            reasons.append("CVD매도압력")

    # ── 규칙 10: RSI 다이버전스 ──
    _rsi_div = ind_15m.get("rsi_divergence")
    if _rsi_div == "bullish":
        long_score += 10
        reasons.append("RSI불리시다이버전스")
    elif _rsi_div == "bearish":
        short_score += 10
        reasons.append("RSI베어리시다이버전스")

    # ── 최종 판정 ──
    _diff = abs(long_score - short_score)
    if _diff < 10:  # 차이 10 미만 → 불확실 → wait
        return {"signal": "wait", "score": max(long_score, short_score), "reasons": reasons, "long_score": long_score, "short_score": short_score}
    if long_score > short_score:
        return {"signal": "long", "score": long_score, "reasons": reasons, "long_score": long_score, "short_score": short_score}
    else:
        return {"signal": "short", "score": short_score, "reasons": reasons, "long_score": long_score, "short_score": short_score}


def calc_confidence(result: dict) -> tuple:
    """분석 결과에서 신뢰도 점수(0-100)와 방향 반환"""
    score = 10  # 베이스 점수
    _agree_count = 0  # 방향 동의 지표 카운트 (최소 4개 필요)

    tj = result.get("trader_json", {})
    direction = tj.get("signal", "wait")
    if direction == "wait":
        return 30, "wait"

    # 트레이더 JSON confidence를 베이스로 활용
    trader_conf = tj.get("confidence", 50)
    score = int(trader_conf * 0.35)  # 35%로 조정 (지표 자체 점수 비중 확대)

    ind    = result.get("indicators", {})    # 15m 지표
    ind_1h = result.get("indicators_1h", {}) # 1h 지표

    # 1. 컨플루언스 (25~30점) — 3TF 방향 일치 시 보너스
    cf_type = result.get("confluence_type", "mixed")
    cf_label = result.get("confluence", "")
    if cf_type == direction:
        score += 25
        _agree_count += 1
        # 3TF 완전 일치 시 추가 5점
        if "3TF" in cf_label:
            score += 5
    elif cf_type == "mixed":
        score += 5
    else:
        score -= 10

    # 2. RL 앙상블 신호 (ETH 전용, exp14+seed700+eth_seed800 만장일치)
    rl = result.get("rl", {})
    if rl.get("available"):
        rl_type = rl.get("type", "wait")
        if rl_type == direction:
            score += 20  # 방향 일치 → 강한 보너스 (15→20)
            _agree_count += 1
        elif rl_type == "wait":
            score -= 10  # RL 거부권: 관망이면 페널티 (3→-10)
        elif rl_type in ("long", "short") and rl_type != direction:
            score -= 8   # 방향 충돌 → 페널티 (0→-8)
    else:
        score += 5

    # 3. 에이전트 합의 메커니즘 (텍스트 방향 일치 분석)
    keywords_long  = ["상승", "매수", "롱", "bullish", "buy", "상방"]
    keywords_short = ["하락", "매도", "숏", "bearish", "sell", "하방"]
    kw_agree = keywords_long if direction == "long" else keywords_short
    kw_oppose = keywords_short if direction == "long" else keywords_long
    _agent_agree = 0
    _agent_oppose = 0
    for txt in [result.get("analyst",""), result.get("news",""), result.get("risk","")]:
        if any(k in txt for k in kw_agree):
            _agent_agree += 1
            score += 6
        elif any(k in txt for k in kw_oppose):
            _agent_oppose += 1
    # 합의 보너스/페널티
    if _agent_agree >= 3:
        score += 8  # 3/3 에이전트 합의 → 강한 보너스
    elif _agent_agree <= 1 and _agent_oppose >= 1:
        score -= 5  # 1개 이하 동의 + 반대 존재 → 페널티

    # 4. RSI 방향 일치 (최대 12점) — 15m 기준
    rsi = ind.get("rsi")
    if rsi:
        if direction == "long" and rsi < 25:
            score += 12; _agree_count += 1  # 극단 과매도 → 강한 반등 기대
        elif direction == "long" and rsi < 40:
            score += 8; _agree_count += 1   # 과매도 → 반등 기대
        elif direction == "long" and 40 <= rsi <= 60:
            score += 4
        elif direction == "long" and rsi > 70:
            score -= 5   # 과매수에서 롱 → 페널티
        elif direction == "short" and rsi > 60:
            score += 8; _agree_count += 1   # 과매수 → 하락 기대
        elif direction == "short" and 40 <= rsi <= 60:
            score += 4
        elif direction == "short" and rsi < 25:
            score -= 8   # 극단 과매도에서 숏 → 강한 페널티
        elif direction == "short" and rsi < 30:
            score -= 5   # 과매도에서 숏 → 페널티

    # 5. MACD 히스토그램 방향 (10점) — 15m
    macd_hist = ind.get("macd_hist")
    if macd_hist is not None:
        if direction == "long" and macd_hist > 0:
            score += 10; _agree_count += 1
        elif direction == "short" and macd_hist < 0:
            score += 10; _agree_count += 1
        else:
            score -= 3

    # 6. EMA 배열 (추세) (8점) — 15m
    ema20 = ind.get("ema20")
    ema50 = ind.get("ema50")
    price = ind.get("price")
    if ema20 and ema50 and price:
        if direction == "long" and ema20 > ema50 and price > ema20:
            score += 8; _agree_count += 1   # 골든크로스 + 가격 위
        elif direction == "short" and ema20 < ema50 and price < ema20:
            score += 8; _agree_count += 1   # 데드크로스 + 가격 아래
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
    from datetime import timezone
    _utc_hour = datetime.now(timezone.utc).hour
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

    # 20. 연속 방향 편향 보정 (±5점) — 한쪽으로 몰리면 페널티
    _hist = _load_history()
    if len(_hist) >= 10:
        _recent = _hist[-20:]
        _short_cnt = sum(1 for h in _recent if "숏" in h.get("decision", ""))
        _long_cnt  = sum(1 for h in _recent if "롱" in h.get("decision", ""))
        _total_sig = _short_cnt + _long_cnt
        if _total_sig >= 5:
            if direction == "short" and _short_cnt / _total_sig >= 0.8:
                score -= 5   # 최근 80%+ 숏 → 숏 편향 페널티
            elif direction == "long" and _long_cnt / _total_sig >= 0.8:
                score -= 5   # 최근 80%+ 롱 → 롱 편향 페널티

    # 21. 지표 동의 최소 개수 캡 — 4개 미만 동의 시 70% 상한
    # 동의 지표: 컨플루언스, RL, RSI, MACD, EMA (5개 중 4+ 필요)
    if _agree_count < 4:
        score = min(score, 69)  # 70% 미만으로 캡 → 자동 진입 불가

    # 22. 신뢰도 캘리브레이션 (실제 승률 기반 보정, 강화)
    # 콜드 스타트: 거래 10건 미만이면 캘리브레이션 데이터 부족 → 보정 스킵
    _total_trades = 0
    try:
        import sqlite3 as _sql3
        _conn = _sql3.connect("trades.db")
        _total_trades = _conn.execute("SELECT COUNT(*) FROM trades WHERE pnl != 0 AND close_price > 0").fetchone()[0]
        _conn.close()
    except Exception:
        pass
    if _total_trades >= 10:
        try:
            _cal = trade_db.get_confidence_calibration()
            for (lo, hi), data in _cal.items():
                if lo <= score <= hi and data["count"] >= 3:
                    _adj = data["adj"] * 2
                    if lo >= 80 and data["win_rate"] == 0 and data["count"] >= 2:
                        _adj = min(_adj, -15)
                    score += _adj
                    break
        except Exception:
            pass

    # 23. 고래 추적 시그널 (±8점) — 거래소 입출금 흐름
    _whale = result.get("whale", {})
    _whale_sig = _whale.get("signal", "neutral")
    if _whale_sig == "bearish":
        if direction == "short": score += 8    # 매도 압력 + 숏 → 강한 근거
        elif direction == "long": score -= 6   # 매도 압력 + 롱 → 역행
    elif _whale_sig == "bullish":
        if direction == "long": score += 8     # 축적 + 롱 → 강한 근거
        elif direction == "short": score -= 6  # 축적 + 숏 → 역행
    elif _whale_sig == "slightly_bearish":
        if direction == "short": score += 4
        elif direction == "long": score -= 3
    elif _whale_sig == "slightly_bullish":
        if direction == "long": score += 4
        elif direction == "short": score -= 3

    # 24. 고신뢰도 과신뢰 방지 — 80+ 구간 데이터 5건 미만이면 70% 캡
    # 콜드 스타트(10건 미만): 이 캡 비활성화 (진입 기회 확보)
    if score >= 80 and _total_trades >= 10:
        try:
            _cal80 = trade_db.get_confidence_calibration().get((80, 100), {})
            if _cal80.get("count", 0) < 5:
                score = min(score, 70)
        except Exception:
            score = min(score, 70)

    return max(0, min(100, score)), direction


def calc_confidence_alt(candidate: dict, analysis: dict, btc_ind: dict = None) -> tuple:
    """알트코인 독립 신뢰도 계산 (0-100) + 방향.
    candidate: 스크리너 결과, analysis: run_alt_analysis 결과, btc_ind: BTC 15m 지표"""
    tj = analysis.get("trader_json", {})
    direction = tj.get("signal", "wait")
    if direction == "wait":
        return 25, "wait"

    score = 20  # 베이스 (스크리너가 이미 필터링했으므로 20 시작)
    ind_15m = candidate.get("indicators", {})
    ind_1h  = candidate.get("indicators_1h", {})

    # 1. 스크리너 점수 (최대 20점) — 스크리너가 높은 점수로 선별했으면 보너스
    _sc_score = candidate.get("score", 0)
    if _sc_score >= 60:
        score += 20
    elif _sc_score >= 40:
        score += 12
    elif _sc_score >= 25:
        score += 6

    # 2. 거래량 배수 (최대 15점) — 거래량 급등은 강한 신호
    _vol_ratio = candidate.get("vol_ratio", 1)
    if _vol_ratio >= 5:
        score += 15
    elif _vol_ratio >= 3:
        score += 10
    elif _vol_ratio >= 2:
        score += 5

    # 3. RSI 방향 일치 (최대 12점)
    # ※ ADX 40+ 강추세에서 RSI 극단은 반전 아닌 추세 지속 → 역추세 보상 제거
    rsi = ind_15m.get("rsi", 50) or 50
    _adx_raw = ind_15m.get("adx", 0) or 0
    _strong_trend = _adx_raw >= 40
    if direction == "long" and rsi < 25:
        score += 12 if not _strong_trend else 4  # 강추세 과매도 롱 → 축소
    elif direction == "long" and rsi < 40:
        score += 8
    elif direction == "short" and rsi > 75:
        score += 12 if not _strong_trend else 4  # 강추세 과매수 숏 → 축소
    elif direction == "short" and rsi > 60:
        score += 8
    elif direction == "long" and rsi > 70:
        score -= 8  # 과매수에서 롱
    elif direction == "short" and rsi < 30:
        score -= 8  # 과매도에서 숏

    # 4. MACD 방향 일치 (8점)
    macd_hist = ind_15m.get("macd_hist")
    if macd_hist is not None:
        if direction == "long" and macd_hist > 0:
            score += 8
        elif direction == "short" and macd_hist < 0:
            score += 8
        else:
            score -= 4

    # 5. EMA 배열 (8점)
    ema20 = ind_15m.get("ema20")
    ema50 = ind_15m.get("ema50")
    price = ind_15m.get("price", 0)
    if ema20 and ema50 and price:
        if direction == "long" and ema20 > ema50 and price > ema20:
            score += 8
        elif direction == "short" and ema20 < ema50 and price < ema20:
            score += 8
        elif direction == "long" and ema20 < ema50:
            score -= 4
        elif direction == "short" and ema20 > ema50:
            score -= 4

    # 6. ADX 추세 강도 (8점) + 초강추세 역추세 페널티
    adx = ind_15m.get("adx", 0) or 0
    dmp = ind_15m.get("adx_dmp", 0) or 0
    dmn = ind_15m.get("adx_dmn", 0) or 0
    _is_counter_trend = (direction == "long" and dmn > dmp) or (direction == "short" and dmp > dmn)
    if adx >= 25:
        if not _is_counter_trend:
            score += 8
        else:
            score -= 4
        # ADX 40+ 초강추세에서 역추세 → 강한 페널티 (ADX 비례)
        if adx >= 40 and _is_counter_trend:
            _adx_penalty = 15 + int((adx - 40) * 0.5)  # ADX 58 → -24pt
            score -= _adx_penalty
    elif adx < 15:
        score -= 3  # 추세 없는 알트 = 위험

    # 7. 1H 방향 일치 (6점)
    macd_1h = ind_1h.get("macd_hist")
    if macd_1h is not None:
        if direction == "long" and macd_1h > 0:
            score += 6
        elif direction == "short" and macd_1h < 0:
            score += 6
        else:
            score -= 3

    # 8. 펀딩비 역추세 (6점)
    funding = candidate.get("funding", 0) or 0
    if direction == "long" and funding < -0.03:
        score += 6  # 숏 과열 → 롱 유리
    elif direction == "short" and funding > 0.03:
        score += 6  # 롱 과열 → 숏 유리
    elif direction == "long" and funding > 0.05:
        score -= 5  # 롱 과열인데 롱
    elif direction == "short" and funding < -0.05:
        score -= 5  # 숏 과열인데 숏

    # 9. BTC 상관관계 (±10점) — 알트 핵심 필터
    if btc_ind:
        btc_ema20 = btc_ind.get("ema20", 0) or 0
        btc_ema50 = btc_ind.get("ema50", 0) or 0
        btc_rsi   = btc_ind.get("rsi", 50) or 50
        btc_change = btc_ind.get("change_pct", 0) or 0
        # BTC 급락 중 알트 롱 → 강한 페널티
        if direction == "long" and btc_change < -1.0 and btc_ema20 < btc_ema50:
            score -= 10
        # BTC 급등 중 알트 숏 → 페널티
        elif direction == "short" and btc_change > 1.0 and btc_ema20 > btc_ema50:
            score -= 10
        # BTC와 같은 방향 → 보너스
        elif direction == "long" and btc_ema20 > btc_ema50 and btc_rsi > 45:
            score += 6
        elif direction == "short" and btc_ema20 < btc_ema50 and btc_rsi < 55:
            score += 6

    # 10. 에이전트 텍스트 합의 (±8점)
    kw_long  = ["상승", "매수", "롱", "bullish", "buy", "상방"]
    kw_short = ["하락", "매도", "숏", "bearish", "sell", "하방"]
    kw_agree = kw_long if direction == "long" else kw_short
    kw_oppose = kw_short if direction == "long" else kw_long
    _agree = 0
    for txt in [analysis.get("analyst",""), analysis.get("news","")]:
        if any(k in txt for k in kw_agree):
            _agree += 1
            score += 4
        elif any(k in txt for k in kw_oppose):
            score -= 3
    if _agree >= 2:
        score += 4  # analyst+news 모두 동의

    # 볼린저밴드 위치 (5점) — 강추세 역추세 시 BB 극단은 추세 지속 신호
    bb_upper = ind_15m.get("bb_upper")
    bb_lower = ind_15m.get("bb_lower")
    if bb_upper and bb_lower and price:
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            pct = (price - bb_lower) / bb_range
            if direction == "long" and pct < 0.2:
                score += 5 if not _strong_trend else 2
            elif direction == "short" and pct > 0.8:
                score += 5 if not _strong_trend else 2

    # 11. 멀티 타임프레임 방향 합의 (최대 10점 / -8점)
    ind_4h = candidate.get("indicators_4h", {})
    _tf_agree = 0
    _tf_total = 0
    # 15m 방향
    if ema20 and ema50:
        _tf_total += 1
        if direction == "long" and ema20 > ema50:
            _tf_agree += 1
        elif direction == "short" and ema20 < ema50:
            _tf_agree += 1
    # 1h 방향
    _ema20_1h = ind_1h.get("ema20", 0) or 0
    _ema50_1h = ind_1h.get("ema50", 0) or 0
    if _ema20_1h and _ema50_1h:
        _tf_total += 1
        if direction == "long" and _ema20_1h > _ema50_1h:
            _tf_agree += 1
        elif direction == "short" and _ema20_1h < _ema50_1h:
            _tf_agree += 1
    # 4h 방향
    _ema20_4h = ind_4h.get("ema20", 0) or 0
    _ema50_4h = ind_4h.get("ema50", 0) or 0
    if _ema20_4h and _ema50_4h:
        _tf_total += 1
        if direction == "long" and _ema20_4h > _ema50_4h:
            _tf_agree += 1
        elif direction == "short" and _ema20_4h < _ema50_4h:
            _tf_agree += 1
    if _tf_total >= 2:
        if _tf_agree >= 3:
            score += 10  # 3TF 모두 동의
        elif _tf_agree >= 2:
            score += 5   # 2TF 동의
        elif _tf_agree == 0:
            score -= 8   # 전부 역행

    # 12. 스크리너↔신뢰도 방향 교차 검증 (±6점)
    _screener_dir = candidate.get("direction", "wait")
    if _screener_dir != "wait" and direction != "wait":
        if _screener_dir == direction:
            score += 6  # 스크리너와 에이전트 방향 일치
        else:
            score -= 6  # 방향 불일치 → 페널티

    # 13. 공포탐욕 지수 보너스 (최대 8점) — 극도 공포 시 역발상 롱 보너스
    _fgi = 50
    try:
        for _sym_key in st.session_state.get("last_analysis", {}):
            _fg_data = st.session_state.last_analysis[_sym_key].get("fear_greed", {})
            if _fg_data and _fg_data.get("available"):
                _fgi = int(_fg_data.get("value", 50))
                break
    except Exception:
        pass
    if _fgi < 20 and direction == "long":
        score += 8
    elif _fgi < 30 and direction == "long":
        score += 4
    elif _fgi > 80 and direction == "short":
        score += 8
    elif _fgi > 70 and direction == "short":
        score += 4

    # 14. BTC 1H 동조 보너스 (±8점) — A+C 전략
    if btc_ind:
        _btc_ema20 = btc_ind.get("ema20", 0) or 0
        _btc_ema50 = btc_ind.get("ema50", 0) or 0
        if _btc_ema20 and _btc_ema50:
            _btc_bullish = _btc_ema20 > _btc_ema50
            if direction == "long" and _btc_bullish:
                score += 8   # BTC 상승 + 알트 롱 = 동조
            elif direction == "short" and not _btc_bullish:
                score += 8   # BTC 하락 + 알트 숏 = 동조
            elif direction == "long" and not _btc_bullish:
                score -= 6   # BTC 하락인데 롱 = 역행
            elif direction == "short" and _btc_bullish:
                score -= 6   # BTC 상승인데 숏 = 역행

    # 15. 알트 시즌 감지 (±6점) — BTC 횡보/하락 + 알트 상승 = 알트 시즌
    if btc_ind:
        _btc_chg = btc_ind.get("change_pct", 0) or 0
        _btc_adx = btc_ind.get("adx", 25) or 25
        _alt_chg = ind_15m.get("change_pct", 0) or 0
        # BTC 횡보(ADX<20) 또는 하락 + 알트 상승 → 알트 시즌
        if (_btc_adx < 20 or _btc_chg < -0.5) and _alt_chg > 0.5 and direction == "long":
            score += 6  # 알트 시즌 롱 보너스
        # BTC 급등 중 알트 정체 → 알트 약세
        elif _btc_chg > 1.5 and _alt_chg < 0.3 and direction == "long":
            score -= 4  # BTC로 자금 쏠림

    # 15. 승률 기반 동적 보정 (±8점) — 종목+방향별 과거 실적
    _sym_name = candidate.get("symbol", "")
    try:
        _wr = trade_db.get_symbol_direction_winrate(_sym_name, direction, recent_n=10)
        if _wr["trades"] >= 3:
            if _wr["win_rate"] >= 70:
                score += 8
            elif _wr["win_rate"] >= 50:
                score += 4
            elif _wr["win_rate"] <= 20:
                score -= 8
            elif _wr["win_rate"] <= 35:
                score -= 4
    except Exception:
        pass

    # 15. 펀딩비 팜 보너스 (6점) — 극단 펀딩비 역방향 포지션 수익 기회
    funding = candidate.get("funding", 0) or 0
    if abs(funding) >= 0.1:
        # 펀딩비 ±0.1% 이상이면 역방향 포지션으로 펀딩비 수취 가능
        if (direction == "long" and funding < -0.1) or (direction == "short" and funding > 0.1):
            score += 6  # 역방향 = 펀딩비 수취 보너스

    # 16. RL 범용모델 시그널 (확률 기반 동적 가중치)
    try:
        _rl_alt = get_rl_signal(_sym_name)
        if _rl_alt.get("available"):
            _rl_type = _rl_alt.get("type", "wait")
            _rl_bonus = _rl_alt.get("confidence_bonus", 0)
            _long_prob = _rl_alt.get("long_prob", 0)
            if _rl_type == "long" and direction == "long":
                # 기본 +10 + 확률 보너스(1~5pt) = 최대 +15pt
                score += 10 + _rl_bonus
            elif _rl_type == "long" and direction == "short":
                score -= 8   # RL 롱인데 숏 진입 → 페널티
            elif _rl_type == "wait" and direction == "long":
                # 관망이지만 롱 확률이 높으면 페널티 축소
                if _long_prob >= 0.3:
                    score -= 1   # 롱 확률 30%+ → 약한 페널티
                else:
                    score -= 4   # 롱 확률 낮음 → 강한 페널티
    except Exception:
        pass

    return max(0, min(100, score)), direction


defaults = {
    "logs": [],
    "agent_status": {"analyst": "대기", "news": "대기", "risk": "대기", "gate": "대기", "trader": "대기"},
    "last_analysis": _load_analysis(),
    "analysis_history": _load_history(),
    "auto_run": True,
    "selected_symbol": "ETHUSDT",
    "last_auto_time": 0,
    "gate_block_streak": {},    # 심볼별 연속 gate 차단 횟수
    "pos_skip_counter":  0,     # 포지션 보유 시 분석 스킵 카운터
    "last_hourly_briefing": -1,  # 마지막 정시 브리핑 시각 (hour)
    "is_analyzing": False,
    "analyzing_symbol": "",
    "tg_token":       TELEGRAM_TOKEN   if BINANCE_READY else "",
    "tg_chat_id":     TELEGRAM_CHAT_ID if BINANCE_READY else "",
    "tg_notify":      True,
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
    "alt_scan_thread_started":   False, # 알트 스캔 스레드 시작 여부
    "alt_auto_trade":            True, # 알트 자동 주문 ON/OFF
    "cooldown_until":            0,    # 연속 손실 쿨다운 종료 timestamp
    "consec_loss_count":   5,     # 쿨다운 발동 연속 손실 횟수 (테스트: 3→5)
    "consec_loss_hours":   0.5,   # 쿨다운 지속 시간 (테스트: 2시간→30분)
    "atr_volatility_mult": 2.0,   # 변동성 필터 배수 (기본 2배)
    "volatility_filter_on": True, # 변동성 필터 ON/OFF
    "trailing_stop_on":    True,  # 트레일링 스탑 ON/OFF
    "trailing_atr_mult":   0.5,   # SL 이동 발동 기준 (수익 ≥ ATR × 배수) — 조기 발동
    "early_exit_on":       True,  # 조기 청산 ON/OFF
    "early_exit_conf":     70,    # 조기 청산 발동 최소 신뢰도
    "immediate_reanalyze": [],    # TP/SL 청산 후 즉시 재분석 대기 심볼 목록
    "last_analysis_time":  {},    # 심볼별 마지막 분석 시각 {symbol: timestamp} — 분석 폭풍 방지
    "daily_trade_count":   {},    # 심볼별 일일 거래 횟수 {"ETHUSDT_2026-03-22": 2}
    "oi_prev": {},               # 직전 OI 값 {symbol: oi_float} — OI delta 계산용
    "candle_confirm_on": True,   # 캔들 종가 확인 후 진입 ON/OFF
    "partial_tp_on":    True,    # 부분 청산 (TP 50%) ON/OFF
    "partial_tp_done":  {},      # 이미 부분 청산한 심볼 set {symbol: True}
    "smart_exit_on":    True,    # 스마트 엑싯 (SL/TP 근접 시 지표 재평가) ON/OFF
    "api_fail_count":   0,        # 연속 API 실패 횟수 (3회 연속 → 비상 청산)
    "_last_weekly_report": "",     # 주간 리포트 중복 방지 키
    "_last_daily_report":  "",     # 일일 마감 리포트 중복 방지 키
    "gate_pass_count":  0,         # 게이트 통과 카운트 (세션 내 누적)
    "gate_block_count": 0,         # 게이트 차단 카운트 (세션 내 누적)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 봇 상태 SQLite 복원 (재시작 안전) ──
try:
    _saved_paused = trade_db.get_bot_state("trading_paused")
    if _saved_paused == "true":
        st.session_state.trading_paused = True
    # daily_start_balance 복원
    _today_init = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.daily_start_date != _today_init or st.session_state.daily_start_balance is None:
        _restored = trade_db.get_daily_balance(_today_init)
        if _restored is not None:
            st.session_state.daily_start_balance = _restored
            st.session_state.daily_start_date = _today_init
except Exception:
    pass

# ── 텔레그램 토큰 안전 조회 (세션 빈값 폴백) ──
def _tg():
    """tg_token, tg_chat_id 반환 — 세션 빈값이면 config에서 직접 가져옴"""
    t = st.session_state.get("tg_token", "")
    c = st.session_state.get("tg_chat_id", "")
    if not t and BINANCE_READY:
        t = TELEGRAM_TOKEN
        st.session_state.tg_token = t
    if not c and BINANCE_READY:
        c = TELEGRAM_CHAT_ID
        st.session_state.tg_chat_id = c
    return t, c

# ── 시작 시 포지션 싱크 (고스트 미청산 레코드 정리) ──
if BINANCE_READY:
    try:
        _sync_pos = {p["symbol"] for p in get_positions()}
        _sync_open = trade_db.get_open_trades()
        for _so in _sync_open:
            if _so["symbol"] not in _sync_pos:
                # 바이낸스 체결 내역에서 실제 PnL 조회
                _sync_pnl = 0.0
                _sync_cpx = 0.0
                _sync_method = "추정"
                try:
                    _sync_trades = get_recent_trades(_so["symbol"], limit=30)
                    _sync_entry_time = _so.get("time", "")
                    # 진입 이후 체결 내역 필터
                    _sync_recent = [t for t in _sync_trades if t["time"] >= _sync_entry_time]
                    _sync_rpnl = sum(t["realized_pnl"] for t in _sync_recent if t["realized_pnl"] != 0)
                    if _sync_rpnl != 0:
                        # 실제 realized PnL 사용 (가장 정확)
                        _sync_pnl = round(_sync_rpnl, 4)
                        _sync_cpx = _sync_recent[-1]["price"] if _sync_recent else 0
                        _sync_method = "체결내역"
                    elif _sync_recent:
                        # realized_pnl=0이면 체결가 기반 추정
                        _sync_entry = _so.get("price") or 0
                        _sync_qty = _so.get("qty") or 0
                        _sync_cpx = _sync_recent[-1]["price"]
                        _sync_is_long = "롱" in (_so.get("side") or "") or "LONG" in (_so.get("side") or "").upper()
                        if _sync_entry and _sync_qty:
                            if _sync_is_long:
                                _sync_pnl = round((_sync_cpx - _sync_entry) * _sync_qty, 4)
                            else:
                                _sync_pnl = round((_sync_entry - _sync_cpx) * _sync_qty, 4)
                        _sync_method = "체결가추정"
                    else:
                        # 체결 내역도 없으면 현재가 기반 최후 추정
                        _sync_cpx = float(get_price(_so["symbol"]))
                        _sync_entry = _so.get("price") or 0
                        _sync_qty = _so.get("qty") or 0
                        _sync_is_long = "롱" in (_so.get("side") or "") or "LONG" in (_so.get("side") or "").upper()
                        if _sync_entry and _sync_qty and _sync_cpx:
                            if _sync_is_long:
                                _sync_pnl = round((_sync_cpx - _sync_entry) * _sync_qty, 4)
                            else:
                                _sync_pnl = round((_sync_entry - _sync_cpx) * _sync_qty, 4)
                except Exception:
                    pass
                trade_db.update_trade_field(
                    _so["id"], pnl=_sync_pnl, action="청산(싱크정리)",
                    close_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    close_price=_sync_cpx)
                _log(f"🧹 싱크 정리: {_so['symbol']} id={_so['id']} PnL=${_sync_pnl:+.4f} ({_sync_method})")
                # 잔존 SL/TP 주문 취소
                try:
                    cancel_open_orders(_so["symbol"])
                except Exception:
                    pass
    except Exception:
        pass

# ── 시작 시 당일 잔고 기록 (balance_history) ──
if BINANCE_READY:
    try:
        _init_bal = get_balance()
        _init_today = datetime.now().strftime("%Y-%m-%d")
        trade_db.record_balance(_init_today, _init_bal["total"])
    except Exception:
        pass

AGENTS = {
    "analyst":       {"emoji": "[분석]",  "name": "분석가",      "desc": "3TF 기술적 분석"},
    "news":          {"emoji": "[뉴스]",   "name": "뉴스",        "desc": "시장 심리"},
    "risk":          {"emoji": "[리스크]", "name": "리스크",      "desc": "포지션 관리"},
    "gate":          {"emoji": "[게이트]", "name": "게이트",      "desc": "ADF 거래 필터"},
    "trader":        {"emoji": "[결정]",  "name": "트레이더",     "desc": "최종 결정"},
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


# ── RL 앙상블 모델 로드 (캐시) ────────────────────────────────────
@st.cache_resource
def _load_ensemble():
    """ETH: exp14+seed700+eth_seed800 (만장일치+폴백) / BTC: exp14+seed100+seed200 (다수결) / ALT: 범용모델"""
    try:
        from stable_baselines3 import PPO
        base = Path(__file__).parent / "rl-lab/models"
        # ETH 모델 (v2: exp08→eth_seed800 교체, WF 4/4전승)
        eth_paths = {
            "exp14": base / "exp14/ppo_eth_30m.zip",
            "seed700": base / "seed700/ppo_eth_30m.zip",
            "eth_seed800": base / "eth_seed800/ppo_eth_30m.zip",
        }
        # BTC 모델
        btc_paths = {
            "btc_exp14": base / "btc_exp14/ppo_btc_30m.zip",
            "btc_seed100": base / "btc_seed100/ppo_btc_30m.zip",
            "btc_seed200": base / "btc_seed200/ppo_btc_30m.zip",
        }
        # ALT 범용 모델 (14코인 학습, 롱전용 3-action)
        alt_path = base / "alt_universal_exp01/ppo_alt.zip"

        result = {"eth": None, "btc": None, "alt": None}
        # ETH 로드
        eth_ok = all(p.exists() for p in eth_paths.values())
        if eth_ok:
            result["eth"] = {k: PPO.load(str(p)) for k, p in eth_paths.items()}
            add_log(f"✅ ETH RL 앙상블 로드 (exp14+seed700+eth_seed800, 만장일치)")
        else:
            add_log(f"⚠️ ETH RL 모델 누락")
        # BTC 로드
        btc_ok = all(p.exists() for p in btc_paths.values())
        if btc_ok:
            result["btc"] = {k: PPO.load(str(p)) for k, p in btc_paths.items()}
            add_log(f"✅ BTC RL 앙상블 로드 (exp14+seed100+seed200, 다수결)")
        else:
            add_log(f"⚠️ BTC RL 모델 누락")
        # ALT 로드
        if alt_path.exists():
            result["alt"] = PPO.load(str(alt_path))
            add_log(f"✅ ALT RL 범용모델 로드 (14코인 학습, 롱전용)")
        else:
            add_log(f"⚠️ ALT RL 모델 누락")
        if not eth_ok and not btc_ok and result["alt"] is None:
            return None
        return result
    except Exception as e:
        add_log(f"⚠️ RL 앙상블 로드 실패: {e}")
        return None


def get_rl_signal(symbol: str) -> dict:
    """ETH/BTC 앙상블 + ALT 범용 RL 신호 (ETH/BTC: 0~3, ALT: 0=관망 1=롱 2=청산)"""
    try:
        import numpy as np
        import pandas_ta as ta
        from collections import Counter

        ensemble = _load_ensemble()
        if ensemble is None:
            return {"available": False, "reason": "앙상블 모델 파일 없음"}

        # 알트코인 → 범용 모델 분기
        is_alt = symbol not in ("ETHUSDT", "BTCUSDT")
        if is_alt:
            alt_model = ensemble.get("alt")
            if alt_model is None:
                return {"available": False, "reason": "ALT 범용모델 미로드"}
        else:
            asset_key = "eth" if symbol == "ETHUSDT" else "btc"
            models = ensemble.get(asset_key)
            if models is None:
                return {"available": False, "reason": f"{asset_key.upper()} 모델 미로드"}

        # 최근 30m 데이터 120캔들
        df = get_klines(symbol, "30m", 120)

        # === v5 피처 13개 계산 (env_v5.py _preprocess 동일) ===
        df["price_chg"]  = df["close"].pct_change().fillna(0).clip(-0.1, 0.1)
        df["rsi"]        = ta.rsi(df["close"], length=14).fillna(50)
        df["rsi_norm"]   = df["rsi"] / 100.0
        macd_df          = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd_norm"]  = (macd_df["MACD_12_26_9"] / df["close"]).fillna(0).clip(-0.05, 0.05)
        bb = ta.bbands(df["close"], length=20, std=2)
        col_u = next(c for c in bb.columns if c.startswith("BBU"))
        col_l = next(c for c in bb.columns if c.startswith("BBL"))
        df["bb_pct"]     = ((df["close"] - bb[col_l]) / (bb[col_u] - bb[col_l])).fillna(0.5).clip(0, 1)
        ema20            = ta.ema(df["close"], length=20)
        ema50            = ta.ema(df["close"], length=50)
        df["ema_ratio"]  = (ema20 / ema50 - 1).fillna(0).clip(-0.1, 0.1)
        atr              = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_norm"]   = (atr / df["close"]).fillna(0).clip(0, 0.1)
        df["vol_ratio"]  = (df["volume"] / df["volume"].rolling(20).mean()).fillna(1.0).clip(0, 5) / 5.0
        # ADX
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            df["adx_norm"] = (adx_df["ADX_14"] / 100.0).fillna(0)
        else:
            df["adx_norm"] = 0.0
        # 1시간 변화율 (30m × 2캔들)
        df["price_chg_1h"] = df["close"].pct_change(2).fillna(0).clip(-0.1, 0.1) / 0.1
        # RSI 다이버전스
        price_dir = df["close"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        rsi_dir   = df["rsi"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        df["rsi_diverge"] = (price_dir != rsi_dir).astype(float)
        # Stochastic RSI
        stoch = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None and "STOCHRSIk_14_14_3_3" in stoch.columns:
            df["stoch_rsi"] = (stoch["STOCHRSIk_14_14_3_3"] / 100.0).fillna(0.5).clip(0, 1)
        else:
            df["stoch_rsi"] = 0.5
        # OBV 기울기
        obv = (df["volume"] * np.where(df["close"].diff() > 0, 1, -1)).cumsum()
        obv_std = obv.rolling(20).std().replace(0, np.nan).fillna(1)
        df["obv_slope"] = (obv.diff(5) / obv_std).fillna(0).clip(-3, 3) / 3.0
        # 변동성 레짐
        df["vol_regime"] = atr.rolling(100).rank(pct=True).fillna(0.5)

        df = df.dropna().reset_index(drop=True)
        if len(df) < 20:
            return {"available": False, "reason": "데이터 부족"}

        # === 실거래 포지션 상태 반영 ===
        position_val = 0.0   # 1=롱, -1=숏, 0=없음
        upnl_val     = 0.0   # 미실현 PnL (비율, -1~1 클립)
        hold_val     = 0.0   # 홀딩 캔들 수 / 50
        cooldown_val = 0.0   # 쿨다운 / 10
        try:
            positions = get_positions()
            sym_pos = next((p for p in positions if p["symbol"] == symbol), None)
            if sym_pos:
                position_val = 1.0 if sym_pos["side"] == "LONG" else -1.0
                # 미실현 PnL을 진입가 대비 비율로 계산
                if sym_pos["entry_price"] > 0:
                    cur_price = float(df["close"].iloc[-1])
                    upnl_raw = (cur_price - sym_pos["entry_price"]) / sym_pos["entry_price"] * position_val
                    upnl_val = float(np.clip(upnl_raw, -1, 1))
                # 홀딩 캔들 수: DB에서 미청산 거래의 진입시각 조회
                import sqlite3 as _sql3
                _db = _sql3.connect(str(Path(__file__).parent / "trades.db"))
                _row = _db.execute(
                    f"SELECT time FROM trades WHERE symbol='{symbol}' AND close_time IS NULL ORDER BY id DESC LIMIT 1"
                ).fetchone()
                _db.close()
                if _row:
                    from datetime import datetime as _dt
                    _entry_time = _dt.strptime(_row[0], "%Y-%m-%d %H:%M:%S")
                    from datetime import timezone as _tz
                    _now = _dt.now(tz=_tz.utc).replace(tzinfo=None)
                    _minutes = (_now - _entry_time).total_seconds() / 60
                    hold_val = float(np.clip(_minutes / 30 / 50, 0, 1))  # 캔들 수 / 50
            else:
                # 포지션 없음 → 쿨다운 계산: 마지막 청산 이후 경과
                import sqlite3 as _sql3
                _db = _sql3.connect(str(Path(__file__).parent / "trades.db"))
                _row = _db.execute(
                    f"SELECT close_time FROM trades WHERE symbol='{symbol}' AND close_time IS NOT NULL ORDER BY id DESC LIMIT 1"
                ).fetchone()
                _db.close()
                if _row:
                    from datetime import datetime as _dt
                    _close_time = _dt.strptime(_row[0], "%Y-%m-%d %H:%M:%S")
                    from datetime import timezone as _tz
                    _now = _dt.now(tz=_tz.utc).replace(tzinfo=None)
                    _candles_since = (_now - _close_time).total_seconds() / 60 / 30
                    cooldown_val = float(np.clip(max(0, 8 - _candles_since) / 10, 0, 1))
        except Exception:
            pass  # API/DB 실패 시 기본값(0) 유지

        # obs 구성: 마지막 20캔들, v5 피처 13개 + 상태 4개 = 17차원 × 20 = 340
        feat_cols = [
            "price_chg", "rsi_norm", "macd_norm", "bb_pct", "ema_ratio",
            "atr_norm", "vol_ratio", "adx_norm", "price_chg_1h", "rsi_diverge",
            "stoch_rsi", "obv_slope", "vol_regime",
        ]
        rows = []
        for i in range(len(df) - 20, len(df)):
            row = [float(df[c].iloc[i]) for c in feat_cols]
            row += [position_val, upnl_val, hold_val, cooldown_val]
            rows.append(row)

        obs = np.array(rows, dtype=np.float32).flatten()

        # --- 모델 예측 + 투표 ---
        import torch as _torch
        if is_alt:
            # ALT 범용 모델: 단일 모델 (0=관망, 1=롱, 2=청산) + action 확률
            a, _ = alt_model.predict(obs, deterministic=True)
            action = int(a)
            # action probability 추출 → 확신도
            try:
                _obs_t = _torch.as_tensor(obs).unsqueeze(0).to(alt_model.device)
                _dist = alt_model.policy.get_distribution(_obs_t)
                _probs = _dist.distribution.probs.detach().cpu().numpy()[0]
                action_prob = float(_probs[int(a)])  # 선택된 행동의 확률
                long_prob = float(_probs[1])  # 롱 확률 (항상 추출)
            except Exception:
                action_prob, long_prob = 0.5, 0.0
            # 알트 3-action → 표준 매핑: 2(청산)→3(청산)
            if action == 2:
                action = 3
            method = "single"
            vote_str = f"[alt_universal:{int(a)}(p={action_prob:.0%})]"
            raw_votes = [int(a)]
            # 확률 기반 동적 보너스: 확신 70%+ → +5pt, 50~70% → +3pt, <50% → +1pt
            if action == 1:
                confidence_bonus = 5 if action_prob >= 0.7 else (3 if action_prob >= 0.5 else 1)
            else:
                confidence_bonus = 0
        else:
            model_keys = list(models.keys())
            votes = []
            for k in model_keys:
                a, _ = models[k].predict(obs, deterministic=True)
                votes.append(int(a))

            # 행동 마스킹: 숏(2) → 관망(0)으로 치환 (롱전용 운영)
            votes = [0 if v == 2 else v for v in votes]

            if symbol == "ETHUSDT":
                # ETH: 만장일치 우선, 폴백으로 다수결
                if len(set(votes)) == 1:
                    action = votes[0]
                    method = "unanimous"
                else:
                    vote_count = Counter(votes)
                    maj = vote_count.most_common(1)[0]
                    if maj[1] >= 2:
                        action = maj[0]
                        method = "majority"
                    else:
                        action = 0
                        method = "no_consensus"
            else:
                # BTC: 다수결 (2/3 이상)
                vote_count = Counter(votes)
                maj = vote_count.most_common(1)[0]
                if maj[1] >= 2:
                    action = maj[0]
                    method = "majority"
                else:
                    action = 0
                    method = "no_consensus"

            vote_str = "[" + " ".join(f"{k}:{v}" for k, v in zip(model_keys, votes)) + "]"
            raw_votes = votes
            # ETH 만장일치 시 +5pt, BTC 다수결도 +5pt (승률 100% 검증)
            confidence_bonus = 5 if method in ("unanimous", "majority") else 0

        action_map = {0: ("⚪ 관망", "wait"), 1: ("🟢 롱", "long"),
                      2: ("🔴 숏", "short"), 3: ("🔵 청산", "close")}
        label, atype = action_map.get(int(action), ("?", "wait"))

        result = {"available": True, "action": int(action), "label": label,
                "type": atype, "votes": vote_str, "raw_votes": raw_votes,
                "method": method, "confidence_bonus": confidence_bonus}
        # ALT 확률 정보 추가
        if is_alt:
            result["action_prob"] = round(action_prob, 3)
            result["long_prob"] = round(long_prob, 3)
        return result

    except Exception as e:
        return {"available": False, "reason": str(e)}


# ── 컨플루언스 계산 (15m + 1h 방향 일치 여부) ─────────────────────
def get_confluence(ind_15m: dict, ind_1h: dict, ind_4h: dict = None) -> tuple[str, str]:
    """3개 타임프레임 지표에서 방향 점수를 계산해 컨플루언스 판단"""
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
    s4h = score(ind_4h) if ind_4h else 0

    # 3TF 전부 양수 → 강한 상승, 3TF 전부 음수 → 강한 하락
    scores = [s15, s1h] + ([s4h] if ind_4h else [])
    all_long  = all(s > 0 for s in scores)
    all_short = all(s < 0 for s in scores)

    if all_long:
        label = "🟢 3TF 상승 컨플루언스" if ind_4h else "🟢 상승 컨플루언스"
        return label, "long"
    elif all_short:
        label = "🔴 3TF 하락 컨플루언스" if ind_4h else "🔴 하락 컨플루언스"
        return label, "short"
    else:
        # 2TF 일치 시 부분 컨플루언스
        if s15 > 0 and s1h > 0:
            return "🟡 부분 상승 (4h 불일치)", "long"
        elif s15 < 0 and s1h < 0:
            return "🟡 부분 하락 (4h 불일치)", "short"
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
_MAX_TRADES_PER_DAY = 15  # 심볼당 일일 최대 거래 수 (테스트: 5→15)

def _should_trade(auto_run: bool, confidence: int = 0, symbol: str = "") -> bool:
    """진입 조건: 자동 주문 ON + 신뢰도 65%+ + 일일 거래 제한 + 동적 시간대 필터"""
    if not auto_run:
        return False
    # 콜드 스타트 감지 (10건 미만 → 최소 신뢰도 5%p 하향)
    _cold_start = False
    try:
        import sqlite3 as _sql3
        _conn = _sql3.connect("trades.db")
        _tc = _conn.execute("SELECT COUNT(*) FROM trades WHERE pnl != 0 AND close_price > 0").fetchone()[0]
        _conn.close()
        _cold_start = _tc < 10
    except Exception:
        pass
    # 변동성 레짐 적응: ADX < 15(횡보) → 최소 75% 요구, ADX ≥ 30(강추세) → 기본 65%
    _adx_val = st.session_state.get("_last_adx", 0) or 0
    _min_conf = 50  # 테스트 모드: 65→50
    if _adx_val < 15:
        _min_conf = 50  # 횡보장 (테스트: 75→50)
    elif _adx_val >= 30:
        _min_conf = 45  # 강한 추세 (테스트: 60→45)
    # 콜드 스타트: 최소 신뢰도 5%p 하향 (진입 기회 확보)
    if _cold_start:
        _min_conf -= 5
    if confidence < _min_conf:
        _reason = f"신뢰도 {confidence}% < 최소 {_min_conf}%"
        if _adx_val:
            _reason += f" (ADX={_adx_val:.0f})"
        if _cold_start:
            _reason += " [콜드스타트 -5%p 적용됨]"
        add_log(f"📊 진입 차단: {symbol} — {_reason}")
        return False
    # 일일 거래 수 제한 (심볼당, SQLite 영속화)
    if symbol and _MAX_TRADES_PER_DAY > 0:
        _today = datetime.now().strftime("%Y-%m-%d")
        _today_cnt = trade_db.get_daily_trade_count(symbol, _today)
        if _today_cnt >= _MAX_TRADES_PER_DAY:
            add_log(f"🚫 {symbol} 일일 거래 한도 {_MAX_TRADES_PER_DAY}회 도달 — 진입 차단")
            return False
    # 동적 시간대 필터 (데이터 기반 — 최소 3건 거래된 시간대 중 평균 손실 < -$0.5)
    try:
        _bad_hours = trade_db.get_bad_hours(min_trades=3, max_avg_loss=-0.5)
        _hour_kst = datetime.now().hour
        if _hour_kst in _bad_hours:
            add_log(f"🚫 {_hour_kst}시(KST) 데이터 기반 손실 시간대 — 진입 차단")
            return False
    except Exception:
        pass
    if confidence >= 55:      # 55%+ → 즉시 진입 (테스트: 70→55)
        return True
    # 신뢰도 50~54: 캔들 경계 확인 (15m 캔들 경계 ±5분 이내)
    if st.session_state.get("candle_confirm_on", True):
        now_min = datetime.now().minute
        _boundaries = [0, 15, 30, 45, 60]
        _dist = min(min(abs(now_min - b), 60 - abs(now_min - b)) for b in _boundaries)
        if _dist > 5:
            add_log(f"📊 진입 보류: {symbol} 신뢰도 {confidence}% (60~69구간) — 캔들 경계 외 ({now_min}분, 거리 {_dist}분)")
            return False
    return True


_ANALYSIS_COOLDOWN = 60  # 동일 심볼 분석 최소 간격 (테스트: 3분→1분)

# Claude 자동 호출 시간당 상한 (수동 분석은 제한 없음)
_CLAUDE_RATE_LIMIT = 10  # 시간당 최대 10회
_claude_call_log = []    # 호출 시각 기록

def _check_claude_rate_limit():
    """시간당 호출 상한 확인. True=허용, False=차단"""
    global _claude_call_log
    now = time.time()
    _claude_call_log = [t for t in _claude_call_log if now - t < 3600]
    if len(_claude_call_log) >= _CLAUDE_RATE_LIMIT:
        return False
    _claude_call_log.append(now)
    return True

def run_analysis(symbol: str, execute_trade: bool = False):
    # ── 분석 폭풍 방지: 동일 심볼 3분 쿨다운 ──
    _last_ts = st.session_state.get("last_analysis_time", {}).get(symbol, 0)
    _since = time.time() - _last_ts
    if _since < _ANALYSIS_COOLDOWN:
        _remain = int(_ANALYSIS_COOLDOWN - _since)
        add_log(f"⏳ {symbol} 분석 쿨다운 중 ({_remain}초 남음) — 스킵")
        return
    # 분석 시작 시각 기록
    _lat = st.session_state.get("last_analysis_time", {})
    _lat[symbol] = time.time()
    st.session_state.last_analysis_time = _lat

    # ── 시간대별 자동 OFF: 손실이 심한 시간대에는 분석만 하고 주문 차단 ──
    _bad_hour_block = False
    try:
        _bad_hours = trade_db.get_bad_hours(min_trades=3, max_avg_loss=-0.5)
        _cur_hour = datetime.now().hour
        if _cur_hour in _bad_hours and execute_trade:
            _bad_hour_block = True
            add_log(f"⏰ 현재 {_cur_hour}시는 손실 시간대 → 주문 차단 (분석만 진행)")
    except Exception:
        pass

    add_log(f"🔍 {symbol} 분석 시작 (15m + 1h + 4h)...")

    # 에이전트 상태 초기화
    for k in st.session_state.agent_status:
        st.session_state.agent_status[k] = "대기"

    result = {}

    # ── 데이터 수집: 15m + 1h + 4h + 실시간 뉴스 병렬 ──
    try:
        def _fetch_15m(): return get_klines(symbol, "15m", CANDLE_CNT)
        def _fetch_1h():  return get_klines(symbol, "1h",  50)
        def _fetch_4h():  return get_klines(symbol, "4h",  60)
        def _fetch_news(): return _fetch_live_news(symbol)
        def _fetch_fg():   return _fetch_fear_greed()
        def _fetch_fr_hist(): return get_funding_rate_history(symbol, limit=8)
        def _fetch_oi():   return get_open_interest(symbol)
        def _fetch_whale():
            try:
                from whale_tracker import get_whale_signals
                return get_whale_signals()
            except Exception as e:
                _log(f"고래 추적 스킵: {e}")
                return {"summary": "[고래 추적] 모듈 로드 실패", "signal": "neutral", "raw_count": 0,
                        "deposits": [], "withdrawals": [], "wallet_moves": [], "net_flow": {}}

        with ThreadPoolExecutor(max_workers=4) as ex:  # 동시 API 호출 제한 (IP 밴 방지)
            fut_15m    = ex.submit(_fetch_15m)
            fut_1h     = ex.submit(_fetch_1h)
            fut_4h     = ex.submit(_fetch_4h)
            fut_news   = ex.submit(_fetch_news)
            fut_fg     = ex.submit(_fetch_fg)
            fut_fr     = ex.submit(_fetch_fr_hist)
            fut_oi     = ex.submit(_fetch_oi)
            fut_whale  = ex.submit(_fetch_whale)
            df_15m      = fut_15m.result()
            df_1h       = fut_1h.result()
            df_4h       = fut_4h.result()
            live_news   = fut_news.result()
            fear_greed  = fut_fg.result()
            fr_history  = fut_fr.result()
            oi_data     = fut_oi.result()
            whale_data  = fut_whale.result()

        ind_15m = calc_indicators(df_15m)
        ind_1h  = calc_indicators(df_1h)
        ind_4h  = calc_indicators(df_4h)

        text_15m = format_for_agent(symbol, ind_15m, label="15분봉")
        text_1h  = format_for_agent(symbol, ind_1h,  label="1시간봉")
        text_4h  = format_for_agent(symbol, ind_4h,  label="4시간봉")

        confluence_label, confluence_type = get_confluence(ind_15m, ind_1h, ind_4h)

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
                _auto_mult = round(max(2.5, min(4.0, 1 + 1.5 * _cv)), 1)
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
            _votes = rl.get("votes", "")
            add_log(f"🤖 앙상블 신호: {rl['label']} {_votes}")

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

        # 고래 추적 결과
        result["whale"] = whale_data
        _whale_cnt = whale_data.get("raw_count", 0)
        _whale_sig = whale_data.get("signal", "neutral")
        if _whale_cnt > 0:
            add_log(f"🐋 고래 추적: {_whale_cnt}건 감지, 시그널={_whale_sig}")
        else:
            add_log(f"🐋 고래: 최근 대형 이동 없음")

        add_log(f"📡 데이터 수집 완료 | 컨플루언스: {confluence_label}")

        # ADX를 session_state에 저장 (변동성 레짐 적응용)
        _adx_15m = ind_15m.get("adx", 0) or 0
        st.session_state["_last_adx"] = _adx_15m

        # ── 사전 필터: 컨플루언스 혼조 + ADX<15 + 연속 차단 3회 → 에이전트 호출 스킵 ──
        _rsi_15m = ind_15m.get("rsi", 50) or 50
        _stoch_k = ind_15m.get("stoch_k", 50) or 50
        _streak = st.session_state.gate_block_streak.get(symbol, 0)
        _is_extreme = _rsi_15m < 30 or _rsi_15m > 70 or _stoch_k < 15 or _stoch_k > 85
        # 테스트 모드: 사전 필터 비활성화 (에이전트 항상 호출)
        # if (confluence_type == "mixed" and _adx_15m < 15 and _streak >= 3 and not _is_extreme):
        #     add_log(f"⏭️ 사전 필터: {symbol} 혼조+ADX {_adx_15m:.1f}+연속차단 {_streak}회 → 에이전트 스킵")
        #     result["trader_json"] = {"signal": "wait", "entry": None, "sl": None, "tp": None,
        #                               "confidence": 0, "reason": "사전 필터: 신호 부재", "condition": "", "raw": ""}
        #     return result
        pass

    except Exception as e:
        add_log(f"⚠️ 데이터 수집 실패: {e}")
        return result

    analyst_input = f"""{symbol} 3개 타임프레임 기술적 지표를 분석해주세요.

{text_15m}

{text_1h}

{text_4h}

3TF 컨플루언스: 15분(진입)/1시간(중기)/4시간(대추세) 신호가 일치할수록 신뢰도를 높게 평가해주세요."""

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

    # ── 교훈 로드 (SQLite + 분석 리포트) ──
    _lessons_section = ""
    try:
        _lessons_section = trade_db.get_trade_lessons()
        # 분석 리포트 요약도 추가 (있으면)
        try:
            from trade_report import get_report_summary
            _report_summary = get_report_summary()
            if _report_summary:
                _lessons_section += f"\n{_report_summary}"
        except Exception:
            pass
        if _lessons_section:
            add_log(f"📚 과거 거래 교훈 + 분석 리포트 로드 완료")
    except Exception:
        pass

    # ── 1단계: analyst + news 병렬 실행 (부분 실패 허용) ──
    st.session_state.agent_status["analyst"] = "작업 중"
    st.session_state.agent_status["news"]    = "작업 중"

    def _run_agent_timed(agent_name, agent_input):
        """에이전트 실행 + 응답 시간/품질 추적"""
        _t0 = time.time()
        try:
            _resp = run_agent(agent_name, agent_input)
            _elapsed = round(time.time() - _t0, 1)
            _ok = not (_resp.startswith("⚠️") if _resp else True)
            # 통계 누적
            _stats = st.session_state.setdefault("_agent_stats", {})
            _s = _stats.setdefault(agent_name, {"calls": 0, "ok": 0, "total_time": 0})
            _s["calls"] += 1
            _s["ok"] += (1 if _ok else 0)
            _s["total_time"] += _elapsed
            return _resp
        except Exception as e:
            _elapsed = round(time.time() - _t0, 1)
            _stats = st.session_state.setdefault("_agent_stats", {})
            _s = _stats.setdefault(agent_name, {"calls": 0, "ok": 0, "total_time": 0})
            _s["calls"] += 1
            _s["total_time"] += _elapsed
            return f"[{agent_name} 오류: {e}]"

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_analyst = executor.submit(_run_agent_timed, "analyst", analyst_input)
        fut_news    = executor.submit(_run_agent_timed, "news", news_input)
        result["analyst"] = fut_analyst.result()
        result["news"]    = fut_news.result()

    st.session_state.agent_status["analyst"] = "완료"
    st.session_state.agent_status["news"]    = "완료"
    add_log("📊 분석가 완료")
    add_log("📰 뉴스 에이전트 완료")

    # ── 2단계: 리스크 평가 (부분 실패 허용) ──
    st.session_state.agent_status["risk"] = "작업 중"
    # 리스크에게는 교훈을 참고 정보로만 전달
    _risk_lessons = ""
    if _lessons_section:
        _risk_lessons = f"\n[참고: 과거 거래 통계 — 차단 근거로 사용하지 마세요]\n{_lessons_section}"
    # 규칙 기반 사전 시그널 생성 (risk에게 독립 근거 제공)
    _pre_rule = generate_rule_signal(ind_15m, result.get("indicators_1h", {}),
                                     oi_delta_pct=oi_delta_pct, cvd_trend=ind_15m.get("cvd_trend"))
    _rule_hint = f"[규칙 기반 시그널] {_pre_rule['signal'].upper()} (L:{_pre_rule['long_score']} S:{_pre_rule['short_score']})"

    risk_input = f"""다음 지표를 독립적으로 분석하고 리스크를 평가해주세요.
기술적 분석과 뉴스 에이전트의 결론에 편향되지 말고, 수치 지표를 직접 해석하세요.

심볼: {symbol}
{ind_text}

{text_4h}

{_rule_hint}

[기술적 분석 요약 (참고만)]
{result.get('analyst', '')[:300]}

[시장 심리 요약 (참고만)]
{result.get('news', '')[:200]}

{whale_data.get('summary', '')}
{_risk_lessons}"""
    result["risk"] = _run_agent_timed("risk", risk_input)
    st.session_state.agent_status["risk"] = "완료"
    add_log("⚖️ 리스크 에이전트 완료")

    # ── 3단계: 게이트 (규칙 기반 우선 + LLM 보조) ──
    st.session_state.agent_status["gate"] = "작업 중"
    gate_passed = True  # 폴백: 통과

    # 규칙 기반 사전 게이트: 명확한 경우 LLM 호출 생략 (비용 절감)
    _rule_score = max(_pre_rule.get("long_score", 0), _pre_rule.get("short_score", 0))
    _rule_diff = abs(_pre_rule.get("long_score", 0) - _pre_rule.get("short_score", 0))
    _gate_adx = ind_15m.get("adx", 0) or 0

    if _rule_diff >= 20 and _gate_adx >= 20:
        # 규칙 신호 명확 + 추세 존재 → 자동 통과 (gate 불필요)
        gate_passed = True
        result["gate"] = f"[규칙 게이트] 자동 통과 (점수차={_rule_diff}, ADX={_gate_adx:.0f})"
        result["gate_json"] = {"should_trade": True, "reason": f"규칙 명확 (diff={_rule_diff})", "risk_level": "low"}
        add_log(f"🚦 게이트: 규칙 자동 통과 (점수차={_rule_diff}, ADX={_gate_adx:.0f})")
        st.session_state.gate_block_streak[symbol] = 0
        st.session_state.gate_pass_count = st.session_state.get("gate_pass_count", 0) + 1
    elif confluence_type == "mixed" and _gate_adx < 10 and _rule_diff < 5:
        # 테스트: 혼조 차단 조건 강화 (ADX<15→10, diff<10→5) — 더 극단적인 경우만 차단
        gate_passed = False
        result["gate"] = f"[규칙 게이트] 자동 차단 (혼조+ADX={_gate_adx:.0f}+규칙불확실)"
        result["gate_json"] = {"should_trade": False, "reason": f"혼조+ADX<10+규칙불확실", "risk_level": "high"}
        add_log(f"🚦 게이트: 규칙 자동 차단 (혼조+ADX={_gate_adx:.0f})")
        st.session_state.gate_block_streak[symbol] = st.session_state.gate_block_streak.get(symbol, 0) + 1
        st.session_state.gate_block_count = st.session_state.get("gate_block_count", 0) + 1
    else:
        # 경계 케이스 → LLM gate 호출
        gate_input = f"""다음 리스크 평가를 비판적으로 재검토하고, 지금 거래해야 하는지 판단해주세요.

심볼: {symbol}

{_rule_hint}

[리스크 평가]
{result.get('risk', '')}

[컨플루언스]
{confluence_label}

{whale_data.get('summary', '')}"""
        try:
            gate_raw = _run_agent_timed("gate", gate_input)
            result["gate"] = gate_raw
            gate_json = _parse_gate_json(gate_raw)
            result["gate_json"] = gate_json
            gate_passed = gate_json.get("should_trade", True)
            # high 위험만 실제 차단 (low/medium은 통과)
            if not gate_passed and gate_json.get("risk_level", "high") != "high":
                gate_passed = True
                gate_json["reason"] = f"[위험 낮아 통과] {gate_json.get('reason', '')}"
            add_log(f"🚦 게이트: {'통과' if gate_passed else '차단'} | {gate_json.get('reason', '')} | 위험: {gate_json.get('risk_level', '?')}")
            if gate_passed:
                st.session_state.gate_block_streak[symbol] = 0
                st.session_state.gate_pass_count = st.session_state.get("gate_pass_count", 0) + 1
            else:
                st.session_state.gate_block_streak[symbol] = st.session_state.gate_block_streak.get(symbol, 0) + 1
                st.session_state.gate_block_count = st.session_state.get("gate_block_count", 0) + 1
        except Exception as e:
            result["gate"] = f"[게이트 오류: {e}]"
            result["gate_json"] = {"should_trade": True, "reason": "게이트 오류 — 폴백 통과", "risk_level": "medium"}
            add_log(f"⚠️ 게이트 오류 (폴백 통과): {e}")
    st.session_state.agent_status["gate"] = "완료"

    # gate 성과 추적 (통과 비율 로깅)
    _gp = st.session_state.get("gate_pass_count", 0)
    _gb = st.session_state.get("gate_block_count", 0)
    _gt = _gp + _gb
    if _gt > 0 and _gt % 5 == 0:  # 5회마다 로그
        add_log(f"📊 게이트 통계: 통과 {_gp}/{_gt} ({_gp/_gt*100:.0f}%) | 차단 {_gb}/{_gt}")

    # ── 4단계: 최종 결정 (gate 통과 시만) ──
    if gate_passed:
        st.session_state.agent_status["trader"] = "작업 중"
        # 과매도/과매수 힌트 생성 (숏 편향 완화: 과매수 조건 엄격화)
        _rsi_now = ind_15m.get("rsi", 50) or 50
        _stoch_now = ind_15m.get("stoch_k", 50) or 50
        _rsi_1h_hint = (result.get("indicators_1h") or {}).get("rsi", 50) or 50
        _bias_hint = ""
        _fg = result.get("fear_greed", {})
        _fg_val = _fg.get("value", 50) if isinstance(_fg, dict) else 50
        # 과매도: 15m OR 1h 극단 → 롱 유도
        if _stoch_now < 20 or (_rsi_now < 35 and _stoch_now < 30):
            _bias_hint = f"""
[필수 지시] 극단 과매도 감지 (RSI={_rsi_now}, Stoch RSI K={_stoch_now}, 공포탐욕={_fg_val})
→ 기본 방향을 LONG으로 설정하세요. 숏을 선택하려면 reason에 롱보다 우월한 근거 3가지를 명시해야 합니다.
→ 과매도 + 공포 = 역발상 롱이 통계적으로 유리한 구간입니다."""
        elif _rsi_now < 35 or _stoch_now < 25:
            _bias_hint = f"\n⚠️ RSI={_rsi_now}, Stoch K={_stoch_now} → 과매도 구간. 롱(반등) 우선 검토."
        # 과매수: 15m AND 1h 모두 과매수일 때만 강제 숏 (편향 방지)
        elif (_stoch_now > 85 and _rsi_1h_hint > 65) or (_rsi_now > 75 and _stoch_now > 80):
            _bias_hint = f"""
[참고] 과매수 감지 (15m RSI={_rsi_now}, Stoch K={_stoch_now}, 1h RSI={_rsi_1h_hint:.0f})
→ 숏(조정) 시나리오를 검토하되, 강한 상승 추세(ADX>25+DI+)라면 롱 유지도 유효합니다.
→ 양쪽 시나리오를 reason에 비교해서 제시하세요."""
        elif _rsi_now > 70 or _stoch_now > 80:
            _bias_hint = f"\n⚠️ RSI={_rsi_now}, Stoch K={_stoch_now} → 과매수 접근. 양방향 균형 분석하세요."

        # 숏 편향 방지: 최근 거래에서 숏 비율 75%+ 이면 롱 우대 힌트 추가
        try:
            _recent_trades = trade_db.get_closed_trades(limit=15)
            if len(_recent_trades) >= 5:
                _short_cnt = sum(1 for t in _recent_trades if "숏" in (t.get("side") or "").upper() or "SHORT" in (t.get("side") or "").upper())
                _short_pct = _short_cnt / len(_recent_trades) * 100
                if _short_pct >= 70:
                    _bias_hint += f"\n⚠️ [편향 경고] 최근 거래 숏 비율 {_short_pct:.0f}% — 롱 시나리오를 적극 검토하세요."
        except Exception:
            pass

        trader_input = f"""다음 분석을 종합해 최종 매매 결정을 내려주세요.
게이트가 이미 통과 판정을 내렸으므로, 기술적 신호에 집중해 방향과 수치를 결정하세요.
과도한 관망은 지양하고, 65% 이상 확신이 있으면 반드시 방향(long/short)을 결정하세요.
{_bias_hint}
심볼: {symbol}

[기술적 분석 (3TF)]
{result.get('analyst', '')}

[4시간봉 지표]
{text_4h}

[시장 심리]
{result.get('news', '')}

[리스크 평가]
{result.get('risk', '')}"""
        try:
            trader_raw = _run_agent_timed("trader", trader_input)
            result["trader"] = trader_raw
            _tj = parse_trader_json(trader_raw)
            # 빈/에러 응답 시 1회 재시도 (프롬프트 변형으로 방향 강제)
            if not trader_raw or trader_raw.startswith("⚠️") or (_tj["signal"] == "wait" and not _tj.get("entry")):
                add_log(f"⚠️ trader 응답 불완전 → 프롬프트 변형 재시도")
                _retry_input = trader_input + """

[재시도 지시] 이전 응답이 불완전했습니다. 반드시 long 또는 short 중 하나를 선택하세요.
규칙 시그널 방향을 참고하되, 기술적 분석 기반으로 entry/sl/tp 수치를 포함해 JSON을 완성하세요.
wait은 절대 금지합니다."""
                trader_raw = _run_agent_timed("trader", _retry_input)
                result["trader"] = trader_raw
                _tj = parse_trader_json(trader_raw)
            result["trader_json"] = _tj
        except Exception as e:
            result["trader"] = f"[트레이더 오류: {e}]"
            result["trader_json"] = {"signal": "wait", "entry": None, "sl": None, "tp": None,
                                      "confidence": 0, "reason": str(e), "condition": "", "raw": ""}
        st.session_state.agent_status["trader"] = "완료"
    else:
        # gate 차단 → trader 생략, wait 처리
        result["trader"] = "🚦 게이트 차단 — 거래 불필요 판단"
        result["trader_json"] = {"signal": "wait", "entry": None, "sl": None, "tp": None,
                                  "confidence": 0, "reason": f"게이트 차단: {result.get('gate_json', {}).get('reason', '')}", "condition": "", "raw": ""}
        st.session_state.agent_status["trader"] = "생략"
        # 연속 차단 시 쿨다운 연장 (같은 신호로 반복 재시도 방지)
        _streak = st.session_state.gate_block_streak.get(symbol, 0)
        if _streak >= 2:
            _extra_cd = min(_streak * 30, 120)  # 테스트: 연속 2회부터 30초씩, 최대 2분
            _lat = st.session_state.get("last_analysis_time", {})
            _lat[symbol] = time.time() + _extra_cd  # 쿨다운 연장
            st.session_state.last_analysis_time = _lat
            add_log(f"🚦 게이트 {_streak}회 연속 차단 → {symbol} 쿨다운 +{_extra_cd//60}분")
        add_log("🚦 게이트 차단 → 트레이더 생략 (wait)")

    tj = result["trader_json"]
    add_log(f"🤖 트레이더 결정: {trader_signal_text(tj['signal'])} | 신뢰도: {tj['confidence']}%")

    # ── 규칙 기반 시그널 생성 (통계적 에지) ──
    _rule_sig = generate_rule_signal(ind_15m, result.get("indicators_1h", {}), result.get("indicators_4h", {}),
                                     oi_delta_pct=result.get("oi_delta_pct", 0),
                                     cvd_trend=ind_15m.get("cvd_trend"))
    result["rule_signal"] = _rule_sig
    add_log(f"📊 규칙 시그널: {_rule_sig['signal']} (L:{_rule_sig['long_score']} S:{_rule_sig['short_score']}) | {', '.join(_rule_sig['reasons'][:3])}")

    # ── 신뢰도 점수 계산 ──
    confidence, conf_dir = calc_confidence(result)

    # ── LLM↔규칙 교차 검증 ──
    _rule_dir = _rule_sig["signal"]
    if _rule_dir != "wait" and conf_dir != "wait":
        if _rule_dir == conf_dir:
            confidence = min(100, confidence + 8)  # 규칙+LLM 일치 → 신뢰 상승
            add_log(f"✅ 규칙↔LLM 일치({_rule_dir}) → +8pt → {confidence}%")
        else:
            confidence = max(0, confidence - 10)  # 불일치 → 신뢰 대폭 하락
            add_log(f"⚠️ 규칙({_rule_dir})↔LLM({conf_dir}) 불일치 → -10pt → {confidence}%")
    elif _rule_dir == "wait":
        confidence = max(0, confidence - 5)  # 규칙이 확신 없음 → 소폭 하락
        add_log(f"⏸ 규칙 시그널 불확실 → -5pt → {confidence}%")

    # ── BTC↔ETH 교차 필터 (±5점) ──
    _other_sym = "BTCUSDT" if symbol == "ETHUSDT" else "ETHUSDT"
    _other_res = st.session_state.last_analysis.get(_other_sym, {})
    _other_dir = _other_res.get("trader_json", {}).get("signal", "wait")
    if _other_dir != "wait" and conf_dir != "wait":
        if _other_dir == conf_dir:
            confidence = min(100, confidence + 3)
            add_log(f"🔗 교차 필터: {_other_sym} 동방향({_other_dir}) → +3pt → {confidence}%")
        else:
            confidence = max(0, confidence - 2)
            add_log(f"⚠️ 교차 필터: {_other_sym} 반방향({_other_dir}) → -2pt → {confidence}%")

    # ── RL+LLM+규칙 3자 일치 보너스 (ETH 전용) ──
    _rl_data = result.get("rl", {})
    if _rl_data.get("available") and conf_dir != "wait":
        _rl_type = _rl_data.get("type", "wait")
        _rl_bonus = _rl_data.get("confidence_bonus", 0)  # 만장일치 +5, 다수결 0
        _rl_method = _rl_data.get("method", "")
        if _rl_type == conf_dir and _rule_dir == conf_dir:
            _pts = 15 + _rl_bonus  # 만장일치 시 +20, 다수결 시 +15
            confidence = min(100, confidence + _pts)
            add_log(f"🎯 RL+LLM+규칙 3자 일치({conf_dir}, {_rl_method}) → +{_pts}pt → {confidence}%")
        elif _rl_type == conf_dir and _rule_dir != conf_dir:
            _pts = 8 + _rl_bonus  # 만장일치 시 +13, 다수결 시 +8
            confidence = min(100, confidence + _pts)
            add_log(f"🤖 RL+LLM 일치({conf_dir}, {_rl_method}), 규칙 불일치 → +{_pts}pt → {confidence}%")
        elif _rl_type != "wait" and _rl_type != conf_dir:
            confidence = max(0, confidence - 8)     # RL 반대 → 페널티
            add_log(f"⚠️ RL 반대({_rl_type} vs {conf_dir}) → -8pt → {confidence}%")
        # RL이 관망(wait)인데 LLM이 진입 → 약한 감점
        elif _rl_type == "wait" and conf_dir in ("long", "short"):
            confidence = max(0, confidence - 3)
            add_log(f"🤖 RL 관망, LLM {conf_dir} → -3pt → {confidence}%")

    # 역발상 보정: 멀티TF 극단 과매도/과매수 종합 판단
    _rsi_15 = ind_15m.get("rsi", 50) or 50
    _stoch_15 = ind_15m.get("stoch_k", 50) or 50
    _rsi_1h = (result.get("indicators_1h") or {}).get("rsi", 50) or 50
    _stoch_1h = (result.get("indicators_1h") or {}).get("stoch_k", 50) or 50
    # 과매도: 15m 또는 1h 중 하나라도 극단이면 (1h가 더 신뢰도 높음)
    _is_oversold = (_stoch_15 < 20 or _stoch_1h < 20 or
                    (_rsi_15 < 35 and _stoch_15 < 30) or
                    (_rsi_1h < 35 and _stoch_1h < 30))
    _is_overbought = (_stoch_15 > 80 or _stoch_1h > 80 or
                      (_rsi_15 > 65 and _stoch_15 > 70) or
                      (_rsi_1h > 65 and _stoch_1h > 70))
    if _is_oversold and _is_overbought:
        pass  # 15m과 1h가 반대 극단 → 보정 스킵 (혼조)
    elif _is_oversold:
        if conf_dir == "short":
            confidence = max(0, confidence - 10)
            add_log(f"📉 역발상 보정: 과매도(15m Stoch={_stoch_15},1h Stoch={_stoch_1h})에서 숏 → -10pt → {confidence}%")
        elif conf_dir == "long":
            confidence = min(100, confidence + 5)
            add_log(f"📈 역발상 보정: 과매도에서 롱 → +5pt → {confidence}%")
    elif _is_overbought:
        if conf_dir == "long":
            confidence = max(0, confidence - 10)
            add_log(f"📈 역발상 보정: 과매수(15m Stoch={_stoch_15},1h Stoch={_stoch_1h})에서 롱 → -10pt → {confidence}%")
        elif conf_dir == "short":
            confidence = min(100, confidence + 5)
            add_log(f"📉 역발상 보정: 과매수에서 숏 → +5pt → {confidence}%")

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
        "rl_label":   (result.get("rl", {}).get("label", "") + " " + result.get("rl", {}).get("votes", "")) if result.get("rl", {}).get("available") else "",
        "trader_conf": tj.get("confidence", 0),
        "confidence": confidence,  # 최종 신뢰도 (calc_confidence + 교차 필터)
    }
    st.session_state.analysis_history.insert(0, history_entry)
    if len(st.session_state.analysis_history) > 100:
        st.session_state.analysis_history = st.session_state.analysis_history[:100]
    _save_history(st.session_state.analysis_history)

    # ── 파일 캐시 저장 ──
    _save_analysis(st.session_state.last_analysis)

    # ── 텔레그램: 신호 변경 추적 (알림은 진입/청산 시에만 발송) ──
    _cur_signal  = tj.get("signal", "wait")
    st.session_state.last_tg_signal[symbol] = _cur_signal

    # ── 조기 청산 체크 (반대 신호 → 단계적 대응) ──
    if BINANCE_READY and st.session_state.get("early_exit_on", True) and st.session_state.get("auto_run"):
        try:
            _positions = get_positions()
            _my_pos = next((p for p in _positions if p["symbol"] == symbol), None)
            if _my_pos:
                _pos_side = _my_pos["side"]   # "LONG" or "SHORT"
                _new_sig  = tj.get("signal", "wait")
                _upnl     = _my_pos.get("unrealized_pnl", 0)
                _is_opp   = (_pos_side == "LONG"  and _new_sig == "short") or \
                            (_pos_side == "SHORT" and _new_sig == "long")
                if _is_opp:
                    # 최소 보유 시간 체크 (진입 후 15분 미만이면 조기 청산 금지)
                    _hold_ok = True
                    try:
                        _entry_time = None
                        for _j in reversed(st.session_state.get("trade_journal", [])):
                            if _j.get("symbol") == symbol and _j.get("action") == "진입":
                                _entry_time = _j.get("time")
                                break
                        if _entry_time:
                            from datetime import datetime as _dt
                            _elapsed = (datetime.now() - _dt.strptime(_entry_time, "%Y-%m-%d %H:%M:%S")).total_seconds()
                            if _elapsed < 900:  # 15분 = 900초
                                add_log(f"⏳ {symbol} 반대신호({_new_sig}, {confidence}%) 감지 — 보유 {_elapsed/60:.0f}분 (최소 15분 미충족, 유지)")
                                _hold_ok = False
                    except Exception:
                        pass
                    if _hold_ok:
                        # 단계적 임계값: 손실 중 65%, 수익 중 70% (저신뢰 반대 신호로 인한 불필요한 청산 방지)
                        if _upnl < 0:
                            _exit_thr = 65   # 손실 + 반대 신호 → 65% 이상이어야 손절
                        else:
                            _exit_thr = st.session_state.get("early_exit_conf", 70)  # 수익 중 → 기존 임계값
                        if confidence >= _exit_thr:
                            _pnl_label = "손절" if _upnl < 0 else "이익확정"
                            add_log(f"🚨 조기 청산({_pnl_label}): {symbol} {_pos_side} | 반대신호({_new_sig}, {confidence}%) | PnL ${_upnl:.2f}")
                            _cr = close_position(symbol)
                            if _cr["success"]:
                                _real_trades = get_recent_trades(symbol, limit=5)
                                _real_pnl = sum(t["realized_pnl"] for t in _real_trades if t["realized_pnl"] != 0)
                                _close_px = _real_trades[0]["price"] if _real_trades else ind.get("price")
                                _final_pnl = _real_pnl if _real_pnl != 0 else _upnl
                                _update_journal_pnl(symbol, _final_pnl, _close_px)
                                add_log(f"✅ 조기 청산 완료: {symbol} PnL ${_final_pnl:.2f}")
                                if st.session_state.get("tg_notify", True):
                                    send_error(*_tg(),
                                               f"🚨 {symbol} 조기 청산({_pnl_label}) | 반대신호 {confidence}% | PnL ${_final_pnl:.2f}")
                            else:
                                add_log(f"❌ 조기 청산 실패: {_cr.get('error', '')}", "error")
                        elif confidence >= 40:
                            # 40~임계값: 경고만 (로그 기록, 다음 분석에서 재평가)
                            add_log(f"⚠️ {symbol} 반대 신호 감지({_new_sig}, {confidence}%) | PnL ${_upnl:.2f} — 다음 분석에서 재평가")
        except Exception as _e:
            add_log(f"조기 청산 체크 오류: {_e}", "error")

    # ── 진입 근거 요약 (3줄, 포지션 표시용) ──
    _entry_reason = ""
    try:
        _reasons = []
        _tr = tj.get("reason", "")
        if _tr:
            _reasons.append(f"AI: {_tr[:60]}")
        _rs = result.get("rule_signal", {}).get("reasons", [])
        if _rs:
            _reasons.append(f"규칙: {', '.join(_rs[:3])}")
        _conf_label = f"신뢰도 {confidence}% ({conf_dir})"
        _reasons.append(_conf_label)
        _entry_reason = " | ".join(_reasons[:3])
    except Exception:
        pass

    # ── 자동 주문 실행 ──
    if execute_trade:
        # updater 관리 종목 충돌 방지
        try:
            import json as _json
            with open('/home/hyeok/01.APCC/00.ai-lab/updater_managed.json') as _mf:
                _managed = _json.load(_mf).get("symbols", [])
            if symbol in _managed:
                add_log(f"⏭️ {symbol}: position_updater 관리 중 → 메인 분석 스킵")
                execute_trade = False
        except Exception:
            pass

    if execute_trade:
        # 일일 손실 한도 체크
        try:
            _bal_now  = get_balance()
            _today    = datetime.now().strftime("%Y-%m-%d")
            # daily_start_balance SQLite 영속화 (Streamlit 재시작 안전)
            if st.session_state.daily_start_date != _today or st.session_state.daily_start_balance is None:
                # 날짜 변경 시 일일 손실 한도 paused 자동 해제
                if st.session_state.get("trading_paused") and st.session_state.daily_start_date != _today:
                    st.session_state.trading_paused = False
                    trade_db.save_bot_state("trading_paused", "false")
                    add_log("🔄 날짜 변경 → 일일 손실 한도 리셋, 거래 자동 재개")
                # DB에서 복구 시도
                _restored_bal = trade_db.get_daily_balance(_today)
                if _restored_bal is not None:
                    st.session_state.daily_start_balance = _restored_bal
                    st.session_state.daily_start_date = _today
                else:
                    st.session_state.daily_start_date    = _today
                    st.session_state.daily_start_balance = _bal_now["total"]
                    trade_db.save_daily_balance(_today, _bal_now["total"])
            # 실현 손실 + 미실현 손실 합산 (미실현 PnL이 음수면 위험 반영)
            _unrealized = _bal_now.get("unrealized_pnl", 0)
            _daily_loss = st.session_state.daily_start_balance - _bal_now["total"]
            _daily_loss_total = _daily_loss + max(0, -_unrealized)  # 미실현 손실만 합산
            if _daily_loss_total >= MAX_DAILY_LOSS:
                add_log(f"🚨 일일 손실 한도 초과 (실현${_daily_loss:.2f} + 미실현${max(0,-_unrealized):.2f} = ${_daily_loss_total:.2f}) — 거래 중단")
                st.session_state.trading_paused = True
                trade_db.save_bot_state("trading_paused", "true")
                # 한도 초과 시 기존 포지션 전량 청산
                _em = emergency_close_all()
                if _em["closed"]:
                    add_log(f"🚨 손실 한도 초과 → 전량 청산: {', '.join(_em['closed'])}")
                if _em["errors"]:
                    for _err in _em["errors"]:
                        add_log(f"🚨 청산 오류: {_err}", "error")
                # 일일 손실 한도 텔레그램 알림 비활성화 (불필요한 알림 제거)
                # if st.session_state.get("tg_notify", True):
                #     send_daily_limit_alert(*_tg(),
                #                            _daily_loss_total, MAX_DAILY_LOSS)
                execute_trade = False
        except Exception as e:
            # 잔고 조회 실패 시 안전하게 거래 차단 (한도 우회 방지)
            add_log(f"⚠️ 일일 손실 체크 실패: {e} — 안전을 위해 거래 보류", "error")
            execute_trade = False

    if execute_trade:
        # 일시 중지 체크 (날짜 리셋 후 확인)
        if st.session_state.get("trading_paused"):
            add_log("⏸ 자동 거래 일시 중지 중 — 주문 건너뜀")
            execute_trade = False

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
                if st.session_state.get("tg_notify", True):
                    send_error(*_tg(),
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

    # ── 신뢰도 0% (파싱 실패) 거래 차단 ──
    if execute_trade and tj.get("confidence", 0) == 0:
        add_log(f"🚫 trader 신뢰도 0% (파싱 실패 가능) — 진입 차단")
        execute_trade = False

    if execute_trade:
        if _bad_hour_block:
            execute_trade = False
            add_log(f"⏰ 불리한 시간대 차단: {datetime.now().hour}시 → 주문 실행 안함")
        else:
            execute_trade = _should_trade(True, confidence, symbol)
            if execute_trade:
                add_log(f"✅ 진입 조건 충족 (신뢰도:{confidence}% | 방향:{conf_dir} | 시각:{datetime.now().minute}분)")

    # 포트폴리오 총 익스포저 체크 (메인 심볼만 합산, 알트 제외)
    if execute_trade and BINANCE_READY:
        try:
            _all_pos = get_positions()
            _main_symbols = set(SYMBOLS)  # ETHUSDT, BTCUSDT
            _total_exposure = sum(
                p["size"] * p["entry_price"] / p["leverage"]
                for p in _all_pos if p["symbol"] in _main_symbols
            )
            _max_exposure = MAX_USDT * 4  # 최대 총 익스포저 (기본 $400, 2심볼 동시 진입 가능)
            if _total_exposure >= _max_exposure:
                add_log(f"🚫 포트폴리오 한도: 메인 익스포저 ${_total_exposure:.0f} ≥ ${_max_exposure:.0f} → 진입 차단")
                execute_trade = False
        except Exception:
            pass

    if execute_trade:
        _signal = tj.get("signal", "wait")
        _atr    = ind.get("atr", 0) or 0
        _price  = ind.get("price", 0) or 0

        # SL/TP: trader JSON 우선, 없으면 ATR 폴백 (ADX 동적 R:R)
        # SL 거리 제한: 최소 2.0%, 최대 10% (건당 손실 한도 초과 방지)
        _MIN_SL_PCT = 0.02
        _MAX_SL_PCT = 0.10

        def _enforce_min_sl(side: str, sl_val, price_val):
            """최소 SL 거리 보장 + 방향 오류 수정"""
            if not sl_val or not price_val:
                return sl_val
            _min_dist = price_val * _MIN_SL_PCT
            # 방향 오류 수정 (롱인데 SL > 진입가, 숏인데 SL < 진입가)
            if side == "BUY" and sl_val >= price_val:
                sl_val = round(price_val - _min_dist, 2)
                add_log(f"🚨 SL 방향 오류 보정 (롱): ${sl_val}")
            elif side == "SELL" and sl_val <= price_val:
                sl_val = round(price_val + _min_dist, 2)
                add_log(f"🚨 SL 방향 오류 보정 (숏): ${sl_val}")
            # 최소 거리 보장
            if side == "BUY" and price_val - sl_val < _min_dist:
                sl_val = round(price_val - _min_dist, 2)
                add_log(f"⚠️ SL 최소거리 보정: ${sl_val} (≥{_MIN_SL_PCT*100}%)")
            elif side == "SELL" and sl_val - price_val < _min_dist:
                sl_val = round(price_val + _min_dist, 2)
                add_log(f"⚠️ SL 최소거리 보정: ${sl_val} (≥{_MIN_SL_PCT*100}%)")
            # 최대 거리 제한 (건당 손실 한도 초과 방지)
            _max_dist = price_val * _MAX_SL_PCT
            if side == "BUY" and price_val - sl_val > _max_dist:
                sl_val = round(price_val - _max_dist, 2)
                add_log(f"🚨 SL 최대거리 보정: ${sl_val} (≤{_MAX_SL_PCT*100}%)")
            elif side == "SELL" and sl_val - price_val > _max_dist:
                sl_val = round(price_val + _max_dist, 2)
                add_log(f"🚨 SL 최대거리 보정: ${sl_val} (≤{_MAX_SL_PCT*100}%)")
            return sl_val

        def _resolve_sl_tp(side: str):
            """반환: (soft_sl, tp, hard_sl, sl_mode)
            - sl_mode="soft": 박스권 — 캔들 종가 확인 후 청산, hard_sl은 급락 안전망
            - sl_mode="hard": 추세 — STOP_MARKET 즉시 청산
            """
            _adx = ind.get("adx", 0) or 0
            _is_ranging = _adx < 20

            sl = tj.get("sl")
            tp = tj.get("tp")
            if sl and tp:
                sl = _enforce_min_sl(side, sl, _price)
                # R:R 최소 1:2 검증 — TP가 SL 거리의 2배 미만이면 자동 보정
                _sl_dist = abs(_price - sl)
                _tp_dist = abs(tp - _price)
                if _sl_dist > 0 and _tp_dist < _sl_dist * 2:
                    _old_tp = tp
                    if side == "BUY":
                        tp = round(_price + _sl_dist * 2, 2)
                    else:
                        tp = round(_price - _sl_dist * 2, 2)
                    add_log(f"⚠️ R:R 보정: TP ${_old_tp} → ${tp} (최소 1:2 보장)")
                if _is_ranging:
                    _dist = abs(_price - sl)
                    hard_sl = round(sl - _dist, 2) if side == "BUY" else round(sl + _dist, 2)
                    add_log(f"📐 트레이더 JSON 소프트SL:${sl} / 하드SL:${hard_sl} / TP:${tp} (박스권)")
                    return sl, tp, hard_sl, "soft"
                add_log(f"📐 트레이더 JSON SL:${sl} / TP:${tp}")
                return sl, tp, sl, "hard"

            # ADX 기반 R:R 동적 조정 (SL은 항상 유지, TP만 조정)
            if _adx >= 30:
                _sl_mult = ATR_SL_MULT
                _tp_mult = ATR_TP_MULT * 1.25  # 4.0→5.0
                add_log(f"📈 강한 추세(ADX={_adx:.0f}): TP 확대 ×{_tp_mult:.1f}")
            elif _is_ranging:
                _sl_mult = ATR_SL_MULT
                _tp_mult = ATR_TP_MULT * 0.625  # 4.0→2.5
                add_log(f"↔️ 횡보(ADX={_adx:.0f}): 소프트 SL 모드, TP 축소 ×{_tp_mult:.1f}")
            else:
                _sl_mult = ATR_SL_MULT
                _tp_mult = ATR_TP_MULT
            dist_sl = _atr * _sl_mult
            dist_tp = _atr * _tp_mult
            # 최소 SL 거리 보장
            _min_sl_dist = _price * _MIN_SL_PCT
            if dist_sl < _min_sl_dist:
                add_log(f"⚠️ ATR SL({dist_sl:.2f}) < 최소 2.0%({_min_sl_dist:.2f}) → 보정")
                dist_sl = _min_sl_dist
            if side == "BUY":
                sl = round(_price - dist_sl, 2) if dist_sl else None
                tp = round(_price + dist_tp, 2) if dist_tp else None
            else:
                sl = round(_price + dist_sl, 2) if dist_sl else None
                tp = round(_price - dist_tp, 2) if dist_tp else None

            if _is_ranging and sl:
                # 박스권: 소프트 SL(캔들 종가), 하드 SL(급락 안전망 = 2배 거리)
                hard_dist = dist_sl * 2
                if side == "BUY":
                    hard_sl = round(_price - hard_dist, 2)
                else:
                    hard_sl = round(_price + hard_dist, 2)
                add_log(f"📐 소프트SL:${sl} / 하드SL:${hard_sl} / TP:${tp} (R:R 1:{_tp_mult/_sl_mult:.1f})")
                return sl, tp, hard_sl, "soft"

            add_log(f"📐 ATR폴백 SL:${sl} / TP:${tp} (R:R 1:{_tp_mult/_sl_mult:.1f})")
            return sl, tp, sl, "hard"

        # ── 연속 손실/승률 기반 진입금 조절 ──
        _loss_mult = 1.0  # 기본 배율
        _recent_closed = trade_db.get_closed_trades(limit=10)
        # 연속 손실 체크
        _consec_losses = 0
        for _rc in _recent_closed:
            if (_rc.get("pnl") or 0) <= 0:
                _consec_losses += 1
            else:
                break
        if _consec_losses >= 3:
            _loss_mult = 0.5  # 3연속 손실 → 진입금 50% 축소
            add_log(f"🚨 {_consec_losses}연속 손실 → 진입금 50% 축소")
            if _consec_losses == 3 and st.session_state.get("tg_notify", True):
                from telegram_notifier import send_consec_loss_alert
                try:
                    _alert_bal = get_balance()["available"]
                    _alert_base = _alert_bal * POSITION_PCT / 100
                except Exception:
                    _alert_base = MAX_USDT
                send_consec_loss_alert(*_tg(),
                                       _consec_losses, _alert_base * 0.5, _alert_base)
        # 일중 드로다운 기반 축소 (당일 누적 손실 / 시작 잔고)
        try:
            _today_str = datetime.now().strftime("%Y-%m-%d")
            _today_trades = [t for t in _recent_closed
                             if t.get("close_time", "").startswith(_today_str)]
            _today_pnl = sum(t.get("pnl", 0) for t in _today_trades)
            _start_bal = get_balance(force=True)["total"] - _today_pnl  # 오늘 시작 추정 잔고
            if _start_bal > 0 and _today_pnl < 0:
                _dd_pct = abs(_today_pnl) / _start_bal * 100
                if _dd_pct >= 10:
                    _loss_mult = min(_loss_mult, 0.3)  # 테스트: 10%+ DD만 축소
                    add_log(f"📉 일중 DD {_dd_pct:.1f}% → 진입금 70% 축소")
                elif _dd_pct >= 7:
                    _loss_mult = min(_loss_mult, 0.5)  # 테스트: 7%+ DD
                    add_log(f"📉 일중 DD {_dd_pct:.1f}% → 진입금 50% 축소")
        except Exception:
            pass

        # 승률 기반 조절 (최근 10건)
        if len(_recent_closed) >= 5:
            _recent_wins = sum(1 for t in _recent_closed if (t.get("pnl") or 0) > 0)
            _recent_wr = _recent_wins / len(_recent_closed) * 100
            if _recent_wr < 30:
                _loss_mult = min(_loss_mult, 0.5)
                add_log(f"📉 최근 승률 {_recent_wr:.0f}% → 진입금 50% 축소")
            elif _recent_wr >= 60 and _consec_losses == 0:
                _loss_mult = 1.0  # 승률 양호 + 연승 → 원래 금액 복원

        # 진입 금액: 잔고 비례(POSITION_PCT) × 신뢰도 배율 × 손실 배율 × 스케일업
        # 신뢰도: 65-69% → ×0.5 / 70-79% → ×0.75 / 80%+ → ×0.75(과신뢰 방지)
        try:
            _avail_bal  = get_balance()["available"]
            _base_usdt  = _avail_bal * POSITION_PCT / 100  # 잔고 비례 기본 진입금
            if confidence >= 80:
                _conf_mult = 0.75
                add_log(f"⚠️ 고신뢰도 {confidence}% — 과신뢰 방지 배율 75% 적용")
            elif confidence >= 70:
                _conf_mult = 0.75
            else:
                _conf_mult = 0.5
            # 스케일업: 누적 성과 기반 배율 (SCALE_TABLE)
            _scale_mult = 1.0
            try:
                _all_closed = trade_db.get_closed_trades(limit=200)
                _valid = [t for t in _all_closed if (t.get("pnl") or 0) != 0 and (t.get("close_price") or 0) > 0]
                _total_cnt = len(_valid)
                _total_wins = sum(1 for t in _valid if (t.get("pnl") or 0) > 0)
                _total_wr = (_total_wins / _total_cnt * 100) if _total_cnt > 0 else 0
                for _min_cnt, _min_wr, _mult in SCALE_TABLE:
                    if _total_cnt >= _min_cnt and _total_wr >= _min_wr:
                        _scale_mult = _mult
                        add_log(f"📈 스케일업: {_total_cnt}건 승률{_total_wr:.0f}% → ×{_mult}")
                        break
            except Exception:
                pass
            _dyn_usdt   = round(_base_usdt * _conf_mult * _loss_mult * _scale_mult, 1)
            _order_usdt = min(_dyn_usdt, MAX_USDT)
        except Exception:
            _conf_mult  = 1.0
            _scale_mult = 1.0
            _order_usdt = min(_avail_bal * POSITION_PCT / 100 if '_avail_bal' in dir() else 15, MAX_USDT)
        add_log(f"💼 진입 금액: ${_order_usdt} (잔고${_avail_bal:.0f}×{POSITION_PCT}%=${_base_usdt:.1f} × 신뢰도{_conf_mult:.0%} × 손실{_loss_mult:.0%} × 스케일{_scale_mult:.1f}×)")


        # 건당 최대 손실 사전 검증 (SL 거리 × 수량 × 레버리지)
        def _check_max_single_loss(side: str, sl_val, order_usdt):
            """예상 손실이 잔고의 MAX_SINGLE_LOSS_PCT를 초과하면 차단"""
            from config import MAX_SINGLE_LOSS_PCT
            if not sl_val or not _price:
                add_log("🚫 SL 가격 미설정 → 진입 차단 (안전장치)")
                return False  # SL 없으면 진입 차단 (기존: True → 허용했음)
            try:
                _bal = get_balance()["total"]
                _sl_pct = abs(_price - sl_val) / _price  # SL 거리 비율
                _est_loss = order_usdt * LEVERAGE * _sl_pct  # 예상 손실 금액
                _max_loss = _bal * MAX_SINGLE_LOSS_PCT / 100
                if _est_loss > _max_loss:
                    add_log(f"🚫 건당 손실 제한: 예상 ${_est_loss:.2f} > 한도 ${_max_loss:.2f} (잔고의 {MAX_SINGLE_LOSS_PCT}%) → 진입 차단")
                    return False
                add_log(f"💰 예상 손실: ${_est_loss:.2f} / 한도: ${_max_loss:.2f}")
                # SL 최소 거리 검증: 2% 미만이면 경고 (슬리피지 위험)
                if _sl_pct < 0.005:
                    add_log(f"🚫 SL 거리 {_sl_pct*100:.2f}% < 0.5% → 너무 가까움, 진입 차단")
                    return False
            except Exception:
                pass
            return True

        # R:R 최종 검증 함수
        def _check_rr(side: str, sl_val, tp_val):
            """R:R < 1.5면 차단"""
            if not sl_val or not tp_val or not _price:
                return True
            _sl_d = abs(_price - sl_val)
            _tp_d = abs(tp_val - _price)
            _rr = _tp_d / _sl_d if _sl_d > 0 else 0
            if _rr < 1.5:
                add_log(f"🚫 R:R {_rr:.1f} < 1.5 → 진입 차단 (SL거리={_sl_d:.2f}, TP거리={_tp_d:.2f})")
                return False
            return True

        # -- 메인 지정가 예약 주문 (분석마다 갱신) --
        _has_main_pos = any(p["symbol"] == symbol for p in get_positions())
        if _has_main_pos:
            add_log(f"중복 진입 방지: {symbol} 이미 포지션 존재")
        elif _signal in ("long", "short"):
            _side_str = "BUY" if _signal == "long" else "SELL"
            _sl, _tp, _hard_sl, _sl_mode = _resolve_sl_tp(_side_str)
            _actual_sl = _hard_sl if _sl_mode == "soft" else _sl
            if not _check_max_single_loss(_side_str, _actual_sl, _order_usdt):
                _signal = "wait"
            if _signal != "wait" and not _check_rr(_side_str, _actual_sl, _tp):
                _signal = "wait"
            if _signal != "wait":
                _lim_entry = result.get("trader_json", {}).get("entry")
                if _lim_entry:
                    _lim_entry = float(_lim_entry)
                    if abs(_lim_entry - _price) / _price * 100 > 2.0:
                        _lim_entry = None
                if not _lim_entry:
                    _lim_entry = round(_price - _atr * 0.3, 2) if _signal == "long" else round(_price + _atr * 0.3, 2)
                _prev_main = st.session_state.get("_main_pending_order", {})
                if _prev_main.get("order_id"):
                    try:
                        cancel_open_orders(symbol)
                        _client_tmp = get_client()
                        for _ak in ("sl_algo_id", "tp_algo_id"):
                            _aid = _prev_main.get(_ak)
                            if _aid:
                                try:
                                    _client_tmp._request_futures_api('delete', 'algoOrder', True, data={'algoId': _aid})
                                except Exception:
                                    pass
                    except Exception:
                        pass
                r = place_limit_order(symbol, _side_str, _order_usdt, _lim_entry, LEVERAGE, sl_price=_actual_sl, tp_price=_tp)
                if r["success"]:
                    _sig_label = "LONG" if _signal == "long" else "SHORT"
                    add_log(f"LIMIT ORDER: {symbol} {_sig_label} @ ${_lim_entry} (now ${_price}, SL ${_actual_sl}, TP ${_tp}, conf {confidence}%)")
                    st.session_state["_main_pending_order"] = {
                        "order_id": r.get("order_id"), "sl_algo_id": r.get("sl_algo_id"), "tp_algo_id": r.get("tp_algo_id"),
                        "time": time.time(), "symbol": symbol, "side": _signal,
                        "entry_price": _lim_entry, "sl": _actual_sl, "tp": _tp,
                        "qty": r["qty"], "confidence": confidence, "sl_mode": _sl_mode, "entry_reason": _entry_reason,
                    }
                    if st.session_state.get("tg_notify", True):
                        try:
                            send_message(*_tg(), f"LIMIT {symbol} {_sig_label} @ ${_lim_entry} | SL ${_actual_sl} TP ${_tp} | conf {confidence}%")
                        except Exception:
                            pass
                else:
                    add_log(f"order fail: {r.get('error', '')}", "error")
            else:
                add_log("filter blocked - no order")
        else:
            _prev_main = st.session_state.get("_main_pending_order", {})
            if _prev_main.get("order_id"):
                try:
                    cancel_open_orders(symbol)
                    _client_tmp = get_client()
                    for _ak in ("sl_algo_id", "tp_algo_id"):
                        _aid = _prev_main.get(_ak)
                        if _aid:
                            try:
                                _client_tmp._request_futures_api('delete', 'algoOrder', True, data={'algoId': _aid})
                            except Exception:
                                pass
                    st.session_state["_main_pending_order"] = {}
                except Exception:
                    pass
            else:
                add_log("wait - no order")
        _main_pend = st.session_state.get("_main_pending_order", {})
        if _main_pend.get("order_id") and not _main_pend.get("filled"):
            _has_pos_now = any(p["symbol"] == _main_pend.get("symbol", symbol) for p in get_positions())
            if _has_pos_now:
                _main_pend["filled"] = True
                _mp_sig = _main_pend["side"]
                _mp_entry = _main_pend["entry_price"]
                add_log(f"FILLED {symbol} {_mp_sig} @ ${_mp_entry}")
                _add_journal_entry({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol, "side": "long" if _mp_sig == "long" else "short",
                    "action": "entry", "qty": _main_pend.get("qty", 0), "price": _mp_entry,
                    "sl": _main_pend.get("sl"), "tp": _main_pend.get("tp"),
                    "atr": _atr, "pnl": None, "confidence": _main_pend.get("confidence", 0),
                    "sl_mode": _main_pend.get("sl_mode"), "fill_price": _mp_entry, "slippage": 0,
                    "entry_reason": _main_pend.get("entry_reason", ""),
                })
                if st.session_state.get("tg_notify", True):
                    try:
                        send_message(*_tg(), f"FILLED {symbol} {_mp_sig} @ ${_mp_entry}")
                    except Exception:
                        pass

    # 분석 이력 저장 (롱/숏/관망 모두)
    try:
        _tj = result.get("trader_json", {})
        _rl_data = result.get("rl", {})
        _btc_cache = st.session_state.get("_btc_ind_cache", {})
        _ind15 = result.get("indicators_15m", {})
        _ind1h = result.get("indicators_1h", {})
        _ind4h = result.get("indicators_4h", {})
        _main_p = st.session_state.get("_main_pending_order", {})
        trade_db.save_analysis({
            "symbol": symbol,
            "decision": _signal if _signal in ("long", "short") else "wait",
            "confidence": confidence,
            "entry_price": _main_p.get("entry_price"),
            "sl": _main_p.get("sl"),
            "tp": _main_p.get("tp"),
            "rsi_15m": _ind15.get("rsi"),
            "rsi_1h": _ind1h.get("rsi"),
            "rsi_4h": _ind4h.get("rsi"),
            "adx_15m": _ind15.get("adx"),
            "ema20_15m": _ind15.get("ema20"),
            "ema50_15m": _ind15.get("ema50"),
            "ema20_4h": _ind4h.get("ema20"),
            "ema50_4h": _ind4h.get("ema50"),
            "atr": _ind15.get("atr"),
            "macd_hist": _ind15.get("macd_hist"),
            "btc_price": _btc_cache.get("price"),
            "btc_trend": "up" if (_btc_cache.get("ema20", 0) or 0) > (_btc_cache.get("ema50", 0) or 0) else "down",
            "rl_signal": _rl_data.get("label", ""),
            "llm_reason": _tj.get("reason", "")[:500],
            "order_type": "LIMIT" if _main_p.get("order_id") else None,
            "order_id": str(_main_p.get("order_id", "")),
            "source": "main",
            "filled": _main_p.get("filled", False),
        })
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────

# ── 헤더 + 분석 버튼 ─────────────────────────────────────────────────
# ── 텔레그램 폴링 스레드 시작 (최초 1회) ────────────────────────────
if BINANCE_READY and not st.session_state.tg_polling_started:
    _tg_t, _tg_c = _tg()
    if _tg_t and _tg_c:
        start_polling_thread(_tg_t, _tg_c)
        st.session_state.tg_polling_started = True

# ── 알트 스캔 백그라운드 스레드 (비활성화: position_updater가 LLM 없이 담당) ────
# if BINANCE_READY and st.session_state.get("alt_auto_scan", True) and not st.session_state.get("alt_scan_thread_started"):
#     from alt_scan_thread import start_alt_scan_thread
#     start_alt_scan_thread()
#     st.session_state.alt_scan_thread_started = True

# ── 텔레그램 명령어 처리 ─────────────────────────────────────────────
if BINANCE_READY:
    for _cmd in read_and_clear_commands():
        _c = _cmd.get("cmd")
        if _c == "pause":
            st.session_state.trading_paused = True
            trade_db.save_bot_state("trading_paused", "true")
            add_log("⏸ 텔레그램 명령: 자동 거래 일시 중지")
        elif _c == "resume":
            st.session_state.trading_paused = False
            trade_db.save_bot_state("trading_paused", "false")
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
                send_status(*_tg(),
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
                    _cl_bal = get_balance().get("total", 0)
                    _cl_today = datetime.now().strftime("%Y-%m-%d")
                    _cl_dpnl = (st.session_state.daily_start_balance or _cl_bal) - _cl_bal
                    send_close(*_tg(), _sym, _tg_pnl,
                               balance=_cl_bal, daily_pnl=-_cl_dpnl)
                else:
                    add_log(f"❌ 텔레그램 청산 실패: {_r['error']}", "error")
            except Exception as e:
                add_log(f"⚠️ close 명령 오류: {e}")
        elif _c == "pnl":
            try:
                _today_str = datetime.now().strftime("%Y-%m-%d")
                _today_trades = [t for t in trade_db.get_all_trades(50)
                                 if t.get("time", "").startswith(_today_str)]
                _daily_pnl = sum(t.get("pnl") or 0 for t in _today_trades if t.get("pnl") is not None)
                _stats = trade_db.get_trade_stats()
                _total_pnl = _stats.get("total_pnl", 0)
                from telegram_notifier import send_pnl_report
                send_pnl_report(*_tg(),
                                _today_trades, _total_pnl, _daily_pnl, _stats)
            except Exception as e:
                add_log(f"⚠️ pnl 명령 오류: {e}")
        elif _c == "balance":
            try:
                _b = get_balance()
                _start = st.session_state.daily_start_balance or _b["total"]
                _bh = trade_db.get_balance_history(7)
                from telegram_notifier import send_balance_report
                send_balance_report(*_tg(),
                                    _b, _start, _bh)
            except Exception as e:
                add_log(f"⚠️ balance 명령 오류: {e}")

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
    _btn_c1, _btn_c2 = st.columns(2)
    with _btn_c1:
        _sym_label = st.session_state.get("symbol_select", SYMBOLS[0])
        run_btn = st.button(f"🔍 {_sym_label}\n분석", type="primary", use_container_width=True)
    with _btn_c2:
        alt_manual_btn = st.button("🔍 알트\n스캔", type="secondary", use_container_width=True)

# ── 알트 수동 스캔+분석+자동진입 ──────────────────────────────────────
if BINANCE_READY and alt_manual_btn:
    with st.spinner("알트코인 스캔 + 분석 중..."):
        try:
            _man_syms = get_alt_futures_symbols(ALT_SCAN_LIMIT)
            _man_results = screen_altcoins(_man_syms, top_n=10)
            _man_top = [r for r in _man_results if r["score"] >= ALT_MIN_SCORE][:3]
            if not _man_top:
                st.warning("스크리너 통과 종목 없음 (점수 50+ & 숏 아닌 종목)")
            else:
                _btc_ind_man = st.session_state.get("btc_indicators", {})
                for _mc in _man_top:
                    _msym = _mc["symbol"]
                    add_log(f"🔍 수동 알트 분석: {_msym} (점수 {_mc['score']})")
                    _mres = run_alt_analysis(_mc)
                    st.session_state.alt_analysis[_msym] = _mres
                    _mconf, _mdir = calc_confidence_alt(_mc, _mres, _btc_ind_man)
                    _mres["alt_confidence"] = _mconf
                    _mres.get("trader_json", {})["confidence"] = _mconf
                    st.info(f"**{_msym}** — 신뢰도 {_mconf}%, 방향 {_mdir}")
                    if _mdir != "wait" and _mconf >= ALT_AUTO_CONFIDENCE:
                        # 직접 지정가 주문 (수동 스캔은 _alt_place_order 정의 전이므로)
                        _mtj = _mres.get("trader_json", {})
                        _m_entry = _mtj.get("entry")
                        _m_atr = _mc.get("atr", 0) or 0
                        _m_px = _mc.get("price", 0)
                        if not _m_entry or abs(float(_m_entry) - _m_px) / _m_px * 100 > 3:
                            _m_entry = round(_m_px - _m_atr * 0.3, 6) if _mdir == "long" else round(_m_px + _m_atr * 0.3, 6)
                        _m_sl = _mtj.get("sl") or (round(_m_px - _m_atr * ALT_ATR_SL_MULT, 4) if _mdir == "long" else round(_m_px + _m_atr * ALT_ATR_SL_MULT, 4))
                        _m_tp = _mtj.get("tp") or (round(_m_px + _m_atr * ALT_ATR_TP_MULT, 4) if _mdir == "long" else round(_m_px - _m_atr * ALT_ATR_TP_MULT, 4))
                        _m_side = "BUY" if _mdir == "long" else "SELL"
                        _m_usdt = min(get_balance()["available"] * POSITION_PCT_ALT / 100, MAX_USDT_ALT)
                        _m_r = place_limit_order(_msym, _m_side, _m_usdt, float(_m_entry), ALT_LEVERAGE, sl_price=float(_m_sl), tp_price=float(_m_tp))
                        if _m_r["success"]:
                            st.success(f"✅ {_msym} {'롱' if _mdir=='long' else '숏'} 지정가 주문! @ ${_m_entry} (신뢰도 {_mconf}%)")
                            add_log(f"📋 수동 알트 지정가: {_msym} {'🟢 롱' if _mdir=='long' else '🔴 숏'} @ ${_m_entry}")
                            _pending = st.session_state.setdefault("_alt_pending_orders", {})
                            _pending[_msym] = {"order_id": _m_r.get("order_id"),
                                               "sl_algo_id": _m_r.get("sl_algo_id"), "tp_algo_id": _m_r.get("tp_algo_id"),
                                               "time": time.time(), "expire": 1800,
                                               "entry_price": float(_m_entry), "sl": float(_m_sl), "tp": float(_m_tp),
                                               "side": _mdir, "qty": _m_r["qty"], "confidence": _mconf,
                                               "source": "alt_manual", "reason": f"수동스캔 | {_mtj.get('reason','')[:50]}", "filled": False}
                        else:
                            st.error(f"❌ {_msym} 주문 실패: {_m_r.get('error')}")
                    else:
                        st.warning(f"⏸ {_msym} 보류 (신뢰도 {_mconf}%, 방향 {_mdir}, 기준 {ALT_AUTO_CONFIDENCE}%)")
        except Exception as _me:
            st.error(f"알트 수동 스캔 오류: {_me}")
            add_log(f"알트 수동 스캔 오류: {_me}", "error")

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
        elif status == "생략":
            cls, icon, badge_color = "done", "⏭️", "#9E9E9E"
        else:
            cls, icon, badge_color = "", "⚪", "#aaa"
        st.markdown(f"""
        <div class="agent-card {cls}">
            <b>{info['emoji']} {info['name']}</b>
            <small style="color:#888"> — {info['desc']}</small><br>
            <small style="color:{badge_color}; font-weight:600">{icon} {status}</small>
        </div>
        """, unsafe_allow_html=True)

    # 에이전트 응답 품질 통계
    _agent_stats = st.session_state.get("_agent_stats", {})
    if _agent_stats:
        with st.expander("📊 에이전트 응답 품질", expanded=False):
            _aq_rows = []
            for _aname, _as in _agent_stats.items():
                _avg_t = round(_as["total_time"] / max(_as["calls"], 1), 1)
                _ok_pct = round(_as["ok"] / max(_as["calls"], 1) * 100, 0)
                _aq_rows.append({
                    "에이전트": _aname,
                    "호출 수": _as["calls"],
                    "성공률": f"{_ok_pct:.0f}%",
                    "평균 응답(초)": f"{_avg_t}s",
                    "총 시간(초)": f"{_as['total_time']:.0f}s",
                })
            st.dataframe(pd.DataFrame(_aq_rows), use_container_width=True, hide_index=True)

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
                trade_db.save_bot_state("trading_paused", "false")
                st.rerun()
        else:
            st.warning("⚠️ 자동 주문 활성화됨")

    # 일일 손실 현황
    _today = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.daily_start_date == _today and st.session_state.daily_start_balance:
        try:
            _bal_cur  = get_balance()
            _dloss    = st.session_state.daily_start_balance - _bal_cur["total"]
            _dunreal  = _bal_cur.get("unrealized_pnl", 0)
            _dloss_total = _dloss + max(0, -_dunreal)
            _dloss_pct = (_dloss_total / MAX_DAILY_LOSS * 100) if MAX_DAILY_LOSS else 0
            _d_color   = "#e53935" if _dloss_total > MAX_DAILY_LOSS * 0.7 else "#555"
            st.markdown(
                f"<div style='font-size:12px; color:{_d_color}; margin-top:4px;'>"
                f"📉 오늘 손실: ${_dloss_total:+.2f} / ${MAX_DAILY_LOSS:.0f} "
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
        index=0,  # 기본 비활성 (position_updater가 LLM 없이 거래 담당)
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
        st.caption("📁 config.py 경로: /home/hyeok/01.APCC/00.ai-lab/config.py")


# ── 실시간 메트릭 + 포지션 (5초 갱신 — rate limit 보호) ──────────────
@st.fragment(run_every=5)
def live_panel(sym: str):
    try:
        price     = get_price(sym)
        balance   = get_balance()
        positions = get_positions()
        # API 성공 → 실패 카운터 리셋
        st.session_state.api_fail_count = 0

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
            _total_bal = balance['total']
            _avail_bal = balance['available']
            _margin_used = _total_bal - _avail_bal
            st.markdown(f"""<div class="metric-box">
                <div class="value">${_avail_bal:,.1f} <span style="font-size:0.55em;color:#aaa">/ ${_total_bal:,.1f}</span></div>
                <div class="label">가용 / 총 잔고 (USDT) &nbsp; <span style="font-size:0.85em;color:#FF9800">마진 ${_margin_used:,.1f}</span></div></div>""", unsafe_allow_html=True)
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
            # 진입 근거 조회 (DB extra 필드)
            _pos_reasons = {}
            try:
                import sqlite3 as _sql3
                _rc = _sql3.connect("trades.db")
                for _pr in _rc.execute("SELECT symbol, extra FROM trades WHERE action='진입' AND pnl IS NULL ORDER BY id DESC").fetchall():
                    _sym = _pr[0]
                    if _sym not in _pos_reasons and _pr[1]:
                        _ex = json.loads(_pr[1])
                        if _ex.get("entry_reason"):
                            _pos_reasons[_sym] = _ex["entry_reason"]
                _rc.close()
            except Exception:
                pass
            for pos in positions:
                side_color = "#4CAF50" if pos["side"] == "LONG" else "#e53935"
                pnl        = pos["unrealized_pnl"]
                pnl_color  = "#4CAF50" if pnl >= 0 else "#e53935"
                # PnL% 계산: (pnl / 진입금액) * leverage * 100
                entry_val  = float(pos.get("entry_price", 1)) * float(pos.get("size", 1))
                lev        = float(pos.get("leverage", 1))
                pnl_pct    = (pnl / entry_val * lev * 100) if entry_val else 0
                _reason_html = ""
                _reason = _pos_reasons.get(pos["symbol"], "")
                if _reason:
                    _reason_html = f'<br><small style="color:#666">📝 {_reason}</small>'
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
                    </small>{_reason_html}
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"❌ {pos['symbol']} 청산", key=f"close_{pos['symbol']}"):
                    _pnl_snap = pos.get("unrealized_pnl", 0)
                    _px_snap  = get_price(pos["symbol"])
                    r = close_position(pos["symbol"])
                    if r["success"]:
                        _update_journal_pnl(pos["symbol"], _pnl_snap, _px_snap)
                        add_log(f"✅ {pos['symbol']} 포지션 청산 완료 | PnL ${_pnl_snap:+.4f}")
                        if st.session_state.get("tg_notify", True):
                            _cl_bal = get_balance().get("total", 0)
                            _cl_dpnl = -((st.session_state.daily_start_balance or _cl_bal) - _cl_bal)
                            send_close(st.session_state.tg_token,
                                       st.session_state.tg_chat_id, pos["symbol"], _pnl_snap,
                                       balance=_cl_bal, daily_pnl=_cl_dpnl)
                        st.success("청산 완료!")
                        st.rerun()
                    else:
                        st.error(f"청산 실패: {r['error']}")

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        # API 연속 실패 카운터 증가 → 3회 연속 시 비상 전량 청산
        _fail_cnt = st.session_state.get("api_fail_count", 0) + 1
        st.session_state.api_fail_count = _fail_cnt
        if _fail_cnt >= 3:
            add_log(f"🚨 API {_fail_cnt}회 연속 실패 → 비상 전량 청산 실행!", "error")
            try:
                _em = emergency_close_all()
                if _em["closed"]:
                    add_log(f"🚨 비상 청산 완료: {', '.join(_em['closed'])}")
                if _em["errors"]:
                    for _err in _em["errors"]:
                        add_log(f"🚨 비상 청산 오류: {_err}", "error")
                st.session_state.trading_paused = True
                trade_db.save_bot_state("trading_paused", "true")
                if st.session_state.get("tg_notify", True):
                    send_error(*_tg(),
                               f"🚨 API {_fail_cnt}회 연속 실패 → 비상 전량 청산 실행!")
            except Exception:
                pass
            st.session_state.api_fail_count = 0

live_panel(symbol)
all_btn = False

# ── 거래 일지 + 분석 히스토리 (현재 포지션 바로 아래) ──
st.divider()
tab_pos, tab_j, tab_h = st.tabs([
    "📌 현재 포지션", "📒 거래 일지", "📋 분석 히스토리"
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
                        if st.session_state.get("tg_notify", True):
                            _cl_bal = get_balance().get("total", 0)
                            _cl_dpnl = -((st.session_state.daily_start_balance or _cl_bal) - _cl_bal)
                            send_close(st.session_state.tg_token,
                                       st.session_state.tg_chat_id, pos["symbol"], _pnl_snap,
                                       balance=_cl_bal, daily_pnl=_cl_dpnl)
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
                f"| {rsi_val} | RL: {rl_val}</small>"
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
                f"<b>🤖 앙상블 신호 (ETH)</b><br>{rl['label']} "
                f"<span style='font-size:11px;color:#666'>{rl.get('votes','')}</span></div>",
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
                # 쿨다운 체크: 최근 3분 이내 분석했으면 스킵 (분석 폭풍 방지)
                _last_t = st.session_state.get("last_analysis_time", {}).get(_rsym, 0)
                if time.time() - _last_t < _ANALYSIS_COOLDOWN:
                    add_log(f"⏳ {_rsym} 재분석 스킵 (쿨다운 {int(_ANALYSIS_COOLDOWN - (time.time() - _last_t))}초)")
                    continue
                add_log(f"⚡ {_rsym} TP/SL 청산 감지 → 즉시 재분석")
                with st.spinner(f"⚡ {_rsym} 즉시 재분석 중..."):
                    run_analysis(_rsym, execute_trade=True)
        st.session_state.last_auto_time = time.time()
        # st.rerun() 제거: 연쇄 rerun 방지 — 다음 자동 분석 주기에 자연 갱신

# ── 자동 분석 처리 (메인 루프 체크) — 멀티심볼 병렬 실행 ────────────
if auto_interval != "비활성":
    interval_map = {"10분": 600, "30분": 1800, "1시간": 3600}
    secs    = interval_map[auto_interval]
    elapsed = time.time() - st.session_state.last_auto_time
    if elapsed >= secs:
        # 캔들 경계 동기화: 15분 캔들 경계(0,15,30,45분) 후 1분 이내 대기 (최대 2분)
        _now_min = datetime.now().minute
        _candle_dist = min(abs(_now_min - b) for b in [0, 15, 30, 45, 60])
        if _candle_dist > 1 and elapsed < secs + 120:
            pass  # 캔들 경계까지 대기
        else:
            # SL/TP 자동 청산 감지 + 소프트SL + 트레일링 스탑 + 부분 청산 + 보유시간 체크 (분석 전 먼저)
            if BINANCE_READY:
                _check_sl_tp_closed()
                _check_soft_sl()
                _smart_exit_check()
                _update_trailing_stop()
                _check_partial_tp()
                _check_max_hold_time()
            # 포지션 보유 심볼은 3사이클(~30분)에 1번만 분석 (비용 절감)
            _active_syms = set()
            if BINANCE_READY:
                try:
                    _active_syms = {p["symbol"] for p in get_positions()}
                except Exception:
                    pass
            _skip_counter = st.session_state.get("pos_skip_counter", 0)
            st.session_state.pos_skip_counter = _skip_counter + 1
            for _sym in SYMBOLS:
                if _sym in _active_syms and _skip_counter % 3 != 0:
                    add_log(f"⏭️ {_sym} 포지션 보유 중 → 분석 스킵 ({_skip_counter % 3}/3)")
                    continue
                # Claude 시간당 상한 체크
                if not _check_claude_rate_limit():
                    add_log(f"⚠️ Claude 시간당 상한 도달 ({_CLAUDE_RATE_LIMIT}회/시간) → 자동 분석 스킵")
                    break
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
    # ── 정시 텔레그램 브리핑 (매 정시 0~1분 사이에 1회) ──
    _now_h = datetime.now().hour
    _now_m = datetime.now().minute
    if (BINANCE_READY and st.session_state.get("tg_notify", True)
            and _now_m < 2 and st.session_state.get("last_hourly_briefing", -1) != _now_h):
        st.session_state.last_hourly_briefing = _now_h
        try:
            _bal = get_balance()
            _pos = get_positions()
            # 오늘 청산 거래
            _today_str = datetime.now().strftime("%Y-%m-%d")
            _closed = []
            _daily_pnl = 0.0
            _trades_today = 0
            try:
                import trade_db as _tdb
                import sqlite3 as _sq
                _conn = _sq.connect(str(_tdb._DB_PATH), timeout=10)
                _conn.row_factory = _sq.Row
                _cur = _conn.cursor()
                _cur.execute("SELECT * FROM trades WHERE close_time IS NOT NULL AND close_time >= ? ORDER BY close_time DESC", (_today_str,))
                _closed = [dict(r) for r in _cur.fetchall()]
                _daily_pnl = sum(c.get("pnl") or 0 for c in _closed)
                _cur.execute("SELECT COUNT(*) FROM trades WHERE time >= ? AND action='진입'", (_today_str,))
                _trades_today = _cur.fetchone()[0]
                _conn.close()
            except Exception:
                pass
            # 게이트 통과/차단 카운트 (세션 내 누적)
            _gp = st.session_state.get("gate_pass_count", 0)
            _gb = st.session_state.get("gate_block_count", 0)
            send_hourly_briefing(
                *_tg(),
                _bal, _pos, _closed, _daily_pnl, _gp, _gb, _trades_today
            )
            _log(f"📋 정시 텔레그램 브리핑 발송 ({_now_h}:00)")
        except Exception as _e:
            _log(f"정시 브리핑 오류: {_e}", "error")
        # ── 정시 잔고 기록 (balance_history 테이블) ──
        try:
            _rec_bal = get_balance()
            _rec_today = datetime.now().strftime("%Y-%m-%d")
            _rec_cnt = trade_db.get_daily_trade_count("ETHUSDT", _rec_today) + trade_db.get_daily_trade_count("BTCUSDT", _rec_today)
            trade_db.record_balance(_rec_today, _rec_bal["total"], _rec_cnt)
        except Exception:
            pass
        # ── 주간 리포트 (일요일 0시) ──
        if datetime.now().weekday() == 6 and _now_h == 0:
            _wr_key = f"weekly_report_{datetime.now().strftime('%Y-%W')}"
            if st.session_state.get("_last_weekly_report") != _wr_key:
                st.session_state._last_weekly_report = _wr_key
                try:
                    _wr_stats = trade_db.get_trade_stats()
                    _wr_hist = trade_db.get_balance_history(7)
                    _wr_start = _wr_hist[-1].get("open_bal", 0) if _wr_hist else 0
                    from telegram_notifier import send_weekly_report
                    send_weekly_report(*_tg(),
                                       _wr_stats, _wr_hist, _wr_start)
                    _log("📊 주간 리포트 발송 완료")
                except Exception as _wr_e:
                    _log(f"주간 리포트 오류: {_wr_e}", "error")
        # ── 일일 마감 리포트 (23시) ──
        if _now_h == 23:
            _dr_key = f"daily_report_{datetime.now().strftime('%Y-%m-%d')}"
            if st.session_state.get("_last_daily_report") != _dr_key:
                st.session_state._last_daily_report = _dr_key
                try:
                    _dr_bal = get_balance().get("total", 0)
                    _dr_start = st.session_state.daily_start_balance or _dr_bal
                    _dr_today = datetime.now().strftime("%Y-%m-%d")
                    _dr_trades = [t for t in trade_db.get_all_trades(50)
                                  if t.get("time", "").startswith(_dr_today)]
                    _dr_pnl = sum(t.get("pnl") or 0 for t in _dr_trades if t.get("pnl") is not None)
                    _dr_stats = trade_db.get_trade_stats()
                    from telegram_notifier import send_daily_close_report
                    send_daily_close_report(*_tg(),
                                            _dr_bal, _dr_start, _dr_trades, _dr_pnl, _dr_stats)
                    # 거래 분석 리포트 자동 생성 (.txt)
                    try:
                        from trade_report import generate_report
                        generate_report()
                        _log("📊 일일 거래 분석 리포트 생성 완료 (reports/)")
                    except Exception as _rpt_e:
                        _log(f"거래 분석 리포트 생성 오류: {_rpt_e}", "error")
                    _log("🌙 일일 마감 리포트 발송 완료")
                except Exception as _dr_e:
                    _log(f"일일 마감 리포트 오류: {_dr_e}", "error")

    # ── 30초마다 SL/TP 청산 감지 + 소프트SL + 트레일링 + 부분 청산 체크 + 즉시 재분석 트리거 ──
    if BINANCE_READY:
        try:
            _check_sl_tp_closed()
            _check_soft_sl()
            _smart_exit_check()
            _update_trailing_stop()
            _check_partial_tp()
            _check_max_hold_time()
            # API 정상 → 실패 카운터 초기화
            st.session_state.api_fail_count = 0
        except (ConnectionError, Exception) as e:
            st.session_state.api_fail_count = st.session_state.get("api_fail_count", 0) + 1
            _fail_cnt = st.session_state.api_fail_count
            _log(f"⚠️ API 오류 ({_fail_cnt}/3): {e}", "error")
            add_log(f"⚠️ API 연속 실패 {_fail_cnt}/3: {str(e)[:80]}", "error")
            if _fail_cnt >= 3:
                # 3회 연속 실패 → 비상 전량 청산
                _log("🚨 API 3회 연속 실패 → 비상 전량 청산 실행", "error")
                add_log("🚨 API 3회 연속 실패 → 비상 전량 청산 실행!", "error")
                _em = emergency_close_all()
                if _em["closed"]:
                    add_log(f"✅ 비상 청산 완료: {', '.join(_em['closed'])}")
                if _em["errors"]:
                    for _err in _em["errors"]:
                        add_log(f"🚨 비상 청산 오류: {_err}", "error")
                st.session_state.trading_paused = True
                trade_db.save_bot_state("trading_paused", "true")
                if st.session_state.get("tg_notify", True):
                    send_error(*_tg(),
                               f"🚨 API 3회 연속 실패 → 비상 전량 청산 실행\n청산: {_em['closed']}\n오류: {_em['errors']}")
                st.session_state.api_fail_count = 0
        if st.session_state.get("immediate_reanalyze"):
            # 쿨다운 내 재분석 큐는 자연 소화 — 불필요한 rerun 방지
            _rq = st.session_state.get("immediate_reanalyze", [])
            _has_ready = any(
                time.time() - st.session_state.get("last_analysis_time", {}).get(s, 0) >= _ANALYSIS_COOLDOWN
                for s in _rq
            )
            if _has_ready:
                st.rerun(scope="app")
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
    # 스캔 주기 동적화: BTC ATR 높으면 90초, 낮으면 300초
    _btc_cache_tmp = st.session_state.get("_btc_ind_cache", {})
    _btc_atr_tmp = _btc_cache_tmp.get("atr", 0) or 0
    _btc_px_tmp = _btc_cache_tmp.get("price", 1) or 1
    _btc_atr_pct = (_btc_atr_tmp / _btc_px_tmp) * 100
    # 스캔 주기 180초 (API 부하 1/3 절감, 60→180)
    _ALT_SCREEN_INTERVAL = 180
    _ALT_ANNOUNCE_INTERVAL = 180  # 공지는 고정 3분

    # 섹터 매핑 — 같은 섹터 동시 진입 제한
    _ALT_SECTORS = {
        # AI
        "FETUSDT": "ai", "RENDERUSDT": "ai", "AGIXUSDT": "ai", "OCEANUSDT": "ai",
        "WLDUSDT": "ai", "ARKMUSDT": "ai", "TAORUSDT": "ai", "NEARUSDT": "ai",
        # L1
        "SOLUSDT": "l1", "AVAXUSDT": "l1", "DOTUSDT": "l1", "ADAUSDT": "l1",
        "ATOMUSDT": "l1", "APTUSDT": "l1", "SUIUSDT": "l1", "SEIUSDT": "l1",
        "INJUSDT": "l1", "TIAUSDT": "l1",
        # L2
        "ARBUSDT": "l2", "OPUSDT": "l2", "MATICUSDT": "l2", "STRKUSDT": "l2",
        "MANTAUSDT": "l2", "BLASTUSDT": "l2",
        # 밈
        "DOGEUSDT": "meme", "SHIBUSDT": "meme", "PEPEUSDT": "meme", "FLOKIUSDT": "meme",
        "BONKUSDT": "meme", "WIFUSDT": "meme", "MEMEUSDT": "meme",
        # DeFi
        "UNIUSDT": "defi", "AAVEUSDT": "defi", "MKRUSDT": "defi", "CRVUSDT": "defi",
        "LDOUSDT": "defi", "PENDLEUSDT": "defi",
        # 게임
        "AXSUSDT": "game", "SANDUSDT": "game", "MANAUSDT": "game", "IMXUSDT": "game",
        "GABORUSDT": "game", "PIXELUSDT": "game",
    }

    # 소스별 SL/TP ATR 배수 (급등은 ATR 부풀려짐 → 타이트하게)
    _ALT_SL_TP_BY_SOURCE = {
        "screener":  (ALT_ATR_SL_MULT, ALT_ATR_TP_MULT),  # x2.0 / x4.0
        "binance":   (1.5, 3.0),
        "upbit":     (1.5, 3.0),
        "okx":       (1.5, 3.0),
        "coinbase":  (1.5, 3.0),
        "surge":     (1.0, 2.0),  # 급등은 변동성 이미 확대 → 타이트
    }

    def _alt_place_order(sym, sig, tj, cand, source="screener"):
        """알트 자동 주문 실행 헬퍼. source: screener/binance/upbit/okx/coinbase/surge"""
        try:
            # updater 관리 종목 충돌 방지
            try:
                import json as _json
                with open('/home/hyeok/01.APCC/00.ai-lab/updater_managed.json') as _mf:
                    _managed = _json.load(_mf).get("symbols", [])
                if sym in _managed:
                    add_log(f"⏭️ {sym}: position_updater 관리 중 → app.py 스킵")
                    return
            except Exception:
                pass

            # 시간대 필터: 해제 (데이터 축적 우선, 50건+ 후 재검토)
            _utc_hour = datetime.now(tz=__import__('datetime').timezone.utc).hour
            _kst_hour = (_utc_hour + 9) % 24
            _low_liq_hours = False  # 저유동성 축소도 해제

            # 연속 손실 종목 강화: 최근 3건 연속 손실 → 6시간 쿨다운
            try:
                _sym_recent = trade_db.get_recent_trades_by_symbol(sym, limit=3)
                if len(_sym_recent) >= 3 and all(t.get("pnl", 0) < 0 for t in _sym_recent):
                    _last_loss_time = _sym_recent[0].get("close_time", "")
                    if _last_loss_time:
                        from datetime import datetime as _dt
                        _lt = _dt.strptime(_last_loss_time, "%Y-%m-%d %H:%M:%S")
                        _hours_since = (datetime.now() - _lt).total_seconds() / 3600
                        if _hours_since < 3:  # 3연속 손실 쿨다운 3시간 (반복 손실 방지)
                            add_log(f"🚫 알트 {sym} 3연속 손실 쿨다운 ({(3-_hours_since)*60:.0f}분 남음)")
                            return
            except Exception:
                pass

            # #10 총 알트 익스포저 한도: 잔고의 50% 초과 시 신규 진입 차단
            try:
                _bal_total = get_balance().get("total", 0)
                _alt_pos_list = [p for p in get_positions() if p["symbol"] not in SYMBOLS]
                _alt_exposure = sum(abs(float(p.get("positionAmt", 0))) * float(p.get("entryPrice", 0)) for p in _alt_pos_list)
                if _bal_total > 0 and _alt_exposure >= _bal_total * 0.5:
                    add_log(f"🚫 알트 {sym} 총 익스포저 한도: ${_alt_exposure:.0f} >= 잔고 50% (${_bal_total*0.5:.0f})")
                    return
            except Exception:
                pass

            # #5 섹터 상관관계: 같은 섹터 2개 이상 동시 보유 시 차단
            _sym_sector = _ALT_SECTORS.get(sym)
            if _sym_sector:
                try:
                    _cur_pos_sec = [p for p in get_positions() if p["symbol"] not in SYMBOLS]
                    _same_sector = sum(1 for p in _cur_pos_sec if _ALT_SECTORS.get(p["symbol"]) == _sym_sector)
                    if _same_sector >= 2:
                        add_log(f"⚠️ 알트 {sym} 섹터 분산: {_sym_sector} 이미 {_same_sector}개")
                        return
                except Exception:
                    pass

            # #9 슬리피지 기반 종목 필터: 평균 슬리피지가 큰 종목 회피
            try:
                _avg_slip = trade_db.get_symbol_avg_slippage(sym, recent_n=5)
                _sym_price = cand.get("price", 1)
                if _sym_price > 0 and _avg_slip > 0:
                    _slip_pct = _avg_slip / _sym_price * 100
                    if _slip_pct > 0.3:  # 0.3% 이상 슬리피지
                        add_log(f"⚠️ 알트 {sym} 고슬리피지: {_slip_pct:.2f}% — 진입 스킵")
                        return
            except Exception:
                pass

            # #8 소스별 성과: 성과 나쁜 소스에서 온 시그널 페널티
            try:
                _src_perf = trade_db.get_source_performance(recent_n=30)
                _src_key = f"alt_{source}"
                if _src_key in _src_perf and _src_perf[_src_key]["trades"] >= 3:
                    if _src_perf[_src_key]["win_rate"] < 25:
                        add_log(f"⚠️ 알트 {sym} 소스 {source} 승률 {_src_perf[_src_key]['win_rate']}% — 진입 스킵")
                        return
            except Exception:
                pass

            # ── A+C 전략: 추세 추종 + BTC 동조 ──────────────────────────

            # [A] 4H EMA 추세 필터 (양방향)
            # 롱: EMA20 > EMA50 필수 / 숏: EMA20 < EMA50 필수
            try:
                _klines_4h = get_klines(sym, "4h", 60)
                _ind_4h = calc_indicators(_klines_4h)
                _ema20_4h = _ind_4h.get("ema20", 0) or 0
                _ema50_4h = _ind_4h.get("ema50", 0) or 0
                if _ema20_4h and _ema50_4h:
                    if sig == "long" and _ema20_4h < _ema50_4h:
                        add_log(f"🚫 알트 {sym} 롱 차단: 4H 하락추세 (EMA20 {_ema20_4h:.4f} < EMA50 {_ema50_4h:.4f})")
                        return
                    if sig == "short" and _ema20_4h > _ema50_4h:
                        add_log(f"🚫 알트 {sym} 숏 차단: 4H 상승추세 (EMA20 {_ema20_4h:.4f} > EMA50 {_ema50_4h:.4f})")
                        return
            except Exception:
                pass

            # [C] BTC 1H 동조 필터 — BTC 1H 방향과 같은 방향만 허용
            _btc_c = st.session_state.get("_btc_ind_cache", {})
            _btc_chg = _btc_c.get("change_pct", 0) or 0
            try:
                _btc_kl_1h = get_klines("BTCUSDT", "1h", 60)
                _btc_ind_1h = calc_indicators(_btc_kl_1h)
                _btc_ema20_1h = _btc_ind_1h.get("ema20", 0) or 0
                _btc_ema50_1h = _btc_ind_1h.get("ema50", 0) or 0
                _btc_rsi_1h = _btc_ind_1h.get("rsi", 50) or 50
                _btc_1h_bullish = _btc_ema20_1h > _btc_ema50_1h
                _btc_1h_bearish = _btc_ema20_1h < _btc_ema50_1h

                if sig == "long" and _btc_1h_bearish and _btc_chg < -1.0:
                    add_log(f"🚫 알트 {sym} 롱 차단: BTC 1H 하락 (EMA20<50, 변동 {_btc_chg:+.1f}%)")
                    return
                if sig == "short" and _btc_1h_bullish and _btc_chg > 1.0:
                    add_log(f"🚫 알트 {sym} 숏 차단: BTC 1H 상승 (EMA20>50, 변동 {_btc_chg:+.1f}%)")
                    return
            except Exception:
                pass

            # 거래량 확인
            _vol_ratio = cand.get("vol_ratio", 1) or 1
            _vol_min = 1.5
            if _vol_ratio < _vol_min:
                add_log(f"🚫 알트 {sym} 차단: 거래량 부족 (vol_ratio {_vol_ratio:.1f}x < {_vol_min}x)")
                return

            # 종목별 승률 피드백: 최근 5건 승률 20% 미만 → 1일 블랙리스트
            try:
                _sym_wr = trade_db.get_symbol_win_rate(sym, recent_n=5)
                if _sym_wr["trades"] >= 5 and _sym_wr["win_rate"] < 20:
                    add_log(f"🚫 알트 {sym} 블랙리스트: 최근 5건 승률 {_sym_wr['win_rate']}%")
                    return
            except Exception:
                pass

            # 동일 종목 재진입 쿨다운 (승리 30분, 패배 4시간 차등)
            _alt_cooldowns = st.session_state.setdefault("_alt_cooldowns", {})
            _cd_info = _alt_cooldowns.get(sym, {})
            _cd_start = _cd_info.get("time", 0) if isinstance(_cd_info, dict) else _cd_info
            _cd_duration = _cd_info.get("duration", 7200) if isinstance(_cd_info, dict) else 7200
            if time.time() - _cd_start < _cd_duration:
                _remain = int((_cd_duration - (time.time() - _cd_start)) / 60)
                add_log(f"⏳ 알트 {sym} 재진입 쿨다운 ({_remain}분 남음)")
                return

            _cur_pos = get_positions()
            _alt_pos = [p for p in _cur_pos if p["symbol"] not in SYMBOLS]
            _alt_cnt = len(_alt_pos)
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
            # 1H 캔들 종가 확인 — 가짜 돌파 필터 (직전 완성 1H 캔들 종가 > 시가여야 롱 허용)
            if sig == "long":
                try:
                    _kl_1h = get_klines(sym, "1h", 3)
                    _prev_close = float(_kl_1h["close"].iloc[-2])  # 직전 완성 캔들
                    _prev_open  = float(_kl_1h["open"].iloc[-2])
                    if _prev_close < _prev_open:
                        add_log(f"🚫 알트 {sym} 롱 차단: 1H 캔들 음봉 (종가 {_prev_close} < 시가 {_prev_open})")
                        return
                except Exception:
                    pass

            # 진입 전 잔존 주문 정리 (SL/TP 잔존으로 인한 -4067 에러 방지)
            _old_alt_orders = get_open_orders(sym)
            if _old_alt_orders:
                cancel_open_orders(sym)
                add_log(f"🧹 알트 {sym} 잔존 주문 {len(_old_alt_orders)}건 정리")

            _atr = cand.get("atr", 0) or 0
            _px  = cand.get("price", 0)
            _side = "BUY" if sig == "long" else "SELL"
            # 소스별 SL/TP ATR 배수 차등화
            _sl_mult, _tp_mult = _ALT_SL_TP_BY_SOURCE.get(source, (ALT_ATR_SL_MULT, ALT_ATR_TP_MULT))
            # #4 동적 익절: ATR% 기반 TP 조정 (고변동=넓게, 저변동=좁게)
            _atr_pct = (_atr / _px * 100) if _px > 0 else 1.0
            if _atr_pct >= 1.5:    # 고변동 종목
                _tp_mult = max(_tp_mult, 5.0)  # TP 넓게
            elif _atr_pct <= 0.5:  # 저변동 종목
                _tp_mult = min(_tp_mult, 3.0)  # TP 좁게
            _sl = tj.get("sl") or (round(_px - _atr * _sl_mult, 4) if sig == "long" else round(_px + _atr * _sl_mult, 4))
            _tp = tj.get("tp") or (round(_px + _atr * _tp_mult, 4) if sig == "long" else round(_px - _atr * _tp_mult, 4))
            # SL 최소 2% / 최대 5% 보장 (노이즈 탈락 방지 + 대형 손실 방지)
            if _px and _sl:
                _sl_dist_pct = abs(_sl - _px) / _px * 100
                if _sl_dist_pct < 2.0:
                    _old_sl = _sl
                    if sig == "long":
                        _sl = round(_px * 0.98, 4)
                    else:
                        _sl = round(_px * 1.02, 4)
                    add_log(f"⚠️ 알트 {sym} SL 최소 보장: {_sl_dist_pct:.2f}%→2.0% (${_old_sl}→${_sl})")
                    # TP도 R:R 1:2 이상 보장
                    _sl_dist_new = abs(_sl - _px)
                    _min_tp = round(_px + _sl_dist_new * 2, 4) if sig == "long" else round(_px - _sl_dist_new * 2, 4)
                    if sig == "long":
                        _tp = max(_tp, _min_tp)
                    else:
                        _tp = min(_tp, _min_tp)
                    _sl_dist_pct = 2.0
                if _sl_dist_pct > 5.0:
                    _old_sl = _sl
                    if sig == "long":
                        _sl = round(_px * 0.95, 4)
                    else:
                        _sl = round(_px * 1.05, 4)
                    add_log(f"⚠️ 알트 {sym} SL 거리 캡: {_sl_dist_pct:.1f}%→5.0% (${_old_sl}→${_sl})")
                    # TP도 R:R 1:2 이상 보장으로 재조정
                    _sl_dist_new = abs(_sl - _px)
                    if sig == "long":
                        _tp = max(_tp, round(_px + _sl_dist_new * 2, 4))
                    else:
                        _tp = min(_tp, round(_px - _sl_dist_new * 2, 4))
            # 잔고 비례 진입금 계산 + 저유동성 축소 + 스케일업
            try:
                _alt_bal = get_balance()["available"]
                _alt_base = _alt_bal * POSITION_PCT_ALT / 100
                # 스케일업 (누적 성과 기반)
                _alt_scale = 1.0
                try:
                    _alt_closed = trade_db.get_closed_trades(limit=200)
                    _alt_valid = [t for t in _alt_closed if t.get("source", "").startswith("alt_") and (t.get("pnl") or 0) != 0 and (t.get("close_price") or 0) > 0]
                    _alt_cnt = len(_alt_valid)
                    _alt_wins = sum(1 for t in _alt_valid if (t.get("pnl") or 0) > 0)
                    _alt_wr = (_alt_wins / _alt_cnt * 100) if _alt_cnt > 0 else 0
                    for _min_cnt, _min_wr, _mult in SCALE_TABLE:
                        if _alt_cnt >= _min_cnt and _alt_wr >= _min_wr:
                            _alt_scale = _mult
                            add_log(f"📈 알트 스케일업: {_alt_cnt}건 승률{_alt_wr:.0f}% → ×{_mult}")
                            break
                except Exception:
                    pass
                _use_usdt = min(round(_alt_base * _alt_scale, 1), MAX_USDT_ALT)
                if _low_liq_hours:
                    _use_usdt = round(_use_usdt * 0.25, 1)  # 저유동성 시간대 75% 축소
                add_log(f"💼 알트 진입금: ${_use_usdt} (잔고${_alt_bal:.0f}×{POSITION_PCT_ALT}%=${_alt_base:.1f} × 스케일{_alt_scale:.1f}×)")
            except Exception:
                _use_usdt = MAX_USDT_ALT * 0.25 if _low_liq_hours else MAX_USDT_ALT
            # 최적 진입가 결정 (LLM entry 또는 ATR 기반 폴백)
            _entry_price = tj.get("entry")
            if _entry_price:
                _entry_price = float(_entry_price)
                # 진입가 유효성 검증: 현재가 대비 2% 이내
                _dist_pct = abs(_entry_price - _px) / _px * 100
                if _dist_pct > 3.0:
                    add_log(f"⚠️ 알트 {sym} LLM 진입가 ${_entry_price} 현재가 대비 {_dist_pct:.1f}% → ATR 폴백")
                    _entry_price = None
            if not _entry_price:
                # ATR 기반 폴백: 롱은 현재가 -0.3×ATR, 숏은 현재가 +0.3×ATR (체결률 우선)
                if sig == "long":
                    _entry_price = round(_px - _atr * 0.3, 6)
                else:
                    _entry_price = round(_px + _atr * 0.3, 6)
            add_log(f"📍 알트 {sym} 지정가 진입: ${_entry_price} (현재 ${_px}, 차이 {abs(_entry_price - _px) / _px * 100:.2f}%)")

            # 지정가(LIMIT) 주문 + SL/TP 동시 배치
            _r = place_limit_order(sym, _side, _use_usdt, _entry_price, ALT_LEVERAGE,
                                   sl_price=_sl, tp_price=_tp)
            if _r["success"]:
                # 미체결 주문 대기 등록 (체결 확인 루프에서 DB 기록 + SL/TP 배치)
                _alt_reason = f"알트스캔({source}) | {tj.get('reason', '')[:60]} | 신뢰도 {tj.get('confidence', 0)}% ({sig})"
                _pending = st.session_state.setdefault("_alt_pending_orders", {})
                _pending[sym] = {
                    "order_id": _r.get("order_id"),
                    "sl_algo_id": _r.get("sl_algo_id"),
                    "tp_algo_id": _r.get("tp_algo_id"),
                    "time": time.time(),
                    "expire": 1800,  # 30분
                    "entry_price": _entry_price,
                    "sl": _sl, "tp": _tp,
                    "side": sig,
                    "qty": _r["qty"],
                    "atr": _atr,
                    "confidence": tj.get("confidence", 0),
                    "source": f"alt_{source}",
                    "reason": _alt_reason,
                    "filled": False,
                }
                _sig_label = "🟢 롱" if sig == "long" else "🔴 숏"
                add_log(f"📋 알트 지정가 주문: {sym} {_sig_label} @ ${_entry_price} (신뢰도 {tj.get('confidence')}%, 체결 대기)")
                # 재진입 쿨다운 기록 (기본 4시간, 승리 시 30분으로 단축)
                st.session_state.setdefault("_alt_cooldowns", {})[sym] = {"time": time.time(), "duration": 14400}
                if st.session_state.get("tg_notify", True):
                    try:
                        send_message(*_tg(),
                                     f"📋 알트 지정가 주문: {sym} {_sig_label} @ ${_entry_price}\n"
                                     f"SL: ${_sl} | TP: ${_tp}\n"
                                     f"신뢰도: {tj.get('confidence')}% | 체결 대기 (최대 30분)")
                    except Exception:
                        pass
            else:
                add_log(f"❌ 알트 자동 주문 실패: {sym} {_r.get('error','')}", "error")
        except Exception as _e:
            add_log(f"알트 자동 주문 오류: {_e}", "error")

    # 분석 중 심볼 세트 (중복 분석 방지)
    _analyzing = st.session_state.setdefault("_alt_analyzing", set())

    # 풀백 대기 워치리스트: {sym: {"price_at_detect": float, "time": float, "source": str, ...}}
    # 급등/공지 감지 시 등록 → 다음 스캔에서 풀백 확인 후 진입
    _pullback_wl = st.session_state.setdefault("_alt_pullback_watchlist", {})
    # 풀백 체크: 등록된 심볼 중 조건 충족 시 진입
    _pb_done = []
    for _pb_sym, _pb_info in list(_pullback_wl.items()):
        _pb_age = time.time() - _pb_info.get("time", 0)
        if _pb_age > 1800:  # 30분 초과 → 만료 제거
            _pb_done.append(_pb_sym)
            continue
        if _pb_age < 60:  # 최소 1분 대기
            continue
        try:
            _pb_kl = get_klines(_pb_sym, "15m", 5)
            _pb_cur = float(_pb_kl["close"].iloc[-1])
            _pb_detect_price = _pb_info["price_at_detect"]
            _pb_dir = _pb_info["direction"]
            _pb_change = (_pb_cur - _pb_detect_price) / _pb_detect_price * 100
            # 풀백 조건: 급등 후 -1%~-3% 되돌림 (롱), 또는 급락 후 +1%~+3% 반등 (숏)
            _pb_ok = False
            if _pb_dir == "long" and -3.0 <= _pb_change <= -0.5:
                _pb_ok = True  # 급등 후 적절한 풀백
            elif _pb_dir == "short" and 0.5 <= _pb_change <= 3.0:
                _pb_ok = True  # 급락 후 적절한 반등
            elif _pb_dir == "long" and _pb_change < -5.0:
                _pb_done.append(_pb_sym)  # 너무 많이 빠짐 → 폐기
                add_log(f"📉 풀백 폐기: {_pb_sym} {_pb_change:+.1f}% (과도한 하락)")
                continue
            elif _pb_dir == "short" and _pb_change > 5.0:
                _pb_done.append(_pb_sym)
                add_log(f"📈 풀백 폐기: {_pb_sym} {_pb_change:+.1f}% (과도한 상승)")
                continue

            if _pb_ok:
                _pb_done.append(_pb_sym)
                add_log(f"🎯 풀백 진입 조건 충족: {_pb_sym} {_pb_change:+.1f}% (감지가 대비)")
                # 풀백 → 시그널 큐 전달 (LLM 없이, position_updater가 우선 스캔)
                push_signal(_pb_sym, 'pullback', direction=_pb_dir, priority=4,
                            meta={'source': _pb_info.get('source', ''), 'change_pct': round(_pb_change, 2)})
                add_log(f"🎯 {_pb_sym} 풀백 → 시그널 큐 전달 ({_pb_change:+.1f}%)")
        except Exception:
            pass
    for _done_sym in _pb_done:
        _pullback_wl.pop(_done_sym, None)

    # BTC 15m 지표 캐시 (알트 상관관계 필터용, 3분마다 갱신)
    _btc_cache_age = time.time() - st.session_state.get("_btc_ind_cache_ts", 0)
    if _btc_cache_age >= 180:
        try:
            _btc_kl = get_klines("BTCUSDT", "15m", 60)
            st.session_state["_btc_ind_cache"] = calc_indicators(_btc_kl)
            st.session_state["_btc_ind_cache_ts"] = time.time()
        except Exception:
            pass
    _btc_ind_cache = st.session_state.get("_btc_ind_cache", {})

    # 공지 스캔 (3분 주기)
    _ann_elapsed = time.time() - st.session_state.get("alt_last_announcement_time", 0)
    if _ann_elapsed >= _ALT_ANNOUNCE_INTERVAL:
        try:
            _ann_r = check_binance_announcements()
            st.session_state.alt_announcements          = _ann_r
            st.session_state.alt_last_announcement_time = time.time()
            if _ann_r.get("found"):
                add_log(f"📢 공지 감지: {_ann_r.get('summary','')}")
                # 공지 → 시그널 큐 전달 (LLM 없이, position_updater가 우선 스캔)
                if st.session_state.get("alt_auto_trade", True):
                    for _ann_item in _ann_r.get("announcements", []):
                        _ann_sym = _ann_item.get("symbol", "")
                        _ann_sig = _ann_item.get("signal", "wait")
                        if not _ann_sym or _ann_sig == "wait":
                            continue
                        push_signal(_ann_sym, 'announcement', direction=_ann_sig, priority=1,
                                    meta={'type': _ann_item.get('type', ''), 'exchange': 'binance'})
                        add_log(f"📢 {_ann_sym} → 시그널 큐 전달 (공지: {_ann_item.get('type','')})")
        except Exception:
            pass

        # 업비트 공지 감지 (바이낸스 공지와 같은 주기)
        try:
            _upbit_r = check_upbit_announcements()
            st.session_state["_upbit_announcements"] = _upbit_r
            if _upbit_r.get("found"):
                add_log(f"🇰🇷 업비트 공지: {_upbit_r.get('summary','')}")
                if st.session_state.get("tg_notify", True):
                    from telegram_notifier import send_message
                    send_message(*_tg(),
                                 f"🇰🇷 {_upbit_r['summary']}")
                if st.session_state.get("alt_auto_trade", True):
                    for _uitem in _upbit_r.get("announcements", []):
                        _usym = _uitem.get("symbol", "")
                        _usig = _uitem.get("signal", "wait")
                        if not _usym or _usig == "wait":
                            continue
                        push_signal(_usym, 'announcement', direction=_usig, priority=2,
                                    meta={'type': _uitem.get('type', ''), 'exchange': 'upbit'})
                        add_log(f"🇰🇷 {_usym} → 시그널 큐 전달 (업비트 공지)")
        except Exception:
            pass

        # OKX 공지 감지
        try:
            _okx_r = check_okx_announcements()
            st.session_state["_okx_announcements"] = _okx_r
            if _okx_r.get("found"):
                add_log(f"🔶 OKX 공지: {_okx_r.get('summary','')}")
                if st.session_state.get("tg_notify", True):
                    from telegram_notifier import send_message
                    send_message(*_tg(),
                                 f"🔶 {_okx_r['summary']}")
                if st.session_state.get("alt_auto_trade", True):
                    for _oitem in _okx_r.get("announcements", []):
                        _osym = _oitem.get("symbol", "")
                        _osig = _oitem.get("signal", "wait")
                        if not _osym or _osig == "wait":
                            continue
                        push_signal(_osym, 'announcement', direction=_osig, priority=2,
                                    meta={'type': _oitem.get('type', ''), 'exchange': 'okx'})
                        add_log(f"🔶 {_osym} → 시그널 큐 전달 (OKX 공지)")
        except Exception:
            pass

        # 코인베이스 신규 상장 감지
        try:
            _cb_r = check_coinbase_listings()
            st.session_state["_coinbase_announcements"] = _cb_r
            if _cb_r.get("found"):
                add_log(f"🔵 코인베이스: {_cb_r.get('summary','')}")
                if st.session_state.get("tg_notify", True):
                    from telegram_notifier import send_message
                    send_message(*_tg(),
                                 f"🔵 {_cb_r['summary']}")
                if st.session_state.get("alt_auto_trade", True):
                    for _citem in _cb_r.get("announcements", []):
                        _csym = _citem.get("symbol", "")
                        _csig = _citem.get("signal", "wait")
                        if not _csym or _csig == "wait":
                            continue
                        push_signal(_csym, 'announcement', direction=_csig, priority=2,
                                    meta={'type': _citem.get('type', ''), 'exchange': 'coinbase'})
                        add_log(f"🔵 {_csym} → 시그널 큐 전달 (코인베이스 상장)")
        except Exception:
            pass

    # 급등/급락 감지 (매 스캔 시 스냅샷 촬영 + 비교)
    try:
        _surges = detect_surges()
        st.session_state["_surge_results"] = _surges
        if _surges:
            _top = _surges[0]
            add_log(f"⚡ 급등 감지 {len(_surges)}건 — 1위: {_top['symbol']} {_top['surge_type']} (score={_top['score']})")
            # 상위 3개 급등 종목 처리: 강한 추세는 즉시 진입, 나머지는 풀백 대기
            if st.session_state.get("alt_auto_trade", True):
                for _sg in _surges[:3]:
                    if _sg["score"] < 50:
                        continue
                    _ssym = _sg["symbol"]
                    if (_ssym in st.session_state.alt_analysis
                            or _ssym in _analyzing
                            or _ssym in _pullback_wl):
                        continue

                    # 급등 → 시그널 큐 전달 (LLM 없이, position_updater가 우선 스캔)
                    if _sg["score"] >= 80:
                        push_signal(_ssym, 'surge', direction=_sg["direction"], priority=3,
                                    meta={'surge_type': _sg['surge_type'], 'score': _sg['score']})
                        add_log(f"⚡ {_ssym} 강한 급등 → 시그널 큐 전달 (score={_sg['score']})")
                        continue

                    # 양방향 풀백 등록 허용
                    _pullback_wl[_ssym] = {
                        "price_at_detect": _sg["price"],
                        "time": time.time(),
                        "direction": _sg["direction"],
                        "source": "surge",
                        "score": _sg["score"],
                        "surge_type": _sg["surge_type"],
                    }
                    add_log(f"⚡ {_ssym} 급등 감지 → 풀백 대기 등록 ({_sg['surge_type']})")
    except Exception:
        pass

    # ── 알트 포지션 모니터링 (트레일링 스탑 + 시간 기반 청산) ──
    try:
        _alt_positions = [p for p in get_positions() if p["symbol"] not in SYMBOLS]
        _alt_peak = st.session_state.setdefault("_alt_peak_pnl", {})  # {sym: peak_upnl_pct}
        _alt_entry_ts = st.session_state.setdefault("_alt_entry_time", {})  # {sym: entry_time}

        for _ap in _alt_positions:
            _ap_sym = _ap["symbol"]
            _ap_upnl = float(_ap.get("unRealizedProfit", 0))
            _ap_amt = abs(float(_ap.get("positionAmt", 0)))
            _ap_entry = float(_ap.get("entryPrice", 0))
            _ap_notional = _ap_amt * _ap_entry if _ap_entry > 0 else (MAX_USDT_ALT * ALT_LEVERAGE)
            _ap_upnl_pct = (_ap_upnl / _ap_notional) * 100 if _ap_notional > 0 else 0

            # 진입 시각 기록 (첫 감지 시)
            if _ap_sym not in _alt_entry_ts:
                _alt_entry_ts[_ap_sym] = time.time()

            # 피크 uPnL 갱신
            _prev_peak = _alt_peak.get(_ap_sym, 0)
            if _ap_upnl_pct > _prev_peak:
                _alt_peak[_ap_sym] = _ap_upnl_pct

            # ── 2단계 TP 관리: 1차(40%) 50% 익절 + BE SL, 2차(100%) 나머지 익절 ──
            _tp_plan = st.session_state.get("_alt_tp_plan", {}).get(_ap_sym)
            _is_long = float(_ap.get("positionAmt", 0)) > 0
            _mark = float(_ap.get("markPrice", 0))

            if _tp_plan and _mark > 0:
                _tp1 = _tp_plan["tp1"]
                _tp2 = _tp_plan["tp2"]
                _tp_entry = _tp_plan["entry"]

                # 1차 TP 도달: 50% 수량 익절 + SL을 진입가(BE)로 이동
                if not _tp_plan["tp1_done"]:
                    _tp1_hit = (_is_long and _mark >= _tp1) or (not _is_long and _mark <= _tp1)
                    if _tp1_hit:
                        try:
                            _half_qty = round(_ap_amt / 2, 4)
                            if _half_qty > 0:
                                _partial_side = "SELL" if _is_long else "BUY"
                                from binance_client import get_client as _gc
                                _cl = _gc()
                                # 50% 시장가 익절
                                _cl.futures_create_order(
                                    symbol=_ap_sym, side=_partial_side, type="MARKET",
                                    quantity=_half_qty, reduceOnly=True
                                )
                                # 기존 SL 취소 + 브레이크이븐 SL 배치
                                cancel_open_orders(_ap_sym)
                                _be_price = round(_tp_entry * (1.001 if _is_long else 0.999), 4)
                                _remaining = round(_ap_amt - _half_qty, 4)
                                _cl.futures_create_order(
                                    symbol=_ap_sym, side="SELL" if _is_long else "BUY",
                                    type="STOP_MARKET", stopPrice=str(_be_price),
                                    quantity=_remaining, reduceOnly=True
                                )
                                _tp_plan["tp1_done"] = True
                                _tp1_pct = abs(_tp1 - _tp_entry) / _tp_entry * 100
                                add_log(f"💰 알트 1차 익절: {_ap_sym} 50% @ ${_mark:.4f} (+{_tp1_pct:.1f}%) + BE SL ${_be_price}")
                                if st.session_state.get("tg_notify", True):
                                    send_message(*_tg(),
                                                 f"💰 {_ap_sym} 1차 익절 50% (+{_tp1_pct:.1f}%)\n🛡️ SL → 진입가 ${_be_price} (무손실)")
                        except Exception as _tp1_e:
                            add_log(f"1차 익절 오류: {_ap_sym} {_tp1_e}", "error")

                # 2차 TP 도달: 나머지 전량 익절
                elif not _tp_plan["tp2_done"]:
                    _tp2_hit = (_is_long and _mark >= _tp2) or (not _is_long and _mark <= _tp2)
                    if _tp2_hit:
                        _cr = close_position(_ap_sym)
                        if _cr.get("success"):
                            _tp2_pct = abs(_tp2 - _tp_entry) / _tp_entry * 100
                            _tp_plan["tp2_done"] = True
                            add_log(f"🎯 알트 2차 익절: {_ap_sym} 전량 @ ${_mark:.4f} (+{_tp2_pct:.1f}%)")
                            if st.session_state.get("tg_notify", True):
                                send_message(*_tg(),
                                             f"🎯 {_ap_sym} 2차 익절 전량 (+{_tp2_pct:.1f}%)")
                        _alt_peak.pop(_ap_sym, None)
                        _alt_entry_ts.pop(_ap_sym, None)
                        continue

                    # 1차 익절 후 트레일링: 피크 대비 40% 되돌림 시 나머지 청산 (50→40% 강화)
                    _peak_val = _alt_peak.get(_ap_sym, 0)
                    if _peak_val >= 2.0 and _ap_upnl_pct < _peak_val * 0.6:
                        add_log(f"📊 알트 트레일링: {_ap_sym} 피크 {_peak_val:.1f}% → {_ap_upnl_pct:.1f}%")
                        _tr_r = close_position(_ap_sym)
                        if _tr_r.get("success"):
                            add_log(f"✅ {_ap_sym} 트레일링 청산 (1차 익절 후 수익 보호)")
                            if st.session_state.get("tg_notify", True):
                                send_message(*_tg(),
                                             f"📊 {_ap_sym} 트레일링 청산\n피크 {_peak_val:.1f}% → {_ap_upnl_pct:.1f}%")
                        _alt_peak.pop(_ap_sym, None)
                        _alt_entry_ts.pop(_ap_sym, None)
                        continue

            _hold_hours = (time.time() - _alt_entry_ts.get(_ap_sym, time.time())) / 3600

            # 손실 포지션 조기 청산 (단계적)
            # 30분+ & -1.0% → 즉시 청산 (큰 손실 빠른 절단)
            # 45분+ & -0.5% → 청산 (기존 1h→45m 단축)
            if (_hold_hours >= 0.5 and _ap_upnl_pct <= -1.0) or (_hold_hours >= 0.75 and _ap_upnl_pct <= -0.5):
                add_log(f"✂️ 알트 손실 조기 청산: {_ap_sym} {_hold_hours:.1f}h 보유, uPnL {_ap_upnl_pct:+.1f}%")
                _lc_r = close_position(_ap_sym)
                if _lc_r.get("success"):
                    add_log(f"✅ {_ap_sym} 손실 조기 청산 완료")
                    if st.session_state.get("tg_notify", True):
                        send_message(*_tg(),
                                     f"✂️ {_ap_sym} 손실 조기 청산 ({_hold_hours:.0f}h, {_ap_upnl_pct:+.1f}%)")
                _alt_peak.pop(_ap_sym, None)
                _alt_entry_ts.pop(_ap_sym, None)
                st.session_state.get("_alt_partial_done", set()).discard(_ap_sym)
                continue

            # 시간 기반 청산: 2시간 경과 + 수익률 ±0.5% 미만 → 횡보 중 청산 (3h→2h 단축)
            if _hold_hours >= 2.0 and abs(_ap_upnl_pct) < 0.5:
                add_log(f"⏰ 알트 시간 청산: {_ap_sym} {_hold_hours:.1f}h 보유, uPnL {_ap_upnl_pct:+.1f}% (횡보)")
                _tc_r = close_position(_ap_sym)
                if _tc_r.get("success"):
                    add_log(f"✅ {_ap_sym} 시간 기반 청산 완료")
                    if st.session_state.get("tg_notify", True):
                        send_message(*_tg(),
                                     f"⏰ {_ap_sym} 시간 청산 ({_hold_hours:.0f}h, {_ap_upnl_pct:+.1f}%)")
                _alt_peak.pop(_ap_sym, None)
                _alt_entry_ts.pop(_ap_sym, None)
                st.session_state.get("_alt_partial_done", set()).discard(_ap_sym)
                continue

        # 청산된 포지션 정리 + 잔존 주문 취소 + 쿨다운 차등 조정
        _active_syms = {p["symbol"] for p in _alt_positions}
        for _old_sym in list(_alt_peak.keys()):
            if _old_sym not in _active_syms:
                _alt_peak.pop(_old_sym, None)
                _alt_entry_ts.pop(_old_sym, None)
                st.session_state.get("_alt_partial_done", set()).discard(_old_sym)
                st.session_state.get("_alt_be_done", set()).discard(_old_sym)
                st.session_state.get("_alt_dca_pending", {}).pop(_old_sym, None)
                # 쿨다운 차등: 마지막 거래 PnL 확인 → 승리 30분, 패배 4시간
                try:
                    _last_trade = trade_db.get_recent_trades_by_symbol(_old_sym, limit=1)
                    if _last_trade and (_last_trade[0].get("pnl") or 0) > 0:
                        st.session_state.setdefault("_alt_cooldowns", {})[_old_sym] = {"time": time.time(), "duration": 1800}  # 승리 → 30분
                except Exception:
                    pass
                # 잔존 SL/TP 주문 정리 (TP/SL 자동 체결 후 반대편 고아 주문)
                try:
                    cancel_open_orders(_old_sym)
                except Exception:
                    pass
    except Exception:
        pass

    # 지정가 주문 관리: 체결 확인 → SL/TP 배치 / 만료 취소
    _pending = st.session_state.get("_alt_pending_orders", {})
    _done_syms = []
    for _psym, _pinfo in _pending.items():
        _elapsed = time.time() - _pinfo.get("time", 0)
        # 포지션 존재 확인 = 체결됨
        _has_pos = any(p["symbol"] == _psym and float(p.get("positionAmt", 0)) != 0
                       for p in get_positions())
        if _has_pos and not _pinfo.get("filled"):
            # 체결 확인! → DB 기록 (SL/TP는 주문 시 이미 배치됨)
            _pinfo["filled"] = True
            add_log(f"✅ 알트 {_psym} 지정가 체결! {'🟢 롱' if _pinfo['side']=='long' else '🔴 숏'} @ ${_pinfo['entry_price']}")
            # DB 기록
            try:
                _alt_reason = _pinfo.get("reason", "")
                _add_journal_entry({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": _psym,
                    "side": "🟢 롱" if _pinfo["side"] == "long" else "🔴 숏",
                    "action": "진입", "qty": _pinfo.get("qty", 0),
                    "price": _pinfo["entry_price"],
                    "sl": _pinfo.get("sl"), "tp": _pinfo.get("tp"),
                    "atr": _pinfo.get("atr", 0), "pnl": None,
                    "confidence": _pinfo.get("confidence", 0),
                    "source": _pinfo.get("source", "alt_screener"),
                    "fill_price": _pinfo["entry_price"], "slippage": 0,
                    "entry_reason": _alt_reason,
                })
            except Exception:
                pass
            if st.session_state.get("tg_notify", True):
                try:
                    send_message(*_tg(),
                                 f"✅ 알트 {_psym} 지정가 체결! {'🟢 롱' if _pinfo['side']=='long' else '🔴 숏'} @ ${_pinfo['entry_price']}")
                except Exception:
                    pass
            _done_syms.append(_psym)
        elif not _has_pos and _elapsed >= _pinfo.get("expire", 1800):
            # 30분 만료 → LIMIT + algo SL/TP 모두 취소
            try:
                cancel_open_orders(_psym)
                # algo 주문 (SL/TP) 도 취소
                _client = get_client()
                for _aid_key in ("sl_algo_id", "tp_algo_id"):
                    _aid = _pinfo.get(_aid_key)
                    if _aid:
                        try:
                            _client._request_futures_api('delete', 'algoOrder', True, data={'algoId': _aid})
                        except Exception:
                            pass
                add_log(f"⏰ 알트 {_psym} 지정가 미체결 만료 (30분) → LIMIT+SL+TP 전체 취소")
            except Exception:
                pass
            _done_syms.append(_psym)
    for _esym in _done_syms:
        _pending.pop(_esym, None)

    # 스크리너 (3분 주기) — 포지션 한도 도달 시 스킵 (API 호출 절감)
    _sc_elapsed = time.time() - st.session_state.get("alt_last_scan_time", 0)
    if _sc_elapsed >= _ALT_SCREEN_INTERVAL:
        # 포지션 한도 체크: 이미 꽉 찬 상태에서 스크리너 + 분석 실행은 낭비
        try:
            _cur_pos_check = get_positions()
            _alt_pos_cnt = len([p for p in _cur_pos_check if p["symbol"] not in SYMBOLS])
            if _alt_pos_cnt >= MAX_ALT_POSITIONS:
                st.session_state.alt_last_scan_time = time.time()  # 타이머 리셋
                add_log(f"⏭️ 알트 스캔 스킵: 이미 {_alt_pos_cnt}/{MAX_ALT_POSITIONS}포지션")
        except Exception:
            _alt_pos_cnt = 0
        if _alt_pos_cnt < MAX_ALT_POSITIONS:
            try:
                _syms_auto    = get_alt_futures_symbols(ALT_SCAN_LIMIT)
                _results_auto = screen_altcoins(_syms_auto, top_n=10)  # 테스트2: 5→10
                st.session_state.alt_scan_results   = _results_auto
                st.session_state.alt_last_scan_time = time.time()
                if _results_auto:
                    _top = _results_auto[0]
                    add_log(f"🔥 알트 스캔: {_top['symbol']} 1위 (점수 {_top['score']})")
                    # 상위 종목 → 시그널 큐 전달 (LLM 없이, position_updater가 스코어링)
                    if st.session_state.get("alt_auto_trade", True):
                        for _rank_cand in _results_auto[:5]:
                            if _rank_cand["score"] < ALT_MIN_SCORE:
                                continue
                            _rc_sym = _rank_cand["symbol"]
                            _rc_dir = _rank_cand.get("direction", "long")
                            push_signal(_rc_sym, 'screener', direction=_rc_dir, priority=5,
                                        meta={'score': _rank_cand['score']})
                        add_log(f"🔥 알트 스캔 상위 → 시그널 큐 전달 ({len(_results_auto[:5])}건)")
            except Exception:
                pass

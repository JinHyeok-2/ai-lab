#!/usr/bin/env python3
# 거래 기록 SQLite 저장소 — JSON 파일 대체
# trade_journal.json → trades.db 마이그레이션 포함

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime

_DB_PATH = Path(__file__).parent / "trades.db"
_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    """테이블 생성 (없으면)"""
    with _db_lock:
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                time        TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                side        TEXT,
                action      TEXT DEFAULT '진입',
                qty         REAL,
                price       REAL,
                sl          REAL,
                tp          REAL,
                atr         REAL,
                pnl         REAL,
                confidence  INTEGER,
                close_time  TEXT,
                close_price REAL,
                sl_mode     TEXT,
                source      TEXT DEFAULT 'main',
                extra       TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(time)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_balance (
                date    TEXT PRIMARY KEY,
                balance REAL NOT NULL
            )
        """)
        # 일별 잔고 히스토리 (매일 자동 기록)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS balance_history (
                date       TEXT PRIMARY KEY,
                open_bal   REAL,
                close_bal  REAL,
                high_bal   REAL,
                low_bal    REAL,
                pnl        REAL,
                trades     INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        # 봇 상태 영속화 (trading_paused 등 — 재시작 후 복원)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # 분석 이력 (매 분석 사이클마다 기록 — 롱/숏/관망 모두 포함)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                time        TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                decision    TEXT,
                confidence  INTEGER,
                entry_price REAL,
                sl          REAL,
                tp          REAL,
                rsi_15m     REAL,
                rsi_1h      REAL,
                rsi_4h      REAL,
                adx_15m     REAL,
                ema20_15m   REAL,
                ema50_15m   REAL,
                ema20_4h    REAL,
                ema50_4h    REAL,
                atr         REAL,
                macd_hist   REAL,
                btc_price   REAL,
                btc_trend   TEXT,
                rl_signal   TEXT,
                llm_reason  TEXT,
                order_type  TEXT,
                order_id    TEXT,
                source      TEXT DEFAULT 'main',
                filled      INTEGER DEFAULT 0,
                extra       TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_time ON analysis_log(time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON analysis_log(symbol)")
        conn.commit()
        conn.close()


def migrate_from_json(json_path: str = None):
    """기존 trade_journal.json → SQLite 마이그레이션 (중복 방지)"""
    if json_path is None:
        json_path = Path(__file__).parent / "trade_journal.json"
    else:
        json_path = Path(json_path)

    if not json_path.exists():
        return 0

    with open(json_path, encoding="utf-8") as f:
        journal = json.load(f)

    if not journal:
        return 0

    with _db_lock:
        conn = _get_conn()
        # 이미 마이그레이션된 경우 스킵
        existing = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        if existing > 0:
            conn.close()
            return 0

        count = 0
        for entry in journal:
            _insert_trade_row(conn, entry)
            count += 1
        conn.commit()
        conn.close()
    return count


def _insert_trade_row(conn: sqlite3.Connection, entry: dict):
    """dict → trades 테이블 INSERT"""
    # 기존 JSON 필드 중 정의된 컬럼 외의 데이터는 extra에 보관
    known_cols = {
        "time", "symbol", "side", "action", "qty", "price",
        "sl", "tp", "atr", "pnl", "confidence",
        "close_time", "close_price", "sl_mode", "source"
    }
    extra = {k: v for k, v in entry.items() if k not in known_cols and k != "id"}
    conn.execute("""
        INSERT INTO trades (time, symbol, side, action, qty, price,
                            sl, tp, atr, pnl, confidence,
                            close_time, close_price, sl_mode, source, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        entry.get("symbol", ""),
        entry.get("side", ""),
        entry.get("action", "진입"),
        entry.get("qty"),
        entry.get("price"),
        entry.get("sl"),
        entry.get("tp"),
        entry.get("atr"),
        entry.get("pnl"),
        entry.get("confidence"),
        entry.get("close_time"),
        entry.get("close_price"),
        entry.get("sl_mode"),
        entry.get("source", "main"),
        json.dumps(extra, ensure_ascii=False) if extra else None,
    ))


def add_trade(entry: dict):
    """새 거래 추가"""
    with _db_lock:
        conn = _get_conn()
        _insert_trade_row(conn, entry)
        conn.commit()
        conn.close()


def update_trade_pnl(symbol: str, pnl: float, close_price: float = None):
    """심볼의 가장 최근 미청산(pnl IS NULL) 거래에 PnL 기록"""
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM trades WHERE symbol=? AND pnl IS NULL ORDER BY id DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        if row:
            conn.execute("""
                UPDATE trades SET pnl=?, action='청산', close_time=?, close_price=?
                WHERE id=?
            """, (
                round(pnl, 4),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                close_price,
                row["id"],
            ))
            conn.commit()
        conn.close()


def update_trade_field(trade_id: int, **kwargs):
    """특정 거래의 필드 업데이트"""
    if not kwargs:
        return
    with _db_lock:
        conn = _get_conn()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [trade_id]
        conn.execute(f"UPDATE trades SET {sets} WHERE id=?", vals)
        conn.commit()
        conn.close()


def get_open_trades(symbol: str = None) -> list:
    """미청산 거래 조회"""
    with _db_lock:
        conn = _get_conn()
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE pnl IS NULL AND symbol=? ORDER BY id DESC",
                (symbol,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE pnl IS NULL ORDER BY id DESC"
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_closed_trades(symbol: str = None, limit: int = 200) -> list:
    """청산 완료 거래 조회"""
    with _db_lock:
        conn = _get_conn()
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE pnl IS NOT NULL AND symbol=? ORDER BY id DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE pnl IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_all_trades(limit: int = 200) -> list:
    """전체 거래 조회 (최신순)"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_daily_summary() -> list:
    """일별 PnL 집계 (SQL 레벨)"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT
                SUBSTR(COALESCE(close_time, time), 1, 10) AS date,
                COUNT(*)                                   AS trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)  AS wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                ROUND(SUM(pnl), 4)                         AS pnl,
                ROUND(AVG(pnl), 4)                         AS avg_pnl,
                ROUND(MAX(pnl), 4)                         AS best_trade,
                ROUND(MIN(pnl), 4)                         AS worst_trade
            FROM trades
            WHERE pnl IS NOT NULL
            GROUP BY date
            ORDER BY date
        """).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_symbol_summary() -> list:
    """심볼별 PnL 집계"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT
                symbol,
                COUNT(*)                                   AS trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)  AS wins,
                ROUND(SUM(pnl), 4)                         AS total_pnl,
                ROUND(AVG(pnl), 4)                         AS avg_pnl
            FROM trades
            WHERE pnl IS NOT NULL
            GROUP BY symbol
            ORDER BY total_pnl DESC
        """).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_hourly_pnl() -> list:
    """시간대별 PnL 집계 (어느 시간에 수익 나는지)"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT
                SUBSTR(time, 12, 2) AS hour,
                COUNT(*)            AS trades,
                ROUND(SUM(pnl), 4)  AS total_pnl,
                ROUND(AVG(pnl), 4)  AS avg_pnl
            FROM trades
            WHERE pnl IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        """).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_trade_stats(symbol: str = None) -> dict:
    """거래 통계 (승률, PF, MDD 등) — 고스트 거래(pnl=0+정리) 제외"""
    closed = get_closed_trades(symbol=symbol, limit=9999)
    if not closed:
        return {}
    # 고스트 필터: pnl=0 + (action에 고스트/싱크 OR close_price=0/NULL) 거래 제외
    closed = [t for t in closed if not (
        (t.get("pnl") or 0) == 0 and (
            any(k in (t.get("action") or "") for k in ["고스트", "싱크"]) or
            not t.get("close_price") or t.get("close_price") == 0
        )
    )]
    if not closed:
        return {}
    wins = [t for t in closed if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("pnl") or 0) <= 0]
    total_pnl = sum(t["pnl"] for t in closed)
    win_sum = sum(t["pnl"] for t in wins) if wins else 0
    loss_sum = sum(t["pnl"] for t in losses) if losses else 0
    avg_win = win_sum / len(wins) if wins else 0
    avg_loss = loss_sum / len(losses) if losses else 0
    pf = abs(win_sum / loss_sum) if loss_sum != 0 else 0

    # MDD 계산
    running, peak, mdd = 0, 0, 0
    for t in reversed(closed):
        running += t["pnl"]
        if running > peak:
            peak = running
        dd = peak - running
        if dd > mdd:
            mdd = dd

    return {
        "total": len(closed), "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(closed) * 100,
        "total_pnl": total_pnl,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": pf, "mdd": mdd,
    }


def save_daily_balance(date_str: str, balance: float):
    """일일 시작 잔고 저장"""
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO daily_balance (date, balance) VALUES (?, ?)",
            (date_str, balance)
        )
        conn.commit()
        conn.close()


def get_daily_balance(date_str: str) -> float | None:
    """일일 시작 잔고 조회"""
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT balance FROM daily_balance WHERE date=?", (date_str,)
        ).fetchone()
        conn.close()
    return row["balance"] if row else None


def get_trade_lessons(limit: int = 20) -> str:
    """과거 거래에서 교훈 추출 — 에이전트에게 전달할 텍스트 생성"""
    closed = get_closed_trades(limit=limit)
    if not closed:
        return ""

    lines = ["[과거 거래 교훈]"]

    # 방향 편향 분석 (side 형식: "🟢 롱", "🔴 숏", "LONG", "SHORT" 등 혼용 대응)
    def _is_long(s):
        s = (s or "").upper()
        return "롱" in s or "LONG" in s or "BUY" in s
    def _is_short(s):
        s = (s or "").upper()
        return "숏" in s or "SHORT" in s or "SELL" in s
    longs  = [t for t in closed if _is_long(t.get("side"))]
    shorts = [t for t in closed if _is_short(t.get("side"))]
    if longs or shorts:
        l_win = sum(1 for t in longs  if (t.get("pnl") or 0) > 0)
        s_win = sum(1 for t in shorts if (t.get("pnl") or 0) > 0)
        l_rate = f"{l_win}/{len(longs)}={l_win/len(longs)*100:.0f}%" if longs else "0건"
        s_rate = f"{s_win}/{len(shorts)}={s_win/len(shorts)*100:.0f}%" if shorts else "0건"
        total = len(longs) + len(shorts)
        lines.append(f"• 방향 분포: 롱 {len(longs)}건({l_rate}) / 숏 {len(shorts)}건({s_rate}) (최근 {limit}건)")
        # 편향 경고 (75% 이상 한쪽이면 경고)
        if total >= 5:
            short_pct = len(shorts) / total * 100
            long_pct = len(longs) / total * 100
            if short_pct >= 75:
                lines.append(f"  ⚠️ 숏 편향 심각 ({short_pct:.0f}%) — 롱 기회를 적극 검토하세요!")
            elif long_pct >= 75:
                lines.append(f"  ⚠️ 롱 편향 심각 ({long_pct:.0f}%) — 숏 기회를 적극 검토하세요!")

    # 연속 손실 패턴
    max_streak, cur_streak = 0, 0
    for t in reversed(closed):
        if (t.get("pnl") or 0) <= 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    if max_streak >= 2:
        lines.append(f"• 최대 연속 손실: {max_streak}회 — 연속 손실 시 포지션 축소 권장")

    # 신뢰도별 승률
    conf_bins = {"높음(70+)": [], "중간(50-69)": [], "낮음(<50)": []}
    for t in closed:
        c = t.get("confidence") or 0
        if c >= 70:
            conf_bins["높음(70+)"].append(t)
        elif c >= 50:
            conf_bins["중간(50-69)"].append(t)
        else:
            conf_bins["낮음(<50)"].append(t)
    conf_parts = []
    for label, trades in conf_bins.items():
        if trades:
            wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
            conf_parts.append(f"{label}: {wins}/{len(trades)}={wins/len(trades)*100:.0f}%")
    if conf_parts:
        lines.append(f"• 신뢰도별 승률: {' / '.join(conf_parts)}")

    # 시간대별 성과 분석 (KST 기준)
    hour_pnl = {}
    for t in closed:
        _time = t.get("time", "")
        if len(_time) >= 13:
            try:
                _h = int(_time[11:13])
                if _h not in hour_pnl:
                    hour_pnl[_h] = []
                hour_pnl[_h].append(t.get("pnl") or 0)
            except ValueError:
                pass
    if hour_pnl:
        bad_hours = []
        good_hours = []
        for h, pnls in sorted(hour_pnl.items()):
            total = sum(pnls)
            if len(pnls) >= 2 and total < 0:
                bad_hours.append(f"{h}시(${total:.1f})")
            elif len(pnls) >= 2 and total > 0:
                good_hours.append(f"{h}시(+${total:.1f})")
        if bad_hours:
            lines.append(f"• 손실 시간대: {', '.join(bad_hours)} — 해당 시간 진입 주의")
        if good_hours:
            lines.append(f"• 수익 시간대: {', '.join(good_hours)}")

    # 최근 3건 요약
    recent = closed[:3]
    if recent:
        lines.append("• 최근 거래:")
        for t in recent:
            pnl = t.get("pnl") or 0
            sym = t.get("symbol", "?")
            side = t.get("side", "?")
            conf = t.get("confidence") or 0
            emoji = "✅" if pnl > 0 else "❌"
            lines.append(f"  {emoji} {sym} {side} | PnL: ${pnl:.2f} | 신뢰도: {conf}%")

    return "\n".join(lines)


def get_daily_trade_count(symbol: str, date_str: str) -> int:
    """특정 심볼의 당일 거래 횟수 조회 (SQLite 기반, 재시작 안전)"""
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE symbol=? AND SUBSTR(time,1,10)=? AND action='진입'",
            (symbol, date_str)
        ).fetchone()
        conn.close()
    return row[0] if row else 0


def get_confidence_calibration() -> dict:
    """신뢰도 구간별 실제 승률 반환 — 캘리브레이션용.
    반환: {(lo, hi): {"count": N, "wins": W, "win_rate": float, "adj": int}}
    adj = 실제 승률 기반 보정값 (양수=과소평가→보정↑, 음수=과대평가→보정↓)
    """
    closed = get_closed_trades(limit=100)
    if not closed:
        return {}
    # 고스트 필터: pnl=0 + (action에 고스트/싱크 OR close_price=0/NULL) 제외
    closed = [t for t in closed if not (
        (t.get("pnl") or 0) == 0 and (
            any(k in (t.get("action") or "") for k in ["고스트", "싱크"]) or
            not t.get("close_price") or t.get("close_price") == 0
        )
    )]
    if not closed:
        return {}
    bins = [(0, 49), (50, 59), (60, 69), (70, 79), (80, 100)]
    result = {}
    for lo, hi in bins:
        group = [t for t in closed if lo <= (t.get("confidence") or 0) <= hi and t.get("pnl") is not None]
        if not group:
            result[(lo, hi)] = {"count": 0, "wins": 0, "win_rate": 0, "adj": 0}
            continue
        wins = len([t for t in group if (t.get("pnl") or 0) > 0])
        wr = wins / len(group)
        mid = (lo + hi) / 200  # 구간 중앙을 확률로 (0~1)
        adj = int((wr - mid) * 20)  # 실제 승률과 예상의 차이 → 보정 포인트
        result[(lo, hi)] = {"count": len(group), "wins": wins, "win_rate": round(wr, 3), "adj": adj}
    return result


def record_balance(date_str: str, balance: float, trade_count: int = None):
    """일별 잔고 기록 — 매 체크 시 high/low 갱신, 하루 마지막이 close_bal"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM balance_history WHERE date=?", (date_str,)
        ).fetchone()
        if row is None:
            # 당일 첫 기록
            conn.execute("""
                INSERT INTO balance_history (date, open_bal, close_bal, high_bal, low_bal, pnl, trades, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """, (date_str, balance, balance, balance, balance, trade_count or 0, now))
        else:
            _high = max(row["high_bal"] or balance, balance)
            _low = min(row["low_bal"] or balance, balance)
            _pnl = round(balance - (row["open_bal"] or balance), 4)
            _trades = trade_count if trade_count is not None else row["trades"]
            conn.execute("""
                UPDATE balance_history
                SET close_bal=?, high_bal=?, low_bal=?, pnl=?, trades=?, updated_at=?
                WHERE date=?
            """, (balance, _high, _low, _pnl, _trades, now, date_str))
        conn.commit()
        conn.close()


def get_balance_history(limit: int = 90) -> list:
    """일별 잔고 히스토리 조회 (최근 N일)"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM balance_history ORDER BY date DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def save_bot_state(key: str, value: str):
    """봇 상태 저장 (trading_paused 등 재시작 안전)"""
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
            (key, value)
        )
        conn.commit()
        conn.close()


def get_bot_state(key: str, default: str = None) -> str | None:
    """봇 상태 조회"""
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT value FROM bot_state WHERE key=?", (key,)
        ).fetchone()
        conn.close()
    return row["value"] if row else default


def get_symbol_win_rate(symbol: str, recent_n: int = 5) -> dict:
    """특정 종목의 최근 N건 승률 반환 — 알트 블랙리스트 판단용"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE symbol=? AND pnl IS NOT NULL ORDER BY id DESC LIMIT ?",
            (symbol, recent_n)
        ).fetchall()
        conn.close()
    if not rows:
        return {"trades": 0, "wins": 0, "win_rate": 100}
    pnls = [r["pnl"] for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    return {"trades": len(pnls), "wins": wins, "win_rate": round(wins / len(pnls) * 100, 1)}


def get_recent_trades_by_symbol(symbol: str, limit: int = 3) -> list:
    """특정 종목의 최근 N건 거래 반환 — 연속 손실 판단용"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT pnl, close_time FROM trades WHERE symbol=? AND pnl IS NOT NULL AND close_time IS NOT NULL ORDER BY id DESC LIMIT ?",
            (symbol, limit)
        ).fetchall()
        conn.close()
    return [{"pnl": r["pnl"], "close_time": r["close_time"]} for r in rows]


def get_symbol_direction_winrate(symbol: str, direction: str, recent_n: int = 10) -> dict:
    """종목+방향별 승률 — 신뢰도 동적 보정용"""
    side_label = "롱" if direction == "long" else "숏"
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE symbol=? AND side LIKE ? AND pnl IS NOT NULL ORDER BY id DESC LIMIT ?",
            (symbol, f"%{side_label}%", recent_n)
        ).fetchall()
        conn.close()
    if not rows:
        return {"trades": 0, "win_rate": 50.0}
    pnls = [r["pnl"] for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    return {"trades": len(pnls), "win_rate": round(wins / len(pnls) * 100, 1)}


def get_source_performance(recent_n: int = 50) -> dict:
    """소스별 성과 통계 — source 필드 기반"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT source, pnl FROM trades WHERE pnl IS NOT NULL AND source IS NOT NULL ORDER BY id DESC LIMIT ?",
            (recent_n,)
        ).fetchall()
        conn.close()
    stats = {}
    for r in rows:
        src = r["source"] or "unknown"
        if src not in stats:
            stats[src] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
        stats[src]["trades"] += 1
        stats[src]["total_pnl"] += r["pnl"]
        if r["pnl"] > 0:
            stats[src]["wins"] += 1
    for src in stats:
        s = stats[src]
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 50.0
    return stats


def get_symbol_avg_slippage(symbol: str, recent_n: int = 10) -> float:
    """종목별 평균 슬리피지 — 저유동성 종목 필터용"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT extra FROM trades WHERE symbol=? AND extra IS NOT NULL ORDER BY id DESC LIMIT ?",
            (symbol, recent_n)
        ).fetchall()
        conn.close()
    slips = []
    import json
    for r in rows:
        try:
            ex = json.loads(r["extra"])
            s = ex.get("slippage", 0)
            if s and s > 0:
                slips.append(s)
        except Exception:
            pass
    return round(sum(slips) / len(slips), 6) if slips else 0.0


def get_bad_hours(min_trades: int = 3, max_avg_loss: float = -0.5) -> list:
    """손실이 심한 시간대 목록 반환 (최소 min_trades건 이상, 평균 PnL < max_avg_loss)"""
    hourly = get_hourly_pnl()
    bad = []
    for h in hourly:
        if h["trades"] >= min_trades and (h["avg_pnl"] or 0) < max_avg_loss:
            bad.append(int(h["hour"]))
    return bad


def get_slippage_stats() -> dict:
    """슬리피지 통계 반환 — 평균, 최대, 총합, 건수"""
    trades = get_closed_trades(limit=200)
    slips = []
    for t in trades:
        s = t.get("slippage")
        if s is not None and s != 0:
            slips.append(float(s))
    if not slips:
        return {"count": 0, "avg": 0, "max": 0, "total": 0, "details": []}
    return {
        "count": len(slips),
        "avg": round(sum(slips) / len(slips), 4),
        "max": round(max(slips, key=abs), 4),
        "total": round(sum(slips), 4),
        "details": slips[-20:],  # 최근 20건
    }


def save_analysis(data: dict):
    """분석 이력 저장 (매 분석 사이클마다 호출)"""
    with _db_lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO analysis_log
            (time, symbol, decision, confidence, entry_price, sl, tp,
             rsi_15m, rsi_1h, rsi_4h, adx_15m, ema20_15m, ema50_15m,
             ema20_4h, ema50_4h, atr, macd_hist,
             btc_price, btc_trend, rl_signal, llm_reason,
             order_type, order_id, source, filled, extra)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            data.get("symbol", ""),
            data.get("decision", "wait"),
            data.get("confidence", 0),
            data.get("entry_price"),
            data.get("sl"),
            data.get("tp"),
            data.get("rsi_15m"),
            data.get("rsi_1h"),
            data.get("rsi_4h"),
            data.get("adx_15m"),
            data.get("ema20_15m"),
            data.get("ema50_15m"),
            data.get("ema20_4h"),
            data.get("ema50_4h"),
            data.get("atr"),
            data.get("macd_hist"),
            data.get("btc_price"),
            data.get("btc_trend"),
            data.get("rl_signal"),
            data.get("llm_reason"),
            data.get("order_type"),
            data.get("order_id"),
            data.get("source", "main"),
            1 if data.get("filled") else 0,
            json.dumps(data.get("extra", {}), ensure_ascii=False) if data.get("extra") else None,
        ))
        conn.commit()
        conn.close()


def get_analysis_log(symbol: str = None, limit: int = 50, decision: str = None) -> list:
    """분석 이력 조회"""
    with _db_lock:
        conn = _get_conn()
        q = "SELECT * FROM analysis_log WHERE 1=1"
        params = []
        if symbol:
            q += " AND symbol = ?"
            params.append(symbol)
        if decision:
            q += " AND decision = ?"
            params.append(decision)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_analysis_stats(symbol: str = None, days: int = 7) -> dict:
    """분석 이력 통계 (결정별 수, 체결률 등)"""
    with _db_lock:
        conn = _get_conn()
        cutoff = (datetime.now().replace(hour=0, minute=0, second=0) -
                  __import__("datetime").timedelta(days=days)).strftime("%Y-%m-%d")
        q = "SELECT decision, COUNT(*) as cnt, SUM(filled) as filled_cnt FROM analysis_log WHERE time >= ?"
        params = [cutoff]
        if symbol:
            q += " AND symbol = ?"
            params.append(symbol)
        q += " GROUP BY decision"
        rows = conn.execute(q, params).fetchall()
        conn.close()
        result = {}
        for r in rows:
            result[r["decision"]] = {"count": r["cnt"], "filled": r["filled_cnt"] or 0}
        return result


# 모듈 로드 시 테이블 자동 생성
init_db()

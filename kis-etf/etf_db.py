#!/usr/bin/env python3
# KIS-ETF 거래 기록 SQLite 저장소
import sys; sys.stdout.reconfigure(line_buffering=True)

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime

_DB_PATH = Path(__file__).parent / "etf_trades.db"
_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    """테이블 생성"""
    with _db_lock:
        conn = _get_conn()
        # 거래 기록
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                time        TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                name        TEXT,
                side        TEXT NOT NULL,
                qty         INTEGER,
                price       INTEGER,
                amount      INTEGER,
                pnl         INTEGER,
                pnl_pct     REAL,
                strategy    TEXT,
                reason      TEXT,
                extra       TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(time)")

        # 보유 종목
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                ticker      TEXT PRIMARY KEY,
                name        TEXT,
                qty         INTEGER NOT NULL,
                avg_price   INTEGER NOT NULL,
                strategy    TEXT,
                buy_time    TEXT,
                extra       TEXT
            )
        """)

        # 일일 신호 기록 (백테스트 vs 실거래 비교용)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_signal (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                strategy    TEXT NOT NULL,
                action      TEXT,
                reason      TEXT,
                price       INTEGER,
                extra       TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_date ON daily_signal(date)")

        # 잔고 히스토리
        conn.execute("""
            CREATE TABLE IF NOT EXISTS balance_history (
                date        TEXT PRIMARY KEY,
                balance     INTEGER,
                deposit     INTEGER,
                holdings_value INTEGER,
                total       INTEGER,
                pnl         INTEGER,
                trades      INTEGER DEFAULT 0,
                updated_at  TEXT
            )
        """)

        # 봇 상태 (KV 스토어)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()


def add_trade(ticker: str, name: str, side: str, qty: int, price: int,
              strategy: str = "", reason: str = "", extra: dict = None) -> int:
    """거래 기록 추가"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    amount = qty * price
    with _db_lock:
        conn = _get_conn()
        cur = conn.execute("""
            INSERT INTO trades (time, ticker, name, side, qty, price, amount, strategy, reason, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, ticker, name, side, qty, price, amount, strategy, reason,
              json.dumps(extra) if extra else None))
        conn.commit()
        trade_id = cur.lastrowid
        conn.close()
        return trade_id


def update_trade_pnl(trade_id: int, pnl: int, pnl_pct: float):
    """거래 손익 업데이트"""
    with _db_lock:
        conn = _get_conn()
        conn.execute("UPDATE trades SET pnl=?, pnl_pct=? WHERE id=?",
                      (pnl, pnl_pct, trade_id))
        conn.commit()
        conn.close()


def upsert_holding(ticker: str, name: str, qty: int, avg_price: int, strategy: str = ""):
    """보유 종목 추가/업데이트"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db_lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO holdings (ticker, name, qty, avg_price, strategy, buy_time)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                qty=excluded.qty, avg_price=excluded.avg_price,
                strategy=excluded.strategy, buy_time=excluded.buy_time
        """, (ticker, name, qty, avg_price, strategy, now))
        conn.commit()
        conn.close()


def remove_holding(ticker: str):
    """보유 종목 제거"""
    with _db_lock:
        conn = _get_conn()
        conn.execute("DELETE FROM holdings WHERE ticker=?", (ticker,))
        conn.commit()
        conn.close()


def get_holdings() -> list:
    """보유 종목 목록"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM holdings").fetchall()
        conn.close()
        return [dict(r) for r in rows]


def add_signal(date_str: str, ticker: str, strategy: str, action: str,
               reason: str = "", price: int = 0):
    """일일 신호 기록"""
    with _db_lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO daily_signal (date, ticker, strategy, action, reason, price)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date_str, ticker, strategy, action, reason, price))
        conn.commit()
        conn.close()


def get_trade_stats(days: int = 30) -> dict:
    """최근 N일 거래 통계"""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE time >= date('now', ?) AND pnl IS NOT NULL
            ORDER BY time
        """, (f"-{days} days",)).fetchall()
        conn.close()

    if not rows:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_pnl": 0}

    trades = [dict(r) for r in rows]
    wins   = [t for t in trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0) < 0]
    total_pnl = sum(t.get("pnl") or 0 for t in trades)

    return {
        "total":     len(trades),
        "wins":      len(wins),
        "losses":    len(losses),
        "win_rate":  len(wins) / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "avg_pnl":   total_pnl / len(trades) if trades else 0,
    }


def save_bot_state(key: str, value: str):
    """봇 상태 저장"""
    with _db_lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO bot_state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        conn.commit()
        conn.close()


def get_bot_state(key: str, default: str = "") -> str:
    """봇 상태 조회"""
    with _db_lock:
        conn = _get_conn()
        row = conn.execute("SELECT value FROM bot_state WHERE key=?", (key,)).fetchone()
        conn.close()
        return row["value"] if row else default


def save_balance(balance: int, deposit: int, holdings_value: int):
    """일일 잔고 기록"""
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = deposit + holdings_value
    with _db_lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO balance_history (date, balance, deposit, holdings_value, total, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                balance=excluded.balance, deposit=excluded.deposit,
                holdings_value=excluded.holdings_value, total=excluded.total,
                updated_at=excluded.updated_at
        """, (today, balance, deposit, holdings_value, total, now))
        conn.commit()
        conn.close()


# 모듈 로드 시 자동 초기화
init_db()

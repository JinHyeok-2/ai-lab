#!/usr/bin/env python3
# 바이낸스 선물거래 봇 설정

import os
from dotenv import load_dotenv

load_dotenv()

# ── API 키 (환경변수에서 로드) ─────────────────────────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ── 테스트넷 여부 ────────────────────────────────────────────────────
# True  → 테스트넷 (가상 자금, 안전)
# False → 실거래 (실제 자금 사용)
TESTNET = False

# ── 거래 설정 ────────────────────────────────────────────────────────
SYMBOLS    = ["ETHUSDT"]               # BTC 제거 (최소 notional $100 미달, $15×3=$45)
LEVERAGE   = 3                         # 레버리지 배수 (BTC/ETH 메이저)
MAX_USDT       = 30                    # 1회 최대 진입 금액 상한 (USDT) — 잔고 비례 계산의 캡
POSITION_PCT   = 12                    # 메인: 잔고의 12% 진입 (예: $139 × 12% = $16.7)
ATR_SL_MULT    = 3.0                   # 손절 = ATR × 3.0 (15분봉 노이즈 대비, 2.0→3.0)
ATR_TP_MULT    = 6.0                   # 익절 = ATR × 6.0 (R:R 1:2 유지)
MAX_DAILY_LOSS = 20                    # 일일 최대 손실 한도 (USDT) — $140 기준 보수적
MAX_SINGLE_LOSS_PCT = 3.0              # 건당 최대 손실 비율 (잔고 대비 %) — 실거래 강화

# ── 분석 설정 ────────────────────────────────────────────────────────
INTERVAL   = "15m"   # 캔들 인터벌 (1m, 5m, 15m, 1h, 4h)
CANDLE_CNT = 100     # 가져올 캔들 수

# ── 알트코인 스캐너 설정 ─────────────────────────────────────────────
MAX_USDT_ALT      = 20    # 알트 1회 최대 진입 금액 상한 (USDT) — 잔고 비례 계산의 캡
POSITION_PCT_ALT  = 8     # 알트: 잔고의 8% 진입 (예: $139 × 8% = $11.1)
MAX_ALT_POSITIONS = 3     # 알트 동시 최대 포지션 수 (7→3, 리스크 집중 관리)
ALT_LEVERAGE      = 2     # 알트 레버리지 (Isolated, 최대 2배)
ALT_ATR_SL_MULT   = 2.0  # 알트 손절 배수 (1.0→2.0, 변동성 대비)
ALT_ATR_TP_MULT   = 4.0  # 알트 익절 배수 (R:R 1:2 유지)
ALT_SCAN_LIMIT    = 50   # 스크리너 유니버스 크기 (거래량 상위 N개)
ALT_MIN_SCORE          = 50   # 분석 진행 최소 스크리너 점수 (30→50, LLM 호출 절감)
ALT_AUTO_CONFIDENCE    = 65   # 자동 주문 발동 최소 신뢰도 (%) (60→65 복원, 65미만 전패 데이터)
ALT_MANUAL_MIN_CONF    = 40   # 수동 주문 버튼 활성화 최소 신뢰도 (%) (30→40)

# ── 누적 성과 기반 스케일업 테이블 ──────────────────────────────────
# (최소 거래수, 최소 승률%) → 진입금 배율
# 거래수 + 승률 모두 충족해야 적용, 위에서부터 검사하므로 높은 조건이 먼저
SCALE_TABLE = [
    (100, 45, 3.0),   # 100건+ 승률45%+ → 3배
    (50,  40, 2.0),   # 50건+  승률40%+ → 2배
    (20,  35, 1.5),   # 20건+  승률35%+ → 1.5배
]

# ── 에이전트 모델 ────────────────────────────────────────────────────
ANALYST_MODEL = "sonnet"
NEWS_MODEL    = "sonnet"
RISK_MODEL    = "sonnet"
TRADER_MODEL  = "sonnet"
GATE_MODEL    = "haiku"

# ── 텔레그램 알림 (환경변수에서 로드) ──────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

#!/usr/bin/env python3
# 바이낸스 선물거래 봇 설정

# ── API 키 ──────────────────────────────────────────────────────────
API_KEY    = "IZABohzjh1kDy66OFjCbQtzguaimEGXhTFT37wWY358vGxhznGeFO8Obg29VY7xT"
API_SECRET = "vnxPbEVGXpjkxUB5ZG9D4RBDMGvDMr7bqErVjtvShKV9OYxfIBSZDGtJM1z8lQ7C"

# ── 테스트넷 여부 ────────────────────────────────────────────────────
# True  → 테스트넷 (가상 자금, 안전)
# False → 실거래 (실제 자금 사용)
TESTNET = True

# ── 거래 설정 ────────────────────────────────────────────────────────
SYMBOLS    = ["ETHUSDT", "BTCUSDT"]   # 감시할 코인 (ETH 기본)
LEVERAGE   = 3                         # 레버리지 배수
MAX_USDT       = 100                   # 1회 최대 진입 금액 (USDT)
ATR_SL_MULT    = 1.5                   # 손절 = ATR × 1.5
ATR_TP_MULT    = 3.0                   # 익절 = ATR × 3.0  (R:R 1:2)
MAX_DAILY_LOSS = 200                   # 일일 최대 손실 한도 (USDT) — 초과 시 거래 중단

# ── 분석 설정 ────────────────────────────────────────────────────────
INTERVAL   = "15m"   # 캔들 인터벌 (1m, 5m, 15m, 1h, 4h)
CANDLE_CNT = 100     # 가져올 캔들 수

# ── 알트코인 스캐너 설정 ─────────────────────────────────────────────
MAX_USDT_ALT      = 50    # 알트 1회 최대 진입 금액 (USDT)
MAX_ALT_POSITIONS = 2     # 알트 동시 최대 포지션 수
ALT_LEVERAGE      = 2     # 알트 레버리지 (Isolated, 최대 2배)
ALT_ATR_SL_MULT   = 1.0  # 알트 손절 배수 (타이트)
ALT_ATR_TP_MULT   = 2.0  # 알트 익절 배수
ALT_SCAN_LIMIT    = 50   # 스크리너 유니버스 크기 (거래량 상위 N개)
ALT_MIN_SCORE          = 30   # 분석 진행 최소 스크리너 점수
ALT_AUTO_CONFIDENCE    = 75   # 자동 주문 발동 최소 신뢰도 (%)
ALT_MANUAL_MIN_CONF    = 60   # 수동 주문 버튼 활성화 최소 신뢰도 (%)

# ── 에이전트 모델 ────────────────────────────────────────────────────
ANALYST_MODEL = "sonnet"
NEWS_MODEL    = "sonnet"
RISK_MODEL    = "sonnet"
TRADER_MODEL  = "sonnet"

# ── 텔레그램 알림 ────────────────────────────────────────────────────
# @BotFather 에서 봇 생성 후 토큰 발급
# Chat ID: @userinfobot 에 /start 보내면 확인 가능
TELEGRAM_TOKEN   = "8690641013:AAEKBcZbMA4Kl-HLXO_5Zh1cLfk3k31l-W4"
TELEGRAM_CHAT_ID = "1765732154"

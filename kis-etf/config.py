#!/usr/bin/env python3
# KIS-ETF 자동매매 봇 설정
import sys; sys.stdout.reconfigure(line_buffering=True)

import os
from pathlib import Path
from dotenv import load_dotenv

# .env는 상위 디렉토리(ai-lab) 또는 현재 디렉토리에서 로드
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

# ── KIS API 키 ────────────────────────────���──────────────────────────
KIS_APP_KEY     = os.getenv("KIS_APP_KEY", "")           # 실전 (시세 조회용)
KIS_APP_SECRET  = os.getenv("KIS_APP_SECRET", "")
KIS_VIRTUAL_APP_KEY    = os.getenv("KIS_VIRTUAL_APP_KEY", "")  # 모의 (주문용)
KIS_VIRTUAL_APP_SECRET = os.getenv("KIS_VIRTUAL_APP_SECRET", "")
KIS_ACCOUNT_NO  = os.getenv("KIS_ACCOUNT_NO", "")       # 모의투자 계좌
KIS_HTS_ID      = os.getenv("KIS_HTS_ID", "")           # HTS 로그인 ID
IS_VIRTUAL      = os.getenv("KIS_IS_VIRTUAL", "true").lower() == "true"

# ── 텔레그램 알림 ────────────────────────────────────────────────────
ETF_TG_TOKEN    = os.getenv("ETF_TELEGRAM_TOKEN", "")
ETF_TG_CHAT_ID  = os.getenv("ETF_TELEGRAM_CHAT_ID", "")

# ── 투자 설정 ────────────────────────────────���───────────────────────
INITIAL_CAPITAL    = 1_000_000   # 초기 자본금 (원)
MAX_POSITION_PCT   = 20          # 종목당 최대 자본 비율 (%)
MAX_POSITIONS      = 5           # 동시 보유 최대 종목 수
MAX_DAILY_BUYS     = 3           # 일일 최대 매수 건수
FEE_RATE           = 0.00015     # ETF 매매 수수료 편도 (0.015%)
TAX_RATE           = 0.0         # ETF 매도 시 거래세 없음

# ── ETF 유니버스 ─────────────────────────────────────────────────────
# 국내 주요 ETF (티커: 이름)
ETF_UNIVERSE = {
    "069500": "KODEX 200",
    "102110": "TIGER 200",
    "148070": "KOSEF 국고채10년",
    "132030": "KODEX 골드선물(H)",
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
    "229200": "KODEX 코스닥150",
    "305720": "KODEX 2차전지산업",
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
}

# ── 자산배분 기본 비중 ───────────────────────────────────────────────
ASSET_ALLOCATION = {
    "069500": 0.60,   # 주식 60%
    "148070": 0.30,   # 채권 30%
    "132030": 0.10,   # 금 10%
}

# ── 듀얼 모멘텀 종목 ────────────────────────────────────────────────
MOMENTUM_EQUITY  = "069500"   # KODEX 200 (공격 자산)
MOMENTUM_BOND    = "148070"   # 국고채10년 (방어 자산)
MOMENTUM_LOOKBACK = 252       # 12개월 영업일

# ── 전략 파라미터 ───────────────────────────────────���────────────────
MA_SHORT   = 5     # 이동평균 단기
MA_LONG    = 20    # 이동평균 장기
BB_PERIOD  = 20    # 볼린저밴드 기간
BB_STD     = 2.0   # 볼린저밴드 표준편차
RSI_PERIOD = 14    # RSI 기간
RSI_LOWER  = 30    # RSI 매수 기준
RSI_UPPER  = 70    # RSI 매도 기준

# ── 토큰 캐시 경로 ────────────────────────────────��──────────────────
TOKEN_CACHE = Path(__file__).parent / ".kis_token.json"

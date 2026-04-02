#!/usr/bin/env python3
# 5분마다 자동 종목 스캔 → 기술적 분석 → 지정가 예약 업데이트
# LIMIT만 배치, 체결 시 SL/TP 자동 배치 (conditional 잔존 방지)

import sys
sys.path.insert(0, '/home/hyeok/01.APCC/00.ai-lab')

import os
import time
import math
import json
import uuid
import sqlite3
from datetime import datetime, timezone, timedelta
from binance_client import (get_klines, get_price, get_balance, get_positions,
                            get_client, place_sl_tp, _get_symbol_filters, _round_price,
                            set_margin_type, set_leverage)
from indicators import calc_indicators
from telegram_notifier import send_message
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from alt_scanner import get_alt_futures_symbols, check_upbit_announcements, check_okx_announcements, check_coinbase_listings
import numpy as np
import pandas_ta as ta
from signal_queue import pop_signals
import trade_db

# ── 설정 ──
INTERVAL = 300          # 5분
MAX_ORDERS = 2          # 4→2 (보수적: 집중 투자)
ETH_USDT = 30           # ETH 진입금 (25→30, 1%/일 목표)
ALT_USDT = 30           # 알트 진입금 (25→30, BB 건당 수익 강화)
ALT_USDT_MAX = 60       # 노셔널 상한 (ETH $20×3x=60, ALT $20×2x=40)
ETH_LEV = 3             # ETH 레버리지
ALT_LEV = 2             # 알트 레버리지
MIN_SCORE = 6           # 최소 진입 점수 (5→6: 그리드서치 1위, 노이즈 제거 강화)
MAX_SL_PCT = 5.0        # SL 최대 거리 (%)
MAX_LOSS_PER_TRADE = 1.0  # #141 건당 최대 손실 캡 $1.0 (수량 조절 방식)
MAX_SAME_DIR = 4        # 동일 방향 최대 (2→4, 롱전용이므로 MAX_ORDERS와 동일)
MAX_DAILY_TRADES = 4    # 3→4 (BB 기회 확보, 1%/일 목표)
NIGHT_HOURS = {5}  # KST 05시 차단 — 13건 손실/큰 손실 5건 집중 (전 시간대 최악)
BLACKLIST = {'BRUSDT', 'SIRENUSDT', 'XAUUSDT', 'XAGUSDT', 'RIVERUSDT', 'SIGNUSDT', 'PAXGUSDT', 'BSBUSDT',
             'A2ZUSDT', 'PTBUSDT', 'VIDTUSDT', 'MEMEFIUSDT', 'AMBUSDT', 'TROYUSDT', 'LEVERUSDT', 'NOMUSDT',
             'PORT3USDT', 'NEIROETHUSDT', 'BSWUSDT', 'AGIXUSDT', 'SXPUSDT',
             'ALPACAUSDT', 'BNXUSDT', 'ALPHAUSDT', 'LINAUSDT'}  # + get_price 에러
# 성과 기반 위험 종목 (승률 0~33%) — BB/CVD 진입 제외
WEAK_SYMBOLS = {'ETHUSDT', 'DOGEUSDT', 'ONTUSDT', 'TAOUSDT'}  # ETH 0%, DOGE 0%, ONT 0%, TAO 33%
TG_TOKEN = TELEGRAM_TOKEN
TG_CHAT  = TELEGRAM_CHAT_ID

# 스캔 대상 (ETH 고정 + 거래량 상위 동적 선택)
SCAN_FIXED = ['ETHUSDT']  # 항상 포함
SCAN_ALT_COUNT = 30       # 알트 동적 선택 수 (25→30, CVD 중형 종목 기회 확대)
_scan_cache = {'ts': 0, 'symbols': []}

LOG_PATH = '/home/hyeok/01.APCC/00.ai-lab/position_updater.log'
LOCK_PATH = '/home/hyeok/01.APCC/00.ai-lab/updater_managed.json'

# 체결 대기 추적
_pending_fills = {}
# SL/TP 배치 완료 종목 (재시작 시 1회만 보완)
_sltp_done = set()
# 포지션별 TP 캐시 (부분 익절에서 DB 조회 대신 사용)
_tp_cache = {}  # {symbol: tp_price}
# 트레일링 스탑: 최고/최저 가격 추적
_trail_peak = {}  # {symbol: best_price_so_far}
_trail_atr = {}   # {symbol: ATR(1h)} — 트레일링 활성화 시 캐시
TRAIL_ACTIVATE_PCT = 0.8  # 수익 0.8%부터 트레일링 활성화 (BB롱 중앙값 0.7% 커버)
# #159 상장 숏 설정
LISTING_SHORT_USDT = 10     # 소액 $10
LISTING_SHORT_LEV = 2       # 레버리지 2x
LISTING_SHORT_SL_PCT = 5.0  # SL: 진입 대비 +5% (레버리지 반영 10% 손실 = $1)
LISTING_SHORT_TP_PCT = 5.0  # TP: 진입 대비 -5% (레버리지 반영 10% 수익 = $1)
# ATR 비례 트레일링 (수익 단계별 ATR 배수)
TRAIL_ATR_TIERS = [
    (5.0, 0.2),   # 수익 5%+ → ATR × 0.2 (타이트, 수익 확보)
    (3.0, 0.3),   # 수익 3~5% → ATR × 0.3 (부분 익절 후 구간)
    (1.5, 0.5),   # 수익 1.5~3% → ATR × 0.5
    (0.8, 0.7),   # 수익 0.8~1.5% → ATR × 0.7 (BB롱 수익권 보호, 넓게)
]
TRAIL_MIN_PCT = 0.5   # 최소 트레일 거리 0.5% (0.3%는 수수료 포함 시 본전)
TRAIL_MAX_PCT = 3.0   # 최대 트레일 거리 3.0% (극변동 종목 캡)
# 부분 익절 완료 종목 (사이클마다 중복 방지)
_partial_done = set()
# 종목별 쿨다운 (청산 후 30분간 재진입 방지)
COOLDOWN_SEC = 1800  # 30분
COOLDOWN_LOSS_SEC = 7200  # 동일 종목 손실 시 2시간 쿨다운
_cooldown = {}  # {symbol: expire_time}
# BTC 지표 캐시 (사이클당 1회만 조회)
_btc_cache = {'ts': 0, 'up': True, 'rsi': 50, 'rsi_prev3': 50}
# 텔레그램 전송 주기 (분석은 5분, 알림은 30분)
TG_INTERVAL = 1800  # 30분
_last_tg_time = 0
# 시장 모드 (#4: 자동 판정)
_bear_mode = False
_bull_mode = False
_market_mode_ts = 0  # 마지막 판정 시각
# #159 상장 숏 — 피크 대기 큐: {symbol: {detected_at, peak_price, peak_rsi, source}}
_listing_watch = {}
_listing_done = set()  # 이미 숏 진입한 종목 (세션 내 중복 방지)
# 일일 거래 횟수 + 연속 손실 추적
_daily_trades = {'date': '', 'count': 0}
_consecutive_losses = 0
_global_cooldown_until = 0  # 연패 쿨다운 만료 시각 (time.time)
# #J/K: 하락장 전용 안전장치
_bear_daily_loss = {'date': '', 'total': 0.0}  # 하락장 일일 손실 누적
_bear_stopped = False  # 하락장 당일 거래 정지 플래그
# 최소 보유시간 (10분 미만 트레일링 청산 방지)
MIN_HOLD_SEC = 600  # 10분
_entry_time = {}  # {symbol: time.time()}
_entry_source = {}  # {symbol: 'cvd_divergence' 등} — 전략별 보유시간 차등

# ── 볼린저 박스 왕복 전략 ──
_bb_box_cache = {'ts': 0}
_bb_limit_orders = {}  # {symbol: {'order_id': ..., 'price': ..., 'qty': ..., 'sl': ..., 'tp': ...}}
BB_BOX_LOG = '/home/hyeok/01.APCC/00.ai-lab/bb_box_signals.jsonl'
# ── BB 숏 (상단 매도) ──
_bb_short_cache = {'ts': 0}
_bb_short_limit_orders = {}  # {symbol: {'order_id': ..., ...}}
_bb_short_cooldown = {}  # {symbol: exit_timestamp} — 동일 종목 4시간 재진입 방지
BB_SHORT_USDT = 20  # 롱보다 보수적
BB_SHORT_LOG = '/home/hyeok/01.APCC/00.ai-lab/bb_short_signals.jsonl'
# ── 추세 추종 숏 (하락장) ──
_trend_short_cache = {'ts': 0}
TREND_SHORT_USDT = 20
TREND_SHORT_LOG = '/home/hyeok/01.APCC/00.ai-lab/trend_short_signals.jsonl'
# ── 역행 과매수 숏 (하락장에서 혼자 오른 알트) ──
_contrarian_short_cache = {'ts': 0}
CONTRARIAN_SHORT_USDT = 35
CONTRARIAN_SHORT_LOG = '/home/hyeok/01.APCC/00.ai-lab/contrarian_short_signals.jsonl'
# ── 모멘텀 브레이크아웃 (중립장 추세 초기 롱) ──
_momentum_cache = {'ts': 0}
MOMENTUM_USDT = 15  # 소액 검증 ($15)
MOMENTUM_LOG = '/home/hyeok/01.APCC/00.ai-lab/momentum_signals.jsonl'
# ── 급등 과매수 숏 (24h 급등 종목 역행) ──
_surge_short_cache = {'ts': 0}
SURGE_SHORT_USDT = 20  # STO +$6.10 실증 → $20
SURGE_SHORT_LOG = '/home/hyeok/01.APCC/00.ai-lab/surge_short_signals.jsonl'
# ── 메가 급등 실시간 캐치 (30초 주기, 24h +80%+) ──
_surge_entered_today = set()  # 당일 급등숏 진입 종목 (중복 방지)
_quick_surge_date = ''        # 날짜 바뀌면 리셋

# ── 킬존 시간대 부스트 (런던/뉴욕 오픈 = 변동성↑) ──
def _get_killzone_boost():
    """UTC 기반 킬존 진입금 배수. 런던(07~10 UTC)/뉴욕(12~15 UTC) = 1.2x"""
    _utc_h = datetime.now(timezone.utc).hour
    if 7 <= _utc_h < 10 or 12 <= _utc_h < 15:  # 킬존
        return 1.2
    return 1.0

# ── 기관 매매법 전략 변수 ──
# #L: CVD 다이버전스
_cvd_cache = {'ts': 0}  # 5분 캐시
CVD_LOG = '/home/hyeok/01.APCC/00.ai-lab/cvd_signals.jsonl'
# #M: OI + 롱숏비율 콤보 (숏 스퀴즈)
_squeeze_cache = {'ts': 0}
SQUEEZE_LOG = '/home/hyeok/01.APCC/00.ai-lab/squeeze_signals.jsonl'
# #N: MTF 컨플루언스 (멀티 타임프레임 과매도)
_mtf_cache = {'ts': 0}
MTF_LOG = '/home/hyeok/01.APCC/00.ai-lab/mtf_signals.jsonl'
# #O: VWAP 평균회귀
_vwap_cache = {'ts': 0}
VWAP_LOG = '/home/hyeok/01.APCC/00.ai-lab/vwap_signals.jsonl'
# #P: 거래량 프로파일 (POC)
_vpoc_cache = {'ts': 0}
VPOC_LOG = '/home/hyeok/01.APCC/00.ai-lab/vpoc_signals.jsonl'


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def get_scan_universe():
    """ETH 고정 + 거래량 상위 알트 동적 선택 (10분 캐시)"""
    now = time.time()
    if now - _scan_cache['ts'] < 600 and _scan_cache['symbols']:
        return _scan_cache['symbols']
    try:
        alts = get_alt_futures_symbols(SCAN_ALT_COUNT)
        alt_syms = [s for s in alts if s not in SCAN_FIXED][:SCAN_ALT_COUNT]
        result = SCAN_FIXED + alt_syms
        _scan_cache.update({'ts': now, 'symbols': result})
        return result
    except Exception:
        return _scan_cache.get('symbols', SCAN_FIXED)


def get_btc_trend():
    """BTC 1H 추세 (사이클당 1회 캐시)"""
    now = time.time()
    if now - _btc_cache['ts'] < 250:  # 4분 캐시
        return _btc_cache['up'], _btc_cache['rsi']
    try:
        df = get_klines('BTCUSDT', '1h', 60)
        btc = calc_indicators(df)
        up = (btc.get('ema20', 0) or 0) > (btc.get('ema50', 0) or 0)
        rsi = btc.get('rsi', 50) or 50
        # BTC RSI 3캔들 전 값 (방향성 판단용, API 호출 없음)
        try:
            import pandas_ta as ta
            _rsi_s = ta.rsi(df['close'], length=14)
            rsi_prev3 = float(_rsi_s.iloc[-4]) if _rsi_s is not None and len(_rsi_s) >= 4 else rsi
        except Exception:
            rsi_prev3 = rsi
        _btc_cache.update({'ts': now, 'up': up, 'rsi': rsi, 'rsi_prev3': rsi_prev3})
        return up, rsi
    except Exception:
        return _btc_cache['up'], _btc_cache['rsi']


def update_market_mode():
    """시장 모드 자동 판정 — 🐻하락/⚪일반/🐂상승"""
    global _bear_mode, _bull_mode, _market_mode_ts
    now = time.time()
    if now - _market_mode_ts < 300:  # 5분 캐시
        return
    _market_mode_ts = now
    btc_up, btc_rsi = get_btc_trend()
    old_bear, old_bull = _bear_mode, _bull_mode

    _bear_mode = (btc_rsi < 35 and not btc_up)
    _bull_mode = (btc_rsi > 60 and btc_up)

    if _bear_mode != old_bear or _bull_mode != old_bull:
        if _bear_mode:
            mode_str = "🐻 하락장 모드 ON"
        elif _bull_mode:
            mode_str = "🐂 상승장 모드 ON"
        else:
            mode_str = "⚪ 일반 모드"
        ema_str = '정배열' if btc_up else '역배열'
        log(f"  {mode_str} (BTC RSI={btc_rsi:.0f}, EMA={ema_str})")
        # 시장 모드 알림 OFF — 로그만


# 파생지표 캐시 (10분 TTL)
_sentiment_cache = {'ts': 0, 'data': {}}

def _get_sentiment_bonus(symbol):
    """펀딩비 + 롱숏비율 + OI 역발상 보너스 (±5점, 10분 캐시)"""
    now = time.time()
    if now - _sentiment_cache['ts'] > 600:
        # 캐시 만료 → 갱신
        _sentiment_cache['ts'] = now
        _sentiment_cache['data'] = {}

    if symbol in _sentiment_cache['data']:
        return _sentiment_cache['data'][symbol]

    bonus = 0
    try:
        client = get_client()

        # 1. 펀딩비 역발상 (±2점)
        fr = client.futures_funding_rate(symbol=symbol, limit=1)
        if fr:
            rate = float(fr[-1]['fundingRate'])
            if rate > 0.0005: bonus -= 2      # 롱 과열 → 숏 유리
            elif rate > 0.0001: bonus -= 1
            elif rate < -0.0005: bonus += 2    # 숏 과열 → 롱 유리
            elif rate < -0.0001: bonus += 1

        # 2. 롱숏비율 역발상 (±2점)
        try:
            ls = client.futures_top_longshort_account_ratio(symbol=symbol, period='1h', limit=1)
            if ls:
                long_pct = float(ls[-1]['longAccount'])
                if long_pct > 0.65: bonus -= 2    # 개인 과도 롱 → 숏 유리
                elif long_pct > 0.58: bonus -= 1
                elif long_pct < 0.35: bonus += 2   # 개인 과도 숏 → 롱 유리
                elif long_pct < 0.42: bonus += 1
        except Exception:
            pass

        # 3. OI 변화율 (±1점)
        try:
            oih = client.futures_open_interest_hist(symbol=symbol, period='1h', limit=2)
            if len(oih) >= 2:
                oi_now = float(oih[-1]['sumOpenInterestValue'])
                oi_prev = float(oih[-2]['sumOpenInterestValue'])
                oi_change = (oi_now - oi_prev) / oi_prev * 100 if oi_prev > 0 else 0
                if oi_change > 3: bonus += 1      # OI 급증 → 추세 강화
                elif oi_change < -3: bonus -= 1    # OI 급감 → 자금 이탈
        except Exception:
            pass
    except Exception:
        pass

    _sentiment_cache['data'][symbol] = bonus
    return bonus


# ── RL 알트 범용모델 (#144) ──
_rl_model = None
_rl_loaded = False

def _load_rl_model():
    """ALT 앙상블 RL 모델 로드 (3모델 다수결, 1회만)"""
    global _rl_model, _rl_loaded
    if _rl_loaded:
        return _rl_model
    _rl_loaded = True
    try:
        from stable_baselines3 import PPO
        from pathlib import Path
        base = Path('/home/hyeok/01.APCC/00.ai-lab/rl-lab/models')
        alt_paths = {
            "alt_exp01": base / "alt_universal_exp01/ppo_alt.zip",
            "alt_seed200": base / "alt_v2_seed200/ppo_alt.zip",
            "alt_seed700": base / "alt_v2_seed700/ppo_alt.zip",
        }
        if all(p.exists() for p in alt_paths.values()):
            _rl_model = {k: PPO.load(str(p)) for k, p in alt_paths.items()}
            log(f"  RL 알트 앙상블 로드 (3모델 다수결)")
        else:
            # 폴백: 단일 모델
            p = base / "alt_universal_exp01/ppo_alt.zip"
            if p.exists():
                _rl_model = {"alt_exp01": PPO.load(str(p))}
                log("  RL 알트 단일모델 폴백")
            else:
                log("  RL 알트 모델 파일 없음")
    except Exception as e:
        log(f"  RL 로드 실패: {str(e)[:40]}")
    return _rl_model


_rl_cache = {}  # {symbol: {'ts': time, 'signal': str, 'bonus': int}}

def get_rl_signal_lite(symbol):
    """RL 알트 신호 (경량, 60초 캐시)
    return: ('long'|'wait'|'close', bonus_점수)"""
    now = time.time()
    if symbol in _rl_cache and now - _rl_cache[symbol]['ts'] < 60:
        c = _rl_cache[symbol]
        return c['signal'], c['bonus']

    model = _load_rl_model()
    if model is None:
        return 'wait', 0

    try:
        df = get_klines(symbol, '30m', 120)
        if len(df) < 60:
            return 'wait', 0

        # v5 피처 13개 계산
        df['price_chg'] = df['close'].pct_change().fillna(0).clip(-0.1, 0.1)
        df['rsi'] = ta.rsi(df['close'], length=14).fillna(50)
        df['rsi_norm'] = df['rsi'] / 100.0
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd_norm'] = (macd_df['MACD_12_26_9'] / df['close']).fillna(0).clip(-0.05, 0.05)
        bb = ta.bbands(df['close'], length=20, std=2)
        col_u = next(c for c in bb.columns if c.startswith('BBU'))
        col_l = next(c for c in bb.columns if c.startswith('BBL'))
        df['bb_pct'] = ((df['close'] - bb[col_l]) / (bb[col_u] - bb[col_l])).fillna(0.5).clip(0, 1)
        ema20 = ta.ema(df['close'], length=20)
        ema50 = ta.ema(df['close'], length=50)
        df['ema_ratio'] = (ema20 / ema50 - 1).fillna(0).clip(-0.1, 0.1)
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_norm'] = (atr / df['close']).fillna(0).clip(0, 0.1)
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(20).mean()).fillna(1.0).clip(0, 5) / 5.0
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx_norm'] = (adx_df['ADX_14'] / 100.0).fillna(0) if adx_df is not None and 'ADX_14' in adx_df.columns else 0.0
        df['price_chg_1h'] = df['close'].pct_change(2).fillna(0).clip(-0.1, 0.1) / 0.1
        price_dir = df['close'].diff(5).apply(lambda x: 1 if x > 0 else -1)
        rsi_dir = df['rsi'].diff(5).apply(lambda x: 1 if x > 0 else -1)
        df['rsi_diverge'] = (price_dir != rsi_dir).astype(float)
        stoch = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
        df['stoch_rsi'] = (stoch['STOCHRSIk_14_14_3_3'] / 100.0).fillna(0.5).clip(0, 1) if stoch is not None and 'STOCHRSIk_14_14_3_3' in stoch.columns else 0.5
        obv = (df['volume'] * np.where(df['close'].diff() > 0, 1, -1)).cumsum()
        obv_std = obv.rolling(20).std().replace(0, np.nan).fillna(1)
        df['obv_slope'] = (obv.diff(5) / obv_std).fillna(0).clip(-3, 3) / 3.0
        df['vol_regime'] = atr.rolling(100).rank(pct=True).fillna(0.5)

        df = df.dropna().reset_index(drop=True)
        if len(df) < 20:
            return 'wait', 0

        feat_cols = ['price_chg', 'rsi_norm', 'macd_norm', 'bb_pct', 'ema_ratio',
                     'atr_norm', 'vol_ratio', 'adx_norm', 'price_chg_1h', 'rsi_diverge',
                     'stoch_rsi', 'obv_slope', 'vol_regime']

        # 포지션 상태 (간이 — 현재 보유 여부만)
        pos_val, upnl_val, hold_val, cd_val = 0.0, 0.0, 0.0, 0.0
        if has_position(symbol):
            pos_val = 1.0

        rows = []
        for i in range(len(df) - 20, len(df)):
            row = [float(df[c].iloc[i]) for c in feat_cols]
            row += [pos_val, upnl_val, hold_val, cd_val]
            rows.append(row)

        obs = np.array(rows, dtype=np.float32).flatten()

        # 앙상블 다수결 투표
        from collections import Counter as _Ctr
        models = model if isinstance(model, dict) else {"m": model}
        votes = []
        for k, m in models.items():
            a, _ = m.predict(obs, deterministic=True)
            votes.append(int(a))
        cnt = _Ctr(votes)
        maj = cnt.most_common(1)[0]
        action = maj[0] if maj[1] >= len(models) // 2 + 1 else 0

        # 0=관망, 1=롱, 2=청산
        signal_map = {0: 'wait', 1: 'long', 2: 'close'}
        signal = signal_map.get(action, 'wait')
        unanimous = len(set(votes)) == 1
        bonus = (5 if unanimous else 3) if action == 1 else (-3 if action == 0 else 0)

        _rl_cache[symbol] = {'ts': now, 'signal': signal, 'bonus': bonus}
        return signal, bonus
    except Exception:
        return 'wait', 0


def score_symbol(symbol):
    """멀티TF 기술적 분석 + 파생지표 → 점수 + 메타데이터"""
    # BTC 4h 수익률 캐시 초기화 (함수 속성)
    if not hasattr(score_symbol, '_btc_4h_cache'):
        score_symbol._btc_4h_cache = 0
        score_symbol._btc_4h_ts = 0
    try:
        px = get_price(symbol)
        i15 = calc_indicators(get_klines(symbol, '15m', 60))
        i1h = calc_indicators(get_klines(symbol, '1h', 60))
        i4h = calc_indicators(get_klines(symbol, '4h', 60))
        btc_up, _ = get_btc_trend()

        # 데이터 부족 종목 필터 (RSI/ADX가 nan이면 스킵)
        _rsi_check = i15.get('rsi', 0) or 0
        _adx_check = i15.get('adx', 0) or 0
        if (isinstance(_rsi_check, float) and math.isnan(_rsi_check)) or \
           (isinstance(_adx_check, float) and math.isnan(_adx_check)):
            return {'symbol': symbol, 'score': 0, 'direction': 'wait', 'error': 'nan_data'}

        score = 0

        # 1. EMA 배열 (15m/1h/4h — 4h 가중치 강화)
        for lb, ind in [('15m', i15), ('1h', i1h), ('4h', i4h)]:
            e20 = ind.get('ema20', 0) or 0
            e50 = ind.get('ema50', 0) or 0
            if e20 and e50:
                w = 5 if lb == '4h' else 2  # 4h: 3→5 (추세 종목 캐치)
                score += w if e20 > e50 else -w

        # 1-1. #155 단기 EMA 9/21 크로스 (15m 데이터 재활용, API 추가 호출 없음)
        _e9 = i15.get('ema9', None)
        _e21 = i15.get('ema21', None)
        # indicators.py에 ema9/21이 없으면 ema20 근사로 대체
        if _e9 is not None and _e21 is not None:
            if _e9 > _e21:
                score += 2  # 단기 골든크로스
            else:
                score -= 1
        else:
            # ema20이 현재가보다 아래면 단기 상승 추세로 간주
            _e20_15 = i15.get('ema20', 0) or 0
            if _e20_15 and px > _e20_15 * 1.002:
                score += 1  # 가격이 EMA20 위 = 약한 상승

        # 2. RSI
        rsi = i15.get('rsi', 50) or 50
        if rsi < 35: score += 3
        elif rsi < 45: score += 1
        elif rsi > 65: score -= 3
        elif rsi > 55: score -= 1
        rsi1h = i1h.get('rsi', 50) or 50
        if rsi1h < 40: score += 2
        elif rsi1h > 60: score -= 2

        # 2-1. 눌림 구간 보너스 (15m RSI 40~60 = 과매수 아닌 진입 적기)
        if 40 <= rsi <= 60:
            score += 2  # 눌림에서 진입 우대

        # 2-2. #156 StochRSI 확인 (커뮤니티 전략: K>D = 모멘텀 상승)
        _stoch_k = i15.get('stoch_k', None)
        _stoch_d = i15.get('stoch_d', None)
        if _stoch_k is not None and _stoch_d is not None:
            if _stoch_k > _stoch_d and _stoch_k < 80:  # K>D + 과매수 아님
                score += 2
            elif _stoch_k < _stoch_d and _stoch_k > 20:  # K<D + 과매도 아님
                score -= 1

        # 3. MACD
        if (i15.get('macd_hist', 0) or 0) > 0: score += 1
        else: score -= 1
        if (i1h.get('macd_hist', 0) or 0) > 0: score += 2
        else: score -= 2

        # 4. ADX (추세 강도)
        adx = i15.get('adx', 0) or 0
        adx_1h = i1h.get('adx', 0) or 0
        if adx < 15:
            score = int(score * 0.5)  # 횡보 감점
        if adx_1h >= 25:
            score += 2  # 1h ADX 25+ = 추세 확인 보너스

        # 5. BTC 동조 (#152: ±2→±1, 상관계수 0.117로 영향 미미 확인)
        if btc_up: score += 1
        else: score -= 1

        # 5-1. #153 BTC 대비 상대 강도 (상관 0.784 — 최강 지표)
        # BTC 4h 수익률 캐시 (사이클당 1회)
        _rs_bonus = 0
        try:
            if not hasattr(score_symbol, '_btc_4h_cache') or time.time() - score_symbol._btc_4h_ts > 250:
                _btc_kl = get_client().futures_klines(symbol='BTCUSDT', interval='1h', limit=6)
                if len(_btc_kl) >= 5:
                    score_symbol._btc_4h_cache = (float(_btc_kl[-1][4]) - float(_btc_kl[-5][4])) / float(_btc_kl[-5][4]) * 100
                else:
                    score_symbol._btc_4h_cache = 0
                score_symbol._btc_4h_ts = time.time()

            # 알트 4h 수익률 (이미 로드한 1h 데이터의 close에서 계산)
            _alt_close_now = i1h.get('close', px) or px
            pass  # 미사용 _alt_closes 제거
            try:
                _df1h = get_klines(symbol, '1h', 6)
                if len(_df1h) >= 5:
                    _alt_4h = (float(_df1h['close'].iloc[-1]) - float(_df1h['close'].iloc[-5])) / float(_df1h['close'].iloc[-5]) * 100
                else:
                    _alt_4h = 0
            except:
                _alt_4h = 0

            _rs = _alt_4h - score_symbol._btc_4h_cache
            if _rs >= 2.0:
                _rs_bonus = 5
            elif _rs >= 0:
                _rs_bonus = 3
            elif _rs >= -2.0:
                _rs_bonus = -2
            else:
                _rs_bonus = -5
            score += _rs_bonus
        except:
            pass

        # 6. 파생지표 역발상 (펀딩비 + 롱숏비율 + OI)
        sentiment_bonus = _get_sentiment_bonus(symbol)
        score += sentiment_bonus

        # 7. 지지/저항 보너스 (±2점)
        bb_upper = i1h.get('bb_upper', 0) or 0
        bb_lower = i1h.get('bb_lower', 0) or 0
        bb_pct = 50  # 기본값
        bb_squeeze = False

        if bb_upper and bb_lower and bb_upper > bb_lower:
            bb_range = bb_upper - bb_lower
            bb_pct = (px - bb_lower) / bb_range * 100
            # 롱: BB 하단 30% 이내 → 지지선 근처 (+2)
            if bb_pct < 30:
                score += 2
            # 숏: BB 상단 30% 이내 → 저항선 근처 (-2)
            elif bb_pct > 70:
                score -= 2
            # 박스 중간 (40~60%) → 방향성 약함, 감점
            elif 40 < bb_pct < 60:
                score = int(score * 0.7)

            # 8. 볼린저 스퀴즈 감지 + 돌파 방향 (#157)
            bb_width_pct = bb_range / px * 100
            if bb_width_pct < 1.5:
                bb_squeeze = True
                # #157: 스퀴즈 + 돌파 방향 확인
                if px > bb_upper * 0.998:
                    score = int(score * 1.5)  # 상단 돌파 = 강한 롱 신호
                elif px < bb_lower * 1.002:
                    score = int(score * 0.5)  # 하단 돌파 = 약세
                else:
                    score = int(score * 1.1)  # 아직 돌파 전 = 약한 증폭

        # 9. 거래량 확인 (±2점)
        vol = i15.get('volume', 0) or 0
        vol_avg = i15.get('vol_ma20', 0) or 0
        if vol_avg and vol_avg > 0:
            vol_ratio = vol / vol_avg
            if vol_ratio > 2.0:
                # 거래량 2배+ → 방향 강화
                score += 2 if score > 0 else -2
            elif vol_ratio < 0.5:
                # 거래량 절반 이하 → 가짜 움직임, 감점
                score = int(score * 0.7)

        # #Q: 연속 음봉 필터 (4+ 음봉 = 떨어지는 칼날 → 감점)
        try:
            _klines_1h = get_klines(symbol, '1h', 12)
            _closes_1h = _klines_1h['close'].astype(float).tolist() if 'close' in _klines_1h.columns else []
            _consecutive_red = 0
            for _ci in range(len(_closes_1h)-1, 0, -1):
                if _closes_1h[_ci] < _closes_1h[_ci-1]:
                    _consecutive_red += 1
                else:
                    break
            if _consecutive_red >= 4:
                score -= 5  # 떨어지는 칼날 차단 (0% 승률)
            elif _consecutive_red >= 3:
                score -= 2

            # #R: 거래량 급증 + 레인지 상단 = 고점 진입 차단
            _highs_1h = _klines_1h['high'].astype(float).tolist() if 'high' in _klines_1h.columns else []
            _lows_1h = _klines_1h['low'].astype(float).tolist() if 'low' in _klines_1h.columns else []
            if _highs_1h and _lows_1h:
                _range_h = max(_highs_1h)
                _range_l = min(_lows_1h)
                _range_pct = (px - _range_l) / (_range_h - _range_l) * 100 if _range_h != _range_l else 50
                _vol_1h = i1h.get('volume', 0) or 0
                _vol_avg_1h = i1h.get('vol_ma20', 0) or 0
                _vol_ratio_1h = _vol_1h / _vol_avg_1h if _vol_avg_1h else 1
                if _vol_ratio_1h > 2.0 and _range_pct > 60:
                    score -= 3  # 고점+거래량급증 = 패닉/FOMO 진입 차단
        except:
            pass

        # 하락추세 종목 필터 (4h 역배열 + 1h RSI 약세 → 롱 차단)
        e20_4h = i4h.get('ema20', 0) or 0
        e50_4h = i4h.get('ema50', 0) or 0
        _downtrend = False
        if e20_4h and e50_4h and e20_4h < e50_4h and rsi1h < 40:
            _downtrend = True
            score = min(score, MIN_SCORE - 1)  # 강제로 진입 불가

        # #144 RL 알트 범용모델 보너스 (ETH/BTC 제외)
        _rl_signal, _rl_bonus = 'wait', 0
        if symbol not in ('ETHUSDT', 'BTCUSDT'):
            try:
                _rl_signal, _rl_bonus = get_rl_signal_lite(symbol)
                score += _rl_bonus  # RL 롱 +3, RL 관망 -3
            except:
                pass

        # #3: TAO — 하락장에서 완전 차단 (19건 58% -$2.64 반복손실)
        if symbol == 'TAOUSDT':
            if _bear_mode:
                score -= 10  # 하락장 TAO 완전 차단
            else:
                score += 3  # 상승/일반장에서만 보너스

        # 방향 — 롱 전용
        # #B: 12-15시 점수 +2 상향 (7건 승률29% 최악 구간)
        _hour = datetime.now().hour
        _min_score = MIN_SCORE  # 그리드서치: 12-15시 보너스 0이 최적
        # 상승장: MIN_SCORE -1 (더 쉽게 진입)
        if _bull_mode:
            _min_score = max(_min_score - 1, 3)
        # ETH: 3건 33% -$1.46 → MIN_SCORE +1 강화
        if symbol == 'ETHUSDT':
            _min_score += 1
        # 화요일: 20건 패평균-$1.71 최악 요일 → MIN_SCORE +1
        if datetime.now().weekday() == 1:  # 0=월, 1=화
            _min_score += 1
        if score >= _min_score: direction = 'long'
        else: direction = 'wait'

        # 10. RSI 극단값 진입 차단 (급등 되돌림 방지)
        if direction == 'long' and rsi > 85:
            direction = 'wait'

        # 숏 시그널 기록 (진입하지 않음, 데이터 축적용)
        _short_signal = None
        if _bear_mode and 55 <= rsi1h <= 70 and e20_4h < e50_4h and not btc_up:
            _short_signal = 'bear_short'
        if _short_signal:
            try:
                _short_log = '/home/hyeok/01.APCC/00.ai-lab/short_signals.jsonl'
                with open(_short_log, 'a') as f:
                    f.write(json.dumps({
                        "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                        "symbol": symbol, "price": px, "rsi_1h": round(rsi1h),
                        "rsi_15m": round(rsi), "adx": round(adx_1h),
                        "ema_trend": "down", "score": score,
                    }) + '\n')
            except: pass

        atr = i1h.get('atr', 0) or 0
        atr_pct = atr / px * 100 if px > 0 else 0
        return {
            'symbol': symbol, 'price': px, 'score': score, 'direction': direction,
            'rsi': rsi, 'adx': adx, 'adx_1h': adx_1h, 'atr_1h': atr, 'atr_pct': atr_pct,
            'ema20_1h': i1h.get('ema20', 0) or 0,
            'ema20_4h': e20_4h, 'ema50_4h': e50_4h,
            'bb_lower': i1h.get('bb_lower', 0) or 0,
            'bb_upper': i1h.get('bb_upper', 0) or 0,
            'btc_up': btc_up, 'sentiment_bonus': sentiment_bonus,
            'bb_pct': round(bb_pct, 0), 'bb_squeeze': bb_squeeze,
            'downtrend': _downtrend, 'rl_signal': _rl_signal, 'rl_bonus': _rl_bonus,
            'rs_bonus': _rs_bonus,
        }
    except Exception as e:
        return {'symbol': symbol, 'score': 0, 'direction': 'wait', 'error': str(e)[:40]}


def calc_entry(a):
    """분석 결과 → 진입가/SL/TP"""
    px = a['price']
    d = a['direction']
    atr = a['atr_1h']
    ema20 = a['ema20_1h']

    if d == 'wait':
        return None, None, None

    # ATR=0 폴백: 가격의 2%를 ATR로 사용
    if atr <= 0:
        atr = px * 0.02

    # 극변동 종목 스킵 (ATR > 가격의 15%)
    if px > 0 and atr / px > 0.15:
        return None, None, None

    is_major = a['symbol'] in ('ETHUSDT', 'BTCUSDT')
    lev = 3 if is_major else 2  # ETH/BTC 3x, 알트 2x
    # #149: 변동성 적응형 SL/TP — ATR% 기반 동적 조절
    atr_pct = atr / px * 100 if px > 0 else 2.0
    if is_major:
        sl_m, tp_m = 1.5, 3.0
    else:
        if atr_pct >= 5.0:    # 고변동 (LIGHT, M 등)
            sl_m, tp_m = 2.5, 5.0  # SL 넓게, TP도 넓게
        elif atr_pct >= 2.0:  # 중변동 (TAO, FET 등) — 기본
            sl_m, tp_m = 2.0, 5.0
        else:                 # 저변동 (BNB, SOL 등)
            sl_m, tp_m = 2.0, 5.0  # 그리드서치: SL2.0 TP5.0 통일

    # 시장 모드별 TP 조절
    if _bear_mode:
        tp_m = tp_m * 0.6  # 하락장: TP 40% 축소 (빠른 익절)
    elif _bull_mode:
        tp_m = tp_m * 1.3  # 상승장: TP 30% 확대 (더 오래 들기)

    if d == 'long':
        entry = px - atr * 0.2  # #132: ATR×0.1→0.2 (평균0.93%↓ 진입, 30분내 SL히트 감소)
        sl = entry - atr * sl_m
        tp = entry + atr * tp_m
    else:
        entry = px + atr * 0.2
        sl = entry + atr * sl_m
        tp = entry - atr * tp_m

    # #154: 이전 저점 기반 SL (커뮤니티 1위 전략 — 구조적 지지선)
    try:
        client = get_client()
        _kl_h4 = client.futures_klines(symbol=a['symbol'], interval='4h', limit=12)
        if len(_kl_h4) >= 6 and d == 'long':
            _recent_lows = [float(k[3]) for k in _kl_h4[-6:]]  # 최근 24시간 4h 저점
            _struct_low = min(_recent_lows) * 0.998  # 저점 -0.2% 버퍼
            # ATR SL보다 구조적 저점이 더 가까우면 (더 넓으면) 구조적 저점 사용
            if _struct_low < sl and _struct_low > entry * (1 - 5.0 / lev / 100):
                sl = _struct_low
    except:
        pass

    # ── 고점/저점 기반 TP (실제 저항/지지선 활용) ──
    # 단, 4h ADX 25+ (강한 추세) → 고점 캡 해제 (추세 종목은 고점 갱신 가능)
    _4h_adx = a.get('adx', 0) or 0  # score_symbol에서 이미 계산된 4h ADX 사용 불가 → 재조회
    _skip_resistance_cap = False
    try:
        client = get_client()
        _kl_4h = client.futures_klines(symbol=a['symbol'], interval='4h', limit=30)
        # 4h ADX 간이 계산 (DX 평균)
        _closes_4h = [float(k[4]) for k in _kl_4h]
        _highs_4h = [float(k[2]) for k in _kl_4h]
        _lows_4h = [float(k[3]) for k in _kl_4h]
        if len(_closes_4h) >= 15:
            _dm_plus, _dm_minus, _tr_list = [], [], []
            for _i in range(1, len(_closes_4h)):
                _h_diff = _highs_4h[_i] - _highs_4h[_i-1]
                _l_diff = _lows_4h[_i-1] - _lows_4h[_i]
                _dm_plus.append(_h_diff if _h_diff > _l_diff and _h_diff > 0 else 0)
                _dm_minus.append(_l_diff if _l_diff > _h_diff and _l_diff > 0 else 0)
                _tr_list.append(max(_highs_4h[_i]-_lows_4h[_i],
                                    abs(_highs_4h[_i]-_closes_4h[_i-1]),
                                    abs(_lows_4h[_i]-_closes_4h[_i-1])))
            _n = 14
            if len(_tr_list) >= _n:
                _atr14 = sum(_tr_list[-_n:]) / _n
                _dp14 = sum(_dm_plus[-_n:]) / _n
                _dn14 = sum(_dm_minus[-_n:]) / _n
                _di_p = (_dp14 / _atr14 * 100) if _atr14 > 0 else 0
                _di_n = (_dn14 / _atr14 * 100) if _atr14 > 0 else 0
                _dx = abs(_di_p - _di_n) / (_di_p + _di_n) * 100 if (_di_p + _di_n) > 0 else 0
                _4h_adx = _dx
                if _4h_adx >= 25:
                    _skip_resistance_cap = True
    except Exception:
        pass

    if not _skip_resistance_cap:
        try:
            kl_1h = client.futures_klines(symbol=a['symbol'], interval='1h', limit=24)
            if d == 'long':
                recent_high = max(float(k[2]) for k in kl_1h)
                tp_resistance = recent_high * 0.998
                if tp_resistance > entry:
                    tp = min(tp, tp_resistance)
            else:
                recent_low = min(float(k[3]) for k in kl_1h)
                tp_support = recent_low * 1.002
                if tp_support < entry:
                    tp = max(tp, tp_support)
        except Exception:
            pass
    # 강한 추세 → ATR 기반 TP 유지 (고점 캡 해제)

    # #F: SL 최소 거리 2.0% (1x 기준) — SL 조기 히트 방지 (11건 승률18%)
    min_sl_1x = 2.0 / lev  # 실제 2% → 1x 기준
    sl_dist_pct = abs(entry - sl) / entry * 100
    if sl_dist_pct < min_sl_1x:
        sl = entry * (1 - min_sl_1x / 100) if d == 'long' else entry * (1 + min_sl_1x / 100)

    # SL 최대 거리 캡 — 레버리지 기준 실제 손실 5% 이내
    max_sl_1x = MAX_SL_PCT / lev
    sl_dist_pct = abs(entry - sl) / entry * 100
    if sl_dist_pct > max_sl_1x:
        sl = entry * (1 - max_sl_1x / 100) if d == 'long' else entry * (1 + max_sl_1x / 100)

    # TP 최소 보장 — 레버리지 기준 실제 수익 최소 4% (고점 기반은 가까울 수 있음)
    min_tp_real = 0.04  # 실제 4% (수수료 0.14% 차감 후 3.86%)
    min_tp_1x = entry * (min_tp_real / lev)
    if abs(tp - entry) < min_tp_1x:
        tp = entry + min_tp_1x if d == 'long' else entry - min_tp_1x

    # R:R 최소 1:1.5 보장
    risk = abs(entry - sl)
    if risk > 0 and abs(tp - entry) / risk < 1.5:
        tp = entry + risk * 1.5 if d == 'long' else entry - risk * 1.5

    # 소수점 정리
    if px > 100:
        entry, sl, tp = round(entry, 2), round(sl, 2), round(tp, 2)
    elif px > 1:
        entry, sl, tp = round(entry, 4), round(sl, 4), round(tp, 4)
    else:
        entry, sl, tp = round(entry, 6), round(sl, 6), round(tp, 6)

    return entry, sl, tp


def place_limit_only(symbol, side, usdt, entry_price, leverage):
    """LIMIT만 배치 (SL/TP 없이)"""
    client = get_client()
    # LIMIT 주문만 취소 (SL/TP 보호 — #D1 버그 수정)
    try:
        orders = client.futures_get_open_orders(symbol=symbol)
        for o in orders:
            if o['type'] == 'LIMIT':
                client.futures_cancel_order(symbol=symbol, orderId=o['orderId'])
        time.sleep(0.3)
    except Exception:
        pass

    set_margin_type(symbol, "ISOLATED")
    set_leverage(symbol, leverage)

    step, tick = _get_symbol_filters(symbol)
    qty_raw = (usdt * leverage) / entry_price
    dec = 0 if step >= 1 else len(str(step).rstrip("0").split(".")[-1])
    qty = round(qty_raw - (qty_raw % step), dec)
    if qty <= 0:
        qty = step
    # 노셔널 $20 미달 시 자동 상향 (상한 $25)
    if qty * entry_price < 20:
        min_qty = 20.0 / entry_price
        qty = round(min_qty + step - (min_qty % step), dec)
        if qty <= 0:
            qty = step
    # 노셔널 상한 캡
    max_usdt = ALT_USDT_MAX if symbol != 'ETHUSDT' else ETH_USDT * ETH_LEV
    if qty * entry_price > max_usdt:
        qty = round(max_usdt / entry_price - (max_usdt / entry_price % step), dec)
        if qty <= 0:
            qty = step

    ep = _round_price(entry_price, tick)
    try:
        order = client.futures_create_order(
            symbol=symbol, side=side, type="LIMIT",
            price=ep, quantity=qty, timeInForce="GTC",
        )
        return {"success": True, "order_id": order.get("orderId"), "qty": qty, "price": ep}
    except Exception as e:
        return {"success": False, "error": str(e)[:80]}


def has_position(symbol):
    return any(p['symbol'] == symbol for p in _get_positions_cached())


_positions_cache = {'ts': 0, 'data': []}

def _get_positions_cached():
    """get_positions() 캐시 (5초 TTL) — 사이클당 API 호출 1회로 축소"""
    now = time.time()
    if now - _positions_cache['ts'] < 5:
        return _positions_cache['data']
    try:
        data = get_positions()
        _positions_cache.update({'ts': now, 'data': data})
        return data
    except Exception:
        return _positions_cache['data']


def get_held_symbols():
    """현재 포지션 심볼 목록"""
    return {p['symbol'] for p in _get_positions_cached()}


def _get_held_direction(symbol):
    """포지션 방향 조회 (long/short/None)"""
    for p in _get_positions_cached():
        if p['symbol'] == symbol:
            return 'long' if p['side'] == 'LONG' else 'short'
    return None


def check_fills():
    """체결 감지 → SL/TP 배치 + 기존 포지션 SL/TP 누락 보완"""
    # 1. pending에서 체결 감지
    filled = []
    for sym, info in list(_pending_fills.items()):
        if has_position(sym):
            try:
                pos = [p for p in _get_positions_cached() if p['symbol'] == sym][0]
                qty = float(pos.get('size', 0))
                if qty > 0:
                    _sltp_result = place_sl_tp(sym, info['side'], qty,
                                sl_price=info['sl'], tp_price=info['tp'])
                    # SL 실패 시 긴급 청산 — 보호 없는 포지션 방치 방지
                    if not _sltp_result.get('sl_placed'):
                        log(f"  🚨 {sym} SL 배치 실패! 긴급 청산")
                        _close_side = 'SELL' if info['side'] == 'BUY' else 'BUY'
                        try:
                            client.futures_create_order(symbol=sym, side=_close_side, type='MARKET',
                                quantity=str(qty), reduceOnly=True)
                        except: pass
                        filled.append(sym)
                        continue
                    _sltp_done.add(sym)
                    _tp_cache[sym] = info['tp']
                    # 기관 전략은 _institutional_post_entry에서 이미 카운트 → updater만 여기서 증가
                    if info.get('source', 'updater') == 'updater':
                        _daily_trades['count'] += 1
                    _entry_time[sym] = time.time()
                    _entry_source[sym] = info.get('source', 'updater')
                    log(f"  ✅ {sym} 체결! SL/TP 배치 (SL ${info['sl']}, TP ${info['tp']}) [오늘 {_daily_trades['count']}/{MAX_DAILY_TRADES}건]")
                    # DB 기록
                    try:
                        _side_label = "🟢 롱" if info['side'] == 'BUY' else "🔴 숏"
                        trade_db.add_trade({
                            "symbol": sym, "side": _side_label, "action": "진입",
                            "qty": qty, "price": info['entry'],
                            "sl": info['sl'], "tp": info['tp'],
                            "atr": info.get('atr', 0),
                            "confidence": abs(info.get('score', 0)),
                            "source": info.get('source', 'updater'),
                            "extra": json.dumps({"score": info.get('score',0), "btc_up": _btc_cache.get('up', True), "btc_rsi": round(_btc_cache.get('rsi', 50)), "rl": info.get('rl_signal', 'none'), "cvd_delta": info.get('cvd_delta', None)}),
                        })
                    except Exception:
                        pass
                    try:
                        send_message(TG_TOKEN, TG_CHAT,
                            f"🎯 <b>{sym} 체결!</b>\n"
                            f"   진입 ${info['entry']}\n"
                            f"   SL ${info['sl']} / TP ${info['tp']}")
                    except Exception:
                        pass
            except Exception as e:
                log(f"  🚨 {sym} 체결 처리 실패: {e} — 긴급 청산 시도")
                try:
                    _close_side = 'SELL' if info['side'] == 'BUY' else 'BUY'
                    client.futures_create_order(symbol=sym, side=_close_side, type='MARKET',
                        quantity=str(qty), reduceOnly=True)
                except: pass
            filled.append(sym)
    for sym in filled:
        del _pending_fills[sym]

    # 1-2. 만료된 pending 주문 정리 (기관 전략 300초 TTL)
    expired = [s for s, info in _pending_fills.items()
               if info.get('expire') and time.time() > info['expire'] and not has_position(s)]
    for sym in expired:
        try:
            client = get_client()
            client.futures_cancel_all_open_orders(symbol=sym)
            log(f"  ⏰ {sym} 미체결 만료 → 주문 취소")
        except: pass
        del _pending_fills[sym]
        _bb_limit_orders.pop(sym, None)  # BB 롱 예약도 동시 정리
        _bb_short_limit_orders.pop(sym, None)  # BB 숏 예약도 동시 정리

    # 2. 재시작 시 기존 포지션 SL/TP 1회 보완 (_sltp_done 비어있으면 실행)
    try:
        client = get_client()
        for pos in _get_positions_cached():
            sym = pos['symbol']
            if sym in _pending_fills or sym in _sltp_done:
                continue
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            side_str = pos.get('side', 'LONG')
            if entry > 0 and qty > 0:
                # 1회만 시도 후 _sltp_done 등록 (algo 주문은 API에서 조회 불가 → 중복 방지)
                is_long = 'LONG' in side_str.upper()
                close_side = 'SELL' if is_long else 'BUY'
                # BB 포지션: DB에서 원래 SL/TP 복원 (ATR 기반 덮어쓰기 방지)
                _src = _entry_source.get(sym, 'updater')
                _db_trade = trade_db.get_open_trades(sym)
                if _db_trade and _src in ('bb_box', 'bb_short'):
                    sl = _db_trade[0].get('sl', 0)
                    tp = _db_trade[0].get('tp', 0)
                else:
                    i1h = calc_indicators(get_klines(sym, '1h', 60))
                    atr = i1h.get('atr', 0) or 0
                    lev = 3 if sym == 'ETHUSDT' else 2
                    sl_m = 2.0
                    tp_m = 3.0
                    sl = _round_price_sym(sym, entry - atr * sl_m) if is_long else _round_price_sym(sym, entry + atr * sl_m)
                    tp = _round_price_sym(sym, entry + atr * tp_m) if is_long else _round_price_sym(sym, entry - atr * tp_m)
                if sl and tp:
                    # 1회 배치 (4130이면 이미 존재 → 무시)
                    try:
                        client.futures_create_order(symbol=sym, side=close_side, type='STOP_MARKET',
                            stopPrice=str(sl), quantity=str(qty), reduceOnly=True)
                    except: pass
                    try:
                        client.futures_create_order(symbol=sym, side=close_side, type='TAKE_PROFIT_MARKET',
                            stopPrice=str(tp), quantity=str(qty), reduceOnly=True)
                    except: pass
                    _sltp_done.add(sym)
                    _tp_cache[sym] = tp
                    # DB에 미기록 포지션 → 진입 기록 추가 (재시작 시 BB LIMIT 체결 등)
                    _existing = trade_db.get_open_trades(sym)
                    if not _existing:
                        _side_label = "🟢 롱" if is_long else "🔴 숏"
                        trade_db.add_trade({
                            "symbol": sym, "side": _side_label, "action": "진입",
                            "qty": qty, "price": entry, "sl": sl, "tp": tp,
                            "source": _entry_source.get(sym, 'updater'),
                        })
                        log(f"  🔧 {sym} SL/TP 보완 + DB 기록: SL ${sl} TP ${tp} src={_entry_source.get(sym, 'updater')}")
                    else:
                        log(f"  🔧 {sym} SL/TP 보완: SL ${sl} TP ${tp}")
    except Exception as e:
        log(f"  SL/TP 보완 오류: {e}")


def check_partial_tp():
    """실제 수익 3%+ 시 절반 청산 + SL 본절 이동 (#137)
    중복 방지: 노셔널 $15 미만이면 이미 부분 익절된 것으로 간주 → 스킵"""
    try:
        client = get_client()
        for pos in _get_positions_cached():
            sym = pos['symbol']
            if sym in _partial_done:
                continue
            # 추세숏: 전량 TP → 부분 익절 불필요 (BB롱/역행숏/bb_short는 부분익절 허용)
            if _entry_source.get(sym) in ('trend_short',):
                continue
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            side = pos.get('side', 'LONG')
            if entry <= 0 or qty <= 0:
                continue

            # 중복 방지: 노셔널 $15 미만 = 이미 부분 익절됨 (재시작 시에도 작동)
            _notional = qty * entry
            if _notional < 15:
                _partial_done.add(sym)
                continue

            cur = get_price(sym)
            is_long = 'LONG' in side.upper()

            if sym not in _tp_cache:
                continue
            tp = float(_tp_cache[sym])

            is_major = sym in ('ETHUSDT', 'BTCUSDT')
            lev = 3 if is_major else 2
            if is_long:
                pnl_real = (cur - entry) / entry * 100 * lev
            else:
                pnl_real = (entry - cur) / entry * 100 * lev

            # 부분익절 임계값: BB롱 중앙값 0.7%(1x) → 2% real이면 대부분 커버
            if pnl_real >= 2.0:
                half_qty = _round_qty(sym, qty / 2)
                if half_qty <= 0:
                    continue
                close_side = 'SELL' if is_long else 'BUY'
                try:
                    client.futures_create_order(
                        symbol=sym, side=close_side, type='MARKET',
                        quantity=half_qty, reduceOnly=True)
                    _positions_cache['ts'] = 0  # 부분 청산 후 캐시 즉시 무효화
                    log(f"  💰 {sym} 부분 익절 {half_qty} (수익 {pnl_real:.1f}% real)")

                    # SL을 본절(진입가)로 이동 — STOP_MARKET으로 통일
                    remain = _round_qty(sym, qty - half_qty)
                    if remain > 0:
                        close_side2 = 'SELL' if is_long else 'BUY'
                        # 기존 SL/TP 전체 취소 후 재배치 (algo 포함)
                        try:
                            client.futures_cancel_all_open_orders(symbol=sym)
                            time.sleep(0.3)
                        except Exception:
                            pass
                        # 본절 SL + 기존 TP — STOP_MARKET 통일
                        try:
                            client.futures_create_order(symbol=sym, side=close_side2, type='STOP_MARKET',
                                stopPrice=str(_round_price_sym(sym, entry)),
                                quantity=str(remain), reduceOnly=True)
                        except: pass
                        try:
                            client.futures_create_order(symbol=sym, side=close_side2, type='TAKE_PROFIT_MARKET',
                                stopPrice=str(_round_price_sym(sym, tp)),
                                quantity=str(remain), reduceOnly=True)
                        except: pass
                        log(f"  🔒 {sym} SL → 본절 ${entry}, 잔여 {remain}")

                    _partial_done.add(sym)

                    try:
                        send_message(TG_TOKEN, TG_CHAT,
                            f"💰 {sym} 절반 익절 +{pnl_real:.1f}% | SL→본절")
                    except Exception:
                        pass
                except Exception as e:
                    log(f"  ⚠️ {sym} 부분 익절 실패: {e}")
    except Exception as e:
        log(f"  부분 익절 체크 오류: {e}")


def _get_trail_distance(pnl_pct_1x, atr_pct, is_partial=False):
    """ATR 비례 + 수익 단계별 트레일 거리 (1x % 반환)
    atr_pct: ATR / 진입가 × 100 (1x 기준 %)
    is_partial: 부분 익절 완료 종목 → 거리 1.5배 (대수익 기회 확보)"""
    for threshold, atr_mult in TRAIL_ATR_TIERS:
        if pnl_pct_1x >= threshold:
            dist = atr_pct * atr_mult
            if is_partial:
                dist *= 1.5  # #143: 부분 익절 후 잔여분은 더 넓게
            return max(TRAIL_MIN_PCT, min(TRAIL_MAX_PCT, dist))
    # 기본
    dist = atr_pct * TRAIL_ATR_TIERS[-1][1]
    if is_partial:
        dist *= 1.5
    return max(TRAIL_MIN_PCT, min(TRAIL_MAX_PCT, dist))


# #131 SL 동기화 상태 (SL은 올리기만, 내리지 않음)
_sl_synced = {}  # {symbol: last_synced_sl_price}


def _update_sl_on_exchange(client, sym, new_sl, entry, lev, is_long):
    """바이낸스 SL 주문을 트레일링 피크 기준으로 업데이트 (올리기만, 내리지 않음)"""
    # 진입가보다 낮으면 의미 없음 (원래 SL이 이미 있음)
    if is_long and new_sl <= entry:
        return

    # 이전에 동기화한 SL보다 낮으면 스킵 (절대 내리지 않음)
    prev_sl = _sl_synced.get(sym, 0)
    if new_sl <= prev_sl * 1.001:  # 0.1% 이상 차이날 때만 업데이트 (API 절약)
        return

    # 가격 정밀도
    step, tick = _get_symbol_filters(sym)
    if new_sl > 100:
        new_sl = round(new_sl, 2)
    elif new_sl > 1:
        new_sl = round(new_sl, 4)
    else:
        new_sl = round(new_sl, 6)

    try:
        # 전체 취소 후 SL+TP 재배치 (algo/conditional 주문은 개별 취소 불가)
        try:
            client.futures_cancel_all_open_orders(symbol=sym)
            time.sleep(0.3)
        except: pass

        # 포지션 수량 조회
        _pos = [p for p in client.futures_position_information(symbol=sym) if float(p['positionAmt']) != 0]
        if not _pos: return
        _qty = abs(float(_pos[0]['positionAmt']))

        close_side = 'SELL' if is_long else 'BUY'
        # 새 SL 배치
        client.futures_create_order(
            symbol=sym, side=close_side, type='STOP_MARKET',
            stopPrice=str(new_sl), quantity=str(_qty), reduceOnly=True
        )
        # TP 재배치 (캐시 또는 ATR 기반 재계산)
        _tp_val = _tp_cache.get(sym)
        if not _tp_val:
            # 캐시 없으면 ATR 기반 재계산
            try:
                _i1h = calc_indicators(get_klines(sym, '1h', 60))
                _atr = _i1h.get('atr', 0) or 0
                _entry = float(_pos[0]['entryPrice'])
                if _atr > 0:
                    _tp_val = _entry + _atr * 5.0 if is_long else _entry - _atr * 5.0
                    _tp_cache[sym] = _tp_val
            except: pass
        if _tp_val:
            _tp_str = str(round(_tp_val, 2 if _tp_val > 100 else (4 if _tp_val > 1 else 6)))
            try:
                client.futures_create_order(
                    symbol=sym, side=close_side, type='TAKE_PROFIT_MARKET',
                    stopPrice=_tp_str, quantity=str(_qty), reduceOnly=True
                )
            except: pass

        _sl_synced[sym] = new_sl
        sl_real = (new_sl - entry) / entry * 100 * lev
        log(f"  🔒 {sym} SL 동기화: ${new_sl} ({sl_real:+.1f}% real)")
    except Exception as e:
        log(f"  ⚠️ {sym} SL 동기화 실패: {str(e)[:60]}")


def check_trailing_stop():
    """트레일링 스탑: 수익 단계별 + 레버리지 반영, 30초 빠른 체크 지원"""
    try:
        client = get_client()
        for pos in _get_positions_cached():
            sym = pos['symbol']
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            side = pos.get('side', 'LONG')
            if entry <= 0 or qty <= 0:
                continue
            # 추세숏: 전량 TP에 의존 → 트레일링 스킵 (BB롱/역행숏/bb_short는 트레일링 허용)
            if _entry_source.get(sym) in ('trend_short',):
                continue

            cur = get_price(sym)
            is_long = 'LONG' in side.upper()
            is_major = sym in ('ETHUSDT', 'BTCUSDT')
            lev = 3 if is_major else 2

            # 수익률 계산 (1x 기준)
            if is_long:
                pnl_pct = (cur - entry) / entry * 100
            else:
                pnl_pct = (entry - cur) / entry * 100

            # 트레일링 활성화 전이면 피크 업데이트만
            if pnl_pct < TRAIL_ACTIVATE_PCT:
                _trail_peak.pop(sym, None)
                continue

            # ATR 캐시 (활성화 시 1회 조회, 이후 재사용)
            if sym not in _trail_atr:
                try:
                    _kl = client.futures_klines(symbol=sym, interval='1h', limit=20)
                    _trs = []
                    for _i in range(1, len(_kl)):
                        _h, _l, _pc = float(_kl[_i][2]), float(_kl[_i][3]), float(_kl[_i-1][4])
                        _trs.append(max(_h-_l, abs(_h-_pc), abs(_l-_pc)))
                    _trail_atr[sym] = sum(_trs[-14:]) / 14 if _trs else entry * 0.02
                except Exception:
                    _trail_atr[sym] = entry * 0.02  # 폴백: 가격의 2%

            # 피크 가격 업데이트
            if sym not in _trail_peak:
                _trail_peak[sym] = cur
                atr_val = _trail_atr[sym]
                atr_pct = atr_val / entry * 100
                real_pnl = pnl_pct * lev
                _dist = _get_trail_distance(pnl_pct, atr_pct)
                log(f"  📈 {sym} 트레일링 활성화 (수익 {pnl_pct:.1f}%×{lev}={real_pnl:.1f}% | ATR={atr_pct:.1f}% 거리={_dist:.2f}%)")
            else:
                if is_long:
                    _trail_peak[sym] = max(_trail_peak[sym], cur)
                else:
                    _trail_peak[sym] = min(_trail_peak[sym], cur)

            # 고점/저점 대비 하락/상승 체크
            peak = _trail_peak[sym]
            if is_long:
                drop_pct = (peak - cur) / peak * 100
            else:
                drop_pct = (cur - peak) / peak * 100

            # ATR 비례 + 수익 단계별 트레일 거리
            atr_pct = _trail_atr.get(sym, entry * 0.02) / entry * 100
            trail_dist = _get_trail_distance(pnl_pct, atr_pct, is_partial=(sym in _partial_done))

            # #131 바이낸스 SL 주문 동기화 — 트레일링 피크 기준으로 SL 올림 (내리지 않음)
            if is_long and pnl_pct >= TRAIL_ACTIVATE_PCT:
                new_sl = peak * (1 - trail_dist / 100)
                _update_sl_on_exchange(client, sym, new_sl, entry, lev, is_long)

            if drop_pct >= trail_dist:
                # 최소 보유시간 체크 (10분 미만이면 트레일링 스킵)
                if sym in _entry_time and (time.time() - _entry_time[sym]) < MIN_HOLD_SEC:
                    _held_sec = int(time.time() - _entry_time[sym])
                    log(f"  ⏳ {sym} 트레일링 대기 (보유 {_held_sec}초 < {MIN_HOLD_SEC}초)")
                    continue
                # 트레일링 스탑 발동 → 전량 마켓 청산
                close_side = 'SELL' if is_long else 'BUY'
                try:
                    client.futures_create_order(
                        symbol=sym, side=close_side, type='MARKET',
                        quantity=qty, reduceOnly=True)
                    _positions_cache['ts'] = 0
                    real_pnl = pnl_pct * lev
                    log(f"  🔔 {sym} 트레일링 스탑! 고점 대비 {drop_pct:.1f}% 하락 → 청산 (수익 {pnl_pct:.1f}% 1x = {real_pnl:.1f}% 실제, 거리={trail_dist:.1f}%)")
                    _trail_peak.pop(sym, None)
                    _trail_atr.pop(sym, None)
                    _sl_synced.pop(sym, None)
                    try:
                        send_message(TG_TOKEN, TG_CHAT,
                            f"🔔 <b>{sym} 트레일링 스탑</b>\n"
                            f"   고점 대비 {drop_pct:.1f}% 하락 (거리={trail_dist:.2f}%) → 청산\n"
                            f"   수익 ~{real_pnl:.1f}% ({lev}x)")
                    except:
                        pass
                except Exception as e:
                    log(f"  ⚠️ {sym} 트레일링 청산 실패: {e}")
    except Exception as e:
        log(f"  트레일링 체크 오류: {e}")


def check_stale_position():
    """#F 2시간+ 보유 미수익 → SL을 진입가로 조임 (본전 SL)"""
    try:
        client = get_client()
        now = time.time()
        for pos in _get_positions_cached():
            sym = pos['symbol']
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            if entry <= 0 or qty <= 0:
                continue

            # 2시간 이상 보유 확인 (BB 박스는 횡보장이라 느린 이동 정상 → 4시간)
            _src = _entry_source.get(sym, 'updater')
            # CVD는 큰 수익 평균 10h → stale 조임 6h로 여유. BB/추세 4h, 나머지 2h
            if _src == 'cvd_divergence': _stale_sec = 21600  # 6h
            elif _src in ('bb_box', 'bb_short', 'trend_short', 'contrarian_short', 'momentum_breakout', 'surge_short', 'mega_surge'): _stale_sec = 14400  # 4h
            else: _stale_sec = 7200  # 2h
            held_sec = now - _entry_time.get(sym, now)
            if held_sec < _stale_sec:
                continue

            # 이미 조임 완료면 스킵
            if hasattr(check_stale_position, '_tightened') and sym in check_stale_position._tightened:
                continue

            cur = get_price(sym)
            is_long = 'LONG' in pos.get('side', 'LONG').upper()
            is_major = sym in ('ETHUSDT', 'BTCUSDT')
            lev = 3 if is_major else 2
            pnl_pct = (cur - entry) / entry * 100 if is_long else (entry - cur) / entry * 100

            # 미수익(pnl < 0.5%) 상태만
            if pnl_pct >= 0.5:
                continue

            # 현재 SL 확인
            orders = client.futures_get_open_orders(symbol=sym)
            old_sl = 0
            for o in orders:
                if o['type'] == 'STOP_MARKET':
                    old_sl = float(o['stopPrice'])

            # 본전 SL (진입가 - 0.3% 마진)
            margin = entry * 0.003 / lev
            new_sl = entry - margin if is_long else entry + margin
            prec = 2 if cur > 100 else (4 if cur > 1 else 6)
            new_sl = round(new_sl, prec)

            # 기존 SL보다 타이트할 때만 조임 (롱: SL 올리기, 숏: SL 내리기)
            if is_long and old_sl > 0 and new_sl <= old_sl:
                continue
            if not is_long and old_sl > 0 and new_sl >= old_sl:
                continue

            # SL 교체
            close_side = 'SELL' if is_long else 'BUY'
            for o in orders:
                if o['type'] in ('STOP_MARKET',):
                    try:
                        client.futures_cancel_order(symbol=sym, orderId=o['orderId'])
                    except: pass

            try:
                client.futures_create_order(
                    symbol=sym, side=close_side, type='STOP_MARKET',
                    stopPrice=str(new_sl), quantity=str(qty), reduceOnly=True)

                if not hasattr(check_stale_position, '_tightened'):
                    check_stale_position._tightened = set()
                check_stale_position._tightened.add(sym)

                held_m = held_sec / 60
                log(f"  ⏰ {sym} {held_m:.0f}분 보유+미수익({pnl_pct:+.1f}%) → SL 조임 {old_sl}→{new_sl}")
                # SL 조임 알림 OFF — 중간 과정, 로그만
            except Exception as e:
                log(f"  ⚠️ {sym} SL 조임 실패: {str(e)[:60]}")
    except Exception as e:
        log(f"  stale 체크 오류: {e}")


def check_long_hold():
    """#148 장기 보유(8h+) 포지션 추세 재검증 — 추세 꺾이면 시장가 청산"""
    try:
        client = get_client()
        now = time.time()
        for pos in _get_positions_cached():
            sym = pos['symbol']
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            side = pos.get('side', 'LONG')
            if entry <= 0 or qty <= 0:
                continue

            is_long = 'LONG' in side.upper()
            # 숏도 지원 (bb_short) — 롱 전용 제한 제거

            # #I: 보유시간 확인 (하락장 2h / 일반 4h)
            held_sec = now - _entry_time.get(sym, now)
            hold_limit = 2 * 3600 if _bear_mode else 4 * 3600
            if held_sec < hold_limit:
                continue

            held_h = held_sec / 3600
            # 1h/4h 지표 재검증
            try:
                i1h = calc_indicators(get_klines(sym, '1h', 60))
                i4h = calc_indicators(get_klines(sym, '4h', 30))
            except:
                continue

            e20_1h = i1h.get('ema20', 0) or 0
            e50_1h = i1h.get('ema50', 0) or 0
            e20_4h = i4h.get('ema20', 0) or 0
            e50_4h = i4h.get('ema50', 0) or 0
            rsi1h = i1h.get('rsi', 50) or 50

            # 추세 꺾임 조건: 1h 역배열 + 4h 역배열 or 1h RSI < 35
            trend_broken = False
            reason = ""
            if e20_1h < e50_1h and e20_4h < e50_4h:
                trend_broken = True
                reason = "1h+4h 역배열"
            elif e20_1h < e50_1h and rsi1h < 35:
                trend_broken = True
                reason = f"1h 역배열 + RSI {rsi1h:.0f}"

            # #T: 강제 청산 — CVD는 12h (기관 매집 후 천천히 상승), 나머지 6h
            _source = _entry_source.get(sym, 'updater')
            # CVD 12h, BB/추세숏 8h, 나머지 6h
            if _source == 'cvd_divergence': _max_hold = 12 * 3600
            elif _source in ('bb_box', 'bb_short', 'trend_short', 'contrarian_short', 'momentum_breakout', 'surge_short', 'mega_surge'): _max_hold = 8 * 3600
            else: _max_hold = 6 * 3600
            if held_sec >= _max_hold:
                trend_broken = True
                reason = f"{_max_hold//3600}h+ 강제 ({held_h:.1f}h)"

            if not trend_broken:
                continue

            # 현재 PnL 확인
            cur = get_price(sym)
            is_major = sym in ('ETHUSDT', 'BTCUSDT')
            lev = 3 if is_major else 2
            pnl_real = ((cur - entry) / entry * 100 * lev) if is_long else ((entry - cur) / entry * 100 * lev)

            # 시장가 청산 (롱→SELL, 숏→BUY)
            _close_side = 'SELL' if is_long else 'BUY'
            try:
                client.futures_create_order(
                    symbol=sym, side=_close_side, type='MARKET',
                    quantity=qty, reduceOnly=True)
                _positions_cache['ts'] = 0
                # SL/TP 정리
                try:
                    client.futures_cancel_all_open_orders(symbol=sym)
                except: pass
                _trail_peak.pop(sym, None)
                _trail_atr.pop(sym, None)
                _sl_synced.pop(sym, None)
                log(f"  🔄 {sym} 장기보유 {held_h:.1f}h + {reason} → 청산 (PnL {pnl_real:+.1f}% real)")
                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"🔄 <b>{sym} 장기보유 청산</b>\n"
                        f"   {held_h:.1f}시간 보유 | {reason}\n"
                        f"   PnL ~{pnl_real:+.1f}% real")
                except: pass
            except Exception as e:
                log(f"  ⚠️ {sym} 장기보유 청산 실패: {str(e)[:60]}")
    except Exception as e:
        log(f"  장기보유 체크 오류: {e}")


# ── #158 과매도 반등 스캘핑 (약세/횡보장 전용) ──
_bounce_cooldown = {}  # {symbol: expire_time}
_bounce_done = set()   # 이번 사이클 진입 완료

def check_oversold_bounce():
    """#158 약세장 과매도 반등: RSI<25 + 거래량 급증 → 시장가 진입, TP +1.5%, SL -1.5%"""
    try:
        client = get_client()
        held = get_held_symbols()
        held_count = len(held)

        # 슬롯 여유 없으면 스킵
        if held_count >= MAX_ORDERS:
            return

        # #B: BTC RSI 체크 — 30 미만이면 약세장, 반등도 위험 (20→30 강화)
        _, btc_rsi = get_btc_trend()
        if btc_rsi < 30:
            return

        # 추세 모드에서 후보가 있으면 스킵 (추세 추종 우선)
        # → 추세 모드 후보가 없을 때만 반등 모드 작동

        now = time.time()
        scan = get_scan_universe()

        for sym in scan:
            if sym in held or sym in BLACKLIST or sym in _bounce_done:
                continue
            if sym in _bounce_cooldown and now < _bounce_cooldown[sym]:
                continue
            if sym in _pending_fills:
                continue

            try:
                i1h = calc_indicators(get_klines(sym, '1h', 30))
                rsi = i1h.get('rsi', 50) or 50
                vol = i1h.get('volume', 0) or 0
                vol_avg = i1h.get('vol_ma20', 0) or 0
                bb_lower = i1h.get('bb_lower', 0) or 0
                px = get_price(sym)

                adx = i1h.get('adx', 0) or 0

                # #A: ADX > 50 강추세 차단 (반등 없이 계속 하락)
                if adx > 50:
                    continue

                # #D: 과매도 반등 조건 (RSI 25→20 강화)
                is_oversold = rsi < 20
                vol_surge = vol_avg > 0 and vol > 0 and vol / vol_avg >= 2.0
                near_bb_low = bb_lower > 0 and px < bb_lower * 1.01  # BB 하단 1% 이내

                if not is_oversold:
                    continue
                # 거래량 급증 OR BB 하단 근접 OR RSI 극단(<15) 중 하나
                if not (vol_surge or near_bb_low or rsi < 15):
                    continue

                # 진입!
                lev = 2
                is_major = sym in ('ETHUSDT', 'BTCUSDT')
                if is_major: lev = 3

                # #E: 비대칭 SL/TP — SL 1.5% TP 2.5% (R:R 1:1.7)
                sl_dist = 1.5 / lev / 100  # 1x 기준
                tp_dist = 2.5 / lev / 100

                entry_px = px
                sl_px = px * (1 - sl_dist)
                tp_px = px * (1 + tp_dist)

                # 손실 캡 적용
                sl_pct = abs(entry_px - sl_px) / entry_px * 100
                max_usdt = MAX_LOSS_PER_TRADE / (lev * sl_pct / 100) if sl_pct > 0 else 10
                usdt = min(10, max_usdt)  # 반등 스캘핑은 소액 ($10)

                # 정밀도
                step, tick = _get_symbol_filters(sym)
                dec = 0 if step >= 1 else len(str(step).rstrip('0').split('.')[-1])
                qty = round((usdt * lev) / entry_px, dec)
                if qty <= 0: qty = step
                if qty * entry_px < 5: continue  # 노셔널 $5 미만 스킵

                prec = 2 if px > 100 else (4 if px > 1 else 6)
                sl_px = round(sl_px, prec)
                tp_px = round(tp_px, prec)

                set_margin_type(sym, 'ISOLATED')
                set_leverage(sym, lev)

                # 시장가 즉시 진입 (반등은 빠르게 잡아야 함)
                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='MARKET', quantity=str(qty))

                time.sleep(0.5)

                # SL/TP 배치
                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='STOP_MARKET',
                        stopPrice=str(sl_px), quantity=str(qty), reduceOnly=True)
                except: pass
                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                        stopPrice=str(tp_px), quantity=str(qty), reduceOnly=True)
                except: pass

                _sltp_done.add(sym)
                _bounce_done.add(sym)
                _bounce_cooldown[sym] = now + 7200  # #C: 2시간 쿨다운 (1h→2h, 연속진입 방지)
                _entry_time[sym] = now
                _tp_cache[sym] = tp_px

                sl_real = sl_pct * lev
                tp_real = tp_dist * 100 * lev
                _vr_str = f"{vol/vol_avg:.1f}x" if vol_avg > 0 else "N/A"
                log(f"  🔄 {sym} 과매도 반등 진입! RSI={rsi:.0f} 거래량={_vr_str} | SL {sl_real:.1f}% TP +{tp_real:.1f}%")

                # DB 기록
                try:
                    trade_db.add_trade({
                        "symbol": sym, "side": "🟢 롱", "action": "진입",
                        "qty": qty, "price": entry_px,
                        "sl": sl_px, "tp": tp_px, "atr": 0,
                        "confidence": 0, "source": "bounce",
                        "extra": json.dumps({"mode": "oversold_bounce", "rsi": round(rsi), "btc_rsi": round(btc_rsi)}),
                    })
                except: pass

                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"🔄 <b>{sym} 과매도 반등!</b>\n"
                        f"   RSI={rsi:.0f} | TP +{tp_real:.1f}% SL -{sl_real:.1f}%")
                except: pass

                held_count += 1
                if held_count >= MAX_ORDERS:
                    break

            except Exception as e:
                continue
            time.sleep(0.3)

        _bounce_done.clear()  # 사이클 끝나면 리셋
    except Exception as e:
        log(f"  반등 스캔 오류: {e}")


# ── 페어 트레이딩 + OI 청산 감지 ──
_pair_cache = {'ts': 0, 'data': {}}  # 상관관계 캐시 (30분)
_oi_cache = {}  # {symbol: [oi_values]}
_oi_cache_ts = 0
_pair_cooldown = {}  # {symbol: expire_time}
_liquidation_cooldown = {}  # {symbol: expire_time}

PAIR_ZSCORE_THRESHOLD = 2.0  # z-score 2 이상이면 이탈
PAIR_USDT = 10  # 소액
PAIR_SYMBOLS = ['SOLUSDT']  # 백테스트 결과: SOL만 유효 (89%승률), XRP/BNB 역효과
OI_DROP_THRESHOLD = -5.0  # OI 3h -5% = 대량 청산


def check_pair_divergence():
    """
    #5 페어 트레이딩 — 상관 높은 종목 중 이탈 감지 → 수렴 방향 베팅
    ETH 기준으로 각 알트의 가격비율 z-score 계산
    """
    global _pair_cache
    now = time.time()

    # 30분 캐시
    if now - _pair_cache['ts'] < 1800 and _pair_cache['data']:
        return
    _pair_cache['ts'] = now

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        # ETH 기준 가격 데이터
        base_sym = 'ETHUSDT'
        base_k = client.futures_klines(symbol=base_sym, interval='1h', limit=48)
        base_closes = [float(x[4]) for x in base_k]
        if len(base_closes) < 30:
            return

        signals = []
        for sym in PAIR_SYMBOLS:
            if sym == base_sym or sym in held or sym in BLACKLIST:
                continue
            if sym in _pair_cooldown and now < _pair_cooldown[sym]:
                continue

            try:
                k = client.futures_klines(symbol=sym, interval='1h', limit=48)
                closes = [float(x[4]) for x in k]
                if len(closes) != len(base_closes):
                    continue

                # 수익률 상관
                ret_sym = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                ret_base = [(base_closes[i] - base_closes[i-1]) / base_closes[i-1] for i in range(1, len(base_closes))]
                corr = float(np.corrcoef(ret_sym, ret_base)[0][1])

                # 상관 0.7 미만이면 페어 부적합
                if corr < 0.7:
                    continue

                # 가격비율 z-score
                ratios = [closes[i] / base_closes[i] for i in range(len(closes))]
                mean_r = float(np.mean(ratios[:-3]))  # 최근 3시간 제외
                std_r = float(np.std(ratios[:-3]))
                if std_r == 0:
                    continue
                zscore = (ratios[-1] - mean_r) / std_r

                # z < -2: 알트가 ETH 대비 비정상 약세 → 알트 롱 (수렴 기대)
                if zscore < -PAIR_ZSCORE_THRESHOLD:
                    signals.append({
                        'symbol': sym, 'zscore': zscore, 'corr': corr,
                        'direction': 'long', 'reason': f'{sym[:4]} ETH대비 약세 (z={zscore:.1f})',
                    })

                # 시그널 로그 (진입 안 해도 기록)
                if abs(zscore) > 1.5:
                    try:
                        _log_path = '/home/hyeok/01.APCC/00.ai-lab/pair_signals.jsonl'
                        with open(_log_path, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "base": base_sym,
                                "zscore": round(zscore, 2), "corr": round(corr, 3),
                            }) + '\n')
                    except: pass

            except Exception:
                continue

        # 가장 강한 이탈 시그널 1개만 진입
        if signals:
            signals.sort(key=lambda x: x['zscore'])  # 가장 약한(음수 큰) 것
            sig = signals[0]
            sym = sig['symbol']

            if len(held) >= MAX_ORDERS:
                return

            try:
                px = get_price(sym)
                lev = 2
                usdt = PAIR_USDT

                # 하락장 모드 진입금 축소
                if _bear_mode:
                    usdt = int(usdt * 0.6)

                i1h = calc_indicators(get_klines(sym, '1h', 30))
                atr = i1h.get('atr', 0) or px * 0.02

                sl_px = px - atr * 2.0
                tp_px = px + atr * 3.0
                if _bear_mode:
                    tp_px = px + atr * 2.0  # 빠른 익절

                # 손실 캡
                sl_pct = abs(px - sl_px) / px * 100
                max_usdt = MAX_LOSS_PER_TRADE / (lev * sl_pct / 100) if sl_pct > 0 else usdt
                usdt = min(usdt, max_usdt)

                step, tick = _get_symbol_filters(sym)
                dec = 0 if step >= 1 else len(str(step).rstrip('0').split('.')[-1])
                prec = 2 if px > 100 else (4 if px > 1 else 6)
                qty = round((usdt * lev) / px, dec)
                if qty <= 0 or qty * px < 5:
                    return

                sl_px = round(sl_px, prec)
                tp_px = round(tp_px, prec)

                set_margin_type(sym, 'ISOLATED')
                set_leverage(sym, lev)

                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='MARKET', quantity=str(qty))
                time.sleep(0.5)

                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='STOP_MARKET',
                        stopPrice=str(sl_px), quantity=str(qty), reduceOnly=True)
                except: pass
                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                        stopPrice=str(tp_px), quantity=str(qty), reduceOnly=True)
                except: pass

                _sltp_done.add(sym)
                _pair_cooldown[sym] = now + 7200
                _entry_time[sym] = now
                _tp_cache[sym] = tp_px

                log(f"  📊 {sym} 페어 수렴 롱! z={sig['zscore']:.1f} corr={sig['corr']:.2f} | ${usdt}×{lev}x")

                try:
                    trade_db.add_trade({
                        "symbol": sym, "side": "🟢 롱", "action": "진입",
                        "qty": qty, "price": px,
                        "sl": sl_px, "tp": tp_px, "atr": 0,
                        "confidence": 0, "source": "pair",
                        "extra": json.dumps({"mode": "pair_trade", "zscore": round(sig['zscore'], 2),
                                             "corr": round(sig['corr'], 3), "base": base_sym}),
                    })
                except: pass

                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"📊 <b>{sym} 페어 수렴 롱!</b>\n"
                        f"   {sym[:4]}/ETH z={sig['zscore']:.1f} (상관={sig['corr']:.2f})\n"
                        f"   ETH 대비 비정상 약세 → 수렴 기대")
                except: pass

            except Exception as e:
                log(f"  페어 진입 실패: {str(e)[:50]}")

    except Exception as e:
        log(f"  페어 분석 오류: {str(e)[:50]}")


def check_liquidation_bounce():
    """
    #6 OI/청산 데이터 기반 — 대량 청산 후 바닥 반등 감지
    OI 3h -5% 이상 급감 → 청산 캐스케이드 종료 후 롱 진입
    """
    global _oi_cache_ts
    now = time.time()

    # 10분 캐시
    if now - _oi_cache_ts < 600:
        return
    _oi_cache_ts = now

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        targets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        for sym in targets:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _liquidation_cooldown and now < _liquidation_cooldown[sym]:
                continue

            try:
                oi_hist = client.futures_open_interest_hist(symbol=sym, period='1h', limit=6)
                if len(oi_hist) < 4:
                    continue

                oi_vals = [float(x['sumOpenInterestValue']) for x in oi_hist]
                oi_cur = oi_vals[-1]
                oi_3h = oi_vals[-3]
                oi_pct_3h = (oi_cur - oi_3h) / oi_3h * 100

                # 시그널 로그 (항상 기록)
                try:
                    _log_path = '/home/hyeok/01.APCC/00.ai-lab/oi_signals.jsonl'
                    with open(_log_path, 'a') as f:
                        f.write(json.dumps({
                            "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                            "symbol": sym, "oi_usd": round(oi_cur),
                            "oi_3h_pct": round(oi_pct_3h, 2),
                        }) + '\n')
                except: pass

                # 대량 청산 감지: OI 3h -5% 이상
                if oi_pct_3h >= OI_DROP_THRESHOLD:
                    continue

                # 추가 조건: RSI 과매도 (청산 후 바닥 확인)
                i1h = calc_indicators(get_klines(sym, '1h', 30))
                rsi = i1h.get('rsi', 50) or 50
                if rsi > 35:  # 아직 바닥 아님
                    continue

                # OI 안정화 확인 (최근 1h OI 변화 > -1% = 청산 캐스케이드 종료)
                oi_1h_pct = (oi_vals[-1] - oi_vals[-2]) / oi_vals[-2] * 100
                if oi_1h_pct < -2:  # 아직 청산 진행 중
                    log(f"  🔥 {sym} OI 3h {oi_pct_3h:+.1f}% 청산 진행 중 (1h {oi_1h_pct:+.1f}%) — 대기")
                    continue

                # 청산 종료 + 바닥 확인 → 롱 진입!
                px = get_price(sym)
                is_major = sym in ('ETHUSDT', 'BTCUSDT')
                lev = 3 if is_major else 2
                usdt = 10
                if _bear_mode:
                    usdt = int(usdt * 0.6)

                atr = i1h.get('atr', 0) or px * 0.02
                sl_px = px - atr * 1.5
                tp_px = px + atr * 3.0
                if _bear_mode:
                    tp_px = px + atr * 2.0

                sl_pct = abs(px - sl_px) / px * 100
                max_usdt = MAX_LOSS_PER_TRADE / (lev * sl_pct / 100) if sl_pct > 0 else usdt
                usdt = min(usdt, max_usdt)

                step, tick = _get_symbol_filters(sym)
                dec = 0 if step >= 1 else len(str(step).rstrip('0').split('.')[-1])
                prec = 2 if px > 100 else (4 if px > 1 else 6)
                qty = round((usdt * lev) / px, dec)
                if qty <= 0 or qty * px < 5:
                    continue

                sl_px = round(sl_px, prec)
                tp_px = round(tp_px, prec)

                set_margin_type(sym, 'ISOLATED')
                set_leverage(sym, lev)

                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='MARKET', quantity=str(qty))
                time.sleep(0.5)

                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='STOP_MARKET',
                        stopPrice=str(sl_px), quantity=str(qty), reduceOnly=True)
                except: pass
                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                        stopPrice=str(tp_px), quantity=str(qty), reduceOnly=True)
                except: pass

                _sltp_done.add(sym)
                _liquidation_cooldown[sym] = now + 7200
                _entry_time[sym] = now
                _tp_cache[sym] = tp_px

                log(f"  🔥 {sym} 청산바닥 롱! OI 3h {oi_pct_3h:+.1f}% → 안정화 | RSI={rsi:.0f} | ${usdt}×{lev}x")

                try:
                    trade_db.add_trade({
                        "symbol": sym, "side": "🟢 롱", "action": "진입",
                        "qty": qty, "price": px,
                        "sl": sl_px, "tp": tp_px, "atr": 0,
                        "confidence": 0, "source": "liquidation",
                        "extra": json.dumps({"mode": "liquidation_bounce",
                                             "oi_3h_pct": round(oi_pct_3h, 1), "rsi": round(rsi)}),
                    })
                except: pass

                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"🔥 <b>{sym} 청산바닥 롱!</b>\n"
                        f"   OI 3h {oi_pct_3h:+.1f}% (대량 청산 후 안정화)\n"
                        f"   RSI={rsi:.0f} | ${usdt}×{lev}x")
                except: pass

                break  # 1종목만

            except Exception:
                continue

    except Exception as e:
        log(f"  OI 분석 오류: {str(e)[:50]}")


# ── 크래시 바이 전략 ──
# 대상: ETH(안정) + TAO(고수익) + 거래량 상위 알트
CRASH_BUY_SYMBOLS = ['ETHUSDT']  # 백테스트 결과: ETH만 유효 (57%승률 +11.5%), TAO/SOL 미체결/저효율
CRASH_BUY_TIERS = [
    (-3.0, 5),   # -3% 지점, $5
    (-5.0, 7),   # -5% 지점, $7
    (-7.0, 10),  # -7% 지점, $10
]
CRASH_BUY_SL_PCT = 1.5   # SL: 진입가 -1.5% (real)
CRASH_BUY_TP_PCT = 4.0   # TP: 진입가 +4.0% (real)
CRASH_BUY_EXPIRE = 24 * 3600  # 24시간 후 취소
_crash_orders = {}  # {symbol: [{order_id, tier, placed_at, qty, limit_px, sl_px, tp_px}]}
_crash_last_refresh = 0  # 마지막 갱신 시각

def manage_crash_buys():
    """
    크래시 바이 — 하락장에서 현재가 -3/-5/-7%에 지정가 분할 배치
    체결 시 SL/TP 자동 배치, 24h 미체결 시 취소 후 재배치
    """
    global _crash_last_refresh
    if not _bear_mode:
        # 하락장 해제 시 → 미체결 전부 취소
        _cancel_all_crash_orders()
        return

    client = get_client()
    now = time.time()

    # 체결 확인 (매 사이클)
    _check_crash_fills(client)

    # 30분마다 갱신 (가격 변동 반영)
    if now - _crash_last_refresh < 1800:
        return
    _crash_last_refresh = now

    # 슬롯 여유 확인 (크래시 바이는 별도 슬롯 아닌 기존 슬롯 사용)
    held = get_held_symbols()
    if len(held) >= MAX_ORDERS:
        return

    for sym in CRASH_BUY_SYMBOLS:
        if sym in held or sym in BLACKLIST:
            continue

        try:
            px = get_price(sym)
            if px <= 0:
                continue

            is_major = sym in ('ETHUSDT', 'BTCUSDT')
            lev = 3 if is_major else 2

            # 기존 크래시 주문 취소 (가격 갱신)
            if sym in _crash_orders:
                for co in _crash_orders[sym]:
                    try:
                        client.futures_cancel_order(symbol=sym, orderId=co['order_id'])
                    except: pass
                _crash_orders[sym] = []

            step, tick = _get_symbol_filters(sym)
            dec = 0 if step >= 1 else len(str(step).rstrip('0').split('.')[-1])
            prec = 2 if px > 100 else (4 if px > 1 else 6)

            set_margin_type(sym, 'ISOLATED')
            set_leverage(sym, lev)

            placed = []
            for tier_pct, usdt in CRASH_BUY_TIERS:
                limit_px = round(px * (1 + tier_pct / 100), prec)

                # 손실 캡 적용
                sl_dist = CRASH_BUY_SL_PCT / lev / 100
                sl_px = round(limit_px * (1 - sl_dist), prec)
                tp_dist = CRASH_BUY_TP_PCT / lev / 100
                tp_px = round(limit_px * (1 + tp_dist), prec)

                max_usdt = MAX_LOSS_PER_TRADE / (lev * CRASH_BUY_SL_PCT / 100) if CRASH_BUY_SL_PCT > 0 else usdt
                usdt = min(usdt, max_usdt)

                qty = round((usdt * lev) / limit_px, dec)
                if qty <= 0: qty = step
                if qty * limit_px < 5: continue

                try:
                    order = client.futures_create_order(
                        symbol=sym, side='BUY', type='LIMIT',
                        price=str(limit_px), quantity=str(qty),
                        timeInForce='GTC')

                    placed.append({
                        'order_id': order['orderId'],
                        'tier': tier_pct,
                        'placed_at': now,
                        'qty': qty,
                        'limit_px': limit_px,
                        'sl_px': sl_px,
                        'tp_px': tp_px,
                        'lev': lev,
                        'prec': prec,
                    })
                except Exception as e:
                    log(f"  크래시바이 {sym} {tier_pct}% 주문 실패: {str(e)[:50]}")

            if placed:
                _crash_orders[sym] = placed
                tiers_str = ' / '.join([f"${co['limit_px']}({co['tier']}%)" for co in placed])
                log(f"  🎯 {sym} 크래시바이 배치: {tiers_str}")
                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"🎯 <b>{sym} 크래시바이</b>\n"
                        f"   현재 ${px} 기준 3단계 지정가:\n"
                        f"   {tiers_str}\n"
                        f"   체결 시 SL -{CRASH_BUY_SL_PCT}% TP +{CRASH_BUY_TP_PCT}%")
                except: pass

        except Exception as e:
            log(f"  크래시바이 {sym} 오류: {str(e)[:50]}")
        time.sleep(0.3)


def _check_crash_fills(client):
    """크래시 바이 체결 확인 → SL/TP 배치"""
    now = time.time()
    for sym, orders in list(_crash_orders.items()):
        remaining = []
        for co in orders:
            try:
                order = client.futures_get_order(symbol=sym, orderId=co['order_id'])
                status = order['status']
            except:
                remaining.append(co)
                continue

            if status == 'FILLED':
                ep = float(order.get('avgPrice', 0)) or co['limit_px']
                # 체결가 기준 SL/TP 재계산
                sl_dist = CRASH_BUY_SL_PCT / co['lev'] / 100
                tp_dist = CRASH_BUY_TP_PCT / co['lev'] / 100
                sl_px = round(ep * (1 - sl_dist), co['prec'])
                tp_px = round(ep * (1 + tp_dist), co['prec'])

                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='STOP_MARKET',
                        stopPrice=str(sl_px), quantity=str(co['qty']), reduceOnly=True)
                except: pass
                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                        stopPrice=str(tp_px), quantity=str(co['qty']), reduceOnly=True)
                except: pass

                _sltp_done.add(sym)
                _entry_time[sym] = now
                _tp_cache[sym] = tp_px

                pnl_to_tp = CRASH_BUY_TP_PCT
                log(f"  🎯 {sym} 크래시바이 체결! {co['tier']}% @ ${ep} | SL ${sl_px} TP ${tp_px}")

                try:
                    trade_db.add_trade({
                        "symbol": sym, "side": "🟢 롱", "action": "진입",
                        "qty": co['qty'], "price": ep,
                        "sl": sl_px, "tp": tp_px, "atr": 0,
                        "confidence": 0, "source": "crash_buy",
                        "extra": json.dumps({"mode": "crash_buy", "tier": co['tier'], "original_px": co['limit_px']}),
                    })
                except: pass

                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"🎯 <b>{sym} 크래시바이 체결!</b>\n"
                        f"   {co['tier']}% 단계 @ ${ep}\n"
                        f"   SL ${sl_px} (-{CRASH_BUY_SL_PCT}%) TP ${tp_px} (+{CRASH_BUY_TP_PCT}%)")
                except: pass

                # 나머지 하위 단계 주문은 유지 (분할 매수)

            elif status in ('CANCELED', 'EXPIRED', 'REJECTED'):
                pass  # 제거
            elif now - co['placed_at'] > CRASH_BUY_EXPIRE:
                # 24시간 만료
                try:
                    client.futures_cancel_order(symbol=sym, orderId=co['order_id'])
                except: pass
            else:
                remaining.append(co)

        _crash_orders[sym] = remaining
        if not remaining:
            _crash_orders.pop(sym, None)


def _cancel_all_crash_orders():
    """하락장 해제 시 미체결 크래시 주문 전부 취소"""
    if not _crash_orders:
        return
    try:
        client = get_client()
        for sym, orders in list(_crash_orders.items()):
            for co in orders:
                try:
                    client.futures_cancel_order(symbol=sym, orderId=co['order_id'])
                except: pass
            log(f"  🎯 {sym} 크래시바이 취소 (하락장 해제)")
        _crash_orders.clear()
    except: pass


_funding_cooldown = {}  # {symbol: expire_time}

def check_funding_long():
    """
    #1 펀딩비 롱 전략 — 음수 펀딩 종목 소액 롱 (하락장 전용)
    음수 펀딩 = 숏 과밀 = 롱 보유자가 8시간마다 수수료 받음
    """
    if not _bear_mode:
        return  # 하락장 모드에서만 작동

    try:
        client = get_client()
        held = get_held_symbols()
        held_count = len(held)
        if held_count >= MAX_ORDERS:
            return

        now = time.time()
        scan = get_scan_universe()

        for sym in scan:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _funding_cooldown and now < _funding_cooldown[sym]:
                continue
            if sym in _pending_fills:
                continue
            if held_count >= MAX_ORDERS:
                break

            try:
                # 펀딩비 확인
                fr = client.futures_funding_rate(symbol=sym, limit=1)
                if not fr:
                    continue
                funding = float(fr[-1]['fundingRate'])

                # 음수 펀딩 -0.03% 이하만 (하루 -0.09% = 롱에 수수료 지급)
                if funding >= -0.0003:
                    continue

                # 추가 필터: RSI 극단 과매수면 스킵 (하락 직전)
                i1h = calc_indicators(get_klines(sym, '1h', 30))
                rsi = i1h.get('rsi', 50) or 50
                adx = i1h.get('adx', 0) or 0
                if rsi > 65:
                    continue
                # ADX > 50 강한 하락 추세면 스킵
                ema20 = i1h.get('ema20', 0) or 0
                ema50 = i1h.get('ema50', 0) or 0
                if adx > 50 and ema20 < ema50:
                    continue

                px = get_price(sym)
                lev = 2
                usdt = 10  # 소액

                # SL/TP: SL 2% real, TP 3% real (펀딩비로 시간 벌기)
                sl_dist = 2.0 / lev / 100
                tp_dist = 3.0 / lev / 100
                sl_px = px * (1 - sl_dist)
                tp_px = px * (1 + tp_dist)

                # 손실 캡
                sl_pct = sl_dist * 100
                max_usdt = MAX_LOSS_PER_TRADE / (lev * sl_pct) if sl_pct > 0 else usdt
                usdt = min(usdt, max_usdt)

                step, tick = _get_symbol_filters(sym)
                dec = 0 if step >= 1 else len(str(step).rstrip('0').split('.')[-1])
                qty = round((usdt * lev) / px, dec)
                if qty <= 0: qty = step
                if qty * px < 5: continue

                prec = 2 if px > 100 else (4 if px > 1 else 6)
                sl_px = round(sl_px, prec)
                tp_px = round(tp_px, prec)

                set_margin_type(sym, 'ISOLATED')
                set_leverage(sym, lev)

                # 시장가 롱 진입
                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='MARKET', quantity=str(qty))
                time.sleep(0.5)

                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='STOP_MARKET',
                        stopPrice=str(sl_px), quantity=str(qty), reduceOnly=True)
                except: pass
                try:
                    client.futures_create_order(symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                        stopPrice=str(tp_px), quantity=str(qty), reduceOnly=True)
                except: pass

                _sltp_done.add(sym)
                _funding_cooldown[sym] = now + 7200  # 2시간 쿨다운
                _entry_time[sym] = now
                _tp_cache[sym] = tp_px

                daily_funding = funding * 3 * 100  # 하루 3회
                log(f"  💰 {sym} 펀딩비 롱! 펀딩={funding*100:+.4f}% (일{daily_funding:+.3f}%) | RSI={rsi:.0f} | ${usdt}×{lev}x")

                try:
                    trade_db.add_trade({
                        "symbol": sym, "side": "🟢 롱", "action": "진입",
                        "qty": qty, "price": px,
                        "sl": sl_px, "tp": tp_px, "atr": 0,
                        "confidence": 0, "source": "funding",
                        "extra": json.dumps({"mode": "funding_long", "funding_rate": round(funding*100, 4), "rsi": round(rsi)}),
                    })
                except: pass

                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"💰 <b>{sym} 펀딩비 롱!</b>\n"
                        f"   펀딩={funding*100:+.4f}% (일{daily_funding:+.3f}%)\n"
                        f"   RSI={rsi:.0f} | SL -{2.0:.0f}% TP +{3.0:.0f}%")
                except: pass

                held_count += 1

            except Exception:
                continue
            time.sleep(0.3)

    except Exception as e:
        log(f"  펀딩비 전략 오���: {e}")


def scan_listing_announcements():
    """#159 거래소 상장 공지 스캔 → 감시 큐에 추가"""
    try:
        for checker, name in [
            (check_upbit_announcements, "업비트"),
            (check_okx_announcements, "OKX"),
            (check_coinbase_listings, "코인베이스"),
        ]:
            try:
                result = checker()
                if not result.get('found'):
                    continue
                for ann in result.get('announcements', []):
                    sym = ann.get('symbol', '')
                    sig = ann.get('signal', 'wait')
                    if not sym or sym in _listing_watch or sym in _listing_done:
                        continue
                    if sym in BLACKLIST:
                        continue
                    # 상장 공지 = long 시그널 → 급등 후 숏 기회 감시
                    if sig == 'long':
                        _listing_watch[sym] = {
                            'detected_at': time.time(),
                            'peak_price': 0,
                            'peak_rsi': 0,
                            'source': name,
                            'title': ann.get('title', '')[:60],
                        }
                        log(f"  📢 {name} 상장 감지: {sym} → 피크 감시 시작")
                        try:
                            send_message(TG_TOKEN, TG_CHAT,
                                f"📢 <b>{name} 상장 감지: {sym}</b>\n"
                                f"   {ann.get('title', '')[:60]}\n"
                                f"   급등 후 숏 기회 감시 중...")
                        except: pass
            except Exception:
                continue
    except Exception as e:
        log(f"  상장 스캔 오류: {e}")


def check_listing_short():
    """
    #159 상장 숏 전략 — 거래소 상장 급등 후 피크에서 숏 진입
    조건: RSI>75 → RSI 꺾임 확인 → 소액 숏 + SL 필수
    """
    if not _listing_watch:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        held_count = len(held)
        now = time.time()

        expired = []
        for sym, info in list(_listing_watch.items()):
            # 6시간 경과 시 감시 종료 (상장 효과 소멸)
            if now - info['detected_at'] > 6 * 3600:
                expired.append(sym)
                continue

            # 슬롯 없으면 스킵
            if held_count >= MAX_ORDERS:
                continue
            # 이미 보유 중이면 스킵
            if sym in held:
                continue

            try:
                # 15m 캔들로 피크 판단
                k15 = get_klines(sym, '15m', 20)
                i15 = calc_indicators(k15)
                px = get_price(sym)
                rsi = i15.get('rsi', 50) or 50

                # 1h 캔들로 급등 확인
                k1h = get_klines(sym, '1h', 6)
                closes_1h = k1h['close'].astype(float).tolist()
                if len(closes_1h) < 3:
                    continue
                pct_3h = (closes_1h[-1] - closes_1h[-3]) / closes_1h[-3] * 100

                # 피크 가격/RSI 추적
                if px > info['peak_price']:
                    info['peak_price'] = px
                if rsi > info['peak_rsi']:
                    info['peak_rsi'] = rsi

                # 진입 조건:
                # 1) 최소 +15% 급등했어야 함 (상장 효과 확인)
                # 2) RSI가 한때 75+ 였다가 지금 70 아래로 꺾임 (피크 확인)
                # 3) 현재가가 피크 대비 3%+ 하락 (되돌림 시작)
                min_pump = pct_3h >= 15 or (info['peak_price'] > 0 and
                    (info['peak_price'] - closes_1h[0]) / closes_1h[0] * 100 >= 15)
                rsi_peaked = info['peak_rsi'] >= 75 and rsi < 70
                price_dropped = info['peak_price'] > 0 and (
                    (info['peak_price'] - px) / info['peak_price'] * 100 >= 3)

                if not min_pump:
                    continue
                if not (rsi_peaked and price_dropped):
                    # 아직 피크 미확인 — 계속 감시
                    if rsi > 75:
                        log(f"  👀 {sym} RSI={rsi:.0f} 피크 형성 중... (3h +{pct_3h:.1f}%)")
                    continue

                # === 숏 진입! ===
                lev = LISTING_SHORT_LEV
                usdt = LISTING_SHORT_USDT

                # SL/TP 계산 (레버리지 반영)
                sl_dist = LISTING_SHORT_SL_PCT / lev / 100  # 실제 가격 변화율
                tp_dist = LISTING_SHORT_TP_PCT / lev / 100

                sl_px = px * (1 + sl_dist)  # 숏이므로 위에 SL
                tp_px = px * (1 - tp_dist)  # 숏이므로 아래에 TP

                # 손실 캡 적용
                max_usdt = MAX_LOSS_PER_TRADE / (lev * LISTING_SHORT_SL_PCT / 100) if LISTING_SHORT_SL_PCT > 0 else usdt
                usdt = min(usdt, max_usdt)

                # 수량 계산
                step, tick = _get_symbol_filters(sym)
                dec = 0 if step >= 1 else len(str(step).rstrip('0').split('.')[-1])
                qty = round((usdt * lev) / px, dec)
                if qty <= 0: qty = step
                if qty * px < 5: continue  # 노셔널 $5 미만 스킵

                prec = 2 if px > 100 else (4 if px > 1 else 6)
                sl_px = round(sl_px, prec)
                tp_px = round(tp_px, prec)

                set_margin_type(sym, 'ISOLATED')
                set_leverage(sym, lev)

                # 시장가 숏 진입
                order = client.futures_create_order(
                    symbol=sym, side='SELL', type='MARKET', quantity=str(qty))

                time.sleep(0.5)

                # SL/TP 배치
                try:
                    client.futures_create_order(symbol=sym, side='BUY', type='STOP_MARKET',
                        stopPrice=str(sl_px), quantity=str(qty), reduceOnly=True)
                except: pass
                try:
                    client.futures_create_order(symbol=sym, side='BUY', type='TAKE_PROFIT_MARKET',
                        stopPrice=str(tp_px), quantity=str(qty), reduceOnly=True)
                except: pass

                _sltp_done.add(sym)
                _listing_done.add(sym)
                _entry_time[sym] = now
                _tp_cache[sym] = tp_px
                expired.append(sym)  # 감시 큐에서 제거

                sl_real = LISTING_SHORT_SL_PCT
                tp_real = LISTING_SHORT_TP_PCT
                drop_from_peak = (info['peak_price'] - px) / info['peak_price'] * 100

                log(f"  📉 {sym} 상장 숏 진입! RSI {info['peak_rsi']:.0f}→{rsi:.0f} | "
                    f"피크 대비 -{drop_from_peak:.1f}% | SL +{sl_real:.0f}% TP -{tp_real:.0f}% | "
                    f"${usdt} × {lev}x | 출처: {info['source']}")

                # DB 기록
                try:
                    trade_db.add_trade({
                        "symbol": sym, "side": "🔴 숏", "action": "진입",
                        "qty": qty, "price": px,
                        "sl": sl_px, "tp": tp_px, "atr": 0,
                        "confidence": 0, "source": "listing_short",
                        "extra": json.dumps({
                            "mode": "listing_short",
                            "exchange": info['source'],
                            "peak_rsi": round(info['peak_rsi']),
                            "entry_rsi": round(rsi),
                            "pump_pct": round(pct_3h, 1),
                            "drop_from_peak": round(drop_from_peak, 1),
                        }),
                    })
                except: pass

                try:
                    send_message(TG_TOKEN, TG_CHAT,
                        f"📉 <b>{sym} 상장 숏!</b> ({info['source']})\n"
                        f"   RSI {info['peak_rsi']:.0f}→{rsi:.0f} | 피크 -{drop_from_peak:.1f}%\n"
                        f"   SL +{sl_real:.0f}% TP -{tp_real:.0f}% | ${usdt}×{lev}x")
                except: pass

                held_count += 1
                if held_count >= MAX_ORDERS:
                    break

            except Exception:
                continue
            time.sleep(0.3)

        # 만료/완료 종목 제거
        for sym in expired:
            _listing_watch.pop(sym, None)

    except Exception as e:
        log(f"  상장 숏 오류: {e}")


# ═══════════════════════════════════════════════════════════
# 기관 매매법 전략 #L~#P (데이터 로깅 + 조건부 진입)
# ═══════════════════════════════════════════════════════════

def _institutional_guard():
    """기관 전략 공통 가드 — 야간/일일한도/마진 체크"""
    kst_hour = (datetime.now(tz=timezone.utc).hour + 9) % 24
    if kst_hour in NIGHT_HOURS:
        return False, "야간"
    today = datetime.now().strftime('%Y-%m-%d')
    if _daily_trades.get('date') == today and _daily_trades.get('count', 0) >= MAX_DAILY_TRADES:
        return False, "일일 한도"
    # 마진 사전 체크: 가용 잔고 < $36이면 진입 불가 (CVD $35 + 수수료 여유)
    try:
        _bal = get_balance()
        if isinstance(_bal, dict):
            _avail = float(_bal.get('available', 0) or 0)
        else:
            _avail = float(_bal or 0)
        if _avail < 36:
            return False, f"마진부족(${_avail:.0f})"
    except: pass
    return True, ""


def _check_already_held(sym):
    """같은 종목 다전략 진입 차단 — 포지션/pending/예약/연패 체크"""
    if has_position(sym):
        return True
    if sym in _pending_fills:
        return True
    if sym in _bb_limit_orders:
        return True
    if sym in _bb_short_limit_orders:
        return True
    # 동일 종목 2연패 시 스킵 (더 확실한 기회 대기)
    try:
        _recent = trade_db.get_recent_trades_by_symbol(sym, limit=2)
        if len(_recent) >= 2 and all(t['pnl'] <= 0 for t in _recent):
            return True
    except: pass
    return False


def _institutional_entry_usdt(bear_mult=0.6, normal_mult=0.7, min_usdt=12):
    """기관 전략 공통 진입금 계산 — 노셔널 보장"""
    usdt = ALT_USDT * (bear_mult if _bear_mode else normal_mult)
    return max(usdt, min_usdt)


def _institutional_post_entry(sym, source):
    """기관 전략 진입 후 공통 처리 — 일일 카운트 + 쿨다운"""
    today = datetime.now().strftime('%Y-%m-%d')
    if _daily_trades.get('date') != today:
        _daily_trades['date'] = today
        _daily_trades['count'] = 0
    _daily_trades['count'] += 1
    # BB 박스 롱/숏: 15분 쿨다운 (횡보장 왕복 재진입 허용), 나머지: 30분
    _cd = 900 if source in ('bb_box', 'bb_short', 'trend_short', 'contrarian_short', 'momentum_breakout', 'surge_short', 'mega_surge') else COOLDOWN_SEC
    _cooldown[sym] = time.time() + _cd

def check_bb_box():
    """
    볼린저 박스 왕복 — 횡보장에서 하단 매수 → 상단 청산
    조건: BB 폭 1~4% (박스권) + 가격이 하단 근처 + RSI 30~55 (과매도 아닌 횡보)
    """
    global _bb_box_cache
    now = time.time()
    if now - _bb_box_cache['ts'] < 300:  # 5분
        return
    _bb_box_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return

    # 하락장 필터: BTC RSI < 40 + ADX > 25이면 BB 롱 스킵 (역추세 방지)
    _btc_rsi = _btc_cache.get('rsi', 50)
    if _btc_rsi < 45:  # 50→45 (BB 롱은 CVD와 동일 기준)
        return
    # BTC RSI 하락 모멘텀 필터: 3시간 동안 -5pt 이상 하락이면 BB롱 스킵
    _btc_rsi_prev3 = _btc_cache.get('rsi_prev3', _btc_rsi)
    _btc_rsi_slope = _btc_rsi - _btc_rsi_prev3
    if _btc_rsi_slope < -5:
        log(f"  ⏸ BB롱 스킵: BTC RSI 하락 모멘텀 ({_btc_rsi_prev3:.0f}→{_btc_rsi:.0f}, slope={_btc_rsi_slope:.1f})")
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        _bb_exclude = {'BTCUSDT'}  # BTC만 제외 (SOL은 BB 폭 필터로 자동 조절)
        symbols = [s for s in get_scan_universe() if s not in _bb_exclude]
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST or sym in WEAK_SYMBOLS:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                # 멀티타임프레임 BB (15m + 30m + 1h)
                px = float(get_price(sym))
                _bb_bottom_count = 0
                _bb_1h = None
                _bb_adx = 0  # ADX 필터용

                for _tf in ['15m', '30m', '1h']:
                    _ind = calc_indicators(get_klines(sym, _tf, 50))
                    _bu = _ind.get('bb_upper', 0) or 0
                    _bl = _ind.get('bb_lower', 0) or 0
                    _bm = _ind.get('bb_mid', 0) or 0
                    if not (_bu > _bl > 0):
                        continue
                    _pos = (px - _bl) / (_bu - _bl) * 100
                    if -5 < _pos < 20:
                        _bb_bottom_count += 1
                    if _tf == '1h':
                        _bb_adx = _ind.get('adx', 0) or 0
                        _bb_1h = {
                            'upper': _bu, 'lower': _bl, 'mid': _bm,
                            'pos': _pos, 'width': (_bu - _bl) / _bm * 100,
                            'rsi': _ind.get('rsi', 50) or 50,
                            'atr': _ind.get('atr', 0) or 0,
                        }

                if not _bb_1h:
                    continue
                # ADX >= 20이면 추세장 → BB 반전 부적합 (최근 3일 0승3패 모두 추세장)
                if _bb_adx >= 20:
                    continue
                rsi = _bb_1h['rsi']
                atr = _bb_1h['atr']
                bb_upper = _bb_1h['upper']
                bb_lower = _bb_1h['lower']
                bb_mid = _bb_1h['mid']
                bb_width = _bb_1h['width']
                bb_pos = _bb_1h['pos']
                if rsi != rsi: continue  # nan

                # 조건: 1h 박스권(폭 2~5.5%) + 3/3 TF 하단 합의 + RSI 과매도(30~50)
                is_box = 2.0 < bb_width < 5.5  # 폭 1.5→2.0 (좁은 밴드 R:R 불리)
                is_mtf_bottom = _bb_bottom_count >= 3  # 3/3 전 TF 합의 (보수적)
                is_sideways_rsi = 30 < rsi < 50  # 55→50 (중립 제외, 과매도만)

                is_signal = is_box and is_mtf_bottom and is_sideways_rsi

                # 로그
                if is_box and (_bb_bottom_count >= 1 or rsi < 40):
                    try:
                        with open(BB_BOX_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px,
                                "bb_width": round(bb_width, 2),
                                "bb_pos": round(bb_pos, 1),
                                "rsi": round(rsi),
                                "mtf_bottom": _bb_bottom_count,
                                "signal": is_signal,
                            }) + '\n')
                    except Exception:
                        pass

                if is_signal:
                    signals.append({
                        'symbol': sym, 'price': px, 'rsi': rsi,
                        'bb_width': bb_width, 'bb_pos': bb_pos,
                        'bb_upper': bb_upper, 'bb_lower': bb_lower,
                        'bb_mid': bb_mid, 'atr': atr,
                        'mtf_bottom': _bb_bottom_count,
                    })

            except Exception as _e:
                log(f"  ⚠️ BB {sym} 스캔 오류: {str(_e)[:60]}")
                continue

        # BB 진입: 위치에 따라 시장가/LIMIT 분기
        # < 10%: 시장가 즉시 (거의 하단) / 10~30%: LIMIT 하단 예약
        # 1. 기존 LIMIT 예약 중 조건 벗어난 것 취소
        _sig_syms = {s['symbol'] for s in signals}
        for _sym in list(_bb_limit_orders.keys()):
            if _sym in held or _sym not in _sig_syms:
                try:
                    client.futures_cancel_order(symbol=_sym, orderId=_bb_limit_orders[_sym]['order_id'])
                    log(f"  📦 BB {_sym} 예약 취소 (조건 벗어남)")
                except: pass
                _pending_fills.pop(_sym, None)
                del _bb_limit_orders[_sym]

        # 2. 시그널 종목 진입
        if signals:
            ranked = sorted(signals, key=lambda x: x['bb_pos'])
            _max_orders = min(3, MAX_ORDERS - len(held))
            _placed = len([s for s in _bb_limit_orders if s not in held])
            _entered = 0

            for best in ranked:
                if _entered + _placed >= _max_orders:
                    break
                sym = best['symbol']
                if _check_already_held(sym):
                    continue
                _today = datetime.now().strftime('%Y-%m-%d')
                if _daily_trades.get('date') == _today and _daily_trades.get('count', 0) >= MAX_DAILY_TRADES:
                    break

                # BB롱 진입금: 잔고 30%, 최대 $40 × 킬존 부스트
                _bal = get_balance() or 100
                usdt = max(min(_bal * 0.30, 40) * (0.7 if _bear_mode else 1.0) * _get_killzone_boost(), 12)
                lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
                try:
                    set_margin_type(sym, "ISOLATED")
                    set_leverage(sym, lev)
                    px = float(get_price(sym))

                    atr = best.get('atr', 0) or 0
                    atr_pct = atr / px * 100 if px else 1
                    if atr_pct < 0.5:
                        atr_pct = 1.0
                    # #A: SL을 BB 하단 -0.5%로 고정 (박스 이탈 시에만 손절)
                    sl_pct_bb = abs(px - best['bb_lower']) / px * 100 + 0.5
                    sl_pct = max(sl_pct_bb, 1.5 / lev)  # 최소 레버리지 보호
                    # TP: BB 중간선 (상단 도달률 5% → 중간선 + 부분익절/트레일링으로 전환)
                    tp = _round_price_sym(sym, best['bb_mid'])
                    if best['bb_upper'] <= px * 1.01:
                        continue
                    _mtf = best.get('mtf_bottom', 0)

                    # === 위치 < 10%: 시장가 즉시 ===
                    if best['bb_pos'] < 10:
                        # 기존 LIMIT 예약 있으면 취소 후 시장가 전환
                        if sym in _bb_limit_orders:
                            try: client.futures_cancel_order(symbol=sym, orderId=_bb_limit_orders[sym]['order_id'])
                            except: pass
                            _pending_fills.pop(sym, None)
                            del _bb_limit_orders[sym]
                        _sl_bb = best['bb_lower'] * 0.997
                        _atr_sl = best.get('atr', 0) or 0
                        _sl_min_pct = max(2.0, (_atr_sl / px * 100) if px else 2.0)  # 최소 1.5% 또는 ATR%
                        _sl_min = px * (1 - _sl_min_pct / 100)
                        sl = _round_price_sym(sym, min(_sl_bb, _sl_min))
                        qty = _round_qty(sym, usdt * lev / px)
                        if qty <= 0 or qty * px < 20:
                            step, _ = _get_symbol_filters(sym)
                            qty = _round_qty(sym, 21.0 / px + float(step))
                        if qty <= 0: continue
                        _sl_dist = abs(px - sl)
                        _tp_dist = abs(tp - px)
                        _rr = _tp_dist / _sl_dist if _sl_dist else 0
                        if _rr < 0.8:
                            log(f"  ⛔ BB롱 R:R {_rr:.2f} < 0.8 → 스킵 ({sym})")
                            continue

                        order = client.futures_create_order(symbol=sym, side='BUY', type='MARKET', quantity=str(qty), newClientOrderId=f'bb_{uuid.uuid4().hex[:16]}')
                        _positions_cache['ts'] = 0
                        _sl_ok = False
                        try:
                            client.futures_create_order(symbol=sym, side='SELL', type='STOP_MARKET',
                                stopPrice=str(sl), quantity=str(qty), reduceOnly=True, newClientOrderId=f'bbsl_{uuid.uuid4().hex[:12]}')
                            _sl_ok = True
                        except: pass
                        _tp_qty = qty  # BB 박스: 전량 TP (횡보장은 추세 없어 트레일링 비효율)
                        try:
                            client.futures_create_order(symbol=sym, side='SELL', type='TAKE_PROFIT_MARKET',
                                stopPrice=str(tp), quantity=str(_tp_qty), reduceOnly=True, newClientOrderId=f'bbtp_{uuid.uuid4().hex[:12]}')
                        except: pass
                        if not _sl_ok:
                            try:
                                client.futures_create_order(symbol=sym, side='SELL', type='MARKET', quantity=str(qty), reduceOnly=True)
                            except: pass
                            continue
                        _sltp_done.add(sym)
                        _tp_cache[sym] = tp
                        _entry_time[sym] = time.time()
                        _entry_source[sym] = 'bb_box'
                        try:
                            _fill = client.futures_get_order(symbol=sym, orderId=order['orderId'])
                            _avg = float(_fill.get('avgPrice', 0))
                            entry_px = _avg if _avg > 0 else px
                        except: entry_px = px
                        trade_db.add_trade({"symbol": sym, "side": "🟢 롱", "action": "진입",
                            "qty": qty, "price": entry_px, "sl": sl, "tp": tp, "source": "bb_box",
                            "extra": json.dumps({"bb_width": round(best['bb_width'], 1), "bb_pos": round(best['bb_pos'], 0), "mode": "market"})})
                        log(f"  ✅ BB 시장가: {sym} @ {entry_px} (위치{best['bb_pos']:.0f}%<10%) R:R=1:{_rr:.1f} MTF={_mtf}/3")
                        send_message(TG_TOKEN, TG_CHAT,
                            f"📦 <b>BB 즉시 롱</b>\n   {sym} @ ${entry_px} (하단 {best['bb_pos']:.0f}%)\n   MTF {_mtf}/3 | R:R 1:{_rr:.1f}")
                        _institutional_post_entry(sym, 'bb_box')
                        _entered += 1

                    # === 위치 10~30%: LIMIT 하단 예약 ===
                    else:
                        limit_px = _round_price_sym(sym, best['bb_lower'] * 1.002)
                        _sl_bb = best['bb_lower'] * 0.997
                        _atr_sl = best.get('atr', 0) or 0
                        _sl_min_pct = max(2.0, (_atr_sl / limit_px * 100) if limit_px else 2.0)
                        _sl_min = limit_px * (1 - _sl_min_pct / 100)
                        sl = _round_price_sym(sym, min(_sl_bb, _sl_min))
                        qty = _round_qty(sym, usdt * lev / limit_px)
                        if qty <= 0 or qty * limit_px < 20:
                            step, _ = _get_symbol_filters(sym)
                            qty = _round_qty(sym, 21.0 / limit_px + float(step))
                        if qty <= 0: continue
                        _sl_dist = abs(limit_px - sl)
                        _tp_dist = abs(tp - limit_px)
                        _rr = _tp_dist / _sl_dist if _sl_dist else 0
                        if _rr < 0.8:
                            log(f"  ⛔ BB롱예약 R:R {_rr:.2f} < 0.8 → 스킵 ({sym})")
                            continue

                        if sym in _bb_limit_orders:
                            _old = _bb_limit_orders[sym]
                            if abs(_old['price'] - limit_px) / limit_px < 0.01:
                                continue
                            try: client.futures_cancel_order(symbol=sym, orderId=_old['order_id'])
                            except: pass
                            _pending_fills.pop(sym, None)
                            del _bb_limit_orders[sym]

                        order = client.futures_create_order(symbol=sym, side='BUY', type='LIMIT',
                            price=str(limit_px), quantity=str(qty), timeInForce='GTC', newClientOrderId=f'bbl_{uuid.uuid4().hex[:16]}')
                        _bb_limit_orders[sym] = {'order_id': order['orderId'], 'price': limit_px, 'qty': qty, 'sl': sl, 'tp': tp}
                        _pending_fills[sym] = {
                            'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                            'side': 'BUY', 'entry': limit_px, 'score': 0, 'atr': atr,
                            'time': time.time(), 'source': 'bb_box', 'expire': time.time() + 600,
                        }
                        log(f"  📦 BB 예약: {sym} @ {limit_px} (위치{best['bb_pos']:.0f}%) R:R=1:{_rr:.1f} MTF={_mtf}/3")
                        send_message(TG_TOKEN, TG_CHAT,
                            f"📦 <b>BB 예약 롱</b>\n   {sym} @ ${limit_px} (BB하단 대기)\n   MTF {_mtf}/3 | R:R 1:{_rr:.1f}")
                        _placed += 1
                except Exception as e:
                    log(f"  ❌ BB 진입 ���패: {e}")


    except Exception as e:
        log(f"  BB 박스 오류: {e}")


def check_momentum_breakout():
    """
    모멘텀 브레이크아웃 롱 — BB 중립장(40~80%)에서 추세 초기 합류
    기존 전략 사각지대 (BB롱: BB<20, CVD: RSI<30, 숏: BTC RSI<55) 커버
    조건: ADX>25 + EMA정배열 + RSI 50~68 + MACD hist>0 + BTC RSI 45~65
    """
    global _momentum_cache
    now = time.time()
    if now - _momentum_cache['ts'] < 300:
        return
    _momentum_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return

    # BTC RSI 45~65만 (기존 전략 비활성 구간)
    _btc_rsi = _btc_cache.get('rsi', 50)
    if _btc_rsi < 45 or _btc_rsi > 65:
        return
    _btc_rsi_prev3 = _btc_cache.get('rsi_prev3', _btc_rsi)
    if (_btc_rsi - _btc_rsi_prev3) < -3:
        return  # BTC RSI 급락 중

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        symbols = [s for s in get_scan_universe() if s not in {'BTCUSDT'}]
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST or sym in WEAK_SYMBOLS:
                continue
            if _check_already_held(sym):
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                ind = calc_indicators(get_klines(sym, '1h', 50))
                px = float(get_price(sym))

                rsi = ind.get('rsi', 50) or 50
                adx = ind.get('adx', 0) or 0
                dmp = ind.get('adx_dmp', 0) or 0
                dmn = ind.get('adx_dmn', 0) or 0
                ema20 = ind.get('ema20', 0) or 0
                ema50 = ind.get('ema50', 0) or 0
                macd_hist = ind.get('macd_hist', 0) or 0
                atr = ind.get('atr', 0) or 0
                bb_upper = ind.get('bb_upper', 0) or 0
                bb_lower = ind.get('bb_lower', 0) or 0

                if not (bb_upper > bb_lower > 0 and ema50 > 0 and px > 0):
                    continue

                bb_pos = (px - bb_lower) / (bb_upper - bb_lower) * 100

                # 7개 조건 모두 충족
                is_signal = (
                    adx > 25 and dmp > dmn and       # 상승 추세
                    50 < rsi < 68 and                 # 초기 강세
                    40 < bb_pos < 80 and              # 중립장
                    ema20 > ema50 and                 # 정배열
                    macd_hist > 0                     # 상승 모멘텀
                )

                # 로그 (부분 충족도 기록)
                if adx > 25 and dmp > dmn and (50 < rsi < 68 or 40 < bb_pos < 80):
                    try:
                        with open(MOMENTUM_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": round(px, 4),
                                "adx": round(adx, 1), "dmp": round(dmp, 1), "dmn": round(dmn, 1),
                                "rsi": round(rsi, 1), "bb_pos": round(bb_pos, 1),
                                "macd_hist": round(macd_hist, 6),
                                "ema20": round(ema20, 4), "ema50": round(ema50, 4),
                                "signal": is_signal,
                            }) + '\n')
                    except Exception:
                        pass

                if not is_signal:
                    continue

                # 24h 거래량 필터
                _kl_vol = get_klines(sym, '1h', 24)
                _vol_24h = (_kl_vol['volume'] * _kl_vol['close']).sum() if len(_kl_vol) > 0 else 0
                if _vol_24h < 100_000_000:
                    continue

                signals.append({
                    'symbol': sym, 'price': px, 'rsi': rsi,
                    'adx': adx, 'bb_pos': bb_pos,
                    'bb_upper': bb_upper, 'ema50': ema50,
                    'atr': atr,
                })

            except Exception:
                continue

        if signals and len(held) < MAX_ORDERS:
            ranked = sorted(signals, key=lambda x: -x['adx'])

            for best in ranked[:1]:
                sym = best['symbol']
                _today = datetime.now().strftime('%Y-%m-%d')
                if _daily_trades.get('date') == _today and _daily_trades.get('count', 0) >= MAX_DAILY_TRADES:
                    break

                # 모멘텀 진입금: 잔고의 15%, 최대 $25 × 킬존 부스트
                _bal = get_balance() or 100
                usdt = max(min(_bal * 0.15, 25) * _get_killzone_boost(), 15)
                lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
                try:
                    set_margin_type(sym, "ISOLATED")
                    set_leverage(sym, lev)
                    px = float(get_price(sym))
                    atr = best['atr']

                    # SL: EMA50 - 0.3% (추세 이탈 시 손절)
                    _sl_ema = best['ema50'] * 0.997
                    _atr_pct = atr / px * 100 if px else 1
                    _sl_atr = px * (1 - max(_atr_pct, 1.5) / 100)
                    sl = _round_price_sym(sym, min(_sl_ema, _sl_atr))
                    # SL 최대 3%
                    if abs(px - sl) / px > 0.03:
                        sl = _round_price_sym(sym, px * 0.97)

                    # TP: BB 상단
                    tp = _round_price_sym(sym, best['bb_upper'])

                    # R:R 최소 1.5
                    _sl_dist = abs(px - sl)
                    _tp_dist = abs(tp - px)
                    _rr = _tp_dist / _sl_dist if _sl_dist else 0
                    if _rr < 1.5:
                        continue

                    # LIMIT 진입 (ATR × 0.15 오프셋)
                    _offset = min(max(atr * 0.15 / px if px else 0.002, 0.001), 0.004)
                    limit_px = _round_price_sym(sym, px * (1 - _offset))

                    qty = _round_qty(sym, usdt * lev / limit_px)
                    if qty <= 0 or qty * limit_px < 20:
                        step, _ = _get_symbol_filters(sym)
                        qty = _round_qty(sym, 21.0 / limit_px + float(step))
                    if qty <= 0:
                        continue

                    # 손실 캡
                    _sl_dist_pct = abs(limit_px - sl) / limit_px * 100
                    _max_usdt = MAX_LOSS_PER_TRADE / (lev * _sl_dist_pct / 100) if _sl_dist_pct > 0 else usdt
                    if usdt > _max_usdt:
                        usdt = _max_usdt
                        qty = _round_qty(sym, usdt * lev / limit_px)

                    order = client.futures_create_order(
                        symbol=sym, side='BUY', type='LIMIT',
                        price=str(limit_px), quantity=str(qty), timeInForce='GTC',
                        newClientOrderId=f'mb_{uuid.uuid4().hex[:16]}')
                    _pending_fills[sym] = {
                        'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                        'side': 'BUY', 'entry': limit_px, 'score': 0, 'atr': atr,
                        'time': time.time(), 'source': 'momentum_breakout', 'expire': time.time() + 600,
                    }
                    log(f"  🚀 모멘텀 진입: {sym} @ {limit_px} ADX={best['adx']:.0f} RSI={best['rsi']:.0f} BB={best['bb_pos']:.0f}%")
                    send_message(TG_TOKEN, TG_CHAT,
                        f"🚀 <b>모멘텀 브레이크아웃</b>\n"
                        f"   {sym} @ ${limit_px}\n"
                        f"   ADX={best['adx']:.0f} RSI={best['rsi']:.0f} BB={best['bb_pos']:.0f}%\n"
                        f"   BTC RSI {_btc_rsi:.0f}\n"
                        f"   SL ${sl} → TP ${tp} | R:R 1:{_rr:.1f}")
                    _institutional_post_entry(sym, 'momentum_breakout')
                    break

                except Exception as e:
                    log(f"  ❌ 모멘텀 실패: {e}")

    except Exception as e:
        log(f"  모멘텀 오류: {e}")


def check_surge_short():
    """
    급등 과매수 숏 — 24h 급등(+30%+) + RSI 80+ 종목 역행 숏
    STO +$6.10 실증 패턴 자동화. BTC RSI 무관 (급등 자체가 시그널)
    기존 역행숏(contrarian_short)은 BTC RSI<45 제한 → 이 전략은 독립 작동
    """
    global _surge_short_cache
    now = time.time()
    if now - _surge_short_cache['ts'] < 300:
        return
    _surge_short_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return
    # 연패 쿨다운 체크 (_quick_surge_scan과 동일)
    if time.time() < _global_cooldown_until or _bear_stopped:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        # 24h 티커에서 급등 종목 필터
        tickers = {t['symbol']: t for t in client.futures_ticker()}
        signals = []

        for sym, t in tickers.items():
            if not sym.endswith('USDT') or sym in BLACKLIST or sym in WEAK_SYMBOLS:
                continue
            if sym in held or _check_already_held(sym):
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            chg_24h = float(t.get('priceChangePercent', 0))
            vol = float(t.get('quoteVolume', 0))

            # 24h +30%+ 급등 + 거래량 $50M+ (유동성)
            if chg_24h < 30 or vol < 50_000_000:
                continue

            try:
                ind = calc_indicators(get_klines(sym, '1h', 50))
                px = float(get_price(sym))
                rsi = ind.get('rsi', 50) or 50
                adx = ind.get('adx', 0) or 0
                bbu = ind.get('bb_upper', 0) or 0
                bbl = ind.get('bb_lower', 0) or 0
                bbm = ind.get('bb_mid', 0) or 0
                atr = ind.get('atr', 0) or 0

                if not (bbu > bbl > 0 and px > 0):
                    continue
                bb_pos = (px - bbl) / (bbu - bbl) * 100

                # 조건: RSI 80+ (극과매수) + BB 100%+ (밴드 상단 이탈)
                is_signal = rsi > 80 and bb_pos > 95

                # 로그 (30%+ 급등이면 무조건 기록)
                try:
                    with open(SURGE_SHORT_LOG, 'a') as f:
                        f.write(json.dumps({
                            "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                            "symbol": sym, "price": round(px, 4),
                            "chg_24h": round(chg_24h, 1), "rsi": round(rsi, 1),
                            "adx": round(adx, 1), "bb_pos": round(bb_pos, 1),
                            "vol_m": round(vol / 1e6, 0),
                            "signal": is_signal,
                        }) + '\n')
                except Exception:
                    pass

                if not is_signal:
                    continue

                # 오더북 매도벽 확인 (매도 > 매수 = 하락 압력)
                try:
                    ob = client.futures_order_book(symbol=sym, limit=10)
                    bid = sum(float(b[1]) for b in ob['bids'][:5])
                    ask = sum(float(a[1]) for a in ob['asks'][:5])
                    _imbalance = (bid - ask) / (bid + ask) * 100 if (bid + ask) else 0
                    if _imbalance > 20:  # 매수벽 강하면 스킵 (숏 스퀴즈 위험)
                        log(f"  ⚠️ {sym} 급등숏 스킵: 매수벽 {_imbalance:+.0f}%")
                        continue
                except Exception:
                    pass

                signals.append({
                    'symbol': sym, 'price': px, 'rsi': rsi,
                    'adx': adx, 'bb_pos': bb_pos, 'chg_24h': chg_24h,
                    'bb_upper': bbu, 'bb_mid': bbm, 'atr': atr,
                })

            except Exception:
                continue

        if signals and len(held) < MAX_ORDERS:
            # RSI 높은 순 (가장 과매수인 종목 우선)
            ranked = sorted(signals, key=lambda x: -x['rsi'])

            for best in ranked[:1]:
                sym = best['symbol']
                _today = datetime.now().strftime('%Y-%m-%d')
                if _daily_trades.get('date') == _today and _daily_trades.get('count', 0) >= MAX_DAILY_TRADES:
                    break

                # 급등숏 진입금: 잔고의 20%, 최대 $30 × 킬존 부스트
                _bal = get_balance() or 100
                usdt = max(min(_bal * 0.20, 30) * _get_killzone_boost(), 15)
                lev = ALT_LEV
                try:
                    set_margin_type(sym, "ISOLATED")
                    set_leverage(sym, lev)
                    px = float(get_price(sym))
                    atr = best['atr']

                    # SL: BB 상단 +0.5% (급등 종목은 SL 넓게)
                    sl = _round_price_sym(sym, best['bb_upper'] * 1.005)
                    # SL 최대 5% 캡
                    if abs(sl - px) / px > 0.05:
                        sl = _round_price_sym(sym, px * 1.05)
                    # SL 최소 2%
                    if abs(sl - px) / px < 0.02:
                        sl = _round_price_sym(sym, px * 1.02)

                    # TP: BB 중간선 (과매수 해소 목표)
                    tp = _round_price_sym(sym, best['bb_mid'])
                    if tp >= px * 0.99:
                        continue  # TP 너무 가까움

                    # R:R 체크
                    _sl_dist = abs(sl - px)
                    _tp_dist = abs(px - tp)
                    _rr = _tp_dist / _sl_dist if _sl_dist else 0
                    if _rr < 1.5:
                        continue

                    # 수량 계산 + 손실 캡
                    qty = _round_qty(sym, usdt * lev / px)
                    if qty <= 0 or qty * px < 20:
                        step, _ = _get_symbol_filters(sym)
                        qty = _round_qty(sym, 21.0 / px + float(step))
                    if qty <= 0:
                        continue

                    _sl_dist_pct = abs(sl - px) / px * 100
                    _max_usdt = MAX_LOSS_PER_TRADE / (lev * _sl_dist_pct / 100) if _sl_dist_pct > 0 else usdt
                    if usdt > _max_usdt:
                        usdt = _max_usdt
                        qty = _round_qty(sym, usdt * lev / px)

                    # 시장가 즉시 진입 (급등 종목은 빠르게 반전 가능)
                    order = client.futures_create_order(
                        symbol=sym, side='SELL', type='MARKET',
                        quantity=str(qty), newClientOrderId=f'ss_{uuid.uuid4().hex[:16]}')

                    # SL/TP 즉시 배치
                    import time as _t; _t.sleep(0.3)
                    try:
                        client.futures_create_order(symbol=sym, side='BUY', type='STOP_MARKET',
                            stopPrice=str(sl), quantity=str(qty), reduceOnly=True)
                    except: pass
                    try:
                        client.futures_create_order(symbol=sym, side='BUY', type='TAKE_PROFIT_MARKET',
                            stopPrice=str(tp), quantity=str(qty), reduceOnly=True)
                    except: pass

                    _entry_source[sym] = 'surge_short'
                    _entry_time[sym] = time.time()
                    _tp_cache[sym] = tp

                    # DB 기록
                    try:
                        trade_db.record_trade({
                            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "symbol": sym, "side": "🔴 숏", "action": "진입",
                            "qty": qty, "price": px, "sl": sl, "tp": tp, "atr": atr,
                            "confidence": 0, "source": "surge_short",
                            "extra": json.dumps({"chg_24h": round(best['chg_24h'], 1), "rsi": round(best['rsi']), "adx": round(best['adx']), "bb_pos": round(best['bb_pos'])}),
                        })
                    except: pass

                    log(f"  🔥 급등숏 진입: {sym} @ {px} 24h{best['chg_24h']:+.0f}% RSI={best['rsi']:.0f} R:R=1:{_rr:.1f}")
                    send_message(TG_TOKEN, TG_CHAT,
                        f"🔥 <b>급등 과매수 숏</b>\n"
                        f"   {sym} @ ${px} (24h {best['chg_24h']:+.0f}%)\n"
                        f"   RSI={best['rsi']:.0f} BB={best['bb_pos']:.0f}% ADX={best['adx']:.0f}\n"
                        f"   SL ${sl} → TP ${tp} | R:R 1:{_rr:.1f}")
                    _sltp_done.add(sym)  # verify_sltp 덮어쓰기 방지
                    _institutional_post_entry(sym, 'surge_short')
                    break

                except Exception as e:
                    log(f"  ❌ 급등숏 실패: {e}")

    except Exception as e:
        log(f"  급등숏 오류: {e}")


def _quick_surge_scan():
    """30초 주기 경량 급등 캐치 — 24h +80%+ 종목 즉시 숏
    position_updater 메인 루프의 30초 트레일링 체크에 통합.
    STO +173% → +$6.10 패턴 자동화. BB 조건 없음 (급등 시 BB 무의미)."""
    global _surge_entered_today, _quick_surge_date

    # 안전장치: 야간/한도/마진 + 연패 쿨다운 체크
    ok, reason = _institutional_guard()
    if not ok:
        return
    if time.time() < _global_cooldown_until or _bear_stopped:
        return

    # 날짜 리셋
    _today = datetime.now().strftime('%Y-%m-%d')
    if _quick_surge_date != _today:
        _quick_surge_date = _today
        _surge_entered_today = set()

    try:
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        client = get_client()
        tickers = client.futures_ticker()  # API 1회 — 전 종목 24h 데이터

        for t in tickers:
            sym = t['symbol']
            if not sym.endswith('USDT') or sym in BLACKLIST or sym in WEAK_SYMBOLS:
                continue
            if sym in _surge_entered_today or sym in held:
                continue
            if _check_already_held(sym):
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            chg = float(t.get('priceChangePercent', 0))
            vol = float(t.get('quoteVolume', 0))

            # 24h +80%+ AND 거래량 $50M+
            if chg < 80 or vol < 50_000_000:
                continue

            try:
                # 15m RSI만 빠르게 확인 (1h 대비 반응 2배 빠름)
                ind = calc_indicators(get_klines(sym, '15m', 30))
                rsi_15m = ind.get('rsi', 50) or 50

                # 로그 (80%+ 급등이면 무조건 기록)
                try:
                    with open(SURGE_SHORT_LOG, 'a') as f:
                        f.write(json.dumps({
                            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "symbol": sym, "chg_24h": round(chg, 1),
                            "rsi_15m": round(rsi_15m, 1), "vol_m": round(vol / 1e6),
                            "type": "quick_scan", "signal": rsi_15m > 75,
                        }) + '\n')
                except Exception:
                    pass

                if rsi_15m <= 75:
                    continue

                # 오더북 매수벽 체크 (숏 스퀴즈 방지)
                try:
                    ob = client.futures_order_book(symbol=sym, limit=10)
                    bid = sum(float(b[1]) for b in ob['bids'][:5])
                    ask = sum(float(a[1]) for a in ob['asks'][:5])
                    if bid > 0 and ask > 0 and (bid - ask) / (bid + ask) > 0.30:
                        log(f"  ⚠️ {sym} 메가급등 스킵: 매수벽 강함")
                        continue
                except Exception:
                    pass

                # 진입!
                px = float(get_price(sym))
                lev = ALT_LEV
                set_margin_type(sym, "ISOLATED")
                set_leverage(sym, lev)

                # SL: +3% 고정 (급등 변동성)
                sl = _round_price_sym(sym, px * 1.03)
                # TP: -5% 고정 (STO 실측 -4.5% 기반)
                tp = _round_price_sym(sym, px * 0.95)

                # 메가급등 진입금: 잔고 20%, 최대 $30 × 킬존 부스트
                _bal = get_balance() or 100
                usdt = max(min(_bal * 0.20, 30) * _get_killzone_boost(), 15)

                qty = _round_qty(sym, usdt * lev / px)
                if qty <= 0 or qty * px < 20:
                    step, _ = _get_symbol_filters(sym)
                    qty = _round_qty(sym, 21.0 / px + float(step))
                if qty <= 0:
                    continue

                # 손실 캡
                _sl_pct = 3.0
                _max_usdt = MAX_LOSS_PER_TRADE / (lev * _sl_pct / 100)
                if usdt > _max_usdt:
                    usdt = _max_usdt
                    qty = _round_qty(sym, usdt * lev / px)
                if qty <= 0:
                    continue

                # 시장가 즉시 숏
                order = client.futures_create_order(
                    symbol=sym, side='SELL', type='MARKET',
                    quantity=str(qty), newClientOrderId=f'qs_{uuid.uuid4().hex[:16]}')

                time.sleep(0.3)
                # SL/TP 즉시 배치
                try:
                    client.futures_create_order(symbol=sym, side='BUY', type='STOP_MARKET',
                        stopPrice=str(sl), quantity=str(qty), reduceOnly=True)
                except: pass
                try:
                    client.futures_create_order(symbol=sym, side='BUY', type='TAKE_PROFIT_MARKET',
                        stopPrice=str(tp), quantity=str(qty), reduceOnly=True)
                except: pass

                _entry_source[sym] = 'mega_surge'
                _entry_time[sym] = time.time()
                _tp_cache[sym] = tp
                _surge_entered_today.add(sym)

                # DB 기록
                try:
                    trade_db.record_trade({
                        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "symbol": sym, "side": "🔴 숏", "action": "진입",
                        "qty": qty, "price": px, "sl": sl, "tp": tp, "atr": 0,
                        "confidence": 0, "source": "mega_surge",
                        "extra": json.dumps({"chg_24h": round(chg, 1), "rsi_15m": round(rsi_15m), "type": "mega_surge"}),
                    })
                except: pass

                log(f"  🔥🔥 메가급등 즉시숏: {sym} +{chg:.0f}% @ {px} RSI15m={rsi_15m:.0f}")
                send_message(TG_TOKEN, TG_CHAT,
                    f"🔥🔥 <b>메가 급등 숏!</b>\n"
                    f"   {sym} @ ${px} (24h <b>+{chg:.0f}%</b>)\n"
                    f"   15m RSI={rsi_15m:.0f} Vol=${vol/1e6:.0f}M\n"
                    f"   SL ${sl} (+3%) → TP ${tp} (-5%)")
                _sltp_done.add(sym)  # verify_sltp 덮어쓰기 방지
                _institutional_post_entry(sym, 'mega_surge')
                break  # 1종목만

            except Exception as e:
                log(f"  ❌ 메가급등 {sym} 실패: {e}")
                continue

    except Exception as e:
        if 'Timestamp' not in str(e):  # 타임스탬프 에러는 무시
            log(f"  급등스캔 오류: {e}")


def check_trend_short():
    """
    추세 추종 숏 — 하락장에서 약한 종목 숏
    조건: BTC RSI < 40 + ADX > 25 (하락 추세 확인)
        + 종목 RSI < 35 + BB pos < 10% (하방 모멘텀)
        + 거래량 1h > 평균의 1.2배 (패닉셀 확인)
    SL: BB mid (반등 시 중간선 위로 가면 추세 끝)
    TP: ATR × 2.5 아래
    """
    global _trend_short_cache
    now = time.time()
    if now - _trend_short_cache['ts'] < 300:
        return
    _trend_short_cache['ts'] = now

    # 하락장 확인: BTC RSI < 35 (40→35, 더 확실한 하락장만)
    _btc_rsi = _btc_cache.get('rsi', 50)
    if _btc_rsi >= 35:
        return

    ok, reason = _institutional_guard()
    if not ok:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        symbols = [s for s in get_scan_universe() if s not in {'BTCUSDT'}]
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                px = float(get_price(sym))
                ind = calc_indicators(get_klines(sym, '1h', 50))
                rsi = ind.get('rsi', 50) or 50
                adx = ind.get('adx', 0) or 0
                atr = ind.get('atr', 0) or 0
                bb_upper = ind.get('bb_upper', 0) or 0
                bb_lower = ind.get('bb_lower', 0) or 0
                bb_mid = ind.get('bb_mid', 0) or 0
                # 거래량 직접 계산 (indicators에 없으므로)
                _kl = get_klines(sym, '1h', 20)
                _vols = _kl['volume'].tolist() if hasattr(_kl, 'tolist') else [float(k[5]) for k in _kl]
                volume = _vols[-1] if _vols else 0
                vol_avg = sum(_vols[:-1]) / max(len(_vols) - 1, 1) if len(_vols) > 1 else volume

                if not (bb_upper > bb_lower > 0) or atr <= 0:
                    continue

                bb_pos = (px - bb_lower) / (bb_upper - bb_lower) * 100

                # 조건: RSI < 40 + BB pos < 25% + ADX > 25 (확실한 하방 추세)
                is_weak = rsi < 40
                is_at_low = bb_pos < 25
                is_trending = adx > 25  # ADX 필터 추가 (추세 확인)
                _vol_bonus = volume > vol_avg * 1.2 if vol_avg > 0 else False

                is_signal = is_weak and is_at_low and is_trending

                # 로그
                if is_weak and is_at_low:
                    try:
                        with open(TREND_SHORT_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px, "rsi": round(rsi),
                                "bb_pos": round(bb_pos, 1), "adx": round(adx),
                                "vol_ratio": round(volume / vol_avg, 2) if vol_avg else 0,
                                "signal": is_signal, "btc_rsi": round(_btc_rsi),
                            }) + '\n')
                    except: pass

                if is_signal:
                    signals.append({
                        'symbol': sym, 'price': px, 'rsi': rsi, 'adx': adx,
                        'atr': atr, 'bb_pos': bb_pos,
                        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_mid': bb_mid,
                    })

            except Exception as _e:
                continue

        # 시그널 종목 진입 (RSI 낮은 순 = 가장 약한 종목 우선)
        if signals:
            ranked = sorted(signals, key=lambda x: x['rsi'])
            _entered = 0

            for best in ranked:
                if _entered >= 2:  # 추세 숏 최대 2개
                    break
                sym = best['symbol']
                if _check_already_held(sym):
                    continue
                _today = datetime.now().strftime('%Y-%m-%d')
                if _daily_trades.get('date') == _today and _daily_trades.get('count', 0) >= MAX_DAILY_TRADES:
                    break

                # 추세숏 진입금: 잔고 20%, 최대 $30 × 킬존 부스트
                _bal = get_balance() or 100
                usdt = max(min(_bal * 0.20, 30) * _get_killzone_boost(), 15)
                lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
                try:
                    set_margin_type(sym, "ISOLATED")
                    set_leverage(sym, lev)
                    px = float(get_price(sym))

                    atr = best['atr']
                    atr_pct = atr / px * 100

                    # SL: ATR × 1.5 위 (BB mid가 너무 멀면 ATR 기반으로 대체)
                    _sl_mid = best['bb_mid']
                    _sl_atr = px + atr * 1.5
                    sl = _round_price_sym(sym, min(_sl_mid, _sl_atr))  # 더 가까운 쪽
                    # TP: ATR × 2.0 아래
                    tp = _round_price_sym(sym, px - atr * 2.0)
                    if tp >= px * 0.995 or sl <= px * 1.001:
                        continue

                    qty = _round_qty(sym, usdt * lev / px)
                    if qty <= 0 or qty * px < 20:
                        step, _ = _get_symbol_filters(sym)
                        qty = _round_qty(sym, 21.0 / px + float(step))
                    if qty <= 0:
                        continue

                    _sl_dist = abs(sl - px)
                    _tp_dist = abs(px - tp)
                    _rr = _tp_dist / _sl_dist if _sl_dist else 0
                    if _rr < 0.8:  # 최소 R:R 1:0.8 (추세 방향이므로 승률로 보상)
                        continue

                    # 시장가 숏 진입
                    order = client.futures_create_order(symbol=sym, side='SELL', type='MARKET',
                        quantity=str(qty), newClientOrderId=f'ts_{uuid.uuid4().hex[:16]}')
                    _positions_cache['ts'] = 0
                    _sl_ok = False
                    try:
                        client.futures_create_order(symbol=sym, side='BUY', type='STOP_MARKET',
                            stopPrice=str(sl), quantity=str(qty), reduceOnly=True,
                            newClientOrderId=f'tssl_{uuid.uuid4().hex[:12]}')
                        _sl_ok = True
                    except: pass
                    try:
                        client.futures_create_order(symbol=sym, side='BUY', type='TAKE_PROFIT_MARKET',
                            stopPrice=str(tp), quantity=str(qty), reduceOnly=True,
                            newClientOrderId=f'tstp_{uuid.uuid4().hex[:12]}')
                    except: pass
                    if not _sl_ok:
                        try:
                            client.futures_create_order(symbol=sym, side='BUY', type='MARKET',
                                quantity=str(qty), reduceOnly=True)
                        except: pass
                        continue

                    _sltp_done.add(sym)
                    _tp_cache[sym] = tp
                    _entry_time[sym] = time.time()
                    _entry_source[sym] = 'trend_short'
                    trade_db.add_trade({"symbol": sym, "side": "🔴 숏", "action": "진입",
                        "qty": qty, "price": px, "sl": sl, "tp": tp, "source": "trend_short",
                        "extra": json.dumps({"rsi": round(best['rsi']), "adx": round(best['adx']),
                                             "bb_pos": round(best['bb_pos']), "btc_rsi": round(_btc_rsi)})})
                    log(f"  📉 추세숏: {sym} @ {px} RSI={best['rsi']:.0f} SL={sl}(mid) TP={tp} R:R=1:{_rr:.1f}")
                    send_message(TG_TOKEN, TG_CHAT,
                        f"📉 <b>추세 숏</b> (BTC RSI {_btc_rsi:.0f})\n"
                        f"   {sym} @ ${px}\n"
                        f"   RSI={best['rsi']:.0f} BB pos={best['bb_pos']:.0f}%\n"
                        f"   SL ${sl} (BB mid) → TP ${tp}\n"
                        f"   R:R 1:{_rr:.1f}")
                    _institutional_post_entry(sym, 'trend_short')
                    _entered += 1

                except Exception as e:
                    log(f"  ❌ 추세숏 실패: {e}")

    except Exception as e:
        log(f"  추세숏 오류: {e}")


def check_contrarian_short():
    """
    역행 과매수 숏 — 하락장에서 혼자 오른 알트를 숏
    조건: BTC RSI < 50 (약세) + 종목 RSI > 60 + BB pos > 60% + ADX > 25
    근거: 하락장에서 역행 상승한 알트는 결국 시장에 끌려 내려옴
    SL: BB 상단 + 최소 2%
    TP: BB mid (과매수 해소 = 중간선 복귀)
    """
    global _contrarian_short_cache
    now = time.time()
    if now - _contrarian_short_cache['ts'] < 300:
        return
    _contrarian_short_cache['ts'] = now

    # 하락장 확인: BTC RSI < 45 (50→45, 더 확실한 하락장만)
    _btc_rsi = _btc_cache.get('rsi', 50)
    if _btc_rsi >= 45:
        return

    ok, reason = _institutional_guard()
    if not ok:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        symbols = [s for s in get_scan_universe() if s not in {'BTCUSDT'}]
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                px = float(get_price(sym))
                ind = calc_indicators(get_klines(sym, '1h', 50))
                rsi = ind.get('rsi', 50) or 50
                adx = ind.get('adx', 0) or 0
                atr = ind.get('atr', 0) or 0
                bb_upper = ind.get('bb_upper', 0) or 0
                bb_lower = ind.get('bb_lower', 0) or 0
                bb_mid = ind.get('bb_mid', 0) or 0

                if not (bb_upper > bb_lower > 0) or atr <= 0:
                    continue

                bb_pos = (px - bb_lower) / (bb_upper - bb_lower) * 100

                # 조건 강화 (AIA 패턴 학습: RSI 70, ADX 67, BB 75%에서 +8.5%)
                is_overbought = rsi > 70   # 65→70 (극단적 과매수만)
                is_high = bb_pos > 70      # 상단 근처
                is_trending = adx > 50     # 과도한 추세 = 되돌림 확률↑

                is_signal = is_overbought and is_high and is_trending

                if is_overbought and is_high:
                    try:
                        with open(CONTRARIAN_SHORT_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px, "rsi": round(rsi),
                                "bb_pos": round(bb_pos, 1), "adx": round(adx),
                                "signal": is_signal, "btc_rsi": round(_btc_rsi),
                            }) + '\n')
                    except: pass

                if is_signal:
                    signals.append({
                        'symbol': sym, 'price': px, 'rsi': rsi, 'adx': adx,
                        'atr': atr, 'bb_pos': bb_pos,
                        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_mid': bb_mid,
                    })

            except Exception:
                continue

        if signals:
            # RSI 높은 순 (가장 과매수인 종목 우선)
            ranked = sorted(signals, key=lambda x: -x['rsi'])

            for best in ranked[:1]:  # 최대 1건만 (보수적)
                sym = best['symbol']
                if _check_already_held(sym):
                    continue
                _today = datetime.now().strftime('%Y-%m-%d')
                if _daily_trades.get('date') == _today and _daily_trades.get('count', 0) >= MAX_DAILY_TRADES:
                    break

                # 역행숏 진입금: 잔고의 35%, 최대 $50
                _bal = get_balance() or 100
                usdt = max(min(_bal * 0.35, 50) * _get_killzone_boost(), 15)
                lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
                try:
                    set_margin_type(sym, "ISOLATED")
                    set_leverage(sym, lev)
                    px = float(get_price(sym))

                    atr = best['atr']
                    # LIMIT 진입가: BB 상단 -0.2% (꼭대기 근처에서만 체결)
                    limit_px = _round_price_sym(sym, best['bb_upper'] * 0.998)
                    if limit_px <= px:  # 이미 상단 위면 현재가로
                        limit_px = _round_price_sym(sym, px * 1.002)

                    # SL: LIMIT 진입가 + 최소 2%
                    _sl_bb = best['bb_upper'] * 1.003
                    _sl_min = limit_px * 1.02
                    sl = _round_price_sym(sym, max(_sl_bb, _sl_min))
                    # TP: BB mid (과매수 해소)
                    tp = _round_price_sym(sym, best['bb_mid'])
                    if tp >= limit_px * 0.995 or sl <= limit_px * 1.005:
                        continue

                    qty = _round_qty(sym, usdt * lev / limit_px)
                    if qty <= 0 or qty * limit_px < 20:
                        step, _ = _get_symbol_filters(sym)
                        qty = _round_qty(sym, 21.0 / limit_px + float(step))
                    if qty <= 0:
                        continue

                    _sl_dist = abs(sl - limit_px)
                    _tp_dist = abs(limit_px - tp)
                    _rr = _tp_dist / _sl_dist if _sl_dist else 0
                    if _rr < 0.8:
                        continue

                    # LIMIT 예약 (BB 상단까지 올라와야 체결 — 관망 후 진입)
                    order = client.futures_create_order(symbol=sym, side='SELL', type='LIMIT',
                        price=str(limit_px), quantity=str(qty), timeInForce='GTC',
                        newClientOrderId=f'cs_{uuid.uuid4().hex[:16]}')
                    _pending_fills[sym] = {
                        'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                        'side': 'SELL', 'entry': limit_px, 'score': 0, 'atr': atr,
                        'time': time.time(), 'source': 'contrarian_short', 'expire': time.time() + 1800,  # 30분 대기
                    }
                    log(f"  📦 역행숏 예약: {sym} @ {limit_px} (BB상단 대기) RSI={best['rsi']:.0f} R:R=1:{_rr:.1f}")
                    send_message(TG_TOKEN, TG_CHAT,
                        f"📦 <b>역행 숏 예약</b> (BB상단 대기)\n"
                        f"   {sym} @ ${limit_px}\n"
                        f"   RSI={best['rsi']:.0f} BB={best['bb_pos']:.0f}% ADX={best['adx']:.0f}\n"
                        f"   BTC RSI {_btc_rsi:.0f}\n"
                        f"   SL ${sl} → TP ${tp} | R:R 1:{_rr:.1f}")

                except Exception as e:
                    log(f"  ❌ 역행숏 실패: {e}")

    except Exception as e:
        log(f"  역행숏 오류: {e}")


def check_bb_short():
    """
    BB 박스 숏 — 횡보장에서 상단 매도 → 하단 청산
    조건: BB 폭 2.5~5.5% + 가격 상단 근처 + RSI 55~70 + MTF 3/3 합의
    BTC RSI > 55이면 스킵 (상승장 숏 금지) + R:R 최소 0.8
    """
    global _bb_short_cache
    now = time.time()
    if now - _bb_short_cache['ts'] < 300:  # 5분
        return
    _bb_short_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return

    # BTC 상승장 필터 — RSI 55 이상이면 숏 금지 (보수적)
    _btc_rsi = _btc_cache.get('rsi', 50)
    if _btc_rsi > 55:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        _bb_exclude = {'BTCUSDT'}
        symbols = [s for s in get_scan_universe() if s not in _bb_exclude]
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue
            # BB숏 동일 종목 4시간 쿨다운 (연속 손실 방지)
            if sym in _bb_short_cooldown and time.time() - _bb_short_cooldown[sym] < 14400:
                continue

            try:
                px = float(get_price(sym))
                _bb_top_count = 0
                _bb_1h = None

                for _tf in ['15m', '30m', '1h']:
                    _ind = calc_indicators(get_klines(sym, _tf, 50))
                    _bu = _ind.get('bb_upper', 0) or 0
                    _bl = _ind.get('bb_lower', 0) or 0
                    _bm = _ind.get('bb_mid', 0) or 0
                    if not (_bu > _bl > 0):
                        continue
                    _pos = (px - _bl) / (_bu - _bl) * 100
                    if 80 < _pos < 105:
                        _bb_top_count += 1
                    if _tf == '1h':
                        _bb_1h = {
                            'upper': _bu, 'lower': _bl, 'mid': _bm,
                            'pos': _pos, 'width': (_bu - _bl) / _bm * 100,
                            'rsi': _ind.get('rsi', 50) or 50,
                            'atr': _ind.get('atr', 0) or 0,
                        }

                if not _bb_1h:
                    continue
                rsi = _bb_1h['rsi']
                atr = _bb_1h['atr']
                bb_upper = _bb_1h['upper']
                bb_lower = _bb_1h['lower']
                bb_mid = _bb_1h['mid']
                bb_width = _bb_1h['width']
                bb_pos = _bb_1h['pos']
                if rsi != rsi: continue  # nan

                is_box = 1.5 < bb_width < 5.5  # 백테스트 1위와 동일
                is_mtf_top = _bb_top_count >= 3  # 3/3 전 TF 합의 (보수적)
                is_overbought_rsi = 60 < rsi < 70  # 55→60 (강한 과매수만)
                is_wide_enough = bb_width >= 3.0  # 2.5→3.0 (밴드 좁으면 R:R 불리)

                is_signal = is_box and is_wide_enough and is_mtf_top and is_overbought_rsi

                # 로그
                if is_box and (_bb_top_count >= 1 or rsi > 60):
                    try:
                        with open(BB_SHORT_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px,
                                "bb_width": round(bb_width, 2),
                                "bb_pos": round(bb_pos, 1),
                                "rsi": round(rsi),
                                "mtf_top": _bb_top_count,
                                "signal": is_signal,
                            }) + '\n')
                    except Exception:
                        pass

                if is_signal:
                    signals.append({
                        'symbol': sym, 'price': px, 'rsi': rsi,
                        'bb_width': bb_width, 'bb_pos': bb_pos,
                        'bb_upper': bb_upper, 'bb_lower': bb_lower,
                        'bb_mid': bb_mid, 'atr': atr,
                        'mtf_top': _bb_top_count,
                    })

            except Exception as _e:
                log(f"  ⚠️ BB숏 {sym} 스캔 오류: {str(_e)[:60]}")
                continue

        # 기존 LIMIT 예약 중 조건 벗어난 것 취소
        _sig_syms = {s['symbol'] for s in signals}
        for _sym in list(_bb_short_limit_orders.keys()):
            if _sym in held or _sym not in _sig_syms:
                try:
                    client.futures_cancel_order(symbol=_sym, orderId=_bb_short_limit_orders[_sym]['order_id'])
                    log(f"  📦 BB숏 {_sym} 예약 취소 (조건 벗어남)")
                except: pass
                _pending_fills.pop(_sym, None)
                del _bb_short_limit_orders[_sym]

        # 시그널 종목 진입 (bb_pos 높은 순)
        if signals:
            ranked = sorted(signals, key=lambda x: -x['bb_pos'])  # 상단에 가까울수록 우선
            _max_orders = min(2, MAX_ORDERS - len(held))  # 숏은 최대 2개
            _placed = len([s for s in _bb_short_limit_orders if s not in held])
            _entered = 0

            for best in ranked:
                if _entered + _placed >= _max_orders:
                    break
                sym = best['symbol']
                if _check_already_held(sym):
                    continue
                _today = datetime.now().strftime('%Y-%m-%d')
                if _daily_trades.get('date') == _today and _daily_trades.get('count', 0) >= MAX_DAILY_TRADES:
                    break

                usdt = max(BB_SHORT_USDT * (0.7 if _bear_mode else 1.0), 12)
                lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
                try:
                    set_margin_type(sym, "ISOLATED")
                    set_leverage(sym, lev)
                    px = float(get_price(sym))

                    atr = best.get('atr', 0) or 0
                    # SL: BB 상단 +0.3%, 진입가 대비 최소 1.5% 또는 ATR%
                    _sl_bb = best['bb_upper'] * 1.003
                    _atr_sl = best.get('atr', 0) or 0
                    _sl_min_pct = max(2.0, (_atr_sl / px * 100) if px else 2.0)
                    _sl_min = px * (1 + _sl_min_pct / 100)
                    sl = _round_price_sym(sym, max(_sl_bb, _sl_min))
                    # TP: BB 중간선 (하단→중간: 도달확률 향상, 부분익절+트레일링이 추가수익 커버)
                    tp = _round_price_sym(sym, best['bb_mid'])
                    if best['bb_mid'] >= px * 0.99:
                        continue  # mid가 현재가 가까우면 스킵
                    _mtf = best.get('mtf_top', 0)

                    # === 위치 > 90%: 시장가 즉시 ===
                    if best['bb_pos'] > 90:
                        if sym in _bb_short_limit_orders:
                            try: client.futures_cancel_order(symbol=sym, orderId=_bb_short_limit_orders[sym]['order_id'])
                            except: pass
                            _pending_fills.pop(sym, None)
                            del _bb_short_limit_orders[sym]
                        qty = _round_qty(sym, usdt * lev / px)
                        if qty <= 0 or qty * px < 20:
                            step, _ = _get_symbol_filters(sym)
                            qty = _round_qty(sym, 21.0 / px + float(step))
                        if qty <= 0: continue
                        _sl_dist = abs(sl - px)
                        _tp_dist = abs(px - tp)
                        _rr = _tp_dist / _sl_dist if _sl_dist else 0
                        if _rr < 0.8:
                            log(f"  ⛔ BB숏 R:R {_rr:.2f} < 0.8 → 스킵 ({sym})")
                            continue

                        order = client.futures_create_order(symbol=sym, side='SELL', type='MARKET', quantity=str(qty), newClientOrderId=f'bbs_{uuid.uuid4().hex[:16]}')
                        _positions_cache['ts'] = 0
                        _sl_ok = False
                        try:
                            client.futures_create_order(symbol=sym, side='BUY', type='STOP_MARKET',
                                stopPrice=str(sl), quantity=str(qty), reduceOnly=True, newClientOrderId=f'bbssl_{uuid.uuid4().hex[:12]}')
                            _sl_ok = True
                        except: pass
                        try:
                            client.futures_create_order(symbol=sym, side='BUY', type='TAKE_PROFIT_MARKET',
                                stopPrice=str(tp), quantity=str(qty), reduceOnly=True, newClientOrderId=f'bbstp_{uuid.uuid4().hex[:12]}')
                        except: pass
                        if not _sl_ok:
                            try:
                                client.futures_create_order(symbol=sym, side='BUY', type='MARKET', quantity=str(qty), reduceOnly=True)
                            except: pass
                            continue
                        _sltp_done.add(sym)
                        _tp_cache[sym] = tp
                        _entry_time[sym] = time.time()
                        _entry_source[sym] = 'bb_short'
                        try:
                            _fill = client.futures_get_order(symbol=sym, orderId=order['orderId'])
                            _avg = float(_fill.get('avgPrice', 0))
                            entry_px = _avg if _avg > 0 else px
                        except: entry_px = px
                        trade_db.add_trade({"symbol": sym, "side": "🔴 숏", "action": "진입",
                            "qty": qty, "price": entry_px, "sl": sl, "tp": tp, "source": "bb_short",
                            "extra": json.dumps({"bb_width": round(best['bb_width'], 1), "bb_pos": round(best['bb_pos'], 0), "mode": "market"})})
                        log(f"  🔻 BB숏 시장가: {sym} @ {entry_px} (위치{best['bb_pos']:.0f}%>90%) R:R=1:{_rr:.1f} MTF={_mtf}/3")
                        send_message(TG_TOKEN, TG_CHAT,
                            f"🔻 <b>BB 숏 진입</b>\n   {sym} @ ${entry_px} (상단 {best['bb_pos']:.0f}%)\n   MTF {_mtf}/3 | R:R 1:{_rr:.1f}")
                        _institutional_post_entry(sym, 'bb_short')
                        _entered += 1

                    # === 위치 80~90%: LIMIT 상단 예약 ===
                    else:
                        limit_px = _round_price_sym(sym, best['bb_upper'] * 0.998)
                        _sl_bb = best['bb_upper'] * 1.003
                        _atr_sl = best.get('atr', 0) or 0
                        _sl_min_pct = max(2.0, (_atr_sl / limit_px * 100) if limit_px else 2.0)
                        _sl_min = limit_px * (1 + _sl_min_pct / 100)
                        sl = _round_price_sym(sym, max(_sl_bb, _sl_min))
                        qty = _round_qty(sym, usdt * lev / limit_px)
                        if qty <= 0 or qty * limit_px < 20:
                            step, _ = _get_symbol_filters(sym)
                            qty = _round_qty(sym, 21.0 / limit_px + float(step))
                        if qty <= 0: continue
                        _sl_dist = abs(sl - limit_px)
                        _tp_dist = abs(limit_px - tp)
                        _rr = _tp_dist / _sl_dist if _sl_dist else 0
                        if _rr < 0.8:
                            log(f"  ⛔ BB숏 R:R {_rr:.2f} < 0.8 → 스킵 ({sym})")
                            continue

                        if sym in _bb_short_limit_orders:
                            _old = _bb_short_limit_orders[sym]
                            if abs(_old['price'] - limit_px) / limit_px < 0.01:
                                continue
                            try: client.futures_cancel_order(symbol=sym, orderId=_old['order_id'])
                            except: pass
                            _pending_fills.pop(sym, None)
                            del _bb_short_limit_orders[sym]

                        order = client.futures_create_order(symbol=sym, side='SELL', type='LIMIT',
                            price=str(limit_px), quantity=str(qty), timeInForce='GTC', newClientOrderId=f'bbsl_{uuid.uuid4().hex[:16]}')
                        _bb_short_limit_orders[sym] = {'order_id': order['orderId'], 'price': limit_px, 'qty': qty, 'sl': sl, 'tp': tp}
                        _pending_fills[sym] = {
                            'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                            'side': 'SELL', 'entry': limit_px, 'score': 0, 'atr': atr,
                            'time': time.time(), 'source': 'bb_short', 'expire': time.time() + 600,
                        }
                        log(f"  📦 BB숏 예약: {sym} @ {limit_px} (위치{best['bb_pos']:.0f}%) R:R=1:{_rr:.1f} MTF={_mtf}/3")
                        send_message(TG_TOKEN, TG_CHAT,
                            f"📦 <b>BB 숏 예약</b>\n   {sym} @ ${limit_px} (BB상단 대기)\n   MTF {_mtf}/3 | R:R 1:{_rr:.1f}")
                        _placed += 1
                except Exception as e:
                    log(f"  ❌ BB숏 진입 실패: {e}")

    except Exception as e:
        log(f"  BB숏 오류: {e}")


def check_cvd_divergence():
    """
    #L: CVD 다이버전스 — 가격 저점 갱신 but 매수 누적 증가 = 기관 매집
    Binance 캔들의 taker buy volume으로 CVD 계산
    """
    global _cvd_cache
    now = time.time()
    if now - _cvd_cache['ts'] < 300:  # 5분 캐시 (CVD 단독 운영 — 기회 극대화)
        return
    _cvd_cache['ts'] = now

    # CVD 시간대 필터 — 17~19시(KST) 승률 0~33%(완전 실패), 00~04시 100%(최고)
    _kst_hour = (datetime.now().hour)  # 서버 KST 가정
    if _kst_hour in (17, 18, 19):
        return  # CVD 17~19시 차단
    _cvd_time_boost = 1.3 if _kst_hour in (0, 1, 2, 3, 4) else 1.0

    ok, reason = _institutional_guard()
    if not ok:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        # BTC 하락 중이면 CVD 롱 스킵 (손실 11건 대부분 BTC 하락장)
        _btc_rsi_1h = _btc_cache.get('rsi', 50)
        if _btc_rsi_1h < 45:
            return

        # 대형 코인 CVD 제외 (반등폭 부족 → 소액 수익만): SOL 4건-$0.32, ETH 2건-$0.80, BNB 2건-$0.29
        _cvd_exclude = {'SOLUSDT', 'ETHUSDT', 'BNBUSDT'}
        symbols = [s for s in get_scan_universe() if s not in _cvd_exclude]
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST or sym in WEAK_SYMBOLS:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                # 1h 캔들 48개 (taker buy volume 포함)
                k = client.futures_klines(symbol=sym, interval='1h', limit=48)
                if len(k) < 30:
                    continue

                closes = [float(x[4]) for x in k]
                volumes = [float(x[5]) for x in k]
                taker_buy_vol = [float(x[9]) for x in k]  # taker buy base vol

                # CVD 계산: 매수량 - 매도량 누적
                cvd = []
                cum = 0
                for i in range(len(k)):
                    buy_v = taker_buy_vol[i]
                    sell_v = volumes[i] - buy_v
                    cum += (buy_v - sell_v)
                    cvd.append(cum)

                # 최근 12시간 내 가격 저점 vs CVD 저점 비교
                recent_closes = closes[-12:]
                recent_cvd = cvd[-12:]

                px = closes[-1]
                cvd_now = cvd[-1]
                cvd_12h_low = min(recent_cvd)
                price_12h_low = min(recent_closes)

                # 조건: 현재가가 12h 저점 근처 (-2% ~ +1%) + 저점이 최근 4시간 내 (신선도)
                _low_dist = (px - price_12h_low) / price_12h_low if price_12h_low > 0 else 999
                _low_idx = recent_closes.index(price_12h_low) if price_12h_low in recent_closes else 0
                _low_fresh = _low_idx >= len(recent_closes) - 8  # 최근 8캔들(=8h) 이내
                near_low = -0.02 < _low_dist < 0.01 and _low_fresh

                # CVD 상승 판정: 저점 대비 5% 회복
                if cvd_12h_low < 0:
                    cvd_rising = cvd_now > cvd_12h_low * 0.95  # 음수: -100→-95 = 5% 회복
                elif cvd_12h_low > 0:
                    cvd_rising = cvd_now > cvd_12h_low * 1.05  # 양수: 100→105 = 5% 증가
                else:
                    cvd_rising = cvd_now > 0

                # RSI 필터
                ind = calc_indicators(get_klines(sym, '1h', 50))
                rsi = ind.get('rsi', 50) or 50
                atr = ind.get('atr', 0) or 0  # ATR도 여기서 가져옴 (진입 시 재사용)

                # cvd_trend_up 제거 — 충족률 7%로 거래 차단. cvd_rising이 이미 매수세 확인
                is_signal = near_low and cvd_rising and rsi < 30  # 35→30 (극단적 과매도만)

                # 로그 (조건 부분 충족이라도 기록)
                if near_low or (rsi < 30 and cvd_rising):
                    try:
                        with open(CVD_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px,
                                "cvd_now": round(cvd_now, 2),
                                "cvd_12h_low": round(cvd_12h_low, 2),
                                "rsi": round(rsi), "near_low": near_low,
                                "cvd_rising": cvd_rising,
                                "signal": is_signal,
                                "entered": False,
                            }) + '\n')
                    except Exception:
                        pass  # 로그 실패는 무시 (진입에 영향 없음)

                if is_signal:
                    # 24시간 거래량 필터 — 저유동성 코인 진입 방지 (BSB -$9.23 사례)
                    _vol_24h = sum(volumes[-24:]) * px  # 24h USDT 거래량 추정
                    if _vol_24h < 100_000_000:  # $1억 미만 → 스킵 (5000만→1억, 유동성 강화)
                        log(f"  ⚠️ CVD {sym} 저유동성 스킵 (24h vol ${_vol_24h/1e6:.0f}M)")
                        continue
                    # cvd_delta를 % 정규화 (거래량 큰 종목 편향 방지)
                    _cvd_delta_pct = abs(cvd_now - cvd_12h_low) / abs(cvd_12h_low) * 100 if cvd_12h_low != 0 else 0
                    # CVD delta 최소 임계값 — delta↔PnL 상관 r=+0.789, 약한 시그널 필터
                    if _cvd_delta_pct < 5.0:
                        log(f"  ⚠️ CVD {sym} 약한 시그널 스킵 (delta {_cvd_delta_pct:.1f}% < 5%)")
                        continue
                    # OI+가격 방향 조합 보너스 (OI증가+가격하락 = 숏축적 → 숏스퀴즈 롱 유리)
                    _oi_bonus = 0
                    try:
                        _oi = get_oi_change(sym, '15m', 3)
                        if _oi and _oi.get('change_pct', 0) > 3 and rsi < 35:
                            _oi_bonus = 5  # OI 증가 + 과매도 = 숏 축적 중 → 롱 보너스
                    except:
                        pass

                    signals.append({
                        'symbol': sym, 'price': px, 'rsi': rsi,
                        'cvd_delta': round(_cvd_delta_pct, 1),
                        'atr': atr,
                        'oi_bonus': _oi_bonus,
                    })

            except Exception as _e:
                log(f"  ⚠️ CVD {sym} 스캔 오류: {str(_e)[:60]}")
                continue

        # 펀딩비 연속 극단 보너스 (3회 연속 -0.03% 이하 = 숏 과밀 → 롱 우선순위 UP)
        for sig in signals:
            try:
                _fr_hist = get_funding_rate_history(sig['symbol'], limit=4)
                if _fr_hist and len(_fr_hist) >= 3:
                    _fr_vals = [float(f['fundingRate']) for f in _fr_hist[-3:]]
                    if all(v < -0.0003 for v in _fr_vals):
                        sig['funding_boost'] = -15  # 낮을수록 우선순위↑ (정렬 기준)
                        log(f"  💰 {sig['symbol']} 펀딩비 3연속 극음수 → 숏스퀴즈 보너스")
                    else:
                        sig['funding_boost'] = 0
                else:
                    sig['funding_boost'] = 0
            except:
                sig['funding_boost'] = 0

        # 시그널 중 최적 진입 (슬롯 여유 시 최대 2개)
        if signals and len(held) < MAX_ORDERS:
            ranked = sorted(signals, key=lambda x: x['rsi'] - x['cvd_delta'] * 0.1 + x.get('funding_boost', 0))
            # 슬롯 여유만큼 진입 (최대 2개)
            _max_entry = min(2, MAX_ORDERS - len(held))
            _entered = 0
            for best in ranked:
                if _entered >= _max_entry:
                    break
                sym = best['symbol']
                if _check_already_held(sym):
                    continue
                # 진입 전 일일 한도 재확인
                _today = datetime.now().strftime("%Y-%m-%d")
                if _daily_trades.get("date") == _today and _daily_trades.get("count", 0) >= MAX_DAILY_TRADES:
                    break
                log(f"  📊 CVD 다이버전스 감지: {sym} RSI={best['rsi']:.0f} CVD↑{best['cvd_delta']}%")

                # CVD 진입금: 잔고의 45% (복리), 최대 $65, 최소 $15
                _bal = get_balance() or 100
                usdt = max(min(_bal * 0.45, 65) * (0.7 if _bear_mode else 1.0) * _cvd_time_boost * _get_killzone_boost(), 15)
                lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
                try:
                    set_margin_type(sym, "ISOLATED")
                    set_leverage(sym, lev)
                    px = float(get_price(sym))
                    # ATR 기반 진입 오프셋 (체결률 개선: 오프셋 축소로 10분 내 체결 확률 향상)
                    _atr_tmp = best.get('atr', 0) or 0
                    _offset = min(max(_atr_tmp * 0.2 / px if px else 0.003, 0.001), 0.005)  # 0.1~0.5%
                    limit_px = _round_price_sym(sym, px * (1 - _offset))

                    qty = _round_qty(sym, usdt * lev / px)
                    # 노셔널 $20 최소 보장 (LIMIT 가격 기준)
                    if qty <= 0 or qty * limit_px < 20:
                        step, _ = _get_symbol_filters(sym)
                        qty = _round_qty(sym, 21.0 / limit_px + float(step))
                    if qty <= 0:
                        continue

                    # ATR: 시그널 스캔에서 이미 계산한 값 재사용
                    atr = best.get('atr', 0) or 0
                    if atr <= 0:
                        _ind = calc_indicators(get_klines(sym, '1h', 50))
                        atr = _ind.get('atr', 0) or 0
                    atr_pct = atr / px * 100 if px else 1

                    _sl_mult = 2.0 if _btc_cache.get('rsi', 50) < 45 else 1.5  # 하락장 SL 넓게
                    sl_pct = max(atr_pct * _sl_mult, 2.0 / lev)
                    tp_pct = max(atr_pct * 4.0, sl_pct * 2.5)  # TP 확대 (TP효율 119% → 초과 이동분 캡처)
                    sl = _round_price_sym(sym, limit_px * (1 - sl_pct / 100))  # SL/TP도 LIMIT 가격 기준
                    tp = _round_price_sym(sym, limit_px * (1 + tp_pct / 100))

                    order = client.futures_create_order(
                        symbol=sym, side='BUY', type='LIMIT',
                        price=str(limit_px),
                        quantity=str(qty), timeInForce='GTC'
                    )
                    _pending_fills[sym] = {
                        'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                        'side': 'BUY', 'entry': limit_px, 'score': 0, 'atr': atr,
                        'time': time.time(), 'source': 'cvd_divergence',
                        'expire': time.time() + 600,  # 5→10분 (체결률 개선)
                        'cvd_delta': best.get('cvd_delta', 0),  # delta 기록 (필터 효과 검증용)
                    }
                    # DB 기록은 check_fills() 체결 시에만 (LIMIT 미체결 유령 방지)
                    _rr = tp_pct / sl_pct if sl_pct else 0
                    log(f"  ✅ CVD 롱 진입: {sym} ${usdt:.0f} @ {limit_px} SL={sl} TP={tp} R:R=1:{_rr:.1f}")
                    send_message(TG_TOKEN, TG_CHAT,
                        f"📊 <b>CVD 다이버전스 롱</b>\n"
                        f"   {sym} @ ${limit_px}\n"
                        f"   RSI={best['rsi']:.0f} | CVD↑{best['cvd_delta']:.0f}%\n"
                        f"   노셔널 ${qty * limit_px:.1f} (${usdt}×{lev}x)\n"
                        f"   SL ${sl} ({sl_pct:.1f}%) → TP ${tp} ({tp_pct:.1f}%)\n"
                        f"   R:R = 1:{_rr:.1f}")
                    _institutional_post_entry(sym, 'cvd_divergence')
                    _entered += 1
                except Exception as e:
                    log(f"  ❌ CVD 진입 실패: {e}")

    except Exception as e:
        log(f"  CVD 오류: {e}")


def check_short_squeeze():
    """
    #M: OI 증가 + 롱숏비율 극단 = 숏 스퀴즈 감지
    Binance API: futures_top_longshort_account_ratio + futures_open_interest_hist
    """
    global _squeeze_cache
    now = time.time()
    if now - _squeeze_cache['ts'] < 600:  # 10분 캐시
        return
    _squeeze_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        symbols = get_scan_universe()
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                # 롱숏 비율 (상위 트레이더 계정 기준)
                ls = client.futures_top_longshort_account_ratio(symbol=sym, period='1h', limit=4)
                if not ls:
                    continue
                latest_ratio = float(ls[-1]['longShortRatio'])  # <1이면 숏 우세

                # OI 변화 (4시간)
                oi_hist = client.futures_open_interest_hist(symbol=sym, period='1h', limit=4)
                if len(oi_hist) < 4:
                    continue
                oi_now = float(oi_hist[-1]['sumOpenInterestValue'])
                oi_4h = float(oi_hist[0]['sumOpenInterestValue'])
                oi_change = (oi_now - oi_4h) / oi_4h * 100 if oi_4h else 0

                # RSI
                df = get_klines(sym, '1h', 50)
                ind = calc_indicators(df)
                rsi = ind.get('rsi', 50) or 50
                px = float(get_price(sym))

                # 숏 스퀴즈 조건: 숏 과밀(ratio<0.8) + OI 증가(+1.5%+) + RSI<40
                is_squeeze = latest_ratio < 0.8 and oi_change > 1.5 and rsi < 40

                # 로그 (ratio<1이면 기록)
                if latest_ratio < 1.0 or oi_change > 3:
                    try:
                        with open(SQUEEZE_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px,
                                "ls_ratio": round(latest_ratio, 3),
                                "oi_change_4h": round(oi_change, 2),
                                "rsi": round(rsi),
                                "signal": is_squeeze,
                            }) + '\n')
                    except:
                        pass

                if is_squeeze:
                    signals.append({
                        'symbol': sym, 'price': px, 'rsi': rsi,
                        'ls_ratio': latest_ratio, 'oi_change': oi_change,
                    })

            except:
                continue

        # 가장 강한 스퀴즈 1개 진입
        if signals and len(held) < MAX_ORDERS:
            best = sorted(signals, key=lambda x: x['ls_ratio'])[0]
            sym = best['symbol']
            if _check_already_held(sym):
                return
            log(f"  🔥 숏 스퀴즈 감지: {sym} 롱숏={best['ls_ratio']:.2f} OI+{best['oi_change']:.1f}% RSI={best['rsi']:.0f}")

            usdt = max(ALT_USDT * (0.6 if _bear_mode else 0.7), 12)
            lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
            try:
                set_margin_type(sym, "ISOLATED")
                set_leverage(sym, lev)
                px = float(get_price(sym))
                qty = _round_qty(sym, usdt * lev / px)
                if qty <= 0 or qty * px < 5:
                    return

                df = get_klines(sym, '1h', 50)
                ind = calc_indicators(df)
                atr = ind.get('atr', 0) or 0
                atr_pct = atr / px * 100 if px else 1

                sl_pct = max(atr_pct * 1.5, 2.0 / lev)
                tp_pct = max(atr_pct * 3.5, sl_pct * 2.5)
                sl = _round_price_sym(sym, px * (1 - sl_pct / 100))
                tp = _round_price_sym(sym, px * (1 + tp_pct / 100))

                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='LIMIT',
                    price=str(_round_price_sym(sym, px * 0.999)),
                    quantity=str(qty), timeInForce='GTC'
                )
                _pending_fills[sym] = {
                    'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                    'side': 'BUY', 'entry': px, 'score': 0, 'atr': atr,
                    'time': time.time(), 'source': 'short_squeeze',
                    'expire': time.time() + 300,
                }
                trade_db.add_trade({"symbol": sym, "side": "🟢 롱", "action": "진입",
                                    "qty": qty, "price": px, "sl": sl, "tp": tp,
                                    "source": "short_squeeze",
                                    "extra": json.dumps({"ls_ratio": best['ls_ratio'], "oi_change": round(best['oi_change'], 1)})})
                log(f"  ✅ 스퀴즈 롱: {sym} ${usdt:.0f} SL={sl} TP={tp}")
                send_message(TG_TOKEN, TG_CHAT, f"🔥 숏 스퀴즈 롱\n{sym} 롱숏={best['ls_ratio']:.2f}\nOI+{best['oi_change']:.1f}%\nSL={sl} TP={tp}")
                _institutional_post_entry(sym, 'short_squeeze')
            except Exception as e:
                log(f"  ❌ 스퀴즈 진입 실패: {e}")

    except Exception as e:
        log(f"  스퀴즈 오류: {e}")


def check_mtf_confluence():
    """
    #N: 멀티 타임프레임 컨플루언스 — 4h+1h+15m 동시 과매도 = 강력 반등
    조건 까다로워서 빈도 낮지만 승률 높음
    """
    global _mtf_cache
    now = time.time()
    if now - _mtf_cache['ts'] < 600:  # 10분 캐시
        return
    _mtf_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        symbols = get_scan_universe()
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                # 3개 타임프레임 지표
                i15 = calc_indicators(get_klines(sym, '15m', 50))
                i1h = calc_indicators(get_klines(sym, '1h', 50))
                i4h = calc_indicators(get_klines(sym, '4h', 50))

                rsi15 = i15.get('rsi', 50) or 50
                rsi1h = i1h.get('rsi', 50) or 50
                rsi4h = i4h.get('rsi', 50) or 50

                bb_lower = i1h.get('bb_lower', 0) or 0
                px = float(get_price(sym))

                # 볼린저 하단 이탈
                below_bb = px < bb_lower if bb_lower else False

                # 컨플루언스: 모든 TF 과매도
                # MTF 조건 완화: 4h<35 + 1h<30 + 15m<25 + BB하단
                is_signal = rsi4h < 35 and rsi1h < 30 and rsi15 < 25 and below_bb

                # 로그 (2개 이상 과매도면 기록)
                oversold_count = (1 if rsi4h < 35 else 0) + (1 if rsi1h < 30 else 0) + (1 if rsi15 < 25 else 0)
                if oversold_count >= 2:
                    try:
                        with open(MTF_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px,
                                "rsi_4h": round(rsi4h), "rsi_1h": round(rsi1h),
                                "rsi_15m": round(rsi15), "below_bb": below_bb,
                                "oversold_count": oversold_count,
                                "signal": is_signal,
                            }) + '\n')
                    except:
                        pass

                if is_signal:
                    signals.append({
                        'symbol': sym, 'price': px,
                        'rsi4h': rsi4h, 'rsi1h': rsi1h, 'rsi15': rsi15,
                    })

            except:
                continue

        # 가장 과매도인 것 1개 진입
        if signals and len(held) < MAX_ORDERS:
            best = sorted(signals, key=lambda x: x['rsi1h'])[0]
            sym = best['symbol']
            if _check_already_held(sym):
                return
            log(f"  🎯 MTF 컨플루언스: {sym} RSI 4h={best['rsi4h']:.0f} 1h={best['rsi1h']:.0f} 15m={best['rsi15']:.0f}")

            usdt = max(ALT_USDT * 0.8, 12)  # MTF는 고승률이라 좀 더
            lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
            try:
                set_margin_type(sym, "ISOLATED")
                set_leverage(sym, lev)
                px = float(get_price(sym))
                qty = _round_qty(sym, usdt * lev / px)
                if qty <= 0 or qty * px < 5:
                    return

                atr = (calc_indicators(get_klines(sym, '1h', 50)).get('atr', 0) or 0)
                atr_pct = atr / px * 100 if px else 1
                sl_pct = max(atr_pct * 1.2, 1.5 / lev)
                tp_pct = max(atr_pct * 3.0, sl_pct * 2.5)
                sl = _round_price_sym(sym, px * (1 - sl_pct / 100))
                tp = _round_price_sym(sym, px * (1 + tp_pct / 100))

                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='LIMIT',
                    price=str(_round_price_sym(sym, px * 0.999)),
                    quantity=str(qty), timeInForce='GTC'
                )
                _pending_fills[sym] = {
                    'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                    'side': 'BUY', 'entry': px, 'score': 0, 'atr': atr,
                    'time': time.time(), 'source': 'mtf_confluence',
                    'expire': time.time() + 300,
                }
                trade_db.add_trade({"symbol": sym, "side": "🟢 롱", "action": "진입",
                                    "qty": qty, "price": px, "sl": sl, "tp": tp,
                                    "source": "mtf_confluence",
                                    "extra": json.dumps({"rsi4h": round(best['rsi4h']), "rsi1h": round(best['rsi1h']), "rsi15": round(best['rsi15'])})})
                log(f"  ✅ MTF 롱: {sym} ${usdt:.0f} SL={sl} TP={tp}")
                send_message(TG_TOKEN, TG_CHAT, f"🎯 MTF 컨플루언스 롱\n{sym}\nRSI 4h={best['rsi4h']:.0f}/1h={best['rsi1h']:.0f}/15m={best['rsi15']:.0f}\nSL={sl} TP={tp}")
                _institutional_post_entry(sym, 'mtf_confluence')
            except Exception as e:
                log(f"  ❌ MTF 진입 실패: {e}")

    except Exception as e:
        log(f"  MTF 오류: {e}")


def check_vwap_reversion():
    """
    #O: VWAP 평균회귀 — 가격이 VWAP-1.5σ 이하 = 과매도 → 롱
    VWAP = Σ(가격×거래량) / Σ(거래량), 24시간 기준
    대형 종목만 허용 (소형 알트 유동성 부족 → PIPPIN -$1.10 손실)
    """
    global _vwap_cache
    now = time.time()
    if now - _vwap_cache['ts'] < 600:  # 10분 캐시
        return
    _vwap_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        # 대형 종목만 (소형 알트 VWAP 이탈 후 복귀 안 함)
        _vwap_allowed = {'ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT'}
        symbols = [s for s in get_scan_universe() if s in _vwap_allowed]
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                # 15분 캔들 96개 = 24시간
                k = client.futures_klines(symbol=sym, interval='15m', limit=96)
                if len(k) < 50:
                    continue

                # VWAP 계산
                typical_prices = [(float(x[2]) + float(x[3]) + float(x[4])) / 3 for x in k]
                volumes = [float(x[5]) for x in k]
                cum_tp_vol = 0
                cum_vol = 0
                vwap_list = []
                for tp, vol in zip(typical_prices, volumes):
                    cum_tp_vol += tp * vol
                    cum_vol += vol
                    vwap_list.append(cum_tp_vol / cum_vol if cum_vol else tp)

                vwap = vwap_list[-1]
                px = float(k[-1][4])  # 현재 종가

                # VWAP 표준편차 (최근 24h typical price 기준)
                diffs = [(tp - vwap_list[i]) ** 2 for i, tp in enumerate(typical_prices)]
                std = (sum(diffs) / len(diffs)) ** 0.5

                if std == 0:
                    continue

                zscore = (px - vwap) / std
                deviation_pct = (px - vwap) / vwap * 100

                # RSI 확인
                df = get_klines(sym, '1h', 50)
                ind = calc_indicators(df)
                rsi = ind.get('rsi', 50) or 50

                # 시그널: VWAP-1.5σ 이하 + RSI<45 (완화)
                is_signal = zscore < -1.5 and rsi < 45

                # 로그 (1.5σ 이탈이면 기록)
                if abs(zscore) > 1.5:
                    try:
                        with open(VWAP_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px,
                                "vwap": round(vwap, 4),
                                "zscore": round(zscore, 2),
                                "dev_pct": round(deviation_pct, 2),
                                "rsi": round(rsi),
                                "signal": is_signal,
                            }) + '\n')
                    except:
                        pass

                if is_signal:
                    signals.append({
                        'symbol': sym, 'price': px, 'vwap': vwap,
                        'zscore': zscore, 'rsi': rsi,
                    })

            except:
                continue

        # VWAP 이탈 가장 큰 것 1개 진입
        if signals and len(held) < MAX_ORDERS:
            best = sorted(signals, key=lambda x: x['zscore'])[0]
            sym = best['symbol']
            if _check_already_held(sym):
                return
            log(f"  📉 VWAP 회귀: {sym} z={best['zscore']:.1f} VWAP={best['vwap']:.4f} RSI={best['rsi']:.0f}")

            usdt = max(ALT_USDT * (0.6 if _bear_mode else 0.7), 12)
            lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
            try:
                set_margin_type(sym, "ISOLATED")
                set_leverage(sym, lev)
                px = float(get_price(sym))
                qty = _round_qty(sym, usdt * lev / px)
                if qty <= 0 or qty * px < 5:
                    return

                atr = (calc_indicators(get_klines(sym, '1h', 50)).get('atr', 0) or 0)
                atr_pct = atr / px * 100 if px else 1
                sl_pct = max(atr_pct * 1.5, 2.0 / lev)
                sl = _round_price_sym(sym, px * (1 - sl_pct / 100))
                # TP: VWAP 복귀 또는 최소 R:R 1.5 보장
                _tp_vwap = best['vwap']
                _tp_min = px * (1 + sl_pct * 1.5 / 100)  # 최소 R:R 1.5
                tp = _round_price_sym(sym, max(_tp_vwap, _tp_min))

                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='LIMIT',
                    price=str(_round_price_sym(sym, px * 0.999)),
                    quantity=str(qty), timeInForce='GTC'
                )
                _pending_fills[sym] = {
                    'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                    'side': 'BUY', 'entry': px, 'score': 0, 'atr': atr,
                    'time': time.time(), 'source': 'vwap_reversion',
                    'expire': time.time() + 300,
                }
                trade_db.add_trade({"symbol": sym, "side": "🟢 롱", "action": "진입",
                                    "qty": qty, "price": px, "sl": sl, "tp": tp,
                                    "source": "vwap_reversion",
                                    "extra": json.dumps({"vwap": round(best['vwap'], 4), "zscore": round(best['zscore'], 2)})})
                log(f"  ✅ VWAP 롱: {sym} ${usdt:.0f} SL={sl} TP={tp}")
                send_message(TG_TOKEN, TG_CHAT, f"📉 VWAP 평균회귀 롱\n{sym} z={best['zscore']:.1f}\nVWAP={best['vwap']:.4f}\nSL={sl} TP={tp}")
                _institutional_post_entry(sym, 'vwap_reversion')
            except Exception as e:
                log(f"  ❌ VWAP 진입 실패: {e}")

    except Exception as e:
        log(f"  VWAP 오류: {e}")


def check_volume_profile():
    """
    #P: 거래량 프로파일 — POC(최다 거래 가격대) 지지 감지
    가격이 POC 아래에서 POC 방향 회귀 시 롱
    """
    global _vpoc_cache
    now = time.time()
    if now - _vpoc_cache['ts'] < 1800:  # 30분 캐시
        return
    _vpoc_cache['ts'] = now

    ok, reason = _institutional_guard()
    if not ok:
        return

    try:
        client = get_client()
        held = get_held_symbols()
        if len(held) >= MAX_ORDERS:
            return

        symbols = get_scan_universe()
        signals = []

        for sym in symbols:
            if sym in held or sym in BLACKLIST:
                continue
            if sym in _cooldown and time.time() < _cooldown[sym]:
                continue

            try:
                # 1h 캔들 72개 = 3일
                k = client.futures_klines(symbol=sym, interval='1h', limit=72)
                if len(k) < 48:
                    continue

                closes = [float(x[4]) for x in k]
                highs = [float(x[2]) for x in k]
                lows = [float(x[3]) for x in k]
                volumes = [float(x[5]) for x in k]

                # 가격대별 거래량 히스토그램 (50 bins)
                price_min = min(lows)
                price_max = max(highs)
                if price_max == price_min:
                    continue

                n_bins = 50
                bin_size = (price_max - price_min) / n_bins
                vol_profile = [0.0] * n_bins

                for i in range(len(k)):
                    # 캔들이 걸치는 bin에 거래량 분배
                    low_bin = int((lows[i] - price_min) / bin_size)
                    high_bin = int((highs[i] - price_min) / bin_size)
                    low_bin = max(0, min(low_bin, n_bins - 1))
                    high_bin = max(0, min(high_bin, n_bins - 1))
                    n_covered = high_bin - low_bin + 1
                    for b in range(low_bin, high_bin + 1):
                        vol_profile[b] += volumes[i] / n_covered

                # POC = 최다 거래량 가격대
                poc_bin = vol_profile.index(max(vol_profile))
                poc_price = price_min + (poc_bin + 0.5) * bin_size

                px = closes[-1]

                # HVN 위아래 구분 (POC 기준)
                below_poc = px < poc_price
                dist_from_poc = (poc_price - px) / poc_price * 100  # 양수면 아래

                # RSI
                df = get_klines(sym, '1h', 50)
                ind = calc_indicators(df)
                rsi = ind.get('rsi', 50) or 50

                # 시그널: POC 아래 0.5~5% + RSI<45 + 상승 모멘텀 (직전 3캔들 상승)
                recent_up = closes[-1] > closes[-3] if len(closes) >= 3 else False
                is_signal = 0.5 < dist_from_poc < 5.0 and rsi < 45 and recent_up

                # 로그
                if below_poc and dist_from_poc > 0.5:
                    try:
                        with open(VPOC_LOG, 'a') as f:
                            f.write(json.dumps({
                                "time": datetime.now().strftime('%Y-%m-%d %H:%M'),
                                "symbol": sym, "price": px,
                                "poc": round(poc_price, 4),
                                "dist_pct": round(dist_from_poc, 2),
                                "rsi": round(rsi),
                                "recent_up": recent_up,
                                "signal": is_signal,
                            }) + '\n')
                    except:
                        pass

                if is_signal:
                    signals.append({
                        'symbol': sym, 'price': px, 'poc': poc_price,
                        'dist': dist_from_poc, 'rsi': rsi,
                    })

            except:
                continue

        # POC 가장 가까운 것 1개 진입
        if signals and len(held) < MAX_ORDERS:
            best = sorted(signals, key=lambda x: x['dist'])[0]
            sym = best['symbol']
            if _check_already_held(sym):
                return
            log(f"  📊 VP POC: {sym} POC={best['poc']:.4f} 거리={best['dist']:.1f}% RSI={best['rsi']:.0f}")

            usdt = max(ALT_USDT * (0.6 if _bear_mode else 0.7), 12)
            lev = ETH_LEV if sym == 'ETHUSDT' else ALT_LEV
            try:
                set_margin_type(sym, "ISOLATED")
                set_leverage(sym, lev)
                px = float(get_price(sym))
                qty = _round_qty(sym, usdt * lev / px)
                if qty <= 0 or qty * px < 5:
                    return

                # SL: ATR 기반, TP: POC (최소 R:R 1.5 보장)
                atr = (calc_indicators(get_klines(sym, '1h', 50)).get('atr', 0) or 0)
                atr_pct = atr / px * 100 if px else 1
                sl_pct = max(atr_pct * 1.5, 2.0 / lev)
                sl = _round_price_sym(sym, px * (1 - sl_pct / 100))
                _tp_poc = best['poc']
                _tp_min = px * (1 + sl_pct * 1.5 / 100)  # 최소 R:R 1.5
                tp = _round_price_sym(sym, max(_tp_poc, _tp_min))

                order = client.futures_create_order(
                    symbol=sym, side='BUY', type='LIMIT',
                    price=str(_round_price_sym(sym, px * 0.999)),
                    quantity=str(qty), timeInForce='GTC'
                )
                _pending_fills[sym] = {
                    'order_id': order['orderId'], 'sl': sl, 'tp': tp,
                    'side': 'BUY', 'entry': px, 'score': 0, 'atr': atr,
                    'time': time.time(), 'source': 'volume_profile',
                    'expire': time.time() + 300,
                }
                trade_db.add_trade({"symbol": sym, "side": "🟢 롱", "action": "진입",
                                    "qty": qty, "price": px, "sl": sl, "tp": tp,
                                    "source": "volume_profile",
                                    "extra": json.dumps({"poc": round(best['poc'], 4), "dist": round(best['dist'], 2)})})
                log(f"  ✅ VP 롱: {sym} ${usdt:.0f} SL={sl} TP={tp}")
                send_message(TG_TOKEN, TG_CHAT, f"📊 거래량프로파일 롱\n{sym}\nPOC={best['poc']:.4f} ({best['dist']:.1f}% 아래)\nSL={sl} TP={tp}")
                _institutional_post_entry(sym, 'volume_profile')
            except Exception as e:
                log(f"  ❌ VP 진입 실패: {e}")

    except Exception as e:
        log(f"  VP 오류: {e}")


def _round_qty(symbol, qty):
    """수량을 심볼 규격에 맞게 반올림 (부동소수점 오차 방지)"""
    try:
        from math import floor
        step, _ = _get_symbol_filters(symbol)
        step = float(step)
        # step의 소수점 자릿수 계산
        _dec = len(str(step).rstrip('0').split('.')[-1]) if '.' in str(step) else 0
        return round(floor(qty / step) * step, _dec)
    except Exception:
        return round(qty, 3)


def _round_price_sym(symbol, price):
    """심볼명으로 tick_size 조회 후 _round_price 호출"""
    try:
        _, tick = _get_symbol_filters(symbol)
        return _round_price(float(price), float(tick))
    except Exception:
        return round(float(price), 4)


def update_cycle():
    """자동 스캔 → 상위 종목 선택 → 주문 업데이트"""
    bal = get_balance()
    log(f"--- 사이클 | 잔고 ${bal['total']:.2f} (가용 ${bal['available']:.2f}) ---")

    held = get_held_symbols()
    held_count = len(held)

    # 시그널 큐 우선 처리 (공지/급등/펀딩비 등 외부 감지기 시그널)
    _signals = pop_signals()
    _signal_syms = set()
    if _signals:
        for sig in _signals:
            _signal_syms.add(sig['symbol'])
            log(f"  [시그널큐] {sig['symbol']} ← {sig['source']} (우선순위={sig['priority']})")

    # ========== updater 스캔/진입 OFF (CVD 전략만 운영) ==========
    # 63건 51% -$17.69 → 거래할수록 손실. CVD(20건 55% +$1.39) 테스트 후 재활성화
    btc_up, btc_rsi = get_btc_trend()
    log(f"  BTC: {'UP' if btc_up else 'DOWN'} RSI={btc_rsi:.0f}")
    scored = []  # updater 스캔 OFF → 빈 리스트
    candidates = []
    top_n = []
    top_syms = set()
    slots = MAX_ORDERS - held_count

    scan_list = get_scan_universe()
    # 시그널 큐 종목을 스캔 리스트 맨 앞에 추가 (중복 제거)
    if _signal_syms:
        scan_list = list(_signal_syms - set(scan_list)) + scan_list
    log(f"  스캔 대상: {len(scan_list)}종목" + (f" (시그널 우선: {list(_signal_syms)})" if _signal_syms else ""))

    # 야간 진입 차단 (KST 00~04시)
    kst_hour = (datetime.now(tz=timezone.utc).hour + 9) % 24
    is_night = kst_hour in NIGHT_HOURS

    # 경제 이벤트 고변동 시간대 주의 (UTC 18:00=KST 03:00 FOMC, UTC 12:30=KST 21:30 CPI 등)
    # 정각 전후 5분은 변동성 급증 가능 → 신규 진입 억제 (기존 포지션은 유지)
    utc_min = datetime.now(tz=timezone.utc).minute
    is_event_risk = utc_min <= 5 or utc_min >= 55  # 정각 ±5분

    scored = []
    for sym in scan_list:
        if sym in held:
            log(f"  {sym}: 포지션 보유 → 스킵")
            continue
        if sym in BLACKLIST:
            continue
        if sym in _cooldown:
            if time.time() < _cooldown[sym]:
                continue
            else:
                del _cooldown[sym]  # 만료된 쿨다운 정리
        a = score_symbol(sym)
        if a.get('error'):
            continue
        scored.append(a)
        time.sleep(0.2)  # API 레이트 리밋

    # 점수 기준 내림차순 (방향 있는 것만)
    # #2: TAO 패턴 보너스 — 1h ADX 25+ & 4h 정배열 & ATR 1~5% → 정렬 가산점
    candidates = [a for a in scored if a['direction'] != 'wait']
    for c in candidates:
        c['_rank_bonus'] = 0
        _adx1h = c.get('adx_1h', 0) or 0
        _e20_4h = c.get('ema20_4h', 0) or 0
        _e50_4h = c.get('ema50_4h', 0) or 0
        _atr_pct = c.get('atr_pct', 0) or 0
        if _adx1h >= 25 and _e20_4h > _e50_4h:
            c['_rank_bonus'] += 5  # 추세 확인 종목 우선
        if 1.0 <= _atr_pct <= 5.0:
            c['_rank_bonus'] += 2  # 적정 변동성 (TAO 2.7%)
    # #146: 8시간+ 재진입 감점 (승률20%, -$9.67)
    for c in candidates:
        sym = c['symbol']
        try:
            _db = sqlite3.connect('/home/hyeok/01.APCC/00.ai-lab/trades.db')
            _row = _db.execute(
                "SELECT close_time FROM trades WHERE symbol=? AND close_time IS NOT NULL ORDER BY id DESC LIMIT 1",
                (sym,)).fetchone()
            _db.close()
            if _row:
                _last_close = datetime.fromisoformat(_row[0])
                _hours_since = (datetime.now() - _last_close).total_seconds() / 3600
                if _hours_since >= 8:
                    c['_rank_bonus'] = c.get('_rank_bonus', 0) - 5
                    c['score'] -= 2  # 점수도 직접 감점
        except:
            pass

    candidates.sort(key=lambda x: abs(x['score']) + x.get('_rank_bonus', 0), reverse=True)

    # 동일 방향 집중 제한 — 현재 포지션 + 대기 주문의 방향 카운트
    long_count = sum(1 for s in held if _get_held_direction(s) == 'long')
    short_count = sum(1 for s in held if _get_held_direction(s) == 'short')
    # 대기 주문도 카운트
    for pf in _pending_fills.values():
        if pf.get('side') == 'BUY': long_count += 1
        elif pf.get('side') == 'SELL': short_count += 1
    if long_count >= MAX_SAME_DIR:
        candidates = [c for c in candidates if c['direction'] != 'long']
        log(f"  ⚖️ 롱 {long_count}개 → 추가 롱 차단")
    if short_count >= MAX_SAME_DIR:
        candidates = [c for c in candidates if c['direction'] != 'short']
        log(f"  ⚖️ 숏 {short_count}개 → 추가 숏 차단")

    # 일일 거래 횟수 제한
    today = datetime.now().strftime('%Y-%m-%d')
    if _daily_trades['date'] != today:
        _daily_trades['date'] = today
        _daily_trades['count'] = 0
    daily_limit_hit = _daily_trades['count'] >= MAX_DAILY_TRADES

    # 연속 손실 시 진입금 축소 (2연패 50%, 3연패+ 진입 중단)
    global _consecutive_losses, _bear_stopped, _bear_daily_loss, _global_cooldown_until
    # #J/K: 하락장 일일 리셋
    if _bear_daily_loss['date'] != today:
        _bear_daily_loss = {'date': today, 'total': 0.0}
        _bear_stopped = False

    if _bear_stopped and _bear_mode:
        loss_block = True
    elif time.time() < _global_cooldown_until:
        loss_block = True  # 연패 시간 쿨다운 중
    elif _consecutive_losses >= 5:
        _bear_stopped = True  # 5연패+ → 당일 거래 정지 (승률 0%)
        loss_block = True
    elif _consecutive_losses >= 4:
        _global_cooldown_until = time.time() + 7200  # 4연패 → 2시간 쿨다운
        loss_block = True
    elif _consecutive_losses >= 3:
        _global_cooldown_until = time.time() + 1800  # 3연패 → 30분 쿨다운
        if _bear_mode:
            _bear_stopped = True
        loss_block = True
    else:
        loss_block = False

    # #8: BTC 급락 차단 (RSI < 35 = BTC 과매도 → 알트 동반 하락 위험)
    _, _btc_rsi = get_btc_trend()
    if _btc_rsi < 35:
        log(f"  🚨 BTC RSI {_btc_rsi:.0f} < 35 → 신규 진입 완전 차단")
        candidates = []
    elif is_night:
        log(f"  🌙 야간 {kst_hour}시 → 신규 진입 차단")
        candidates = []
    elif is_event_risk:
        log(f"  ⏰ 정각 전후 5분 → 진입 보류")
        candidates = []
    elif daily_limit_hit:
        log(f"  🛑 일일 거래 한도 {MAX_DAILY_TRADES}건 도달 → 진입 중단")
        candidates = []
    elif loss_block:
        if _bear_stopped:
            log(f"  🛑 {_consecutive_losses}연패 → 당일 거래 정지")
        elif time.time() < _global_cooldown_until:
            _remain = int(_global_cooldown_until - time.time())
            log(f"  🛑 연속 {_consecutive_losses}패 쿨다운 중 (잔여 {_remain//60}분{_remain%60}초)")
        else:
            log(f"  🛑 연속 {_consecutive_losses}패 → 쿨다운 시작")
        candidates = []

    # 전체 스캔 결과 로그
    for a in scored:
        flag = '🟢' if a['direction'] == 'long' else ('🔴' if a['direction'] == 'short' else '⚪')
        sb = a.get('sentiment_bonus', 0)
        sb_str = f" S{sb:+d}" if sb != 0 else ""
        rl_str = f" RL={a.get('rl_signal','?')}" if a.get('rl_bonus', 0) != 0 else ""
        rs_str = f" RS{a.get('rs_bonus',0):+d}" if a.get('rs_bonus', 0) != 0 else ""
        log(f"  {flag} {a['symbol']:12s} ${a.get('price',0):.4f} 점수={a['score']:+3d}{sb_str}{rl_str}{rs_str} RSI={a.get('rsi',0):.0f} ADX={a.get('adx',0):.0f}")

    # 알트 상관관계 필터: 같은 방향 후보가 3개+ 이면 점수 상위 2개만
    long_cands = [c for c in candidates if c['direction'] == 'long']
    short_cands = [c for c in candidates if c['direction'] == 'short']
    if len(long_cands) > 2:
        long_cands = long_cands[:2]
        log(f"  🔗 롱 후보 {len([c for c in candidates if c['direction']=='long'])}개 → 상관관계 필터 → 상위 2개만")
    if len(short_cands) > 2:
        short_cands = short_cands[:2]
        log(f"  🔗 숏 후보 {len([c for c in candidates if c['direction']=='short'])}개 → 상관관계 필터 → 상위 2개만")
    candidates = sorted(long_cands + short_cands, key=lambda x: abs(x['score']), reverse=True)

    # 상위 N개 선택 (점수 절대값 내림차순)
    client = get_client()
    top_n = candidates[:MAX_ORDERS - held_count] if candidates else []
    top_syms = {c['symbol'] for c in top_n}

    # 기존 LIMIT 주문 중 상위 N에 없는 것 정리 (SL/TP는 절대 건드리지 않음)
    try:
        open_orders = client.futures_get_open_orders()
        for o in open_orders:
            osym = o['symbol']
            otype = o.get('type', '')
            # SL/TP 주문은 보호 (STOP_MARKET, TAKE_PROFIT_MARKET)
            if otype in ('STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP', 'TAKE_PROFIT'):
                continue
            if osym not in top_syms and osym not in held:
                try:
                    client.futures_cancel_order(symbol=osym, orderId=o['orderId'])
                    log(f"  {osym}: 상위 탈락 → LIMIT 주문 취소")
                    _pending_fills.pop(osym, None)
                except Exception:
                    pass
    except Exception:
        pass

    # 슬롯 계산
    slots = MAX_ORDERS - held_count
    log(f"  슬롯: {slots}개 (보유 {held_count}/{MAX_ORDERS}) | 후보: {[c['symbol']+'('+str(c['score'])+')' for c in top_n]}")
    # ===== updater 진입 OFF — CVD만 운영 (63건 -$17.69 → 재검증 후 재활성화) =====
    log(f"  🚫 updater 진입 OFF — CVD 전략만 운영 중")
    if False:  # updater 진입 비활성화
    # =====
     if slots <= 0:
        log(f"  슬롯 없음")
     else:
        for a in top_n:
            sym = a['symbol']
            try:
                entry, sl, tp = calc_entry(a)
                if entry is None:
                    log(f"  ⚠️ {sym}: 진입가 계산 실패 (ATR=0?)")
                    continue

                is_eth = sym == 'ETHUSDT'
                base_usdt = ETH_USDT if is_eth else ALT_USDT
                lev = ETH_LEV if is_eth else ALT_LEV
                # 점수 비례 사이징: |점수| 5+ → 1.5배, 8+ → 2배
                # 점수 비례 사이징
                abs_sc = abs(a['score'])
                if abs_sc >= 8: size_mult = 2.0
                elif abs_sc >= 5: size_mult = 1.5
                else: size_mult = 1.0
                # 연속 손실 시 진입금 축소 (2연패 → 50%)
                if _consecutive_losses >= 2:
                    size_mult *= 0.5
                    log(f"  ⚠️ 연속 {_consecutive_losses}패 → 진입금 50% 축소")
                # 시장 모드별 진입금 조절
                if _bear_mode:
                    size_mult *= 0.6   # 하락장: 60%
                elif _bull_mode:
                    size_mult *= 1.5   # 상승장: 150%
                usdt = min(base_usdt * size_mult, ALT_USDT_MAX / lev)  # 노셔널 상한 기준 USDT 캡

                # #141 건당 최대 손실 캡 — SL 거리 기반 수량 조절
                sl_dist_pct = abs(entry - sl) / entry * 100 if entry and sl else 2.5
                # 노셔널 × SL거리% = 예상 손실. 예상 손실 ≤ MAX_LOSS_PER_TRADE
                # 노셔널 = usdt × lev, 예상 손실 = usdt × lev × sl_dist_pct / 100
                max_usdt_by_loss = MAX_LOSS_PER_TRADE / (lev * sl_dist_pct / 100) if sl_dist_pct > 0 else usdt
                if usdt > max_usdt_by_loss:
                    log(f"  💰 {sym} 손실캡: ${usdt:.1f}→${max_usdt_by_loss:.1f} (SL {sl_dist_pct:.1f}% × {lev}x → 최대 ${MAX_LOSS_PER_TRADE})")
                    usdt = max_usdt_by_loss

                side = 'BUY' if a['direction'] == 'long' else 'SELL'

                # #133: 2단계 진입 — 이전 사이클에서 미체결이면 오프셋 축소
                _narrowed = False
                if sym in _pending_fills and _pending_fills[sym].get('placed_at'):
                    elapsed = time.time() - _pending_fills[sym]['placed_at']
                    if elapsed >= 280 and not _pending_fills[sym].get('narrowed'):
                        atr_val = a.get('atr_1h', 0) or 0
                        if atr_val > 0:
                            entry = a['price'] - atr_val * 0.1 if a['direction'] == 'long' else a['price'] + atr_val * 0.1
                            _narrowed = True
                            log(f"  🔄 {sym} 미체결 → 오프셋 축소 ATR×0.2→0.1")

                r = place_limit_only(sym, side, usdt, entry, lev)
                if r['success']:
                    _pending_fills[sym] = {
                        'order_id': r['order_id'], 'sl': sl, 'tp': tp,
                        'side': side, 'qty': r['qty'], 'entry': entry,
                        'score': a.get('score', 0), 'atr': a.get('atr_1h', 0),
                        'placed_at': time.time(), 'narrowed': _narrowed,
                        'rl_signal': a.get('rl_signal', 'none'),
                    }
                    dist = abs(entry - a['price']) / a['price'] * 100
                    sl_pct = abs(entry - sl) / entry * 100
                    tp_pct = abs(tp - entry) / entry * 100
                    log(f"  📋 {sym} {'LONG' if a['direction']=='long' else 'SHORT'} @ ${entry} ({dist:.1f}%) SL({sl_pct:.1f}%) TP({tp_pct:.1f}%) R:R 1:{tp_pct/sl_pct:.1f}")
                else:
                    log(f"  ❌ {sym}: {r.get('error','')[:60]}")
            except Exception as e:
                log(f"  ❌ {sym} 주문 예외: {str(e)[:80]}")
            time.sleep(0.5)

    # #158: bounce OFF — 14건 29% 승률 -$1.86, 수익/리스크 불균형으로 비활성화
    # 데이터 더 쌓이고 조건 개선 후 재활성화 검토
    # if not top_n and slots > 0:
    #     check_oversold_bounce()

    # #4: 하락장 모드 판정
    update_market_mode()

    # ===== CVD만 운영 — 나머지 전략 OFF =====
    # if _bear_mode and slots > 0:
    #     check_funding_long()
    # manage_crash_buys()
    # check_pair_divergence()    # SOL 페어 OFF — 실전 10건 30% PF 0.10, 최근 5연패
    # check_liquidation_bounce()
    # scan_listing_announcements()
    # check_listing_short()

    # 활성 전략: CVD + BB 롱/숏 + 추세숏 + 역행숏
    check_cvd_divergence()       # CVD 다이버전스 (BTC RSI>45일 때만)
    check_bb_box()               # BB 롱 (BTC RSI>50일 때만)
    # check_bb_short()           # BB 숏 OFF — PF 0.06, 최근 1승5패, 방향 문제 (데이터 축적 후 재검토)
    check_trend_short()          # 추세 숏 (하락장 모멘텀)
    check_contrarian_short()     # 역행 숏 (하락장에서 과매수 알트)
    check_momentum_breakout()    # 모멘텀 롱 (중립장 추세 초기, BTC RSI 45~65)
    check_surge_short()          # 급등숏 (24h +30% + RSI 80+ 역행)

    # 나머지 기관 전략 OFF (데이터 축적 후 재검토)
    check_short_squeeze()        # #M 숏스퀴즈 롱 — 롱숏비율+OI 기반, 소액 $15
    # check_mtf_confluence()       # #N
    # check_vwap_reversion()       # #O: -$1.77
    # check_volume_profile()       # #P

    # 30분 스캔 보고 OFF — 알림 과다 방지 (진입/청산 알림만 유지)
    # global _last_tg_time
    # now = time.time()
    if False:  # 30분 보고 OFF
        _last_tg_time = now
        try:
            ts = datetime.now().strftime("%H:%M")
            lines = [f"<b>📊 자동 스캔</b> ({ts})", f"💰 ${bal['total']:.2f} | BTC {'↑' if btc_up else '↓'}", ""]

            for a in scored:
                if a['direction'] != 'wait':
                    e = '🟢' if a['direction'] == 'long' else '🔴'
                    d = '롱' if a['direction'] == 'long' else '숏'
                    lines.append(f"{e} <b>{a['symbol']}</b> ${a.get('price',0):.4f} {d}({a['score']:+d}) RSI {a.get('rsi',0):.0f}")

            pending_syms = list(_pending_fills.keys())
            if pending_syms:
                lines.append("")
                lines.append(f"📋 예약: {', '.join(pending_syms)}")

            if held:
                lines.append(f"✅ 보유: {', '.join(held)}")

            waits = [a for a in scored if a['direction'] == 'wait']
            if waits:
                names = ', '.join(a['symbol'].replace('USDT','') for a in waits[:6])
                lines.append(f"⏸ 관망: {names}")

            send_message(TG_TOKEN, TG_CHAT, "\n".join(lines))
        except Exception as e:
            log(f"  TG 전송 실패: {e}")

    # 관리 종목 파일 갱신 (app.py 충돌 방지)
    managed = list(held | set(_pending_fills.keys()) | top_syms)
    try:
        with open(LOCK_PATH, 'w') as f:
            json.dump({"symbols": managed, "ts": datetime.now().isoformat()}, f)
    except Exception:
        pass


def check_extreme_funding():
    """펀딩비 극단값 감지 → 역발상 시그널 큐 전달 (사이클당 1회)"""
    try:
        client = get_client()
        # 거래량 상위 10개만 빠르게 체크
        tickers = client.futures_ticker()
        tickers = [t for t in tickers if t['symbol'].endswith('USDT') and t['symbol'] not in BLACKLIST]
        tickers.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        top_syms = [t['symbol'] for t in tickers[:10]]

        held = get_held_symbols()
        from signal_queue import push_signal

        for sym in top_syms:
            if sym in held:
                continue
            try:
                fr = client.futures_funding_rate(symbol=sym, limit=1)
                if not fr:
                    continue
                rate = float(fr[-1]['fundingRate'])
                # 극단 펀딩비: ±0.1% 이상 → 역발상 시그널
                if rate >= 0.001:  # +0.1% 이상 → 롱 과열 → 숏 (현재 롱전용이므로 스킵)
                    log(f"  [펀딩] {sym} 펀딩={rate*100:+.3f}% (롱과열 → 스킵, 롱전용)")
                elif rate <= -0.001:  # -0.1% 이상 → 숏 과열 → 롱 역발상
                    push_signal(sym, 'funding', direction='long', priority=3,
                                meta={'funding_rate': round(rate * 100, 4)})
                    log(f"  [펀딩] {sym} 펀딩={rate*100:+.3f}% (숏과열 → 롱 역발상 시그널)")
            except Exception:
                pass
            time.sleep(0.1)
    except Exception as e:
        log(f"  펀딩비 감지 오류: {e}")


_sltp_verify_ts = 0  # 마지막 검증 시각

def verify_sltp():
    """포지션 보유 중인데 _sltp_done에 없는 것 감지 → SL/TP 배치 (2분마다)
    바이낸스 get_open_orders가 algo 주문을 안 보여주는 이슈 → 더미 배치 방식 금지
    대신 _sltp_done 세트로 추적. 재시작 시에는 check_fills 보완이 1회 처리."""
    global _sltp_verify_ts
    now = time.time()
    if now - _sltp_verify_ts < 120:
        return
    _sltp_verify_ts = now

    try:
        for pos in _get_positions_cached():
            sym = pos['symbol']
            entry = float(pos.get('entry_price', 0))
            qty = float(pos.get('size', 0))
            if entry <= 0 or qty <= 0:
                continue

            # _sltp_done에 있으면 이미 배치됨
            if sym in _sltp_done:
                continue
            if sym in _pending_fills:
                continue
            # 외부 모니터 포지션 보호 — _entry_source에 없으면 스킵
            if sym not in _entry_source:
                _sltp_done.add(sym)  # 다음 체크 방지
                log(f"  ℹ️ {sym} 외부 포지션 감지 → SL/TP 재배치 스킵")
                continue

            # 미추적 포지션 → ATR 기반 SL/TP 배치 (롱/숏 자동 판별)
            client = get_client()
            _ind = calc_indicators(get_klines(sym, '1h', 50))
            atr = _ind.get('atr', 0) or 0
            lev = 3 if sym in ('ETHUSDT', 'BTCUSDT') else 2
            atr_pct = atr / entry * 100 if entry else 1
            sl_pct = max(atr_pct * 1.5, 2.0 / lev)
            tp_pct = max(atr_pct * 3.0, sl_pct * 2.0)  # CVD TP 도달률 개선
            _is_long = 'LONG' in pos.get('side', 'LONG').upper()
            if _is_long:
                sl = _round_price_sym(sym, entry * (1 - sl_pct / 100))
                tp = _round_price_sym(sym, entry * (1 + tp_pct / 100))
                _close_side = 'SELL'
            else:
                sl = _round_price_sym(sym, entry * (1 + sl_pct / 100))
                tp = _round_price_sym(sym, entry * (1 - tp_pct / 100))
                _close_side = 'BUY'

            try:
                client.futures_create_order(
                    symbol=sym, side=_close_side, type='STOP_MARKET',
                    stopPrice=str(sl), quantity=str(qty), reduceOnly=True)
            except Exception as e:
                if '4130' not in str(e):
                    log(f"  verify {sym} SL: {str(e)[:40]}")
            try:
                client.futures_create_order(
                    symbol=sym, side=_close_side, type='TAKE_PROFIT_MARKET',
                    stopPrice=str(tp), quantity=str(qty), reduceOnly=True)
            except Exception as e:
                if '4130' not in str(e):
                    log(f"  verify {sym} TP: {str(e)[:40]}")

            _sltp_done.add(sym)
            _tp_cache[sym] = tp
            log(f"  🔧 {sym} SL/TP 미추적 → 배치 SL={sl} TP={tp}")
    except Exception as e:
        log(f"  SL/TP 검증 오류: {e}")



_last_daily_report = ""

def daily_summary():
    """#151 매일 23시 일일 성과 + 분석 리포트 텔레그램 발송"""
    global _last_daily_report
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.hour != 23 or _last_daily_report == today:
        return
    _last_daily_report = today
    try:
        bal = get_balance()
        all_trades = trade_db.get_all_trades(limit=200)
        today_trades = [t for t in all_trades if (t.get("time") or "").startswith(today)]
        today_pnl = sum(t.get("pnl") or 0 for t in today_trades if t.get("pnl") is not None)
        wins = len([t for t in today_trades if (t.get("pnl") or 0) > 0])
        losses = len([t for t in today_trades if (t.get("pnl") or 0) < 0])
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        # 전체 누적
        all_pnl = sum(t.get('pnl', 0) or 0 for t in all_trades if t.get('pnl'))
        all_closed = len([t for t in all_trades if t.get('pnl')])
        all_wins = len([t for t in all_trades if (t.get('pnl') or 0) > 0])
        all_wr = all_wins / all_closed * 100 if all_closed > 0 else 0

        # 보유 포지션
        pos = _get_positions_cached()
        pos_lines = []
        for p in pos:
            _ep = float(p.get('entry_price', 0))
            _cur = get_price(p['symbol'])
            _lev = 3 if p['symbol'] in ('ETHUSDT', 'BTCUSDT') else 2
            _pnl = (_cur - _ep) / _ep * 100 * _lev if _ep > 0 else 0
            pos_lines.append(f"  {p['symbol']} {_pnl:+.1f}%")

        # 최근 7일 추이
        week_pnl = {}
        for t in all_trades:
            d = (t.get('time') or '')[:10]
            if d >= (now - timedelta(days=6)).strftime('%Y-%m-%d'):
                week_pnl[d] = week_pnl.get(d, 0) + (t.get('pnl') or 0)

        # 오늘 최고/최저 거래
        best = max(today_trades, key=lambda t: t.get('pnl') or -999) if today_trades else None
        worst = min(today_trades, key=lambda t: t.get('pnl') or 999) if today_trades else None

        lines = [
            f"<b>🌙 일일 리포트</b> ({today})",
            f"",
            f"💰 잔고: ${bal['total']:.2f}",
            f"📊 오늘: {len(today_trades)}건 | 승{wins} 패{losses} | 승률{wr:.0f}%",
            f"💵 실현: ${today_pnl:+.2f}",
        ]
        if best and best.get('pnl'):
            lines.append(f"⬆️ 최고: {best['symbol']} ${best['pnl']:+.4f}")
        if worst and worst.get('pnl'):
            lines.append(f"⬇️ 최저: {worst['symbol']} ${worst['pnl']:+.4f}")

        if pos_lines:
            lines.append(f"\n📌 보유({len(pos_lines)}개):")
            lines.extend(pos_lines)

        lines.append(f"\n📈 누적: {all_closed}건 승률{all_wr:.0f}% PnL ${all_pnl:+.2f}")

        # 7일 추이
        if week_pnl:
            lines.append(f"\n📅 7일:")
            cum = 0
            for d in sorted(week_pnl):
                cum += week_pnl[d]
                lines.append(f"  {d[5:]}: ${week_pnl[d]:+.2f} (누적${cum:+.2f})")

        send_message(TG_TOKEN, TG_CHAT, "\n".join(lines))
        log(f"  🌙 일일 리포트 발송")
    except Exception as e:
        log(f"  일일 리포트 오류: {e}")


_funding_cycle = 0  # 펀딩비 체크 사이클 카운터

def main():
    global _funding_cycle, _consecutive_losses, _global_cooldown_until, _bear_stopped, _bear_daily_loss

    # #S: 이중 인스턴스 방지 — PID lock 파일
    import fcntl
    _lock_fd = open('/home/hyeok/01.APCC/00.ai-lab/.updater.lock', 'w')
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
    except IOError:
        print("이미 다른 인스턴스 실행 중 → 종료", flush=True)
        sys.exit(1)

    log("=" * 60)
    log(f"포지션 업데이터 시작 (5분 스캔, 최대 {MAX_ORDERS}슬롯, 롱≥{MIN_SCORE} 변동성SL TP×5 트레일1.0%)")
    log(f"  시그널 큐 활성 | 펀딩비 감지 30분 주기 | 스캔 {SCAN_ALT_COUNT}종목 | 일일 {MAX_DAILY_TRADES}건")
    log("=" * 60)

    # 재시작 시 _entry_source / _entry_time DB에서 복원
    try:
        _open = trade_db.get_open_trades()
        for _ot in _open:
            _sym = _ot.get('symbol', '')
            _src = _ot.get('source', 'updater')
            if _sym and _sym not in _entry_source:
                _entry_source[_sym] = _src
            if _sym and _sym not in _entry_time:
                try:
                    _t = datetime.strptime(_ot.get('time', ''), '%Y-%m-%d %H:%M:%S')
                    _entry_time[_sym] = _t.timestamp()
                except: _entry_time[_sym] = time.time() - 3600  # 폴백 1시간 전
        if _entry_source:
            log(f"  🔄 DB에서 포지션 source 복원: {_entry_source}")
        # 최근 청산 종목 쿨다운 복원 (재시작 시 반복 진입 방지)
        _recent = trade_db.get_closed_trades(limit=20)
        for _rt in _recent:
            _sym = _rt.get('symbol', '')
            _ct = _rt.get('close_time', '')
            if _sym and _ct:
                try:
                    _closed_at = datetime.strptime(_ct, '%Y-%m-%d %H:%M:%S').timestamp()
                    _elapsed = time.time() - _closed_at
                    if _elapsed < COOLDOWN_SEC:  # 아직 쿨다운 중
                        _cooldown[_sym] = _closed_at + COOLDOWN_SEC
                except: pass
        if _cooldown:
            log(f"  🔄 쿨다운 복원: {list(_cooldown.keys())}")
    except Exception as _e:
        log(f"  ⚠️ source 복원 실패: {_e}")

    while True:
        try:
            _positions_cache['ts'] = 0  # 사이클 시작 시 캐시 리셋
            check_fills()
            check_trailing_stop()
            check_partial_tp()
            check_stale_position()  # #F 2h+ 미수익 SL 조임
            check_long_hold()  # #148 장기보유 추세 재검증
            # 청산된 종목 → DB 업데이트 + 쿨다운 등록 + 연속 손실 추적
            _held = get_held_symbols()
            for sym in (_sltp_done - _held):
                _cooldown[sym] = time.time() + COOLDOWN_SEC
                # BB숏 동일 종목 4시간 재진입 쿨다운
                if _entry_source.get(sym) == 'bb_short':
                    _bb_short_cooldown[sym] = time.time()
                # DB 미청산 거래 자동 정리 (SL/TP 히트 감지)
                try:
                    _trades_api = get_client().futures_account_trades(symbol=sym, limit=10)
                    # 롱 청산=SELL(buyer=False), 숏 청산=BUY(buyer=True) — 양쪽 캡처
                    _src = _entry_source.get(sym, 'updater')
                    _is_short = _src in ('bb_short', 'trend_short', 'contrarian_short')
                    _close_trades = [t for t in _trades_api if t.get('buyer') == _is_short]  # 숏이면 buyer=True
                    if _close_trades:
                        _close_px = float(_close_trades[-1]['price'])
                        _realized = sum(float(t.get('realizedPnl', 0)) for t in _close_trades[-5:])
                        _open_rows = trade_db.get_open_trades(sym)
                        if _open_rows:
                            for _orow in _open_rows:
                                _entry_px = _orow.get('price', 0) or 0
                                _qty = _orow.get('qty', 0) or 0
                                # 폴백 PnL: 롱=(close-entry)*qty, 숏=(entry-close)*qty
                                if _realized:
                                    _calc_pnl = round(_realized / len(_open_rows), 4)
                                elif _is_short:
                                    _calc_pnl = round((_entry_px - _close_px) * _qty, 4)
                                else:
                                    _calc_pnl = round((_close_px - _entry_px) * _qty, 4)
                                trade_db.update_trade_field(_orow['id'], pnl=_calc_pnl, close_price=_close_px,
                                    close_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action='청산')
                                log(f"  📝 {sym} DB 청산 기록: PnL=${_calc_pnl:+.4f} close=${_close_px}")
                except Exception as _e:
                    log(f"  ⚠️ {sym} DB 청산 기록 실패: {_e}")
                # DB에서 마지막 거래 PnL 확인 → 연속 손실 추적
                try:
                    _db = sqlite3.connect('/home/hyeok/01.APCC/00.ai-lab/trades.db')
                    _row = _db.execute(
                        "SELECT pnl FROM trades WHERE symbol=? AND pnl IS NOT NULL ORDER BY id DESC LIMIT 1",
                        (sym,)).fetchone()
                    _db.close()
                    if _row and _row[0] is not None:
                        if _row[0] <= 0:
                            _consecutive_losses += 1
                            _cooldown[sym] = time.time() + COOLDOWN_LOSS_SEC
                            # SL 히트 알림
                            try:
                                send_message(TG_TOKEN, TG_CHAT,
                                    f"❌ {sym} 손절 ${_row[0]:+.2f} | 연속 {_consecutive_losses}패")
                            except: pass
                            if _bear_mode:
                                _bear_daily_loss['total'] += abs(_row[0])
                                if _bear_daily_loss['total'] >= 3.0:
                                    _bear_stopped = True
                                    log(f"  🛑 하락장 일일 손실 ${_bear_daily_loss['total']:.2f} ≥ $3 → 당일 정지")
                                    try:
                                        send_message(TG_TOKEN, TG_CHAT,
                                            f"🛑 하락장 손실 한도 ${_bear_daily_loss['total']:.2f} → 당일 정지")
                                    except: pass
                            log(f"  ⏳ {sym} 손절 → 연속 {_consecutive_losses}패 | {COOLDOWN_LOSS_SEC//60}분 쿨다운")
                        else:
                            _consecutive_losses = 0
                            _global_cooldown_until = 0  # 익절 → 연패 쿨다운도 해제
                            log(f"  ⏳ {sym} 익절 → 연패 리셋 | {COOLDOWN_SEC//60}분 쿨다운")
                    else:
                        log(f"  ⏳ {sym} 청산 → {COOLDOWN_SEC//60}분 쿨다운")
                except Exception as _pnl_err:
                    # DB 조회 실패해도 안전하게 손실로 간주 (연패 카운터 누락 방지)
                    _consecutive_losses += 1
                    _cooldown[sym] = time.time() + COOLDOWN_LOSS_SEC
                    log(f"  ⚠️ {sym} PnL 조회 실패({_pnl_err}) → 손실 간주, 연속 {_consecutive_losses}패")
            # #150: RL v2 학습 데이터 수집 — 청산 시 30분봉 스냅샷 저장
            for _closed_sym in (_sltp_done - _held):
                try:
                    _rl_data_path = '/home/hyeok/01.APCC/00.ai-lab/rl-lab/data/alt_trades_v2.jsonl'
                    _db2 = sqlite3.connect('/home/hyeok/01.APCC/00.ai-lab/trades.db')
                    _tr = _db2.execute(
                        "SELECT symbol, pnl, price, close_price, confidence, time, close_time FROM trades WHERE symbol=? AND pnl IS NOT NULL ORDER BY id DESC LIMIT 1",
                        (_closed_sym,)).fetchone()
                    _db2.close()
                    if _tr:
                        _record = {
                            'symbol': _tr[0], 'pnl': _tr[1], 'entry': _tr[2], 'exit': _tr[3],
                            'score': _tr[4], 'open_time': _tr[5], 'close_time': _tr[6],
                            'btc_up': _btc_cache.get('up', True), 'btc_rsi': round(_btc_cache.get('rsi', 50)),
                            'ts': datetime.now().isoformat(),
                        }
                        with open(_rl_data_path, 'a') as _f:
                            _f.write(json.dumps(_record) + '\n')
                except: pass

            _partial_done.difference_update(_partial_done - _held)
            _sltp_done.difference_update(_sltp_done - _held)
            # stale _tightened 정리 (재진입 시 stale 체크가 영구 스킵되는 버그 방지)
            if hasattr(check_stale_position, '_tightened'):
                check_stale_position._tightened.difference_update(check_stale_position._tightened - _held)
            for sym in list(_tp_cache):
                if sym not in _held:
                    del _tp_cache[sym]
            for sym in list(_trail_peak):
                if sym not in _held:
                    del _trail_peak[sym]
            for sym in list(_trail_atr):
                if sym not in _held:
                    del _trail_atr[sym]
            for sym in list(_entry_time):
                if sym not in _held:
                    del _entry_time[sym]
            for sym in list(_entry_source):
                if sym not in _held:
                    del _entry_source[sym]
            for sym in list(_bb_limit_orders):
                if sym not in _held and not has_position(sym):
                    _bb_limit_orders.pop(sym, None)
            for sym in list(_bb_short_limit_orders):
                if sym not in _held and not has_position(sym):
                    _bb_short_limit_orders.pop(sym, None)
            for sym in list(_sl_synced):
                if sym not in _held:
                    del _sl_synced[sym]
            update_cycle()
            # 펀딩비 극단 감지 (30분마다, 상위 10종목 ±0.1% 극단)
            _funding_cycle += 1
            if _funding_cycle % 6 == 0:
                check_extreme_funding()
            daily_summary()
        except Exception as e:
            log(f"사이클 오류: {e}")
        log(f"--- 다음: {INTERVAL}초 후 (트레일링 30초 체크) ---")
        # 5분 대기 중 30초마다 트레일링 스탑 빠른 체크
        _wait_start = time.time()
        while time.time() - _wait_start < INTERVAL:
            time.sleep(30)
            try:
                _positions_cache['ts'] = 0  # 캐시 리셋
                check_fills()  # LIMIT 체결 감지 → SL/TP 즉시 배치
                check_trailing_stop()
                verify_sltp()  # SL/TP 누락 자동 감지 + 재배치 (2분마다)
                _quick_surge_scan()  # 30초마다 24h +80%+ 메가급등 캐치
            except Exception:
                pass


if __name__ == "__main__":
    main()

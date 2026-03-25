#!/usr/bin/env python3
# 알트코인 스캐너 — 거래량/가격/RSI/펀딩비 기반 스크리닝 + 바이낸스 공지 감지

import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from binance_client import get_client, get_klines, get_funding_rate, get_open_interest, get_oi_change, get_long_short_ratio
from indicators import calc_indicators
from agents import run_agent

# ── 바이낸스 선물 심볼 캐시 (5분 TTL) ─────────────────────────────────
_FUTURES_SYMS_CACHE: set = set()
_FUTURES_SYMS_TIME: float = 0.0
_FUTURES_SYMS_TTL = 300  # 5분


def get_futures_symbols_cached() -> set:
    """바이낸스 선물 TRADING 심볼 세트 (5분 캐시)"""
    global _FUTURES_SYMS_CACHE, _FUTURES_SYMS_TIME
    if _FUTURES_SYMS_CACHE and time.time() - _FUTURES_SYMS_TIME < _FUTURES_SYMS_TTL:
        return _FUTURES_SYMS_CACHE
    try:
        client = get_client()
        info = client.futures_exchange_info()
        _FUTURES_SYMS_CACHE = {s["symbol"] for s in info["symbols"] if s["status"] == "TRADING"}
        _FUTURES_SYMS_TIME = time.time()
    except Exception:
        pass
    return _FUTURES_SYMS_CACHE


# 마지막으로 확인한 공지 ID 캐시 파일
_ANN_CACHE_PATH = Path(__file__).parent / "last_announcement.json"

# 바이낸스 공지 API 엔드포인트 (카탈로그 48 = 신규 상장/공지)
_BINANCE_ANN_URLS = [
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=48&pageNo=1&pageSize=10",
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=161&pageNo=1&pageSize=5",  # 선물 공지
]
_ANN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}

# 키워드 → 신호 매핑
_LONG_KEYWORDS  = ["will list", "new listing", "adds", "launchpad", "launchpool",
                   "resumes deposit", "resumes withdrawal", "opens deposit",
                   "new futures", "will launch", "perpetual contract"]
_SHORT_KEYWORDS = ["will delist", "removes", "delisting", "suspends", "terminates",
                   "hack", "exploit", "security incident"]
_URGENCY_HIGH   = ["will list", "new futures", "will launch", "will delist", "launchpad"]

# 제외 심볼 (BTC, ETH, 스테이블코인, 인덱스)
EXCLUDE_SYMBOLS = {
    "BTCUSDT", "ETHUSDT",
    "BUSDUSDT", "USDCUSDT", "USDTUSDT", "FDUSDUSDT", "DAIUSDT",
    "TUSDUSDT", "USDDUSDT", "USDPUSDT",
    "BTCDOMUSDT", "DEFIUSDT", "ALTUSDT",
}


def get_alt_futures_symbols(limit: int = 50) -> list:
    """바이낸스 USDT 선물 유니버스 — 거래량 + 변동률 + 최소유동성 필터"""
    try:
        client = get_client()
        tickers = client.futures_ticker()
        filtered = []
        movers = []  # 급등/급락 종목 별도 수집
        _MIN_QUOTE_VOL = 5_000_000  # 최소 24h 거래대금 $5M (슬리피지 방지)

        for t in tickers:
            sym = t["symbol"]
            if not sym.endswith("USDT"):
                continue
            if sym in EXCLUDE_SYMBOLS:
                continue
            quote_vol = float(t.get("quoteVolume", 0))
            if quote_vol <= 0:
                continue
            price_chg = abs(float(t.get("priceChangePercent", 0)))

            # 급등/급락 ±3% 이상 → 거래량 무관 추가 (최소 $1M은 필요)
            if price_chg >= 3.0 and quote_vol >= 1_000_000:
                movers.append(t)

            # 최소 유동성 필터
            if quote_vol < _MIN_QUOTE_VOL:
                continue
            filtered.append(t)

        # 거래량 기본 정렬 + 최근 변동률 가중 (vol_score = quoteVol × (1 + |변동률|/10))
        for t in filtered:
            _chg = abs(float(t.get("priceChangePercent", 0)))
            t["_rank_score"] = float(t.get("quoteVolume", 0)) * (1 + _chg / 10)

        filtered.sort(key=lambda x: x.get("_rank_score", 0), reverse=True)
        result_syms = [t["symbol"] for t in filtered[:limit]]

        # 급등/급락 종목 중 아직 포함되지 않은 것 추가 (최대 5개)
        _added = 0
        for m in sorted(movers, key=lambda x: abs(float(x.get("priceChangePercent", 0))), reverse=True):
            if m["symbol"] not in result_syms and _added < 5:
                result_syms.append(m["symbol"])
                _added += 1

        return result_syms
    except Exception as e:
        return []


def _score_symbol(sym: str) -> dict | None:
    """단일 심볼 스크리닝 — 점수 계산 후 결과 반환 (실패 시 None)"""
    try:
        klines_15m = get_klines(sym, "15m", 60)
        klines_1h  = get_klines(sym, "1h",  60)
        klines_4h  = get_klines(sym, "4h",  60)

        ind_15m = calc_indicators(klines_15m)
        ind_1h  = calc_indicators(klines_1h)
        ind_4h  = calc_indicators(klines_4h)

        price = ind_15m.get("price", 0)
        if price <= 0:
            return None

        # 유동성 필터: 24h 거래대금 $5M 미만 제외 (저유동성 종목 대형 손실 방지)
        try:
            _vol_24h = float(klines_1h["volume"].sum()) * price  # 1h × 60캔들 ≈ 60h 근사
            _daily_vol = _vol_24h * 24 / max(len(klines_1h), 1)  # 24h 추정
            if _daily_vol < 5_000_000:
                return None
        except Exception:
            pass

        # 신규 상장 필터: 4h 데이터 42캔들(7일) 미만이면 제외
        if len(klines_4h) < 42:
            return None

        score    = 0
        signals  = []
        direction = "wait"

        # ── 사전 계산 (이후 로직에서 참조) ──
        rsi = ind_15m.get("rsi", 50) or 50
        recent_vol = float(klines_15m["volume"].iloc[-1])
        avg_vol    = float(klines_15m["volume"].iloc[:-3].mean())
        vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1

        # 거래량 최소 필터: 1.5배 미만이면 모멘텀 부족 → 스킵 (LLM 비용 절감)
        if vol_ratio < 1.5:
            return None

        # ADX 추세 강도 확인 (방향 결정에 활용)
        adx = ind_15m.get("adx", 0) or 0
        dmp = ind_15m.get("adx_dmp", 0) or 0
        dmn = ind_15m.get("adx_dmn", 0) or 0
        _is_trending = adx >= 25

        # 1. 1시간 가격 변동률 — 추세추종 롱 강화
        price_change_1h = ind_1h.get("change_pct", 0) or 0
        abs_change = abs(price_change_1h)
        if abs_change >= 5:
            score += 40
            signals.append(f"급변동 {price_change_1h:+.1f}%")
            if _is_trending and price_change_1h > 0:
                direction = "long"
                score += 10  # 추세추종 롱 보너스
                signals.append("추세추종롱강화")
            elif _is_trending and price_change_1h < 0:
                direction = "wait"  # 강추세 급락은 기본 관망
                signals.append("강추세급락=관망")
                # 예외: RSI 극과매도(20이하) + 거래량 3배+ → 매도 소진 반등 롱
                _rsi_1h_tmp = ind_1h.get("rsi", 50) or 50
                if _rsi_1h_tmp <= 20 and vol_ratio >= 3:
                    direction = "long"
                    signals.append("급락+RSI극과매도+거래량=반등롱")
            else:
                direction = "wait"
                signals.append("약추세=관망")
        elif abs_change >= 3:
            score += 25
            signals.append(f"변동 {price_change_1h:+.1f}%")
            if _is_trending and price_change_1h > 0:
                direction = "long"
                signals.append("추세추종롱")
            else:
                direction = "wait"

        # 2. 거래량 급등 (최근 캔들 vs 직전 20캔들 평균) — 매수세 유입 시 롱 허용
        if vol_ratio >= 5:
            score += 35
            signals.append(f"거래량 {vol_ratio:.1f}배↑")
            # RSI<35면 패닉 매도 거래량 → 롱 금지 (FIL 손절 대응)
            if direction == "wait" and price_change_1h > 0 and rsi >= 35:
                direction = "long"
                signals.append("거래량급등+상승+RSI정상=롱")
            elif direction == "wait" and price_change_1h > 0 and rsi < 35:
                signals.append("거래량급등+과매도=패닉매도관망")
        elif vol_ratio >= 3:
            score += 20
            signals.append(f"거래량 {vol_ratio:.1f}배↑")
            if direction == "wait" and price_change_1h > 1.0 and rsi >= 35:
                direction = "long"
                signals.append("거래량증가+상승+RSI정상=롱")

        # 3. RSI 극단값 — 점수만 부여, 방향 결정 안 함 (롱 전용 모드)
        if rsi <= 20:
            score += 25
            signals.append(f"RSI 극과매도 {rsi}")
        elif rsi >= 80:
            score += 15  # 과매수는 점수 축소 (롱에 불리한 신호)
            signals.append(f"RSI 극과매수 {rsi}")
        elif rsi <= 30:
            score += 12
            signals.append(f"RSI 과매도 {rsi}")
        elif rsi >= 70:
            score += 8   # 과매수 점수 축소
            signals.append(f"RSI 과매수 {rsi}")

        # 4. 펀딩비 극단 — 숏스퀴즈 롱이 최고 수익 전략, 가중치 강화
        fr      = get_funding_rate(sym)
        funding = fr.get("rate", 0)
        if abs(funding) >= 0.1:
            score += 30  # 20→30 강화
            signals.append(f"펀딩비 극단 {funding:+.3f}%")
            if funding < 0 and direction == "wait":
                direction = "long"  # 숏 과열 → 숏스퀴즈 롱
                signals.append("숏스퀴즈롱")
            elif funding < 0 and direction == "long":
                score += 10  # 이미 롱인데 펀딩비도 음수 → 추가 보너스
                signals.append("롱+숏스퀴즈합류")
        elif abs(funding) >= 0.05:
            score += 10
            signals.append(f"펀딩비 {funding:+.3f}%")
            if funding < -0.05 and direction == "wait":
                direction = "long"

        # 5. 4H 대추세 컨플루언스 (최대 15점 / 감점 -10점)
        ema20_4h = ind_4h.get("ema20", 0) or 0
        ema50_4h = ind_4h.get("ema50", 0) or 0
        rsi_4h   = ind_4h.get("rsi", 50) or 50
        if ema20_4h and ema50_4h:
            _4h_up = ema20_4h > ema50_4h
            if direction == "long" and _4h_up:
                score += 15
                signals.append("4H상승추세")
            elif direction == "long" and not _4h_up:
                score -= 10
                signals.append("4H역행주의")

        # 6. ADX 추세 강도 보너스 (최대 10점)
        if adx >= 30:
            score += 10
            signals.append(f"ADX강추세{adx:.0f}")
        elif adx >= 25:
            score += 5

        # 7. 오더북 임밸런스 (최대 15점)
        ob_imbalance = 0.0
        try:
            client = get_client()
            _depth = client.futures_order_book(symbol=sym, limit=10)
            _bid_vol = sum(float(b[1]) for b in _depth.get("bids", []))
            _ask_vol = sum(float(a[1]) for a in _depth.get("asks", []))
            _total = _bid_vol + _ask_vol
            if _total > 0:
                ob_imbalance = (_bid_vol - _ask_vol) / _total  # +1=매수벽, -1=매도벽
                # 롱 전용: 매수벽은 롱 강화, 매도벽은 롱 차단/감점
                if ob_imbalance > 0.3 and direction == "long":
                    score += 15
                    signals.append(f"매수벽{ob_imbalance:.0%}")
                elif ob_imbalance > 0.3 and direction == "wait":
                    direction = "long"
                    score += 10
                    signals.append(f"매수벽{ob_imbalance:.0%}")
                elif ob_imbalance < -0.3 and direction == "long":
                    score -= 5  # 매도벽인데 롱
                elif ob_imbalance < -0.3:
                    score += 5  # 매도벽 감지만 (참고 점수)
                    signals.append(f"매도벽{ob_imbalance:.0%}")
                # 강한 매도벽(-50%+) → 롱 차단 (가격 상승 저항)
                if ob_imbalance < -0.5 and direction == "long":
                    direction = "wait"
                    score -= 10
                    signals.append(f"강매도벽{ob_imbalance:.0%}→롱차단")
        except Exception:
            pass

        # 8. OI — 항목 12에서 실제 히스토리 API로 대체 (프록시 제거)
        oi_change_pct = 0.0

        # 9. RSI 다이버전스 (최대 20점) — 가격↔RSI 방향 불일치 = 반전 신호
        _rsi_div = ind_15m.get("rsi_divergence")
        if _rsi_div == "bullish":
            score += 20
            signals.append("RSI bullish 다이버전스")
            if direction == "wait":
                direction = "long"
        elif _rsi_div == "bearish":
            score += 10  # 롱 전용: bearish div는 참고만 (점수 축소)
            signals.append("RSI bearish 다이버전스(참고)")

        # 10. 15m 직전 캔들 모멘텀 (최대 15점)
        _price_chg_15m = ind_15m.get("change_pct", 0) or 0
        _abs_15m = abs(_price_chg_15m)
        if _abs_15m >= 2.0:
            score += 15
            signals.append(f"15m모멘텀{_price_chg_15m:+.1f}%")
        elif _abs_15m >= 1.0:
            score += 8
            signals.append(f"15m모멘텀{_price_chg_15m:+.1f}%")

        # 11. 볼린저밴드 위치 (최대 15점)
        _bb_upper = ind_15m.get("bb_upper", 0) or 0
        _bb_lower = ind_15m.get("bb_lower", 0) or 0
        if _bb_lower > 0 and _bb_upper > _bb_lower:
            _bb_pos = (price - _bb_lower) / (_bb_upper - _bb_lower)  # 0~1
            if _bb_pos <= 0.05:  # 하단 터치 — 점수만, 단독 방향 설정 안 함
                score += 15
                signals.append(f"BB하단터치({_bb_pos:.0%})")
            elif _bb_pos >= 0.95:  # 상단 터치 — 점수만
                score += 15
                signals.append(f"BB상단터치({_bb_pos:.0%})")
            elif _bb_pos <= 0.15:
                score += 8
                signals.append(f"BB하단근접({_bb_pos:.0%})")
            elif _bb_pos >= 0.85:
                score += 8
                signals.append(f"BB상단근접({_bb_pos:.0%})")

        # 12. OI 히스토리 실제 변화율 (최대 20점) — 프록시 대체
        try:
            _oi_hist = get_oi_change(sym, period="15m", limit=5)
            if _oi_hist.get("available"):
                _oi_chg = _oi_hist["change_pct"]
                oi_change_pct = _oi_chg
                if abs(_oi_chg) >= 5:
                    score += 20
                    signals.append(f"OI{_oi_chg:+.1f}%급변")
                elif abs(_oi_chg) >= 2:
                    score += 10
                    signals.append(f"OI{_oi_chg:+.1f}%")
        except Exception:
            pass

        # 13. 1H RSI/MACD 컨플루언스 (최대 15점)
        _rsi_1h = ind_1h.get("rsi", 50) or 50
        _macd_hist_1h = ind_1h.get("macd_hist", 0) or 0
        if direction == "long":
            if _rsi_1h < 40 and _macd_hist_1h > 0:
                score += 15
                signals.append("1H_RSI과매도+MACD양전환")
            elif _rsi_1h < 40:
                score += 5
                signals.append(f"1H_RSI{_rsi_1h:.0f}")
        # (롱 전용: short 분기 제거)

        # 14. OI+가격 방향 조합 (최대 15점 / -10점) — 롱 방향만
        if oi_change_pct != 0 and direction == "long":
            if oi_change_pct > 2 and price_change_1h > 0:
                score += 15  # OI↑ + 가격↑ = 신규 롱 유입 → 롱 강화
                signals.append("OI↑가격↑롱강화")
            elif oi_change_pct > 2 and price_change_1h < 0:
                score -= 10  # OI↑ + 가격↓인데 롱 → 위험
                signals.append("OI↑가격↓롱위험")
            elif oi_change_pct < -2:
                score -= 5  # OI↓ = 포지션 청산 중 → 추세 약화
                signals.append("OI↓청산중")

        # 15. 탑 트레이더 롱/숏 비율 (최대 12점) — 롱 방향만
        try:
            _ls = get_long_short_ratio(sym, period="15m", limit=1)
            if _ls.get("available"):
                _long_pct = _ls["long_pct"]
                if direction == "long" and _long_pct < 0.35:
                    score += 12  # 숏 과밀 → 롱 역행 기회
                    signals.append(f"롱비율{_long_pct:.0%}역행롱")
                elif direction == "long" and _long_pct > 0.65:
                    score -= 5  # 이미 롱 과밀인데 롱
                    signals.append(f"롱과밀{_long_pct:.0%}")
        except Exception:
            pass

        # 16. 방향 일관성 점수 (최대 10점 / -15점) — 각 요소가 같은 방향을 가리키는지
        _dir_votes = {"long": 0, "short": 0}
        # 가격변동 방향
        if price_change_1h > 1: _dir_votes["long"] += 1
        elif price_change_1h < -1: _dir_votes["short"] += 1
        # RSI 방향
        if rsi < 35: _dir_votes["long"] += 1
        elif rsi > 65: _dir_votes["short"] += 1
        # MACD 1H
        if _macd_hist_1h > 0: _dir_votes["long"] += 1
        elif _macd_hist_1h < 0: _dir_votes["short"] += 1
        # 4H EMA
        if ema20_4h and ema50_4h:
            if ema20_4h > ema50_4h: _dir_votes["long"] += 1
            else: _dir_votes["short"] += 1
        _agree = _dir_votes.get(direction, 0)
        _oppose = _dir_votes.get("short" if direction == "long" else "long", 0)
        if _agree >= 3 and _oppose == 0:
            score += 10
            signals.append(f"방향합의{_agree}/4")
        elif _oppose >= 3:
            score -= 15
            signals.append(f"방향불일치{_oppose}/4역행")

        # 17. CVD 기반 스코어링 (최대 12점 / -8점)
        _cvd = ind_15m.get("cvd_trend")
        _cvd_pct = ind_15m.get("cvd_delta_pct", 0) or 0
        if direction == "long" and _cvd == "up" and _cvd_pct > 10:
            score += 12
            signals.append(f"CVD매수압력↑{_cvd_pct:.0f}%")
        elif direction == "long" and _cvd == "down" and _cvd_pct < -10:
            score -= 8
            signals.append(f"CVD매도압력↑롱위험")

        # 18. VWAP 위치 필터 (최대 10점 / -8점)
        _vwap = ind_15m.get("vwap")
        if _vwap and price > 0:
            if direction == "long" and price > _vwap:
                score += 10
                signals.append("VWAP위(매수우위)")
            elif direction == "long" and price < _vwap * 0.98:
                score -= 8
                signals.append("VWAP아래(매수약세)")

        # 19. EMA 크로스 (최대 15점 / -10점)
        _ema_cross = ind_15m.get("ema_cross")
        if _ema_cross == "golden":
            score += 15
            signals.append("EMA골든크로스")
            if direction == "wait":
                direction = "long"
        elif _ema_cross == "death" and direction == "long":
            score -= 10
            signals.append("EMA데드크로스→롱감점")

        if score < 20:
            return None

        # 숏 허용: 스크리너 방향 그대로 LLM에 전달

        atr     = ind_15m.get("atr", 0) or 0
        atr_pct = round(atr / price * 100, 3) if price > 0 else 0

        # 저변동 종목 제외: ATR% 0.5% 미만 (SL 노이즈 탈락 방지)
        if atr_pct < 0.5:
            return None

        return {
            "symbol":          sym,
            "score":           score,
            "signals":         signals,
            "direction":       direction,
            "price_change_1h": round(price_change_1h, 2),
            "vol_ratio":       round(vol_ratio, 2),
            "rsi":             rsi,
            "funding":         funding,
            "price":           price,
            "atr":             atr,
            "atr_pct":         atr_pct,
            "adx":             round(adx, 1),
            "ob_imbalance":    round(ob_imbalance, 3),
            "indicators":      ind_15m,
            "indicators_1h":   ind_1h,
            "indicators_4h":   ind_4h,
        }
    except Exception:
        return None


def screen_altcoins(symbols: list, top_n: int = 3, max_workers: int = 10) -> list:
    """알트코인 병렬 스크리닝 — 점수 상위 top_n 반환"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_score_symbol, sym): sym for sym in symbols}
        for fut in futures:
            r = fut.result()
            if r is not None:
                results.append(r)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def _load_seen_ids() -> set:
    """마지막으로 확인한 공지 ID 목록 로드"""
    try:
        if _ANN_CACHE_PATH.exists():
            return set(json.loads(_ANN_CACHE_PATH.read_text()).get("ids", []))
    except Exception:
        pass
    return set()


def _save_seen_ids(ids: set):
    """확인한 공지 ID 저장"""
    try:
        _ANN_CACHE_PATH.write_text(json.dumps({"ids": list(ids)[-200:]}))
    except Exception:
        pass


def _extract_symbol(title: str) -> str | None:
    """공지 제목에서 코인 심볼 추출"""
    # 패턴 1: 괄호 안 심볼 (예: "Will List ABC (ABCUSDT)")
    m = re.search(r'\(([A-Z]{2,10}USDT)\)', title)
    if m:
        return m.group(1)
    # 패턴 2: 괄호 안 심볼 (비USDT, USDT 붙여서 반환)
    m = re.search(r'\(([A-Z]{2,10})\)', title)
    if m:
        sym = m.group(1)
        if sym not in {"USD", "USDT", "BTC", "ETH", "BNB"}:
            return sym + "USDT"
    # 패턴 3: "USDⓈ-M XYZ Perpetual" → XYZUSDT
    m = re.search(r'USDS?-M\s+([A-Z]{2,10})\s+Perpetual', title)
    if m:
        return m.group(1) + "USDT"
    return None


def _classify_announcement(title: str) -> tuple[str, str, str]:
    """
    공지 제목 분류 → (type, signal, urgency)
    """
    t = title.lower()
    if any(k in t for k in ["will list", "new listing", "adds"]):
        ann_type = "spot_listing"
        signal   = "long"
    elif any(k in t for k in ["new futures", "will launch", "perpetual contract", "usds-m"]):
        ann_type = "futures_listing"
        signal   = "long"
    elif any(k in t for k in ["launchpad", "launchpool"]):
        ann_type = "launchpad"
        signal   = "long"
    elif any(k in t for k in ["resumes deposit", "resumes withdrawal", "opens deposit", "reopens"]):
        ann_type = "deposit_resume"
        signal   = "long"
    elif any(k in t for k in ["will delist", "removes", "delisting", "terminates trading"]):
        ann_type = "delisting"
        signal   = "short"
    elif any(k in t for k in ["suspends", "hack", "exploit", "security incident"]):
        ann_type = "security"
        signal   = "short"
    else:
        ann_type = "other"
        signal   = "wait"

    urgency = "high" if any(k in t for k in _URGENCY_HIGH) else "medium"
    return ann_type, signal, urgency


def check_binance_announcements() -> dict:
    """
    바이낸스 공지 API 직접 폴링 (Claude 웹검색 불필요)
    새 공지 ID 감지 시 반환, 없으면 found=False
    반환: {"found": bool, "announcements": [...], "summary": str, "new_count": int}
    """
    seen_ids   = _load_seen_ids()
    new_items  = []
    all_new_ids = set()

    for url in _BINANCE_ANN_URLS:
        try:
            resp = requests.get(url, headers=_ANN_HEADERS, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("data", {}).get("articles", [])
            if not articles:
                _cats = data.get("data", {}).get("catalogs", [])
                articles = _cats[0].get("articles", []) if _cats else []
            for art in articles:
                art_id    = str(art.get("id", ""))
                art_title = art.get("title", "")
                if not art_id or not art_title:
                    continue
                all_new_ids.add(art_id)
                if art_id in seen_ids:
                    continue  # 이미 본 공지

                # 새 공지 분류
                ann_type, signal, urgency = _classify_announcement(art_title)
                if signal == "wait":
                    continue  # 트레이딩 무관 공지 무시

                symbol = _extract_symbol(art_title)
                if not symbol:
                    continue

                new_items.append({
                    "symbol":  symbol,
                    "type":    ann_type,
                    "signal":  signal,
                    "title":   art_title,
                    "reason":  f"바이낸스 공지: {ann_type}",
                    "urgency": urgency,
                    "id":      art_id,
                })
        except Exception:
            continue

    # 새로 본 ID 저장
    _save_seen_ids(seen_ids | all_new_ids)

    if new_items:
        syms = [a["symbol"] for a in new_items]
        return {
            "found":         True,
            "announcements": new_items,
            "summary":       f"새 공지 {len(new_items)}건: {', '.join(syms)}",
            "new_count":     len(new_items),
        }

    return {
        "found":         False,
        "announcements": [],
        "summary":       f"새 공지 없음 (총 {len(all_new_ids)}건 확인)",
        "new_count":     0,
    }


def run_alt_analysis(candidate: dict) -> dict:
    """
    스크리너 선택 종목 전체 분석
    alt_analyst + alt_news 병렬 → alt_trader 순차
    반환: 분석 결과 dict (trader_json 포함)
    """
    sym        = candidate["symbol"]
    price      = candidate["price"]
    ind_15m    = candidate.get("indicators", {})
    signals    = candidate.get("signals", [])
    direction  = candidate.get("direction", "wait")

    screener_summary = (
        f"스크리너 점수: {candidate['score']}점\n"
        f"감지 신호: {', '.join(signals)}\n"
        f"예상 방향: {direction}\n"
        f"1h 변동: {candidate['price_change_1h']:+.2f}%\n"
        f"거래량 배수: {candidate['vol_ratio']}배\n"
        f"RSI: {candidate['rsi']}\n"
        f"펀딩비: {candidate['funding']:+.3f}%\n"
        f"ATR: {candidate['atr']} ({candidate['atr_pct']}%)"
    )

    ind_1h  = candidate.get("indicators_1h", {})
    ind_4h  = candidate.get("indicators_4h", {})

    analyst_prompt = f"""{sym} 알트코인 분석을 해주세요.

현재가: ${price:,}
{screener_summary}

기술적 지표 (15분봉):
RSI: {ind_15m.get('rsi')} / MACD Hist: {ind_15m.get('macd_hist')} / EMA20: {ind_15m.get('ema20')} / EMA50: {ind_15m.get('ema50')}
ATR: {ind_15m.get('atr')} / ADX: {ind_15m.get('adx')} / 볼린저밴드 상단: {ind_15m.get('bb_upper')} / 하단: {ind_15m.get('bb_lower')}

기술적 지표 (1시간봉):
RSI: {ind_1h.get('rsi')} / MACD Hist: {ind_1h.get('macd_hist')} / EMA20: {ind_1h.get('ema20')} / EMA50: {ind_1h.get('ema50')}

기술적 지표 (4시간봉):
RSI: {ind_4h.get('rsi')} / EMA20: {ind_4h.get('ema20')} / EMA50: {ind_4h.get('ema50')} / 추세: {'상승' if (ind_4h.get('ema20') or 0) > (ind_4h.get('ema50') or 0) else '하락'}"""

    news_prompt = f"""{sym} 알트코인의 최신 뉴스, 트위터/X 트렌드, 커뮤니티 분위기를 웹에서 검색해서 분석해주세요.
현재가: ${price:,} (1시간 변동: {candidate['price_change_1h']:+.2f}%)
이 코인 관련 최근 24시간 이내 뉴스, 공시, 개발팀 발표 등을 찾아주세요."""

    # 뉴스 에이전트 완전 스킵 (비용 대비 효과 불분명, 기술적 분석으로 충분)
    analyst_out = run_agent("alt_analyst", analyst_prompt)
    news_out = "뉴스 스킵 (기술적 분석 전용 모드)"

    # alt_trader 최종 결정 — 롱/숏 양방향 + 최적 진입가
    _atr_val = candidate['atr'] or 0
    _bb_lower = ind_15m.get("bb_lower", 0) or 0
    _bb_upper = ind_15m.get("bb_upper", 0) or 0
    _ema20_15m = ind_15m.get("ema20", 0) or 0
    _ema50_15m = ind_15m.get("ema50", 0) or 0
    _ema20_1h = ind_1h.get("ema20", 0) or 0
    _vwap = ind_15m.get("vwap", 0) or 0

    trader_prompt = f"""알트코인 {sym} 매매 결정 + 최적 진입가를 제시해주세요.

현재가: ${price:,}

[스크리너 신호 — 예상 방향: {direction.upper()}]
{screener_summary}

[기술적 분석]
{analyst_out}

[뉴스/센티멘트]
{news_out}

[지지/저항 참고]
- BB 하단: ${_bb_lower:.4f} / 상단: ${_bb_upper:.4f}
- 15m EMA20: ${_ema20_15m:.4f} / EMA50: ${_ema50_15m:.4f}
- 1H EMA20: ${_ema20_1h:.4f}
- ATR(15m): ${_atr_val:.4f} ({candidate['atr_pct']}%)

ATR 기반 SL/TP 참고:
- 롱 SL: ${round(price - _atr_val, 4)} / TP: ${round(price + _atr_val * 2, 4)}
- 숏 SL: ${round(price + _atr_val, 4)} / TP: ${round(price - _atr_val * 2, 4)}

**중요: 시장가가 아닌 지정가(LIMIT) 주문입니다.**
최적 진입가(entry)를 반드시 제시하세요:
- 롱: 현재가보다 약간 아래 (지지선, EMA, BB하단 근처에서 풀백 매수)
- 숏: 현재가보다 약간 위 (저항선, EMA, BB상단 근처에서 반등 매도)
- 현재가 대비 0.3%~1.5% 범위 내에서 설정 (너무 멀면 미체결, 너무 가까우면 시장가와 다를 바 없음)

JSON 응답: {{"signal":"long/short/wait", "entry":최적진입가, "sl":손절가, "tp":익절가, "confidence":0-100, "reason":"근거"}}"""

    trader_raw  = run_agent("alt_trader", trader_prompt)
    trader_json = _parse_alt_trader_json(trader_raw)

    return {
        "symbol":      sym,
        "candidate":   candidate,
        "analyst":     analyst_out,
        "news":        news_out,
        "trader":      trader_raw,
        "trader_json": trader_json,
        "time":        __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _parse_alt_trader_json(raw: str) -> dict:
    """alt_trader JSON 파싱. 실패 시 wait 반환."""
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data   = json.loads(match.group())
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
    return {"signal": "wait", "entry": None, "sl": None, "tp": None,
            "confidence": 0, "reason": raw[:200], "condition": "", "raw": raw}


# ── 업비트 상장 공지 감지 ────────────────────────────────────────────
_UPBIT_ANN_URL = "https://api-manager.upbit.com/api/v1/announcements?os=web&page=1&per_page=10&category=trade"
_UPBIT_CACHE_PATH = Path(__file__).parent / "last_upbit_announcement.json"

# 업비트 심볼 → 바이낸스 심볼 매핑 (괄호 안 티커 추출)
_UPBIT_SYM_PATTERN = re.compile(r'[（(]([A-Z]{2,10})[)）]')

# 업비트 공지 분류 키워드
_UPBIT_LONG_KW  = ["신규 거래지원", "디지털 자산 추가", "마켓 추가", "거래 재개"]
_UPBIT_SHORT_KW = ["거래지원 종료", "거래 유의 종목 지정", "상장폐지"]
_UPBIT_WARN_KW  = ["유의 촉구"]


def _load_upbit_seen() -> set:
    try:
        if _UPBIT_CACHE_PATH.exists():
            return set(json.loads(_UPBIT_CACHE_PATH.read_text()).get("ids", []))
    except Exception:
        pass
    return set()


def _save_upbit_seen(ids: set):
    try:
        _UPBIT_CACHE_PATH.write_text(json.dumps({"ids": list(ids)[-200:]}))
    except Exception:
        pass


def _extract_upbit_symbol(title: str) -> str | None:
    """업비트 공지 제목에서 코인 티커 추출 → 바이낸스 USDT 심볼 반환"""
    m = _UPBIT_SYM_PATTERN.search(title)
    if m:
        ticker = m.group(1).upper()
        if ticker not in {"KRW", "BTC", "USDT", "USD", "ETH", "BNB"}:
            return ticker + "USDT"
    return None


def _classify_upbit_announcement(title: str) -> tuple[str, str]:
    """업비트 공지 분류 → (signal, ann_type)"""
    for kw in _UPBIT_LONG_KW:
        if kw in title:
            return "long", "upbit_listing"
    for kw in _UPBIT_SHORT_KW:
        if kw in title:
            return "short", "upbit_delisting"
    for kw in _UPBIT_WARN_KW:
        if kw in title:
            return "short", "upbit_warning"
    return "wait", "upbit_other"


def check_upbit_announcements() -> dict:
    """
    업비트 공지 폴링 — 신규 상장/거래지원 감지
    반환: {"found": bool, "announcements": [...], "summary": str, "new_count": int}
    """
    seen_ids = _load_upbit_seen()
    new_items = []
    all_ids = set()

    try:
        resp = requests.get(_UPBIT_ANN_URL, headers=_ANN_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        notices = data.get("data", {}).get("notices", [])

        for n in notices:
            nid = str(n.get("id", ""))
            title = n.get("title", "")
            if not nid or not title:
                continue
            all_ids.add(nid)
            if nid in seen_ids:
                continue

            signal, ann_type = _classify_upbit_announcement(title)
            if signal == "wait":
                continue

            symbol = _extract_upbit_symbol(title)
            if not symbol:
                continue

            # 바이낸스에 해당 선물 존재하는지 확인
            _binance_syms = get_futures_symbols_cached()
            if _binance_syms and symbol not in _binance_syms:
                continue

            new_items.append({
                "symbol":  symbol,
                "type":    ann_type,
                "signal":  signal,
                "title":   title,
                "reason":  f"업비트 공지: {title[:50]}",
                "urgency": "high" if "신규 거래지원" in title or "거래지원 종료" in title else "medium",
                "source":  "upbit",
                "id":      nid,
            })

    except Exception:
        pass

    _save_upbit_seen(seen_ids | all_ids)

    if new_items:
        syms = [a["symbol"] for a in new_items]
        return {
            "found":         True,
            "announcements": new_items,
            "summary":       f"업비트 새 공지 {len(new_items)}건: {', '.join(syms)}",
            "new_count":     len(new_items),
        }

    return {
        "found":         False,
        "announcements": [],
        "summary":       "업비트 새 공지 없음",
        "new_count":     0,
    }


# ── OKX 상장/상폐 공지 감지 ────────────────────────────────────────────
_OKX_ANN_URL = "https://www.okx.com/api/v5/support/announcements?page=1"
_OKX_CACHE_PATH = Path(__file__).parent / "last_okx_announcement.json"

# OKX annType 매핑
_OKX_LISTING_TYPES = {"announcements-new-listings"}
_OKX_DELIST_TYPES  = {"announcements-delistings"}

# OKX 제목에서 심볼 추출 패턴
_OKX_SYM_PATTERNS = [
    re.compile(r'list\s+(?:perpetual\s+futures\s+for\s+)?(\w+)\s*\(', re.IGNORECASE),    # "list KAT (Katana)"
    re.compile(r'launch\s+(\w+)/USD', re.IGNORECASE),                                     # "launch KAT/USD"
    re.compile(r'delist\s+(.+?)\s+spot', re.IGNORECASE),                                  # "delist RSS3, MEMEFI..."
    re.compile(r'futures\s+for\s+(\w+)\s+crypto', re.IGNORECASE),                         # "futures for OPN crypto"
    re.compile(r'perpetual\s+futures\s+for\s+(\w+)$', re.IGNORECASE),                     # "perpetual futures for NG"
]

# 제외 (주식, TradFi 등)
_OKX_EXCLUDE = {"SELECTED", "EQUITIES", "SEVERAL", "NG", "XAU", "XAG"}


def _load_okx_seen() -> set:
    try:
        if _OKX_CACHE_PATH.exists():
            return set(json.loads(_OKX_CACHE_PATH.read_text()).get("ids", []))
    except Exception:
        pass
    return set()


def _save_okx_seen(ids: set):
    try:
        _OKX_CACHE_PATH.write_text(json.dumps({"ids": list(ids)[-200:]}))
    except Exception:
        pass


def _extract_okx_symbols(title: str) -> list[str]:
    """OKX 공지 제목에서 코인 심볼 추출 (여러 개 가능)"""
    for pat in _OKX_SYM_PATTERNS:
        m = pat.search(title)
        if m:
            raw = m.group(1).strip()
            # 쉼표 구분된 여러 심볼 (delist 공지)
            if "," in raw:
                syms = [s.strip().upper() for s in raw.split(",")]
            else:
                syms = [raw.upper()]
            return [s + "USDT" for s in syms
                    if 2 <= len(s) <= 10 and s not in _OKX_EXCLUDE and s.isalpha()]
    return []


def check_okx_announcements() -> dict:
    """
    OKX 공지 API 폴링 — 신규 상장/상폐 감지
    반환: {"found": bool, "announcements": [...], "summary": str, "new_count": int}
    """
    seen_ids = _load_okx_seen()
    new_items = []
    all_ids = set()

    try:
        resp = requests.get(_OKX_ANN_URL, headers=_ANN_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            return {"found": False, "announcements": [], "summary": "OKX API 에러", "new_count": 0}

        for group in data.get("data", []):
            for item in group.get("details", []):
                ann_type = item.get("annType", "")
                title = item.get("title", "")
                p_time = item.get("pTime", "")
                # ID = pTime (고유)
                nid = p_time
                if not nid or not title:
                    continue
                all_ids.add(nid)
                if nid in seen_ids:
                    continue

                # 상장/상폐만 처리
                if ann_type in _OKX_LISTING_TYPES:
                    signal = "long"
                    at = "okx_listing"
                elif ann_type in _OKX_DELIST_TYPES:
                    signal = "short"
                    at = "okx_delisting"
                else:
                    continue

                symbols = _extract_okx_symbols(title)
                if not symbols:
                    continue

                # 바이낸스 선물에 있는 심볼만
                _binance_syms = get_futures_symbols_cached()

                for sym in symbols:
                    if _binance_syms and sym not in _binance_syms:
                        continue
                    new_items.append({
                        "symbol":  sym,
                        "type":    at,
                        "signal":  signal,
                        "title":   title,
                        "reason":  f"OKX 공지: {title[:50]}",
                        "urgency": "high" if signal == "long" else "medium",
                        "source":  "okx",
                        "id":      nid,
                    })

    except Exception:
        pass

    _save_okx_seen(seen_ids | all_ids)

    if new_items:
        syms = list(set(a["symbol"] for a in new_items))
        return {
            "found":         True,
            "announcements": new_items,
            "summary":       f"OKX 새 공지 {len(new_items)}건: {', '.join(syms)}",
            "new_count":     len(new_items),
        }

    return {
        "found":         False,
        "announcements": [],
        "summary":       "OKX 새 공지 없음",
        "new_count":     0,
    }


# ── 코인베이스 신규 상장 감지 ───────────────────────────────────────────
_CB_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
_CB_CACHE_PATH = Path(__file__).parent / "last_coinbase_products.json"


def _load_cb_known() -> set:
    """이전에 알려진 코인베이스 상품 ID 세트"""
    try:
        if _CB_CACHE_PATH.exists():
            return set(json.loads(_CB_CACHE_PATH.read_text()).get("products", []))
    except Exception:
        pass
    return set()


def _save_cb_known(products: set):
    try:
        _CB_CACHE_PATH.write_text(json.dumps({"products": list(products)}))
    except Exception:
        pass


def check_coinbase_listings() -> dict:
    """
    코인베이스 신규 상장 감지 — 상품 목록 캐시 비교 방식
    첫 실행: 캐시 저장만 (새 공지 0건)
    이후 실행: 새로 추가된 USD 마켓 감지
    반환: {"found": bool, "announcements": [...], "summary": str, "new_count": int}
    """
    known = _load_cb_known()
    new_items = []

    try:
        resp = requests.get(_CB_PRODUCTS_URL, headers=_ANN_HEADERS, timeout=10)
        resp.raise_for_status()
        products = resp.json()

        # USD 마켓 + online 상태만
        usd_products = {
            p["id"]: p["base_currency"]
            for p in products
            if p.get("quote_currency") == "USD"
            and p.get("status") == "online"
            and not p.get("trading_disabled", False)
        }

        current_ids = set(usd_products.keys())

        if not known:
            # 첫 실행: 캐시 초기화
            _save_cb_known(current_ids)
            return {
                "found":         False,
                "announcements": [],
                "summary":       f"코인베이스 캐시 초기화 ({len(current_ids)}개 상품)",
                "new_count":     0,
            }

        # 새로 추가된 상품 탐지
        new_products = current_ids - known
        # 상폐 (사라진 상품)
        delisted = known - current_ids

        # 바이낸스 선물 심볼 확인
        try:
            _binance_syms = get_futures_symbols_cached()
        except Exception:
            _binance_syms = set()

        for pid in new_products:
            base = usd_products.get(pid, pid.split("-")[0])
            sym = base + "USDT"
            if sym in EXCLUDE_SYMBOLS:
                continue
            if _binance_syms and sym not in _binance_syms:
                continue
            new_items.append({
                "symbol":  sym,
                "type":    "coinbase_listing",
                "signal":  "long",
                "title":   f"Coinbase 신규 상장: {base} ({pid})",
                "reason":  f"코인베이스 신규 상장: {base}",
                "urgency": "high",
                "source":  "coinbase",
                "id":      pid,
            })

        for pid in delisted:
            base = pid.split("-")[0]
            sym = base + "USDT"
            if sym in EXCLUDE_SYMBOLS:
                continue
            if _binance_syms and sym not in _binance_syms:
                continue
            new_items.append({
                "symbol":  sym,
                "type":    "coinbase_delisting",
                "signal":  "short",
                "title":   f"Coinbase 상폐: {base} ({pid})",
                "reason":  f"코인베이스 상폐: {base}",
                "urgency": "medium",
                "source":  "coinbase",
                "id":      pid,
            })

        # 캐시 업데이트
        _save_cb_known(current_ids)

    except Exception:
        pass

    if new_items:
        syms = list(set(a["symbol"] for a in new_items))
        return {
            "found":         True,
            "announcements": new_items,
            "summary":       f"코인베이스 변동 {len(new_items)}건: {', '.join(syms)}",
            "new_count":     len(new_items),
        }

    return {
        "found":         False,
        "announcements": [],
        "summary":       "코인베이스 변동 없음",
        "new_count":     0,
    }

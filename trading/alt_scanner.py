#!/usr/bin/env python3
# 알트코인 스캐너 — 거래량/가격/RSI/펀딩비 기반 스크리닝 + 바이낸스 공지 감지

import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from binance_client import get_client, get_klines, get_funding_rate
from indicators import calc_indicators
from agents import run_agent

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
    """바이낸스 USDT 선물 거래량 상위 N개 (BTC/ETH/스테이블 제외)"""
    try:
        client = get_client()
        tickers = client.futures_ticker()
        filtered = []
        for t in tickers:
            sym = t["symbol"]
            if not sym.endswith("USDT"):
                continue
            if sym in EXCLUDE_SYMBOLS:
                continue
            # 거래량 0 제외
            if float(t.get("quoteVolume", 0)) <= 0:
                continue
            filtered.append(t)
        filtered.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
        return [t["symbol"] for t in filtered[:limit]]
    except Exception as e:
        return []


def _score_symbol(sym: str) -> dict | None:
    """단일 심볼 스크리닝 — 점수 계산 후 결과 반환 (실패 시 None)"""
    try:
        klines_15m = get_klines(sym, "15m", 30)
        klines_1h  = get_klines(sym, "1h",  24)

        ind_15m = calc_indicators(klines_15m)
        ind_1h  = calc_indicators(klines_1h)

        price = ind_15m.get("price", 0)
        if price <= 0:
            return None

        score    = 0
        signals  = []
        direction = "wait"

        # 1. 1시간 가격 변동률
        price_change_1h = ind_1h.get("change_pct", 0) or 0
        abs_change = abs(price_change_1h)
        if abs_change >= 5:
            score += 40
            signals.append(f"급변동 {price_change_1h:+.1f}%")
            direction = "short" if price_change_1h > 0 else "long"
        elif abs_change >= 3:
            score += 25
            signals.append(f"변동 {price_change_1h:+.1f}%")
            direction = "short" if price_change_1h > 0 else "long"

        # 2. 거래량 급등 (최근 캔들 vs 직전 20캔들 평균)
        recent_vol = float(klines_15m["volume"].iloc[-1])
        avg_vol    = float(klines_15m["volume"].iloc[:-3].mean())
        vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1
        if vol_ratio >= 5:
            score += 35
            signals.append(f"거래량 {vol_ratio:.1f}배↑")
        elif vol_ratio >= 3:
            score += 20
            signals.append(f"거래량 {vol_ratio:.1f}배↑")

        # 3. RSI 극단값
        rsi = ind_15m.get("rsi", 50) or 50
        if rsi <= 20:
            score += 25
            signals.append(f"RSI 극과매도 {rsi}")
            if direction == "wait":
                direction = "long"
        elif rsi >= 80:
            score += 25
            signals.append(f"RSI 극과매수 {rsi}")
            if direction == "wait":
                direction = "short"
        elif rsi <= 30:
            score += 12
            signals.append(f"RSI 과매도 {rsi}")
            if direction == "wait":
                direction = "long"
        elif rsi >= 70:
            score += 12
            signals.append(f"RSI 과매수 {rsi}")
            if direction == "wait":
                direction = "short"

        # 4. 펀딩비 극단
        fr      = get_funding_rate(sym)
        funding = fr.get("rate", 0)
        if abs(funding) >= 0.1:
            score += 20
            signals.append(f"펀딩비 극단 {funding:+.3f}%")
            if funding > 0 and direction == "wait":
                direction = "short"
            elif funding < 0 and direction == "wait":
                direction = "long"
        elif abs(funding) >= 0.05:
            score += 10
            signals.append(f"펀딩비 {funding:+.3f}%")

        if score < 20:
            return None

        atr     = ind_15m.get("atr", 0) or 0
        atr_pct = round(atr / price * 100, 3) if price > 0 else 0

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
            "indicators":      ind_15m,
            "indicators_1h":   ind_1h,
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
            articles = (
                data.get("data", {}).get("articles", [])
                or data.get("data", {}).get("catalogs", [{}])[0].get("articles", [])
            )
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

    analyst_prompt = f"""{sym} 알트코인 분석을 해주세요.

현재가: ${price:,}
{screener_summary}

기술적 지표 (15분봉):
RSI: {ind_15m.get('rsi')} / MACD Hist: {ind_15m.get('macd_hist')} / EMA20: {ind_15m.get('ema20')} / EMA50: {ind_15m.get('ema50')}
ATR: {ind_15m.get('atr')} / ADX: {ind_15m.get('adx')} / 볼린저밴드 상단: {ind_15m.get('bb_upper')} / 하단: {ind_15m.get('bb_lower')}"""

    news_prompt = f"""{sym} 알트코인의 최신 뉴스, 트위터/X 트렌드, 커뮤니티 분위기를 웹에서 검색해서 분석해주세요.
현재가: ${price:,} (1시간 변동: {candidate['price_change_1h']:+.2f}%)
이 코인 관련 최근 24시간 이내 뉴스, 공시, 개발팀 발표 등을 찾아주세요."""

    # analyst + news 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_analyst = ex.submit(run_agent, "alt_analyst", analyst_prompt)
        fut_news    = ex.submit(run_agent, "alt_news",    news_prompt)
        analyst_out = fut_analyst.result()
        news_out    = fut_news.result()

    # alt_trader 최종 결정
    trader_prompt = f"""알트코인 {sym} 매매 결정을 내려주세요.

현재가: ${price:,}

[스크리너 신호]
{screener_summary}

[기술적 분석]
{analyst_out}

[뉴스/센티멘트]
{news_out}

ATR 기반 SL/TP 참고:
- 롱 SL: ${round(price - (candidate['atr'] or 0), 4)} / TP: ${round(price + (candidate['atr'] or 0) * 2, 4)}
- 숏 SL: ${round(price + (candidate['atr'] or 0), 4)} / TP: ${round(price - (candidate['atr'] or 0) * 2, 4)}"""

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

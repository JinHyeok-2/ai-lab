#!/usr/bin/env python3
# 바이낸스 선물 실시간 급등/급락 감지 모듈
# REST 폴링 + 가격 스냅샷 비교 방식

import time
import requests
import logging
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)

# ── 설정 ──
_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# 급등 감지 임계값
SURGE_PCT_5M  = 3.0   # 5분 내 3% 이상 변동 → 급등/급락
SURGE_PCT_15M = 5.0   # 15분 내 5% 이상 변동
VOL_SPIKE_MULT = 5.0  # 거래량 평균 대비 5배 이상

# 스냅샷 보관 개수 (1분 간격 × 20 = 최근 20분)
MAX_SNAPSHOTS = 20

# 제외 심볼
_EXCLUDE = {
    "BTCUSDT", "ETHUSDT",
    "BUSDUSDT", "USDCUSDT", "USDTUSDT", "FDUSDUSDT", "DAIUSDT",
    "TUSDUSDT", "BTCDOMUSDT", "DEFIUSDT", "ALTUSDT",
}

# ── 가격 스냅샷 저장소 (모듈 레벨 싱글턴) ──
_snapshots: deque = deque(maxlen=MAX_SNAPSHOTS)
_lock = Lock()
_last_fetch_time = 0.0
_FETCH_COOLDOWN = 30  # 최소 30초 간격


def _fetch_all_tickers() -> dict[str, dict]:
    """바이낸스 선물 전종목 24h 틱커 조회 → {symbol: ticker_data}"""
    try:
        resp = requests.get(_TICKER_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        return {
            t["symbol"]: {
                "price": float(t["lastPrice"]),
                "volume": float(t["quoteVolume"]),
                "change_24h": float(t["priceChangePercent"]),
            }
            for t in resp.json()
            if t["symbol"].endswith("USDT") and float(t.get("quoteVolume", 0)) > 0
        }
    except Exception as e:
        logger.warning(f"틱커 조회 실패: {e}")
        return {}


def _take_snapshot() -> dict[str, dict] | None:
    """가격 스냅샷 촬영 + 저장"""
    global _last_fetch_time
    now = time.time()

    # 쿨다운 체크
    if now - _last_fetch_time < _FETCH_COOLDOWN:
        return None

    tickers = _fetch_all_tickers()
    if not tickers:
        return None

    with _lock:
        _snapshots.append({"time": now, "data": tickers})
        _last_fetch_time = now

    return tickers


def detect_surges(min_volume_usd: float = 10_000_000) -> list[dict]:
    """
    급등/급락 종목 감지
    1. 새 스냅샷 촬영
    2. 5분 전/15분 전 스냅샷과 비교
    3. 임계값 초과 종목 반환

    min_volume_usd: 최소 24h 거래대금 필터 (노이즈 제거)

    반환: [{"symbol", "price", "change_5m", "change_15m", "volume_24h",
            "direction", "surge_type", "score"}]
    """
    current = _take_snapshot()
    if current is None:
        # 쿨다운 중이면 마지막 스냅샷 사용
        with _lock:
            if _snapshots:
                current = _snapshots[-1]["data"]
            else:
                return []

    now = time.time()
    surges = []

    # 과거 스냅샷 찾기 (5분 전, 15분 전)
    snap_5m = None
    snap_15m = None
    with _lock:
        for snap in reversed(_snapshots):
            age = now - snap["time"]
            if age >= 270 and snap_5m is None:  # ~4.5분 이상
                snap_5m = snap["data"]
            if age >= 840 and snap_15m is None:  # ~14분 이상
                snap_15m = snap["data"]
                break

    if not snap_5m and not snap_15m:
        # 아직 충분한 히스토리 없음
        return []

    for sym, ticker in current.items():
        if sym in _EXCLUDE:
            continue
        if ticker["volume"] < min_volume_usd:
            continue

        price = ticker["price"]
        if price <= 0:
            continue

        # 5분 변동률
        change_5m = 0.0
        if snap_5m and sym in snap_5m and snap_5m[sym]["price"] > 0:
            change_5m = (price - snap_5m[sym]["price"]) / snap_5m[sym]["price"] * 100

        # 15분 변동률
        change_15m = 0.0
        if snap_15m and sym in snap_15m and snap_15m[sym]["price"] > 0:
            change_15m = (price - snap_15m[sym]["price"]) / snap_15m[sym]["price"] * 100

        # 급등/급락 판단
        is_surge_5m  = abs(change_5m) >= SURGE_PCT_5M
        is_surge_15m = abs(change_15m) >= SURGE_PCT_15M

        if not is_surge_5m and not is_surge_15m:
            continue

        # 거래량 스파이크 감지 (5분 전 대비)
        vol_ratio = 1.0
        if snap_5m and sym in snap_5m and snap_5m[sym]["volume"] > 0:
            vol_ratio = ticker["volume"] / snap_5m[sym]["volume"]

        # 점수 계산
        score = 0
        surge_type = []
        if is_surge_5m:
            score += min(int(abs(change_5m) / SURGE_PCT_5M * 30), 60)
            surge_type.append(f"5분 {change_5m:+.1f}%")
        if is_surge_15m:
            score += min(int(abs(change_15m) / SURGE_PCT_15M * 20), 40)
            surge_type.append(f"15분 {change_15m:+.1f}%")

        # 거래량 스파이크 보너스 (5배 이상 → +15점)
        if vol_ratio >= VOL_SPIKE_MULT:
            score += 15
            surge_type.append(f"거래량 {vol_ratio:.1f}x")

        # 방향 결정
        main_change = change_5m if abs(change_5m) > abs(change_15m) else change_15m
        direction = "long" if main_change > 0 else "short"

        surges.append({
            "symbol":     sym,
            "price":      price,
            "change_5m":  round(change_5m, 2),
            "change_15m": round(change_15m, 2),
            "change_24h": round(ticker["change_24h"], 2),
            "volume_24h": ticker["volume"],
            "direction":  direction,
            "surge_type": ", ".join(surge_type),
            "score":      min(score, 100),
        })

    # 점수순 정렬
    surges.sort(key=lambda x: x["score"], reverse=True)
    return surges[:10]  # 상위 10개만


def get_snapshot_count() -> int:
    """현재 저장된 스냅샷 개수"""
    with _lock:
        return len(_snapshots)


def get_snapshot_age_minutes() -> float:
    """가장 오래된 스냅샷 나이 (분)"""
    with _lock:
        if not _snapshots:
            return 0.0
        return (time.time() - _snapshots[0]["time"]) / 60


# ── 테스트용 ──
if __name__ == "__main__":
    print("=== 급등 감지 테스트 ===")
    print("스냅샷 1 촬영...")
    _take_snapshot()
    print(f"스냅샷: {get_snapshot_count()}개")

    # 실제로는 5분+ 대기 필요하지만 테스트용으로 바로 실행
    surges = detect_surges()
    print(f"급등 감지: {len(surges)}건")
    for s in surges[:5]:
        print(f"  {s['symbol']} {s['surge_type']} (score={s['score']})")

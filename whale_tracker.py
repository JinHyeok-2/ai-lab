#!/usr/bin/env python3
# 고래 추적 모듈 — Whale Alert 텔레그램 채널에서 대형 거래 파싱
# Telethon으로 @whale_alert_io 공개 채널 최근 메시지 읽기

import os
import re
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# .env 로드 (Telegram API 키)
try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

logger = logging.getLogger(__name__)

# ── 설정 ──
WHALE_CHANNEL = "whale_alert_io"  # @whale_alert_io
LOOKBACK_HOURS = 6                # 최근 N시간 메시지 분석
MIN_USD_AMOUNT = 10_000_000       # $10M 이상만 추적 (노이즈 제거)
TARGET_ASSETS = {"BTC", "ETH", "USDT", "USDC", "SOL"}  # 관심 자산

# 주요 거래소 키워드
EXCHANGES = {
    "binance", "coinbase", "kraken", "okx", "okex", "bybit", "bitfinex",
    "huobi", "kucoin", "gemini", "bitstamp", "upbit", "bithumb",
    "crypto.com", "gate.io", "mexc", "htx", "robinhood",
    "bitget", "deribit", "phemex", "bitmart", "lbank",
}

# ── 메시지 파싱 ──
# 형식: "🚨 15,121 #ETH (53,375,343 USD) transferred from Coinbase to unknown wallet"
_PATTERN = re.compile(
    r"([\d,]+(?:\.\d+)?)\s+#(\w+)\s+"           # 수량 + 토큰
    r"\(([\d,]+(?:\.\d+)?)\s+USD\)\s+"           # USD 금액
    r"transferred\s+from\s+(.+?)\s+to\s+(.+)",   # 출발지 → 도착지
    re.IGNORECASE
)


def _parse_whale_message(text: str) -> dict | None:
    """Whale Alert 메시지 한 줄 파싱 → dict 또는 None"""
    if not text or "transferred" not in text.lower():
        return None
    m = _PATTERN.search(text)
    if not m:
        return None
    amount_str, asset, usd_str, src, dst = m.groups()
    asset = asset.upper()
    if asset not in TARGET_ASSETS:
        return None
    usd = float(usd_str.replace(",", ""))
    if usd < MIN_USD_AMOUNT:
        return None

    # 해시태그 제거 (#Binance → binance)
    src_clean = src.strip().lstrip("#").lower()
    dst_clean = dst.strip().lstrip("#").lower()
    # [Details] 링크 제거
    src_clean = re.sub(r"\[details\].*", "", src_clean).strip()
    dst_clean = re.sub(r"\[details\].*", "", dst_clean).strip()
    src_is_exchange = any(ex in src_clean for ex in EXCHANGES)
    dst_is_exchange = any(ex in dst_clean for ex in EXCHANGES)

    # 방향 판단
    if dst_is_exchange and not src_is_exchange:
        direction = "exchange_deposit"   # 거래소 입금 → 매도 압력 (bearish)
    elif src_is_exchange and not dst_is_exchange:
        direction = "exchange_withdrawal"  # 거래소 출금 → 축적 (bullish)
    elif src_is_exchange and dst_is_exchange:
        direction = "exchange_transfer"    # 거래소 간 이동 (중립)
    else:
        direction = "wallet_transfer"      # 개인 지갑 간 이동 (중립)

    return {
        "asset": asset,
        "amount": float(amount_str.replace(",", "")),
        "usd": usd,
        "source": src.strip(),
        "destination": dst.strip(),
        "direction": direction,
        "src_is_exchange": src_is_exchange,
        "dst_is_exchange": dst_is_exchange,
    }


async def _fetch_whale_messages(limit: int = 200) -> list[dict]:
    """Telethon으로 Whale Alert 채널 최근 메시지 읽기"""
    from telethon import TelegramClient

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        logger.warning("TELEGRAM_API_ID / TELEGRAM_API_HASH 미설정")
        return []

    session_path = os.path.join(os.path.dirname(__file__), "whale_session")
    client = TelegramClient(session_path, int(api_id), api_hash)

    results = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    try:
        await client.start()
        channel = await client.get_entity(WHALE_CHANNEL)
        async for msg in client.iter_messages(channel, limit=limit):
            if msg.date and msg.date < cutoff:
                break
            if not msg.text:
                continue
            parsed = _parse_whale_message(msg.text)
            if parsed:
                parsed["timestamp"] = msg.date.isoformat()
                results.append(parsed)
    except Exception as e:
        logger.error(f"고래 추적 에러: {e}")
    finally:
        await client.disconnect()

    return results


def get_whale_signals() -> dict:
    """
    동기 래퍼 — run_analysis에서 호출용
    반환: {
        "summary": "텍스트 요약 (에이전트 전달용)",
        "deposits": [...],      # 거래소 입금 목록
        "withdrawals": [...],   # 거래소 출금 목록
        "net_flow": {"BTC": ..., "ETH": ...},  # 순유입 (양수=입금>출금, bearish)
        "signal": "bearish" | "bullish" | "neutral",
        "raw_count": int,
    }
    """
    try:
        loop = asyncio.new_event_loop()
        messages = loop.run_until_complete(_fetch_whale_messages())
        loop.close()
    except Exception as e:
        logger.error(f"고래 추적 실패: {e}")
        return _empty_result("고래 데이터 수집 실패")

    if not messages:
        return _empty_result(f"최근 {LOOKBACK_HOURS}시간 대형 고래 이동 없음")

    deposits = [m for m in messages if m["direction"] == "exchange_deposit"]
    withdrawals = [m for m in messages if m["direction"] == "exchange_withdrawal"]
    wallet_moves = [m for m in messages if m["direction"] == "wallet_transfer"]

    # 자산별 순유입 계산 (입금 - 출금, USD 기준)
    net_flow = {}
    for m in messages:
        asset = m["asset"]
        if asset not in net_flow:
            net_flow[asset] = 0.0
        if m["direction"] == "exchange_deposit":
            net_flow[asset] += m["usd"]
        elif m["direction"] == "exchange_withdrawal":
            net_flow[asset] -= m["usd"]

    # 전체 시그널 판단
    total_deposit_usd = sum(m["usd"] for m in deposits)
    total_withdrawal_usd = sum(m["usd"] for m in withdrawals)
    total_wallet_usd = sum(m["usd"] for m in wallet_moves)
    net_total = total_deposit_usd - total_withdrawal_usd

    if net_total > 50_000_000:
        signal = "bearish"
    elif net_total < -50_000_000:
        signal = "bullish"
    elif net_total > 20_000_000:
        signal = "slightly_bearish"
    elif net_total < -20_000_000:
        signal = "slightly_bullish"
    else:
        # 거래소 입출금이 없어도 대형 지갑 이동 자체가 시그널
        if total_wallet_usd > 500_000_000:
            signal = "large_movement"
        else:
            signal = "neutral"

    # 에이전트 전달용 요약 텍스트
    summary_lines = [f"[고래 추적 - 최근 {LOOKBACK_HOURS}시간, ${MIN_USD_AMOUNT/1e6:.0f}M+ 거래]"]
    summary_lines.append(f"감지 건수: {len(messages)}건 (거래소입금 {len(deposits)}, 출금 {len(withdrawals)}, 지갑이동 {len(wallet_moves)})")

    if deposits or withdrawals:
        summary_lines.append(f"거래소 입금: ${total_deposit_usd/1e6:.1f}M / 출금: ${total_withdrawal_usd/1e6:.1f}M")
        summary_lines.append(f"순유입: ${net_total/1e6:+.1f}M -> 시그널: {signal}")

    if wallet_moves:
        summary_lines.append(f"대형 지갑 이동: ${total_wallet_usd/1e6:.1f}M ({len(wallet_moves)}건)")

    for asset in sorted(net_flow.keys()):
        nf = net_flow.get(asset, 0)
        if nf != 0:
            summary_lines.append(f"  {asset}: ${nf/1e6:+.1f}M 순유입")

    # 전체 대형 5건 표시
    top5 = sorted(messages, key=lambda x: x["usd"], reverse=True)[:5]
    if top5:
        summary_lines.append("주요 건:")
        _dir_kr = {"exchange_deposit": "거래소입금", "exchange_withdrawal": "거래소출금",
                   "exchange_transfer": "거래소간이동", "wallet_transfer": "지갑이동"}
        for t in top5:
            summary_lines.append(
                f"  {t['asset']} ${t['usd']/1e6:.1f}M {_dir_kr.get(t['direction'], t['direction'])} "
                f"({t['source']} -> {t['destination']})"
            )

    return {
        "summary": "\n".join(summary_lines),
        "deposits": deposits,
        "withdrawals": withdrawals,
        "wallet_moves": wallet_moves,
        "net_flow": net_flow,
        "signal": signal,
        "raw_count": len(messages),
    }


def _empty_result(reason: str) -> dict:
    return {
        "summary": f"[고래 추적] {reason}",
        "deposits": [],
        "withdrawals": [],
        "net_flow": {},
        "signal": "neutral",
        "raw_count": 0,
    }


# ── 테스트용 ──
if __name__ == "__main__":
    # 파싱 테스트
    test_msgs = [
        "🚨 🚨 15,121 #ETH (53,375,343 USD) transferred from Coinbase Institutional to unknown new wallet",
        "🚨 99,999,969 #USDT (100,001,868 USD) transferred from unknown wallet to #Binance",
        "🚨 2,500 #BTC (215,000,000 USD) transferred from unknown wallet to Coinbase",
        "🚨 500 #BTC (43,000,000 USD) transferred from Binance to unknown wallet",
    ]
    print("=== 파싱 테스트 ===")
    for msg in test_msgs:
        r = _parse_whale_message(msg)
        if r:
            print(f"  {r['asset']} ${r['usd']/1e6:.1f}M {r['direction']}")

    # 실제 채널 연결 테스트 (API 키 필요)
    import dotenv
    dotenv.load_dotenv()
    result = get_whale_signals()
    print(f"\n=== 채널 조회 ===")
    print(result["summary"])

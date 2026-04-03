#!/usr/bin/env python3
# pykrx 기반 ETF 일봉 데이터 수집
import sys; sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from pathlib import Path
from pykrx import stock as krx

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_etf(ticker: str, start: str = "20190101", end: str = "20260403",
                 use_cache: bool = True) -> pd.DataFrame:
    """
    ETF 일봉 데이터 다운로드 (pykrx)

    Args:
        ticker: 종목코드 (예: '069500')
        start/end: 'YYYYMMDD' 형식
        use_cache: True면 로컬 캐시 사용

    Returns:
        DataFrame(index=날짜, columns=[open, high, low, close, volume])
    """
    cache_file = DATA_DIR / f"{ticker}_{start}_{end}.parquet"

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        print(f"  [캐시] {ticker} {len(df)}일치 로드")
        return df

    print(f"  [다운로드] {ticker} {start}~{end}...")
    df = krx.get_etf_ohlcv_by_date(start, end, ticker)

    if df is None or df.empty:
        # ETF 전용 함수 실패 시 일반 주식 함수로 재시도
        df = krx.get_market_ohlcv_by_date(start, end, ticker)

    if df is None or df.empty:
        print(f"  [경고] {ticker} 데이터 없음")
        return pd.DataFrame()

    # 컬럼명 영문 통일
    col_map = {
        "시가": "open", "고가": "high", "저가": "low",
        "종가": "close", "거래량": "volume",
        "NAV": "nav", "거래대금": "amount",
    }
    df = df.rename(columns=col_map)

    # 필수 컬럼만 유지
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    # 0 거래량 제거 (비거래일)
    df = df[df["volume"] > 0]

    # 캐시 저장
    df.to_parquet(cache_file)
    print(f"  [저장] {ticker} {len(df)}일치 → {cache_file.name}")
    return df


def download_universe(tickers: dict, start: str = "20190101", end: str = "20260403") -> dict:
    """
    여러 ETF 일괄 다운로드

    Args:
        tickers: {종목코드: 이름} dict
        start/end: 기간

    Returns:
        {종목코드: DataFrame} dict
    """
    data = {}
    for ticker, name in tickers.items():
        print(f"[{name}] ({ticker})")
        df = download_etf(ticker, start, end)
        if not df.empty:
            data[ticker] = df
    return data


# 백테스트 대상 ETF 유니버스
BACKTEST_UNIVERSE = {
    # 지수
    "069500": "KODEX 200",
    "229200": "KODEX 코스닥150",
    # 섹터
    "091160": "KODEX 반도체",
    "305720": "KODEX 2차전지산업",
    # 안전자산
    "148070": "KOSEF 국고채10년",
    "132030": "KODEX 골드선물(H)",
    # 레버리지/인버스 (스캘핑용)
    "122630": "KODEX 레버리지",
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
}


if __name__ == "__main__":
    # 테스트: 전체 유니버스 다운로드
    data = download_universe(BACKTEST_UNIVERSE)
    print(f"\n다운로드 완료: {len(data)}종목")
    for t, df in data.items():
        print(f"  {t}: {len(df)}일, {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

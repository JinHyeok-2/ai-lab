#!/usr/bin/env python3
# 분봉 데이터 일일 수집기 — 매일 장마감 후 실행
# KIS API: 당일 분봉만 조회 가능 → 매일 저장해서 축적
import sys; sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from pathlib import Path
from datetime import datetime, date

from kis_client import connect

# 저장 경로
DATA_DIR = Path(__file__).parent / "minute_data"
DATA_DIR.mkdir(exist_ok=True)

# 수집 대상 ETF (스캘핑 후보)
SCALP_TICKERS = {
    "069500": "KODEX 200",
    "122630": "KODEX 레버리지",
    "252670": "KODEX 인버스2X",
    "229200": "KODEX 코스닥150",
    "091160": "KODEX 반도체",
    "305720": "KODEX 2차전지산업",
}

# 수집 타임프레임 (분)
TIMEFRAMES = [1, 5, 15, 30, 60]


def collect_minute(ticker: str, name: str, period: int) -> pd.DataFrame:
    """특정 종목의 분봉 데이터 수집"""
    kis = connect()
    stock = kis.stock(ticker)

    try:
        chart = stock.chart(period=period)
        df = chart.df()
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        print(f"  [오류] {name} {period}분봉: {e}")

    return pd.DataFrame()


def collect_all():
    """전체 ETF × 전체 타임프레임 수집"""
    today = date.today().strftime("%Y%m%d")
    print(f"={'='*60}")
    print(f"  분봉 데이터 수집 — {today}")
    print(f"={'='*60}")

    total_saved = 0

    for ticker, name in SCALP_TICKERS.items():
        print(f"\n[{name}] ({ticker})")
        ticker_dir = DATA_DIR / ticker
        ticker_dir.mkdir(exist_ok=True)

        for tf in TIMEFRAMES:
            tf_label = f"{tf}m" if tf < 60 else f"{tf//60}h"
            df = collect_minute(ticker, name, tf)

            if df.empty:
                print(f"  {tf_label}: 데이터 없음")
                continue

            # 저장 (일자별 파일)
            filename = f"{today}_{tf_label}.parquet"
            filepath = ticker_dir / filename
            df.to_parquet(filepath)
            print(f"  {tf_label}: {len(df)}봉 → {filepath.name}")
            total_saved += 1

    print(f"\n{'='*60}")
    print(f"  수집 완료: {total_saved}개 파일 저장")
    print(f"  저장 위치: {DATA_DIR}")
    print(f"{'='*60}")

    return total_saved


def get_accumulated_stats():
    """축적된 데이터 현황 출력"""
    print(f"\n분봉 데이터 축적 현황:")
    print(f"  경로: {DATA_DIR}")
    total_files = 0
    for ticker_dir in sorted(DATA_DIR.iterdir()):
        if ticker_dir.is_dir():
            files = list(ticker_dir.glob("*.parquet"))
            days = len(set(f.name.split("_")[0] for f in files))
            total_files += len(files)
            name = SCALP_TICKERS.get(ticker_dir.name, ticker_dir.name)
            print(f"  {name} ({ticker_dir.name}): {days}일, {len(files)}파일")
    print(f"  총: {total_files}파일")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="분봉 데이터 수집")
    parser.add_argument("--stats", action="store_true", help="축적 현황 출력")
    args = parser.parse_args()

    if args.stats:
        get_accumulated_stats()
    else:
        collect_all()

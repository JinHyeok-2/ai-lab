#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT RL 시그널 실거래 성능 검증 — A/B/C 그룹 비교
# 50건+ 축적 후 실행: conda run -n ai-lab python analyze_alt_rl_performance.py

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent.parent / "trades.db"

# RL 프로덕션 적용 시점 (이후 거래만 분석)
RL_START = "2026-03-26 12:00:00"

# 학습에 포함된 코인 (14종 → 26종 v2 기준)
TRAINED_COINS_V1 = {
    "SOL", "DOGE", "XRP", "ADA", "AVAX", "LINK", "DOT", "NEAR",
    "ARB", "OP", "SUI", "APT", "WIF", "TAO",
}
TRAINED_COINS_V2 = TRAINED_COINS_V1 | {
    "ONT", "ZEC", "FIL", "JTO", "AAVE", "UNI", "ATOM", "FTM",
    "ETC", "TIA", "SEI", "DYDX",
}


def load_trades():
    """RL 적용 이후 알트 거래 로드"""
    db = sqlite3.connect(str(DB_PATH))
    query = f"""
        SELECT id, symbol, side, action, pnl, confidence, time, close_time, extra
        FROM trades
        WHERE symbol NOT IN ('ETHUSDT', 'BTCUSDT')
          AND pnl != 0
          AND close_price > 0
          AND time >= '{RL_START}'
        ORDER BY time
    """
    df = pd.read_sql(query, db)
    db.close()
    return df


def classify_trade(row):
    """거래를 A/B/C 그룹으로 분류
    A: RL 롱 합의 (RL=롱, 에이전트=롱)
    B: RL 관망인데 진입 (RL=관망, 에이전트=롱)
    C: RL 미적용/실패 (RL 정보 없음)
    """
    extra = row.get("extra", "")
    if not extra or pd.isna(extra):
        return "C"

    try:
        import json
        ext = json.loads(extra) if isinstance(extra, str) else {}
    except:
        return "C"

    rl_type = ext.get("rl_type", "")
    if rl_type == "long" and "롱" in str(row.get("side", "")):
        return "A"
    elif rl_type == "wait" and "롱" in str(row.get("side", "")):
        return "B"
    else:
        return "C"


def analyze_groups(df):
    """A/B/C 그룹별 성과 분석"""
    print(f"\n{'='*60}")
    print(f"그룹별 성과 비교")
    print(f"{'='*60}")
    print(f"{'그룹':<20} {'건수':>6} {'승률':>8} {'총PnL':>10} {'평균PnL':>10} {'평균conf':>10}")
    print(f"{'-'*65}")

    for group in ["A", "B", "C"]:
        g = df[df["group"] == group]
        if len(g) == 0:
            print(f"{group_label(group):<20} {'0':>6}")
            continue
        wins = (g["pnl"] > 0).sum()
        wr = wins / len(g) * 100
        total_pnl = g["pnl"].sum()
        avg_pnl = g["pnl"].mean()
        avg_conf = g["confidence"].mean()
        print(f"{group_label(group):<20} {len(g):>6} {wr:>7.1f}% ${total_pnl:>+8.2f} ${avg_pnl:>+8.3f} {avg_conf:>9.1f}")

    # 검증 결론
    a = df[df["group"] == "A"]
    b = df[df["group"] == "B"]
    c = df[df["group"] == "C"]

    print(f"\n--- 검증 결론 ---")
    if len(a) >= 5 and len(b) >= 5:
        a_wr = (a["pnl"] > 0).mean() * 100
        b_wr = (b["pnl"] > 0).mean() * 100
        if a_wr > b_wr + 10:
            print(f"✅ RL 효과 검증 성공: A그룹 승률 {a_wr:.0f}% >> B그룹 {b_wr:.0f}%")
            print(f"   → RL 가중치 유지 또는 강화 권장")
        elif a_wr > b_wr:
            print(f"⚠️ RL 약한 효과: A그룹 {a_wr:.0f}% > B그룹 {b_wr:.0f}% (차이 {a_wr-b_wr:.0f}%p)")
            print(f"   → 추가 데이터 축적 후 재검증")
        else:
            print(f"❌ RL 효과 미확인: A그룹 {a_wr:.0f}% <= B그룹 {b_wr:.0f}%")
            print(f"   → RL 가중치 축소 검토")
    else:
        print(f"⏳ 데이터 부족: A={len(a)}건, B={len(b)}건 (각 5건+ 필요)")


def analyze_trained_vs_untrained(df):
    """학습 포함 vs 미포함 코인 비교"""
    print(f"\n{'='*60}")
    print(f"학습 포함 vs 미포함 코인 비교")
    print(f"{'='*60}")

    df["coin"] = df["symbol"].str.replace("USDT", "")
    df["in_training"] = df["coin"].isin(TRAINED_COINS_V2)

    for label, mask in [("학습 포함", True), ("학습 미포함", False)]:
        g = df[df["in_training"] == mask]
        if len(g) == 0:
            print(f"{label}: 0건")
            continue
        wins = (g["pnl"] > 0).sum()
        wr = wins / len(g) * 100
        print(f"{label}: {len(g)}건, 승률 {wr:.0f}%, PnL ${g['pnl'].sum():+.2f}")
        coins = g.groupby("coin").agg(cnt=("pnl", "count"), wr=("pnl", lambda x: (x>0).mean()*100), pnl=("pnl", "sum")).sort_values("pnl", ascending=False)
        for _, r in coins.iterrows():
            print(f"  {r.name:<8} {int(r['cnt'])}건 승률{r['wr']:.0f}% PnL${r['pnl']:+.2f}")

    # 차이 분석
    trained = df[df["in_training"]]
    untrained = df[~df["in_training"]]
    if len(trained) >= 5 and len(untrained) >= 5:
        t_wr = (trained["pnl"] > 0).mean() * 100
        u_wr = (untrained["pnl"] > 0).mean() * 100
        diff = t_wr - u_wr
        if diff > 15:
            print(f"\n⚠️ 학습 포함 코인이 {diff:.0f}%p 우수 → 미포함 코인 RL 가중치 축소 권장")
        else:
            print(f"\n✅ 학습 유무 차이 {diff:.0f}%p — 범용 모델 효과 양호")


def analyze_confidence_calibration(df):
    """신뢰도 구간별 승률 캘리브레이션"""
    print(f"\n{'='*60}")
    print(f"신뢰도 구간별 승률")
    print(f"{'='*60}")

    bins = [(0, 5), (5, 8), (8, 10), (10, 13), (13, 100)]
    print(f"{'구간':<12} {'건수':>6} {'승률':>8} {'평균PnL':>10}")
    print(f"{'-'*40}")
    for lo, hi in bins:
        g = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
        if len(g) == 0:
            continue
        wr = (g["pnl"] > 0).mean() * 100
        avg = g["pnl"].mean()
        print(f"{lo:>2}~{hi:<4}      {len(g):>6} {wr:>7.1f}% ${avg:>+8.3f}")


def group_label(g):
    labels = {"A": "A(RL롱+에이전트롱)", "B": "B(RL관망+진입)", "C": "C(RL미적용)"}
    return labels.get(g, g)


def main():
    print("=== ALT RL 시그널 실거래 성능 검증 ===")
    print(f"분석 기간: {RL_START} ~ 현재\n")

    df = load_trades()
    print(f"알트 거래 총 {len(df)}건")

    if len(df) < 10:
        print(f"\n⏳ 데이터 부족 ({len(df)}건). 최소 50건+ 축적 후 재실행하세요.")
        return

    # 그룹 분류
    df["group"] = df.apply(classify_trade, axis=1)
    print(f"그룹 분포: A={len(df[df['group']=='A'])}건, B={len(df[df['group']=='B'])}건, C={len(df[df['group']=='C'])}건")

    # 분석
    analyze_groups(df)
    analyze_trained_vs_untrained(df)
    analyze_confidence_calibration(df)

    # 종합 권장사항
    print(f"\n{'='*60}")
    print("종합 권장사항")
    print(f"{'='*60}")

    a_count = len(df[df["group"] == "A"])
    total = len(df)
    if a_count >= 20:
        a_wr = (df[df["group"] == "A"]["pnl"] > 0).mean() * 100
        if a_wr >= 70:
            print("→ RL 가중치 강화 (+15pt 이상 검토)")
        elif a_wr >= 55:
            print("→ RL 가중치 유지 (현재 설정 적정)")
        else:
            print("→ RL 가중치 축소 검토 (A그룹 승률 저조)")
    else:
        print(f"→ 추가 데이터 필요 (현재 A={a_count}건, 목표 20건+)")

    print(f"\n총 {total}건 분석 완료")


if __name__ == "__main__":
    main()

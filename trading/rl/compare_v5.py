#!/usr/bin/env python3
# v4 vs v4.1 vs v5 백테스트 비교
# 실행: conda activate APCC && python trading/rl/compare_v5.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from stable_baselines3 import PPO

from rl.data import load_data
from rl.env import ETHTradingEnv
from rl.env_v41 import ETHTradingEnvV41
from rl.env_v5 import ETHTradingEnvV5

INTERVAL = "30m"
RL_DIR   = Path(__file__).parent


def run_model(env, model_path):
    """공통 백테스트 루프"""
    model = PPO.load(str(model_path))
    obs, _ = env.reset()
    balances = [env.initial_balance]
    actions  = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
        balances.append(info["balance"])
        actions.append(int(action))

    return balances, actions, info


def calc_stats(balances, info, bh_ret):
    final = balances[-1]
    ret   = (final - 10000) / 10000 * 100
    arr   = np.array(balances)
    peak  = np.maximum.accumulate(arr)
    mdd   = ((arr - peak) / peak * 100).min()
    return {
        "final": final, "return": ret, "buy_hold": bh_ret,
        "mdd": mdd, "trades": info["total_trades"],
        "win_rate": info["win_rate"],
    }


def print_stats(label, s, actions):
    dist  = Counter(actions)
    total = len(actions)
    names = {0: "관망", 1: "롱", 2: "숏", 3: "청산"}
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  최종 잔고:   ${s['final']:,.2f}")
    print(f"  총 수익률:   {s['return']:+.2f}%")
    print(f"  Buy & Hold:  {s['buy_hold']:+.2f}%")
    print(f"  최대낙폭:    {s['mdd']:.2f}%")
    print(f"  총 거래:     {s['trades']}회 | 승률: {s['win_rate']:.1%}")
    print(f"  행동 분포:   " + " / ".join(
        f"{names[k]}:{v}({v/total:.0%})" for k, v in sorted(dist.items())))


def plot_comparison(test_prices, results, stats_list, labels):
    """3개 모델 비교 그래프"""
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    fig.suptitle(f"v4 vs v4.1 vs v5 백테스트 비교 [{INTERVAL}]",
                 fontsize=14, fontweight="bold")

    colors = ["#1565C0", "#E65100", "#2E7D32"]

    # 1. 잔고 비교
    ax1 = axes[0]
    for i, (bal, label) in enumerate(zip([r[0] for r in results], labels)):
        s = stats_list[i]
        ax1.plot(bal, color=colors[i], lw=1.8,
                 label=f"{label}  {s['return']:+.1f}%")

    # Buy & Hold
    bh = [10000 * (p / test_prices[0]) for p in test_prices]
    ax1.plot(range(len(bh)), bh, color="#aaa", lw=1, ls="--",
             label=f"Buy & Hold  {stats_list[0]['buy_hold']:+.1f}%")
    ax1.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax1.set_ylabel("잔고 (USDT)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("잔고 변화")

    # 2. v5 매매 포인트
    ax2 = axes[1]
    v5_actions = results[2][1]
    n_prices   = min(len(test_prices), len(v5_actions))
    prices     = test_prices[:n_prices]
    ax2.plot(range(len(prices)), prices, color="#555", lw=1)
    markers = {1: ("#4CAF50", "^", "롱"), 2: ("#e53935", "v", "숏"),
               3: ("#FF9800", "x", "청산")}
    for a, (col, mk, lbl) in markers.items():
        idx = [i for i, x in enumerate(v5_actions) if x == a and i < n_prices]
        ax2.scatter(idx, [prices[i] for i in idx],
                    marker=mk, color=col, s=25, label=f"v5 {lbl}", zorder=5)
    ax2.set_ylabel("ETH 가격 (USDT)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title("v5 매매 포인트")

    # 3. 성과 비교표
    ax3 = axes[2]
    ax3.axis("off")
    header = ["지표"] + labels
    rows = [
        ["총 수익률"]  + [f"{s['return']:+.2f}%" for s in stats_list],
        ["Buy & Hold"] + [f"{s['buy_hold']:+.2f}%" for s in stats_list],
        ["최대낙폭"]   + [f"{s['mdd']:.2f}%" for s in stats_list],
        ["총 거래"]    + [f"{s['trades']}회" for s in stats_list],
        ["승률"]       + [f"{s['win_rate']:.1%}" for s in stats_list],
    ]
    tbl = ax3.table(cellText=rows, colLabels=header,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    for j in range(len(header)):
        tbl[0, j].set_facecolor("#1a237e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # v5 열 하이라이트
    for i in range(1, len(rows) + 1):
        tbl[i, 3].set_facecolor("#e8f5e9")

    plt.tight_layout()
    out = RL_DIR / "backtest_compare_v5.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장: {out}")
    plt.close()


if __name__ == "__main__":
    # ── 공통 테스트 데이터 (eth_30m_v41.csv, 2025-09-01~) ──
    v41_csv = RL_DIR / "eth_30m_v41.csv"
    if not v41_csv.exists():
        print("❌ eth_30m_v41.csv 없음 — 먼저 데이터 수집 필요")
        sys.exit(1)

    full_df = pd.read_csv(v41_csv, parse_dates=["time"])
    cutoff  = pd.Timestamp("2025-09-01")
    test_df = full_df[full_df["time"] >= cutoff].reset_index(drop=True)

    print(f"테스트 구간: {test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()}")
    print(f"테스트 캔들: {len(test_df):,}개")
    bh_ret = (test_df["close"].iloc[-1] - test_df["close"].iloc[20]) \
             / test_df["close"].iloc[20] * 100
    print(f"Buy & Hold: {bh_ret:+.2f}%\n")

    all_results = []
    all_stats   = []
    labels      = []

    # ── v4 (LEV×5, 피처 7개) ────────────────────────────
    v4_path = RL_DIR / "models/v4/ppo_eth_30m.zip"
    if v4_path.exists():
        print("▶ v4 백테스트...")
        env_v4 = ETHTradingEnv(test_df, initial_balance=10000.0,
                               leverage=5, window_size=20, min_hold_steps=4)
        bal, act, info = run_model(env_v4, v4_path)
        all_results.append((bal, act, info))
        all_stats.append(calc_stats(bal, info, bh_ret))
        labels.append("v4 (LEV×5)")
        print_stats("v4 (LEV×5, 피처 7개)", all_stats[-1], act)
    else:
        print("⚠️ v4 모델 없음, 건너뜀")

    # ── v4.1 (LEV×3, 피처 10개) ──────────────────────────
    v41_path = RL_DIR / "models/v41/ppo_eth_30m.zip"
    if v41_path.exists():
        print("\n▶ v4.1 백테스트...")
        env_v41 = ETHTradingEnvV41(test_df, initial_balance=10000.0,
                                   leverage=3, window_size=20, min_hold_steps=4)
        bal, act, info = run_model(env_v41, v41_path)
        all_results.append((bal, act, info))
        all_stats.append(calc_stats(bal, info, bh_ret))
        labels.append("v4.1 (LEV×3)")
        print_stats("v4.1 (LEV×3, 피처 10개)", all_stats[-1], act)
    else:
        print("⚠️ v4.1 모델 없음, 건너뜀")

    # ── v5 (LEV×3, 피처 13개) ────────────────────────────
    v5_path = RL_DIR / "models/v5/ppo_eth_30m.zip"
    if v5_path.exists():
        print("\n▶ v5 백테스트...")
        env_v5 = ETHTradingEnvV5(
            test_df, initial_balance=10000.0,
            leverage=3, window_size=20, min_hold_steps=4,
            max_episode_len=len(test_df) + 100,  # 전체 데이터 사용
            max_drawdown=1.0,                     # 백테스트 시 MDD 종료 비활성
            curriculum=False,
        )
        bal, act, info = run_model(env_v5, v5_path)
        all_results.append((bal, act, info))
        all_stats.append(calc_stats(bal, info, bh_ret))
        labels.append("v5 (LEV×3)")
        print_stats("v5 (LEV×3, 피처 13개, 커리큘럼)", all_stats[-1], act)
    else:
        print("⚠️ v5 모델 없음 — 먼저 학습 필요: python trading/rl/train_v5.py")

    # ── 비교 그래프 ──────────────────────────────────────
    if len(all_results) >= 2:
        # 가격 데이터 (가장 긴 백테스트 기준)
        max_len = max(len(r[0]) for r in all_results)
        prices  = [float(test_df["close"].iloc[min(i + 20, len(test_df) - 1)])
                   for i in range(max_len)]
        plot_comparison(prices, all_results, all_stats, labels)

    print("\n✅ 비교 완료")

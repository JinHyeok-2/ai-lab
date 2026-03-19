#!/usr/bin/env python3
# v4 vs v4.1 백테스트 비교
# 실행: python trading/rl/compare_v4_v41.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.data import load_data
from rl.env import ETHTradingEnv
from rl.env_v41 import ETHTradingEnvV41

INTERVAL = "30m"

def run_v4(test_df):
    model_path = Path(__file__).parent / "models/v4/ppo_eth_30m.zip"
    env   = ETHTradingEnv(test_df, initial_balance=10000.0, leverage=5,
                          window_size=20, min_hold_steps=4)
    model = PPO.load(str(model_path))
    obs, _ = env.reset()
    balances, prices, actions = [10000.0], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
        balances.append(info["balance"])
        idx = min(env.current_step, len(test_df) - 1)
        prices.append(float(test_df["close"].iloc[idx - 1]))
        actions.append(int(action))
    return balances, prices, actions, info

def run_v41(test_df):
    model_path = Path(__file__).parent / "models/v41/ppo_eth_30m.zip"
    env   = ETHTradingEnvV41(test_df, initial_balance=10000.0, leverage=3,
                             window_size=20, min_hold_steps=4)
    model = PPO.load(str(model_path))
    obs, _ = env.reset()
    balances, prices, actions = [10000.0], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
        balances.append(info["balance"])
        idx = min(env.current_step, len(test_df) - 1)
        prices.append(float(test_df["close"].iloc[idx - 1]))
        actions.append(int(action))
    return balances, prices, actions, info

def calc_stats(balances, info, bh_ret):
    final = balances[-1]
    ret   = (final - 10000) / 10000 * 100
    arr   = np.array(balances)
    peak  = np.maximum.accumulate(arr)
    mdd   = ((arr - peak) / peak * 100).min()
    return {"final": final, "return": ret, "buy_hold": bh_ret,
            "mdd": mdd, "trades": info["total_trades"], "win_rate": info["win_rate"]}

def print_stats(label, s, actions):
    dist  = Counter(actions)
    total = len(actions)
    names = {0:"관망", 1:"롱", 2:"숏", 3:"청산"}
    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"{'='*48}")
    print(f"  최종 잔고:   ${s['final']:,.2f}")
    print(f"  총 수익률:   {s['return']:+.2f}%")
    print(f"  Buy & Hold:  {s['buy_hold']:+.2f}%")
    print(f"  최대낙폭:    {s['mdd']:.2f}%")
    print(f"  총 거래:     {s['trades']}회 | 승률: {s['win_rate']:.1%}")
    print(f"  행동 분포:   " + " / ".join(f"{names[k]}:{v}({v/total:.0%})" for k,v in sorted(dist.items())))

def plot_comparison(test_df, v4_bal, v4_prices, v4_act, v41_bal, v41_prices, v41_act, s4, s41):
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f"v4 vs v4.1 백테스트 비교 [{INTERVAL}]", fontsize=14, fontweight="bold")

    # 1. 잔고 비교
    ax1 = axes[0]
    bh  = [10000 * (p / v4_prices[0]) for p in v4_prices]
    ax1.plot(v4_bal,  color="#1565C0", lw=1.8, label=f"v4  (LEV×5)   {s4['return']:+.1f}%")
    ax1.plot(v41_bal, color="#E65100", lw=1.8, label=f"v4.1(LEV×3)  {s41['return']:+.1f}%")
    ax1.plot(range(len(bh)), bh, color="#aaa", lw=1, ls="--", label=f"Buy & Hold  {s4['buy_hold']:+.1f}%")
    ax1.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax1.set_ylabel("잔고 (USDT)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("잔고 변화")

    # 2. ETH 가격 + v4.1 매매 포인트
    ax2 = axes[1]
    prices = v41_prices[:len(v41_act)]
    ax2.plot(range(len(prices)), prices, color="#555", lw=1)
    colors = {1: ("#4CAF50", "^", "롱▲"), 2: ("#e53935", "v", "숏▼"), 3: ("#FF9800", "x", "청산×")}
    for a, (col, mk, lbl) in colors.items():
        idx = [i for i, x in enumerate(v41_act) if x == a and i < len(prices)]
        ax2.scatter(idx, [prices[i] for i in idx], marker=mk, color=col, s=25, label=f"v4.1 {lbl}", zorder=5)
    ax2.set_ylabel("ETH 가격 (USDT)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title("v4.1 매매 포인트")

    # 3. 성과 비교 표
    ax3 = axes[2]
    ax3.axis("off")
    rows = [
        ["지표", "v4 (LEV×5)", "v4.1 (LEV×3)"],
        ["총 수익률",    f"{s4['return']:+.2f}%",  f"{s41['return']:+.2f}%"],
        ["Buy & Hold",  f"{s4['buy_hold']:+.2f}%", f"{s41['buy_hold']:+.2f}%"],
        ["최대낙폭(MDD)", f"{s4['mdd']:.2f}%",      f"{s41['mdd']:.2f}%"],
        ["총 거래",      f"{s4['trades']}회",        f"{s41['trades']}회"],
        ["승률",         f"{s4['win_rate']:.1%}",   f"{s41['win_rate']:.1%}"],
    ]
    tbl = ax3.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    for j in range(3):
        tbl[0, j].set_facecolor("#1a237e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)):
        tbl[i, 2].set_facecolor("#fff3e0")

    plt.tight_layout()
    out = Path(__file__).parent / "backtest_compare_v4_v41.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장: {out}")
    plt.close()

if __name__ == "__main__":
    # v4 기준 테스트 구간 사용
    meta = json.loads((Path(__file__).parent / "models/v4/meta_30m.json").read_text())
    df   = load_data(INTERVAL)
    test_df = df.iloc[meta["train_candles"]:].reset_index(drop=True)

    # v4.1은 자체 데이터로 테스트 (더 최신 데이터 포함)
    v41_df_path = Path(__file__).parent / "eth_30m_v41.csv"
    v41_full = pd.read_csv(v41_df_path, parse_dates=["time"])
    v41_meta = json.loads((Path(__file__).parent / "models/v41/meta_30m.json").read_text())
    v41_test = v41_full.iloc[v41_meta["train_candles"]:].reset_index(drop=True)

    print(f"테스트 구간 (v4):   {test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()}  ({len(test_df)}캔들)")
    print(f"테스트 구간 (v4.1): {v41_test['time'].iloc[0].date()} ~ {v41_test['time'].iloc[-1].date()}  ({len(v41_test)}캔들)")

    bh_ret = (test_df["close"].iloc[-1] - test_df["close"].iloc[20]) / test_df["close"].iloc[20] * 100

    print("\n▶ v4 백테스트...")
    v4_bal, v4_prices, v4_act, v4_info = run_v4(test_df)
    s4 = calc_stats(v4_bal, v4_info, bh_ret)

    bh_ret41 = (v41_test["close"].iloc[-1] - v41_test["close"].iloc[20]) / v41_test["close"].iloc[20] * 100
    print("▶ v4.1 백테스트...")
    v41_bal, v41_prices, v41_act, v41_info = run_v41(v41_test)
    s41 = calc_stats(v41_bal, v41_info, bh_ret41)

    print_stats("v4   (LEV×5, 7920캔들 학습, 피처 7개)", s4, v4_act)
    print_stats("v4.1 (LEV×3, 29760캔들 학습, 피처 10개)", s41, v41_act)

    plot_comparison(test_df, v4_bal, v4_prices, v4_act, v41_bal, v41_prices, v41_act, s4, s41)
    print("\n✅ 비교 완료")

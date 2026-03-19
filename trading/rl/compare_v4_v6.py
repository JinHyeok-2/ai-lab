#!/usr/bin/env python3
# v4 vs v6 백테스트 비교 스크립트
# 실행: python trading/rl/compare_v4_v6.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.data import load_data
from rl.env import ETHTradingEnv

INTERVAL = "30m"

def run_v4(test_df):
    model_path = Path(__file__).parent / "models/v4/ppo_eth_30m.zip"
    env = ETHTradingEnv(test_df, initial_balance=10000.0, leverage=5,
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

def run_v6(test_df):
    model_path = Path(__file__).parent / "models/v6/ppo_eth_30m.zip"
    vec_path   = Path(__file__).parent / "models/v6/vecnormalize_30m.pkl"

    # VecNormalize로 래핑
    venv = DummyVecEnv([lambda: ETHTradingEnv(test_df, initial_balance=10000.0,
                        leverage=3, window_size=20, min_hold_steps=4)])
    venv = VecNormalize.load(str(vec_path), venv)
    venv.training = False   # 추론 시 통계 업데이트 중단
    venv.norm_reward = False

    model = PPO.load(str(model_path), env=venv)

    obs = venv.reset()
    balances, prices, actions = [10000.0], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        done = dones[0]
        balances.append(info["balance"])
        idx = min(
            venv.envs[0].current_step if hasattr(venv, 'envs') else len(test_df) - 1,
            len(test_df) - 1
        )
        prices.append(float(test_df["close"].iloc[idx - 1]))
        actions.append(int(action[0]))

    return balances, prices, actions, info

def calc_stats(balances, info, buy_hold_ret):
    final = balances[-1]
    ret   = (final - 10000) / 10000 * 100
    arr   = np.array(balances)
    peak  = np.maximum.accumulate(arr)
    mdd   = ((arr - peak) / peak * 100).min()
    return {
        "final":       final,
        "return":      ret,
        "buy_hold":    buy_hold_ret,
        "mdd":         mdd,
        "trades":      info["total_trades"],
        "win_rate":    info["win_rate"],
    }

def plot_comparison(test_df, v4_bal, v4_prices, v4_act,
                                  v6_bal, v6_prices, v6_act,
                                  s4, s6):
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f"v4 vs v6 백테스트 비교 [{INTERVAL}]", fontsize=14, fontweight="bold")

    # 1. 잔고 비교
    ax1 = axes[0]
    bh_bal = [10000 * (p / v4_prices[0]) for p in v4_prices]
    ax1.plot(v4_bal, color="#1565C0", lw=1.8, label=f"v4 (LEV×5)  {s4['return']:+.1f}%")
    ax1.plot(v6_bal, color="#2E7D32", lw=1.8, label=f"v6 (LEV×3)  {s6['return']:+.1f}%")
    ax1.plot(range(len(bh_bal)), bh_bal, color="#aaa", lw=1, ls="--",
             label=f"Buy & Hold  {s4['buy_hold']:+.1f}%")
    ax1.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax1.set_ylabel("잔고 (USDT)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("잔고 변화")

    # 2. 매매 포인트 (v4 위, v6 아래)
    ax2 = axes[1]
    prices = v4_prices[:len(v4_act)]
    ax2.plot(range(len(prices)), prices, color="#555", lw=1, label="ETH 가격")
    action_names = {1: ("롱▲", "#4CAF50", "^", 30), 2: ("숏▼", "#e53935", "v", 30), 3: ("청산×", "#FF9800", "x", 20)}
    for a, (lbl, col, mk, sz) in action_names.items():
        idx = [i for i, x in enumerate(v4_act) if x == a and i < len(prices)]
        ax2.scatter(idx, [prices[i] for i in idx], marker=mk, color=col, s=sz, label=f"v4 {lbl}", zorder=5, alpha=0.7)
    for a, (lbl, col, mk, sz) in action_names.items():
        prices6 = v6_prices[:len(v6_act)]
        idx = [i for i, x in enumerate(v6_act) if x == a and i < len(prices6)]
        ax2.scatter(idx, [prices6[i] for i in idx], marker=mk, color=col, s=sz//2,
                    label=f"v6 {lbl}", zorder=4, alpha=0.4, edgecolors="black", linewidths=0.3)
    ax2.set_ylabel("ETH 가격")
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(alpha=0.3)
    ax2.set_title("매매 포인트 (진한색=v4, 연한색=v6)")

    # 3. 성과 비교 표
    ax3 = axes[2]
    ax3.axis("off")
    rows = [
        ["지표", "v4 (LEV×5)", "v6 (LEV×3)"],
        ["총 수익률",   f"{s4['return']:+.2f}%",  f"{s6['return']:+.2f}%"],
        ["Buy & Hold", f"{s4['buy_hold']:+.2f}%", f"{s6['buy_hold']:+.2f}%"],
        ["최대낙폭(MDD)", f"{s4['mdd']:.2f}%",    f"{s6['mdd']:.2f}%"],
        ["총 거래",     f"{s4['trades']}회",       f"{s6['trades']}회"],
        ["승률",        f"{s4['win_rate']:.1%}",   f"{s6['win_rate']:.1%}"],
    ]
    tbl = ax3.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    # 헤더 색상
    for j in range(3):
        tbl[0, j].set_facecolor("#1a237e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # v6 열 강조
    for i in range(1, len(rows)):
        tbl[i, 2].set_facecolor("#e8f5e9")

    plt.tight_layout()
    out = Path(__file__).parent / "backtest_compare_v4_v6.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장: {out}")
    plt.close()

def print_stats(label, s, actions):
    dist = Counter(actions)
    total = len(actions)
    names = {0:"관망", 1:"롱", 2:"숏", 3:"청산"}
    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  최종 잔고:   ${s['final']:,.2f}")
    print(f"  총 수익률:   {s['return']:+.2f}%")
    print(f"  Buy & Hold:  {s['buy_hold']:+.2f}%")
    print(f"  최대낙폭:    {s['mdd']:.2f}%")
    print(f"  총 거래:     {s['trades']}회 | 승률: {s['win_rate']:.1%}")
    print(f"  행동 분포:   " + " / ".join(f"{names[k]}:{v}({v/total:.0%})" for k,v in sorted(dist.items())))

if __name__ == "__main__":
    df = load_data(INTERVAL)

    # v4 메타 기준 test 구간 사용
    meta_path = Path(__file__).parent / "models/v4/meta_30m.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        split_idx = meta["train_candles"]
    else:
        split_idx = int(len(df) * 0.8)

    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"테스트 구간: {test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()}  ({len(test_df)}개 캔들)\n")

    bh_ret = (test_df["close"].iloc[-1] - test_df["close"].iloc[20]) / test_df["close"].iloc[20] * 100

    print("▶ v4 백테스트 실행 중...")
    v4_bal, v4_prices, v4_act, v4_info = run_v4(test_df)
    s4 = calc_stats(v4_bal, v4_info, bh_ret)

    print("▶ v6 백테스트 실행 중...")
    v6_bal, v6_prices, v6_act, v6_info = run_v6(test_df)
    s6 = calc_stats(v6_bal, v6_info, bh_ret)

    print_stats("v4  (레버리지 ×5, 기존)", s4, v4_act)
    print_stats("v6  (레버리지 ×3, VecNorm)", s6, v6_act)

    plot_comparison(test_df, v4_bal, v4_prices, v4_act,
                              v6_bal, v6_prices, v6_act, s4, s6)
    print("\n✅ 비교 완료")

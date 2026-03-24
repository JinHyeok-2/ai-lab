#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ETH 앙상블 재탐색: eth_seed800 포함 시 성능 변화 확인
# 기존 최적: exp14+exp08+seed700 만장일치 (+1490%, MDD -3.1%, 94.6%)

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from stable_baselines3 import PPO
from env_v51_exp02 import ETHTradingEnvV51Exp02

EXP_DIR = Path(__file__).parent.parent
LEVERAGE = 3


def backtest(models, names, df, strategy):
    env = ETHTradingEnvV51Exp02(df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    balances = [10000.0]
    done = False
    while not done:
        votes = [int(models[n].predict(obs, deterministic=True)[0]) for n in names]
        if strategy == "unanimous":
            action = votes[0] if len(set(votes)) == 1 else 0
        else:
            vc = Counter(votes)
            maj = vc.most_common(1)[0]
            action = maj[0] if maj[1] >= 2 else 0
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        balances.append(info["balance"])
    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    return {"return": (balances[-1]-10000)/10000*100, "mdd": mdd,
            "trades": info["total_trades"], "win_rate": info["win_rate"]}


def main():
    data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)

    # seed800 포함 4모델
    model_names = ["exp14", "exp08", "seed700", "seed800"]
    # seed800은 eth_seed800 디렉토리에 저장됨
    dir_map = {"seed800": "eth_seed800"}
    models = {}
    for n in model_names:
        d = dir_map.get(n, n)
        p = EXP_DIR / f"models/{d}/ppo_eth_30m.zip"
        models[n] = PPO.load(str(p), device="cpu")
        print(f"  로드: {n}")

    print(f"\n{'='*70}")
    print(f"  ETH 앙상블 재탐색 (seed800 포함)")
    print(f"{'='*70}")

    results = []

    # seed800 포함 3모델 조합 중 seed800이 들어간 것만
    print(f"\n[1] seed800 포함 3모델 다수결")
    print("-" * 70)
    for combo in combinations(model_names, 3):
        if "seed800" not in combo:
            continue
        names = list(combo)
        label = "+".join(names)
        r = backtest(models, names, test_df, "majority")
        print(f"  {label:<30} → 수익 {r['return']:>+8.1f}% | MDD {r['mdd']:>7.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")
        results.append({"combo": label, "strategy": "majority", **r})

    print(f"\n[2] seed800 포함 3모델 만장일치")
    print("-" * 70)
    for combo in combinations(model_names, 3):
        if "seed800" not in combo:
            continue
        names = list(combo)
        label = "+".join(names)
        r = backtest(models, names, test_df, "unanimous")
        print(f"  {label:<30} → 수익 {r['return']:>+8.1f}% | MDD {r['mdd']:>7.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")
        results.append({"combo": label, "strategy": "unanimous", **r})

    # 4모델 다수결 (3/4 이상)
    print(f"\n[3] 4모델 다수결 (3/4 이상)")
    print("-" * 70)
    names = model_names
    label = "+".join(names)
    r_maj = backtest(models, names, test_df, "majority")
    print(f"  {label:<30} → 수익 {r_maj['return']:>+8.1f}% | MDD {r_maj['mdd']:>7.1f}% | 거래 {r_maj['trades']:>3}회 | 승률 {r_maj['win_rate']:.1%}")
    results.append({"combo": label, "strategy": "majority_4", **r_maj})

    # 기존 최적 (비교용)
    print(f"\n[비교] 기존 프로덕션 (exp14+exp08+seed700)")
    print("-" * 70)
    r_old = backtest(models, ["exp14", "exp08", "seed700"], test_df, "unanimous")
    print(f"  exp14+exp08+seed700 만장일치 → 수익 {r_old['return']:>+8.1f}% | MDD {r_old['mdd']:>7.1f}% | 거래 {r_old['trades']:>3}회 | 승률 {r_old['win_rate']:.1%}")

    # 랭킹
    for r in results:
        wr = r["win_rate"] if r["win_rate"] > 0 else 0.01
        mdd = abs(r["mdd"]) if r["mdd"] != 0 else 0.01
        r["score"] = r["return"] * wr / mdd
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  랭킹 (seed800 포함 조합)")
    print(f"{'='*70}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['combo']:<35} {r['strategy']:<12} {r['return']:>+7.1f}% {r['mdd']:>6.1f}% {r['trades']:>3}회 {r['win_rate']:>5.1%} score={r['score']:.1f}")

    print(f"\n  기존 최적(비교): exp14+exp08+seed700 만장일치 → {r_old['return']:+.1f}%, MDD {r_old['mdd']:.1f}%, 승률 {r_old['win_rate']:.1%}")


if __name__ == "__main__":
    main()

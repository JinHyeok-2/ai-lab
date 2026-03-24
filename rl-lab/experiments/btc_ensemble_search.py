#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# BTC 앙상블 조합 탐색 — 4모델(btc_exp14, seed100, seed200, seed300)
# 3모델 다수결/만장일치 + 4모델 다수결/가중투표

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

# 개별 모델 백테스트 수익률 (가중투표용)
SOLO_RETURNS = {
    "btc_exp14": 378.3,
    "btc_seed100": 50.2,
    "btc_seed200": 67.1,
    "btc_seed300": 42.5,
}


def backtest_ensemble(models, model_names, df, strategy="majority"):
    """앙상블 백테스트"""
    env = ETHTradingEnvV51Exp02(df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    balances = [10000.0]
    done = False
    n = len(model_names)

    while not done:
        votes = []
        for name in model_names:
            a, _ = models[name].predict(obs, deterministic=True)
            votes.append(int(a))

        if strategy == "unanimous":
            action = votes[0] if len(set(votes)) == 1 else 0
        elif strategy == "weighted":
            weights = {name: SOLO_RETURNS[name] for name in model_names}
            score = {}
            for name, v in zip(model_names, votes):
                score[v] = score.get(v, 0) + weights[name]
            action = max(score, key=score.get)
        else:  # majority
            vc = Counter(votes)
            maj = vc.most_common(1)[0]
            if n == 3:
                action = maj[0] if maj[1] >= 2 else 0
            else:  # 4모델
                action = maj[0] if maj[1] >= 3 else 0  # 3/4 이상

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        balances.append(info["balance"])

    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (balances[-1] - 10000) / 10000 * 100
    return {"return": ret, "mdd": mdd,
            "trades": info["total_trades"], "win_rate": info["win_rate"]}


def main():
    # 데이터
    data_path = PROJECT_ROOT / "rl" / "btc_30m.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)
    print(f"BTC 테스트: {len(test_df)}캔들")

    # 모델 로드
    model_names = ["btc_exp14", "btc_seed100", "btc_seed200", "btc_seed300"]
    models = {}
    for name in model_names:
        p = EXP_DIR / f"models/{name}/ppo_btc_30m.zip"
        models[name] = PPO.load(str(p), device="cpu")
        print(f"  로드: {name}")

    print(f"\n{'='*70}")
    print(f"  BTC 앙상블 조합 탐색")
    print(f"  후보: {model_names}")
    print(f"{'='*70}")

    results = []

    # [1] 3모델 다수결 (C(4,3) = 4)
    print(f"\n\n[1] 3모델 다수결 조합")
    print("-" * 70)
    for combo in combinations(model_names, 3):
        names = list(combo)
        label = "+".join(n.replace("btc_", "") for n in names)
        r = backtest_ensemble(models, names, test_df, "majority")
        print(f"  {label:<30} → 수익 {r['return']:>+8.1f}% | MDD {r['mdd']:>7.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")
        results.append({"combo": label, "strategy": "majority", **r})

    # [2] 3모델 만장일치 (C(4,3) = 4)
    print(f"\n\n[2] 3모델 만장일치 조합")
    print("-" * 70)
    for combo in combinations(model_names, 3):
        names = list(combo)
        label = "+".join(n.replace("btc_", "") for n in names)
        r = backtest_ensemble(models, names, test_df, "unanimous")
        print(f"  {label:<30} → 수익 {r['return']:>+8.1f}% | MDD {r['mdd']:>7.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")
        results.append({"combo": label, "strategy": "unanimous", **r})

    # [3] 4모델 다수결 (3/4 이상)
    print(f"\n\n[3] 4모델 다수결 (3/4 이상)")
    print("-" * 70)
    label = "+".join(n.replace("btc_", "") for n in model_names)
    r = backtest_ensemble(models, model_names, test_df, "majority")
    print(f"  {label:<30} → 수익 {r['return']:>+8.1f}% | MDD {r['mdd']:>7.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")
    results.append({"combo": label, "strategy": "majority_4", **r})

    # [4] 4모델 가중투표
    print(f"\n\n[4] 4모델 가중투표 (수익률 비례)")
    print("-" * 70)
    r = backtest_ensemble(models, model_names, test_df, "weighted")
    print(f"  {label:<30} → 수익 {r['return']:>+8.1f}% | MDD {r['mdd']:>7.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")
    results.append({"combo": label, "strategy": "weighted", **r})

    # [5] 3모델 가중투표
    print(f"\n\n[5] 3모델 가중투표")
    print("-" * 70)
    for combo in combinations(model_names, 3):
        names = list(combo)
        label = "+".join(n.replace("btc_", "") for n in names)
        r = backtest_ensemble(models, names, test_df, "weighted")
        print(f"  {label:<30} → 수익 {r['return']:>+8.1f}% | MDD {r['mdd']:>7.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")
        results.append({"combo": label, "strategy": "weighted_3", **r})

    # 최종 랭킹
    for r in results:
        wr = r["win_rate"] if r["win_rate"] > 0 else 0.01
        mdd = abs(r["mdd"]) if r["mdd"] != 0 else 0.01
        r["score"] = r["return"] * wr / mdd

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n\n{'='*70}")
    print(f"  최종 랭킹 (수익률 x 승률 / |MDD| 스코어)")
    print(f"{'='*70}")
    print(f"\n  {'순위'} {'조합':<35} {'전략':<12} {'수익률':>8} {'MDD':>7} {'거래':>4} {'승률':>6} {'스코어':>8}")
    print("-" * 90)
    for i, r in enumerate(results):
        print(f"  {i+1:>2}. {r['combo']:<35} {r['strategy']:<12} {r['return']:>+7.1f}% {r['mdd']:>6.1f}% {r['trades']:>4}회 {r['win_rate']:>5.1%} {r['score']:>8.1f}")

    best = results[0]
    print(f"\n  최적 조합: {best['combo']} ({best['strategy']})")
    print(f"  수익: {best['return']:+.1f}% | MDD: {best['mdd']:.1f}% | 거래: {best['trades']}회 | 승률: {best['win_rate']:.1%}")


if __name__ == "__main__":
    main()

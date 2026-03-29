#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ETH/BTC 앙상블 조합 탐색 — 기존 3모델 + 신규 3시드 = 6모델
# 숏 마스킹 적용, 다수결/만장일치 비교
# conda run -n ai-lab --no-capture-output python eval_eth_btc_ensemble.py

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np, pandas as pd, json
from itertools import combinations
from collections import Counter
from stable_baselines3 import PPO
from env_v51_exp02 import ETHTradingEnvV51Exp02

MODELS_DIR = Path(__file__).parent.parent / "models"

WF_WINDOWS = [
    ("2025-09-01", "2025-11-01"),
    ("2025-11-01", "2026-01-01"),
    ("2026-01-01", "2026-03-01"),
    ("2026-03-01", "2026-03-29"),
]


def load_all_models():
    """ETH/BTC 전체 모델 로드"""
    eth_models, btc_models = {}, {}

    # ETH 기존
    for name, path in [
        ("exp14", "exp14/ppo_eth_30m.zip"),
        ("seed700", "seed700/ppo_eth_30m.zip"),
        ("eth_seed800", "eth_seed800/ppo_eth_30m.zip"),
    ]:
        p = MODELS_DIR / path
        if p.exists():
            eth_models[name] = PPO.load(str(p), device="cuda")
            print(f"  ETH 로드: {name}")

    # ETH 신규
    for seed in [400, 500, 600]:
        for sub in [f"eth_v2_seed{seed}"]:
            p = MODELS_DIR / sub / "ppo_eth_30m.zip"
            if p.exists():
                eth_models[f"v2s{seed}"] = PPO.load(str(p), device="cuda")
                print(f"  ETH 로드: v2s{seed}")

    # BTC 기존
    for name, path in [
        ("exp14", "btc_exp14/ppo_btc_30m.zip"),
        ("seed100", "btc_seed100/ppo_btc_30m.zip"),
        ("seed200", "btc_seed200/ppo_btc_30m.zip"),
    ]:
        p = MODELS_DIR / path
        if p.exists():
            btc_models[name] = PPO.load(str(p), device="cuda")
            print(f"  BTC 로드: {name}")

    # BTC 신규
    for seed in [400, 500, 600]:
        p = MODELS_DIR / f"btc_v2_seed{seed}" / "ppo_btc_30m.zip"
        if p.exists():
            btc_models[f"v2s{seed}"] = PPO.load(str(p), device="cuda")
            print(f"  BTC 로드: v2s{seed}")

    return eth_models, btc_models


def bt_ensemble(models_dict, keys, test_df, leverage=3, method="majority", mask_short=True):
    """앙상블 백테스트 (숏 마스킹 포함)"""
    env = ETHTradingEnvV51Exp02(
        test_df, initial_balance=10000.0, leverage=leverage, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    bals = [10000.0]
    while not done:
        votes = []
        for k in keys:
            a, _ = models_dict[k].predict(obs, deterministic=True)
            v = int(a)
            if mask_short and v == 2:
                v = 0  # 숏 → 관망
            votes.append(v)

        if method == "unanimous":
            if len(set(votes)) == 1:
                action = votes[0]
            else:
                cnt = Counter(votes)
                maj = cnt.most_common(1)[0]
                action = maj[0] if maj[1] >= 2 else 0
        else:
            cnt = Counter(votes)
            maj = cnt.most_common(1)[0]
            action = maj[0] if maj[1] >= len(keys) // 2 + 1 else 0

        obs, _, term, trunc, info = env.step(action); done = term or trunc
        bals.append(info["balance"])

    arr = np.array(bals); peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (bals[-1] - 10000) / 10000 * 100
    return ret, mdd, info["total_trades"], info["win_rate"] * 100


def bt_single(model, test_df, leverage=3, mask_short=True):
    env = ETHTradingEnvV51Exp02(
        test_df, initial_balance=10000.0, leverage=leverage, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    bals = [10000.0]
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        action = int(a)
        if mask_short and action == 2:
            action = 0
        obs, _, term, trunc, info = env.step(action); done = term or trunc
        bals.append(info["balance"])
    arr = np.array(bals); peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (bals[-1] - 10000) / 10000 * 100
    return ret, mdd, info["total_trades"], info["win_rate"] * 100


def evaluate_asset(asset, models_dict, csv_path, leverage=3):
    print(f"\n{'='*60}")
    print(f"{asset} 앙상블 조합 탐색 ({len(models_dict)}모델, 숏 마스킹)")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)
    keys = sorted(models_dict.keys())

    # 1. 개별 성능
    print(f"\n--- 개별 모델 (숏 마스킹) ---")
    print(f"{'모델':<12} {'수익률':>10} {'MDD':>8} {'승률':>7}")
    for k in keys:
        ret, mdd, trades, wr = bt_single(models_dict[k], test_df, leverage)
        print(f"{k:<12} {ret:>+9.1f}% {mdd:>7.1f}% {wr:>6.1f}%")

    # 2. 3모델 조합
    print(f"\n--- 3모델 다수결 (상위 10개) ---")
    results = []
    for combo in combinations(keys, 3):
        ret, mdd, trades, wr = bt_ensemble(models_dict, list(combo), test_df, leverage)
        results.append({"combo": list(combo), "ret": round(ret, 1), "mdd": round(mdd, 1),
                        "trades": trades, "wr": round(wr, 1)})

    results.sort(key=lambda x: x["ret"], reverse=True)
    for i, r in enumerate(results[:10]):
        print(f"  {'+'.join(r['combo']):<30} {r['ret']:>+7.1f}% MDD{r['mdd']:>7.1f}% 승률{r['wr']:>5.1f}%")

    # 3. 상위 3개 WF 검증
    print(f"\n--- 상위 3개 WF 검증 ---")
    for i, cr in enumerate(results[:3]):
        combo = cr["combo"]
        wf_ok, wf_tot = 0, 0
        for s, e in WF_WINDOWS:
            wdf = df[(df["time"] > pd.Timestamp(s)) & (df["time"] <= pd.Timestamp(e))].reset_index(drop=True)
            if len(wdf) < 50: continue
            ret, _, _, _ = bt_ensemble(models_dict, combo, wdf, leverage)
            wf_tot += 1
            if ret > 0: wf_ok += 1
        cr["wf"] = f"{wf_ok}/{wf_tot}"
        print(f"  #{i+1} {'+'.join(combo):<30} WF {wf_ok}/{wf_tot}")

    # 4. 현재 프로덕션 앙상블 비교
    print(f"\n--- 현재 프로덕션 vs 최적 비교 ---")
    return results


def main():
    print("=== ETH/BTC 앙상블 조합 탐색 ===\n")
    eth_models, btc_models = load_all_models()
    print(f"\nETH: {len(eth_models)}모델, BTC: {len(btc_models)}모델\n")

    eth_csv = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    btc_csv = PROJECT_ROOT / "rl" / "btc_30m.csv"

    eth_results = evaluate_asset("ETH", eth_models, eth_csv, leverage=3)
    btc_results = evaluate_asset("BTC", btc_models, btc_csv, leverage=3)

    # 최종 요약
    print(f"\n{'='*60}")
    print("최종 추천")
    print(f"{'='*60}")
    if eth_results:
        r = eth_results[0]
        print(f"  ETH: {'+'.join(r['combo'])} → {r['ret']:+.1f}% MDD{r['mdd']:.1f}% WF {r.get('wf','-')}")
    if btc_results:
        r = btc_results[0]
        print(f"  BTC: {'+'.join(r['combo'])} → {r['ret']:+.1f}% MDD{r['mdd']:.1f}% WF {r.get('wf','-')}")

    # 저장
    out = MODELS_DIR / "eth_btc_ensemble_results.json"
    out.write_text(json.dumps({
        "eth_top5": [r for r in eth_results[:5]],
        "btc_top5": [r for r in btc_results[:5]],
    }, indent=2, ensure_ascii=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT v2 앙상블 조합 탐색 — 초경량 (대표 5코인 + 상위 WF만)
# conda run -n ai-lab --no-capture-output python eval_alt_v2_quick.py

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np, pandas as pd, json
from itertools import combinations
from collections import Counter
from stable_baselines3 import PPO
from env_alt_universal import AltUniversalTradingEnv

MODELS_DIR = Path(__file__).parent.parent / "models"
ALT_DATA_DIR = PROJECT_ROOT / "rl" / "alt_30m"

# 대표 코인 (실거래 활발 + 다양한 특성)
EVAL_COINS = ["TAO", "SOL", "AVAX", "LINK", "SUI", "ONT", "FIL", "DOGE", "XRP", "APT"]

WF_WINDOWS = [
    ("2025-09-01", "2025-11-01"),
    ("2025-11-01", "2026-01-01"),
    ("2026-01-01", "2026-03-01"),
    ("2026-03-01", "2026-03-27"),
]


def load_models():
    models = {}
    for sub in ["alt_universal_exp01"]:
        path = MODELS_DIR / sub / "ppo_alt.zip"
        best = MODELS_DIR / sub / "best" / "ppo_alt.zip"
        p = best if best.exists() else path
        if p.exists():
            models["exp01"] = PPO.load(str(p), device="cuda")
            print(f"  exp01 로드")
    for d in sorted(MODELS_DIR.glob("alt_v2_seed*")):
        name = d.name.replace("alt_v2_", "")
        best = d / "best" / "ppo_alt.zip"
        final = d / "ppo_alt.zip"
        p = best if best.exists() else final
        if p.exists():
            models[name] = PPO.load(str(p), device="cuda")
            print(f"  {name} 로드")
    return models


def bt_ensemble(models_dict, keys, df, method="majority"):
    env = AltUniversalTradingEnv(
        [df], initial_balance=10000.0, leverage=2, window_size=20,
        min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    bals = [10000.0]
    while not done:
        votes = [int(models_dict[k].predict(obs, deterministic=True)[0]) for k in keys]
        cnt = Counter(votes)
        maj = cnt.most_common(1)[0]
        action = maj[0] if maj[1] >= len(keys) // 2 + 1 else 0
        obs, _, term, trunc, info = env.step(action); done = term or trunc
        bals.append(info["balance"])
    arr = np.array(bals); peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (bals[-1] - 10000) / 10000 * 100
    return ret, mdd, info["total_trades"], info["win_rate"] * 100


def bt_single(model, df):
    env = AltUniversalTradingEnv(
        [df], initial_balance=10000.0, leverage=2, window_size=20,
        min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(a)); done = term or trunc
    return (info["balance"] - 10000) / 10000 * 100


def main():
    print("=== ALT v2 앙상블 조합 탐색 (대표 10코인) ===\n")
    models = load_models()
    keys = sorted(models.keys())
    print(f"\n모델 {len(keys)}개: {keys}\n")

    # 데이터 로드 (대표 코인만)
    data = {}
    full_data = {}
    for name in EVAL_COINS:
        p = ALT_DATA_DIR / f"{name.lower()}_30m.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["time"])
            full_data[name] = df
            test = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)
            if len(test) >= 100:
                data[name] = test
    print(f"평가 코인: {list(data.keys())} ({len(data)}종)\n")

    # 1. 개별 성능
    print(f"{'='*55}")
    print(f"{'모델':<12} {'평균수익':>10} {'수익코인':>10}")
    print(f"{'-'*35}")
    indiv = {}
    for k in keys:
        rets = [bt_single(models[k], df) for df in data.values()]
        avg = np.mean(rets)
        wins = sum(1 for r in rets if r > 0)
        indiv[k] = {"avg": round(avg,1), "wins": wins}
        print(f"{k:<12} {avg:>+9.1f}% {wins:>6}/{len(rets)}")

    # 2. 3모델 조합
    print(f"\n{'='*55}")
    print("3모델 다수결 앙상블")
    print(f"{'='*55}")
    results = []
    for combo in combinations(keys, 3):
        rets, mdds = [], []
        for df in data.values():
            r, m, _, _ = bt_ensemble(models, list(combo), df)
            rets.append(r); mdds.append(m)
        avg_r = np.mean(rets); avg_m = np.mean(mdds)
        wins = sum(1 for r in rets if r > 0)
        results.append({"combo": list(combo), "ret": round(avg_r,1), "mdd": round(avg_m,1), "wins": wins})
        print(f"  {'+'.join(combo):<35} {avg_r:>+7.1f}% MDD{avg_m:>6.1f}% {wins}/{len(rets)}")
    results.sort(key=lambda x: x["ret"], reverse=True)

    # 3. 상위 3개 WF
    print(f"\n{'='*55}")
    print("상위 3개 WF 검증 (10코인×4구간)")
    print(f"{'='*55}")
    for i, cr in enumerate(results[:3]):
        combo = cr["combo"]
        wf_ok, wf_tot = 0, 0
        for name, fdf in full_data.items():
            for s, e in WF_WINDOWS:
                wdf = fdf[(fdf["time"]>pd.Timestamp(s))&(fdf["time"]<=pd.Timestamp(e))].reset_index(drop=True)
                if len(wdf) < 50: continue
                r, _, _, _ = bt_ensemble(models, combo, wdf)
                wf_tot += 1
                if r > 0: wf_ok += 1
        cr["wf"] = f"{wf_ok}/{wf_tot}"
        print(f"  #{i+1} {'+'.join(combo):<35} WF {wf_ok}/{wf_tot}")

    # 4. 5시드 전체
    print(f"\n{'='*55}")
    print("5시드 전체 앙상블")
    print(f"{'='*55}")
    v2 = [k for k in keys if k.startswith("seed")]
    if len(v2) >= 5:
        rets, mdds = [], []
        for df in data.values():
            r, m, _, _ = bt_ensemble(models, v2, df)
            rets.append(r); mdds.append(m)
        print(f"  5시드: {np.mean(rets):>+.1f}% MDD{np.mean(mdds):.1f}% {sum(1 for r in rets if r>0)}/{len(rets)}")

    # 6모델 전체
    if len(keys) >= 6:
        rets, mdds = [], []
        for df in data.values():
            r, m, _, _ = bt_ensemble(models, keys, df)
            rets.append(r); mdds.append(m)
        print(f"  6모델: {np.mean(rets):>+.1f}% MDD{np.mean(mdds):.1f}% {sum(1 for r in rets if r>0)}/{len(rets)}")

    # 최종 순위
    print(f"\n{'='*55}")
    print("최종 순위")
    print(f"{'='*55}")
    for i, cr in enumerate(results[:5]):
        wf = cr.get("wf", "-")
        print(f"  #{i+1}: {'+'.join(cr['combo']):<35} {cr['ret']:>+7.1f}% MDD{cr['mdd']:>6.1f}% WF {wf}")

    # 저장
    out = MODELS_DIR / "alt_v2_ensemble_results.json"
    out.write_text(json.dumps({"individual": indiv, "top_combos": results[:10]}, indent=2, ensure_ascii=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

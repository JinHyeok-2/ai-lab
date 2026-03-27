#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT v2 앙상블 조합 탐색 — 경량 버전 (WF는 상위 3개만)
# conda run -n ai-lab python eval_alt_v2_fast.py

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

WF_WINDOWS = [
    ("2025-09-01", "2025-11-01"),
    ("2025-11-01", "2026-01-01"),
    ("2026-01-01", "2026-03-01"),
    ("2026-03-01", "2026-03-27"),
]


def load_models():
    models = {}
    # 기존 exp01
    for sub in ["alt_universal_exp01"]:
        best = MODELS_DIR / sub / "best" / "ppo_alt.zip"
        final = MODELS_DIR / sub / "ppo_alt.zip"
        path = best if best.exists() else final
        if path.exists():
            models["exp01"] = PPO.load(str(path), device="cuda")
            print(f"  로드: exp01")
    # v2 시드 모델
    for d in sorted(MODELS_DIR.glob("alt_v2_seed*")):
        seed_name = d.name.replace("alt_v2_", "")
        best = d / "best" / "ppo_alt.zip"
        final = d / "ppo_alt.zip"
        path = best if best.exists() else final
        if path.exists():
            models[seed_name] = PPO.load(str(path), device="cuda")
            print(f"  로드: {seed_name}")
    return models


def backtest_single(model, test_df):
    """단일 모델 백테스트"""
    env = AltUniversalTradingEnv(
        [test_df], initial_balance=10000.0, leverage=2, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(a)); done = term or trunc
    ret = (info["balance"] - 10000) / 10000 * 100
    arr_b = info["balance"]
    return ret, info["total_trades"], info["win_rate"] * 100


def backtest_ensemble(models_dict, model_keys, test_df, method="majority"):
    """앙상블 백테스트"""
    env = AltUniversalTradingEnv(
        [test_df], initial_balance=10000.0, leverage=2, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    balances = [10000.0]
    while not done:
        votes = [int(models_dict[k].predict(obs, deterministic=True)[0]) for k in model_keys]
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
            action = maj[0] if maj[1] >= len(model_keys) // 2 + 1 else 0
        obs, _, term, trunc, info = env.step(action); done = term or trunc
        balances.append(info["balance"])
    arr = np.array(balances); peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (balances[-1] - 10000) / 10000 * 100
    return ret, mdd, info["total_trades"], info["win_rate"] * 100


def main():
    print("=== ALT v2 앙상블 조합 탐색 (경량) ===\n")

    models_dict = load_models()
    model_keys = sorted(models_dict.keys())
    print(f"\n모델: {model_keys} ({len(model_keys)}개)\n")

    # 테스트 데이터 로드
    all_data = {}
    for csv_path in sorted(ALT_DATA_DIR.glob("*_30m.csv")):
        name = csv_path.stem.replace("_30m", "").upper()
        df = pd.read_csv(csv_path, parse_dates=["time"])
        test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)
        if len(test_df) >= 100:
            all_data[name] = test_df
    print(f"테스트 코인: {len(all_data)}종\n")

    # === 1. 개별 모델 성능 ===
    print(f"{'='*60}")
    print("1. 개별 모델 성능")
    print(f"{'='*60}")
    print(f"{'모델':<12} {'평균수익':>10} {'수익코인':>10}")
    print(f"{'-'*35}")

    individual = {}
    for k in model_keys:
        rets = []
        for name, df in all_data.items():
            ret, _, _ = backtest_single(models_dict[k], df)
            rets.append(ret)
        avg = np.mean(rets)
        wins = sum(1 for r in rets if r > 0)
        individual[k] = {"avg": round(avg, 1), "wins": wins, "total": len(rets)}
        print(f"{k:<12} {avg:>+9.1f}% {wins:>6}/{len(rets)}")
    print()

    # === 2. 3모델 조합 (다수결) ===
    print(f"{'='*60}")
    print("2. 앙상블 조합 (3모델 다수결)")
    print(f"{'='*60}")

    combo_results = []
    for combo in combinations(model_keys, 3):
        rets, mdds = [], []
        for name, df in all_data.items():
            ret, mdd, _, _ = backtest_ensemble(models_dict, list(combo), df)
            rets.append(ret); mdds.append(mdd)
        avg_ret = np.mean(rets)
        avg_mdd = np.mean(mdds)
        wins = sum(1 for r in rets if r > 0)
        combo_results.append({
            "combo": list(combo), "avg_return": round(avg_ret, 1),
            "avg_mdd": round(avg_mdd, 1), "wins": wins, "total": len(rets)
        })
        combo_str = "+".join(combo)
        print(f"  {combo_str:<35} {avg_ret:>+7.1f}% MDD {avg_mdd:>6.1f}% 코인 {wins}/{len(rets)}")

    combo_results.sort(key=lambda x: x["avg_return"], reverse=True)

    # === 3. 상위 3개 WF 검증 ===
    print(f"\n{'='*60}")
    print("3. 상위 3개 조합 WF 검증")
    print(f"{'='*60}")

    # WF용 전체 데이터 로드
    full_data = {}
    for csv_path in sorted(ALT_DATA_DIR.glob("*_30m.csv")):
        name = csv_path.stem.replace("_30m", "").upper()
        full_data[name] = pd.read_csv(csv_path, parse_dates=["time"])

    for rank, cr in enumerate(combo_results[:3]):
        combo = cr["combo"]
        combo_str = "+".join(combo)
        print(f"\n--- #{rank+1}: {combo_str} (avg {cr['avg_return']:+.1f}%) ---")

        wf_total, wf_pass = 0, 0
        for name in list(all_data.keys())[:10]:  # 상위 10코인만 WF
            df = full_data.get(name)
            if df is None: continue
            coin_wins = 0
            for start, end in WF_WINDOWS:
                wf_df = df[(df["time"] > pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(end))].reset_index(drop=True)
                if len(wf_df) < 50: continue
                ret, _, _, _ = backtest_ensemble(models_dict, combo, wf_df)
                wf_total += 1
                if ret > 0:
                    wf_pass += 1
                    coin_wins += 1
            print(f"  {name:<8} WF {coin_wins}/4")

        cr["wf_pass"] = f"{wf_pass}/{wf_total}"
        print(f"  전체 WF: {wf_pass}/{wf_total}")

    # === 4. 5모델 전체 앙상블 ===
    if len(model_keys) >= 5:
        print(f"\n{'='*60}")
        print("4. 전체 모델 앙상블 (다수결)")
        print(f"{'='*60}")
        # v2 시드만 (exp01 제외)
        v2_keys = [k for k in model_keys if k.startswith("seed")]
        rets, mdds = [], []
        for name, df in all_data.items():
            ret, mdd, _, _ = backtest_ensemble(models_dict, v2_keys, df)
            rets.append(ret); mdds.append(mdd)
        avg_ret = np.mean(rets)
        avg_mdd = np.mean(mdds)
        wins = sum(1 for r in rets if r > 0)
        print(f"  5시드 다수결: {avg_ret:>+.1f}% MDD {avg_mdd:.1f}% 코인 {wins}/{len(rets)}")

        # 6모델 전체
        if len(model_keys) == 6:
            rets2, mdds2 = [], []
            for name, df in all_data.items():
                ret, mdd, _, _ = backtest_ensemble(models_dict, model_keys, df)
                rets2.append(ret); mdds2.append(mdd)
            avg2 = np.mean(rets2); mdd2 = np.mean(mdds2)
            wins2 = sum(1 for r in rets2 if r > 0)
            print(f"  6모델 다수결: {avg2:>+.1f}% MDD {mdd2:.1f}% 코인 {wins2}/{len(rets2)}")

    # === 최종 요약 ===
    print(f"\n{'='*60}")
    print("최종 순위 (상위 5)")
    print(f"{'='*60}")
    for i, cr in enumerate(combo_results[:5]):
        wf = cr.get("wf_pass", "-")
        print(f"  #{i+1}: {'+'.join(cr['combo']):<35} {cr['avg_return']:>+7.1f}% MDD {cr['avg_mdd']:>6.1f}% 코인 {cr['wins']}/{cr['total']} WF {wf}")

    # 결과 저장
    result_file = MODELS_DIR / "alt_v2_ensemble_results.json"
    result_file.write_text(json.dumps({
        "individual": individual,
        "top_combos": combo_results[:10],
    }, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {result_file}")


if __name__ == "__main__":
    main()

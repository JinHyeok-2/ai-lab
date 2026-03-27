#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT v2 앙상블 조합 탐색 — 시드별 모델 로드 → 투표 조합 비교
# conda run -n ai-lab python eval_alt_v2_ensemble.py

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

# WF 검증 구간
WF_WINDOWS = [
    ("2025-09-01", "2025-11-01"),
    ("2025-11-01", "2026-01-01"),
    ("2026-01-01", "2026-03-01"),
    ("2026-03-01", "2026-03-26"),
]


def load_models():
    """모든 alt_v2 시드 모델 + 기존 exp01 로드"""
    models = {}

    # 기존 exp01
    for sub in ["alt_universal_exp01"]:
        best = MODELS_DIR / sub / "best" / "ppo_alt.zip"
        final = MODELS_DIR / sub / "ppo_alt.zip"
        path = best if best.exists() else final
        if path.exists():
            models["exp01"] = PPO.load(str(path), device="cuda")
            print(f"  로드: exp01 ({path.name})")

    # v2 시드 모델
    for d in sorted(MODELS_DIR.glob("alt_v2_seed*")):
        seed_name = d.name.replace("alt_v2_", "")  # seed42, seed100, ...
        best = d / "best" / "ppo_alt.zip"
        final = d / "ppo_alt.zip"
        path = best if best.exists() else final
        if path.exists():
            models[seed_name] = PPO.load(str(path), device="cuda")
            print(f"  로드: {seed_name} ({path.name})")

    return models


def load_test_data():
    """전체 코인 테스트 데이터 로드"""
    all_data = {}
    for csv_path in sorted(ALT_DATA_DIR.glob("*_30m.csv")):
        name = csv_path.stem.replace("_30m", "").upper()
        df = pd.read_csv(csv_path, parse_dates=["time"])
        test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)
        if len(test_df) >= 100:
            all_data[name] = test_df
    return all_data


def backtest_ensemble(models_dict, model_keys, test_df, method="majority"):
    """앙상블 백테스트 — 투표 방식으로 행동 결정"""
    env = AltUniversalTradingEnv(
        [test_df], initial_balance=10000.0, leverage=2, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    balances = [10000.0]; actions = []

    while not done:
        # 각 모델 투표
        votes = []
        for k in model_keys:
            a, _ = models_dict[k].predict(obs, deterministic=True)
            votes.append(int(a))

        # 투표 결정
        if method == "unanimous":
            if len(set(votes)) == 1:
                action = votes[0]
            else:
                # 폴백: 다수결
                cnt = Counter(votes)
                maj = cnt.most_common(1)[0]
                action = maj[0] if maj[1] >= 2 else 0
        else:  # majority
            cnt = Counter(votes)
            maj = cnt.most_common(1)[0]
            action = maj[0] if maj[1] >= len(model_keys) // 2 + 1 else 0

        obs, _, term, trunc, info = env.step(action); done = term or trunc
        balances.append(info["balance"]); actions.append(action)

    arr = np.array(balances); peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (balances[-1] - 10000) / 10000 * 100
    return {
        "return_pct": ret, "mdd_pct": mdd,
        "trades": info["total_trades"],
        "win_rate": info["win_rate"] * 100,
    }


def evaluate_combination(models_dict, combo, all_data, method="majority"):
    """코인별 백테스트 + WF 검증"""
    results = []
    for name, df in all_data.items():
        r = backtest_ensemble(models_dict, list(combo), df, method)
        r["name"] = name
        results.append(r)

    avg_ret = np.mean([r["return_pct"] for r in results])
    avg_mdd = np.mean([r["mdd_pct"] for r in results])
    wins = sum(1 for r in results if r["return_pct"] > 0)

    # WF 검증 (대표 코인 5개)
    wf_pass = 0
    wf_total = 0
    top_coins = sorted(results, key=lambda x: x["return_pct"], reverse=True)[:5]
    for coin_r in top_coins:
        name = coin_r["name"]
        df = all_data[name]
        full_df = pd.read_csv(ALT_DATA_DIR / f"{name.lower()}_30m.csv", parse_dates=["time"])
        coin_wf_wins = 0
        for start, end in WF_WINDOWS:
            wf_df = full_df[(full_df["time"] > pd.Timestamp(start)) & (full_df["time"] <= pd.Timestamp(end))].reset_index(drop=True)
            if len(wf_df) < 50:
                continue
            wf_r = backtest_ensemble(models_dict, list(combo), wf_df, method)
            wf_total += 1
            if wf_r["return_pct"] > 0:
                wf_pass += 1
                coin_wf_wins += 1

    return {
        "combo": list(combo), "method": method,
        "avg_return": round(avg_ret, 1), "avg_mdd": round(avg_mdd, 1),
        "profitable_coins": f"{wins}/{len(results)}",
        "wf_pass": f"{wf_pass}/{wf_total}",
        "details": results,
    }


def main():
    print("=== ALT v2 앙상블 조합 탐색 ===\n")

    # 모델 로드
    models_dict = load_models()
    if len(models_dict) < 2:
        print(f"모델 {len(models_dict)}개 — 최소 2개 필요. 학습 완료 후 재실행하세요.")
        return

    model_keys = sorted(models_dict.keys())
    print(f"\n사용 가능 모델: {model_keys} ({len(model_keys)}개)")

    # 테스트 데이터 로드
    all_data = load_test_data()
    print(f"테스트 코인: {len(all_data)}종\n")

    # === 1. 개별 모델 성능 ===
    print(f"{'='*70}")
    print("1. 개별 모델 성능")
    print(f"{'='*70}")
    print(f"{'모델':<12} {'평균수익':>10} {'평균MDD':>10} {'수익코인':>10}")
    print(f"{'-'*45}")

    individual = {}
    for k in model_keys:
        results = []
        for name, df in all_data.items():
            env = AltUniversalTradingEnv(
                [df], initial_balance=10000.0, leverage=2, window_size=20,
                min_hold_steps=4, max_episode_len=len(df)+100,
                max_drawdown=1.0, cooldown_steps=8, curriculum=False
            )
            obs, _ = env.reset(); done = False
            while not done:
                a, _ = models_dict[k].predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(a)); done = term or trunc
            ret = (info["balance"] - 10000) / 10000 * 100
            results.append({"name": name, "return_pct": ret})
        avg = np.mean([r["return_pct"] for r in results])
        wins = sum(1 for r in results if r["return_pct"] > 0)
        individual[k] = {"avg": avg, "wins": wins, "total": len(results)}
        print(f"{k:<12} {avg:>+9.1f}% {'':>10} {wins:>6}/{len(results)}")

    # === 2. 앙상블 조합 탐색 (3모델) ===
    print(f"\n{'='*70}")
    print("2. 앙상블 조합 탐색 (3모델 다수결)")
    print(f"{'='*70}")

    all_combos = []
    for combo in combinations(model_keys, 3):
        r = evaluate_combination(models_dict, combo, all_data, "majority")
        all_combos.append(r)
        combo_str = "+".join(combo)
        print(f"  {combo_str:<30} 수익 {r['avg_return']:>+6.1f}% MDD {r['avg_mdd']:>6.1f}% "
              f"코인 {r['profitable_coins']} WF {r['wf_pass']}")

    # 상위 5개 상세
    all_combos.sort(key=lambda x: x["avg_return"], reverse=True)
    print(f"\n--- 상위 5개 조합 ---")
    for i, r in enumerate(all_combos[:5]):
        print(f"  #{i+1}: {'+'.join(r['combo'])}")
        print(f"      수익 {r['avg_return']:>+.1f}%, MDD {r['avg_mdd']:.1f}%, 코인 {r['profitable_coins']}, WF {r['wf_pass']}")

    # === 3. 5모델 전체 앙상블 (있으면) ===
    if len(model_keys) >= 5:
        print(f"\n{'='*70}")
        print("3. 5모델 전체 앙상블 (다수결)")
        print(f"{'='*70}")
        r = evaluate_combination(models_dict, model_keys, all_data, "majority")
        print(f"  수익 {r['avg_return']:>+.1f}%, MDD {r['avg_mdd']:.1f}%, 코인 {r['profitable_coins']}, WF {r['wf_pass']}")

    # 결과 저장
    result_file = MODELS_DIR / "alt_v2_ensemble_results.json"
    result_file.write_text(json.dumps({
        "individual": {k: v for k, v in individual.items()},
        "top_combos": [{k: v for k, v in c.items() if k != "details"} for c in all_combos[:10]],
    }, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {result_file}")


if __name__ == "__main__":
    main()

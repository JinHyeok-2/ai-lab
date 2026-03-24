#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# 앙상블 조합 탐색: 다양한 모델 조합 + 투표 전략 비교
# 후보: exp14, exp08, exp05(LSTM), exp13, exp15, seed600, seed700

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import logging
from collections import Counter
from itertools import combinations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from env_v51_exp02 import ETHTradingEnvV51Exp02

EXP_DIR = Path(__file__).parent.parent
OUT_DIR = EXP_DIR / "experiments"
LEVERAGE = 3

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "ensemble_search.log", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# 후보 모델 정의 (단일 백테스트 성능 기준 상위)
CANDIDATES = {
    "exp14": {"path": "models/exp14/ppo_eth_30m.zip", "lstm": False, "single_ret": 864, "single_wr": 77.8},
    "exp08": {"path": "models/exp08/ppo_eth_30m.zip", "lstm": False, "single_ret": 832, "single_wr": 70.5},
    "exp05": {"path": "models/exp05/ppo_eth_30m.zip", "lstm": True,  "single_ret": 600, "single_wr": 68.0},
    "exp13": {"path": "models/exp13/ppo_eth_30m.zip", "lstm": False, "single_ret": 527, "single_wr": 72.7},
    "exp15": {"path": "models/exp15/ppo_eth_30m.zip", "lstm": False, "single_ret": 684, "single_wr": 86.7},
    "seed700": {"path": "models/seed700/ppo_eth_30m.zip", "lstm": False, "single_ret": 689, "single_wr": 70.3},
    "seed600": {"path": "models/seed600/ppo_eth_30m.zip", "lstm": False, "single_ret": 197, "single_wr": 68.9},
}


def load_model(name):
    """모델 로드"""
    info = CANDIDATES[name]
    path = str(EXP_DIR / info["path"])
    if info["lstm"]:
        return RecurrentPPO.load(path, device="cpu"), True
    else:
        return PPO.load(path, device="cpu"), False


def predict_actions(models_with_info, obs, lstm_states_dict, episode_start):
    """모든 모델의 행동 예측"""
    actions = []
    for name, model, is_lstm in models_with_info:
        if is_lstm:
            ac, lstm_states_dict[name] = model.predict(
                obs, state=lstm_states_dict.get(name),
                episode_start=episode_start, deterministic=True)
        else:
            ac, _ = model.predict(obs, deterministic=True)
        actions.append((name, int(ac)))
    return actions


def majority_vote(actions, threshold=None):
    """다수결 투표. threshold=None이면 과반, 아니면 해당 비율 이상"""
    votes = [a for _, a in actions]
    n = len(votes)
    min_agree = (n // 2 + 1) if threshold is None else max(2, int(n * threshold + 0.5))

    vote_count = Counter(votes)
    majority = vote_count.most_common(1)[0]
    if majority[1] >= min_agree:
        return majority[0]
    return 0  # 합의 없으면 관망


def weighted_vote(actions, weights):
    """가중 투표: 각 모델 가중치 기반"""
    score = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for (name, ac), w in zip(actions, weights):
        score[ac] += w

    best = max(score, key=score.get)
    # 관망이 아닌 행동이 가중합 50% 이상이어야 실행
    total_w = sum(weights)
    if best != 0 and score[best] < total_w * 0.4:
        return 0
    return best


def backtest_ensemble(model_names, strategy="majority", custom_weights=None, test_df=None):
    """앙상블 백테스트"""
    # 모델 로드
    models_with_info = []
    for name in model_names:
        model, is_lstm = load_model(name)
        models_with_info.append((name, model, is_lstm))

    env = ETHTradingEnvV51Exp02(test_df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)

    obs, _ = env.reset()
    balances = [env.initial_balance]
    done = False
    lstm_states_dict = {}
    episode_start = np.ones((1,), dtype=bool)
    trade_actions = []

    # 가중치 설정
    if custom_weights is None:
        weights = [1.0] * len(model_names)
    else:
        weights = custom_weights

    while not done:
        actions = predict_actions(models_with_info, obs, lstm_states_dict, episode_start)

        if strategy == "majority":
            action = majority_vote(actions)
        elif strategy == "weighted":
            action = weighted_vote(actions, weights)
        elif strategy == "unanimous":
            # 만장일치: 전원 동의해야 행동
            votes = [a for _, a in actions]
            action = votes[0] if len(set(votes)) == 1 else 0
        else:
            action = majority_vote(actions)

        obs, _, term, trunc, info = env.step(action)
        episode_start = np.array([term or trunc])
        done = term or trunc
        balances.append(info["balance"])
        trade_actions.append(action)

    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()

    dist = Counter(trade_actions)
    short_pct = dist.get(2, 0) / max(info["total_trades"], 1) * 100

    return {
        "return": (balances[-1] - 10000) / 10000 * 100,
        "mdd": mdd,
        "trades": info["total_trades"],
        "win_rate": info["win_rate"],
        "short_pct": short_pct,
        "balances": balances,
    }


def run():
    # 데이터 로드
    data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)

    log.info("=" * 70)
    log.info("  앙상블 조합 탐색")
    log.info(f"  후보 모델: {list(CANDIDATES.keys())}")
    log.info("=" * 70)

    all_results = []

    # === 1. 3모델 조합 다수결 (기존 + 새 조합) ===
    log.info("\n\n[1] 3모델 다수결 조합")
    log.info("-" * 70)

    # 핵심 후보: exp14(최고 단일) 포함 조합 위주
    combos_3 = [
        ("exp14", "exp08", "exp05"),        # 기존 프로덕션 후보
        ("exp14", "exp08", "seed700"),       # seed700 교체
        ("exp14", "exp05", "seed700"),       # exp08→seed700
        ("exp14", "exp15", "exp08"),         # exp15(승률최고) 추가
        ("exp14", "exp15", "seed700"),       # exp15+seed700
        ("exp14", "exp13", "exp08"),         # exp13(gamma=0.98)
        ("exp14", "seed700", "seed600"),     # 시드 조합
        ("exp08", "exp05", "seed700"),       # exp14 제외
        ("exp15", "exp08", "exp05"),         # exp15 베이스
        ("exp14", "exp15", "exp05"),         # 상위3 (exp14,exp15,exp05)
    ]

    for combo in combos_3:
        try:
            r = backtest_ensemble(combo, strategy="majority", test_df=test_df)
            label = "+".join(combo)
            log.info(f"  {label:35s} → 수익 {r['return']:+8.1f}% | MDD {r['mdd']:6.1f}% | "
                     f"거래 {r['trades']:3d}회 | 승률 {r['win_rate']:.1%} | 숏 {r['short_pct']:.0f}%")
            all_results.append({"combo": label, "n": 3, "strategy": "majority", **r})
        except Exception as e:
            log.info(f"  {'+'.join(combo):35s} → 실패: {e}")

    # === 2. 5모델 다수결 (3/5 이상 동의) ===
    log.info("\n\n[2] 5모델 다수결 조합")
    log.info("-" * 70)

    combos_5 = [
        ("exp14", "exp08", "exp05", "exp15", "seed700"),      # 상위 5
        ("exp14", "exp08", "exp05", "exp13", "seed700"),      # exp13 포함
        ("exp14", "exp08", "exp05", "exp15", "exp13"),        # 시드 제외
        ("exp14", "exp08", "exp15", "seed700", "seed600"),    # LSTM 제외
    ]

    for combo in combos_5:
        try:
            r = backtest_ensemble(combo, strategy="majority", test_df=test_df)
            label = "+".join(combo)
            log.info(f"  {label:50s} → 수익 {r['return']:+8.1f}% | MDD {r['mdd']:6.1f}% | "
                     f"거래 {r['trades']:3d}회 | 승률 {r['win_rate']:.1%} | 숏 {r['short_pct']:.0f}%")
            all_results.append({"combo": label, "n": 5, "strategy": "majority", **r})
        except Exception as e:
            log.info(f"  {'+'.join(combo):50s} → 실패: {e}")

    # === 3. 가중 투표 (수익률 비례 가중치) ===
    log.info("\n\n[3] 가중 투표 (단일 수익률 비례)")
    log.info("-" * 70)

    weighted_combos = [
        ("exp14", "exp08", "exp05"),
        ("exp14", "exp08", "seed700"),
        ("exp14", "exp15", "seed700"),
        ("exp14", "exp08", "exp05", "exp15", "seed700"),
    ]

    for combo in weighted_combos:
        try:
            weights = [CANDIDATES[n]["single_ret"] for n in combo]
            # 정규화
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            r = backtest_ensemble(combo, strategy="weighted", custom_weights=weights, test_df=test_df)
            label = "+".join(combo)
            log.info(f"  {label:50s} → 수익 {r['return']:+8.1f}% | MDD {r['mdd']:6.1f}% | "
                     f"거래 {r['trades']:3d}회 | 승률 {r['win_rate']:.1%} | 숏 {r['short_pct']:.0f}%")
            all_results.append({"combo": label, "n": len(combo), "strategy": "weighted", **r})
        except Exception as e:
            log.info(f"  {'+'.join(combo):50s} → 실패: {e}")

    # === 4. 만장일치 (보수적) ===
    log.info("\n\n[4] 만장일치 (보수적)")
    log.info("-" * 70)

    for combo in [("exp14", "exp08", "exp05"), ("exp14", "exp08", "seed700")]:
        try:
            r = backtest_ensemble(combo, strategy="unanimous", test_df=test_df)
            label = "+".join(combo)
            log.info(f"  {label:35s} → 수익 {r['return']:+8.1f}% | MDD {r['mdd']:6.1f}% | "
                     f"거래 {r['trades']:3d}회 | 승률 {r['win_rate']:.1%} | 숏 {r['short_pct']:.0f}%")
            all_results.append({"combo": label, "n": 3, "strategy": "unanimous", **r})
        except Exception as e:
            log.info(f"  {'+'.join(combo):35s} → 실패: {e}")

    # === 최종 랭킹 ===
    log.info("\n\n" + "=" * 70)
    log.info("  최종 랭킹 (수익률 × 승률 / |MDD| 스코어)")
    log.info("=" * 70)

    for r in all_results:
        # 종합 스코어: 수익률 × 승률 / |MDD| (높을수록 좋음)
        mdd_abs = max(abs(r["mdd"]), 0.1)
        r["score"] = r["return"] * r["win_rate"] / mdd_abs

    ranked = sorted(all_results, key=lambda x: x["score"], reverse=True)

    log.info(f"\n{'순위':>4s} {'조합':50s} {'전략':10s} {'수익률':>8s} {'MDD':>7s} {'거래':>4s} {'승률':>6s} {'스코어':>7s}")
    log.info("-" * 100)
    for i, r in enumerate(ranked[:20], 1):
        log.info(f"  {i:2d}. {r['combo']:50s} {r['strategy']:10s} "
                 f"{r['return']:+7.1f}% {r['mdd']:6.1f}% {r['trades']:4d} {r['win_rate']:.1%} {r['score']:7.1f}")

    # 최고 조합 상세
    best = ranked[0]
    log.info(f"\n  최적 조합: {best['combo']} ({best['strategy']})")
    log.info(f"  수익: {best['return']:+.1f}% | MDD: {best['mdd']:.1f}% | 거래: {best['trades']}회 | 승률: {best['win_rate']:.1%}")

    # 그래프: 상위 5개 조합
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Ensemble Combination Search - Top Results", fontsize=14, fontweight="bold")

    # 상단: 수익률/MDD 바 차트
    ax = axes[0]
    top5 = ranked[:8]
    labels = [f"{r['combo'][:30]}\n({r['strategy']})" for r in top5]
    x = np.arange(len(top5))
    ax.bar(x - 0.2, [r["return"] for r in top5], 0.35, label="Return %", color="#2E7D32", alpha=0.8)
    ax.bar(x + 0.2, [r["mdd"] for r in top5], 0.35, label="MDD %", color="#C62828", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Return vs MDD (Top 8)")

    # 하단: 상위 5개 자산 곡선
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, min(5, len(ranked))))
    for i, r in enumerate(ranked[:5]):
        ax2.plot(r["balances"], color=colors[i], lw=1.5,
                 label=f"#{i+1} {r['combo'][:25]} ({r['strategy']}) {r['return']:+.0f}%")
    ax2.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_ylabel("Balance ($)")
    ax2.set_title("Top 5 Equity Curves")

    plt.tight_layout()
    out = OUT_DIR / "ensemble_search_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"\n그래프: {out}")
    plt.close()


if __name__ == "__main__":
    run()

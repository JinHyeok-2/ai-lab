#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# Walk-forward 검증: 테스트 기간을 2개월 단위로 분할하여 구간별 성능 측정
# exp14 단일 + 앙상블(exp14+exp08+exp05)의 과적합 여부 확인

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import logging
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from env_v51_exp02 import ETHTradingEnvV51Exp02

EXP_DIR = Path(__file__).parent.parent
OUT_DIR = EXP_DIR / "experiments"
LEVERAGE = 3

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "walkforward.log", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def backtest_single(model, df, use_lstm=False):
    """단일 모델 백테스트, 결과 dict 반환"""
    env = ETHTradingEnvV51Exp02(df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    balances = [env.initial_balance]
    actions = []
    done = False
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        if use_lstm:
            ac, lstm_states = model.predict(obs, state=lstm_states,
                episode_start=episode_start, deterministic=True)
        else:
            ac, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(ac))
        episode_start = np.array([term or trunc])
        done = term or trunc
        balances.append(info["balance"])
        actions.append(int(ac))

    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    return {
        "return": (balances[-1] - 10000) / 10000 * 100,
        "mdd": mdd,
        "trades": info["total_trades"],
        "win_rate": info["win_rate"],
        "balances": balances,
    }


def backtest_ensemble(models, df):
    """앙상블(다수결) 백테스트"""
    m14, m08, m05 = models
    env = ETHTradingEnvV51Exp02(df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    balances = [env.initial_balance]
    actions = []
    done = False
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        a14, _ = m14.predict(obs, deterministic=True)
        a08, _ = m08.predict(obs, deterministic=True)
        a05, lstm_states = m05.predict(obs, state=lstm_states,
            episode_start=episode_start, deterministic=True)

        votes = [int(a14), int(a08), int(a05)]
        vote_count = Counter(votes)
        majority = vote_count.most_common(1)[0]
        action = majority[0] if majority[1] >= 2 else 0

        obs, _, term, trunc, info = env.step(action)
        episode_start = np.array([term or trunc])
        done = term or trunc
        balances.append(info["balance"])
        actions.append(action)

    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    return {
        "return": (balances[-1] - 10000) / 10000 * 100,
        "mdd": mdd,
        "trades": info["total_trades"],
        "win_rate": info["win_rate"],
        "balances": balances,
    }


def run():
    # 데이터 로드
    data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)

    # 2개월(~2880 캔들) 단위로 분할
    window_candles = 2880  # 약 2개월 (30분 * 48/일 * 60일)
    total = len(test_df)
    windows = []
    for start in range(0, total - window_candles + 1, window_candles):
        end = min(start + window_candles, total)
        w_df = test_df.iloc[start:end].reset_index(drop=True)
        t0 = test_df["time"].iloc[start].strftime("%Y-%m-%d")
        t1 = test_df["time"].iloc[end-1].strftime("%Y-%m-%d")
        windows.append((f"{t0} ~ {t1}", w_df))

    # 나머지 구간 추가 (마지막 불완전 윈도우)
    last_start = (total // window_candles) * window_candles
    if last_start < total and last_start > 0:
        w_df = test_df.iloc[last_start:].reset_index(drop=True)
        if len(w_df) > 100:  # 최소 100캔들 이상
            t0 = test_df["time"].iloc[last_start].strftime("%Y-%m-%d")
            t1 = test_df["time"].iloc[-1].strftime("%Y-%m-%d")
            windows.append((f"{t0} ~ {t1}", w_df))

    # 전체 기간도 추가
    t0 = test_df["time"].iloc[0].strftime("%Y-%m-%d")
    t1 = test_df["time"].iloc[-1].strftime("%Y-%m-%d")
    windows.append((f"전체 {t0} ~ {t1}", test_df))

    log.info("=" * 70)
    log.info("  Walk-Forward 검증: 2개월 롤링 윈도우")
    log.info(f"  전체 테스트 캔들: {total}, 윈도우 크기: {window_candles}")
    log.info(f"  총 {len(windows)}개 구간 (분할 {len(windows)-1} + 전체 1)")
    log.info("=" * 70)

    # 모델 로드
    m14 = PPO.load(str(EXP_DIR / "models/exp14/ppo_eth_30m.zip"), device="cpu")
    m08 = PPO.load(str(EXP_DIR / "models/exp08/ppo_eth_30m.zip"), device="cpu")
    m05 = RecurrentPPO.load(str(EXP_DIR / "models/exp05/ppo_eth_30m.zip"), device="cpu")

    # 구간별 테스트
    results_14 = []
    results_ens = []

    for label, w_df in windows:
        log.info(f"\n--- {label} (캔들 {len(w_df)}) ---")

        r14 = backtest_single(m14, w_df, use_lstm=False)
        rens = backtest_ensemble((m14, m08, m05), w_df)

        results_14.append({"period": label, **r14})
        results_ens.append({"period": label, **rens})

        log.info(f"  exp14:  수익 {r14['return']:+7.1f}% | MDD {r14['mdd']:6.1f}% | 거래 {r14['trades']:3d}회 | 승률 {r14['win_rate']:.1%}")
        log.info(f"  앙상블: 수익 {rens['return']:+7.1f}% | MDD {rens['mdd']:6.1f}% | 거래 {rens['trades']:3d}회 | 승률 {rens['win_rate']:.1%}")

    # 요약 테이블 (전체 제외)
    split_results_14 = results_14[:-1]
    split_results_ens = results_ens[:-1]

    log.info("\n" + "=" * 70)
    log.info("  Walk-Forward 요약 (전체 기간 제외)")
    log.info("=" * 70)

    positive_14 = sum(1 for r in split_results_14 if r["return"] > 0)
    positive_ens = sum(1 for r in split_results_ens if r["return"] > 0)
    n_splits = len(split_results_14)

    avg_ret_14 = np.mean([r["return"] for r in split_results_14])
    avg_ret_ens = np.mean([r["return"] for r in split_results_ens])
    std_ret_14 = np.std([r["return"] for r in split_results_14])
    std_ret_ens = np.std([r["return"] for r in split_results_ens])
    worst_mdd_14 = min(r["mdd"] for r in split_results_14)
    worst_mdd_ens = min(r["mdd"] for r in split_results_ens)
    avg_wr_14 = np.mean([r["win_rate"] for r in split_results_14])
    avg_wr_ens = np.mean([r["win_rate"] for r in split_results_ens])

    log.info(f"\n  exp14 단일:")
    log.info(f"    수익 구간: {positive_14}/{n_splits} ({positive_14/n_splits:.0%})")
    log.info(f"    평균 수익: {avg_ret_14:+.1f}% (±{std_ret_14:.1f}%)")
    log.info(f"    최악 MDD:  {worst_mdd_14:.1f}%")
    log.info(f"    평균 승률: {avg_wr_14:.1%}")

    log.info(f"\n  앙상블(exp14+exp08+exp05):")
    log.info(f"    수익 구간: {positive_ens}/{n_splits} ({positive_ens/n_splits:.0%})")
    log.info(f"    평균 수익: {avg_ret_ens:+.1f}% (±{std_ret_ens:.1f}%)")
    log.info(f"    최악 MDD:  {worst_mdd_ens:.1f}%")
    log.info(f"    평균 승률: {avg_wr_ens:.1%}")

    # 과적합 판정
    log.info(f"\n  [과적합 판정]")
    if positive_ens >= n_splits * 0.7:
        log.info(f"  ✓ 앙상블: {positive_ens}/{n_splits} 구간 수익 → 과적합 가능성 낮음")
    else:
        log.info(f"  ✗ 앙상블: {positive_ens}/{n_splits} 구간만 수익 → 과적합 주의")

    if std_ret_ens < std_ret_14:
        log.info(f"  ✓ 앙상블 수익 변동성({std_ret_ens:.1f}%) < 단일({std_ret_14:.1f}%) → 안정적")
    else:
        log.info(f"  △ 앙상블 수익 변동성({std_ret_ens:.1f}%) ≥ 단일({std_ret_14:.1f}%)")

    # 그래프
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Walk-Forward Validation (2-month rolling windows)", fontsize=14, fontweight="bold")

    # 상단: 구간별 수익률 바 차트
    ax = axes[0]
    periods = [r["period"].split(" ~ ")[0][:7] for r in split_results_14]
    x = np.arange(len(periods))
    w = 0.35
    bars1 = ax.bar(x - w/2, [r["return"] for r in split_results_14], w, label="exp14", color="#E65100", alpha=0.8)
    bars2 = ax.bar(x + w/2, [r["return"] for r in split_results_ens], w, label="Ensemble", color="#2E7D32", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Return (%)")
    ax.set_title("Return by Period")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # 바 위에 값 표시
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():+.0f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():+.0f}%', ha='center', va='bottom', fontsize=8)

    # 하단: 전체 기간 자산 곡선
    ax2 = axes[1]
    ax2.plot(results_14[-1]["balances"], color="#E65100", lw=1.2, ls="--",
             label=f'exp14 {results_14[-1]["return"]:+.0f}%')
    ax2.plot(results_ens[-1]["balances"], color="#2E7D32", lw=2,
             label=f'Ensemble {results_ens[-1]["return"]:+.0f}%')
    ax2.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax2.set_ylabel("Balance ($)")
    ax2.set_title("Full Period Equity Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "walkforward_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"\n그래프: {out}")
    plt.close()


if __name__ == "__main__":
    run()

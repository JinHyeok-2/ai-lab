#!/usr/bin/env python3
# 학습된 모델 백테스트 및 성과 분석 — VecNormalize 지원
# 실행 예시:
#   python rl/backtest.py --interval 30m --version v6
#   python rl/backtest.py --interval 30m --version v2          (구버전: VecNormalize 없어도 동작)
#   python rl/backtest.py --interval 30m --version v6 --test-start 2025-10-01

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.data import load_data
from rl.env import ETHTradingEnv


def run_backtest(interval: str = "1h", version: str = "v6",
                 split: float = None, test_start: str = None):
    model_path   = Path(__file__).parent / "models" / version / f"ppo_eth_{interval}.zip"
    meta_path    = Path(__file__).parent / "models" / version / f"meta_{interval}.json"
    vecnorm_path = Path(__file__).parent / "models" / version / f"vecnorm_{interval}.pkl"

    # 메타데이터 로드 (버전별 설정 자동 적용)
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # 레버리지 / window_size: 메타에 있으면 사용, 없으면 버전 추정
    leverage    = meta.get("leverage",    3)
    window_size = meta.get("window_size", 40)

    df = load_data(interval)

    # 테스트 구간 결정: test-start > split > meta > 기본 0.8
    if test_start:
        cutoff    = pd.Timestamp(test_start)
        split_idx = df[df["time"] >= cutoff].index[0]
        print(f"날짜 기준 테스트 시작: {test_start}")
    elif split is not None:
        split_idx = int(len(df) * split)
        print(f"비율 기준 분할: {split:.0%}")
    elif meta.get("train_candles"):
        split_idx = meta["train_candles"]
        print(f"메타데이터 기준 분할")
    else:
        split_idx = int(len(df) * 0.8)
        print(f"기본 분할: 80%")

    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"[{interval}/{version}] 백테스트: {test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()}")
    print(f"총 {len(test_df)}개 캔들  |  레버리지 {leverage}x  |  window {window_size}\n")

    # 환경 생성
    from rl.train import MIN_HOLD
    env = ETHTradingEnv(
        test_df,
        initial_balance=10000.0,
        leverage=leverage,
        window_size=window_size,
        min_hold_steps=MIN_HOLD.get(interval, 3),
    )

    # VecNormalize 로드 (v6 이후 모델)
    vecnorm = None
    if vecnorm_path.exists():
        dummy = DummyVecEnv([lambda: ETHTradingEnv(
            test_df, initial_balance=10000.0, leverage=leverage,
            window_size=window_size, min_hold_steps=MIN_HOLD.get(interval, 3),
        )])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy)
        vecnorm.training    = False
        vecnorm.norm_reward = False
        print("VecNormalize 로드됨 (관측값 정규화 적용)")

    model = PPO.load(str(model_path))

    # 백테스트 실행
    obs, _ = env.reset()
    balance_history = [env.initial_balance]
    price_history   = [float(test_df["close"].iloc[env.window_size])]
    actions_taken   = []
    done = False

    while not done:
        # VecNormalize 있으면 obs 정규화 후 예측
        if vecnorm is not None:
            obs_input = vecnorm.normalize_obs(obs.reshape(1, -1))[0]
        else:
            obs_input = obs

        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        step_idx = min(env.current_step, len(test_df) - 1)
        balance_history.append(info["balance"])
        price_history.append(float(test_df["close"].iloc[step_idx - 1]))
        actions_taken.append(int(action))

    # 성과 계산
    final_balance = info["balance"]
    total_return  = (final_balance - 10000) / 10000 * 100
    buy_hold_ret  = (test_df["close"].iloc[-1] - test_df["close"].iloc[env.window_size]) \
                    / test_df["close"].iloc[env.window_size] * 100

    balance_arr = np.array(balance_history)
    peak        = np.maximum.accumulate(balance_arr)
    drawdown    = (balance_arr - peak) / peak * 100
    mdd         = drawdown.min()

    win_rate     = info["win_rate"]
    total_trades = info["total_trades"]

    print("=" * 50)
    print(f"백테스트 결과 [{version}]")
    print("=" * 50)
    print(f"초기 잔고:     $10,000")
    print(f"최종 잔고:     ${final_balance:,.2f}")
    print(f"총 수익률:     {total_return:+.2f}%")
    print(f"Buy & Hold:    {buy_hold_ret:+.2f}%")
    print(f"최대 낙폭:     {mdd:.2f}%")
    print(f"총 거래 횟수:  {total_trades}회")
    print(f"승률:          {win_rate:.1%}")
    print("=" * 50)

    # 행동 분포
    action_names = {0: "관망", 1: "롱", 2: "숏", 3: "청산"}
    from collections import Counter
    dist = Counter(actions_taken)
    print("\n행동 분포:")
    for k, v in sorted(dist.items()):
        print(f"  {action_names[k]}: {v}회 ({v/len(actions_taken)*100:.1f}%)")

    # 시각화
    _plot_results(test_df, balance_history, price_history, actions_taken,
                  total_return, buy_hold_ret, mdd, win_rate, total_trades,
                  interval, version)

    return {
        "version": version, "interval": interval,
        "total_return": total_return, "buy_hold_ret": buy_hold_ret,
        "mdd": mdd, "win_rate": win_rate, "total_trades": total_trades,
    }


def _plot_results(test_df, balance_history, price_history, actions_taken,
                  total_return, buy_hold_ret, mdd, win_rate, total_trades,
                  interval="1h", version="v6"):
    _font = "AppleGothic" if platform.system() == "Darwin" else "Noto Sans CJK JP"
    plt.rcParams["font.family"] = _font
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"ETH RL 트레이딩 백테스트 [{interval} / {version}]", fontsize=14, fontweight="bold")

    steps = range(len(balance_history))

    # 1. 잔고 곡선
    ax1 = axes[0]
    ax1.plot(steps, balance_history, color="#1a237e", linewidth=1.5, label="RL 에이전트")
    bh_balance = [10000 * (p / price_history[0]) for p in price_history]
    ax1.plot(range(len(bh_balance)), bh_balance, color="#aaa", linewidth=1,
             linestyle="--", label="Buy & Hold")
    ax1.axhline(10000, color="#e53935", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("잔고 (USDT)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title(f"잔고 변화  |  RL: {total_return:+.1f}%  vs  B&H: {buy_hold_ret:+.1f}%")

    # 2. ETH 가격 + 매매 포인트
    ax2 = axes[1]
    prices    = price_history[:len(actions_taken)]
    long_idx  = [i for i, a in enumerate(actions_taken) if a == 1]
    short_idx = [i for i, a in enumerate(actions_taken) if a == 2]
    close_idx = [i for i, a in enumerate(actions_taken) if a == 3]
    ax2.plot(range(len(prices)), prices, color="#555", linewidth=1, label="ETH 가격")
    ax2.scatter(long_idx,  [prices[i] for i in long_idx  if i < len(prices)],
                marker="^", color="#4CAF50", s=30, label="롱 진입", zorder=5)
    ax2.scatter(short_idx, [prices[i] for i in short_idx if i < len(prices)],
                marker="v", color="#e53935", s=30, label="숏 진입", zorder=5)
    ax2.scatter(close_idx, [prices[i] for i in close_idx if i < len(prices)],
                marker="x", color="#FF9800", s=20, label="청산", zorder=5)
    ax2.set_ylabel("ETH 가격 (USDT)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title("매매 포인트")

    # 3. 성과 요약
    ax3 = axes[2]
    ax3.axis("off")
    summary = (
        f"총 수익률: {total_return:+.2f}%   |   Buy & Hold: {buy_hold_ret:+.2f}%   |   "
        f"최대낙폭: {mdd:.2f}%   |   거래횟수: {total_trades}회   |   승률: {win_rate:.1%}"
    )
    ax3.text(0.5, 0.5, summary, transform=ax3.transAxes,
             ha="center", va="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f0fe", edgecolor="#1a237e"))

    plt.tight_layout()
    out_path = Path(__file__).parent / f"backtest_{interval}_{version}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장: {out_path}")
    if platform.system() == "Darwin":
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval",   default="1h", choices=["15m","30m","1h","2h","4h","1d"])
    parser.add_argument("--version",    default="v6", help="모델 버전 (예: v6, v7)")
    parser.add_argument("--split",      type=float,   default=None,
                        help="테스트 시작 비율 (예: 0.6 → 60%% 이후 테스트)")
    parser.add_argument("--test-start", default=None,
                        help="테스트 시작 날짜 (예: 2025-09-01)")
    args = parser.parse_args()
    run_backtest(interval=args.interval, version=args.version,
                 split=args.split, test_start=args.test_start)

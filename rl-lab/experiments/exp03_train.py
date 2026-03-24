#!/usr/bin/env python3
# exp03: 보상 대칭화 + 과다매매 억제 + DD 패널티 + ADX 필터
# 실행: conda run -n ai-lab python rl-lab/experiments/exp03_train.py
#
# exp01 대비 변경:
#   1. 추세 진입 보상 축소: 0.012 → 0.006
#   2. ADX > 0.25일 때만 추세 보상 적용
#   3. cooldown 5 → 8
#   4. DD 단계별 패널티 (5%: -0.05, 10%: -0.15)
#   5. PnL 보상 스케일: 20 → 15

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import logging
from datetime import datetime
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from rl.data import fetch_eth_data
from env_v51_exp03 import ETHTradingEnvV51Exp03
from rl.env_v5 import ETHTradingEnvV5

# ── 설정 ─────────────────────────────────────────────────
EXP_NAME    = "exp03"
INTERVAL    = "30m"
LEVERAGE    = 3
TRAIN_END   = "2025-09-01"
DATA_DAYS   = 820
N_ENVS      = 4
TOTAL_STEPS = 3_000_000

EXP_DIR   = Path(__file__).parent.parent
MODEL_DIR = EXP_DIR / "models" / EXP_NAME
OUT_DIR   = EXP_DIR / "experiments"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / f"{EXP_NAME}.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── 커리큘럼 콜백 ────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    def __init__(self, total_steps: int, check_freq: int = 10000):
        super().__init__()
        self.total_steps = total_steps
        self.check_freq  = check_freq
        self.start_time  = None
        self.best_reward = -np.inf

    def _on_training_start(self):
        self.start_time = datetime.now()
        log.info(f"[{self.start_time.strftime('%H:%M:%S')}] {EXP_NAME} 학습 시작 ({self.total_steps:,} 스텝)")

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_steps

        # 커리큘럼: 50% → 30% → 자연
        if progress < 0.25:
            bear_ratio = 0.5
            phase = "1-하락집중(50%)"
        elif progress < 0.65:
            bear_ratio = 0.3
            phase = "2-혼합(30%)"
        else:
            bear_ratio = 0.0
            phase = "3-자연"

        self.training_env.env_method("set_bear_ratio", bear_ratio)

        new_ent = max(0.01 - 0.007 * progress, 0.003)
        self.model.ent_coef = new_ent

        if self.num_timesteps % self.check_freq < N_ENVS:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            pct     = progress * 100
            eta_m   = int(elapsed / max(self.num_timesteps, 1) *
                          (self.total_steps - self.num_timesteps) / 60)
            mean_r  = "N/A"
            n_eps   = len(self.model.ep_info_buffer)
            if n_eps > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                mean_r  = f"{np.mean(rewards):+.3f}"
                if np.mean(rewards) > self.best_reward:
                    self.best_reward = np.mean(rewards)
            log.info(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{self.num_timesteps:>9,}/{self.total_steps:,} ({pct:5.1f}%) | "
                f"보상: {mean_r} (ep:{n_eps}) | ent: {new_ent:.4f} | "
                f"{phase} | ETA: ~{eta_m}분"
            )
        return True

    def _on_training_end(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log.info(f"\n학습 완료: {elapsed/60:.1f}분 소요 | 최고 보상: {self.best_reward:+.3f}")


# ── 학습 ─────────────────────────────────────────────────
def train():
    data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    if not data_path.exists():
        log.info(f"API 데이터 수집 ({DATA_DAYS}일치)...")
        df = fetch_eth_data(interval=INTERVAL, days=DATA_DAYS)
        df.to_csv(data_path, index=False)
    else:
        log.info(f"데이터 로드: {data_path.name}")
        df = pd.read_csv(data_path, parse_dates=["time"])

    cutoff    = pd.Timestamp(TRAIN_END)
    split_idx = df[df["time"] <= cutoff].index[-1] + 1
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    test_df   = df.iloc[split_idx:].reset_index(drop=True)

    log.info(f"학습: {len(train_df):,}캔들 ({train_df['time'].iloc[0].date()} ~ {train_df['time'].iloc[-1].date()})")
    log.info(f"테스트: {len(test_df):,}캔들 ({test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()})")

    def make_env(seed):
        def _init():
            env = ETHTradingEnvV51Exp03(
                train_df, initial_balance=10000.0, leverage=LEVERAGE,
                fee_rate=0.0004, window_size=20, min_hold_steps=4,
                max_episode_len=2000, max_drawdown=0.15,
                cooldown_steps=6, curriculum=True,
            )
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_env(seed=i * 42) for i in range(N_ENVS)])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"ppo_eth_{INTERVAL}"

    model = PPO(
        "MlpPolicy", env,
        learning_rate=lambda p: 3e-4 * (0.3 + 0.7 * p),
        n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        verbose=0, device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=CurriculumCallback(total_steps=TOTAL_STEPS, check_freq=10000),
        progress_bar=False,
    )

    model.save(str(model_path))
    meta = {
        "experiment": EXP_NAME,
        "description": "대칭+DD완화+cooldown6+관망패널티",
        "changes": [
            "추세 진입 보상 축소: 0.012 → 0.006",
            "ADX > 0.25일 때만 추세 보상 적용",
            "cooldown 5 → 8",
            "DD 단계별 패널티: 5%→-0.05, 10%→-0.15, 15%→truncate",
            "PnL 보상 스케일: 20 → 15",
        ],
        "interval": INTERVAL, "leverage": LEVERAGE,
        "steps": TOTAL_STEPS, "n_envs": N_ENVS,
        "train_start": str(train_df["time"].iloc[0].date()),
        "train_end": str(train_df["time"].iloc[-1].date()),
        "test_start": str(test_df["time"].iloc[0].date()),
        "test_end": str(test_df["time"].iloc[-1].date()),
        "train_candles": len(train_df),
        "test_candles": len(test_df),
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"\n모델 저장: {model_path}.zip")
    return model, test_df


# ── 백테스트 ─────────────────────────────────────────────
def backtest(test_df):
    log.info("\n" + "=" * 60)
    log.info("  백테스트: exp03 vs v5 (베이스라인)")
    log.info("=" * 60)

    bh_ret = (test_df["close"].iloc[-1] - test_df["close"].iloc[20]) \
             / test_df["close"].iloc[20] * 100
    log.info(f"테스트: {test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()}")
    log.info(f"Buy & Hold: {bh_ret:+.2f}%\n")

    results = {}

    # v5 베이스라인
    v5_path = PROJECT_ROOT / "rl" / "models" / "v5" / "ppo_eth_30m.zip"
    if v5_path.exists():
        log.info("v5 백테스트...")
        env_v5 = ETHTradingEnvV5(
            test_df, initial_balance=10000.0, leverage=LEVERAGE,
            window_size=20, min_hold_steps=4,
            max_episode_len=len(test_df) + 100,
            max_drawdown=1.0, curriculum=False,
        )
        results["v5"] = _run_backtest(env_v5, v5_path)

    # exp03
    exp_path = MODEL_DIR / f"ppo_eth_{INTERVAL}.zip"
    if exp_path.exists():
        log.info("exp03 백테스트...")
        env_exp = ETHTradingEnvV51Exp03(
            test_df, initial_balance=10000.0, leverage=LEVERAGE,
            window_size=20, min_hold_steps=4,
            max_episode_len=len(test_df) + 100,
            max_drawdown=1.0, cooldown_steps=6, curriculum=False,
        )
        results["exp03"] = _run_backtest(env_exp, exp_path)

    for name, r in results.items():
        s = r["stats"]
        dist = Counter(r["actions"])
        total = len(r["actions"])
        names = {0: "관망", 1: "롱", 2: "숏", 3: "청산"}
        log.info(f"\n{'='*52}")
        log.info(f"  {name}")
        log.info(f"{'='*52}")
        log.info(f"  최종 잔고:   ${s['final']:,.2f}")
        log.info(f"  총 수익률:   {s['return']:+.2f}%")
        log.info(f"  Buy & Hold:  {bh_ret:+.2f}%")
        log.info(f"  최대낙폭:    {s['mdd']:.2f}%")
        log.info(f"  총 거래:     {s['trades']}회 | 승률: {s['win_rate']:.1%}")
        log.info(f"  행동 분포:   " + " / ".join(
            f"{names[k]}:{v}({v/total:.0%})" for k, v in sorted(dist.items())))

    if len(results) == 2:
        _plot_comparison(test_df, results, bh_ret)

    return results


def _run_backtest(env, model_path):
    model = PPO.load(str(model_path), device="cpu")
    obs, _ = env.reset()
    balances = [env.initial_balance]
    actions  = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
        balances.append(info["balance"])
        actions.append(int(action))

    arr  = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd  = ((arr - peak) / peak * 100).min()

    return {
        "balances": balances,
        "actions": actions,
        "stats": {
            "final":    balances[-1],
            "return":   (balances[-1] - 10000) / 10000 * 100,
            "mdd":      mdd,
            "trades":   info["total_trades"],
            "win_rate": info["win_rate"],
        },
    }


def _plot_comparison(test_df, results, bh_ret):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"exp03 (exp03: balanced trade frequency) vs v5 [{INTERVAL}]",
                 fontsize=14, fontweight="bold")

    colors = {"v5": "#1565C0", "exp03": "#E65100"}

    ax1 = axes[0]
    for name, r in results.items():
        s = r["stats"]
        ax1.plot(r["balances"], color=colors[name], lw=1.8,
                 label=f"{name}  {s['return']:+.1f}% (MDD {s['mdd']:.1f}%)")

    prices = [float(test_df["close"].iloc[min(i + 20, len(test_df) - 1)])
              for i in range(max(len(r["balances"]) for r in results.values()))]
    bh = [10000 * (p / prices[0]) for p in prices]
    ax1.plot(bh, color="#aaa", lw=1, ls="--", label=f"Buy & Hold  {bh_ret:+.1f}%")
    ax1.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax1.set_ylabel("Balance (USDT)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("Balance Curve")

    ax2 = axes[1]
    action_names = ["Hold", "Long", "Short", "Close"]
    x = np.arange(4)
    width = 0.35
    for i, (name, r) in enumerate(results.items()):
        dist = Counter(r["actions"])
        total = len(r["actions"])
        pcts = [dist.get(a, 0) / total * 100 for a in range(4)]
        bars = ax2.bar(x + i * width, pcts, width, label=name, color=colors[name], alpha=0.8)
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(action_names)
    ax2.set_ylabel("Action %")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("Action Distribution")

    plt.tight_layout()
    out = OUT_DIR / f"{EXP_NAME}_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"\n그래프 저장: {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--backtest-only", action="store_true")
    args = parser.parse_args()

    if args.backtest_only:
        data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
        df = pd.read_csv(data_path, parse_dates=["time"])
        cutoff = pd.Timestamp(TRAIN_END)
        test_df = df[df["time"] > cutoff].reset_index(drop=True)
        backtest(test_df)
    else:
        TOTAL_STEPS = args.steps
        model, test_df = train()
        backtest(test_df)

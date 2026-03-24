#!/usr/bin/env python3
# exp06: 학습량 2배 (6M 스텝, exp02 환경 동일)
# exp02가 3M에서 수렴 부족일 가능성 검증

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
from env_v51_exp02 import ETHTradingEnvV51Exp02
from rl.env_v5 import ETHTradingEnvV5

EXP_NAME    = "exp06"
INTERVAL    = "30m"
LEVERAGE    = 3
TRAIN_END   = "2025-09-01"
N_ENVS      = 4
TOTAL_STEPS = 6_000_000  # 3M → 6M

EXP_DIR   = Path(__file__).parent.parent
MODEL_DIR = EXP_DIR / "models" / EXP_NAME
OUT_DIR   = EXP_DIR / "experiments"

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / f"{EXP_NAME}.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    def __init__(self, total_steps, check_freq=20000):
        super().__init__()
        self.total_steps = total_steps
        self.check_freq = check_freq
        self.start_time = None
        self.best_reward = -np.inf

    def _on_training_start(self):
        self.start_time = datetime.now()
        log.info(f"[{self.start_time.strftime('%H:%M:%S')}] {EXP_NAME} 학습 시작 ({self.total_steps:,} 스텝)")

    def _on_step(self):
        progress = self.num_timesteps / self.total_steps
        # 커리큘럼: 6M 기준으로 조정
        if progress < 0.20:
            bear_ratio, phase = 0.5, "1-하락집중(50%)"
        elif progress < 0.50:
            bear_ratio, phase = 0.3, "2-혼합(30%)"
        else:
            bear_ratio, phase = 0.0, "3-자연"
        self.training_env.env_method("set_bear_ratio", bear_ratio)
        new_ent = max(0.01 - 0.005 * progress, 0.003)
        self.model.ent_coef = new_ent
        if self.num_timesteps % self.check_freq < N_ENVS:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            eta_m = int(elapsed / max(self.num_timesteps, 1) * (self.total_steps - self.num_timesteps) / 60)
            mean_r = "N/A"
            n_eps = len(self.model.ep_info_buffer)
            if n_eps > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                mean_r = f"{np.mean(rewards):+.3f}"
                if np.mean(rewards) > self.best_reward:
                    self.best_reward = np.mean(rewards)
            log.info(f"[{datetime.now().strftime('%H:%M:%S')}] "
                     f"{self.num_timesteps:>9,}/{self.total_steps:,} ({progress*100:5.1f}%) | "
                     f"보상: {mean_r} | {phase} | ETA: ~{eta_m}분")
        return True

    def _on_training_end(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log.info(f"\n학습 완료: {elapsed/60:.1f}분 소요 | 최고 보상: {self.best_reward:+.3f}")


def train():
    data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    cutoff = pd.Timestamp(TRAIN_END)
    split_idx = df[df["time"] <= cutoff].index[-1] + 1
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    log.info(f"학습: {len(train_df):,}캔들 | 테스트: {len(test_df):,}캔들")

    def make_env(seed):
        def _init():
            env = ETHTradingEnvV51Exp02(
                train_df, initial_balance=10000.0, leverage=LEVERAGE,
                fee_rate=0.0004, window_size=20, min_hold_steps=4,
                max_episode_len=2000, max_drawdown=0.15,
                cooldown_steps=8, curriculum=True,
            )
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_env(seed=i * 42) for i in range(N_ENVS)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"ppo_eth_{INTERVAL}"

    # exp02 동일 네트워크, 6M 스텝
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
        callback=CurriculumCallback(total_steps=TOTAL_STEPS, check_freq=20000),
        progress_bar=False,
    )

    model.save(str(model_path))
    meta = {
        "experiment": EXP_NAME,
        "description": "학습량 2배 (6M 스텝, exp02 환경 동일)",
        "changes": ["TOTAL_STEPS: 3M → 6M", "커리큘럼 비율 조정 (20/50/자연)"],
        "interval": INTERVAL, "leverage": LEVERAGE,
        "steps": TOTAL_STEPS, "n_envs": N_ENVS,
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info(f"\n모델 저장: {model_path}.zip")
    return model, test_df


def backtest(test_df):
    log.info("\n" + "=" * 60)
    log.info(f"  백테스트: {EXP_NAME} vs exp02 vs v5")
    log.info("=" * 60)
    bh_ret = (test_df["close"].iloc[-1] - test_df["close"].iloc[20]) / test_df["close"].iloc[20] * 100
    results = {}

    v5_path = PROJECT_ROOT / "rl" / "models" / "v5" / "ppo_eth_30m.zip"
    if v5_path.exists():
        env_v5 = ETHTradingEnvV5(test_df, initial_balance=10000.0, leverage=LEVERAGE,
                                  window_size=20, min_hold_steps=4,
                                  max_episode_len=len(test_df)+100, max_drawdown=1.0, curriculum=False)
        results["v5"] = _run_backtest(env_v5, v5_path)

    exp02_path = EXP_DIR / "models" / "exp02" / f"ppo_eth_{INTERVAL}.zip"
    if exp02_path.exists():
        env_exp02 = ETHTradingEnvV51Exp02(test_df, initial_balance=10000.0, leverage=LEVERAGE,
                                           window_size=20, min_hold_steps=4,
                                           max_episode_len=len(test_df)+100, max_drawdown=1.0,
                                           cooldown_steps=8, curriculum=False)
        results["exp02"] = _run_backtest(env_exp02, exp02_path)

    exp_path = MODEL_DIR / f"ppo_eth_{INTERVAL}.zip"
    if exp_path.exists():
        env_exp = ETHTradingEnvV51Exp02(test_df, initial_balance=10000.0, leverage=LEVERAGE,
                                         window_size=20, min_hold_steps=4,
                                         max_episode_len=len(test_df)+100, max_drawdown=1.0,
                                         cooldown_steps=8, curriculum=False)
        results[EXP_NAME] = _run_backtest(env_exp, exp_path)

    for name, r in results.items():
        s = r["stats"]
        dist = Counter(r["actions"])
        total = len(r["actions"])
        names = {0: "관망", 1: "롱", 2: "숏", 3: "청산"}
        log.info(f"\n{'='*52}\n  {name}\n{'='*52}")
        log.info(f"  수익률: {s['return']:+.2f}% | MDD: {s['mdd']:.2f}% | "
                 f"거래: {s['trades']}회 | 승률: {s['win_rate']:.1%}")
        log.info(f"  행동: " + " / ".join(
            f"{names[k]}:{v}({v/total:.0%})" for k, v in sorted(dist.items())))

    if results:
        _plot(test_df, results, bh_ret)
    return results


def _run_backtest(env, model_path):
    model = PPO.load(str(model_path), device="cpu")
    obs, _ = env.reset()
    balances = [env.initial_balance]
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
        balances.append(info["balance"])
        actions.append(int(action))
    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    return {
        "balances": balances, "actions": actions,
        "stats": {"final": balances[-1], "return": (balances[-1]-10000)/10000*100,
                  "mdd": mdd, "trades": info["total_trades"], "win_rate": info["win_rate"]},
    }


def _plot(test_df, results, bh_ret):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle(f"{EXP_NAME} (6M steps) vs exp02 vs v5", fontsize=13, fontweight="bold")
    colors = {"v5": "#1565C0", "exp02": "#E65100", EXP_NAME: "#2E7D32"}
    for name, r in results.items():
        s = r["stats"]
        ax.plot(r["balances"], color=colors.get(name, "#666"), lw=1.8,
                label=f"{name}  {s['return']:+.1f}% (MDD {s['mdd']:.1f}%)")
    ax.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"{EXP_NAME}_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"\n그래프 저장: {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-only", action="store_true")
    args = parser.parse_args()
    if args.backtest_only:
        data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
        df = pd.read_csv(data_path, parse_dates=["time"])
        test_df = df[df["time"] > pd.Timestamp(TRAIN_END)].reset_index(drop=True)
        backtest(test_df)
    else:
        model, test_df = train()
        backtest(test_df)

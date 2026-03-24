#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# BTC 30m PPO 학습 — exp14 설정 기반 (gamma=0.975)
# ETH 환경(env_v51_exp02) 재사용, BTC 데이터

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_v51_exp02 import ETHTradingEnvV51Exp02  # BTC에도 동일 환경 사용

SEED = 42
TOTAL_STEPS = 3_000_000
SAVE_DIR = Path(__file__).parent.parent / "models" / "btc_exp14"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class CurriculumCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.phase = 0
    def _on_step(self):
        progress = self.num_timesteps / TOTAL_STEPS
        if progress > 0.6 and self.phase < 2:
            self.phase = 2
            for env in self.training_env.envs:
                env.set_bear_ratio(0.0)
            print(f"[{self.num_timesteps:,}] Phase 3: 자연분포")
        elif progress > 0.3 and self.phase < 1:
            self.phase = 1
            for env in self.training_env.envs:
                env.set_bear_ratio(0.3)
            print(f"[{self.num_timesteps:,}] Phase 2: 하락장 30%")
        return True


class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best_reward = -999
    def _on_step(self):
        if self.num_timesteps % 50_000 == 0:
            rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            if rew > self.best_reward:
                self.best_reward = rew
            print(f"[{self.num_timesteps:>8,}] reward={rew:+.3f} best={self.best_reward:+.3f}")
        return True


def main():
    data_path = PROJECT_ROOT / "rl" / "btc_30m.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    train_df = df[df["time"] <= pd.Timestamp("2025-09-01")].reset_index(drop=True)
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)
    print(f"BTC 학습: {len(train_df)}캔들, 테스트: {len(test_df)}캔들")

    def make_env(seed_offset=0):
        def _init():
            env = ETHTradingEnvV51Exp02(
                train_df, initial_balance=10000.0, leverage=3,
                window_size=20, min_hold_steps=4, max_episode_len=2000,
                max_drawdown=0.15, cooldown_steps=8, curriculum=True
            )
            env.reset(seed=SEED + seed_offset)
            return env
        return _init

    env = DummyVecEnv([make_env(i) for i in range(4)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.975, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]),
                           activation_fn=torch.nn.ReLU),
        seed=SEED, device="cuda", verbose=0,
    )

    print(f"학습 시작: {TOTAL_STEPS:,} 스텝, gamma=0.975, GPU")
    model.learn(total_timesteps=TOTAL_STEPS,
                callback=[CurriculumCallback(), LogCallback()])

    save_path = SAVE_DIR / "ppo_btc_30m"
    model.save(str(save_path))
    print(f"\n모델 저장: {save_path}.zip")

    # 백테스트
    test_env = ETHTradingEnvV51Exp02(
        test_df, initial_balance=10000.0, leverage=3,
        window_size=20, min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = test_env.reset()
    done = False; balances = [10000.0]; actions = []
    while not done:
        ac, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = test_env.step(int(ac))
        done = term or trunc
        balances.append(info["balance"]); actions.append(int(ac))

    from collections import Counter
    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    dist = Counter(actions); total = len(actions)
    nm = {0:"관망", 1:"롱", 2:"숏", 3:"청산"}
    print(f"\n=== BTC 백테스트 ===")
    print(f"수익률: {(balances[-1]-10000)/10000*100:+.1f}%")
    print(f"MDD: {mdd:.1f}%")
    print(f"거래: {info['total_trades']}회, 승률: {info['win_rate']:.1%}")
    print(f"행동: " + " / ".join(f"{nm[k]}:{v}({v/total:.0%})" for k, v in sorted(dist.items())))


if __name__ == "__main__":
    main()

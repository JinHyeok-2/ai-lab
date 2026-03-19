#!/usr/bin/env python3
# RL v6 학습 스크립트 — 3x 레버리지, VecNormalize, 4 병렬 env
# 실행: python trading/rl/train_v6.py --steps 1000000

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import logging
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from rl.data import fetch_eth_data, save_data, load_data
from rl.env import ETHTradingEnv

VERSION  = "v6"
INTERVAL = "30m"
N_ENVS   = 4
LEVERAGE = 3

LOG_PATH = Path(__file__).parent / "train_v6.log"

# 로그 파일 설정 (파일 + 콘솔 동시 출력)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

class TrainCallback(BaseCallback):
    def __init__(self, check_freq: int = 10000, total_steps: int = 1_000_000):
        super().__init__()
        self.check_freq  = check_freq
        self.total_steps = total_steps
        self.start_time  = None

    def _on_training_start(self):
        self.start_time = datetime.now()
        log.info(f"[{self.start_time.strftime('%H:%M:%S')}] 학습 시작")

    def _on_step(self) -> bool:
        # num_timesteps = 실제 누적 타임스텝 (n_envs 반영됨)
        if self.num_timesteps % (self.check_freq * 4) < 4:
            now     = datetime.now()
            elapsed = (now - self.start_time).total_seconds()
            pct     = self.num_timesteps / self.total_steps * 100
            eta_s   = elapsed / max(self.num_timesteps, 1) * (self.total_steps - self.num_timesteps)
            eta_m   = int(eta_s // 60)

            mean_r = "N/A"
            if len(self.model.ep_info_buffer) > 0:
                mean_r = f"{np.mean([ep['r'] for ep in self.model.ep_info_buffer]):+.4f}"

            log.info(
                f"[{now.strftime('%H:%M:%S')}] "
                f"스텝 {self.num_timesteps:>9,}/{self.total_steps:,} ({pct:5.1f}%) | "
                f"평균보상: {mean_r} | "
                f"ETA: ~{eta_m}분"
            )
        return True


def make_env(df, seed: int = 0):
    def _init():
        env = ETHTradingEnv(
            df,
            initial_balance=10000.0,
            leverage=LEVERAGE,
            fee_rate=0.0008,   # 수수료 2배 → 과다매매 억제
            window_size=20,
            min_hold_steps=6,  # 최소 보유 6캔들(3시간) 강제
        )
        env.reset(seed=seed)
        return env
    return _init


def train(total_timesteps: int = 1_000_000, split: float = 0.8):
    model_dir  = Path(__file__).parent / "models" / VERSION
    model_path = model_dir / f"ppo_eth_{INTERVAL}"
    vec_path   = model_dir / f"vecnormalize_{INTERVAL}.pkl"

    # 1. 데이터 준비
    data_path = Path(__file__).parent / f"eth_{INTERVAL}.csv"
    if data_path.exists():
        log.info(f"기존 데이터 로드: eth_{INTERVAL}.csv")
        df = load_data(INTERVAL)
    else:
        log.info(f"API로 데이터 수집 ({INTERVAL})...")
        df = fetch_eth_data(interval=INTERVAL)
        save_data(df, INTERVAL)

    # 2. 학습/테스트 분할
    split_idx = int(len(df) * split)
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    test_df   = df.iloc[split_idx:].reset_index(drop=True)
    log.info(f"학습: {len(train_df)}개 ({train_df['time'].iloc[0].date()} ~ {train_df['time'].iloc[-1].date()})")
    log.info(f"테스트: {len(test_df)}개 ({test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()})")
    log.info(f"레버리지: {LEVERAGE}x | 병렬 env: {N_ENVS}개\n")

    # 3. 병렬 환경 + VecNormalize
    env = SubprocVecEnv([make_env(train_df, seed=i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # 4. PPO 모델
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_path.with_suffix(".zip").exists() and vec_path.exists():
        print(f"기존 모델 로드 후 추가 학습: {model_path}.zip")
        env = VecNormalize.load(str(vec_path), env.venv)
        model = PPO.load(str(model_path), env=env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.003,   # 낮춰서 과도한 탐험 억제
            verbose=0,
        )

    # 5. 학습
    log.info(f"학습 시작 [{INTERVAL}] — {total_timesteps:,} 스텝\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TrainCallback(check_freq=10000, total_steps=total_timesteps),
        progress_bar=False,
        reset_num_timesteps=False,
    )

    # 6. 저장 (모델 + VecNormalize 통계 + 메타)
    model.save(str(model_path))
    env.save(str(vec_path))

    meta = {
        "version":    VERSION,
        "interval":   INTERVAL,
        "leverage":   LEVERAGE,
        "n_envs":     N_ENVS,
        "steps":      total_timesteps,
        "vecnorm":    True,
        "train_start": str(train_df["time"].iloc[0].date()),
        "train_end":   str(train_df["time"].iloc[-1].date()),
        "test_start":  str(test_df["time"].iloc[0].date()),
        "test_end":    str(test_df["time"].iloc[-1].date()),
        "train_candles": len(train_df),
        "test_candles":  len(test_df),
    }
    with open(model_dir / f"meta_{INTERVAL}.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"\n✅ 모델 저장: {model_path}.zip")
    log.info(f"✅ VecNormalize: {vec_path}")
    log.info(f"백테스트: python trading/rl/backtest.py --interval {INTERVAL} --version {VERSION}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000, help="학습 스텝 수")
    parser.add_argument("--split", type=float, default=0.8, help="학습 비율 (기본 0.8)")
    args = parser.parse_args()

    train(total_timesteps=args.steps, split=args.split)

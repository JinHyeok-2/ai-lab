#!/usr/bin/env python3
# v4.1 학습 스크립트 — 레버리지 3배, 피처 확장, 추세 보상, 2M 스텝
# 실행: python trading/rl/train_v41.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import logging
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from rl.data import fetch_eth_data, save_data, load_data
from rl.env_v41 import ETHTradingEnvV41

VERSION   = "v41"
INTERVAL  = "30m"
LEVERAGE  = 3
TRAIN_END = "2025-09-01"
DATA_DAYS = 820   # 2024-01-01 포함 위해 820일치

LOG_PATH = Path(__file__).parent / "train_v41.log"

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
    def __init__(self, check_freq: int = 20000, total_steps: int = 2_000_000):
        super().__init__()
        self.check_freq  = check_freq
        self.total_steps = total_steps
        self.start_time  = None

    def _on_training_start(self):
        self.start_time = datetime.now()
        log.info(f"[{self.start_time.strftime('%H:%M:%S')}] 학습 시작")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq < 1:
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
                f"평균보상: {mean_r} | ETA: ~{eta_m}분"
            )
        return True


def train(total_timesteps: int = 2_000_000):
    import pandas as pd

    model_dir  = Path(__file__).parent / "models" / VERSION
    model_path = model_dir / f"ppo_eth_{INTERVAL}"

    # 1. 데이터 준비 (820일치 재수집 or 로드)
    data_path = Path(__file__).parent / f"eth_{INTERVAL}_v41.csv"
    if data_path.exists():
        log.info(f"기존 데이터 로드: eth_{INTERVAL}_v41.csv")
        df = pd.read_csv(data_path, parse_dates=["time"])
    else:
        log.info(f"API로 데이터 수집 ({DATA_DAYS}일치)...")
        df = fetch_eth_data(interval=INTERVAL, days=DATA_DAYS)
        df.to_csv(data_path, index=False)
        log.info(f"저장: {data_path}")

    # 2. 학습/테스트 분할 (TRAIN_END 기준)
    cutoff    = pd.Timestamp(TRAIN_END)
    split_idx = df[df["time"] <= cutoff].index[-1] + 1
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    test_df   = df.iloc[split_idx:].reset_index(drop=True)

    log.info(f"학습: {len(train_df)}개 ({train_df['time'].iloc[0].date()} ~ {train_df['time'].iloc[-1].date()})")
    log.info(f"테스트: {len(test_df)}개 ({test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()})")
    log.info(f"레버리지: {LEVERAGE}x | 피처: 10개 | 스텝: {total_timesteps:,}\n")

    # 3. 환경 생성 (DummyVecEnv, VecNormalize 없음)
    env = DummyVecEnv([lambda: ETHTradingEnvV41(
        train_df,
        initial_balance=10000.0,
        leverage=LEVERAGE,
        fee_rate=0.0004,
        window_size=20,
        min_hold_steps=4,
    )])

    # 4. PPO 모델 (v4와 동일 구조, ent_coef만 조정)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_path.with_suffix(".zip").exists():
        log.info(f"기존 모델 이어서 학습: {model_path}.zip")
        model = PPO.load(str(model_path), env=env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,   # v4(0.008)보다 낮게 — 안정적 학습
            verbose=0,
        )

    # 5. 학습
    model.learn(
        total_timesteps=total_timesteps,
        callback=TrainCallback(check_freq=20000, total_steps=total_timesteps),
        progress_bar=False,
        reset_num_timesteps=False,
    )

    # 6. 저장
    model.save(str(model_path))

    meta = {
        "version":    VERSION,
        "interval":   INTERVAL,
        "leverage":   LEVERAGE,
        "steps":      total_timesteps,
        "train_end":  TRAIN_END,
        "train_start": str(train_df["time"].iloc[0].date()),
        "train_end_actual": str(train_df["time"].iloc[-1].date()),
        "test_start":  str(test_df["time"].iloc[0].date()),
        "test_end":    str(test_df["time"].iloc[-1].date()),
        "train_candles": len(train_df),
        "test_candles":  len(test_df),
        "features": 10,
        "env": "ETHTradingEnvV41",
    }
    with open(model_dir / f"meta_{INTERVAL}.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"\n✅ 모델 저장: {model_path}.zip")
    log.info(f"백테스트: python trading/rl/compare_v4_v41.py")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    args = parser.parse_args()
    train(total_timesteps=args.steps)

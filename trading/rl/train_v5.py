#!/usr/bin/env python3
# v5 학습 — 에피소드 분할 + 커리큘럼 러닝 + 보상 리스케일
# 실행: conda activate APCC && python trading/rl/train_v5.py [--steps 3000000]
#
# v4.1 대비 핵심 변경:
# - DummyVecEnv × 4 병렬 (v4.1: × 1) → 에피소드 4배 빠르게 수집
# - max_episode_len=2000 → 에피소드 다수 완료 → ep_info_buffer 정상 동작
# - 커리큘럼 3단계: 하락장 70% → 40% → 자연분포
# - 엔트로피 감소: 0.01 → 0.003 (탐험→수렴)
# - 학습률 감소: 3e-4 → 9e-5
# - 네트워크 확장: [256, 128] (기본 [64, 64])

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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from rl.data import fetch_eth_data
from rl.env_v5 import ETHTradingEnvV5

VERSION    = "v5"
INTERVAL   = "30m"
LEVERAGE   = 3
TRAIN_END  = "2025-09-01"
DATA_DAYS  = 820
N_ENVS     = 4
TOTAL_STEPS = 3_000_000

LOG_PATH = Path(__file__).parent / "train_v5.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """커리큘럼 러닝 + 엔트로피 감소 + 학습 모니터링"""

    def __init__(self, total_steps: int, check_freq: int = 10000):
        super().__init__()
        self.total_steps = total_steps
        self.check_freq  = check_freq
        self.start_time  = None
        self.best_reward = -np.inf

    def _on_training_start(self):
        self.start_time = datetime.now()
        log.info(f"[{self.start_time.strftime('%H:%M:%S')}] v5 학습 시작 ({self.total_steps:,} 스텝)")
        log.info(f"  환경: {N_ENVS}개 | 에피소드: 2000캔들 | 커리큘럼: ON\n")

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_steps

        # ── 커리큘럼 페이즈 ──────────────────────────────
        if progress < 0.25:
            bear_ratio = 0.7
            phase = "1-하락집중"
        elif progress < 0.65:
            bear_ratio = 0.4
            phase = "2-혼합"
        else:
            bear_ratio = 0.0
            phase = "3-자연"

        self.training_env.env_method("set_bear_ratio", bear_ratio)

        # ── 엔트로피 감소: 0.01 → 0.003 ─────────────────
        new_ent = max(0.01 - 0.007 * progress, 0.003)
        self.model.ent_coef = new_ent

        # ── 주기적 로깅 ─────────────────────────────────
        if self.num_timesteps % self.check_freq < N_ENVS:
            now     = datetime.now()
            elapsed = (now - self.start_time).total_seconds()
            pct     = progress * 100
            eta_m   = int(elapsed / max(self.num_timesteps, 1) *
                          (self.total_steps - self.num_timesteps) / 60)

            # ep_info_buffer 에서 보상 추출
            mean_r = "N/A"
            n_eps  = len(self.model.ep_info_buffer)
            if n_eps > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                mean_r  = f"{np.mean(rewards):+.3f}"
                # 최고 보상 갱신 체크
                current_mean = np.mean(rewards)
                if current_mean > self.best_reward:
                    self.best_reward = current_mean

            log.info(
                f"[{now.strftime('%H:%M:%S')}] "
                f"{self.num_timesteps:>9,}/{self.total_steps:,} ({pct:5.1f}%) | "
                f"보상: {mean_r} (ep:{n_eps}) | ent: {new_ent:.4f} | "
                f"{phase} | ETA: ~{eta_m}분"
            )

        return True

    def _on_training_end(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log.info(f"\n학습 완료: {elapsed/60:.1f}분 소요")
        if self.best_reward > -np.inf:
            log.info(f"최고 평균 보상: {self.best_reward:+.3f}")


def train(total_timesteps: int = TOTAL_STEPS):
    import pandas as pd

    model_dir  = Path(__file__).parent / "models" / VERSION
    model_path = model_dir / f"ppo_eth_{INTERVAL}"

    # 1. 데이터 로드 (v4.1과 동일 데이터셋 사용)
    data_path = Path(__file__).parent / f"eth_{INTERVAL}_v41.csv"
    if data_path.exists():
        log.info(f"기존 데이터 로드: {data_path.name}")
        df = pd.read_csv(data_path, parse_dates=["time"])
    else:
        log.info(f"API 데이터 수집 ({DATA_DAYS}일치)...")
        df = fetch_eth_data(interval=INTERVAL, days=DATA_DAYS)
        df.to_csv(data_path, index=False)

    # 2. 학습/테스트 분할 (2025-09-01 기준)
    cutoff    = pd.Timestamp(TRAIN_END)
    split_idx = df[df["time"] <= cutoff].index[-1] + 1
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    test_df   = df.iloc[split_idx:].reset_index(drop=True)

    log.info(f"학습: {len(train_df):,}캔들 ({train_df['time'].iloc[0].date()} ~ {train_df['time'].iloc[-1].date()})")
    log.info(f"테스트: {len(test_df):,}캔들 ({test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()})")
    log.info(f"레버리지: {LEVERAGE}x | 피처: 13개 | 환경: {N_ENVS}개 | 스텝: {total_timesteps:,}")

    # 3. 병렬 환경 (DummyVecEnv × 4, Monitor 래핑)
    def make_env(seed: int):
        def _init():
            env = ETHTradingEnvV5(
                train_df,
                initial_balance=10000.0,
                leverage=LEVERAGE,
                fee_rate=0.0004,
                window_size=20,
                min_hold_steps=4,
                max_episode_len=2000,
                max_drawdown=0.15,
                curriculum=True,
            )
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_env(seed=i * 42) for i in range(N_ENVS)])

    # 4. PPO 모델
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_path.with_suffix(".zip").exists():
        log.info(f"기존 모델 이어서 학습: {model_path}.zip")
        model = PPO.load(str(model_path), env=env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=lambda p: 3e-4 * (0.3 + 0.7 * p),  # 3e-4 → 9e-5
            n_steps=2048,
            batch_size=256,       # v4.1: 128
            n_epochs=10,
            gamma=0.995,          # v4.1: 0.99 — 장기 보상 더 중시
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # 초기값 (콜백에서 0.003까지 감소)
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
            ),
        )

    # 5. 학습
    log.info("")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CurriculumCallback(
            total_steps=total_timesteps,
            check_freq=10000,
        ),
        progress_bar=False,
        reset_num_timesteps=False,
    )

    # 6. 저장
    model.save(str(model_path))

    bear_count = 0
    try:
        inner = env.envs[0]
        # Monitor 래핑된 경우 unwrap
        while hasattr(inner, "env"):
            inner = inner.env
        bear_count = len(inner.bear_starts) if hasattr(inner, "bear_starts") else 0
    except Exception:
        pass

    meta = {
        "version":      VERSION,
        "interval":     INTERVAL,
        "leverage":     LEVERAGE,
        "steps":        total_timesteps,
        "train_end":    TRAIN_END,
        "train_start":  str(train_df["time"].iloc[0].date()),
        "train_end_actual": str(train_df["time"].iloc[-1].date()),
        "test_start":   str(test_df["time"].iloc[0].date()),
        "test_end":     str(test_df["time"].iloc[-1].date()),
        "train_candles": len(train_df),
        "test_candles":  len(test_df),
        "features":     13,
        "env":          "ETHTradingEnvV5",
        "n_envs":       N_ENVS,
        "max_episode_len": 2000,
        "curriculum":   True,
        "bear_segments": bear_count,
        "net_arch":     "pi=[256,128] vf=[256,128]",
        "gamma":        0.995,
        "batch_size":   256,
    }
    with open(model_dir / f"meta_{INTERVAL}.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"\n✅ 모델 저장: {model_path}.zip")
    log.info(f"하락장 구간: {bear_count}개 감지됨")
    log.info(f"백테스트: conda activate APCC && python trading/rl/compare_v5.py")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()
    train(total_timesteps=args.steps)

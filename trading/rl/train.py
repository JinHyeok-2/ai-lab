#!/usr/bin/env python3
# 강화학습 에이전트 학습 — 멀티 타임프레임 + 기간 지정 지원
# 실행 예시:
#   python trading/rl/train.py --interval 30m --steps 1000000 --version v2
#   python trading/rl/train.py --interval 30m --steps 1000000 --version v3 --split 0.6
#   python trading/rl/train.py --interval 30m --steps 1000000 --version v4 --train-end 2025-09-01

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from rl.data import fetch_eth_data, save_data, load_data
from rl.env import ETHTradingEnv

# 타임프레임별 최소 보유 캔들 설정
MIN_HOLD = {"15m": 5, "30m": 4, "1h": 4, "2h": 3, "4h": 3, "1d": 2}

# ── 학습 콜백 ────────────────────────────────────────────────────────
class TrainCallback(BaseCallback):
    def __init__(self, check_freq: int = 10000):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                print(f"  스텝 {self.n_calls:>8,} | 평균 보상: {mean_r:+.4f}")
        return True

# ── 메인 학습 ────────────────────────────────────────────────────────
def train(interval: str = "1h", total_timesteps: int = 500_000,
          version: str = "v2", split: float = 0.8, train_end: str = None):
    data_path  = Path(__file__).parent / f"eth_{interval}.csv"
    model_path = Path(__file__).parent / "models" / version / f"ppo_eth_{interval}"

    # 1. 데이터 준비
    if data_path.exists():
        print(f"기존 데이터 로드: eth_{interval}.csv")
        df = load_data(interval)
    else:
        print(f"실거래 API로 데이터 수집 ({interval})...")
        df = fetch_eth_data(interval=interval)
        save_data(df, interval)

    # 2. 학습/테스트 분할 (train-end 날짜 우선, 없으면 split 비율)
    if train_end:
        cutoff = pd.Timestamp(train_end)
        split_idx = df[df["time"] <= cutoff].index[-1] + 1
        print(f"날짜 기준 분할: ~ {train_end}")
    else:
        split_idx = int(len(df) * split)
        print(f"비율 기준 분할: {split:.0%}")

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df  = df.iloc[split_idx:].reset_index(drop=True)
    print(f"학습: {len(train_df)}개 ({train_df['time'].iloc[0].date()} ~ {train_df['time'].iloc[-1].date()})")
    print(f"테스트: {len(test_df)}개 ({test_df['time'].iloc[0].date()} ~ {test_df['time'].iloc[-1].date()})\n")

    # 3. 환경 생성
    min_hold = MIN_HOLD.get(interval, 3)
    env = DummyVecEnv([lambda: ETHTradingEnv(
        train_df,
        initial_balance=10000.0,
        leverage=5,
        fee_rate=0.0004,
        window_size=20,
        min_hold_steps=min_hold,
    )])

    # 타임프레임별 PPO 하이퍼파라미터
    HYP = {
        "15m": dict(learning_rate=3e-4, gamma=0.99,  ent_coef=0.01,  n_steps=2048, batch_size=128),
        "30m": dict(learning_rate=3e-4, gamma=0.99,  ent_coef=0.008, n_steps=2048, batch_size=128),
        "1h":  dict(learning_rate=2e-4, gamma=0.995, ent_coef=0.005, n_steps=2048, batch_size=128),
        "2h":  dict(learning_rate=2e-4, gamma=0.995, ent_coef=0.005, n_steps=2048, batch_size=128),
        "4h":  dict(learning_rate=2e-4, gamma=0.995, ent_coef=0.005, n_steps=2048, batch_size=128),
        "1d":  dict(learning_rate=1e-4, gamma=0.999, ent_coef=0.003, n_steps=1024, batch_size=64),
    }
    hyp = HYP.get(interval, HYP["1h"])

    # 4. PPO 모델 (기존 모델 있으면 이어서 학습)
    if model_path.with_suffix(".zip").exists():
        print(f"기존 모델 로드 후 추가 학습: {model_path}.zip")
        model = PPO.load(str(model_path), env=env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=hyp["learning_rate"],
            n_steps=hyp["n_steps"],
            batch_size=hyp["batch_size"],
            n_epochs=10,
            gamma=hyp["gamma"],
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=hyp["ent_coef"],
            verbose=0,
            tensorboard_log=None,
        )

    # 5. 학습
    print(f"학습 시작 [{interval}] — {total_timesteps:,} 스텝 (최소보유: {min_hold} 캔들)\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TrainCallback(check_freq=10000),
        progress_bar=False,
        reset_num_timesteps=False,
    )

    # 6. 저장 (모델 + 메타데이터)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    meta = {
        "interval":    interval,
        "version":     version,
        "steps":       total_timesteps,
        "split":       split if not train_end else None,
        "train_end":   train_end,
        "train_start": str(train_df["time"].iloc[0].date()),
        "train_end_actual": str(train_df["time"].iloc[-1].date()),
        "test_start":  str(test_df["time"].iloc[0].date()),
        "test_end":    str(test_df["time"].iloc[-1].date()),
        "train_candles": len(train_df),
        "test_candles":  len(test_df),
    }
    meta_path = model_path.parent / f"meta_{interval}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n모델 저장: {model_path}.zip")
    print(f"메타데이터: {meta_path}")
    print(f"백테스트: python trading/rl/backtest.py --interval {interval} --version {version}")
    return model, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval",   default="1h",  choices=["15m","30m","1h","2h","4h","1d"],
                        help="캔들 타임프레임")
    parser.add_argument("--steps",      type=int, default=500_000,
                        help="학습 스텝 수")
    parser.add_argument("--version",    default="v2",
                        help="모델 버전 (예: v2, v3)")
    parser.add_argument("--split",      type=float, default=0.8,
                        help="학습/테스트 비율 (기본: 0.8 = 80%% 학습)")
    parser.add_argument("--train-end",  default=None,
                        help="학습 종료 날짜 (예: 2025-09-01) — 지정 시 --split 무시")
    args = parser.parse_args()

    train(interval=args.interval, total_timesteps=args.steps,
          version=args.version, split=args.split, train_end=args.train_end)

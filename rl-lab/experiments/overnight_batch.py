#!/usr/bin/env python3
# 야간 GPU 병렬 학습 배치 — BTC 시드 3종 + ETH 시드 1종
# 동시 4개 GPU 학습 (각 ~300MB VRAM, 총 ~1.2GB / 24GB)

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import time
import json
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_v51_exp02 import ETHTradingEnvV51Exp02

TOTAL_STEPS = 3_000_000
LOG_DIR = Path(__file__).parent.parent / "experiments"
STATUS_FILE = LOG_DIR / "overnight_status.json"


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
        elif progress > 0.3 and self.phase < 1:
            self.phase = 1
            for env in self.training_env.envs:
                env.set_bear_ratio(0.3)
        return True


class ProgressCallback(BaseCallback):
    """진행률 + 보상 추적 (파일 기반)"""
    def __init__(self, job_name, status_file):
        super().__init__()
        self.job_name = job_name
        self.status_file = status_file
        self.best_reward = -999
    def _on_step(self):
        if self.num_timesteps % 100_000 == 0:
            rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            if rew > self.best_reward:
                self.best_reward = rew
            # 상태 파일 업데이트
            try:
                status = json.loads(self.status_file.read_text()) if self.status_file.exists() else {}
            except:
                status = {}
            status[self.job_name] = {
                "steps": self.num_timesteps,
                "progress": f"{self.num_timesteps/TOTAL_STEPS*100:.0f}%",
                "reward": f"{rew:+.3f}",
                "best_reward": f"{self.best_reward:+.3f}",
                "updated": datetime.now().strftime("%H:%M:%S"),
            }
            self.status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False))
        return True


def backtest(model, test_df, leverage=3):
    """백테스트 실행"""
    env = ETHTradingEnvV51Exp02(
        test_df, initial_balance=10000.0, leverage=leverage,
        window_size=20, min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    done = False; balances = [10000.0]; actions = []
    while not done:
        ac, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(ac))
        done = term or trunc
        balances.append(info["balance"]); actions.append(int(ac))
    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    dist = Counter(actions); total = len(actions)
    return {
        "return": (balances[-1]-10000)/10000*100,
        "mdd": mdd,
        "trades": info["total_trades"],
        "win_rate": info["win_rate"],
        "actions": {k: v for k, v in sorted(dist.items())},
        "total_actions": total,
    }


def train_one(job):
    """단일 모델 학습 + 백테스트"""
    name = job["name"]
    asset = job["asset"]
    seed = job["seed"]
    save_dir = Path(__file__).parent.parent / "models" / name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{name}] 학습 시작 — {asset}, seed={seed}, GPU")
    print(f"{'='*60}")

    # 데이터 로드
    if asset == "BTC":
        data_path = PROJECT_ROOT / "rl" / "btc_30m.csv"
    else:
        data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    train_df = df[df["time"] <= pd.Timestamp("2025-09-01")].reset_index(drop=True)
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)
    print(f"  [{name}] 학습: {len(train_df)}캔들, 테스트: {len(test_df)}캔들")

    def make_env(seed_offset=0):
        def _init():
            env = ETHTradingEnvV51Exp02(
                train_df, initial_balance=10000.0, leverage=3,
                window_size=20, min_hold_steps=4, max_episode_len=2000,
                max_drawdown=0.15, cooldown_steps=8, curriculum=True
            )
            env.reset(seed=seed + seed_offset)
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
        seed=seed, device="cuda", verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS,
                callback=[CurriculumCallback(),
                          ProgressCallback(name, STATUS_FILE)])
    elapsed = time.time() - t0

    # 저장
    save_path = save_dir / f"ppo_{asset.lower()}_30m"
    model.save(str(save_path))
    print(f"  [{name}] 모델 저장: {save_path}.zip ({elapsed/60:.0f}분)")

    # 백테스트
    result = backtest(model, test_df)
    nm = {0:"관망", 1:"롱", 2:"숏", 3:"청산"}
    print(f"\n  [{name}] === 백테스트 ===")
    print(f"  수익률: {result['return']:+.1f}%")
    print(f"  MDD: {result['mdd']:.1f}%")
    print(f"  거래: {result['trades']}회, 승률: {result['win_rate']:.1%}")
    acts = result['actions']
    total = result['total_actions']
    print(f"  행동: " + " / ".join(f"{nm[k]}:{v}({v/total:.0%})" for k, v in sorted(acts.items())))

    # 결과 기록
    result["name"] = name
    result["asset"] = asset
    result["seed"] = seed
    result["elapsed_min"] = round(elapsed / 60, 1)

    # 상태 업데이트
    try:
        status = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
    except:
        status = {}
    status[name] = {
        "status": "DONE",
        "return": f"{result['return']:+.1f}%",
        "mdd": f"{result['mdd']:.1f}%",
        "trades": result["trades"],
        "win_rate": f"{result['win_rate']:.1%}",
        "elapsed": f"{elapsed/60:.0f}분",
        "updated": datetime.now().strftime("%H:%M:%S"),
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False))

    return result


# === 학습 작업 정의 ===
JOBS = [
    {"name": "btc_seed100", "asset": "BTC", "seed": 100},
    {"name": "btc_seed200", "asset": "BTC", "seed": 200},
    {"name": "btc_seed300", "asset": "BTC", "seed": 300},
    {"name": "eth_seed800", "asset": "ETH", "seed": 800},
]


def main():
    print(f"\n{'#'*60}")
    print(f"  야간 GPU 병렬 학습 배치")
    print(f"  작업: {len(JOBS)}개, 각 {TOTAL_STEPS:,} 스텝")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # 초기 상태 파일
    STATUS_FILE.write_text(json.dumps(
        {j["name"]: {"status": "QUEUED", "updated": datetime.now().strftime("%H:%M:%S")}
         for j in JOBS}, indent=2, ensure_ascii=False))

    # 순차 실행 (GPU 메모리 안전 + CUDA 컨텍스트 충돌 방지)
    # PPO는 CPU 바운드라 순차로도 GPU 활용률 충분
    results = []
    for job in JOBS:
        try:
            r = train_one(job)
            results.append(r)
        except Exception as e:
            print(f"\n  [ERROR] {job['name']}: {e}")
            try:
                status = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
            except:
                status = {}
            status[job["name"]] = {"status": "FAILED", "error": str(e),
                                    "updated": datetime.now().strftime("%H:%M:%S")}
            STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False))

    # 최종 요약
    print(f"\n\n{'#'*60}")
    print(f"  야간 학습 완료 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"\n{'이름':<16} {'자산':<5} {'시드':<6} {'수익률':>8} {'MDD':>8} {'거래':>5} {'승률':>6} {'시간':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<16} {r['asset']:<5} {r['seed']:<6} "
              f"{r['return']:>+7.1f}% {r['mdd']:>7.1f}% {r['trades']:>5}회 "
              f"{r['win_rate']:>5.1%} {r['elapsed_min']:>5.0f}분")

    # 결과 JSON 저장
    result_file = LOG_DIR / "overnight_results.json"
    result_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n결과 저장: {result_file}")


if __name__ == "__main__":
    main()

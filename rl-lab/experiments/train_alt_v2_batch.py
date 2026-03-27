#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT 범용모델 v2 — 27코인 학습, 시드 앙상블 배치
# 시드별 순차 학습 (GPU 1장이므로 동시 불가)
# conda run -n ai-lab python train_alt_v2_batch.py
# 예상 소요: 시드당 ~2시간, 5시드 = ~10시간

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np, pandas as pd, torch, time, json
from datetime import datetime
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_alt_universal import AltUniversalTradingEnv

TOTAL_STEPS = 5_000_000  # 3M → 5M (27코인에 맞게 증가)
SEEDS = [42, 100, 200, 300, 700]  # 5개 시드
MODELS_BASE = Path(__file__).parent.parent / "models"
STATUS_FILE = Path(__file__).parent / "train_status.json"
ALT_DATA_DIR = PROJECT_ROOT / "rl" / "alt_30m"
TRAIN_END = "2025-09-01"


def load_all_datasets():
    """27종 알트코인 데이터 로드"""
    train_datasets, test_datasets, names = [], [], []
    for csv_path in sorted(ALT_DATA_DIR.glob("*_30m.csv")):
        df = pd.read_csv(csv_path, parse_dates=["time"])
        name = csv_path.stem.replace("_30m", "").upper()
        train_df = df[df["time"] <= pd.Timestamp(TRAIN_END)].reset_index(drop=True)
        test_df = df[df["time"] > pd.Timestamp(TRAIN_END)].reset_index(drop=True)
        if len(train_df) > 1000:
            train_datasets.append(train_df)
            test_datasets.append(test_df)
            names.append(name)
    return train_datasets, test_datasets, names


class LogCallback(BaseCallback):
    def __init__(self, seed, save_dir, total_steps):
        super().__init__()
        self.seed = seed
        self.save_dir = save_dir
        self.total_steps = total_steps
        self.best = -999

    def _on_step(self):
        if self.num_timesteps % 100_000 == 0:
            rew = np.mean([e["r"] for e in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            if rew > self.best:
                self.best = rew
                best_path = self.save_dir / "best"
                best_path.mkdir(exist_ok=True)
                self.model.save(str(best_path / "ppo_alt.zip"))
            pct = self.num_timesteps / self.total_steps * 100
            print(f"  [seed{self.seed}] {self.num_timesteps:>8,} ({pct:.0f}%) reward={rew:+.3f} best={self.best:+.3f}")
            # 상태 파일
            try:
                s = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
                s[f"alt_v2_seed{self.seed}"] = {
                    "steps": self.num_timesteps,
                    "progress": f"{pct:.0f}%",
                    "reward": f"{rew:+.3f}",
                    "updated": datetime.now().strftime("%H:%M:%S")
                }
                STATUS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))
            except:
                pass
        # 체크포인트
        if self.num_timesteps in (2_000_000, 4_000_000):
            cp = self.save_dir / f"checkpoint_{self.num_timesteps//1_000_000}M"
            cp.mkdir(exist_ok=True)
            self.model.save(str(cp / "ppo_alt.zip"))
        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, total_steps):
        super().__init__()
        self.phase = 0
        self.total_steps = total_steps

    def _on_step(self):
        p = self.num_timesteps / self.total_steps
        if p > 0.6 and self.phase < 2:
            self.phase = 2
            for e in self.training_env.envs:
                e.set_bear_ratio(0.0)
        elif p > 0.3 and self.phase < 1:
            self.phase = 1
            for e in self.training_env.envs:
                e.set_bear_ratio(0.3)
        return True


def backtest_single(model, test_df, name=""):
    env = AltUniversalTradingEnv(
        [test_df], initial_balance=10000.0, leverage=2, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    balances = [10000.0]; actions = []
    while not done:
        ac, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(ac)); done = term or trunc
        balances.append(info["balance"]); actions.append(int(ac))
    arr = np.array(balances); peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (balances[-1] - 10000) / 10000 * 100
    dist = Counter(actions); total = len(actions)
    return {
        "name": name, "return_pct": ret, "mdd_pct": mdd,
        "trades": info["total_trades"],
        "win_rate": info["win_rate"] * 100,
        "long_pct": dist.get(1, 0) / total * 100,
    }


def train_seed(seed, train_datasets, test_datasets, names):
    """단일 시드 학습 + 백테스트"""
    save_dir = MODELS_BASE / f"alt_v2_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"시드 {seed} 학습 시작 — {TOTAL_STEPS:,}스텝, {len(train_datasets)}코인")
    print(f"저장: {save_dir}")
    print(f"{'='*60}")

    def make_env(offset=0):
        def _init():
            env = AltUniversalTradingEnv(
                train_datasets, leverage=2, window_size=20,
                min_hold_steps=4, max_episode_len=2000,
                max_drawdown=0.20, cooldown_steps=8, curriculum=True
            )
            env.reset(seed=seed + offset)
            return env
        return _init

    env = DummyVecEnv([make_env(i) for i in range(4)])

    model = PPO("MlpPolicy", env,
                learning_rate=3e-4, n_steps=2048, batch_size=256,
                n_epochs=10, gamma=0.975, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 128], vf=[256, 128]),
                    activation_fn=torch.nn.ReLU
                ),
                seed=seed, device="cuda", verbose=0)

    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[
            CurriculumCallback(TOTAL_STEPS),
            LogCallback(seed, save_dir, TOTAL_STEPS),
        ]
    )
    elapsed = time.time() - t0
    model.save(str(save_dir / "ppo_alt"))
    print(f"\n  [seed{seed}] 학습 완료: {elapsed/60:.0f}분")

    # 간단 백테스트 (상위 5코인)
    print(f"\n  --- seed{seed} 백테스트 (전체) ---")
    results = []
    for test_df, name in zip(test_datasets, names):
        if len(test_df) < 100:
            continue
        r = backtest_single(model, test_df, name)
        results.append(r)

    # 요약
    if results:
        avg_ret = np.mean([r["return_pct"] for r in results])
        avg_mdd = np.mean([r["mdd_pct"] for r in results])
        wins = sum(1 for r in results if r["return_pct"] > 0)
        print(f"  평균 수익: {avg_ret:+.1f}%, 평균 MDD: {avg_mdd:.1f}%, 수익코인: {wins}/{len(results)}")

        # 결과 저장
        result_file = save_dir / "backtest_results.json"
        result_file.write_text(json.dumps({
            "seed": seed, "steps": TOTAL_STEPS, "coins": len(results),
            "avg_return": round(avg_ret, 2), "avg_mdd": round(avg_mdd, 2),
            "profitable_coins": wins, "elapsed_min": round(elapsed/60, 1),
            "details": [{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in results]
        }, indent=2, ensure_ascii=False))

    return results


def main():
    print("=== ALT 범용모델 v2 — 시드 앙상블 배치 학습 ===\n")

    train_datasets, test_datasets, names = load_all_datasets()
    print(f"학습 코인: {len(names)}종 — {', '.join(names)}")
    print(f"총 학습 캔들: {sum(len(d) for d in train_datasets):,}")
    print(f"시드: {SEEDS}")
    print(f"스텝: {TOTAL_STEPS:,}/시드")
    print(f"예상 소요: ~{len(SEEDS) * 2.5:.0f}시간\n")

    all_results = {}
    for seed in SEEDS:
        results = train_seed(seed, train_datasets, test_datasets, names)
        all_results[seed] = results

    # 최종 요약
    print(f"\n{'='*60}")
    print("시드별 최종 요약")
    print(f"{'='*60}")
    print(f"{'시드':<8} {'평균수익':>10} {'평균MDD':>10} {'수익코인':>10}")
    print(f"{'-'*40}")
    for seed, results in all_results.items():
        if results:
            avg_ret = np.mean([r["return_pct"] for r in results])
            avg_mdd = np.mean([r["mdd_pct"] for r in results])
            wins = sum(1 for r in results if r["return_pct"] > 0)
            print(f"seed{seed:<4} {avg_ret:>+9.1f}% {avg_mdd:>9.1f}% {wins:>6}/{len(results)}")

    print(f"\n학습 완료! eval_alt_v2_ensemble.py로 앙상블 조합 탐색 진행하세요.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT v3 Sharpe 보상 + 보유시간 보너스 실험 — 2시드 학습
# conda run -n ai-lab python train_alt_v3_sharpe.py

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
from env_alt_v3_sharpe import AltV3SharpeEnv

TOTAL_STEPS = 5_000_000
SEEDS = [42, 300]  # 2시드만 (v2에서 최고/최저 성능 시드)
MODELS_BASE = Path(__file__).parent.parent / "models"
STATUS_FILE = Path(__file__).parent / "train_status.json"
ALT_DATA_DIR = PROJECT_ROOT / "rl" / "alt_30m"
TRAIN_END = "2025-09-01"


def load_all_datasets():
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


class CurriculumCallback(BaseCallback):
    def __init__(self, total_steps):
        super().__init__(); self.phase = 0; self.total_steps = total_steps
    def _on_step(self):
        p = self.num_timesteps / self.total_steps
        if p > 0.6 and self.phase < 2:
            self.phase = 2
            for e in self.training_env.envs: e.set_bear_ratio(0.0)
        elif p > 0.3 and self.phase < 1:
            self.phase = 1
            for e in self.training_env.envs: e.set_bear_ratio(0.3)
        return True


class LogCallback(BaseCallback):
    def __init__(self, seed, save_dir, total_steps):
        super().__init__(); self.seed = seed; self.save_dir = save_dir
        self.total_steps = total_steps; self.best = -999
    def _on_step(self):
        if self.num_timesteps % 100_000 == 0:
            rew = np.mean([e["r"] for e in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            if rew > self.best:
                self.best = rew
                best_path = self.save_dir / "best"
                best_path.mkdir(exist_ok=True)
                self.model.save(str(best_path / "ppo_alt.zip"))
            pct = self.num_timesteps / self.total_steps * 100
            print(f"  [v3_seed{self.seed}] {self.num_timesteps:>8,} ({pct:.0f}%) rew={rew:+.3f} best={self.best:+.3f}")
            try:
                s = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
                s[f"alt_v3_seed{self.seed}"] = {
                    "steps": self.num_timesteps, "progress": f"{pct:.0f}%",
                    "reward": f"{rew:+.3f}", "updated": datetime.now().strftime("%H:%M:%S")
                }
                STATUS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))
            except: pass
        return True


def backtest_single(model, test_df, name=""):
    env = AltV3SharpeEnv(
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
        "trades": info["total_trades"], "win_rate": info["win_rate"] * 100,
        "long_pct": dist.get(1, 0) / total * 100,
    }


def train_seed(seed, train_datasets, test_datasets, names):
    save_dir = MODELS_BASE / f"alt_v3_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"v3 Sharpe 시드 {seed} — {TOTAL_STEPS:,}스텝, {len(train_datasets)}코인")
    print(f"{'='*60}")

    def make_env(offset=0):
        def _init():
            env = AltV3SharpeEnv(
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
    model.learn(total_timesteps=TOTAL_STEPS,
                callback=[CurriculumCallback(TOTAL_STEPS), LogCallback(seed, save_dir, TOTAL_STEPS)])
    elapsed = time.time() - t0
    model.save(str(save_dir / "ppo_alt"))
    print(f"\n  [v3_seed{seed}] 완료: {elapsed/60:.0f}분")

    # 백테스트
    results = []
    for test_df, name in zip(test_datasets, names):
        if len(test_df) < 100: continue
        r = backtest_single(model, test_df, name)
        results.append(r)

    if results:
        avg_ret = np.mean([r["return_pct"] for r in results])
        avg_mdd = np.mean([r["mdd_pct"] for r in results])
        wins = sum(1 for r in results if r["return_pct"] > 0)
        print(f"  평균: {avg_ret:+.1f}%, MDD: {avg_mdd:.1f}%, 수익: {wins}/{len(results)}")
        (save_dir / "backtest_results.json").write_text(json.dumps({
            "seed": seed, "steps": TOTAL_STEPS, "coins": len(results),
            "avg_return": round(avg_ret, 2), "avg_mdd": round(avg_mdd, 2),
            "profitable_coins": wins, "elapsed_min": round(elapsed/60, 1),
            "env": "v3_sharpe",
        }, indent=2, ensure_ascii=False))
    return results


def main():
    print("=== ALT v3 Sharpe 보상 + 보유시간 실험 ===\n")
    train_datasets, test_datasets, names = load_all_datasets()
    print(f"코인: {len(names)}종, 시드: {SEEDS}\n")

    for seed in SEEDS:
        train_seed(seed, train_datasets, test_datasets, names)

    print("\n학습 완료!")


if __name__ == "__main__":
    main()

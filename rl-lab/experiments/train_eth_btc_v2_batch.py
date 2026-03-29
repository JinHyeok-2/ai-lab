#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ETH/BTC 5M스텝 시드 앙상블 배치 학습
# 기존 3M → 5M, 시드 3개 추가 (ETH: 400,500,600 / BTC: 400,500,600)
# conda run -n ai-lab python train_eth_btc_v2_batch.py

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
from env_v51_exp02 import ETHTradingEnvV51Exp02

TOTAL_STEPS = 5_000_000
MODELS_BASE = Path(__file__).parent.parent / "models"
STATUS_FILE = Path(__file__).parent / "train_status.json"
TRAIN_END = "2025-09-01"

# ETH 3시드, BTC 3시드
CONFIGS = [
    {"asset": "ETH", "csv": PROJECT_ROOT / "rl" / "eth_30m_v41.csv", "seeds": [400, 500, 600],
     "leverage": 3, "prefix": "eth"},
    {"asset": "BTC", "csv": PROJECT_ROOT / "rl" / "btc_30m.csv", "seeds": [400, 500, 600],
     "leverage": 3, "prefix": "btc"},
]


class LogCallback(BaseCallback):
    def __init__(self, tag, save_dir, total_steps):
        super().__init__(); self.tag = tag; self.save_dir = save_dir
        self.total_steps = total_steps; self.best = -999

    def _on_step(self):
        if self.num_timesteps % 100_000 == 0:
            rew = np.mean([e["r"] for e in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            if rew > self.best:
                self.best = rew
                best_path = self.save_dir / "best"
                best_path.mkdir(exist_ok=True)
                self.model.save(str(best_path / f"ppo_{self.tag.split('_')[0]}_30m.zip"))
            pct = self.num_timesteps / self.total_steps * 100
            print(f"  [{self.tag}] {self.num_timesteps:>8,} ({pct:.0f}%) rew={rew:+.3f} best={self.best:+.3f}")
            try:
                s = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
                s[self.tag] = {
                    "steps": self.num_timesteps, "progress": f"{pct:.0f}%",
                    "reward": f"{rew:+.3f}", "updated": datetime.now().strftime("%H:%M:%S")
                }
                STATUS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))
            except: pass
        return True


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


def backtest(model, test_df, leverage=3):
    env = ETHTradingEnvV51Exp02(
        test_df, initial_balance=10000.0, leverage=leverage, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    balances = [10000.0]; actions = []
    while not done:
        ac, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(ac)); done = term or trunc
        balances.append(info["balance"]); actions.append(int(ac))
    # 숏 마스킹 적용
    arr = np.array(balances); peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    ret = (balances[-1] - 10000) / 10000 * 100
    dist = Counter(actions); total = len(actions)
    return {
        "return_pct": ret, "mdd_pct": mdd,
        "trades": info["total_trades"], "win_rate": info["win_rate"] * 100,
        "long_pct": dist.get(1, 0) / total * 100,
        "short_pct": dist.get(2, 0) / total * 100,
    }


def train_one(asset, csv_path, seed, leverage, prefix):
    tag = f"{prefix}_v2_seed{seed}"
    save_dir = MODELS_BASE / f"{prefix}_v2_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"ppo_{prefix}_30m"

    print(f"\n{'='*60}")
    print(f"{asset} seed{seed} — {TOTAL_STEPS:,}스텝, leverage={leverage}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path, parse_dates=["time"])
    train_df = df[df["time"] <= pd.Timestamp(TRAIN_END)].reset_index(drop=True)
    test_df = df[df["time"] > pd.Timestamp(TRAIN_END)].reset_index(drop=True)
    print(f"  학습: {len(train_df)}캔들, 테스트: {len(test_df)}캔들")

    def make_env(offset=0):
        def _init():
            env = ETHTradingEnvV51Exp02(
                train_df, leverage=leverage, window_size=20,
                min_hold_steps=4, max_episode_len=2000,
                max_drawdown=0.15, cooldown_steps=8, curriculum=True
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
                callback=[CurriculumCallback(TOTAL_STEPS), LogCallback(tag, save_dir, TOTAL_STEPS)])
    elapsed = time.time() - t0
    model.save(str(save_dir / model_name))
    print(f"\n  [{tag}] 완료: {elapsed/60:.0f}분")

    # 백테스트
    r = backtest(model, test_df, leverage)
    print(f"  수익: {r['return_pct']:+.1f}%, MDD: {r['mdd_pct']:.1f}%, "
          f"거래: {r['trades']}회, 승률: {r['win_rate']:.1f}%, 롱: {r['long_pct']:.0f}%")

    (save_dir / "backtest_results.json").write_text(json.dumps({
        "asset": asset, "seed": seed, "steps": TOTAL_STEPS,
        "return_pct": round(r["return_pct"], 2), "mdd_pct": round(r["mdd_pct"], 2),
        "trades": r["trades"], "win_rate": round(r["win_rate"], 1),
        "elapsed_min": round(elapsed/60, 1),
    }, indent=2, ensure_ascii=False))
    return r


def main():
    print("=== ETH/BTC 5M 시드 앙상블 배치 학습 ===\n")
    print(f"스텝: {TOTAL_STEPS:,}/시드, 예상: ~{len(CONFIGS)*3*3.5:.0f}시간\n")

    all_results = {}
    for cfg in CONFIGS:
        for seed in cfg["seeds"]:
            r = train_one(cfg["asset"], cfg["csv"], seed, cfg["leverage"], cfg["prefix"])
            all_results[f"{cfg['prefix']}_seed{seed}"] = r

    print(f"\n{'='*60}")
    print("최종 요약")
    print(f"{'='*60}")
    for k, r in all_results.items():
        print(f"  {k:<20} {r['return_pct']:>+8.1f}% MDD{r['mdd_pct']:>7.1f}% 승률{r['win_rate']:>5.1f}%")
    print("\n완료!")


if __name__ == "__main__":
    main()

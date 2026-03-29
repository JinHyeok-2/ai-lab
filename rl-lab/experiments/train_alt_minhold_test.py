#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT min_hold_steps 파라미터 실험 — 4 vs 6 vs 8 비교
# 보상 함수 변경 없이 파라미터만 조정
# ETH/BTC 학습 완료 후 실행: conda run -n ai-lab python train_alt_minhold_test.py

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

TOTAL_STEPS = 3_000_000  # 파라미터 비교용이므로 3M으로 충분
SEED = 42
MODELS_BASE = Path(__file__).parent.parent / "models"
STATUS_FILE = Path(__file__).parent / "train_status.json"
ALT_DATA_DIR = PROJECT_ROOT / "rl" / "alt_30m"
TRAIN_END = "2025-09-01"

# 테스트할 min_hold_steps 값
HOLD_VALUES = [4, 6, 8]


def load_datasets():
    train_ds, test_ds, names = [], [], []
    for p in sorted(ALT_DATA_DIR.glob("*_30m.csv")):
        df = pd.read_csv(p, parse_dates=["time"])
        name = p.stem.replace("_30m", "").upper()
        tr = df[df["time"] <= pd.Timestamp(TRAIN_END)].reset_index(drop=True)
        te = df[df["time"] > pd.Timestamp(TRAIN_END)].reset_index(drop=True)
        if len(tr) > 1000:
            train_ds.append(tr); test_ds.append(te); names.append(name)
    return train_ds, test_ds, names


class LogCB(BaseCallback):
    def __init__(self, tag):
        super().__init__(); self.tag = tag; self.best = -999
    def _on_step(self):
        if self.num_timesteps % 500_000 == 0:
            rew = np.mean([e["r"] for e in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            if rew > self.best: self.best = rew
            pct = self.num_timesteps / TOTAL_STEPS * 100
            print(f"  [{self.tag}] {self.num_timesteps:>8,} ({pct:.0f}%) best={self.best:+.3f}")
            try:
                s = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
                s[self.tag] = {"steps": self.num_timesteps, "progress": f"{pct:.0f}%",
                               "updated": datetime.now().strftime("%H:%M:%S")}
                STATUS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))
            except: pass
        return True


class CurrCB(BaseCallback):
    def __init__(self):
        super().__init__(); self.phase = 0
    def _on_step(self):
        p = self.num_timesteps / TOTAL_STEPS
        if p > 0.6 and self.phase < 2:
            self.phase = 2
            for e in self.training_env.envs: e.set_bear_ratio(0.0)
        elif p > 0.3 and self.phase < 1:
            self.phase = 1
            for e in self.training_env.envs: e.set_bear_ratio(0.3)
        return True


def bt(model, test_df):
    env = AltUniversalTradingEnv(
        [test_df], initial_balance=10000.0, leverage=2, window_size=20,
        min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False
    )
    obs, _ = env.reset(); done = False
    bals = [10000.0]
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, t, tr, info = env.step(int(a)); done = t or tr
        bals.append(info["balance"])
    arr = np.array(bals); pk = np.maximum.accumulate(arr)
    return (bals[-1]-10000)/10000*100, ((arr-pk)/pk*100).min(), info["total_trades"], info["win_rate"]*100


def main():
    print("=== ALT min_hold_steps 파라미터 실험 ===\n")
    train_ds, test_ds, names = load_datasets()
    print(f"코인: {len(names)}종\n")

    # 대표 10코인으로 평가
    eval_coins = ["TAO", "SOL", "AVAX", "LINK", "SUI", "ONT", "FIL", "DOGE", "XRP", "APT"]
    eval_idx = [i for i, n in enumerate(names) if n in eval_coins]

    results = {}
    for mh in HOLD_VALUES:
        tag = f"minhold_{mh}"
        save_dir = MODELS_BASE / f"alt_minhold_{mh}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"min_hold_steps = {mh}")
        print(f"{'='*50}")

        def make_env(offset=0):
            def _init():
                env = AltUniversalTradingEnv(
                    train_ds, leverage=2, window_size=20,
                    min_hold_steps=mh, max_episode_len=2000,
                    max_drawdown=0.20, cooldown_steps=8, curriculum=True
                )
                env.reset(seed=SEED + offset)
                return env
            return _init

        env = DummyVecEnv([make_env(i) for i in range(4)])
        model = PPO("MlpPolicy", env,
                    learning_rate=3e-4, n_steps=2048, batch_size=256,
                    n_epochs=10, gamma=0.975, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]),
                                       activation_fn=torch.nn.ReLU),
                    seed=SEED, device="cuda", verbose=0)

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_STEPS, callback=[CurrCB(), LogCB(tag)])
        elapsed = time.time() - t0
        model.save(str(save_dir / "ppo_alt"))
        print(f"  완료: {elapsed/60:.0f}분")

        # 평가 (대표 10코인)
        rets, mdds = [], []
        for i in eval_idx:
            if len(test_ds[i]) < 100: continue
            r, m, _, _ = bt(model, test_ds[i])
            rets.append(r); mdds.append(m)
        avg_r = np.mean(rets); avg_m = np.mean(mdds)
        wins = sum(1 for r in rets if r > 0)
        results[mh] = {"avg_ret": avg_r, "avg_mdd": avg_m, "wins": wins, "total": len(rets)}
        print(f"  결과: {avg_r:+.1f}%, MDD {avg_m:.1f}%, 수익 {wins}/{len(rets)}")

    # 비교
    print(f"\n{'='*50}")
    print("min_hold_steps 비교")
    print(f"{'='*50}")
    for mh, r in results.items():
        print(f"  mh={mh}: {r['avg_ret']:>+8.1f}% MDD{r['avg_mdd']:>7.1f}% 코인 {r['wins']}/{r['total']}")


if __name__ == "__main__":
    main()

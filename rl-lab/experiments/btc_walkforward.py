#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# BTC Walk-Forward 검증: exp14+seed100+seed200 다수결

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from collections import Counter
from stable_baselines3 import PPO
from env_v51_exp02 import ETHTradingEnvV51Exp02

EXP_DIR = Path(__file__).parent.parent
LEVERAGE = 3


def backtest_majority(models, df):
    env = ETHTradingEnvV51Exp02(df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    balances = [10000.0]
    done = False
    while not done:
        votes = [int(models[n].predict(obs, deterministic=True)[0]) for n in models]
        vc = Counter(votes)
        maj = vc.most_common(1)[0]
        action = maj[0] if maj[1] >= 2 else 0
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        balances.append(info["balance"])
    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    return {"return": (balances[-1]-10000)/10000*100, "mdd": mdd,
            "trades": info["total_trades"], "win_rate": info["win_rate"]}


def main():
    data_path = PROJECT_ROOT / "rl" / "btc_30m.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)

    models = {
        "btc_exp14": PPO.load(str(EXP_DIR / "models/btc_exp14/ppo_btc_30m.zip"), device="cpu"),
        "btc_seed100": PPO.load(str(EXP_DIR / "models/btc_seed100/ppo_btc_30m.zip"), device="cpu"),
        "btc_seed200": PPO.load(str(EXP_DIR / "models/btc_seed200/ppo_btc_30m.zip"), device="cpu"),
    }

    # 2개월 롤링 윈도우
    window_candles = 2880
    total = len(test_df)
    windows = []
    for start in range(0, total - window_candles + 1, window_candles):
        end = min(start + window_candles, total)
        w_df = test_df.iloc[start:end].reset_index(drop=True)
        t0 = test_df["time"].iloc[start].strftime("%Y-%m-%d")
        t1 = test_df["time"].iloc[end-1].strftime("%Y-%m-%d")
        windows.append((f"{t0} ~ {t1}", w_df))
    last_start = (total // window_candles) * window_candles
    if last_start < total and last_start > 0:
        w_df = test_df.iloc[last_start:].reset_index(drop=True)
        if len(w_df) > 100:
            t0 = test_df["time"].iloc[last_start].strftime("%Y-%m-%d")
            t1 = test_df["time"].iloc[-1].strftime("%Y-%m-%d")
            windows.append((f"{t0} ~ {t1}", w_df))
    windows.append(("전체", test_df))

    print(f"{'='*60}")
    print(f"  BTC Walk-Forward: exp14+seed100+seed200 다수결")
    print(f"  총 {len(windows)}개 구간")
    print(f"{'='*60}")

    for label, w_df in windows:
        print(f"\n--- {label} (캔들 {len(w_df)}) ---")
        r = backtest_majority(models, w_df)
        print(f"  수익 {r['return']:>+7.1f}% | MDD {r['mdd']:>6.1f}% | 거래 {r['trades']:>3}회 | 승률 {r['win_rate']:.1%}")

    print("\n완료")


if __name__ == "__main__":
    main()

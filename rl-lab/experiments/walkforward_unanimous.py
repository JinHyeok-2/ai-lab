#!/usr/bin/env python3
# Walk-forward 검증: exp14+exp08+seed700 만장일치 (새 프로덕션 앙상블)

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import logging
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from env_v51_exp02 import ETHTradingEnvV51Exp02

EXP_DIR = Path(__file__).parent.parent
OUT_DIR = EXP_DIR / "experiments"
LEVERAGE = 3

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "walkforward_unanimous.log", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def backtest_unanimous(models, df):
    """만장일치 앙상블 백테스트"""
    env = ETHTradingEnvV51Exp02(df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    balances = [env.initial_balance]
    done = False

    while not done:
        a14, _ = models["exp14"].predict(obs, deterministic=True)
        a08, _ = models["exp08"].predict(obs, deterministic=True)
        a700, _ = models["seed700"].predict(obs, deterministic=True)
        votes = [int(a14), int(a08), int(a700)]
        action = votes[0] if len(set(votes)) == 1 else 0

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        balances.append(info["balance"])

    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    return {"return": (balances[-1]-10000)/10000*100, "mdd": mdd,
            "trades": info["total_trades"], "win_rate": info["win_rate"], "balances": balances}


def backtest_majority(models, df):
    """다수결 앙상블 백테스트 (비교용)"""
    env = ETHTradingEnvV51Exp02(df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)
    obs, _ = env.reset()
    balances = [env.initial_balance]
    done = False

    while not done:
        a14, _ = models["exp14"].predict(obs, deterministic=True)
        a08, _ = models["exp08"].predict(obs, deterministic=True)
        a700, _ = models["seed700"].predict(obs, deterministic=True)
        votes = [int(a14), int(a08), int(a700)]
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
            "trades": info["total_trades"], "win_rate": info["win_rate"], "balances": balances}


def run():
    data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp("2025-09-01")].reset_index(drop=True)

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

    windows.append((f"전체", test_df))

    log.info("=" * 70)
    log.info("  Walk-Forward: exp14+exp08+seed700 (만장일치 vs 다수결)")
    log.info(f"  총 {len(windows)}개 구간")
    log.info("=" * 70)

    models = {
        "exp14": PPO.load(str(EXP_DIR / "models/exp14/ppo_eth_30m.zip"), device="cpu"),
        "exp08": PPO.load(str(EXP_DIR / "models/exp08/ppo_eth_30m.zip"), device="cpu"),
        "seed700": PPO.load(str(EXP_DIR / "models/seed700/ppo_eth_30m.zip"), device="cpu"),
    }

    for label, w_df in windows:
        log.info(f"\n--- {label} (캔들 {len(w_df)}) ---")
        ru = backtest_unanimous(models, w_df)
        rm = backtest_majority(models, w_df)
        log.info(f"  만장일치: 수익 {ru['return']:+7.1f}% | MDD {ru['mdd']:6.1f}% | 거래 {ru['trades']:3d}회 | 승률 {ru['win_rate']:.1%}")
        log.info(f"  다수결:   수익 {rm['return']:+7.1f}% | MDD {rm['mdd']:6.1f}% | 거래 {rm['trades']:3d}회 | 승률 {rm['win_rate']:.1%}")

    log.info("\n완료")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
# 앙상블 백테스트: exp14 + exp08 + exp05 다수결 투표
# 학습 없이 기존 모델 3개 조합

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import logging
from collections import Counter
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from env_v51_exp02 import ETHTradingEnvV51Exp02
from rl.env_v5 import ETHTradingEnvV5

INTERVAL = "30m"; LEVERAGE = 3; TRAIN_END = "2025-09-01"
EXP_DIR = Path(__file__).parent.parent
OUT_DIR = EXP_DIR / "experiments"

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "ensemble.log", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def run():
    data_path = PROJECT_ROOT / "rl" / "eth_30m_v41.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    test_df = df[df["time"] > pd.Timestamp(TRAIN_END)].reset_index(drop=True)
    bh_ret = (test_df["close"].iloc[-1] - test_df["close"].iloc[20]) / test_df["close"].iloc[20] * 100

    log.info("=" * 60)
    log.info("  앙상블 백테스트: exp14 + exp08 + exp05 다수결")
    log.info("=" * 60)

    # 모델 로드
    m14 = PPO.load(str(EXP_DIR / "models/exp14/ppo_eth_30m.zip"), device="cpu")
    m08 = PPO.load(str(EXP_DIR / "models/exp08/ppo_eth_30m.zip"), device="cpu")
    m05 = RecurrentPPO.load(str(EXP_DIR / "models/exp05/ppo_eth_30m.zip"), device="cpu")

    # 환경 (exp02 기반)
    env = ETHTradingEnvV51Exp02(test_df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(test_df)+100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)

    obs, _ = env.reset()
    balances = [env.initial_balance]
    actions = []
    done = False

    # LSTM 상태
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        # 3개 모델 예측
        a14, _ = m14.predict(obs, deterministic=True)
        a08, _ = m08.predict(obs, deterministic=True)
        a05, lstm_states = m05.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)

        votes = [int(a14), int(a08), int(a05)]

        # 다수결: 2/3 이상 동의하면 해당 행동, 아니면 관망(0)
        vote_count = Counter(votes)
        majority = vote_count.most_common(1)[0]
        if majority[1] >= 2:
            action = majority[0]
        else:
            action = 0  # 합의 없으면 관망

        obs, _, term, trunc, info = env.step(action)
        episode_start = np.array([term or trunc])
        done = term or trunc
        balances.append(info["balance"])
        actions.append(action)

    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()

    ens_stats = {"final": balances[-1], "return": (balances[-1]-10000)/10000*100,
                 "mdd": mdd, "trades": info["total_trades"], "win_rate": info["win_rate"]}

    # 개별 모델 백테스트 (비교용)
    results = {"ensemble": {"balances": balances, "actions": actions, "stats": ens_stats}}

    for name, mp, use_lstm in [
        ("exp14", EXP_DIR/"models/exp14/ppo_eth_30m.zip", False),
        ("exp08", EXP_DIR/"models/exp08/ppo_eth_30m.zip", False),
        ("exp05", EXP_DIR/"models/exp05/ppo_eth_30m.zip", True),
    ]:
        e = ETHTradingEnvV51Exp02(test_df, initial_balance=10000.0, leverage=LEVERAGE,
            window_size=20, min_hold_steps=4, max_episode_len=len(test_df)+100,
            max_drawdown=1.0, cooldown_steps=8, curriculum=False)
        if use_lstm:
            m = RecurrentPPO.load(str(mp), device="cpu")
            o, _ = e.reset(); b=[e.initial_balance]; a=[]; d=False
            ls=None; es=np.ones((1,),dtype=bool)
            while not d:
                ac,ls=m.predict(o,state=ls,episode_start=es,deterministic=True)
                o,_,te,tr,inf=e.step(int(ac)); es=np.array([te or tr]); d=te or tr
                b.append(inf["balance"]); a.append(int(ac))
        else:
            m = PPO.load(str(mp), device="cpu")
            o, _ = e.reset(); b=[e.initial_balance]; a=[]; d=False
            while not d:
                ac,_=m.predict(o,deterministic=True); o,_,te,tr,inf=e.step(int(ac)); d=te or tr
                b.append(inf["balance"]); a.append(int(ac))
        ar=np.array(b); pk=np.maximum.accumulate(ar); md=((ar-pk)/pk*100).min()
        results[name]={"balances":b,"actions":a,"stats":{"final":b[-1],"return":(b[-1]-10000)/10000*100,
                       "mdd":md,"trades":inf["total_trades"],"win_rate":inf["win_rate"]}}

    # 결과 출력
    for name, r in results.items():
        s = r["stats"]; dist = Counter(r["actions"]); total = len(r["actions"])
        nm = {0:"관망",1:"롱",2:"숏",3:"청산"}
        log.info(f"\n{'='*52}\n  {name}\n{'='*52}")
        log.info(f"  수익률: {s['return']:+.2f}% | MDD: {s['mdd']:.2f}% | 거래: {s['trades']}회 | 승률: {s['win_rate']:.1%}")
        log.info(f"  행동: " + " / ".join(f"{nm[k]}:{v}({v/total:.0%})" for k, v in sorted(dist.items())))

    # 그래프
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle("Ensemble (exp14+exp08+exp05 majority vote) vs individuals", fontsize=13, fontweight="bold")
    colors = {"ensemble": "#2E7D32", "exp14": "#E65100", "exp08": "#1565C0", "exp05": "#9C27B0"}
    for name, r in results.items():
        s = r["stats"]
        ax.plot(r["balances"], color=colors.get(name, "#666"), lw=2 if name=="ensemble" else 1.2,
                ls="-" if name=="ensemble" else "--",
                label=f"{name}  {s['return']:+.1f}% (MDD {s['mdd']:.1f}%)")
    ax.axhline(10000, color="#e53935", lw=0.8, ls=":"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "ensemble_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"\n그래프: {out}"); plt.close()


if __name__ == "__main__":
    run()

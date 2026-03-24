#!/usr/bin/env python3
# exp20: 롱 전용 모델 재학습
# exp14 기반 + 숏 행동 제거(3→2 action space: 관망/롱/청산) + 최신 데이터 포함
# 앙상블에 추가하여 롱 전용 환경에 특화된 모델 확보 목표

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import json, numpy as np, logging
from datetime import datetime
from collections import Counter
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from gymnasium import spaces
import pandas_ta as ta

EXP_NAME = "exp20"; INTERVAL = "30m"; LEVERAGE = 3; TRAIN_END = "2025-09-01"
N_ENVS = 4; TOTAL_STEPS = 3_000_000; GAMMA = 0.975
EXP_DIR = Path(__file__).parent.parent
MODEL_DIR = EXP_DIR / "models" / EXP_NAME; OUT_DIR = EXP_DIR / "experiments"

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(OUT_DIR / f"{EXP_NAME}.log", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


class ETHTradingEnvLongOnly(gym.Env):
    """롱 전용 환경 — 3개 행동: 0=관망, 1=롱, 2=청산 (숏 없음)"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 leverage: int = 3,
                 fee_rate: float = 0.0004,
                 window_size: int = 20,
                 min_hold_steps: int = 4,
                 max_episode_len: int = 2000,
                 max_drawdown: float = 0.15,
                 cooldown_steps: int = 8,
                 curriculum: bool = True):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.window_size = window_size
        self.min_hold_steps = min_hold_steps
        self.max_episode_len = max_episode_len
        self.max_drawdown = max_drawdown
        self.cooldown_steps = cooldown_steps
        self.curriculum = curriculum
        self.bear_ratio = 0.5 if curriculum else 0.0

        self.feature_cols = [
            "price_chg", "rsi_norm", "macd_norm", "bb_pct", "ema_ratio",
            "atr_norm", "vol_ratio", "adx_norm", "price_chg_1h", "rsi_diverge",
            "stoch_rsi", "obv_slope", "vol_regime",
        ]
        n_features = len(self.feature_cols) + 4  # +position, upnl, hold, cooldown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features,), dtype=np.float32
        )
        # 롱 전용: 관망(0), 롱(1), 청산(2)
        self.action_space = spaces.Discrete(3)

        self._preprocess()
        self._find_bear_segments()

    def _preprocess(self):
        df = self.df.copy()
        df["price_chg"] = df["close"].pct_change().fillna(0).clip(-0.1, 0.1)
        df["rsi_norm"] = df["rsi"] / 100.0
        df["macd_norm"] = (df["macd"] / df["close"]).fillna(0).clip(-0.05, 0.05)
        df["ema_ratio"] = (df["ema20"] / df["ema50"] - 1).fillna(0).clip(-0.1, 0.1)
        df["atr_norm"] = (df["atr"] / df["close"]).fillna(0).clip(0, 0.1)
        df["vol_ratio"] = df["vol_ratio"].fillna(1.0).clip(0, 5) / 5.0
        df["bb_pct"] = df["bb_pct"].fillna(0.5).clip(0, 1)

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            df["adx_norm"] = (adx_df["ADX_14"] / 100.0).fillna(0)
        else:
            df["adx_norm"] = 0.0

        df["price_chg_1h"] = df["close"].pct_change(2).fillna(0).clip(-0.1, 0.1) / 0.1

        price_dir = df["close"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        rsi_dir = df["rsi"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        df["rsi_diverge"] = (price_dir != rsi_dir).astype(float)

        stoch = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None and "STOCHRSIk_14_14_3_3" in stoch.columns:
            df["stoch_rsi"] = (stoch["STOCHRSIk_14_14_3_3"] / 100.0).fillna(0.5).clip(0, 1)
        else:
            df["stoch_rsi"] = 0.5

        obv = (df["volume"] * np.where(df["close"].diff() > 0, 1, -1)).cumsum()
        obv_std = obv.rolling(20).std().replace(0, np.nan).fillna(1)
        df["obv_slope"] = (obv.diff(5) / obv_std).fillna(0).clip(-3, 3) / 3.0

        atr = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vol_regime"] = atr.rolling(100).rank(pct=True).fillna(0.5)

        self._df = df
        self._valid_start = max(100, self.window_size)

    def _find_bear_segments(self):
        """하락장 구간 인덱스 (커리큘럼 용)"""
        df = self._df
        self._bear_idx = []
        self._bull_idx = []
        window = 48  # 24시간
        for i in range(self._valid_start, len(df) - self.max_episode_len):
            ret = (df["close"].iloc[i + window] - df["close"].iloc[i]) / df["close"].iloc[i]
            if ret < -0.03:
                self._bear_idx.append(i)
            else:
                self._bull_idx.append(i)
        if not self._bear_idx:
            self._bear_idx = list(range(self._valid_start, len(df) - self.max_episode_len))
        if not self._bull_idx:
            self._bull_idx = list(range(self._valid_start, len(df) - self.max_episode_len))

    def set_bear_ratio(self, ratio):
        self.bear_ratio = ratio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.curriculum and self.np_random.random() < self.bear_ratio:
            self._start = self.np_random.choice(self._bear_idx)
        else:
            self._start = self.np_random.choice(self._bull_idx)

        self._step = self._start
        self._end = min(self._start + self.max_episode_len, len(self._df) - 1)
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = 0   # 0=없음, 1=롱
        self.entry_price = 0.0
        self.hold_steps = 0
        self.cooldown = 0
        self.total_trades = 0
        self.wins = 0
        self.win_rate = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        rows = []
        for i in range(self._step - self.window_size, self._step):
            row = [float(self._df[c].iloc[i]) for c in self.feature_cols]
            # 상태: position(0/1), upnl, hold, cooldown
            pos_val = float(self.position)
            upnl = 0.0
            if self.position == 1 and self.entry_price > 0:
                cur = float(self._df["close"].iloc[min(i, len(self._df)-1)])
                upnl = np.clip((cur - self.entry_price) / self.entry_price, -1, 1)
            hold_val = min(self.hold_steps / 50.0, 1.0)
            cd_val = min(self.cooldown / 10.0, 1.0)
            row += [pos_val, upnl, hold_val, cd_val]
            rows.append(row)
        return np.array(rows, dtype=np.float32).flatten()

    def step(self, action):
        # action: 0=관망, 1=롱, 2=청산
        self._step += 1
        price = float(self._df["close"].iloc[self._step])
        reward = 0.0
        info = {}

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.position == 1:
            self.hold_steps += 1
            pnl_pct = (price - self.entry_price) / self.entry_price
            upnl = pnl_pct * self.leverage

            # 청산 조건: action=2 또는 자동
            do_close = False
            if action == 2 and self.hold_steps >= self.min_hold_steps:
                do_close = True
            elif pnl_pct <= -0.015:  # SL 1.5%
                do_close = True
            elif pnl_pct >= 0.0225:  # TP 2.25%
                do_close = True

            if do_close:
                realized = pnl_pct * self.leverage
                fee_cost = self.fee_rate * 2  # 왕복 수수료
                net = realized - fee_cost
                self.balance *= (1 + net)
                reward = net * 15  # PnL 보상 스케일
                self.position = 0
                self.entry_price = 0
                self.hold_steps = 0
                self.cooldown = self.cooldown_steps
                self.total_trades += 1
                if net > 0:
                    self.wins += 1
            else:
                # 홀딩 보상: 수익 중이면 소량 보상, 손실 중이면 패널티
                if upnl > 0:
                    reward = 0.001
                elif upnl < -0.005:
                    reward = -0.002

        elif action == 1 and self.cooldown == 0:
            # 롱 진입
            self.position = 1
            self.entry_price = price
            self.hold_steps = 0

            # ADX 추세 필터 보상
            adx = float(self._df["adx_norm"].iloc[self._step])
            if adx > 0.25:
                reward = 0.003  # 추세 시 진입 소량 보상
        else:
            # 관망
            reward = 0.0

        # DD 체크
        self.peak_balance = max(self.peak_balance, self.balance)
        dd = (self.peak_balance - self.balance) / self.peak_balance
        truncated = False

        if dd > 0.05:
            reward -= 0.05
        if dd > 0.10:
            reward -= 0.15
        if dd > self.max_drawdown:
            reward -= 1.0
            truncated = True

        terminated = self._step >= self._end
        self.win_rate = self.wins / max(self.total_trades, 1)

        info = {
            "balance": self.balance,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
        }

        return self._get_obs(), reward, terminated, truncated, info


class CB(BaseCallback):
    def __init__(self, total_steps, check_freq=10000):
        super().__init__()
        self.total_steps = total_steps
        self.check_freq = check_freq
        self.start_time = None
        self.best_reward = -np.inf

    def _on_training_start(self):
        self.start_time = datetime.now()
        log.info(f"[{self.start_time.strftime('%H:%M:%S')}] {EXP_NAME} 롱전용 학습 시작")

    def _on_step(self):
        progress = self.num_timesteps / self.total_steps
        # 커리큘럼: 하락장 비중 조절
        if progress < 0.25:
            br, ph = 0.5, "1-하락(50%)"
        elif progress < 0.65:
            br, ph = 0.3, "2-혼합(30%)"
        else:
            br, ph = 0.0, "3-자연"
        self.training_env.env_method("set_bear_ratio", br)
        ne = max(0.01 - 0.007 * progress, 0.003)
        self.model.ent_coef = ne

        if self.num_timesteps % self.check_freq < N_ENVS:
            el = (datetime.now() - self.start_time).total_seconds()
            eta = int(el / max(self.num_timesteps, 1) * (self.total_steps - self.num_timesteps) / 60)
            mr = "N/A"
            ne2 = len(self.model.ep_info_buffer)
            if ne2 > 0:
                rr = [ep["r"] for ep in self.model.ep_info_buffer]
                mr = f"{np.mean(rr):+.3f}"
                if np.mean(rr) > self.best_reward:
                    self.best_reward = np.mean(rr)
            log.info(f"[{datetime.now().strftime('%H:%M:%S')}] {self.num_timesteps:>9,}/{self.total_steps:,} "
                     f"({progress*100:5.1f}%) | 보상: {mr} | {ph} | ETA: ~{eta}분")
        return True

    def _on_training_end(self):
        el = (datetime.now() - self.start_time).total_seconds()
        log.info(f"\n학습 완료: {el/60:.1f}분 | 최고 보상: {self.best_reward:+.3f}")


def train():
    df = pd.read_csv(PROJECT_ROOT / "rl" / "eth_30m_v41.csv", parse_dates=["time"])
    cutoff = pd.Timestamp(TRAIN_END)
    si = df[df["time"] <= cutoff].index[-1] + 1
    train_df = df.iloc[:si].reset_index(drop=True)
    test_df = df.iloc[si:].reset_index(drop=True)
    log.info(f"학습: {len(train_df):,}캔들 | 테스트: {len(test_df):,}캔들")

    def make_env(seed):
        def _init():
            env = ETHTradingEnvLongOnly(train_df, initial_balance=10000.0, leverage=LEVERAGE,
                fee_rate=0.0004, window_size=20, min_hold_steps=4, max_episode_len=2000,
                max_drawdown=0.15, cooldown_steps=8, curriculum=True)
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_env(seed=i * 42) for i in range(N_ENVS)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mp = MODEL_DIR / f"ppo_eth_{INTERVAL}"

    model = PPO("MlpPolicy", env,
        learning_rate=lambda p: 3e-4 * (0.3 + 0.7 * p),
        n_steps=2048, batch_size=256, n_epochs=10,
        gamma=GAMMA, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, verbose=0, device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])))

    model.learn(total_timesteps=TOTAL_STEPS, callback=CB(TOTAL_STEPS), progress_bar=False)
    model.save(str(mp))

    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump({
            "experiment": EXP_NAME,
            "gamma": GAMMA,
            "type": "long_only",
            "actions": "0=관망, 1=롱, 2=청산",
            "note": "숏 제거, 롱 전용 환경",
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n모델 저장: {mp}.zip")
    return model, test_df


def backtest(model_or_path, test_df):
    """롱 전용 백테스트"""
    if isinstance(model_or_path, (str, Path)):
        model = PPO.load(str(model_or_path), device="cpu")
    else:
        model = model_or_path

    env = ETHTradingEnvLongOnly(test_df, initial_balance=10000.0, leverage=LEVERAGE,
        window_size=20, min_hold_steps=4, max_episode_len=len(test_df) + 100,
        max_drawdown=1.0, cooldown_steps=8, curriculum=False)

    obs, _ = env.reset()
    balances = [env.initial_balance]
    actions = []
    done = False

    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
        balances.append(info["balance"])
        actions.append(int(a))

    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    mdd = ((arr - peak) / peak * 100).min()
    bh = (test_df["close"].iloc[-1] - test_df["close"].iloc[20]) / test_df["close"].iloc[20] * 100

    d = Counter(actions)
    t = len(actions)
    nm = {0: "관망", 1: "롱", 2: "청산"}

    log.info(f"\n{'=' * 52}")
    log.info(f"  {EXP_NAME} 롱전용 백테스트 결과")
    log.info(f"{'=' * 52}")
    log.info(f"  수익률: {(balances[-1]-10000)/10000*100:+.2f}% | MDD: {mdd:.2f}% | B&H: {bh:+.2f}%")
    log.info(f"  거래: {info['total_trades']}회 | 승률: {info['win_rate']:.1%}")
    log.info(f"  행동: " + " / ".join(f"{nm[k]}:{v}({v/t:.0%})" for k, v in sorted(d.items())))

    # 그래프
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle(f"{EXP_NAME} Long-Only Backtest", fontsize=13, fontweight="bold")
    ax.plot(balances, color="#2E7D32", lw=1.8,
            label=f"{EXP_NAME} {(balances[-1]-10000)/100:+.1f}% (MDD {mdd:.1f}%)")
    ax.axhline(10000, color="#e53935", lw=0.8, ls=":")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"{EXP_NAME}_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"  그래프: {out}")
    plt.close()

    return {"balances": balances, "actions": actions}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--backtest-only", action="store_true")
    a = p.parse_args()

    if a.backtest_only:
        df = pd.read_csv(PROJECT_ROOT / "rl" / "eth_30m_v41.csv", parse_dates=["time"])
        test_df = df[df["time"] > pd.Timestamp(TRAIN_END)].reset_index(drop=True)
        mp = MODEL_DIR / f"ppo_eth_{INTERVAL}.zip"
        backtest(mp, test_df)
    else:
        model, test_df = train()
        backtest(model, test_df)

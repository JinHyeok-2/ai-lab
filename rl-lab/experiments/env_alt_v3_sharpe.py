#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(line_buffering=True)
# ALT v3 환경 — Sharpe 보상 + 보유시간 보너스
# env_alt_universal.py 기반, 보상 함수만 변경

import numpy as np
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class AltV3SharpeEnv(gym.Env):
    """ALT v3: Sharpe 기반 보상 + 보유시간 보상"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, datasets: list[pd.DataFrame],
                 initial_balance: float = 10000.0,
                 leverage: int = 2,
                 fee_rate: float = 0.0004,
                 window_size: int = 20,
                 min_hold_steps: int = 4,
                 max_episode_len: int = 2000,
                 max_drawdown: float = 0.20,
                 cooldown_steps: int = 8,
                 curriculum: bool = True):
        super().__init__()
        self.datasets = datasets
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
        n_features = len(self.feature_cols) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 관망/롱/청산

        self.processed_datasets = []
        self.bear_starts_list = []
        for df in self.datasets:
            proc = self._preprocess(df)
            self.processed_datasets.append(proc)
            self.bear_starts_list.append(self._find_bear_segments(proc))

        self.current_dataset_idx = 0
        self.processed = self.processed_datasets[0]
        self.bear_starts = self.bear_starts_list[0]

    def _preprocess(self, df):
        df = df.copy().reset_index(drop=True)
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
        df["vol_regime"] = df["atr"].rolling(100).rank(pct=True).fillna(0.5)
        return df

    def _find_bear_segments(self, df):
        if len(df) < 300: return []
        returns = df["close"].pct_change(200).fillna(0)
        bear_mask = returns < -0.03
        max_start = len(df) - self.max_episode_len - 1
        bear_starts = [i for i in bear_mask[bear_mask].index.tolist()
                       if self.window_size <= i <= max_start]
        return bear_starts if len(bear_starts) >= 10 else []

    def _get_trend(self):
        i = self.current_step
        if i < self.window_size: return 0
        row = self.processed.iloc[i]
        ema_ratio = float(row.get("ema_ratio", 0))
        price_chg_1h = float(row.get("price_chg_1h", 0))
        adx = float(row.get("adx_norm", 0))
        score = ema_ratio * 10 + price_chg_1h * 3
        if adx > 0.25: score *= 1.5
        return 1 if score > 0.3 else (-1 if score < -0.3 else 0)

    def set_bear_ratio(self, ratio): self.bear_ratio = ratio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_dataset_idx = int(self.np_random.integers(0, len(self.datasets)))
        self.processed = self.processed_datasets[self.current_dataset_idx]
        self.bear_starts = self.bear_starts_list[self.current_dataset_idx]

        max_start = len(self.processed) - self.max_episode_len - 1
        if max_start <= self.window_size:
            start = self.window_size
            self.episode_end = len(self.processed) - 1
        else:
            if (self.bear_ratio > 0 and self.bear_starts
                    and self.np_random.random() < self.bear_ratio):
                start = int(self.np_random.choice(self.bear_starts))
            else:
                start = int(self.np_random.integers(self.window_size, max_start + 1))
            self.episode_end = min(start + self.max_episode_len, len(self.processed) - 1)

        self.current_step = start
        self.episode_start = start
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.cooldown = 0
        self.total_trades = 0
        self.win_trades = 0
        self.total_pnl = 0.0
        self.trade_history = []
        # Sharpe용 수익률 기록
        self.returns_buffer = deque(maxlen=100)
        return self._get_obs(), {}

    def _get_obs(self):
        rows = []
        for i in range(self.current_step - self.window_size, self.current_step):
            row = self.processed.iloc[i]
            price = self.processed["close"].iloc[self.current_step - 1]
            upnl = 0.0
            if self.position != 0 and self.entry_price > 0:
                upnl = (price - self.entry_price) / self.entry_price * self.position
            feat = [float(row[c]) for c in self.feature_cols]
            feat += [float(self.position), float(np.clip(upnl, -1, 1)),
                     float(min(self.hold_steps, 50) / 50),
                     float(min(self.cooldown, 10) / 10)]
            rows.append(feat)
        return np.array(rows, dtype=np.float32).flatten()

    def step(self, action: int):
        price = float(self.processed["close"].iloc[self.current_step])
        reward = 0.0
        trend = self._get_trend()

        if self.cooldown > 0:
            self.cooldown -= 1

        def close_pos():
            nonlocal reward
            if self.position == 0: return
            pnl_pct = (price - self.entry_price) / self.entry_price * self.position
            fee = self.fee_rate * 2
            trade_pnl = pnl_pct * self.leverage - fee
            self.balance += self.balance * trade_pnl
            self.returns_buffer.append(trade_pnl)

            # === Sharpe 보상 (핵심 변경) ===
            if len(self.returns_buffer) >= 5:
                rets = np.array(self.returns_buffer)
                sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(48)  # 연간화
                reward += np.clip(sharpe, -2, 2) * 0.5
            else:
                reward += trade_pnl * 10  # 초기에는 단순 PnL

            # 보유시간 보너스 (핵심 변경: 30분~2h = 1~4스텝 최적)
            if self.hold_steps < self.min_hold_steps:
                reward -= 0.02 * (self.min_hold_steps - self.hold_steps)  # 너무 빠른 청산
            elif 2 <= self.hold_steps <= 8:  # 30분~4시간 (최적 구간)
                if trade_pnl > 0:
                    reward += 0.02  # 적정 보유 + 수익 보너스
            elif self.hold_steps > 20:
                reward -= 0.005  # 과도하게 오래 보유

            if trade_pnl > 0:
                self.win_trades += 1
                reward += 0.01
            else:
                reward -= 0.005

            self.total_trades += 1
            self.total_pnl += trade_pnl
            self.trade_history.append({
                "step": self.current_step, "pnl": trade_pnl,
                "hold_steps": self.hold_steps, "price": price,
            })
            self.position = 0
            self.entry_price = 0.0
            self.hold_steps = 0
            self.cooldown = self.cooldown_steps

        # 행동 처리
        adx = float(self.processed.iloc[self.current_step].get("adx_norm", 0))
        adx_gate = adx > 0.25

        if action == 1:  # 롱
            if self.position == 0:
                if self.cooldown > 0:
                    reward -= 0.008 * self.cooldown
                elif adx_gate:
                    if trend == 1: reward += 0.005
                    elif trend == -1: reward -= 0.005
                self.balance *= (1 - self.fee_rate)
                self.position = 1
                self.entry_price = price
                self.hold_steps = 0
            else:
                reward -= 0.005
        elif action == 2:  # 청산
            if self.position != 0:
                close_pos()
            else:
                reward -= 0.005
        else:  # 관망
            if self.position != 0:
                self.hold_steps += 1
                if self.position * trend > 0:
                    reward += 0.001
                elif self.position * trend < 0:
                    reward -= 0.002
                if self.hold_steps > 150:
                    reward -= 0.002

        # DD 패널티
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance

        if drawdown > 0.10:
            reward -= 0.2
        elif drawdown > 0.05:
            reward -= 0.08

        if self.balance < self.initial_balance * 0.3:
            reward -= 0.3

        self.current_step += 1
        terminated = self.current_step >= self.episode_end
        truncated = (self.balance < self.initial_balance * 0.05
                     or drawdown > self.max_drawdown)
        if truncated:
            reward -= 0.5

        info = {
            "balance": self.balance, "position": self.position,
            "total_trades": self.total_trades,
            "win_rate": self.win_trades / max(self.total_trades, 1),
            "total_pnl": self.total_pnl, "drawdown": drawdown,
            "dataset_idx": self.current_dataset_idx,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        price = float(self.processed["close"].iloc[self.current_step - 1])
        dd = (self.peak_balance - self.balance) / self.peak_balance * 100
        print(f"Step {self.current_step:5d} | ${price:,.2f} | "
              f"잔고 ${self.balance:,.0f} | DD {dd:.1f}%")

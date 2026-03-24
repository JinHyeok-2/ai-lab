#!/usr/bin/env python3
# ETH 선물거래 RL 환경 v5.1-exp04
# exp02 기반 + 거래 빈도 소폭 증가 + 네트워크 확대 (학습 스크립트에서)
#
# exp02 대비 변경:
#   1. PnL 보상 스케일: 15 → 17 (거래 인센티브 소폭 증가)
#   2. cooldown: 8 → 7 (진입 기회 소폭 증가)
#   3. DD 패널티: exp02 그대로 유지 (5%/-0.05, 10%/-0.15)
#   4. 쿨다운 패널티 소폭 완화: 0.008 → 0.006 (진입 장벽 낮춤)
#   5. 나머지 전부 exp02 동일

import numpy as np
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
from gymnasium import spaces


class ETHTradingEnvV51Exp04(gym.Env):
    """v5.1-exp04: exp02 기반 + 거래 빈도 소폭 증가"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 leverage: int = 3,
                 fee_rate: float = 0.0004,
                 window_size: int = 20,
                 min_hold_steps: int = 4,
                 max_episode_len: int = 2000,
                 max_drawdown: float = 0.15,
                 cooldown_steps: int = 7,       # [변경2] 8 → 7
                 curriculum: bool = True):
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage        = leverage
        self.fee_rate        = fee_rate
        self.window_size     = window_size
        self.min_hold_steps  = min_hold_steps
        self.max_episode_len = max_episode_len
        self.max_drawdown    = max_drawdown
        self.cooldown_steps  = cooldown_steps
        self.curriculum      = curriculum
        self.bear_ratio      = 0.5 if curriculum else 0.0

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
        self.action_space = spaces.Discrete(4)

        self._preprocess()
        self._find_bear_segments()

    def _preprocess(self):
        df = self.df.copy()
        df["price_chg"]  = df["close"].pct_change().fillna(0).clip(-0.1, 0.1)
        df["rsi_norm"]   = df["rsi"] / 100.0
        df["macd_norm"]  = (df["macd"] / df["close"]).fillna(0).clip(-0.05, 0.05)
        df["ema_ratio"]  = (df["ema20"] / df["ema50"] - 1).fillna(0).clip(-0.1, 0.1)
        df["atr_norm"]   = (df["atr"] / df["close"]).fillna(0).clip(0, 0.1)
        df["vol_ratio"]  = df["vol_ratio"].fillna(1.0).clip(0, 5) / 5.0
        df["bb_pct"]     = df["bb_pct"].fillna(0.5).clip(0, 1)

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            df["adx_norm"] = (adx_df["ADX_14"] / 100.0).fillna(0)
        else:
            df["adx_norm"] = 0.0

        df["price_chg_1h"] = df["close"].pct_change(2).fillna(0).clip(-0.1, 0.1) / 0.1

        price_dir = df["close"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        rsi_dir   = df["rsi"].diff(5).apply(lambda x: 1 if x > 0 else -1)
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
        self.processed = df

    def _find_bear_segments(self):
        if len(self.df) < 300:
            self.bear_starts = []
            return
        returns  = self.df["close"].pct_change(200).fillna(0)
        bear_mask = returns < -0.03
        max_start = len(self.df) - self.max_episode_len - 1
        self.bear_starts = [
            i for i in bear_mask[bear_mask].index.tolist()
            if self.window_size <= i <= max_start
        ]
        if len(self.bear_starts) < 10:
            self.bear_starts = []

    def _get_trend(self):
        i = self.current_step
        if i < self.window_size:
            return 0
        row = self.processed.iloc[i]
        ema_ratio    = float(row.get("ema_ratio", 0))
        price_chg_1h = float(row.get("price_chg_1h", 0))
        adx          = float(row.get("adx_norm", 0))
        score = ema_ratio * 10 + price_chg_1h * 3
        if adx > 0.25:
            score *= 1.5
        if score > 0.3:
            return 1
        elif score < -0.3:
            return -1
        return 0

    def _get_adx(self):
        i = self.current_step
        if i >= len(self.processed):
            return 0.0
        return float(self.processed.iloc[i].get("adx_norm", 0))

    def set_bear_ratio(self, ratio: float):
        self.bear_ratio = ratio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = len(self.df) - self.max_episode_len - 1

        if max_start <= self.window_size:
            start = self.window_size
            self.episode_end = len(self.df) - 1
        else:
            if (self.bear_ratio > 0 and self.bear_starts
                    and self.np_random.random() < self.bear_ratio):
                start = int(self.np_random.choice(self.bear_starts))
            else:
                start = int(self.np_random.integers(self.window_size, max_start + 1))
            self.episode_end = min(start + self.max_episode_len, len(self.df) - 1)

        self.current_step  = start
        self.episode_start = start
        self.balance       = self.initial_balance
        self.peak_balance  = self.initial_balance
        self.position      = 0
        self.entry_price   = 0.0
        self.hold_steps    = 0
        self.cooldown      = 0
        self.total_trades  = 0
        self.win_trades    = 0
        self.total_pnl     = 0.0
        self.trade_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        rows = []
        for i in range(self.current_step - self.window_size, self.current_step):
            row   = self.processed.iloc[i]
            price = self.processed["close"].iloc[self.current_step - 1]
            upnl  = 0.0
            if self.position != 0 and self.entry_price > 0:
                upnl = (price - self.entry_price) / self.entry_price * self.position
            feat = [float(row[c]) for c in self.feature_cols]
            feat += [
                float(self.position),
                float(np.clip(upnl, -1, 1)),
                float(min(self.hold_steps, 50) / 50),
                float(min(self.cooldown, 10) / 10),
            ]
            rows.append(feat)
        return np.array(rows, dtype=np.float32).flatten()

    def step(self, action: int):
        price  = float(self.processed["close"].iloc[self.current_step])
        reward = 0.0
        trend  = self._get_trend()
        adx    = self._get_adx()

        if self.cooldown > 0:
            self.cooldown -= 1

        def close_pos():
            nonlocal reward
            if self.position == 0:
                return
            pnl_pct   = (price - self.entry_price) / self.entry_price * self.position
            fee       = self.fee_rate * 2
            trade_pnl = pnl_pct * self.leverage - fee

            self.balance += self.balance * trade_pnl

            if self.hold_steps < self.min_hold_steps:
                reward -= 0.015 * (self.min_hold_steps - self.hold_steps)

            # [변경1] PnL 보상 스케일: 15 → 17
            reward += trade_pnl * 17

            if trade_pnl > 0:
                self.win_trades += 1
                reward += 0.01
            else:
                reward -= 0.008

            self.total_trades += 1
            self.total_pnl    += trade_pnl
            self.trade_history.append({
                "step": self.current_step, "pnl": trade_pnl,
                "hold_steps": self.hold_steps, "price": price,
                "position": self.position,
            })
            self.position    = 0
            self.entry_price = 0.0
            self.hold_steps  = 0
            self.cooldown    = self.cooldown_steps  # [변경2] 7캔들

        # ── 행동 처리 ────────────────────────────────────
        # 추세 진입 보상: exp02 동일 0.006
        TREND_BONUS   = 0.006
        TREND_PENALTY = 0.006
        adx_gate = adx > 0.25

        if action == 1:   # 롱 진입
            if self.position == -1:
                close_pos()
            if self.position == 0:
                if self.cooldown > 0:
                    # [변경4] 쿨다운 패널티: 0.008 → 0.006
                    reward -= 0.006 * self.cooldown
                elif adx_gate:
                    if trend == 1:
                        reward += TREND_BONUS
                    elif trend == -1:
                        reward -= TREND_PENALTY
                self.balance    *= (1 - self.fee_rate)
                self.position    = 1
                self.entry_price = price
                self.hold_steps  = 0
            else:
                reward -= 0.005

        elif action == 2:  # 숏 진입
            if self.position == 1:
                close_pos()
            if self.position == 0:
                if self.cooldown > 0:
                    reward -= 0.006 * self.cooldown
                elif adx_gate:
                    if trend == -1:
                        reward += TREND_BONUS
                    elif trend == 1:
                        reward -= TREND_PENALTY
                self.balance    *= (1 - self.fee_rate)
                self.position    = -1
                self.entry_price = price
                self.hold_steps  = 0
            else:
                reward -= 0.005

        elif action == 3:  # 청산
            if self.position != 0:
                close_pos()
            else:
                reward -= 0.005

        else:  # 관망
            if self.position != 0:
                self.hold_steps += 1
                if self.position * trend > 0:
                    reward += 0.002
                elif self.position * trend < 0:
                    reward -= 0.003
                if self.hold_steps > 150:
                    reward -= 0.002

        # ── DD 단계별 패널티 (exp02 동일) ──────────────
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance

        if drawdown > 0.10:
            reward -= 0.15   # DD 10% 이상: 강한 패널티
        elif drawdown > 0.05:
            reward -= 0.05   # DD 5% 이상: 경고

        # 잔고 소진 경고
        if self.balance < self.initial_balance * 0.3:
            reward -= 0.3

        self.current_step += 1

        terminated = self.current_step >= self.episode_end
        truncated = (self.balance < self.initial_balance * 0.05
                     or drawdown > self.max_drawdown)
        if truncated:
            reward -= 0.5

        info = {
            "balance":      self.balance,
            "position":     self.position,
            "total_trades": self.total_trades,
            "win_rate":     self.win_trades / max(self.total_trades, 1),
            "total_pnl":    self.total_pnl,
            "drawdown":     drawdown,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        price = float(self.processed["close"].iloc[self.current_step - 1])
        pos_str = {1: "LONG", -1: "SHORT", 0: "없음"}[self.position]
        dd = (self.peak_balance - self.balance) / self.peak_balance * 100
        print(f"Step {self.current_step:5d} | ${price:,.2f} | "
              f"잔고 ${self.balance:,.0f} | {pos_str} | "
              f"거래 {self.total_trades}회 | DD {dd:.1f}%")

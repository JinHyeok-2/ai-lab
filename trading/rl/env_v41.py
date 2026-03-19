#!/usr/bin/env python3
# ETH 선물거래 RL 환경 v4.1
# 개선: 피처 추가(ADX, 1h변화율, RSI다이버전스) + 추세 방향 보상

import numpy as np
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
from gymnasium import spaces


class ETHTradingEnvV41(gym.Env):
    """
    v4.1 개선 환경
    - 피처 10개: price_chg, rsi_norm, macd_norm, bb_pct, ema_ratio,
                 atr_norm, vol_ratio, adx_norm, price_chg_1h, rsi_diverge
    - 추세 인식 보상: 하락 구간 숏 진입 추가 보상 / 롱 진입 패널티
    - 레버리지 3배
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 leverage: int = 3,
                 fee_rate: float = 0.0004,
                 window_size: int = 20,
                 min_hold_steps: int = 4):
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage        = leverage
        self.fee_rate        = fee_rate
        self.window_size     = window_size
        self.min_hold_steps  = min_hold_steps

        self.feature_cols = [
            "price_chg", "rsi_norm", "macd_norm", "bb_pct", "ema_ratio",
            "atr_norm", "vol_ratio", "adx_norm", "price_chg_1h", "rsi_diverge",
        ]
        n_features = len(self.feature_cols) + 4  # +포지션, +손익, +보유시간, +쿨다운

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self._preprocess()

    def _preprocess(self):
        df = self.df.copy()

        # 기존 피처
        df["price_chg"] = df["close"].pct_change().fillna(0)
        df["rsi_norm"]  = df["rsi"] / 100.0
        df["macd_norm"] = (df["macd"] / df["close"]).fillna(0)
        df["ema_ratio"] = (df["ema20"] / df["ema50"] - 1).fillna(0)
        df["atr_norm"]  = (df["atr"] / df["close"]).fillna(0)
        df["vol_ratio"] = df["vol_ratio"].fillna(1.0).clip(0, 5) / 5.0
        df["bb_pct"]    = df["bb_pct"].fillna(0.5).clip(0, 1)

        # 추가 피처 1: ADX (추세 강도)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            df["adx_norm"] = (adx_df["ADX_14"] / 100.0).fillna(0)
        else:
            df["adx_norm"] = 0.0

        # 추가 피처 2: 1시간 누적 변화율 (30m 2캔들 = 1h)
        df["price_chg_1h"] = df["close"].pct_change(2).fillna(0).clip(-0.1, 0.1) / 0.1

        # 추가 피처 3: RSI 다이버전스 (가격 방향 vs RSI 방향)
        price_dir = df["close"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        rsi_dir   = df["rsi"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        df["rsi_diverge"] = (price_dir != rsi_dir).astype(float)  # 1=다이버전스, 0=동일

        self.processed = df

    def _get_trend(self):
        """최근 20캔들 기준 추세 판별: 1=상승, -1=하락, 0=횡보"""
        i = self.current_step
        if i < self.window_size:
            return 0
        recent = self.processed["close"].iloc[i - self.window_size: i]
        chg = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]
        if chg > 0.02:
            return 1
        elif chg < -0.02:
            return -1
        return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step   = self.window_size
        self.balance        = self.initial_balance
        self.position       = 0
        self.entry_price    = 0.0
        self.hold_steps     = 0
        self.cooldown       = 0
        self.total_trades   = 0
        self.win_trades     = 0
        self.total_pnl      = 0.0
        self.trade_history  = []
        return self._get_obs(), {}

    def _get_obs(self):
        rows = []
        for i in range(self.current_step - self.window_size, self.current_step):
            row   = self.processed.iloc[i]
            price = self.processed["close"].iloc[self.current_step - 1]
            upnl  = 0.0
            if self.position != 0 and self.entry_price > 0:
                upnl = (price - self.entry_price) / self.entry_price * self.position

            feat = [row[c] for c in self.feature_cols]
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
                reward -= 0.01 * (self.min_hold_steps - self.hold_steps)

            reward += trade_pnl * 8
            if trade_pnl > 0:
                self.win_trades += 1
                reward += 0.005
            else:
                reward -= 0.005

            self.total_trades += 1
            self.total_pnl    += trade_pnl
            self.trade_history.append({
                "step": self.current_step, "pnl": trade_pnl,
                "hold_steps": self.hold_steps, "price": price
            })
            self.position    = 0
            self.entry_price = 0.0
            self.hold_steps  = 0
            self.cooldown    = 5

        if action == 1:   # 롱 진입
            if self.position == -1:
                close_pos()
            if self.position == 0:
                if self.cooldown > 0:
                    reward -= 0.005 * self.cooldown
                # 하락 추세에서 롱 진입 패널티
                if trend == -1:
                    reward -= 0.008
                self.balance    *= (1 - self.fee_rate)
                self.position    = 1
                self.entry_price = price
                self.hold_steps  = 0
            else:
                reward -= 0.003

        elif action == 2:  # 숏 진입
            if self.position == 1:
                close_pos()
            if self.position == 0:
                if self.cooldown > 0:
                    reward -= 0.005 * self.cooldown
                # 하락 추세에서 숏 진입 추가 보상
                if trend == -1:
                    reward += 0.01
                elif trend == 1:
                    reward -= 0.008
                self.balance    *= (1 - self.fee_rate)
                self.position    = -1
                self.entry_price = price
                self.hold_steps  = 0
            else:
                reward -= 0.003

        elif action == 3:  # 청산
            if self.position != 0:
                close_pos()
            else:
                reward -= 0.003

        else:  # 관망
            if self.position != 0:
                upnl = (price - self.entry_price) / self.entry_price * self.position
                reward += upnl * self.leverage * 0.005
                self.hold_steps += 1
                if self.hold_steps > 200:
                    reward -= 0.001

        if self.balance < self.initial_balance * 0.2:
            reward -= 0.5

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated  = self.balance < self.initial_balance * 0.05

        info = {
            "balance":      self.balance,
            "position":     self.position,
            "total_trades": self.total_trades,
            "win_rate":     self.win_trades / max(self.total_trades, 1),
            "total_pnl":    self.total_pnl,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        price   = float(self.processed["close"].iloc[self.current_step - 1])
        pos_str = {1: "LONG", -1: "SHORT", 0: "없음"}[self.position]
        print(f"Step {self.current_step:5d} | ${price:,.2f} | "
              f"잔고 ${self.balance:,.0f} | {pos_str} | "
              f"거래 {self.total_trades}회 | 승률 {self.win_trades/max(self.total_trades,1):.1%}")

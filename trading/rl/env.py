#!/usr/bin/env python3
# ETH 선물거래 강화학습 환경 — 과다매매 방지 강화 버전

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class ETHTradingEnv(gym.Env):
    """
    ETH 선물거래 시뮬레이션 환경

    Action:
        0 = 관망 (Hold)
        1 = 롱 진입 (Long)
        2 = 숏 진입 (Short)
        3 = 포지션 청산 (Close)

    과다매매 방지:
        - 최소 보유 시간 강제 (min_hold_steps)
        - 높은 거래 수수료 반영
        - 연속 매매 쿨다운 패널티
        - 짧은 보유 후 손절 패널티
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 leverage: int = 5,
                 fee_rate: float = 0.0004,
                 window_size: int = 20,
                 min_hold_steps: int = 3):   # 최소 보유 캔들 수
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage        = leverage
        self.fee_rate        = fee_rate
        self.window_size     = window_size
        self.min_hold_steps  = min_hold_steps  # 최소 보유 캔들

        self.feature_cols = [
            "price_chg", "rsi_norm", "macd_norm",
            "bb_pct", "ema_ratio", "atr_norm", "vol_ratio",
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
        df["price_chg"] = df["close"].pct_change().fillna(0)
        df["rsi_norm"]  = df["rsi"] / 100.0
        df["macd_norm"] = (df["macd"] / df["close"]).fillna(0)
        df["ema_ratio"] = (df["ema20"] / df["ema50"] - 1).fillna(0)
        df["atr_norm"]  = (df["atr"] / df["close"]).fillna(0)
        df["vol_ratio"] = df["vol_ratio"].fillna(1.0).clip(0, 5) / 5.0
        df["bb_pct"]    = df["bb_pct"].fillna(0.5).clip(0, 1)
        self.processed  = df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step   = self.window_size
        self.balance        = self.initial_balance
        self.position       = 0        # 0: 없음 / 1: 롱 / -1: 숏
        self.entry_price    = 0.0
        self.hold_steps     = 0
        self.cooldown       = 0        # 매매 후 쿨다운 카운터
        self.total_trades   = 0
        self.win_trades     = 0
        self.total_pnl      = 0.0
        self.trade_history  = []
        self.last_trade_step = -999
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
                float(min(self.cooldown, 10) / 10),   # 쿨다운 상태
            ]
            rows.append(feat)

        return np.array(rows, dtype=np.float32).flatten()

    def step(self, action: int):
        price    = float(self.processed["close"].iloc[self.current_step])
        reward   = 0.0

        # 쿨다운 감소
        if self.cooldown > 0:
            self.cooldown -= 1

        def close_pos(reason: str = ""):
            nonlocal reward
            if self.position == 0:
                return
            pnl_pct   = (price - self.entry_price) / self.entry_price * self.position
            fee       = self.fee_rate * 2
            trade_pnl = pnl_pct * self.leverage - fee
            self.balance += self.balance * trade_pnl

            # 짧은 보유 패널티 (min_hold_steps 미만)
            if self.hold_steps < self.min_hold_steps:
                reward -= 0.01 * (self.min_hold_steps - self.hold_steps)

            # 손익 보상
            reward += trade_pnl * 8

            if trade_pnl > 0:
                self.win_trades += 1
                reward += 0.005   # 승리 추가 보상
            else:
                reward -= 0.005   # 패배 추가 패널티

            self.total_trades += 1
            self.total_pnl    += trade_pnl
            self.trade_history.append({
                "step": self.current_step, "pnl": trade_pnl,
                "hold_steps": self.hold_steps, "price": price
            })
            self.position    = 0
            self.entry_price = 0.0
            self.hold_steps  = 0
            self.cooldown    = 5   # 청산 후 5 캔들 쿨다운

        # ── 행동 처리 ──────────────────────────────────────────────
        if action == 1:   # 롱 진입
            if self.position == -1:
                close_pos("숏→롱 전환")
            if self.position == 0:
                # 쿨다운 중 진입 패널티
                if self.cooldown > 0:
                    reward -= 0.005 * self.cooldown
                fee = self.fee_rate
                self.balance   *= (1 - fee)
                self.position   = 1
                self.entry_price = price
                self.hold_steps = 0
                self.last_trade_step = self.current_step
            else:
                reward -= 0.003   # 이미 롱인데 또 롱

        elif action == 2:  # 숏 진입
            if self.position == 1:
                close_pos("롱→숏 전환")
            if self.position == 0:
                if self.cooldown > 0:
                    reward -= 0.005 * self.cooldown
                fee = self.fee_rate
                self.balance   *= (1 - fee)
                self.position   = -1
                self.entry_price = price
                self.hold_steps = 0
                self.last_trade_step = self.current_step
            else:
                reward -= 0.003

        elif action == 3:  # 청산
            if self.position != 0:
                close_pos("수동 청산")
            else:
                reward -= 0.003   # 포지션 없는데 청산

        else:  # 관망
            if self.position != 0:
                upnl = (price - self.entry_price) / self.entry_price * self.position
                # 미실현 손익 소보상 (포지션 유지 장려)
                reward += upnl * self.leverage * 0.005
                self.hold_steps += 1
                # 과도한 장기 보유 패널티 (200 캔들 초과)
                if self.hold_steps > 200:
                    reward -= 0.001

        # 잔고 소진 패널티
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
              f"거래 {self.total_trades}회 | 승률 {self.win_trades/max(self.total_trades,1):.1%} | "
              f"쿨다운 {self.cooldown}")

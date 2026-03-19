#!/usr/bin/env python3
# ETH 선물거래 강화학습 환경
# v6 개선: 3x 레버리지, window=40, 트렌드 특징, 청산가 로직, 보상 스케일 균형

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

    v6 개선사항:
        - 레버리지 3x (5x → 리스크 감소, 청산가 -33.3%)
        - window_size 40 (20 → 더 긴 컨텍스트 20시간)
        - 특징 추가: ema200_ratio (트렌드 방향), trend_slope (모멘텀)
        - 보상 균형: 관망 보상 0.005 → 0.1 (20x 증가)
        - 청산가 로직: 33.3% 역방향 이동 시 강제청산 + 패널티
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 leverage: int = 3,
                 fee_rate: float = 0.0004,
                 window_size: int = 40,
                 min_hold_steps: int = 3,
                 _legacy: bool = False):
        """
        _legacy=True : v2~v5 구버전 호환 (window=20, 7특징, obs=220)
        _legacy=False: v6 이후 기본값  (window=40, 9특징, obs=520)
        """
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage        = leverage
        self.fee_rate        = fee_rate
        self.window_size     = window_size
        self.min_hold_steps  = min_hold_steps

        self._legacy = _legacy   # 강제청산 로직 등 v6 신규 동작 제어

        if _legacy:
            # v2~v5 구버전 특징 세트 (7개)
            self.feature_cols = [
                "price_chg", "rsi_norm", "macd_norm",
                "bb_pct", "ema_ratio", "atr_norm", "vol_ratio",
            ]
        else:
            # v6 이후 특징 세트 (9개, 트렌드 방향성 포함)
            self.feature_cols = [
                "price_chg", "rsi_norm", "macd_norm",
                "bb_pct", "ema_ratio", "atr_norm", "vol_ratio",
                "ema200_ratio", "trend_slope",
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
        df["price_chg"]  = df["close"].pct_change().fillna(0)
        df["rsi_norm"]   = df["rsi"] / 100.0
        df["macd_norm"]  = (df["macd"] / df["close"]).fillna(0)
        df["ema_ratio"]  = (df["ema20"] / df["ema50"] - 1).fillna(0)
        df["atr_norm"]   = (df["atr"] / df["close"]).fillna(0)
        df["vol_ratio"]  = df["vol_ratio"].fillna(1.0).clip(0, 5) / 5.0
        df["bb_pct"]     = df["bb_pct"].fillna(0.5).clip(0, 1)
        # 트렌드 특징: EMA200 대비 현재가 위치, 10캔들 기울기
        ema200 = df["close"].ewm(span=200, adjust=False).mean()
        df["ema200_ratio"] = (df["close"] / ema200 - 1).clip(-0.5, 0.5).fillna(0)
        df["trend_slope"]  = df["close"].pct_change(10).clip(-0.2, 0.2).fillna(0)
        self.processed = df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = self.window_size
        self.balance         = self.initial_balance
        self.position        = 0       # 0: 없음 / 1: 롱 / -1: 숏
        self.entry_price     = 0.0
        self.hold_steps      = 0
        self.cooldown        = 0
        self.total_trades    = 0
        self.win_trades      = 0
        self.total_pnl       = 0.0
        self.trade_history   = []
        self.last_trade_step = -999
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size
        rows  = []
        for i in range(start, self.current_step):
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
        price      = float(self.processed["close"].iloc[self.current_step])
        reward     = 0.0
        liquidated = False

        # 쿨다운 감소
        if self.cooldown > 0:
            self.cooldown -= 1

        # ── 강제청산 체크 (v6 이후만 적용, legacy 모델은 스킵) ──────
        if not self._legacy and self.position != 0 and self.entry_price > 0:
            price_move = (price - self.entry_price) / self.entry_price * self.position
            liq_thresh = -(1.0 / self.leverage - 0.01)   # -30.3% (1% 버퍼)
            if price_move <= liq_thresh:
                trade_pnl = max(price_move * self.leverage - self.fee_rate * 2, -0.99)
                self.balance = max(self.balance * (1 + trade_pnl), 1.0)
                self.total_trades += 1
                self.trade_history.append({
                    "step": self.current_step, "pnl": trade_pnl,
                    "hold_steps": self.hold_steps, "price": price,
                    "reason": "강제청산",
                })
                self.position    = 0
                self.entry_price = 0.0
                self.hold_steps  = 0
                self.cooldown    = 5
                reward    -= 1.0
                liquidated = True

        if not liquidated:
            def close_pos(reason: str = ""):
                nonlocal reward
                if self.position == 0:
                    return
                pnl_pct   = (price - self.entry_price) / self.entry_price * self.position
                fee       = self.fee_rate * 2
                trade_pnl = pnl_pct * self.leverage - fee
                self.balance += self.balance * trade_pnl

                # 짧은 보유 패널티
                if self.hold_steps < self.min_hold_steps:
                    reward -= 0.01 * (self.min_hold_steps - self.hold_steps)

                # 손익 보상 (스케일 8 → 5, VecNormalize가 추가 정규화)
                reward += trade_pnl * 5

                if trade_pnl > 0:
                    self.win_trades += 1
                    reward += 0.005
                else:
                    reward -= 0.005

                self.total_trades += 1
                self.total_pnl    += trade_pnl
                self.trade_history.append({
                    "step": self.current_step, "pnl": trade_pnl,
                    "hold_steps": self.hold_steps, "price": price,
                })
                self.position    = 0
                self.entry_price = 0.0
                self.hold_steps  = 0
                self.cooldown    = 5

            # ── 행동 처리 ──────────────────────────────────────────────
            if action == 1:   # 롱 진입
                if self.position == -1:
                    close_pos("숏→롱 전환")
                if self.position == 0:
                    if self.cooldown > 0:
                        reward -= 0.005 * self.cooldown
                    self.balance    *= (1 - self.fee_rate)
                    self.position    = 1
                    self.entry_price = price
                    self.hold_steps  = 0
                    self.last_trade_step = self.current_step
                else:
                    reward -= 0.003

            elif action == 2:  # 숏 진입
                if self.position == 1:
                    close_pos("롱→숏 전환")
                if self.position == 0:
                    if self.cooldown > 0:
                        reward -= 0.005 * self.cooldown
                    self.balance    *= (1 - self.fee_rate)
                    self.position    = -1
                    self.entry_price = price
                    self.hold_steps  = 0
                    self.last_trade_step = self.current_step
                else:
                    reward -= 0.003

            elif action == 3:  # 청산
                if self.position != 0:
                    close_pos("수동 청산")
                else:
                    reward -= 0.003

            else:  # 관망 (action == 0)
                if self.position != 0:
                    upnl = (price - self.entry_price) / self.entry_price * self.position
                    # 미실현 손익 보상 스케일 균형 (0.005 → 0.1, 20x 증가)
                    reward += upnl * self.leverage * 0.1
                    self.hold_steps += 1
                    if self.hold_steps > 200:
                        reward -= 0.001

            # 잔고 소진 패널티
            if self.balance < self.initial_balance * 0.2:
                reward -= 0.5

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated  = liquidated or (self.balance < self.initial_balance * 0.05)

        return self._get_obs(), reward, terminated, truncated, self._info()

    def _info(self):
        return {
            "balance":      self.balance,
            "position":     self.position,
            "total_trades": self.total_trades,
            "win_rate":     self.win_trades / max(self.total_trades, 1),
            "total_pnl":    self.total_pnl,
        }

    def render(self):
        price   = float(self.processed["close"].iloc[self.current_step - 1])
        pos_str = {1: "LONG", -1: "SHORT", 0: "없음"}[self.position]
        print(f"Step {self.current_step:5d} | ${price:,.2f} | "
              f"잔고 ${self.balance:,.0f} | {pos_str} | "
              f"거래 {self.total_trades}회 | 승률 {self.win_trades/max(self.total_trades,1):.1%} | "
              f"쿨다운 {self.cooldown}")

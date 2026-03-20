#!/usr/bin/env python3
# ETH 선물거래 RL 환경 v5
# 핵심 개선: 에피소드 분할 + 커리큘럼 러닝 + 보상 리스케일 + 피처 13개

import numpy as np
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
from gymnasium import spaces


class ETHTradingEnvV5(gym.Env):
    """
    v5 개선 사항 (v4.1 대비):
    1. 에피소드 분할 (max_episode_len=2000) — 짧은 에피소드 → 다수 학습 사이클
       v4.1은 29,760캔들 1개 에피소드 → ep_info_buffer 미충전 → 평균보상 N/A
    2. 랜덤 시작점 + 커리큘럼 (하락장 70% → 40% → 자연분포)
    3. 보상 리스케일: pnl × 20 (v4.1: × 8) + 추세 추종 강화
    4. 피처 13개: 기존 10 + StochRSI, OBV 기울기, 변동성 레짐
    5. MDD 조기 종료 (15% 낙폭 → truncate + 패널티)
    6. 관망 보상 개선: upnl 기반 → 추세 정렬 기반 (과도한 보유 방지)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 leverage: int = 3,
                 fee_rate: float = 0.0004,
                 window_size: int = 20,
                 min_hold_steps: int = 4,
                 max_episode_len: int = 2000,
                 max_drawdown: float = 0.15,
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
        self.curriculum      = curriculum
        self.bear_ratio      = 0.7 if curriculum else 0.0

        self.feature_cols = [
            "price_chg", "rsi_norm", "macd_norm", "bb_pct", "ema_ratio",
            "atr_norm", "vol_ratio", "adx_norm", "price_chg_1h", "rsi_diverge",
            "stoch_rsi", "obv_slope", "vol_regime",
        ]
        # 13 피처 + 4 상태 (포지션, 미실현손익, 보유시간, 쿨다운)
        n_features = len(self.feature_cols) + 4

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 0:관망 1:롱 2:숏 3:청산

        self._preprocess()
        self._find_bear_segments()

    # ── 전처리 ──────────────────────────────────────────────
    def _preprocess(self):
        df = self.df.copy()

        # 기본 피처 (v4.1 동일)
        df["price_chg"]  = df["close"].pct_change().fillna(0).clip(-0.1, 0.1)
        df["rsi_norm"]   = df["rsi"] / 100.0
        df["macd_norm"]  = (df["macd"] / df["close"]).fillna(0).clip(-0.05, 0.05)
        df["ema_ratio"]  = (df["ema20"] / df["ema50"] - 1).fillna(0).clip(-0.1, 0.1)
        df["atr_norm"]   = (df["atr"] / df["close"]).fillna(0).clip(0, 0.1)
        df["vol_ratio"]  = df["vol_ratio"].fillna(1.0).clip(0, 5) / 5.0
        df["bb_pct"]     = df["bb_pct"].fillna(0.5).clip(0, 1)

        # ADX (추세 강도)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            df["adx_norm"] = (adx_df["ADX_14"] / 100.0).fillna(0)
        else:
            df["adx_norm"] = 0.0

        # 1시간 변화율 (30m × 2캔들)
        df["price_chg_1h"] = df["close"].pct_change(2).fillna(0).clip(-0.1, 0.1) / 0.1

        # RSI 다이버전스
        price_dir = df["close"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        rsi_dir   = df["rsi"].diff(5).apply(lambda x: 1 if x > 0 else -1)
        df["rsi_diverge"] = (price_dir != rsi_dir).astype(float)

        # === v5 신규 피처 (3개) ===

        # 1) Stochastic RSI (0~1, 모멘텀 오실레이터)
        stoch = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None and "STOCHRSIk_14_14_3_3" in stoch.columns:
            df["stoch_rsi"] = (stoch["STOCHRSIk_14_14_3_3"] / 100.0).fillna(0.5).clip(0, 1)
        else:
            df["stoch_rsi"] = 0.5

        # 2) OBV 기울기 (5캔들, 표준화 — 거래량-가격 확인)
        obv = (df["volume"] * np.where(df["close"].diff() > 0, 1, -1)).cumsum()
        obv_std = obv.rolling(20).std().replace(0, np.nan).fillna(1)
        df["obv_slope"] = (obv.diff(5) / obv_std).fillna(0).clip(-3, 3) / 3.0

        # 3) 변동성 레짐 (ATR 퍼센타일 — 최근 100캔들 기준)
        df["vol_regime"] = df["atr"].rolling(100).rank(pct=True).fillna(0.5)

        self.processed = df

    def _find_bear_segments(self):
        """하락장 구간 시작점 탐색 (200캔들 롤링 수익률 < -3%)"""
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

    # ── 추세 판별 ───────────────────────────────────────────
    def _get_trend(self):
        """EMA + ADX 기반 추세: 1=상승, -1=하락, 0=횡보"""
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

    # ── 커리큘럼 인터페이스 ─────────────────────────────────
    def set_bear_ratio(self, ratio: float):
        """콜백에서 호출 — 하락장 샘플링 비율 조정"""
        self.bear_ratio = ratio

    # ── 환경 리셋 ───────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.df) - self.max_episode_len - 1

        if max_start <= self.window_size:
            # 백테스트 모드: 데이터 전체를 한 에피소드로 실행
            start = self.window_size
            self.episode_end = len(self.df) - 1
        else:
            # 커리큘럼: bear_ratio 확률로 하락장 시작점 선택
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

    # ── 관측 ────────────────────────────────────────────────
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

    # ── 스텝 ────────────────────────────────────────────────
    def step(self, action: int):
        price  = float(self.processed["close"].iloc[self.current_step])
        reward = 0.0
        trend  = self._get_trend()

        if self.cooldown > 0:
            self.cooldown -= 1

        # ── 포지션 청산 내부 함수 ───────────────────────────
        def close_pos():
            nonlocal reward
            if self.position == 0:
                return
            pnl_pct   = (price - self.entry_price) / self.entry_price * self.position
            fee       = self.fee_rate * 2
            trade_pnl = pnl_pct * self.leverage - fee
            self.balance += self.balance * trade_pnl

            # 짧은 보유 패널티
            if self.hold_steps < self.min_hold_steps:
                reward -= 0.015 * (self.min_hold_steps - self.hold_steps)

            # 핵심 보상: PnL × 20 (v4.1: × 8)
            reward += trade_pnl * 20

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
            self.cooldown    = 5

        # ── 행동 처리 ───────────────────────────────────────
        if action == 1:   # 롱 진입
            if self.position == -1:
                close_pos()
            if self.position == 0:
                if self.cooldown > 0:
                    reward -= 0.005 * self.cooldown
                # 추세 반영 (v5: 더 강한 시그널)
                if trend == -1:
                    reward -= 0.015   # 하락장 롱 = 강한 패널티
                elif trend == 1:
                    reward += 0.008   # 상승장 롱 = 보너스
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
                # 숏 진입 보상 강화 (v5 핵심)
                if trend == -1:
                    reward += 0.02    # 하락장 숏 = 강한 보너스 (v4.1: 0.01)
                elif trend == 1:
                    reward -= 0.015   # 상승장 숏 = 강한 패널티
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
                self.hold_steps += 1
                # v5: upnl 보상 제거 → 추세 정렬 보상으로 교체
                if ((self.position == 1 and trend == 1) or
                        (self.position == -1 and trend == -1)):
                    reward += 0.002   # 추세 방향 유지
                elif ((self.position == 1 and trend == -1) or
                        (self.position == -1 and trend == 1)):
                    reward -= 0.003   # 추세 역행 보유
                # 과도한 보유 패널티
                if self.hold_steps > 150:
                    reward -= 0.002

        # 피크 잔고 갱신
        self.peak_balance = max(self.peak_balance, self.balance)

        # 잔고 소진 경고
        if self.balance < self.initial_balance * 0.3:
            reward -= 0.3

        self.current_step += 1

        # ── 종료 조건 ───────────────────────────────────────
        terminated = self.current_step >= self.episode_end

        drawdown = (self.peak_balance - self.balance) / self.peak_balance
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

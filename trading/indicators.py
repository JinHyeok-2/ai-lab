#!/usr/bin/env python3
# 기술적 지표 계산 모듈

import pandas as pd
import pandas_ta as ta

def calc_indicators(df: pd.DataFrame) -> dict:
    """
    OHLCV 데이터프레임으로 주요 지표 계산 후 딕셔너리 반환
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # RSI (14)
    rsi = ta.rsi(close, length=14)
    rsi_val = round(float(rsi.iloc[-1]), 2) if rsi is not None else None

    # MACD
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        macd_val   = round(float(macd_df["MACD_12_26_9"].iloc[-1]), 4)
        signal_val = round(float(macd_df["MACDs_12_26_9"].iloc[-1]), 4)
        hist_val   = round(float(macd_df["MACDh_12_26_9"].iloc[-1]), 4)
    else:
        macd_val = signal_val = hist_val = None

    # 볼린저밴드 (20, 2)
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        col_u = next((c for c in bb.columns if c.startswith("BBU")), None)
        col_m = next((c for c in bb.columns if c.startswith("BBM")), None)
        col_l = next((c for c in bb.columns if c.startswith("BBL")), None)
        bb_upper = round(float(bb[col_u].iloc[-1]), 2) if col_u else None
        bb_mid   = round(float(bb[col_m].iloc[-1]), 2) if col_m else None
        bb_lower = round(float(bb[col_l].iloc[-1]), 2) if col_l else None
    else:
        bb_upper = bb_mid = bb_lower = None

    # EMA 20, 50
    ema20 = ta.ema(close, length=20)
    ema50 = ta.ema(close, length=50)
    ema20_val = round(float(ema20.iloc[-1]), 2) if ema20 is not None else None
    ema50_val = round(float(ema50.iloc[-1]), 2) if ema50 is not None else None

    # ATR (변동성)
    atr = ta.atr(high, low, close, length=14)
    atr_val = round(float(atr.iloc[-1]), 2) if atr is not None else None

    # ADX (추세 강도) — 25+ 강한 추세, 50+ 매우 강한 추세
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None:
        adx_col = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
        dmp_col = next((c for c in adx_df.columns if c.startswith("DMP_")), None)
        dmn_col = next((c for c in adx_df.columns if c.startswith("DMN_")), None)
        adx_val = round(float(adx_df[adx_col].iloc[-1]), 2) if adx_col else None
        dmp_val = round(float(adx_df[dmp_col].iloc[-1]), 2) if dmp_col else None  # +DI
        dmn_val = round(float(adx_df[dmn_col].iloc[-1]), 2) if dmn_col else None  # -DI
    else:
        adx_val = dmp_val = dmn_val = None

    # Stochastic RSI (과매수/과매도 민감 신호)
    stochrsi = ta.stochrsi(close, length=14)
    if stochrsi is not None:
        k_col = next((c for c in stochrsi.columns if "STOCHRSIk" in c), None)
        d_col = next((c for c in stochrsi.columns if "STOCHRSId" in c), None)
        stoch_k = round(float(stochrsi[k_col].iloc[-1]), 2) if k_col else None
        stoch_d = round(float(stochrsi[d_col].iloc[-1]), 2) if d_col else None
    else:
        stoch_k = stoch_d = None

    # OBV (On-Balance Volume) — 거래량 방향성
    obv = ta.obv(close, volume)
    if obv is not None and len(obv) >= 5:
        obv_now  = float(obv.iloc[-1])
        obv_prev = float(obv.iloc[-5])
        obv_trend = "up" if obv_now > obv_prev else "down"
    else:
        obv_now = obv_prev = None
        obv_trend = None

    # VWAP (기관 기준선) — 당일 기준, 가격과 비교
    try:
        vwap_s = ta.vwap(high, low, close, volume)
        vwap_val = round(float(vwap_s.iloc[-1]), 2) if vwap_s is not None else None
    except Exception:
        vwap_val = None

    # CVD (Cumulative Volume Delta) — 매수/매도 압력 누적
    # taker_buy_base 칼럼이 있을 때만 계산 (get_klines에서 제공)
    cvd_trend = None
    cvd_delta_pct = None
    if "taker_buy_base" in df.columns:
        taker_buy  = df["taker_buy_base"].astype(float)
        taker_sell = volume - taker_buy
        cvd = (taker_buy - taker_sell).cumsum()
        if len(cvd) >= 10:
            cvd_now   = float(cvd.iloc[-1])
            cvd_prev5 = float(cvd.iloc[-5])
            cvd_trend = "up" if cvd_now > cvd_prev5 else "down"
            # 변화율 % (절대값 기준)
            cvd_range = abs(float(cvd.iloc[-10:].max()) - float(cvd.iloc[-10:].min()))
            cvd_delta_pct = round((cvd_now - cvd_prev5) / max(cvd_range, 1e-6) * 100, 1)

    current_price = round(float(close.iloc[-1]), 2)
    prev_close    = round(float(close.iloc[-2]), 2)
    change_pct    = round((current_price - prev_close) / prev_close * 100, 2)

    return {
        "price":       current_price,
        "change_pct":  change_pct,
        "rsi":         rsi_val,
        "macd":        macd_val,
        "macd_signal": signal_val,
        "macd_hist":   hist_val,
        "bb_upper":    bb_upper,
        "bb_mid":      bb_mid,
        "bb_lower":    bb_lower,
        "ema20":       ema20_val,
        "ema50":       ema50_val,
        "atr":         atr_val,
        "adx":         adx_val,
        "adx_dmp":     dmp_val,
        "adx_dmn":     dmn_val,
        "stoch_k":     stoch_k,
        "stoch_d":     stoch_d,
        "obv_trend":      obv_trend,
        "vwap":           vwap_val,
        "cvd_trend":      cvd_trend,
        "cvd_delta_pct":  cvd_delta_pct,
    }

def format_for_agent(symbol: str, indicators: dict, label: str = "") -> str:
    """에이전트에게 전달할 지표 요약 텍스트. label로 타임프레임 표시 (예: '15분봉')"""
    header = f"[{symbol} {label} 기술적 지표 요약]" if label else f"[{symbol} 기술적 지표 요약]"
    adx_str   = f"{indicators.get('adx')} (+DI:{indicators.get('adx_dmp')} / -DI:{indicators.get('adx_dmn')})" if indicators.get('adx') else "N/A"
    stoch_str = f"K:{indicators.get('stoch_k')} / D:{indicators.get('stoch_d')}" if indicators.get('stoch_k') is not None else "N/A"
    obv_str   = indicators.get('obv_trend', 'N/A')
    vwap_str  = f"${indicators.get('vwap'):,}" if indicators.get('vwap') else "N/A"
    cvd_str   = f"{indicators.get('cvd_trend', 'N/A')} ({indicators.get('cvd_delta_pct', 0):+.1f}%)" if indicators.get('cvd_trend') else "N/A"
    return f"""
{header}
현재가: ${indicators['price']:,}  ({indicators['change_pct']:+.2f}%)

RSI(14): {indicators['rsi']}
MACD: {indicators['macd']} / Signal: {indicators['macd_signal']} / Hist: {indicators['macd_hist']}
볼린저밴드: 상단 {indicators['bb_upper']} / 중간 {indicators['bb_mid']} / 하단 {indicators['bb_lower']}
EMA20: {indicators['ema20']} / EMA50: {indicators['ema50']}
ATR(변동성): {indicators['atr']}
ADX(추세강도): {adx_str}
Stoch RSI: {stoch_str}
OBV 방향: {obv_str}
CVD(매수압력): {cvd_str}
VWAP: {vwap_str}
""".strip()

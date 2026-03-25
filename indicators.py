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
        _vwap_df = df[["high", "low", "close", "volume"]].copy()
        if "timestamp" in df.columns:
            _vwap_df.index = pd.to_datetime(df["timestamp"], unit="ms")
        elif "time" in df.columns:
            _vwap_df.index = pd.to_datetime(df["time"])
        if not isinstance(_vwap_df.index, pd.DatetimeIndex):
            _vwap_df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(_vwap_df), freq="15min")
        _vwap_df = _vwap_df.sort_index()
        vwap_s = ta.vwap(_vwap_df["high"], _vwap_df["low"], _vwap_df["close"], _vwap_df["volume"])
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

    # RSI 다이버전스 탐지 — 최근 20봉 기준 가격↔RSI 방향 불일치
    rsi_divergence = None
    if rsi is not None and len(rsi.dropna()) >= 20:
        _rsi_20 = rsi.iloc[-20:]
        _close_20 = close.iloc[-20:]
        # 최근 저점 2개 비교 (불리시 다이버전스: 가격↓ but RSI↑)
        _half = len(_rsi_20) // 2
        _p1_low, _p2_low = float(_close_20.iloc[:_half].min()), float(_close_20.iloc[_half:].min())
        _r1_low, _r2_low = float(_rsi_20.iloc[:_half].min()), float(_rsi_20.iloc[_half:].min())
        if _p2_low < _p1_low and _r2_low > _r1_low:
            rsi_divergence = "bullish"  # 가격 저점 하락, RSI 저점 상승
        # 최근 고점 2개 비교 (베어리시 다이버전스: 가격↑ but RSI↓)
        _p1_hi, _p2_hi = float(_close_20.iloc[:_half].max()), float(_close_20.iloc[_half:].max())
        _r1_hi, _r2_hi = float(_rsi_20.iloc[:_half].max()), float(_rsi_20.iloc[_half:].max())
        if _p2_hi > _p1_hi and _r2_hi < _r1_hi:
            rsi_divergence = "bearish"  # 가격 고점 상승, RSI 고점 하락

    # 지지/저항 수준 (피봇 포인트 기반, 최근 20봉)
    support = None
    resistance = None
    if len(df) >= 20:
        _recent = df.iloc[-20:]
        _lows = _recent["low"].values
        _highs = _recent["high"].values
        # 피봇 저점/고점: 좌우 2봉보다 낮은/높은 봉
        for i in range(2, len(_lows) - 2):
            if _lows[i] < _lows[i-1] and _lows[i] < _lows[i-2] and _lows[i] < _lows[i+1] and _lows[i] < _lows[i+2]:
                if support is None or _lows[i] > support:  # 가장 가까운 지지
                    support = round(float(_lows[i]), 4)
            if _highs[i] > _highs[i-1] and _highs[i] > _highs[i-2] and _highs[i] > _highs[i+1] and _highs[i] > _highs[i+2]:
                if resistance is None or _highs[i] < resistance:  # 가장 가까운 저항
                    resistance = round(float(_highs[i]), 4)

    # EMA 크로스 감지 (최근 3봉 내 크로스)
    ema_cross = None
    if ema20 is not None and ema50 is not None and len(ema20.dropna()) >= 3:
        _e20_now, _e50_now = float(ema20.iloc[-1]), float(ema50.iloc[-1])
        _e20_prev, _e50_prev = float(ema20.iloc[-3]), float(ema50.iloc[-3])
        if _e20_prev <= _e50_prev and _e20_now > _e50_now:
            ema_cross = "golden"  # 골든크로스
        elif _e20_prev >= _e50_prev and _e20_now < _e50_now:
            ema_cross = "death"   # 데드크로스

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
        "rsi_divergence": rsi_divergence,
        "ema_cross":      ema_cross,
        "support":        support,
        "resistance":     resistance,
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
RSI 다이버전스: {indicators.get('rsi_divergence') or '없음'}
""".strip()


def generate_entry_chart(df: pd.DataFrame, symbol: str = "", side: str = "",
                         entry_price: float = None, sl: float = None, tp: float = None) -> str:
    """진입 시점 차트 PNG 생성 — 텔레그램 전송용. 저장 경로 반환."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pathlib import Path

    save_dir = Path(__file__).parent / "charts"
    save_dir.mkdir(exist_ok=True)
    # 오래된 차트 정리 (30개 유지)
    try:
        _files = sorted(save_dir.glob("*.png"), key=lambda f: f.stat().st_mtime)
        for _old in _files[:-30]:
            _old.unlink()
    except Exception:
        pass

    ohlc = df[["open", "high", "low", "close"]].tail(60).copy()
    ohlc.index = range(len(ohlc))
    if "time" in df.columns:
        times = pd.to_datetime(df["time"]).tail(60).values
    else:
        times = range(len(ohlc))

    # EMA
    ema20 = ta.ema(df["close"], length=20)
    ema50 = ta.ema(df["close"], length=50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.05})

    # 캔들차트 (간이)
    for i in range(len(ohlc)):
        o, h, l, c = ohlc.iloc[i]
        color = "#26a69a" if c >= o else "#ef5350"
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
        ax1.plot([i, i], [o, c], color=color, linewidth=3)

    if ema20 is not None:
        ax1.plot(range(len(ohlc)), ema20.tail(60).values, color="#2196F3", linewidth=1, label="EMA20")
    if ema50 is not None:
        ax1.plot(range(len(ohlc)), ema50.tail(60).values, color="#FF9800", linewidth=1, label="EMA50")

    # 진입가/SL/TP 라인
    if entry_price:
        ax1.axhline(entry_price, color="#1976D2", linewidth=1.5, linestyle="--", label=f"진입 ${entry_price:,.2f}")
    if sl:
        ax1.axhline(sl, color="#e53935", linewidth=1.2, linestyle=":", label=f"SL ${sl:,.2f}")
    if tp:
        ax1.axhline(tp, color="#4CAF50", linewidth=1.2, linestyle=":", label=f"TP ${tp:,.2f}")

    _side_kr = "롱" if "BUY" in (side or "").upper() or "롱" in (side or "") else "숏"
    ax1.set_title(f"{symbol} {_side_kr} 진입", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    # RSI
    rsi = ta.rsi(df["close"], length=14)
    if rsi is not None:
        ax2.plot(range(len(ohlc)), rsi.tail(60).values, color="#E91E63", linewidth=1)
        ax2.axhline(70, color="#888", linewidth=0.5, linestyle="--")
        ax2.axhline(30, color="#888", linewidth=0.5, linestyle="--")
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

    fname = f"{symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    fpath = str(save_dir / fname)
    fig.savefig(fpath, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fpath

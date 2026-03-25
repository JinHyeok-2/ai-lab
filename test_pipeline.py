#!/usr/bin/env python3
# 4대 기능 파이프라인 전체 검증 스크립트 (Streamlit 없이 독립 실행)

import sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor
from binance_client import get_klines
from indicators import calc_indicators, format_for_agent, generate_chart_image
from agents import run_agent
import trade_db

SYMBOL = "ETHUSDT"

def log(msg):
    print(f"  {msg}")

def parse_json(raw):
    import json, re
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        return json.loads(m.group())
    return {}

print("=" * 60)
print("  4대 기능 파이프라인 검증")
print("=" * 60)

# 1. 데이터 수집
print("\n[1/6] 데이터 수집 (15m + 1h + 4h)...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=3) as ex:
    f15 = ex.submit(get_klines, SYMBOL, "15m", 100)
    f1h = ex.submit(get_klines, SYMBOL, "1h", 50)
    f4h = ex.submit(get_klines, SYMBOL, "4h", 30)
    df_15m = f15.result()
    df_1h = f1h.result()
    df_4h = f4h.result()

ind_15m = calc_indicators(df_15m)
ind_1h = calc_indicators(df_1h)
ind_4h = calc_indicators(df_4h)
text_15m = format_for_agent(SYMBOL, ind_15m, label="15분봉")
text_1h = format_for_agent(SYMBOL, ind_1h, label="1시간봉")
text_4h = format_for_agent(SYMBOL, ind_4h, label="4시간봉")
log(f"OK ({time.time()-t0:.1f}s) | 가격: ${ind_15m['price']:,}")

# 2. 교훈 로드
print("\n[2/6] 거래 교훈 로드...")
lessons = trade_db.get_trade_lessons()
log(f"OK | {'교훈 있음' if lessons else '교훈 없음 (거래 기록 없음)'}")
if lessons:
    for line in lessons.split('\n')[:5]:
        log(f"  {line}")

# 3. 차트 이미지 생성
print("\n[3/6] 차트 이미지 생성...")
t0 = time.time()
chart_path = generate_chart_image(df_15m, SYMBOL)
log(f"OK ({time.time()-t0:.1f}s) | {chart_path}")

# 4. 에이전트 3개 병렬 (analyst + news + chart_analyst)
print("\n[4/6] 에이전트 병렬 실행 (analyst + news + chart_analyst)...")
analyst_input = f"""{SYMBOL} 3개 타임프레임 기술적 지표를 분석해주세요.
{text_15m}
{text_1h}
{text_4h}
3TF 컨플루언스: 15분(진입)/1시간(중기)/4시간(대추세) 신호가 일치할수록 신뢰도를 높게 평가해주세요."""

news_input = f"{SYMBOL} 현재 시장 심리와 최근 암호화폐 뉴스를 분석해주세요. 현재가: ${ind_15m.get('price', 'N/A')}"
chart_input = f"{SYMBOL} 15분봉 캔들차트를 분석해주세요. 패턴, 추세선, 지지/저항 구간을 식별해주세요."

t0 = time.time()
with ThreadPoolExecutor(max_workers=3) as ex:
    fa = ex.submit(run_agent, "analyst", analyst_input)
    fn = ex.submit(run_agent, "news", news_input)
    fc = ex.submit(run_agent, "chart_analyst", chart_input, chart_path)
    analyst_result = fa.result()
    news_result = fn.result()
    chart_result = fc.result()
elapsed = time.time() - t0
log(f"OK ({elapsed:.1f}s)")
log(f"analyst: {analyst_result[:100]}...")
log(f"news: {news_result[:100]}...")
log(f"chart: {chart_result[:100]}...")

# 5. risk -> gate
print("\n[5/6] risk -> gate 순차 실행...")
risk_input = f"""다음 분석 결과를 바탕으로 리스크를 평가해주세요.
심볼: {SYMBOL}
{text_15m}
{text_4h}
[기술적 분석 결과]
{analyst_result}
[시장 심리 결과]
{news_result}
{lessons}"""

t0 = time.time()
risk_result = run_agent("risk", risk_input)
log(f"risk OK ({time.time()-t0:.1f}s): {risk_result[:100]}...")

gate_input = f"""다음 리스크 평가를 비판적으로 재검토하고, 지금 거래해야 하는지 판단해주세요.
심볼: {SYMBOL}
[리스크 평가]
{risk_result}
{lessons}"""

t0 = time.time()
gate_result = run_agent("gate", gate_input)
gate_json = parse_json(gate_result)
log(f"gate OK ({time.time()-t0:.1f}s)")
log(f"  should_trade: {gate_json.get('should_trade')}")
log(f"  reason: {gate_json.get('reason')}")
log(f"  risk_level: {gate_json.get('risk_level')}")

# 6. trader (gate 통과 시만)
print("\n[6/6] trader (gate 결과에 따라)...")
gate_passed = gate_json.get("should_trade", True)
if gate_passed:
    trader_input = f"""다음 분석을 종합해 최종 매매 결정을 내려주세요.
심볼: {SYMBOL}
[기술적 분석 (3TF)]
{analyst_result}
[4시간봉 지표]
{text_4h}
[시장 심리]
{news_result}
[차트 패턴 분석]
{chart_result}
[리스크 평가]
{risk_result}
{lessons}"""
    t0 = time.time()
    trader_result = run_agent("trader", trader_input)
    trader_json = parse_json(trader_result)
    log(f"trader OK ({time.time()-t0:.1f}s)")
    log(f"  signal: {trader_json.get('signal')}")
    log(f"  confidence: {trader_json.get('confidence')}")
    log(f"  reason: {trader_json.get('reason')}")
    log(f"  entry: {trader_json.get('entry')} / SL: {trader_json.get('sl')} / TP: {trader_json.get('tp')}")
else:
    log("gate 차단 -> trader 생략 (wait)")

print("\n" + "=" * 60)
print("  검증 완료!")
print("=" * 60)

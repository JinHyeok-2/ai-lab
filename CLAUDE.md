# CLAUDE.md — 00.ai-lab (AI 트레이딩 봇)

## 프로젝트 개요

바이낸스 선물거래 멀티에이전트 트레이딩 봇 (Streamlit GUI)

- GitHub: `https://github.com/JinHyeok-2/ai-lab.git` (private)
- 접속: `http://168.188.99.183:8002/proxy/8503/` (Jupyter 프록시 경유)

## 실행 환경

- **conda 환경**: 반드시 `ai-lab` (Python 3.12) 사용
  - 경로: `/home/hyeok/.conda/envs/ai-lab/bin/python`
  - 시스템 Python 사용 금지 (numpy/pyarrow 충돌 발생)
- **Streamlit 실행**: `conda run -n ai-lab python -m streamlit run app.py --server.port 8503`
- **패키지 설치**: `conda run -n ai-lab pip install <패키지>`

## 프로젝트 구조

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit GUI (메인 진입점, ~4700줄) |
| `config.py` | 설정 (TESTNET, SYMBOLS, LEVERAGE, MAX_USDT 등) |
| `binance_client.py` | 바이낸스 선물 API 래퍼 (싱글턴, 캐싱, 재시도) |
| `agents.py` | 5개 에이전트 (analyst, news, risk, gate, trader) + Claude CLI subprocess |
| `indicators.py` | 기술적 지표 계산 (20+종, RSI 다이버전스 포함) |
| `alt_scanner.py` | 알트코인 스캐너 + 거래소 공지 감지 (바이낸스/업비트/OKX/코인베이스) |
| `surge_detector.py` | 바이낸스 선물 실시간 급등/급락 감지 (REST 폴링 + 스냅샷 비교) |
| `whale_tracker.py` | 고래 추적 (Telethon + Whale Alert 텔레그램 채널) |
| `telegram_notifier.py` | 텔레그램 알림 (15+ 함수, 6개 명령어) |
| `trade_db.py` | 거래 기록 SQLite 저장소 (`trades.db`) |
| `.env` | API 키 (바이낸스, 텔레그램, Telegram User API) — 절대 커밋 금지 |
| `rl/` | 강화학습 모델 (v1~v6) |
| `rl-lab/` | RL 개선 실험실 (서브 프로젝트) |

## 현재 설정 (config.py)

- `TESTNET=False` (실거래 모드)
- `SYMBOLS=["ETHUSDT", "BTCUSDT"]`
- `LEVERAGE=3`, `MAX_USDT=15`, `MAX_DAILY_LOSS=20`
- `MAX_SINGLE_LOSS_PCT=3.0` (건당 잔고 대비 3%)
- 알트: `MAX_USDT_ALT=10`, `ALT_LEVERAGE=2`, 동시 최대 2포지션
- 에이전트: analyst/news/risk/trader=sonnet, gate=haiku

## 파이프라인

```
데이터 수집 [병렬 7개: 15m+1h+4h+뉴스RSS+공포탐욕+펀딩비+OI]
→ 지표 계산(15m+1h+4h) + 규칙 시그널(10개 규칙: EMA/RSI/Stoch/ADX/MACD/1H/BB/OBV/OI/CVD + 4H TF + RSI다이버전스)
→ 사전 필터: 혼조+ADX<15+연속차단3회 → 에이전트 스킵
→ [병렬] analyst(3TF) + news(RSS+웹검색)  ← 응답 시간/품질 추적
→ risk (양방향 시나리오 + 교훈)
→ gate (규칙 기반 우선: 명확→자동통과/차단, 경계→haiku LLM)
→ [통과 시만] trader (방향+가격만 결정, confidence=50 고정)
→ calc_confidence (22개 지표 기반 독립 채점 + 캘리브레이션 보정)
→ 에이전트 합의 메커니즘 (3/3 동의 보너스 +8pt, 반대 존재 -5pt)
→ 교차 검증 (규칙↔LLM 일치 +8pt, 불일치 -10pt)
→ 역발상 보정 (과매도+롱 +5pt, 과매도+숏 -10pt 등)
→ 변동성 레짐 적응 (ADX<15→최소75%, ADX≥30→최소60%)
→ 드로다운 기반 포지션 축소 (DD 3%→50%, DD 5%→30%)
→ 진입 판단 (_should_trade: 신뢰도 임계값, 캔들 경계 확인)
```

## 완료된 개선사항

### 2026-03-20 (기반 구축)
| # | 항목 |
|---|------|
| 1 | SL/TP 부분 청산 후 SL 업데이트 실패 버그 수정 |
| 2 | 에러 핸들링 강화 — API 3회 연속 실패 시 비상 전량 청산 |
| 3 | 일일 손실 한도 — 한도 초과 시 전량 청산 + daily_balance SQLite 영속화 |
| 6 | 백테스트 재검증 — v5 30m: -0.17%, MDD -16%, 23거래, 승률 61% |
| 9 | systemd 자동 재시작 — `ai-trading-bot.service`, linger 활성화 |
| 10 | 거래 로그 SQLite 전환 — `trade_db.py` + `trades.db` |
| 11 | PnL 대시보드 — 일별/심볼별/시간대별 차트 + 신뢰도 캘리브레이션 |

### 2026-03-22 오전 (실거래 전환 + 핵심 수정)
| # | 항목 |
|---|------|
| 7 | 텔레그램 알림 — 8종 + 잔고/진입금 표시 정상 확인 |
| 8 | 실거래 보수적 설정 — BTC/ETH x3 $15, 알트 x2 $10, 일일한도 $20 |
| 12 | trading_paused 영속화 — bot_state 테이블, 재시작 후 복원 |
| 13 | 정시 브리핑 SQL 버그 수정 |
| 14 | 알트 SL 미배치 안전장치 |
| 15 | gate 연속 차단 쿨다운 |
| 16 | PnL 부분 청산 합산 정확도 개선 |
| 17 | 일별 잔고 히스토리 — balance_history 테이블 |
| 18 | 실거래 API 전환 — TESTNET=False, $143.90 시작 |
| 19 | 캘리브레이션 기반 신뢰도 보정 강화 |
| 20 | 숏 편향 완화 — 과매수 조건 엄격화 |
| 21 | trader confidence 이중 채점 제거 |
| 22 | gate 규칙 기반 우선 — 비용 절감 |
| 23 | 고스트 거래 통계 필터 |
| 24 | 블록 시간대 동적화 |
| 25 | 규칙 시그널에 OI/CVD 추가 |
| 26 | trader 재시도 프롬프트 변형 |

### 2026-03-22 오후 (16개 개선 일괄 적용)
| # | 항목 |
|---|------|
| 27 | pnl=0 고스트 필터 강화 — close_price=0/NULL 조건 추가 |
| 28 | obv_slope → obv_trend 버그 수정 + OBV 규칙 추가 |
| 29 | 고신뢰도(80+) 과신뢰 방지 — 데이터 5건 미만 시 70% 캡 |
| 30 | 일일마감/주간 리포트 session state 키 초기화 |
| 31 | get_balance() 10초 TTL 캐싱 |
| 32 | 4h 지표 규칙 시그널 (EMA/RSI/MACD 3종 추가) |
| 33 | 대형 손실 방지 — SL 미설정 차단, 최소거리 0.5% 검증 |
| 34 | RSI 다이버전스 탐지 + 규칙 10 추가 |
| 35 | 슬리피지 분석 대시보드 (trade_db + UI) |
| 36 | 변동성 레짐 적응 — ADX 기반 최소 신뢰도 동적 조절 |
| 37 | 에이전트 응답 품질 모니터링 (시간/성공률 추적) |
| 38 | 드로다운 기반 동적 포지션 축소 (DD 3%/5% 단계) |
| 39 | 에이전트 합의 메커니즘 (3/3 합의 보너스) |
| 40 | 게이트 성과 통계 추적 (5회마다 로그) |
| 41 | 메인 vs 알트 성과 분리 통계 |
| 42 | DB 초기화 — 실거래 첫 거래부터 깨끗한 통계 시작 |

### 2026-03-22 저녁 (고래추적 + 알트 개선 + 멀티 거래소 공지)
| # | 항목 |
|---|------|
| 43 | 고래 추적 — Telethon + Whale Alert 채널, 파이프라인 통합 (🐋 탭) |
| 44 | 알트 독립 신뢰도 채점 (calc_confidence_alt, 10요소) |
| 45 | BTC 상관 필터 — BTC 역방향 시 알트 진입 차단 |
| 46 | 알트 gate/4h/쿨다운 필터 |
| 47 | 업비트 상장 공지 감지 — API 폴링 + 자동 분석/진입 |
| 48 | OKX 상장/상폐 공지 감지 — REST API 폴링 |
| 49 | 코인베이스 신규 상장 감지 — 상품 목록 캐시 비교 |
| 50 | 실시간 급등/급락 감지 — REST 스냅샷 비교 (surge_detector.py) |

### 2026-03-23 오후 (RL 검증 + 가중치 + 리포트 차트)
| # | 항목 |
|---|------|
| 61 | Walk-forward 검증 — 2개월 롤링 윈도우, 3/4 구간 수익, 과적합 가능성 낮음 확인 |
| 62 | 앙상블 조합 탐색 완료 — 20개 조합 비교, exp14+exp08+seed700 만장일치 1위 (+1490%, MDD -3.1%) |
| 63 | RL 시그널 가중치 강화 — RL반대 -8pt, RL관망 -3pt 페널티 추가 |
| 64 | get_rl_signal() hold_val/cooldown_val 구현 — DB 기반 실거래 상태 추적 |
| 65 | 데이터 파생 컬럼 재계산 — eth_30m_v41.csv 176캔들 추가 + rsi/macd 재계산 |
| 66 | 거래 분석 차트 자동 생성 — trade_report.py에 6패널 차트 첨부 |
| 67 | 롱 전용 RL 환경 + 학습 — v1 파산, v2 과다매매, 보류 |
| 68 | 프로덕션 앙상블 교체 — exp05→seed700, 다수결→만장일치+폴백 |
| 69 | 숏 행동 마스킹 — 투표 전 숏→관망 치환 (롱전용 운영) |
| 70 | BTC RL 데이터 수집 + 학습 시작 — btc_30m.csv 39,513캔들, GPU 학습 중 |
| 71 | 데이터 자동 갱신 크론 — 매일 01:03 ETH+BTC 캔들 다운로드 |
| 72 | 모델 정리 — 94MB→27MB, 불필요 19개 삭제, 8개 보존 |

### 미완료
| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 70 | BTC RL 모델 학습 | 🔄 진행 중 | GPU ~2시간 |

## 핵심 안전장치

- SL 최소 0.5% 거리 검증, SL 미설정 시 진입 차단
- R:R 최소 1:1.5 (자동 보정), SL 미배치 시 즉시 청산
- 건당 손실 한도 잔고의 3%, 일일 손실 한도 $20
- 연속 3손실 쿨다운 2시간, 진입금 50% 축소
- 일중 드로다운 3%→50% 축소, 5%→30% 축소
- 변동성 필터: ATR > 평균 × 동적배수
- 변동성 레짐: ADX<15 횡보→최소 신뢰도 75%, ADX≥30 추세→60%
- 포트폴리오 한도: 메인 총 익스포저 $60 이하
- 고신뢰도(80+) 과신뢰 방지: 데이터 5건 미만 시 70% 캡

## 작업 규칙

- compact 후 체크리스트 진행 상황 확인 → 다음 작업 추천
- `.env` 파일은 읽기만, 수정 시 사용자 확인 필수

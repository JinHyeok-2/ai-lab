# ETH RL 트레이딩 에이전트 — 서버 이관 가이드

## 이관할 파일 목록

```
trading/rl/
├── data.py          ← 바이낸스 데이터 수집
├── env.py           ← 강화학습 환경
├── train.py         ← PPO 학습
├── backtest.py      ← 백테스트
├── __init__.py
├── requirements.txt ← 패키지 목록
├── eth_30m.csv      ← 30분봉 데이터 (17,454개, 2025-03~2026-03)
├── eth_1h.csv       ← 1시간봉 데이터 (8,725개)
├── eth_4h.csv       ← 4시간봉 데이터 (2,955개)
├── eth_2h.csv       ← 2시간봉 데이터 (4,339개)
├── eth_15m.csv      ← 15분봉 데이터 (8,651개)
├── eth_1d.csv       ← 일봉 데이터 (682개)
└── models/
    ├── v1/          ← 첫 번째 학습 (1h, 4h, 2h, 15m, 1d)
    ├── v2/          ← 개선 버전 (30m: +26%, MDD -3.9%, 승률 85.7%)
    ├── v3/          ← 60/40 split
    ├── v4/          ← train-end 2025-09-01 (상승장 학습)
    └── v5/          ← 70/30 split
```

또한 상위 폴더에서 config.py 필요:
```
trading/config.py    ← API 키 (직접 입력 필요)
```

---

## 1단계: 서버에 파일 전송 (로컬에서 실행)

```bash
# 서버 IP와 경로 설정
SERVER="user@your-server-ip"
DEST="~/ai-lab/trading"

# rl 폴더 전체 전송 (모델 포함)
scp -r /Users/sunny/Desktop/ai-lab/trading/rl $SERVER:$DEST/

# config.py 전송 (API 키 포함 — 주의)
scp /Users/sunny/Desktop/ai-lab/trading/config.py $SERVER:$DEST/
```

또는 tar로 압축 후 전송:
```bash
cd /Users/sunny/Desktop/ai-lab/trading
tar -czf rl_package.tar.gz rl/ config.py
scp rl_package.tar.gz $SERVER:~/
```

---

## 2단계: 서버 환경 설정

```bash
# 가상환경 생성 (권장)
conda create -n trading python=3.11
conda activate trading

# 또는 venv
python -m venv trading_env
source trading_env/bin/activate

# 패키지 설치
pip install -r trading/rl/requirements.txt

# PyTorch는 서버 GPU 환경에 맞게 설치
# CPU 서버:
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# GPU 서버 (CUDA 12.1):
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## 3단계: config.py 설정

```python
# trading/config.py
API_KEY    = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_API_SECRET"
```

---

## 4단계: 동작 확인

```bash
cd ~/ai-lab

# 데이터 로드 확인
python -c "from trading.rl.data import load_data; df = load_data('30m'); print(df.shape)"

# 백테스트 확인 (기존 모델 v2)
python trading/rl/backtest.py --interval 30m --version v2

# 새 학습 시작 (서버에서)
python trading/rl/train.py --interval 30m --steps 1000000 --version v6 --split 0.8
```

---

## 학습 명령어 모음

```bash
# 30m 모델 (핵심)
python trading/rl/train.py --interval 30m --steps 1000000 --version v6

# 비율 지정
python trading/rl/train.py --interval 30m --steps 1000000 --version v7 --split 0.6

# 날짜 지정 (특정 기간만 학습)
python trading/rl/train.py --interval 30m --steps 1000000 --version v8 --train-end 2025-09-01

# 백테스트
python trading/rl/backtest.py --interval 30m --version v6

# 특정 기간 테스트
python trading/rl/backtest.py --interval 30m --version v6 --test-start 2025-10-01
```

---

## 현재 모델 성능 현황

| 버전 | 타임프레임 | 수익률 | MDD | 거래수 | 승률 | 비고 |
|------|-----------|--------|-----|--------|------|------|
| v1 | 1h | +13.4% | -8.1% | 3 | 33.3% | 하락장 |
| v1 | 4h | +7.5% | -16.3% | 3 | 66.7% | 하락장 |
| v1 | 30m | +126% | -14.6% | 26 | 76.9% | 상승장 포함 |
| **v2** | **30m** | **+26%** | **-3.9%** | 7 | **85.7%** | **가장 신뢰** |
| v3 | 30m | - | - | - | - | 60/40 split |
| v4 | 30m | - | - | - | - | 상승장 학습 |
| v5 | 30m | - | - | - | - | 70/30 split |

---

## 환경 구조 (env.py 요약)

- 행동: 0=관망, 1=롱, 2=숏, 3=청산
- 레버리지: 5x, 수수료: 0.04%
- 특징(features): price_chg, rsi_norm, macd_norm, bb_pct, ema_ratio, atr_norm, vol_ratio + 포지션/손익/보유시간/쿨다운
- 과다매매 방지: min_hold_steps, cooldown(5캔들), 짧은 보유 패널티
- window_size=20, n_features=11, obs_shape=(220,)

---

## 주의사항

1. **API 키**: config.py는 실거래 바이낸스 키 (klines는 공개 데이터라 학습/데이터수집에 사용)
2. **모델 삭제 금지**: 기존 버전은 보존, 항상 새 버전 번호 사용
3. **메타데이터**: 각 버전 폴더에 `meta_{interval}.json` 자동 저장됨 (학습 기간 기록)
4. **그래프**: backtest 실행 시 `plt.show()` → 서버에서는 에러 날 수 있음

서버에서 그래프 저장만 하고 표시 안 하려면 backtest.py 마지막 줄 수정:
```python
# plt.show()  ← 이 줄 주석 처리
```

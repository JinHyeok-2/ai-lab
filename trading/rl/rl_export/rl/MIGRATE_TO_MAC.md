# v4 모델 → iMac 이관 가이드

## 포함된 파일
```
rl/
├── env.py              ← RL 환경 (v6 개선버전, legacy 호환)
├── train.py            ← PPO 학습
├── backtest.py         ← 백테스트 (macOS 자동감지: AppleGothic, plt.show())
├── data.py             ← 바이낸스 데이터 수집
├── compare_versions.py ← 다중 버전 병렬 비교
├── __init__.py
├── requirements.txt
├── eth_30m.csv         ← 30분봉 데이터 (2025-03 ~ 2026-03, 17,454개)
└── models/
    └── v4/
        ├── ppo_eth_30m.zip   ← 학습된 모델 (★ 최고 성능 +36.1%)
        └── meta_30m.json     ← 학습 메타데이터
```

---

## 1단계: 파일 압축 해제

```bash
cd ~/Desktop   # 원하는 위치로 이동
tar -xzf v4_mac_export.tar.gz
cd rl_export
```

---

## 2단계: conda 환경 생성

```bash
conda create -n trading python=3.11
conda activate trading
pip install -r requirements.txt
```

> **Apple Silicon(M1/M2/M3)** PyTorch 설치:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
>
> **Intel Mac** PyTorch 설치 (requirements.txt에 포함):
> ```bash
> pip install torch==2.10.0
> ```

---

## 3단계: 동작 확인

```bash
# 압축 해제 폴더의 상위에서 실행
cd ~/Desktop/rl_export

# v4 백테스트 (그래프 자동 표시)
python rl/backtest.py --interval 30m --version v4 --test-start 2025-09-01

# 버전 비교
python rl/compare_versions.py --versions v4
```

---

## 4단계: config.py 설정 (새 데이터 수집 시)

```python
# rl_export/config.py  (상위 폴더에 생성)
API_KEY    = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_API_SECRET"
```

새 데이터 수집:
```bash
python rl/data.py --interval 30m
```

---

## 5단계: 추가 학습 (서버 대신 맥에서 할 경우)

```bash
# v7 학습 (서버에서 하는 게 더 빠름)
python rl/train.py --interval 30m --steps 1000000 --version v7
```

> macOS는 MPS(Apple Silicon GPU) 지원:
> `train.py`의 `device="cpu"` → `device="mps"` 로 변경하면 M1/M2/M3에서 GPU 가속 가능

---

## 현재 모델 성능 (2026-03-19 기준)

| 버전 | 수익률 | MDD | 거래 | 승률 | 테스트기간 |
|------|--------|-----|------|------|-----------|
| **v4** ★ | **+36.1%** | -4.6% | 15회 | 80% | 2025-09-01 ~ 2026-03-18 |
| v2 | +26.1% | -3.9% | 7회 | 85.7% | 2026-01-04 ~ 2026-03-18 |
| v3 | +24.9% | -1.7% | 12회 | 75% | 2025-10-24 ~ 2026-03-18 |
| v5 | +14.8% | -38% | 210회 | 55.7% | 2025-11-29 ~ 2026-03-18 |

---

## 서버에서 진행 중인 작업

- **v6 학습 중** (3x 레버리지, 9특징, VecNormalize, 4개 병렬환경)
- v6 완료 후 → v7 (Trailing Stop + BTC 보조 특징) 예정
- 최종 목표: 앙상블 (v4 + v6 + v7)

---

## 주의사항

1. **실행 위치**: 반드시 `rl/` 폴더의 **상위 폴더**에서 `python rl/backtest.py ...` 형태로 실행
2. **모델 삭제 금지**: 새 학습 시 새 버전 번호 사용 (v7, v8, ...)
3. **v4 모델 구버전**: `_legacy=True` 자동 적용 (7특징, window=20, obs=220)

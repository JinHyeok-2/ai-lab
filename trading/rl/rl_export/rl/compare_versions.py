#!/usr/bin/env python3
# 여러 버전 모델을 병렬로 백테스트하고 결과를 비교 테이블로 출력
# 실행 예시:
#   python rl/compare_versions.py
#   python rl/compare_versions.py --versions v2 v3 v4 v5 v6
#   python rl/compare_versions.py --interval 1h --versions v1

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import platform
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from rl.data import load_data
from rl.env import ETHTradingEnv


def _run_single(args: dict) -> dict:
    """단일 버전 백테스트 (서브프로세스에서 실행)"""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.data import load_data
    from rl.env import ETHTradingEnv
    from rl.train import MIN_HOLD

    version  = args["version"]
    interval = args["interval"]
    base     = Path(__file__).parent / "models" / version

    model_path   = base / f"ppo_eth_{interval}.zip"
    meta_path    = base / f"meta_{interval}.json"
    vecnorm_path = base / f"vecnorm_{interval}.pkl"

    if not model_path.exists():
        return {"version": version, "interval": interval, "error": "모델 없음"}

    # 메타 로드
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    leverage    = meta.get("leverage",    5)   # v2 이전 기본값 5
    window_size = meta.get("window_size", 20)  # v2 이전 기본값 20

    df = load_data(interval)

    # 테스트 구간 결정
    if meta.get("train_candles"):
        split_idx = meta["train_candles"]
    else:
        split_idx = int(len(df) * 0.8)

    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # 모델 로드 후 obs_size 자동 감지 → 레거시 모드 결정
    model   = PPO.load(str(model_path))
    obs_size = model.observation_space.shape[0]
    # obs_size = window_size × n_features
    # 레거시: 20 × 11 = 220 / 신버전: 40 × 13 = 520
    is_legacy = (obs_size != window_size * 13)

    # 환경 생성 (obs_size 일치하도록 레거시 모드 적용)
    env = ETHTradingEnv(
        test_df,
        initial_balance=10000.0,
        leverage=leverage,
        window_size=window_size,
        min_hold_steps=MIN_HOLD.get(interval, 3),
        _legacy=is_legacy,
    )

    # VecNormalize (v6 이후)
    vecnorm = None
    if vecnorm_path.exists() and not is_legacy:
        dummy = DummyVecEnv([lambda: ETHTradingEnv(
            test_df, initial_balance=10000.0, leverage=leverage,
            window_size=window_size, min_hold_steps=MIN_HOLD.get(interval, 3),
            _legacy=False,
        )])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy)
        vecnorm.training    = False
        vecnorm.norm_reward = False

    # 백테스트 실행
    obs, _ = env.reset()
    balance_history = [env.initial_balance]
    actions_taken   = []
    done = False

    while not done:
        obs_input = vecnorm.normalize_obs(obs.reshape(1, -1))[0] if vecnorm else obs
        action, _ = model.predict(obs_input, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        balance_history.append(info["balance"])
        actions_taken.append(int(action))

    # 성과 계산
    final_balance = info["balance"]
    total_return  = (final_balance - 10000) / 10000 * 100
    buy_hold_ret  = (test_df["close"].iloc[-1] - test_df["close"].iloc[env.window_size]) \
                    / test_df["close"].iloc[env.window_size] * 100

    balance_arr = np.array(balance_history)
    peak        = np.maximum.accumulate(balance_arr)
    mdd         = ((balance_arr - peak) / peak * 100).min()

    from collections import Counter
    dist  = Counter(actions_taken)
    total = len(actions_taken)

    return {
        "version":      version,
        "interval":     interval,
        "leverage":     leverage,
        "test_start":   str(test_df["time"].iloc[0].date()),
        "test_end":     str(test_df["time"].iloc[-1].date()),
        "total_return": round(total_return, 2),
        "buy_hold_ret": round(buy_hold_ret, 2),
        "mdd":          round(mdd, 2),
        "win_rate":     round(info["win_rate"] * 100, 1),
        "total_trades": info["total_trades"],
        "act_hold":     round(dist.get(0, 0) / total * 100, 1),
        "act_long":     round(dist.get(1, 0) / total * 100, 1),
        "act_short":    round(dist.get(2, 0) / total * 100, 1),
        "act_close":    round(dist.get(3, 0) / total * 100, 1),
        "error":        None,
    }


def compare(versions: list, interval: str = "30m", max_workers: int = 4):
    print(f"\n{'='*60}")
    print(f"  버전 비교 백테스트  [{interval}]  —  {len(versions)}개 버전")
    print(f"{'='*60}\n")

    # 병렬 실행
    tasks  = [{"version": v, "interval": interval} for v in versions]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single, t): t["version"] for t in tasks}
        for future in as_completed(futures):
            ver = futures[future]
            try:
                res = future.result()
                results.append(res)
                status = f"오류: {res['error']}" if res["error"] else f"완료 ({res['total_return']:+.1f}%)"
                print(f"  [{ver}] {status}")
            except Exception as e:
                results.append({"version": ver, "interval": interval, "error": str(e)})
                print(f"  [{ver}] 실패: {e}")

    # 정상 결과만 분리
    ok  = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]

    if not ok:
        print("\n정상 결과 없음.")
        return

    # 수익률 기준 정렬
    ok.sort(key=lambda r: r["total_return"], reverse=True)

    # ── 성과 테이블 ─────────────────────────────────────────────────
    col_w = 10
    header = (f"\n{'버전':^6} {'레버리지':^6} {'테스트기간':^22} "
              f"{'수익률':^9} {'B&H':^9} {'MDD':^9} {'거래':^5} {'승률':^7}")
    sep    = "-" * len(header)
    print(f"\n{'[ 성과 요약 ]':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for r in ok:
        ret_str = f"{r['total_return']:+.2f}%"
        bh_str  = f"{r['buy_hold_ret']:+.2f}%"
        mdd_str = f"{r['mdd']:.2f}%"
        period  = f"{r['test_start']} ~ {r['test_end']}"
        flag    = " ★" if r == ok[0] else ""
        print(f"  {r['version']:^4}  {r['leverage']:^6}x  {period:^22}  "
              f"{ret_str:^9} {bh_str:^9} {mdd_str:^9} {r['total_trades']:^5} "
              f"{r['win_rate']:^5.1f}%{flag}")
    print(sep)

    # ── 행동 분포 테이블 ────────────────────────────────────────────
    print(f"\n{'[ 행동 분포 (%) ]':^{len(header)}}")
    print(sep)
    print(f"{'버전':^6} {'관망':^10} {'롱':^10} {'숏':^10} {'청산':^10}")
    print(sep)
    for r in ok:
        print(f"  {r['version']:^4}  {r['act_hold']:^8.1f}  {r['act_long']:^8.1f}  "
              f"{r['act_short']:^8.1f}  {r['act_close']:^8.1f}")
    print(sep)

    # ── 오류 목록 ───────────────────────────────────────────────────
    if err:
        print("\n[ 오류 목록 ]")
        for r in err:
            print(f"  {r['version']}: {r['error']}")

    print()
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions",  nargs="+", default=["v2","v3","v4","v5"],
                        help="비교할 버전 목록 (기본: v2 v3 v4 v5)")
    parser.add_argument("--interval",  default="30m",
                        choices=["15m","30m","1h","2h","4h","1d"],
                        help="캔들 타임프레임 (기본: 30m)")
    parser.add_argument("--workers",   type=int, default=4,
                        help="병렬 프로세스 수 (기본: 4)")
    args = parser.parse_args()

    compare(versions=args.versions, interval=args.interval, max_workers=args.workers)

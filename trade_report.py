#!/usr/bin/env python3
# 거래 분석 리포트 생성기 — trades.db 기반 통계 분석 → .txt 저장
# 용도: 일일 자동 실행 + 에이전트 컨텍스트 주입

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import trade_db
import json
from datetime import datetime, timedelta

REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def _ghost_filter(trades):
    """고스트 거래 제외 (pnl=0 + close_price 없음)"""
    return [t for t in trades if not (
        (t.get("pnl") or 0) == 0 and (
            not t.get("close_price") or t.get("close_price") == 0
        )
    )]


def _is_long(side):
    s = (side or "").upper()
    return "롱" in s or "LONG" in s or "BUY" in s


def _is_short(side):
    s = (side or "").upper()
    return "숏" in s or "SHORT" in s or "SELL" in s


def _hold_minutes(t):
    """보유 시간(분) 계산"""
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        open_t = datetime.strptime(t["time"], fmt)
        close_t = datetime.strptime(t["close_time"], fmt)
        return (close_t - open_t).total_seconds() / 60
    except Exception:
        return None


def generate_report():
    """전체 분석 리포트 생성 → .txt 저장, 리포트 텍스트 반환"""
    closed = trade_db.get_closed_trades(limit=9999)
    closed = _ghost_filter(closed)

    if not closed:
        return "거래 데이터 없음 — 리포트 생성 불가"

    now = datetime.now()
    lines = []
    lines.append("=" * 60)
    lines.append(f"  거래 분석 리포트 — {now.strftime('%Y-%m-%d %H:%M KST')}")
    lines.append("=" * 60)
    lines.append(f"  분석 대상: {len(closed)}건 (고스트 제외)")
    lines.append("")

    # ── 1. 전체 통계 ──
    stats = trade_db.get_trade_stats()
    if stats:
        lines.append("[ 1. 전체 통계 ]")
        lines.append(f"  거래: {stats['total']}건 | 승: {stats['wins']} | 패: {stats['losses']}")
        lines.append(f"  승률: {stats['win_rate']:.1f}%")
        lines.append(f"  총 PnL: ${stats['total_pnl']:.2f}")
        lines.append(f"  평균 익절: ${stats['avg_win']:.3f} | 평균 손절: ${stats['avg_loss']:.3f}")
        lines.append(f"  Profit Factor: {stats['profit_factor']:.2f}")
        lines.append(f"  MDD: ${stats['mdd']:.2f}")
        lines.append("")

    # ── 2. 방향별 분석 ──
    longs = [t for t in closed if _is_long(t.get("side"))]
    shorts = [t for t in closed if _is_short(t.get("side"))]
    lines.append("[ 2. 방향별 성과 ]")
    for label, group in [("롱", longs), ("숏", shorts)]:
        if group:
            wins = sum(1 for t in group if (t.get("pnl") or 0) > 0)
            total_pnl = sum(t.get("pnl") or 0 for t in group)
            avg_pnl = total_pnl / len(group)
            wr = wins / len(group) * 100
            lines.append(f"  {label}: {len(group)}건 | 승률 {wr:.0f}% | PnL ${total_pnl:+.2f} | 평균 ${avg_pnl:+.3f}")
        else:
            lines.append(f"  {label}: 0건")
    if longs and shorts:
        l_wr = sum(1 for t in longs if (t.get("pnl") or 0) > 0) / len(longs) * 100
        s_wr = sum(1 for t in shorts if (t.get("pnl") or 0) > 0) / len(shorts) * 100
        if l_wr > s_wr + 15:
            lines.append(f"  >> 교훈: 롱이 숏보다 승률 {l_wr - s_wr:.0f}%p 우세 — 롱 우선 전략 유효")
        elif s_wr > l_wr + 15:
            lines.append(f"  >> 교훈: 숏이 롱보다 승률 {s_wr - l_wr:.0f}%p 우세")
    lines.append("")

    # ── 3. 심볼별 분석 ──
    sym_stats = trade_db.get_symbol_summary()
    if sym_stats:
        lines.append("[ 3. 심볼별 성과 ]")
        for s in sym_stats:
            wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            flag = " ★" if wr >= 50 and s["trades"] >= 3 else ""
            flag = " ✗" if wr < 25 and s["trades"] >= 3 else flag
            lines.append(f"  {s['symbol']:12s} | {s['trades']:3d}건 | 승률 {wr:5.1f}% | PnL ${s['total_pnl']:+.2f}{flag}")
        # 블랙리스트 후보
        blacklist = [s["symbol"] for s in sym_stats if s["trades"] >= 3 and s["wins"] / s["trades"] < 0.25]
        if blacklist:
            lines.append(f"  >> 교훈: 블랙리스트 후보 (3건+ 승률25%미만): {', '.join(blacklist)}")
        # 우수 종목
        good = [s["symbol"] for s in sym_stats if s["trades"] >= 3 and s["wins"] / s["trades"] >= 0.5]
        if good:
            lines.append(f"  >> 교훈: 우수 종목 (3건+ 승률50%이상): {', '.join(good)}")
        lines.append("")

    # ── 4. 소스별 분석 ──
    src_perf = trade_db.get_source_performance(recent_n=9999)
    if src_perf:
        lines.append("[ 4. 소스별 성과 ]")
        for src, s in sorted(src_perf.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
            lines.append(f"  {src:20s} | {s['trades']:3d}건 | 승률 {s['win_rate']:5.1f}% | PnL ${s['total_pnl']:+.2f}")
        bad_src = [src for src, s in src_perf.items() if s["trades"] >= 3 and s["win_rate"] < 25]
        if bad_src:
            lines.append(f"  >> 교훈: 저성과 소스 (승률25%미만): {', '.join(bad_src)} — 차단 검토")
        lines.append("")

    # ── 5. 신뢰도 구간별 분석 ──
    conf_bins = [(0, 49), (50, 59), (60, 69), (70, 79), (80, 100)]
    lines.append("[ 5. 신뢰도 구간별 성과 ]")
    for lo, hi in conf_bins:
        group = [t for t in closed if lo <= (t.get("confidence") or 0) <= hi]
        if group:
            wins = sum(1 for t in group if (t.get("pnl") or 0) > 0)
            total_pnl = sum(t.get("pnl") or 0 for t in group)
            wr = wins / len(group) * 100
            lines.append(f"  {lo:3d}~{hi:3d}% | {len(group):3d}건 | 승률 {wr:5.1f}% | PnL ${total_pnl:+.2f}")
    # 최적 구간 찾기
    best_bin = None
    best_wr = 0
    for lo, hi in conf_bins:
        group = [t for t in closed if lo <= (t.get("confidence") or 0) <= hi]
        if len(group) >= 3:
            wr = sum(1 for t in group if (t.get("pnl") or 0) > 0) / len(group) * 100
            if wr > best_wr:
                best_wr = wr
                best_bin = f"{lo}~{hi}%"
    if best_bin:
        lines.append(f"  >> 교훈: 최적 신뢰도 구간 {best_bin} (승률 {best_wr:.0f}%)")
    lines.append("")

    # ── 6. 시간대별 분석 (KST 2시간 단위) ──
    lines.append("[ 6. 시간대별 성과 (KST) ]")
    hour_groups = {}
    for t in closed:
        try:
            utc_h = int(t["time"][11:13])
            kst_h = (utc_h + 9) % 24
            bucket = (kst_h // 2) * 2  # 2시간 단위
            if bucket not in hour_groups:
                hour_groups[bucket] = []
            hour_groups[bucket].append(t)
        except Exception:
            pass
    for bucket in sorted(hour_groups.keys()):
        group = hour_groups[bucket]
        wins = sum(1 for t in group if (t.get("pnl") or 0) > 0)
        total_pnl = sum(t.get("pnl") or 0 for t in group)
        wr = wins / len(group) * 100
        bar = "+" * int(abs(total_pnl)) if total_pnl > 0 else "-" * int(abs(total_pnl))
        lines.append(f"  KST {bucket:02d}~{bucket+2:02d}시 | {len(group):3d}건 | 승률 {wr:5.1f}% | PnL ${total_pnl:+.2f} {bar}")
    bad_hours = [b for b, g in hour_groups.items()
                 if len(g) >= 2 and sum(t.get("pnl") or 0 for t in g) < -1]
    if bad_hours:
        lines.append(f"  >> 교훈: 손실 시간대 KST {', '.join(f'{h:02d}~{h+2:02d}시' for h in bad_hours)} — 진입 주의")
    lines.append("")

    # ── 7. 보유 시간 분석 ──
    lines.append("[ 7. 보유 시간 분석 ]")
    win_holds = []
    loss_holds = []
    for t in closed:
        h = _hold_minutes(t)
        if h is not None:
            if (t.get("pnl") or 0) > 0:
                win_holds.append(h)
            else:
                loss_holds.append(h)
    if win_holds:
        lines.append(f"  익절 평균 보유: {sum(win_holds)/len(win_holds):.0f}분 (최소 {min(win_holds):.0f}분, 최대 {max(win_holds):.0f}분)")
    if loss_holds:
        lines.append(f"  손절 평균 보유: {sum(loss_holds)/len(loss_holds):.0f}분 (최소 {min(loss_holds):.0f}분, 최대 {max(loss_holds):.0f}분)")
    if win_holds and loss_holds:
        avg_w = sum(win_holds) / len(win_holds)
        avg_l = sum(loss_holds) / len(loss_holds)
        if avg_l > avg_w * 1.5:
            lines.append(f"  >> 교훈: 손절이 익절보다 {avg_l/avg_w:.1f}배 오래 보유 — 빠른 손절 필요")
        elif avg_w > avg_l * 2:
            lines.append(f"  >> 교훈: 익절 포지션을 충분히 보유 중 — 양호")
    lines.append("")

    # ── 8. 연속 손실 패턴 ──
    lines.append("[ 8. 연속 손실 패턴 ]")
    max_streak, cur_streak = 0, 0
    streaks = []
    for t in reversed(closed):
        if (t.get("pnl") or 0) <= 0:
            cur_streak += 1
        else:
            if cur_streak >= 2:
                streaks.append(cur_streak)
            max_streak = max(max_streak, cur_streak)
            cur_streak = 0
    if cur_streak >= 2:
        streaks.append(cur_streak)
        max_streak = max(max_streak, cur_streak)
    lines.append(f"  최대 연속 손실: {max_streak}회")
    lines.append(f"  2회+ 연속 손실 발생: {len(streaks)}번")
    if max_streak >= 3:
        lines.append(f"  >> 교훈: 3회+ 연속 손실 발생 — 쿨다운 + 진입금 축소 작동 확인 필요")
    lines.append("")

    # ── 9. 일별 추세 ──
    daily = trade_db.get_daily_summary()
    if daily:
        lines.append("[ 9. 일별 추세 ]")
        for d in daily[-7:]:  # 최근 7일
            wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
            lines.append(f"  {d['date']} | {d['trades']:2d}건 | 승률 {wr:5.1f}% | PnL ${d['pnl']:+.2f} | 최고 ${d['best_trade']:+.2f} 최악 ${d['worst_trade']:+.2f}")
        # 추세 판단
        if len(daily) >= 3:
            recent_3 = daily[-3:]
            recent_pnl = [d["pnl"] for d in recent_3]
            if all(p < 0 for p in recent_pnl):
                lines.append(f"  >> 교훈: 3일 연속 손실 추세 — 전략 점검 필요")
            elif all(p > 0 for p in recent_pnl):
                lines.append(f"  >> 교훈: 3일 연속 수익 — 현재 전략 유효")
        lines.append("")

    # ── 10. 종합 교훈 ──
    lines.append("[ 10. 종합 교훈 (자동 생성) ]")
    lessons = []

    # 방향 교훈
    if longs and shorts:
        l_pnl = sum(t.get("pnl") or 0 for t in longs)
        s_pnl = sum(t.get("pnl") or 0 for t in shorts)
        if s_pnl < -3 and l_pnl > s_pnl:
            lessons.append(f"숏 누적 손실 ${s_pnl:.2f} — 롱 전용 운영 유지 권장")
        if l_pnl > 0:
            lessons.append(f"롱 누적 수익 ${l_pnl:+.2f} — 현재 롱 전략 유효")

    # 소스 교훈
    for src, s in src_perf.items():
        if s["trades"] >= 5 and s["win_rate"] < 20:
            lessons.append(f"소스 '{src}' 승률 {s['win_rate']:.0f}% — 비활성화 검토")

    # 시간대 교훈
    if bad_hours:
        lessons.append(f"KST {', '.join(f'{h:02d}~{h+2:02d}시' for h in bad_hours)} 손실 집중 — 해당 시간대 진입 제한 강화")

    # 신뢰도 교훈
    low_conf = [t for t in closed if (t.get("confidence") or 0) < 50]
    if low_conf and len(low_conf) >= 3:
        low_wr = sum(1 for t in low_conf if (t.get("pnl") or 0) > 0) / len(low_conf) * 100
        if low_wr < 30:
            lessons.append(f"신뢰도 50% 미만 승률 {low_wr:.0f}% — 최소 진입 임계값 상향 검토")

    # 보유 시간 교훈
    if win_holds and loss_holds:
        avg_w = sum(win_holds) / len(win_holds)
        avg_l = sum(loss_holds) / len(loss_holds)
        if avg_l > avg_w * 1.5:
            lessons.append(f"손절 보유 {avg_l:.0f}분 > 익절 {avg_w:.0f}분 — 빠른 손절 필요")

    if not lessons:
        lessons.append("데이터 부족 — 50건+ 축적 후 정밀 분석 가능")

    for i, lesson in enumerate(lessons, 1):
        lines.append(f"  {i}. {lesson}")
    lines.append("")

    # ── 잔고 현황 ──
    bal_hist = trade_db.get_balance_history(limit=1)
    if bal_hist:
        b = bal_hist[0]
        lines.append(f"[ 잔고 현황 ]")
        lines.append(f"  현재: ${b['close_bal']:.2f} | 시작($143.90) 대비: ${b['close_bal'] - 143.90:+.2f}")
        lines.append("")

    # ── 11. 자동 개선 제안 ──
    try:
        suggestions = generate_improvement_suggestions()
        lines.append(format_suggestions(suggestions))
    except Exception:
        pass
    lines.append("")

    lines.append("=" * 60)
    lines.append(f"  리포트 생성: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  다음 갱신: 매일 00:00 KST 자동 / 수동 실행 가능")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    # 파일 저장
    date_str = now.strftime("%Y-%m-%d")
    filepath = REPORT_DIR / f"trade_analysis_{date_str}.txt"
    filepath.write_text(report_text, encoding="utf-8")

    # 최신 리포트 심볼릭 링크 (에이전트 컨텍스트 주입용)
    latest = REPORT_DIR / "latest.txt"
    latest.write_text(report_text, encoding="utf-8")

    # 분석 차트 자동 생성
    try:
        _generate_chart(closed)
    except Exception:
        pass  # 차트 실패해도 리포트는 반환

    return report_text


def _generate_chart(closed):
    """거래 분석 6패널 차트 생성"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trades = []
    for t in closed:
        try:
            open_t = datetime.strptime(t["time"], "%Y-%m-%d %H:%M:%S")
            close_t = datetime.strptime(t["close_time"], "%Y-%m-%d %H:%M:%S")
            trades.append({
                "pnl": t.get("pnl", 0),
                "confidence": t.get("confidence", 0),
                "direction": "Long" if _is_long(t.get("side")) else "Short",
                "hour_kst": (open_t.hour + 9) % 24,
                "hold_min": (close_t - open_t).total_seconds() / 60,
                "symbol": t.get("symbol", "?"),
            })
        except Exception:
            continue

    if len(trades) < 3:
        return

    import pandas as pd
    df = pd.DataFrame(trades)
    df["cum_pnl"] = df["pnl"].cumsum()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Live Trading Analysis ({len(df)} trades)", fontsize=15, fontweight="bold")

    # 1. 누적 PnL
    ax = axes[0, 0]
    x = range(1, len(df) + 1)
    ax.plot(x, df["cum_pnl"].values, "o-", color="#1565C0", lw=2, ms=5)
    ax.axhline(0, color="red", lw=0.8, ls=":")
    ax.set_title("Cumulative PnL"); ax.set_xlabel("Trade #"); ax.set_ylabel("PnL ($)"); ax.grid(alpha=0.3)

    # 2. 방향별 PnL
    ax = axes[0, 1]
    for i, (d, c) in enumerate([("Long", "#4CAF50"), ("Short", "#F44336")]):
        data = df[df["direction"] == d]["pnl"].values
        if len(data) > 0:
            ax.scatter([i] * len(data), data, c=c, s=60, alpha=0.7)
            ax.text(i, max(data.max(), 0) + 0.3, f"n={len(data)}\n${data.sum():.1f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Long", "Short"])
    ax.axhline(0, color="gray", lw=0.8, ls=":"); ax.set_title("PnL by Direction"); ax.grid(alpha=0.3)

    # 3. 시간대별 (KST)
    ax = axes[0, 2]
    hg = df.groupby("hour_kst")["pnl"].agg(["sum", "count"]).reindex(range(24), fill_value=0)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in hg["sum"]]
    ax.bar(hg.index, hg["sum"], color=colors, alpha=0.8)
    ax.set_title("PnL by Hour (KST)"); ax.set_xlabel("Hour"); ax.set_xticks(range(0, 24, 2)); ax.grid(axis="y", alpha=0.3)

    # 4. 심볼별 PnL
    ax = axes[1, 0]
    sp = df.groupby("symbol")["pnl"].sum().sort_values()
    colors_s = ["#4CAF50" if v >= 0 else "#F44336" for v in sp]
    sp.plot.barh(ax=ax, color=colors_s, alpha=0.8)
    ax.set_title("PnL by Symbol"); ax.axvline(0, color="gray", lw=0.8, ls=":"); ax.grid(axis="x", alpha=0.3)

    # 5. 보유시간 vs PnL
    ax = axes[1, 1]
    cs = ["green" if p > 0 else "red" for p in df["pnl"]]
    ax.scatter(df["hold_min"], df["pnl"], c=cs, s=60, alpha=0.7)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_title("Hold Time vs PnL"); ax.set_xlabel("Minutes"); ax.set_ylabel("PnL ($)"); ax.grid(alpha=0.3)

    # 6. 신뢰도 vs PnL
    ax = axes[1, 2]
    ax.scatter(df["confidence"], df["pnl"], c=cs, s=60, alpha=0.7)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_title("Confidence vs PnL"); ax.set_xlabel("Confidence"); ax.set_ylabel("PnL ($)"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "trade_analysis_chart.png", dpi=150, bbox_inches="tight")
    plt.close()


def get_latest_report() -> str:
    """최신 리포트 텍스트 반환 (에이전트 프롬프트 주입용)"""
    latest = REPORT_DIR / "latest.txt"
    if latest.exists():
        text = latest.read_text(encoding="utf-8")
        # 에이전트 컨텍스트 길이 제한 (최대 2000자)
        if len(text) > 2000:
            # 종합 교훈 섹션만 추출
            idx = text.find("[ 10. 종합 교훈")
            if idx >= 0:
                return text[idx:]
        return text
    return ""


def get_report_summary() -> str:
    """에이전트 프롬프트용 축약 요약 (500자 이내)"""
    closed = trade_db.get_closed_trades(limit=9999)
    closed = _ghost_filter(closed)
    if not closed:
        return "거래 데이터 없음"

    stats = trade_db.get_trade_stats()
    if not stats:
        return "통계 계산 불가"

    longs = [t for t in closed if _is_long(t.get("side"))]
    shorts = [t for t in closed if _is_short(t.get("side"))]
    l_pnl = sum(t.get("pnl") or 0 for t in longs)
    s_pnl = sum(t.get("pnl") or 0 for t in shorts)

    lines = [
        f"[거래 성과 요약] {stats['total']}건 승률{stats['win_rate']:.0f}% PnL${stats['total_pnl']:+.2f} PF:{stats['profit_factor']:.1f}",
        f"롱{len(longs)}건(${l_pnl:+.2f}) 숏{len(shorts)}건(${s_pnl:+.2f})",
    ]

    # 소스별 한줄 요약
    src = trade_db.get_source_performance(recent_n=9999)
    src_parts = []
    for name, s in sorted(src.items(), key=lambda x: x[1]["trades"], reverse=True)[:3]:
        src_parts.append(f"{name}:{s['trades']}건/{s['win_rate']:.0f}%")
    if src_parts:
        lines.append(f"소스: {' | '.join(src_parts)}")

    return " / ".join(lines)


def generate_improvement_suggestions() -> list:
    """거래 데이터 분석 → 파라미터 조정 제안 목록 반환 (사람 승인 후 적용)"""
    closed = trade_db.get_closed_trades(limit=9999)
    closed = _ghost_filter(closed)

    if len(closed) < 10:
        return [{"type": "info", "msg": f"데이터 {len(closed)}건 — 10건+ 축적 후 제안 가능"}]

    suggestions = []
    stats = trade_db.get_trade_stats()

    # 1. 승률 기반 신뢰도 임계값 조정 제안
    conf_bins = [(50, 59), (60, 69), (70, 79), (80, 100)]
    for lo, hi in conf_bins:
        group = [t for t in closed if lo <= (t.get("confidence") or 0) <= hi]
        if len(group) >= 5:
            wr = sum(1 for t in group if (t.get("pnl") or 0) > 0) / len(group) * 100
            pnl = sum(t.get("pnl") or 0 for t in group)
            if wr < 25 and pnl < -2:
                suggestions.append({
                    "type": "confidence",
                    "severity": "high",
                    "msg": f"신뢰도 {lo}~{hi}% 구간: 승률 {wr:.0f}%, PnL ${pnl:+.2f} — 최소 임계값 {hi+1}%로 상향 검토",
                    "param": "MIN_CONFIDENCE",
                    "current": lo,
                    "suggested": hi + 1,
                })

    # 2. 시간대별 손실 패턴 → 차단 시간 조정 제안
    hour_pnl = {}
    for t in closed:
        try:
            utc_h = int(t["time"][11:13])
            kst_h = (utc_h + 9) % 24
            bucket = (kst_h // 2) * 2
            if bucket not in hour_pnl:
                hour_pnl[bucket] = {"count": 0, "pnl": 0, "wins": 0}
            hour_pnl[bucket]["count"] += 1
            hour_pnl[bucket]["pnl"] += (t.get("pnl") or 0)
            if (t.get("pnl") or 0) > 0:
                hour_pnl[bucket]["wins"] += 1
        except Exception:
            pass

    for bucket, s in hour_pnl.items():
        if s["count"] >= 5 and s["pnl"] < -3:
            wr = s["wins"] / s["count"] * 100
            suggestions.append({
                "type": "time_block",
                "severity": "medium",
                "msg": f"KST {bucket:02d}~{bucket+2:02d}시: {s['count']}건 승률 {wr:.0f}% PnL ${s['pnl']:+.2f} — 해당 시간대 차단/축소 검토",
                "param": "BLOCK_HOURS",
                "current": f"KST {bucket:02d}-{bucket+2:02d}",
                "suggested": "차단 추가",
            })

    # 3. 소스별 성과 → 저성과 소스 비활성화 제안
    src_perf = trade_db.get_source_performance(recent_n=9999)
    for src, s in src_perf.items():
        if s["trades"] >= 5 and s["win_rate"] < 20:
            suggestions.append({
                "type": "source",
                "severity": "high",
                "msg": f"소스 '{src}': {s['trades']}건 승률 {s['win_rate']:.0f}% — 비활성화 검토",
                "param": f"SOURCE_{src.upper()}",
                "current": "활성",
                "suggested": "비활성화",
            })
        elif s["trades"] >= 10 and s["win_rate"] >= 50 and s["total_pnl"] > 0:
            suggestions.append({
                "type": "source",
                "severity": "positive",
                "msg": f"소스 '{src}': {s['trades']}건 승률 {s['win_rate']:.0f}% PnL ${s['total_pnl']:+.2f} — 우수 소스, 가중치 강화 검토",
            })

    # 4. 보유 시간 분석 → 조기 청산 파라미터 제안
    win_holds, loss_holds = [], []
    for t in closed:
        h = _hold_minutes(t)
        if h is not None:
            if (t.get("pnl") or 0) > 0:
                win_holds.append(h)
            else:
                loss_holds.append(h)

    if loss_holds and len(loss_holds) >= 5:
        avg_loss_hold = sum(loss_holds) / len(loss_holds)
        median_loss_hold = sorted(loss_holds)[len(loss_holds) // 2]
        if avg_loss_hold > 90:  # 손절 평균 90분 이상
            suggestions.append({
                "type": "early_exit",
                "severity": "medium",
                "msg": f"손절 평균 보유 {avg_loss_hold:.0f}분 (중앙값 {median_loss_hold:.0f}분) — 조기 청산 기준 강화 검토",
                "param": "EARLY_EXIT",
                "current": "30m/-1.0% or 45m/-0.5%",
                "suggested": f"중앙값({median_loss_hold:.0f}분)의 80%에서 청산",
            })

    # 5. 연속 손실 패턴 → 쿨다운 조정 제안
    max_streak, cur_streak = 0, 0
    for t in reversed(closed):
        if (t.get("pnl") or 0) <= 0:
            cur_streak += 1
        else:
            max_streak = max(max_streak, cur_streak)
            cur_streak = 0
    max_streak = max(max_streak, cur_streak)

    if max_streak >= 5:
        suggestions.append({
            "type": "cooldown",
            "severity": "high",
            "msg": f"최대 연속 손실 {max_streak}회 — 쿨다운 강화 필요 (현재 3연패→2시간)",
            "param": "CONSECUTIVE_LOSS_COOLDOWN",
            "current": "3연패→2시간",
            "suggested": "2연패→2시간 또는 3연패→4시간",
        })

    # 6. 전체 승률 기반 포지션 크기 제안 (50건+ 시만)
    if stats and stats["total"] >= 50:
        wr = stats["win_rate"]
        if wr >= 45 and stats["profit_factor"] >= 1.5:
            suggestions.append({
                "type": "position_size",
                "severity": "positive",
                "msg": f"승률 {wr:.0f}% PF {stats['profit_factor']:.1f} — POSITION_PCT 15%로 증액 검토 가능",
                "param": "POSITION_PCT",
                "current": 12,
                "suggested": 15,
            })
        elif wr < 30:
            suggestions.append({
                "type": "position_size",
                "severity": "high",
                "msg": f"승률 {wr:.0f}% — POSITION_PCT 8%로 축소 검토",
                "param": "POSITION_PCT",
                "current": 12,
                "suggested": 8,
            })

    if not suggestions:
        suggestions.append({"type": "info", "msg": "현재 파라미터 적정 — 변경 불필요"})

    return suggestions


def format_suggestions(suggestions: list) -> str:
    """제안 목록을 텍스트로 포맷"""
    severity_icon = {"high": "🔴", "medium": "🟡", "positive": "🟢", "info": "ℹ️"}
    lines = ["", "[ 자동 개선 제안 ]"]
    for i, s in enumerate(suggestions, 1):
        icon = severity_icon.get(s.get("severity", "info"), "ℹ️")
        lines.append(f"  {icon} {i}. {s['msg']}")
        if "current" in s and "suggested" in s:
            lines.append(f"     현재: {s['current']} → 제안: {s['suggested']}")
    lines.append("  ※ 자동 적용 아님 — 사람 확인 후 수동 반영")
    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report()
    print(report)

    # 개선 제안도 함께 출력
    suggestions = generate_improvement_suggestions()
    print(format_suggestions(suggestions))

    print(f"\n저장됨: {REPORT_DIR / 'latest.txt'}")

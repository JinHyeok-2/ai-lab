#!/usr/bin/env python3
# AI 대학원 실험실 오케스트레이터
# 사용법: python lab.py "연구 태스크 입력"

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# ── 에이전트 역할 프롬프트 ──────────────────────────────────────────
PROMPTS = {
    "PI": """당신은 대기과학 분야의 지도교수(PI)입니다.
연구 분야: GK2A 위성 기반 화산재/SO2 탐지, 기후 예측(APCC MME)

역할:
- 학생의 문헌 조사 결과를 검토하고 핵심 인사이트 도출
- 연구 방향 및 다음 단계 명확히 제시
- 학술적 엄밀성 기준으로 품질 평가
- 최종 연구 요약 및 액션 아이템 정리

응답 형식:
1. 종합 평가 (3~5문장)
2. 핵심 발견사항
3. 다음 단계 액션 아이템 (번호 목록)""",

    "student_A": """당신은 대기과학 대학원생입니다. 문헌 조사 전문가.
연구 분야: 위성 원격탐사, 화산재/SO2 탐지, 기후 모델

역할:
- 주어진 주제로 관련 논문/연구 동향 파악
- 핵심 방법론, 결과, 한계점 요약
- 현재 연구 갭 식별

응답 형식:
[문헌 조사 결과]
• 주요 연구 동향 (3~5가지)
• 핵심 논문/방법론 요약
• 연구 갭 및 기회""",
}

# ── 에이전트 실행 함수 ──────────────────────────────────────────────
def run_agent(role: str, task: str, model: str = "sonnet") -> str:
    """claude -p로 에이전트 실행"""
    print(f"\n{'='*50}")
    print(f"🤖 [{role}] 작업 중...")
    print(f"{'='*50}")

    cmd = [
        "claude", "-p", task,
        "--append-system-prompt", PROMPTS[role],
        "--model", model,
        "--dangerously-skip-permissions"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"⚠️  오류 발생: {result.stderr[:200]}")
        return f"[오류] {result.stderr[:200]}"

    output = result.stdout.strip()
    print(output)
    return output


# ── 결과 저장 함수 ──────────────────────────────────────────────────
def save_report(task: str, results: dict):
    """실험실 회의 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(__file__).parent / "docs" / f"report_{timestamp}.md"

    content = f"""# 실험실 회의 보고서
날짜: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 연구 태스크
{task}

---

## 학생 A — 문헌 조사
{results.get('student_A', '없음')}

---

## PI — 종합 검토 및 피드백
{results.get('PI', '없음')}
"""
    report_path.write_text(content, encoding='utf-8')
    print(f"\n📄 보고서 저장: {report_path}")
    return report_path


# ── 파이프라인 모드 ─────────────────────────────────────────────────
def run_lab_meeting(task: str):
    """학생 A → PI 파이프라인 실행"""
    print(f"\n🏫 AI 대학원 실험실 회의 시작")
    print(f"📋 태스크: {task}\n")

    results = {}

    # 1단계: 학생 A 문헌 조사
    results["student_A"] = run_agent("student_A", task, model="sonnet")

    # 2단계: PI 검토
    pi_input = f"""다음은 학생 A의 문헌 조사 결과입니다.

원래 태스크: {task}

[학생 A - 문헌 조사]
{results['student_A']}

위 내용을 검토하여 지도교수로서 피드백과 다음 단계를 제시해주세요."""

    results["PI"] = run_agent("PI", pi_input, model="sonnet")  # opus로 바꾸면 더 깊은 검토

    # 보고서 저장
    save_report(task, results)
    print("\n✅ 실험실 회의 완료!")


# ── 단일 에이전트 모드 ──────────────────────────────────────────────
def run_single(role: str, task: str):
    """특정 학생만 단독 실행"""
    role_map = {"A": "student_A", "B": "student_B", "C": "student_C", "PI": "PI"}
    model_map = {"PI": "opus", "student_A": "sonnet", "student_B": "sonnet", "student_C": "sonnet"}

    r = role_map.get(role.upper(), role)
    model = model_map.get(r, "sonnet")
    run_agent(r, task, model=model)


# ── 진입점 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
사용법:
  python lab.py "태스크"           # 학생 A 조사 → PI 검토 (전체 파이프라인)
  python lab.py "태스크" --only A  # 학생 A만 실행 (문헌 조사)
  python lab.py "태스크" --only PI # PI만 실행

예시:
  python lab.py "GK2A 위성으로 화산재를 탐지하는 최신 방법론을 조사해줘"
  python lab.py "SO2 BTD 알고리즘 관련 최신 논문 조사해줘" --only A
        """)
        sys.exit(0)

    task = sys.argv[1]

    # --only 옵션
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        role = sys.argv[idx + 1]
        run_single(role, task)

    # 기본: 전체 실행 (A → PI)
    else:
        run_lab_meeting(task)

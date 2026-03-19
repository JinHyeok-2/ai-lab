#!/usr/bin/env python3
# 최교수의 AI 랩 — 슬랙 스타일 GUI
# 실행: streamlit run app.py

import streamlit as st
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="최교수의 AI 랩",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 슬랙 스타일 CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
.stApp { background-color: #f8f8f8; }

/* 사이드바 — 슬랙 다크 퍼플 */
[data-testid="stSidebar"] {
    background-color: #1a1d2e !important;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #3a3d4e !important;
}

/* 워크스페이스 헤더 */
.ws-header {
    padding: 14px 16px 10px;
    border-bottom: 1px solid #3a3d4e;
    margin-bottom: 8px;
}
.ws-name {
    font-size: 17px;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.3px;
}
.ws-status {
    font-size: 11px;
    color: #7a8599 !important;
    margin-top: 2px;
}

/* 채널 섹션 헤더 */
.section-label {
    font-size: 12px;
    font-weight: 600;
    color: #7a8599 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 12px 16px 4px;
}

/* 채널/멤버 아이템 */
.channel-item {
    padding: 5px 16px;
    border-radius: 4px;
    font-size: 14px;
    color: #c9d1d9 !important;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Git 상태 박스 */
.git-status-box {
    background: #2a2d3e;
    border: 1px solid #3a3d4e;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 12px;
    font-family: monospace;
}
.git-status-clean {
    color: #2bac76 !important;
}
.git-status-dirty {
    color: #e8a838 !important;
}
.git-changed-file {
    color: #e8a838 !important;
    font-size: 11px;
    padding: 1px 0;
}
.git-badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
}
.git-badge-push { background: #e01e5a; color: #fff !important; }
.git-badge-pull { background: #1264a3; color: #fff !important; }
.git-badge-ok   { background: #2bac76; color: #fff !important; }

/* 멤버 온라인 상태 */
.status-online  { color: #2bac76 !important; }
.status-busy    { color: #e8a838 !important; }

/* 메인 영역 */
.channel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 20px 12px;
    border-bottom: 1px solid #e8e8e8;
    background: #ffffff;
}
.channel-header-name { font-size: 17px; font-weight: 700; color: #1d1c1d; }
.channel-header-desc { font-size: 13px; color: #616061; margin-left: 8px; }

/* 슬랙 메시지 */
.slack-msg {
    display: flex;
    gap: 10px;
    padding: 6px 20px;
    transition: background 0.1s;
}
.slack-msg:hover { background: #f8f8f8; }
.slack-msg.first-in-group { padding-top: 14px; }

.msg-avatar {
    width: 36px; height: 36px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar-user   { background: #4a154b; }
.avatar-a      { background: #1264a3; }
.avatar-b      { background: #7c3aed; }
.avatar-pi     { background: #007a5a; }
.avatar-git    { background: #24292e; }
.avatar-system { background: #e8e8e8; }

.msg-body { flex: 1; min-width: 0; }
.msg-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 2px;
}
.msg-name { font-size: 14px; font-weight: 700; color: #1d1c1d; }
.msg-name.name-user   { color: #4a154b; }
.msg-name.name-a      { color: #1264a3; }
.msg-name.name-b      { color: #7c3aed; }
.msg-name.name-pi     { color: #007a5a; }
.msg-name.name-git    { color: #24292e; }
.msg-name.name-system { color: #616061; }
.msg-time { font-size: 11px; color: #aaaaaa; }
.msg-text {
    font-size: 14px;
    color: #1d1c1d;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}
.msg-text.system-text { font-size: 13px; color: #888; font-style: italic; }
.msg-text.git-text    { font-family: monospace; font-size: 13px; background: #f6f8fa; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #24292e; }

/* 날짜 구분선 */
.date-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px 8px;
    color: #616061;
    font-size: 12px;
    font-weight: 600;
}
.date-divider::before, .date-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #e8e8e8;
}

/* 입력창 */
[data-testid="stChatInput"] {
    border: 1.5px solid #e0e0e0 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    margin: 0 16px 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #1d1c1d !important;
    font-size: 14px !important;
}

/* 버튼 */
.stButton > button {
    background: transparent;
    color: #c9d1d9 !important;
    border: 1px solid #3a3d4e;
    border-radius: 6px;
    font-size: 13px;
    width: 100%;
    text-align: left;
    padding: 5px 12px;
}
.stButton > button:hover {
    background: #2a2d3e !important;
    color: #fff !important;
    border-color: #4a4d5e;
}

/* selectbox 다크 */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
    background: #2a2d3e !important;
    border: 1px solid #3a3d4e !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] * {
    color: #c9d1d9 !important;
}

/* expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #2a2d3e;
    border: 1px solid #3a3d4e;
    border-radius: 4px;
}

/* 상태 박스 */
[data-testid="stStatus"] {
    background: #f8f8f8;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    margin: 4px 20px;
}

/* 메인 패딩 제거 */
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Git 설정 ───────────────────────────────────────────────────────
VOLCANO_PATH = Path("/Users/sunny/Desktop/volcano-paper")

def git_run(*args):
    """git 명령어 실행 → (성공여부, 출력)"""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=VOLCANO_PATH,
        capture_output=True, text=True
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output

def git_status_porcelain():
    """변경된 파일 목록 반환 (없으면 빈 문자열)"""
    _, out = git_run("status", "--porcelain")
    return out.strip()

def git_pull():
    ok, out = git_run("pull")
    return ok, out

def git_push(commit_msg: str = None):
    changed = git_status_porcelain()
    if not changed:
        return True, "변경사항 없음 — 이미 최신 상태입니다."
    git_run("add", ".")
    msg = commit_msg or f"[자동] {datetime.now().strftime('%Y-%m-%d %H:%M')} 변경사항 저장"
    git_run("commit", "-m", msg)
    ok, out = git_run("push")
    return ok, f"커밋: {msg}\n{out}"

def git_log_short():
    _, out = git_run("log", "--oneline", "-5")
    return out

# ── 자동 Push 백그라운드 스레드 ────────────────────────────────────
_auto_push_lock = threading.Lock()

def _auto_push_worker(interval: int, log_queue: list):
    """interval초마다 변경사항 체크 → 자동 push"""
    while True:
        time.sleep(interval)
        if not st.session_state.get("auto_push_enabled", False):
            break
        changed = git_status_porcelain()
        if changed:
            ok, out = git_push()
            ts = datetime.now().strftime("%H:%M")
            status = "✅ 자동 Push 완료" if ok else "⚠️ Push 실패"
            with _auto_push_lock:
                log_queue.append({"time": ts, "msg": f"{status}\n{out}"})

def start_auto_push(interval: int = 60):
    if "auto_push_log" not in st.session_state:
        st.session_state.auto_push_log = []
    t = threading.Thread(
        target=_auto_push_worker,
        args=(interval, st.session_state.auto_push_log),
        daemon=True
    )
    t.start()
    st.session_state.auto_push_thread = t

# ── 채널 & 멤버 정의 ──────────────────────────────────────────────────
CHANNELS = {
    "일반": {
        "icon": "#",
        "desc": "전체 회의 — 학생1 조사 → 학생2 집필 → PI 검토",
        "target": "전체 회의"
    },
    "문헌조사": {
        "icon": "#",
        "desc": "학생1 전담 — 논문 및 연구 동향 조사",
        "target": "학생1만"
    },
    "논문작성": {
        "icon": "#",
        "desc": "학생2 전담 — 논문 초안 작성 및 문장 다듬기",
        "target": "학생2만"
    },
    "pi-검토": {
        "icon": "#",
        "desc": "PI 전담 — 방향 설정 및 피드백",
        "target": "PI만"
    },
    "git-현황": {
        "icon": "#",
        "desc": "volcano-paper 저장소 — Push / Pull / 자동 동기화",
        "target": None
    },
}

MEMBERS = {
    "학생1": {"emoji": "🔬", "name": "김연구", "role": "학생 1 (문헌조사)", "model": "sonnet", "color": "#1264a3"},
    "학생2": {"emoji": "✍️", "name": "이집필", "role": "학생 2 (논문작성)", "model": "sonnet", "color": "#7c3aed"},
    "PI":    {"emoji": "👨‍🏫", "name": "최교수", "role": "지도교수 (PI)",    "model": "opus",   "color": "#007a5a"},
}

PROMPTS = {
    "학생1": """당신은 대기과학 대학원생 '김연구'입니다. 문헌 조사 전문가.
연구 분야: 위성 원격탐사, 화산재/SO2 탐지, 기후 모델

역할:
- 주어진 주제로 관련 논문/연구 동향 파악
- 핵심 방법론, 결과, 한계점 요약
- 현재 연구 갭 식별

말투: 공손하고 성실한 대학원생 말투 ("네, 교수님" "조사해보겠습니다" 등)

응답 형식:
[문헌 조사 결과]
• 주요 연구 동향
• 핵심 논문 요약
• 연구 갭 및 기회""",

    "학생2": """당신은 대기과학 대학원생 '이집필'입니다. 논문 작성 전문가.
연구 분야: GK-2A/AMI SO₂ RGB 화산 플룸 탐지, 다중위성 비교 (TROPOMI, IASI)
현재 작업 논문: Remote Sensing of Environment (RSE) 투고 예정 — v7 원고

역할:
- 논문 섹션(Introduction, Methods, Results, Discussion, Conclusion) 초안 작성 및 개선
- RSE 스타일에 맞는 영어 학술 문장 작성
- 문장 흐름, 논리 구조, 학술적 표현 다듬기
- 기존 내용 기반 보완 및 확장 제안

핵심 수치 (반드시 정확하게 사용):
- R² = 0.76 [95% CI: 0.72–0.80], n = 1,247, RMSE = 35 DU
- 회귀식: RGB = 255 − 0.95 × DU
- Ruang 고도: 19–21 km (성층권), TROPOMI 편향: ±20–30 DU
- VEI 범위: 2–3

말투: 꼼꼼하고 학구적인 대학원생 ("이 문장은 이렇게 고치면 어떨까요?" "RSE 심사기준에 맞게 수정했습니다" 등)

응답 형식:
[논문 작성 결과]
• 작성/수정된 텍스트 (영문 우선, 필요시 한국어 설명 추가)
• 수정 근거 및 RSE 스타일 설명""",

    "PI": """당신은 대기과학 지도교수 '최교수'입니다.
연구 분야: GK2A 위성 기반 화산재/SO2 탐지, 기후 예측

역할:
- 학생 결과물 검토 및 피드백
- 연구 방향 제시
- 다음 단계 액션 아이템 지시

말투: 권위 있지만 친근한 교수 말투 ("잘 했어", "이 부분은 더 보완해야지" 등)
응답은 간결하고 핵심만. 3~5문장 + 액션 아이템."""
}

# ── 세션 상태 초기화 ───────────────────────────────────────────────
if "channel_messages" not in st.session_state:
    st.session_state.channel_messages = {ch: [] for ch in CHANNELS}

if "active_channel" not in st.session_state:
    st.session_state.active_channel = "일반"

if "member_status" not in st.session_state:
    st.session_state.member_status = {k: "online" for k in MEMBERS}

if "model_overrides" not in st.session_state:
    st.session_state.model_overrides = {k: v["model"] for k, v in MEMBERS.items()}

if "auto_push_enabled" not in st.session_state:
    st.session_state.auto_push_enabled = False

if "auto_push_log" not in st.session_state:
    st.session_state.auto_push_log = []

# ── 에이전트 실행 ──────────────────────────────────────────────────
def run_agent(member_key: str, task: str) -> str:
    model = st.session_state.model_overrides[member_key]
    cmd = [
        "claude", "-p", task,
        "--append-system-prompt", PROMPTS[member_key],
        "--model", model,
        "--dangerously-skip-permissions"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"⚠️ 오류: {result.stderr[:200]}"
    return result.stdout.strip()

def add_message(channel: str, speaker: str, content: str):
    st.session_state.channel_messages[channel].append({
        "speaker": speaker,
        "content": content,
        "time": datetime.now().strftime("%H:%M"),
        "date": datetime.now().strftime("%Y년 %m월 %d일"),
    })

def save_report(channel: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(__file__).parent / "docs" / f"report_{channel}_{timestamp}.md"
    lines = [f"# 최교수의 AI 랩 — #{channel} 회의록\n날짜: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"]
    for msg in st.session_state.channel_messages[channel]:
        lines.append(f"**[{msg['time']}] {msg['speaker']}**\n{msg['content']}\n\n---\n\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path

# ── 메시지 렌더링 ──────────────────────────────────────────────────
def render_messages(channel: str):
    msgs = st.session_state.channel_messages[channel]
    if not msgs:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:80px 0;color:#aaa;">
            <div style="font-size:48px;margin-bottom:12px;">#</div>
            <div style="font-size:18px;font-weight:700;color:#1d1c1d;">{channel}</div>
            <div style="font-size:14px;color:#888;margin-top:6px;">
                이 채널의 첫 번째 메시지를 보내보세요
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    prev_date = None
    prev_speaker = None

    for msg in msgs:
        speaker = msg["speaker"]
        content = msg["content"]
        time_str = msg["time"]
        date     = msg["date"]

        if date != prev_date:
            st.markdown(f'<div class="date-divider">{date}</div>', unsafe_allow_html=True)
            prev_date = date
            prev_speaker = None

        if speaker == "나":
            avatar_class, name_class, avatar_icon, display_name = "avatar-user", "name-user", "🧑‍💻", "나"
        elif speaker == "학생1":
            avatar_class, name_class, avatar_icon, display_name = "avatar-a", "name-a", "🔬", "김연구 (학생1)"
        elif speaker == "학생2":
            avatar_class, name_class, avatar_icon, display_name = "avatar-b", "name-b", "✍️", "이집필 (학생2)"
        elif speaker == "PI":
            avatar_class, name_class, avatar_icon, display_name = "avatar-pi", "name-pi", "👨‍🏫", "최교수 (PI)"
        elif speaker == "Git":
            avatar_class, name_class, avatar_icon, display_name = "avatar-git", "name-git", "⚙️", "Git Bot"
        else:
            avatar_class, name_class, avatar_icon, display_name = "avatar-system", "name-system", "🏫", "시스템"

        text_class = "git-text" if speaker == "Git" else ("system-text" if speaker == "system" else "")
        is_continuation = (speaker == prev_speaker)

        if is_continuation:
            st.markdown(f"""
            <div class="slack-msg" style="padding-left:66px;">
                <div class="msg-body">
                    <div class="msg-text {text_class}">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="slack-msg first-in-group">
                <div class="msg-avatar {avatar_class}">{avatar_icon}</div>
                <div class="msg-body">
                    <div class="msg-header">
                        <span class="msg-name {name_class}">{display_name}</span>
                        <span class="msg-time">{time_str}</span>
                    </div>
                    <div class="msg-text {text_class}">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        prev_speaker = speaker

# ── Git 현황 채널 렌더링 ────────────────────────────────────────────
def render_git_channel():
    changed = git_status_porcelain()

    # 현재 상태 카드
    if changed:
        files = changed.split("\n")
        files_html = "".join([f'<div class="git-changed-file">  {f}</div>' for f in files])
        st.markdown(f"""
        <div class="git-status-box">
            <span class="git-badge git-badge-push">● {len(files)}개 변경</span>
            <div style="margin-top:6px; color:#e8a838 !important;">{files_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="git-status-box">
            <span class="git-badge git-badge-ok">✓ 최신 상태</span>
            <span style="color:#7a8599; font-size:12px; margin-left:8px;">변경사항 없음</span>
        </div>
        """, unsafe_allow_html=True)

    # 버튼 행
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Pull", use_container_width=True, key="btn_pull"):
            with st.spinner("Pull 중..."):
                ok, out = git_pull()
            icon = "✅" if ok else "⚠️"
            add_message("git-현황", "Git", f"{icon} git pull\n{out}")
            st.rerun()
    with col2:
        if st.button("📤 Push", use_container_width=True, key="btn_push"):
            with st.spinner("Push 중..."):
                ok, out = git_push()
            icon = "✅" if ok else "⚠️"
            add_message("git-현황", "Git", f"{icon} git push\n{out}")
            st.rerun()
    with col3:
        if st.button("🔄 새로고침", use_container_width=True, key="btn_refresh"):
            st.rerun()

    # 커밋 메시지 직접 입력 Push
    with st.expander("✏️ 커밋 메시지 직접 입력", expanded=False):
        custom_msg = st.text_input("커밋 메시지", placeholder="[원고] v7 수정 내용", key="custom_commit_msg", label_visibility="collapsed")
        if st.button("📤 커스텀 Push", use_container_width=True, key="btn_custom_push"):
            if custom_msg:
                with st.spinner("Push 중..."):
                    ok, out = git_push(custom_msg)
                icon = "✅" if ok else "⚠️"
                add_message("git-현황", "Git", f"{icon} git push\n커밋: {custom_msg}\n{out}")
                st.rerun()

    # 자동 Push 토글
    st.divider()
    auto_col1, auto_col2 = st.columns([2, 1])
    with auto_col1:
        st.markdown('<span style="color:#c9d1d9;font-size:13px;font-weight:600;">⚡ 자동 Push</span>', unsafe_allow_html=True)
        st.markdown('<span style="color:#7a8599;font-size:11px;">변경 감지 시 60초마다 자동 커밋+푸시</span>', unsafe_allow_html=True)
    with auto_col2:
        prev_auto = st.session_state.auto_push_enabled
        new_auto = st.toggle("", value=prev_auto, key="auto_push_toggle")
        if new_auto != prev_auto:
            st.session_state.auto_push_enabled = new_auto
            if new_auto:
                start_auto_push(interval=60)
                add_message("git-현황", "Git", "⚡ 자동 Push 활성화 — 60초마다 변경사항 감지")
            else:
                add_message("git-현황", "Git", "⏹ 자동 Push 비활성화")
            st.rerun()

    if st.session_state.auto_push_enabled:
        st.markdown('<span class="git-badge git-badge-ok" style="font-size:11px;">● 자동 Push 실행 중</span>', unsafe_allow_html=True)

    # 자동 Push 로그 처리
    with _auto_push_lock:
        if st.session_state.auto_push_log:
            for entry in st.session_state.auto_push_log:
                add_message("git-현황", "Git", entry["msg"])
            st.session_state.auto_push_log.clear()

    st.divider()

    # 최근 커밋 로그
    st.markdown('<div style="color:#7a8599;font-size:12px;font-weight:600;padding:4px 0;">최근 커밋</div>', unsafe_allow_html=True)
    log = git_log_short()
    if log:
        st.markdown(f'<div class="git-status-box" style="color:#c9d1d9 !important;">{log}</div>', unsafe_allow_html=True)

    st.divider()

    # 메시지 피드
    msgs = st.session_state.channel_messages["git-현황"]
    if msgs:
        st.markdown('<div style="color:#7a8599;font-size:12px;font-weight:600;padding:4px 0 8px;">활동 로그</div>', unsafe_allow_html=True)
        for msg in reversed(msgs[-10:]):
            st.markdown(f"""
            <div class="slack-msg first-in-group">
                <div class="msg-avatar avatar-git">⚙️</div>
                <div class="msg-body">
                    <div class="msg-header">
                        <span class="msg-name name-git">Git Bot</span>
                        <span class="msg-time">{msg['time']}</span>
                    </div>
                    <div class="msg-text git-text">{msg['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── 사이드바 ───────────────────────────────────────────────────────
with st.sidebar:
    # 워크스페이스 헤더
    st.markdown("""
    <div class="ws-header">
        <div class="ws-name">🏫 최교수의 AI 랩</div>
        <div class="ws-status">🟢 실험실 운영 중</div>
    </div>
    """, unsafe_allow_html=True)

    # 채널 목록
    st.markdown('<div class="section-label">채널</div>', unsafe_allow_html=True)
    for ch_name, ch_info in CHANNELS.items():
        msg_count = len(st.session_state.channel_messages[ch_name])
        # git 채널은 변경사항 있으면 뱃지 표시
        if ch_name == "git-현황":
            changed_count = len(git_status_porcelain().split("\n")) if git_status_porcelain() else 0
            suffix = f"  ⬆{changed_count}" if changed_count > 0 else ""
        else:
            suffix = f"  ·{msg_count}" if msg_count > 0 else ""
        btn_label = f"# {ch_name}{suffix}"
        if st.button(btn_label, key=f"ch_{ch_name}", use_container_width=True):
            st.session_state.active_channel = ch_name
            st.rerun()

    st.divider()

    # 멤버 목록
    st.markdown('<div class="section-label">멤버</div>', unsafe_allow_html=True)
    for key, info in MEMBERS.items():
        status = st.session_state.member_status[key]
        dot = "🟢" if status == "online" else "🟠"
        st.markdown(f"""
        <div class="channel-item" style="padding: 4px 8px;">
            {dot} {info['emoji']} {info['name']}
            <span style="color:#7a8599;font-size:12px;margin-left:4px;">{info['role']}</span>
        </div>
        """, unsafe_allow_html=True)
        new_model = st.selectbox(
            "모델",
            ["sonnet", "opus", "haiku"],
            index=["sonnet", "opus", "haiku"].index(st.session_state.model_overrides[key]),
            key=f"model_{key}",
            label_visibility="collapsed"
        )
        st.session_state.model_overrides[key] = new_model

    st.divider()

    # 회의록 저장 (git 채널 제외)
    ch = st.session_state.active_channel
    if ch != "git-현황":
        if st.button("📄 회의록 저장", use_container_width=True):
            if st.session_state.channel_messages[ch]:
                path = save_report(ch)
                st.success("저장 완료!")
                st.download_button("📥 다운로드", path.read_text(encoding="utf-8"), path.name, "text/markdown", use_container_width=True)
        if st.button("🔄 채널 초기화", use_container_width=True):
            st.session_state.channel_messages[ch] = []
            st.rerun()

        st.divider()
        st.markdown('<div class="section-label">이전 회의록</div>', unsafe_allow_html=True)
        docs_path = Path(__file__).parent / "docs"
        reports = sorted(docs_path.glob("report_*.md"), reverse=True)[:3]
        for r in reports:
            with st.expander(r.stem[-15:]):
                st.markdown(r.read_text(encoding="utf-8")[:400] + "...")

# ── 메인 영역 ─────────────────────────────────────────────────────
active_ch = st.session_state.active_channel
ch_info   = CHANNELS[active_ch]

# 채널 헤더
git_indicator = ""
if active_ch == "git-현황":
    changed = git_status_porcelain()
    if changed:
        n = len(changed.split("\n"))
        git_indicator = f'<span style="background:#e01e5a;color:#fff;font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px;">⬆ {n}개 미push</span>'
    else:
        git_indicator = '<span style="background:#2bac76;color:#fff;font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px;">✓ 동기화됨</span>'

st.markdown(f"""
<div class="channel-header">
    <span style="font-size:20px;color:#616061;">#</span>
    <span class="channel-header-name">{active_ch}</span>
    {git_indicator}
    <span style="color:#e8e8e8;font-size:18px;">|</span>
    <span class="channel-header-desc">{ch_info['desc']}</span>
</div>
""", unsafe_allow_html=True)

# ── Git 채널 vs 일반 채널 분기 ─────────────────────────────────────
if active_ch == "git-현황":
    render_git_channel()
else:
    chat_container = st.container(height=520, border=False)
    with chat_container:
        render_messages(active_ch)

    target = ch_info["target"]
    placeholder_map = {
        "전체 회의":  f"#{active_ch} 에 메시지 보내기 (학생1 조사 → 학생2 집필 → PI 검토)",
        "학생1만":    f"#{active_ch} 에 메시지 보내기 (김연구에게 지시)",
        "학생2만":    f"#{active_ch} 에 메시지 보내기 (이집필에게 지시)",
        "PI만":       f"#{active_ch} 에 메시지 보내기 (최교수에게 질문)",
    }

    if prompt := st.chat_input(placeholder_map[target]):
        add_message(active_ch, "나", prompt)
        history = "\n".join([
            f"[{m['speaker']}]: {m['content']}"
            for m in st.session_state.channel_messages[active_ch][-4:]
        ])
        result_1 = ""
        result_2 = ""

        # 학생1 — 문헌 조사
        if target in ["전체 회의", "학생1만"]:
            st.session_state.member_status["학생1"] = "busy"
            with st.status("🔬 김연구가 조사 중...", expanded=False):
                result_1 = run_agent("학생1", f"이전 대화:\n{history}\n\n지시: {prompt}")
                add_message(active_ch, "학생1", result_1)
            st.session_state.member_status["학생1"] = "online"

        # 학생2 — 논문 작성
        if target in ["전체 회의", "학생2만"]:
            st.session_state.member_status["학생2"] = "busy"
            with st.status("✍️ 이집필이 작성 중...", expanded=False):
                if target == "전체 회의":
                    task_2 = f"학생1의 문헌 조사 결과를 바탕으로 논문 내용을 작성/개선해주세요.\n\n원래 지시: {prompt}\n\n[학생1 문헌 조사 결과]\n{result_1}\n\nRSE 스타일에 맞게 작성해주세요."
                else:
                    task_2 = f"이전 대화:\n{history}\n\n지시: {prompt}"
                result_2 = run_agent("학생2", task_2)
                add_message(active_ch, "학생2", result_2)
            st.session_state.member_status["학생2"] = "online"

        # PI — 검토
        if target in ["전체 회의", "PI만"]:
            st.session_state.member_status["PI"] = "busy"
            with st.status("👨‍🏫 최교수가 검토 중...", expanded=False):
                if target == "전체 회의":
                    pi_task = (
                        f"학생들의 작업 결과를 최종 검토해주세요.\n\n원래 지시: {prompt}\n\n"
                        f"[학생1 문헌 조사]\n{result_1}\n\n"
                        f"[학생2 논문 작성]\n{result_2}\n\n"
                        f"간결하게 피드백과 다음 지시사항을 주세요."
                    )
                else:
                    pi_task = f"이전 대화:\n{history}\n\n질문/지시: {prompt}"
                result_pi = run_agent("PI", pi_task)
                add_message(active_ch, "PI", result_pi)
            st.session_state.member_status["PI"] = "online"

        st.rerun()

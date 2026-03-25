#!/usr/bin/env python3
# 시그널 큐 — 감지기 → position_updater 즉시 스캔 트리거
# 공지/급등/펀딩비 등 외부 감지기가 시그널을 쓰면, position_updater가 우선 처리
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import os
import fcntl

QUEUE_PATH = '/tmp/trade_signals.json'

def push_signal(symbol, source, direction='long', priority=5, meta=None):
    """시그널 큐에 추가 (파일 잠금으로 안전하게)"""
    signal = {
        'symbol': symbol,
        'source': source,       # 'announcement', 'surge', 'funding', 'wick'
        'direction': direction,  # 'long', 'short', 'any'
        'priority': priority,    # 1=최우선, 5=보통, 10=낮음
        'time': time.time(),
        'meta': meta or {},
    }
    try:
        # 파일 잠금
        fd = os.open(QUEUE_PATH, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        f = os.fdopen(fd, 'r+')
        try:
            content = f.read()
            queue = json.loads(content) if content.strip() else []
        except (json.JSONDecodeError, ValueError):
            queue = []

        # 같은 종목+소스 중복 방지 (5분 이내)
        queue = [s for s in queue
                 if not (s['symbol'] == symbol and s['source'] == source
                         and time.time() - s['time'] < 300)]
        queue.append(signal)

        f.seek(0)
        f.truncate()
        json.dump(queue, f, indent=2)
        f.flush()
        fcntl.flock(fd, fcntl.LOCK_UN)
        f.close()
        return True
    except Exception as e:
        print(f"[시그널큐] push 오류: {e}")
        return False


def pop_signals(max_age=600):
    """시그널 큐에서 유효한 시그널 꺼내기 (10분 이내만, 꺼낸 후 삭제)"""
    if not os.path.exists(QUEUE_PATH):
        return []
    try:
        fd = os.open(QUEUE_PATH, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        f = os.fdopen(fd, 'r+')
        try:
            content = f.read()
            queue = json.loads(content) if content.strip() else []
        except (json.JSONDecodeError, ValueError):
            queue = []

        now = time.time()
        valid = [s for s in queue if now - s['time'] < max_age]
        expired = len(queue) - len(valid)

        # 큐 비우기
        f.seek(0)
        f.truncate()
        json.dump([], f)
        f.flush()
        fcntl.flock(fd, fcntl.LOCK_UN)
        f.close()

        # 우선순위 정렬 (낮은 숫자 = 높은 우선순위)
        valid.sort(key=lambda x: x['priority'])
        return valid
    except Exception as e:
        print(f"[시그널큐] pop 오류: {e}")
        return []


def peek_signals():
    """큐 내용 확인 (삭제하지 않음)"""
    if not os.path.exists(QUEUE_PATH):
        return []
    try:
        with open(QUEUE_PATH, 'r') as f:
            return json.load(f)
    except:
        return []

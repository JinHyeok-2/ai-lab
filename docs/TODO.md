# 할 일 리스트 — AI 대학원 실험실

## Phase 1: Dify 설치 & 환경 구성
- [ ] Docker Desktop 설치 확인
- [ ] Dify docker-compose로 로컬 설치
  ```bash
  git clone https://github.com/langgenius/dify.git
  cd dify/docker
  docker compose up -d
  ```
- [ ] localhost:80 접속 확인
- [ ] Anthropic API 키 등록 (Settings → Model Provider)
- [ ] Claude Opus / Sonnet 모델 연결 확인

## Phase 2: 에이전트 프롬프트 완성
- [ ] PI 시스템 프롬프트 고도화
- [ ] 학생 A 프롬프트 + 웹검색 도구 연결
- [ ] 학생 B 프롬프트 + 코드실행 도구 연결
- [ ] 학생 C 프롬프트 + 코드실행 도구 연결

## Phase 3: Dify 워크플로우 구성
- [ ] 각 에이전트를 Dify "Agent" 앱으로 생성
- [ ] PI → 학생 A/B/C 태스크 분배 노드 구성
- [ ] 학생 결과물 → PI 리뷰 노드 연결
- [ ] 실험실 회의 플로우 (입력 → 토론 → 결론) 구성

## Phase 4: 파이프라인 구현
- [ ] 논문 서치 파이프라인 (주제 입력 → A가 논문 서치 → PI 검토)
- [ ] 데이터 분석 파이프라인 (데이터 경로 입력 → B가 분석 → 보고서)
- [ ] 코드 개발 파이프라인 (알고리즘 명세 → C가 구현 → 테스트)

## Phase 5: 실제 연구 태스크 테스트
- [ ] GK2A 화산재 탐지 관련 논문 서치 (학생 A 테스트)
- [ ] SO2 RGB 데이터 분석 (학생 B 테스트)
- [ ] Ash detection 알고리즘 코드 작성 (학생 C 테스트)
- [ ] PI 종합 리뷰 테스트

## Phase 6: 고도화 (선택)
- [ ] n8n으로 자동 트리거 연결 (새 데이터 → 자동 분석)
- [ ] Slack/메신저 알림 연동
- [ ] 결과물 자동 Git 저장
- [ ] 주간 회의록 자동 생성

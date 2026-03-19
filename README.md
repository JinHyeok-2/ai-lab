# AI 대학원 실험실 (AI-Lab)

## 프로젝트 개요
Dify 기반의 GUI 멀티에이전트 가상 실험실.
각 에이전트가 대학원생 역할을 맡아 연구 프로젝트를 협력 수행.

## 실험실 구성원
| 역할 | 에이전트 | 담당 업무 |
|------|---------|---------|
| 지도교수 (PI) | Opus 모델 | 연구 방향 결정, 최종 검토 |
| 대학원생 A | Sonnet | 논문 서치 & 문헌 정리 |
| 대학원생 B | Sonnet | 데이터 분석 & 시각화 |
| 대학원생 C | Sonnet | 코드 개발 & 실험 실행 |

## 기술 스택
- **오케스트레이션**: Dify (GUI 워크플로우)
- **에이전트 로직**: Claude API (Anthropic)
- **자동화**: n8n (선택적)
- **모델**: claude-opus-4-6 (PI), claude-sonnet-4-6 (학생들)

## 프로젝트 구조
```
ai-lab/
├── README.md
├── docs/           # 프로젝트 문서
├── agents/         # 각 에이전트 프롬프트 & 설정
│   ├── PI.md
│   ├── student_A.md
│   ├── student_B.md
│   └── student_C.md
└── workflows/      # Dify 워크플로우 DSL 파일
```

## 진행 단계
- [ ] Phase 1: Dify 설치 & 기본 설정
- [ ] Phase 2: 에이전트 역할 프롬프트 작성
- [ ] Phase 3: Dify 워크플로우 구성 (노드 연결)
- [ ] Phase 4: 실험실 회의 파이프라인 구현
- [ ] Phase 5: 논문 자동화 파이프라인 구현
- [ ] Phase 6: 실제 연구 태스크 투입 테스트

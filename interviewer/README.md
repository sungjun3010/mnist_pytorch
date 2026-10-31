# 토스 커머스 추천 MLE 모의 면접관

토스 **ML Engineer (Commerce 추천)** JD(`jd.md`)를 바탕으로 프리 인터뷰 / 심층 기술 / ML 시스템 설계 / 문화적합성 면접을 진행하는 면접관 에이전트입니다.

- `jd.md` — 공고 원문
- `interviewer_prompt.md` — 면접관 시스템 프롬프트 (질문 설계 원칙, 진행 방식, 피드백 형식)
- `run.py` — 터미널에서 돌리는 CLI (Claude API)
- `../.claude/skills/toss-interview/` — Claude Code 안에서 `/toss-interview` 로 실행

## 1) Claude Code에서 (가장 간단)

이 저장소를 연 Claude Code 세션에서:

```
/toss-interview design
```

## 2) 터미널 CLI

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # 또는 `ant auth login`
python interviewer/run.py --stage tech --resume my_resume.md
```

`--stage`: `pre`(프리 인터뷰) · `tech`(심층 기술, 기본값) · `design`(ML 시스템 설계) · `culture`(문화적합성)

면접 중 명령어: `/hint` 힌트, `/skip` 다음 주제, `/end` 종료 후 피드백, `/quit` 피드백 없이 종료.
기록은 `interviewer/transcripts/`에 저장됩니다 (git에서는 제외).

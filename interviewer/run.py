"""Toss ML Engineer (Commerce 추천) mock interviewer CLI.

Usage:
    python interviewer/run.py [--stage tech|design|pre|culture] [--resume resume.md]

Commands during the interview: /hint, /skip, /end (feedback), /quit (exit without feedback).
"""

import argparse
import datetime
import pathlib

import anthropic

HERE = pathlib.Path(__file__).parent
MODEL = "claude-opus-5-5"

STAGES = {
    "pre": "프리 인터뷰",
    "tech": "직무 인터뷰 – 심층 기술",
    "design": "직무 인터뷰 – ML 시스템 설계",
    "culture": "문화적합성 인터뷰",
}


def build_system_prompt() -> str:
    jd = (HERE / "jd.md").read_text(encoding="utf-8")
    template = (HERE / "interviewer_prompt.md").read_text(encoding="utf-8")
    return template.replace("{jd}", jd)


def build_opening(stage: str, resume: str | None) -> str:
    opening = f"면접을 시작해 주세요. 진행할 단계: {STAGES[stage]}."
    if resume:
        opening += f"\n\n제 이력서입니다.\n<resume>\n{resume}\n</resume>"
    else:
        opening += "\n\n이력서는 따로 제공하지 않았습니다."
    return opening


def ask(client: anthropic.Anthropic, system: str, messages: list) -> list:
    """Stream one interviewer turn to stdout and return the full content blocks."""
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        response = stream.get_final_message()
    print("\n")
    if response.stop_reason == "refusal":
        print("[응답이 거절되었습니다. 질문을 바꿔서 다시 시도해 주세요.]\n")
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="tech")
    parser.add_argument("--resume", type=pathlib.Path, help="이력서 텍스트/마크다운 파일 경로")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    system = build_system_prompt()
    resume = args.resume.read_text(encoding="utf-8") if args.resume else None
    messages = [{"role": "user", "content": build_opening(args.stage, resume)}]
    transcript = [f"# 모의 면접 – {STAGES[args.stage]} ({datetime.datetime.now():%Y-%m-%d %H:%M})\n"]

    print(f"=== 토스 커머스 추천 MLE 모의 면접: {STAGES[args.stage]} ===")
    print("명령어: /hint 힌트, /skip 다음 주제, /end 종료 후 피드백, /quit 그냥 종료\n")

    while True:
        content = ask(client, system, messages)
        messages.append({"role": "assistant", "content": content})
        transcript.append("**면접관:** " + "".join(b.text for b in content if b.type == "text") + "\n")

        if messages[-2]["content"] == "/end":
            break
        try:
            answer = input("나> ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "/quit"
        if answer == "/quit":
            break
        if not answer:
            answer = "(무응답)"
        messages.append({"role": "user", "content": answer})
        transcript.append(f"**나:** {answer}\n")
        print()

    out = HERE / "transcripts" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{args.stage}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(transcript), encoding="utf-8")
    print(f"기록 저장: {out}")


if __name__ == "__main__":
    main()

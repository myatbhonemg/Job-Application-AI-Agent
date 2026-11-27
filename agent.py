"""
Job application agent: a hand-written tool-use loop on the Anthropic SDK.

Usage:
    python agent.py <job_url> --cv path/to/cv.pdf [--max-turns 10] [--chat] [--yes]

With --chat, you can ask follow-up questions after the evaluation. The
conversation history is kept, so Claude remembers the posting and your CV.
"""

import argparse
import json
import os
import sys

import anthropic
from dotenv import load_dotenv

from tools import TOOLS, run_tool

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a job-application assistant helping a software engineering student.

For the job posting and CV you are given:
1. Fetch the job posting and read the CV with your tools.
2. Compare the role's requirements against the candidate's actual experience. Be honest:
   do not inflate the score.
3. Call save_application exactly once per job with the company, role, a 0-100 match score, and
   notes covering strengths, gaps, and concrete ways to tailor the CV for this role.
4. Finish with a short summary for the candidate.

The candidate may then ask follow-up questions (e.g. rewrite a CV section, draft a cover letter,
prepare for interviews). Answer them using what you already know; only call tools again if you
need new information or are asked to evaluate a different job.

save_application needs the candidate's approval. If they decline with feedback, revise the
record accordingly and call save_application again. If they decline without feedback, do not
retry; just carry on.

If a tool fails, say what went wrong and continue with what you have if that makes sense."""

# Tools that change something (here: write to applications.json) need a human
# to say yes first. Read-only tools run straight away.
NEEDS_APPROVAL = {"save_application"}


def preview(text: str, limit: int = 200) -> str:
    """One-line preview of a tool result for the terminal."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def ask_approval(name: str, tool_input: dict) -> tuple[bool, str]:
    """Show what Claude wants to do and ask the human. Returns (approved, feedback)."""
    print(f"\n✋ Claude wants to run {name}:")
    for key, value in tool_input.items():
        print(f"   {key}: {value}")
    try:
        answer = input("   Approve? [y = yes / n = no / or type feedback for Claude] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = "n"
    if answer.lower() in ("y", "yes"):
        return True, ""
    if answer.lower() in ("", "n", "no"):
        return False, ""
    return False, answer  # anything else is feedback, e.g. "score is too high"


def add_user_text(messages: list, text: str) -> None:
    """Add the user's typed text to the conversation.

    Normally that's a new user message. But if the loop stopped right after
    sending tool results (e.g. it hit --max-turns), the last message is
    already a user message, so we add the text to it instead of sending two
    user messages in a row.
    """
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"].append({"type": "text", "text": text})
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": text}]})


def run_agent(client: anthropic.Anthropic, messages: list, max_turns: int,
              auto_approve: bool = False) -> None:
    """Run the tool-use loop until Claude stops calling tools.

    `messages` is the conversation so far. The API is stateless: this list IS
    the agent's memory, and we send all of it on every turn. We modify it in
    place, so the caller keeps the full history for follow-up questions.
    """
    for turn in range(1, max_turns + 1):
        print(f"\n── turn {turn} " + "─" * 50)

        # 1. Ask Claude what to do next, telling it which tools exist.
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
        )
        print(f"   (tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out,"
              f" stop_reason={response.stop_reason})")

        # 2. Save Claude's reply in the history *unchanged*. It can contain
        #    text, tool_use and thinking blocks, and the API needs them all back.
        messages.append({"role": "assistant", "content": response.content})

        # Show anything Claude said in plain text (its narration / final answer).
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n🤖 {block.text.strip()}")

        # 3. Decide what to do based on WHY Claude stopped.
        if response.stop_reason == "end_turn":
            print("\n✅ Agent finished.")
            return
        if response.stop_reason == "max_tokens":
            print("\n⚠️  Claude hit max_tokens before finishing; stopping.")
            return
        if response.stop_reason == "refusal":
            print("\n⚠️  Claude declined this request; stopping.")
            return
        if response.stop_reason != "tool_use":
            print(f"\n⚠️  Unexpected stop_reason {response.stop_reason!r}; stopping.")
            return

        # 4. stop_reason == "tool_use": run every tool Claude asked for.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\n🔧 {block.name}({json.dumps(block.input, ensure_ascii=False)})")

            # Human in the loop: Claude only *asked* for this tool, so we can
            # still say no. A "no" is sent back as the tool result, which
            # Claude reads like any other result and can react to.
            approved, feedback = True, ""
            if block.name in NEEDS_APPROVAL and not auto_approve:
                approved, feedback = ask_approval(block.name, block.input)

            if approved:
                result, is_error = run_tool(block.name, block.input)
            else:
                result = "The user declined this action. Nothing was saved."
                if feedback:
                    result += f" Their feedback: {feedback}"
                is_error = False  # a decision, not a failure
            print(f"   {'❌' if is_error else '↳'} {preview(result)}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # links this result to Claude's request
                "content": result,
                "is_error": is_error,
            })

        # 5. Send ALL results back in ONE user message, then loop.
        messages.append({"role": "user", "content": tool_results})

    print(f"\n⚠️  Stopped after {max_turns} turns (safety limit).")


def run_agent_safely(client: anthropic.Anthropic, messages: list, max_turns: int,
                     auto_approve: bool = False) -> bool:
    """Run the loop, turning API errors into messages. Returns False on failure."""
    try:
        run_agent(client, messages, max_turns, auto_approve)
        return True
    except anthropic.AuthenticationError:
        print("\n❌ Authentication failed: check the API key in your .env file.")
    except anthropic.RateLimitError:
        print("\n❌ Rate limited by the API. Wait a moment and try again.")
    except anthropic.APIStatusError as exc:
        print(f"\n❌ API error {exc.status_code}: {exc.message}")
    except anthropic.APIConnectionError:
        print("\n❌ Could not reach the Anthropic API. Check your internet connection.")
    return False


def chat(client: anthropic.Anthropic, messages: list, max_turns: int,
         auto_approve: bool = False) -> None:
    """Follow-up questions: read a line, add it to the SAME history, run the loop again."""
    print("\n💬 Chat mode: ask follow-up questions. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            return

        add_user_text(messages, question)
        before = len(messages)
        if not run_agent_safely(client, messages, max_turns, auto_approve):
            # The request failed, so Claude never saw the question. Take it
            # back out of the history so you can just type it again.
            del messages[before:]
            last = messages[-1]["content"]
            last.pop()
            if not last:
                messages.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a job posting against your CV.")
    parser.add_argument("job_url", help="URL of the job posting")
    parser.add_argument("--cv", required=True, help="Path to your CV (.pdf, .txt or .md)")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Safety limit on loop turns (per question in chat mode)")
    parser.add_argument("--chat", action="store_true",
                        help="Keep chatting after the evaluation to ask follow-up questions")
    parser.add_argument("--yes", action="store_true",
                        help="Auto-approve save_application without asking")
    args = parser.parse_args()

    load_dotenv()  # copies values from .env into os.environ
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    request = (
        f"Please evaluate this job for me.\n"
        f"Job posting URL: {args.job_url}\n"
        f"My CV is at: {args.cv}"
    )

    messages: list = []
    add_user_text(messages, request)

    if not run_agent_safely(client, messages, args.max_turns, args.yes):
        sys.exit(1)
    if args.chat:
        chat(client, messages, args.max_turns, args.yes)


if __name__ == "__main__":
    main()

# Job Application Agent

A small command-line AI agent that reads a job posting and your CV, judges how well they match, and logs the result to `applications.json`. In chat mode it can also look back over past evaluations and write cover letters.

## Tools

| Tool | What it does | Needs approval? |
|---|---|---|
| `fetch_job_posting(url)` | Downloads a job page and returns its text | no |
| `read_cv(path)` | Reads your CV (`.pdf`, `.txt`, `.md`) | no |
| `save_application(company, role, match_score, notes)` | Appends an evaluation to `applications.json` | **yes** |
| `list_applications()` | Returns every saved evaluation, so Claude can answer "which jobs scored above 70?" | no |
| `draft_cover_letter(company, role, letter)` | Saves a cover letter **Claude wrote** to `cover_letters/<company>_<role>.md` (never overwrites; extra drafts get `_2`, `_3`) | **yes** |

`draft_cover_letter` shows a pattern worth noticing: **the tool doesn't generate the letter. Claude does.** Claude writes the text and passes it as the `letter` argument, and the tool only saves it. Since saving needs approval, you read the whole letter first and can reply with feedback ("mention my test-coverage result") to get a revised draft.

It uses the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) directly. The **agent loop is written by hand** (no agent framework, no SDK Tool Runner), so you can see exactly how an agent works.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # skip if .venv already exists
pip install -r requirements.txt
cp .env.example .env        # then open .env and paste your real API key
```

`.env` is listed in `.gitignore`, so your key never gets committed. Check with `git status`: `.env` should not appear.

## Usage

```bash
python agent.py https://example.com/jobs/backend-engineer --cv sample_cv.txt
```

| Option | Meaning |
|---|---|
| `job_url` | The job posting page |
| `--cv` | Your CV as `.pdf`, `.txt` or `.md` |
| `--max-turns` | Safety limit on loop iterations (default 10, applied to each question in chat mode) |
| `--chat` | After the evaluation, stay open for follow-up questions |
| `--yes` | Skip the approval prompt and save automatically |

### Approving saves (human in the loop)

Before anything is written to `applications.json`, the agent shows you the record and asks:

```
✋ Claude wants to run save_application:
   company: Acme Ltd
   role: Graduate Backend Engineer
   match_score: 85
   notes: STRENGTHS: ...
   Approve? [y = yes / n = no / or type feedback for Claude]
```

- `y`: save it.
- `n` (or Enter): don't save. Claude is told you declined.
- Anything else is **feedback**, e.g. `score is too high, make it 70`. Nothing is saved; Claude reads your feedback, revises, and asks again.

This works because of how tool use is designed: Claude only *requests* a tool, and **your code decides whether to run it.** A "no" goes back as the `tool_result`, just like real output would:

```python
if block.name in NEEDS_APPROVAL and not auto_approve:
    approved, feedback = ask_approval(block.name, block.input)
if approved:
    result, is_error = run_tool(block.name, block.input)
else:
    result = "The user declined this action. Nothing was saved. Their feedback: ..."
```

Read-only tools (`fetch_job_posting`, `read_cv`, `list_applications`) run without asking. `save_application` and `draft_cover_letter` write files, so they need a yes. Only tools listed in `NEEDS_APPROVAL` in `agent.py` need a yes. Add a tool's name there to gate it too.

### Chat mode

```bash
python agent.py https://example.com/jobs/backend-engineer --cv sample_cv.txt --chat
```

After the evaluation you get a `you>` prompt. Ask things like *"rewrite my summary for this role"*, *"draft a cover letter, under 200 words"*, *"which of my saved jobs scored highest?"* or *"what interview questions should I prepare for?"*. Type `exit` or press Ctrl-D to quit.

Chat mode is a second loop **around** the agent loop:

```python
while True:
    question = input("you> ")
    add_user_text(messages, question)   # same messages list as before
    run_agent(client, messages, ...)     # the exact same tool-use loop
```

Nothing new is needed for Claude to "remember" the job posting and your CV: they're already in `messages` as earlier tool results, and the whole list is resent every turn. Claude can still call tools during chat, for example if you paste a second job URL.

Each question makes the history longer, so watch the `tokens: ... in` number grow.

Example output:

```
── turn 1 ──────────────────────────────────────────────────
   (tokens: 1210 in / 140 out, stop_reason=tool_use)

🤖 I'll start by fetching the job posting and reading your CV.

🔧 fetch_job_posting({"url": "https://example.com/jobs/backend-engineer"})
   ↳ Backend Engineer – Acme Ltd. We're looking for a graduate engineer with Python...

🔧 read_cv({"path": "sample_cv.txt"})
   ↳ Alex Tan Final-year BSc Software Engineering student ...

── turn 2 ──────────────────────────────────────────────────
   (tokens: 4630 in / 410 out, stop_reason=tool_use)

🔧 save_application({"company": "Acme Ltd", "role": "Backend Engineer", "match_score": 72, "notes": "..."})
   ↳ Saved application #1: Backend Engineer at Acme Ltd (score 72).

── turn 3 ──────────────────────────────────────────────────
   (tokens: 5110 in / 180 out, stop_reason=end_turn)

🤖 You're a solid match (72/100). Strengths: ... Gaps: ...

✅ Agent finished.
```

Notice that Claude asked for **two tools in one turn** (parallel tool use) and the loop ran both before calling the API again.

## How the agent loop works

An "agent" is just this: **a language model, some tools, and a loop that lets the model use those tools until it decides it's done.**

```
 you ──► agent.py ──► first user message: "evaluate this job for this CV"
                │
                ▼
        ┌──── AGENT LOOP (run_agent in agent.py) ─────────────────────┐
        │ 1. client.messages.create(model, system, tools, messages)   │
        │ 2. append Claude's reply (role=assistant) to messages       │
        │ 3. look at response.stop_reason                             │
        │      "tool_use" → run each requested tool in tools.py,      │
        │                   send ALL results back in ONE user         │
        │                   message, go to 1                          │
        │      "end_turn" → Claude is done, print answer, exit        │
        │      other      → stop and say why                          │
        └─────────────────────────────────────────────────────────────┘
```

### Step by step

**0. Describe the tools.** In `tools.py`, `TOOLS` is a list of JSON schemas. Each one has a `name`, a `description` saying *when* to use it, and an `input_schema` describing its arguments. This is everything Claude knows about a tool. Claude never sees your Python code.

**1. Call the model.** `client.messages.create(...)` sends the system prompt, the tool schemas and the **entire conversation so far**. The API is *stateless*: it remembers nothing between calls, so the `messages` list is the agent's whole memory.

**2. Claude replies with content blocks.** `response.content` is a list. Each block is one of:
- `text`: Claude talking ("I'll fetch the posting first...")
- `tool_use`: a *request* to run a tool, with an `id`, a `name` and an `input` dict
- `thinking`: Claude's internal reasoning (this model thinks by default)

We append the whole `response.content` to `messages` **unchanged**. If you dropped the `tool_use` or `thinking` blocks, the next request would fail or lose context.

**3. Check `stop_reason`.** This tells you *why* Claude stopped generating:

| `stop_reason` | Meaning | What the loop does |
|---|---|---|
| `tool_use` | "I want you to run these tools" | run them, continue |
| `end_turn` | "I'm finished" | print the answer, exit |
| `max_tokens` | ran out of output space | stop |
| `refusal` | declined the request | stop |

**4. Run the tools.** For every `tool_use` block, `run_tool(name, input)` calls the matching Python function. **Claude doesn't execute anything. Your code does.** This is the key idea: the model decides, the program acts.

**5. Return the results.** Each result goes into a `tool_result` block whose `tool_use_id` matches the request's `id`, so Claude knows which answer belongs to which call. All results from one turn go back together in **one** `user` message. Then loop to step 1.

### Errors are just information

If a tool fails (bad URL, missing file), `run_tool` catches the exception and returns its message with `is_error: True`. Claude reads the error like any other result and can retry, try something else, or explain the problem to you. The program doesn't crash.

### Safety limit

`--max-turns` caps how many times the loop can call the API. Without a cap, a confused model could keep calling tools forever and run up a bill.

### Cost

Every turn resends the whole conversation, so input tokens grow each turn (see the `tokens:` line). That's why `fetch_job_posting` trims page text to 15,000 characters.

## Project layout

```
agent.py          CLI + system prompt + the hand-written agent loop
tools.py          tool schemas (what Claude sees) + implementations (what runs)
requirements.txt  dependencies
.env.example      template for your API key (.env itself is git-ignored)
sample_cv.txt     a fake CV to test with
applications.json created on first save (git-ignored)
cover_letters/    saved cover letters (git-ignored)
```

## Limitations

- Pages that render their content with JavaScript (some LinkedIn and Workday pages) return little or no text. Paste the posting into a `.txt` file instead, or add a tool that reads local files.
- Scanned (image-only) PDF CVs have no text to extract. Use a text-based PDF or a `.txt` copy.

## Ideas for v2

- **More tools:** `update_application(id, ...)` to change a saved record, `export_cover_letter_pdf`, or a `search_jobs` tool backed by a jobs API.
- **Streaming:** use `client.messages.stream(...)` to print Claude's text as it's generated.
- **Prompt caching:** add `cache_control` so the repeated system prompt, tools and history are billed at the cheaper cached rate.
- **Refusal fallback:** the API has a beta `fallbacks` parameter that retries a refused request on another model automatically.
- **Compare with the SDK's Tool Runner** (`client.beta.messages.tool_runner`). It runs this same loop for you. Now that you've written the loop yourself, you'll know what it does.

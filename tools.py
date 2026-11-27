"""
The agent's tools.

Each tool has two halves:
  1. A JSON schema (in TOOLS) that we send to Claude. This is all Claude ever
     sees: a name, a description of when to use it, and the shape of its input.
  2. A plain Python function that actually does the work. Claude never runs
     this; our agent loop does, when Claude asks for it.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

APPLICATIONS_FILE = Path(__file__).parent / "applications.json"
COVER_LETTERS_DIR = Path(__file__).parent / "cover_letters"
MAX_POSTING_CHARS = 15_000  # keep the page text small enough to be cheap to send
VISA_FIT_VALUES = ["ok", "needs sponsorship", "unclear"]

# Every job posting text Claude has been shown in this run (fetched pages and
# --job-file text). save_application checks its quotes against these, so
# Claude can't put words in the posting's mouth.
_seen_postings: list[str] = []


def _normalize(text: str) -> str:
    """Lowercase, unify curly quotes, collapse whitespace: for fair quote matching."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(text.lower().split())


def remember_posting(text: str) -> None:
    """Record a posting text so quotes can be checked against it."""
    _seen_postings.append(_normalize(text))


def _is_real_quote(quote: str) -> bool:
    """True if every part of the quote (split at '...') appears in a seen posting."""
    if not _seen_postings:
        return True  # nothing to check against (e.g. tool called directly)
    parts = [_normalize(part).strip(" \"'") for part in re.split(r"\.\.\.|\u2026", quote)]
    parts = [part for part in parts if part]
    return any(all(part in posting for part in parts) for posting in _seen_postings)


# ---------------------------------------------------------------------------
# 1. Tool implementations (plain Python)
# ---------------------------------------------------------------------------

def fetch_job_posting(url: str) -> str:
    """Download a web page and return its visible text."""
    response = httpx.get(
        url,
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (job-application-agent; learning project)"},
    )
    response.raise_for_status()  # 404 / 500 etc. become an exception -> error result

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        tag.decompose()

    # Collapse whitespace: one non-empty line per line of text.
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)

    if not text:
        raise ValueError("The page had no readable text (it may need JavaScript to render).")
    if len(text) > MAX_POSTING_CHARS:
        text = text[:MAX_POSTING_CHARS] + f"\n\n[...truncated, page was {len(text)} characters]"
    remember_posting(text)
    return text


def read_cv(path: str) -> str:
    """Read a CV from a .pdf, .txt or .md file and return its text."""
    file = Path(path).expanduser()
    if not file.exists():
        raise FileNotFoundError(f"No file at {file}")

    suffix = file.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".txt", ".md"):
        text = file.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported CV format '{suffix}'. Use .pdf, .txt or .md.")

    if not text.strip():
        raise ValueError("The CV file contained no extractable text (scanned PDF?).")
    return text


def save_application(company: str, role: str, match_score: int, notes: str,
                     visa_fit: str, visa_reason: str,
                     programme_length_quote: str, sponsorship_quote: str) -> str:
    """Append one application record to applications.json.

    The visa checks below are enforced in code, not just asked for in the
    prompt: if Claude breaks a rule, it gets an error back and must fix it.
    """
    if not 0 <= match_score <= 100:
        raise ValueError(f"match_score must be between 0 and 100, got {match_score}")
    if visa_fit not in VISA_FIT_VALUES:
        raise ValueError(f"visa_fit must be one of {VISA_FIT_VALUES}, got {visa_fit!r}")

    visa_reason = " ".join(visa_reason.split())  # force a single line
    if not visa_reason:
        raise ValueError("visa_reason must be a one-line reason, not empty.")

    # Rule 1: quotes must really come from the posting.
    for field, quote in (("programme_length_quote", programme_length_quote),
                         ("sponsorship_quote", sponsorship_quote)):
        if quote.strip() and not _is_real_quote(quote):
            raise ValueError(
                f"{field} is not an exact quote from the job posting: {quote!r}. "
                'Copy the posting\'s words exactly, or use "" if it says nothing about this.'
            )

    # Rule 2: no sponsorship statement in the posting -> no guessing.
    if not sponsorship_quote.strip() and visa_fit != "unclear":
        raise ValueError(
            "The posting says nothing about visa sponsorship (sponsorship_quote is empty), "
            f'so visa_fit must be "unclear", not {visa_fit!r}.'
        )

    records = []
    if APPLICATIONS_FILE.exists():
        records = json.loads(APPLICATIONS_FILE.read_text(encoding="utf-8"))

    records.append({
        "company": company,
        "role": role,
        "match_score": match_score,
        "notes": notes,
        "visa_fit": visa_fit,
        "visa_reason": visa_reason,
        "programme_length_quote": programme_length_quote.strip(),
        "sponsorship_quote": sponsorship_quote.strip(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    APPLICATIONS_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return (f"Saved application #{len(records)}: {role} at {company} "
            f"(score {match_score}, visa_fit: {visa_fit}).")


def list_applications() -> str:
    """Return every saved application as JSON, numbered from 1."""
    if not APPLICATIONS_FILE.exists():
        return "No applications saved yet."
    records = json.loads(APPLICATIONS_FILE.read_text(encoding="utf-8"))
    if not records:
        return "No applications saved yet."
    numbered = [{"id": i, **record} for i, record in enumerate(records, start=1)]
    return json.dumps(numbered, indent=2, ensure_ascii=False)


def _slug(text: str) -> str:
    """'Acme Ltd.' -> 'acme-ltd' (safe to use in a file name)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


def draft_cover_letter(company: str, role: str, letter: str) -> str:
    """Save a cover letter Claude has written to cover_letters/<company>_<role>.md.

    Claude does the writing; this tool just saves the text. Existing files
    are never overwritten: a second draft becomes ..._2.md, and so on.
    """
    if not letter.strip():
        raise ValueError("The cover letter is empty.")
    COVER_LETTERS_DIR.mkdir(exist_ok=True)
    stem = f"{_slug(company)}_{_slug(role)}"
    path = COVER_LETTERS_DIR / f"{stem}.md"
    n = 2
    while path.exists():
        path = COVER_LETTERS_DIR / f"{stem}_{n}.md"
        n += 1
    path.write_text(letter.strip() + "\n", encoding="utf-8")
    return f"Saved cover letter to {path.relative_to(Path(__file__).parent)} ({len(letter.split())} words)."


# ---------------------------------------------------------------------------
# 2. Tool definitions (what Claude sees)
# ---------------------------------------------------------------------------
# `strict: True` makes the API guarantee that Claude's tool input matches the
# schema exactly (needs `additionalProperties: False` and a `required` list).

TOOLS = [
    {
        "name": "fetch_job_posting",
        "description": (
            "Download a job posting web page and return its visible text. "
            "Use this first to learn the role's requirements."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full http(s) URL of the job posting."},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_cv",
        "description": (
            "Read the candidate's CV from a local .pdf, .txt or .md file and return its text."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the CV file."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "save_application",
        "description": (
            "Record the evaluation of one job in the applications log, including a visa "
            "eligibility check backed by exact quotes from the posting. "
            "Call this exactly once, after you have read both the posting and the CV."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name."},
                "role": {"type": "string", "description": "Job title."},
                "match_score": {
                    "type": "integer",
                    "description": "How well the CV fits the role, from 0 (no fit) to 100 (perfect fit).",
                },
                "notes": {
                    "type": "string",
                    "description": "Strengths, gaps, and concrete tips for tailoring the CV.",
                },
                "visa_fit": {
                    "type": "string",
                    "enum": VISA_FIT_VALUES,
                    "description": (
                        "Whether the candidate's eligibility rules (from their profile) fit this job. "
                        '"ok": the posting clearly works for them. "needs sponsorship": the posting '
                        'shows they would need sponsorship. "unclear": the posting does not say '
                        "enough (always use this if it says nothing about sponsorship, or if no "
                        "profile was given)."
                    ),
                },
                "visa_reason": {
                    "type": "string",
                    "description": "One line explaining visa_fit, based on the quotes and the profile.",
                },
                "programme_length_quote": {
                    "type": "string",
                    "description": (
                        "The posting's exact words about programme or contract length, copied "
                        'verbatim. Use "" if the posting says nothing about it. Never paraphrase.'
                    ),
                },
                "sponsorship_quote": {
                    "type": "string",
                    "description": (
                        "The posting's exact words about visa sponsorship or right to work, copied "
                        'verbatim. Use "" if the posting says nothing about it. Never paraphrase.'
                    ),
                },
            },
            "required": ["company", "role", "match_score", "notes", "visa_fit", "visa_reason",
                         "programme_length_quote", "sponsorship_quote"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_applications",
        "description": (
            "List every job the candidate has evaluated so far, with company, role, match score, "
            "notes and date. Use this to answer questions about past applications or compare jobs."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "draft_cover_letter",
        "description": (
            "Save a cover letter you have written for a specific job to a Markdown file. "
            "Write the complete letter yourself, tailored to the posting and grounded only in "
            "facts from the CV, and pass the full text. Only use this when the candidate asks "
            "for a cover letter."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name."},
                "role": {"type": "string", "description": "Job title."},
                "letter": {
                    "type": "string",
                    "description": "The full cover letter text (Markdown), ready to send.",
                },
            },
            "required": ["company", "role", "letter"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# 3. Dispatcher: the agent loop calls this for every tool_use block
# ---------------------------------------------------------------------------

_FUNCTIONS = {
    "fetch_job_posting": fetch_job_posting,
    "read_cv": read_cv,
    "save_application": save_application,
    "list_applications": list_applications,
    "draft_cover_letter": draft_cover_letter,
}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a tool by name. Returns (result_text, is_error).

    We never let an exception escape: errors go back to Claude as text so it
    can decide what to do (retry, ask, or explain the problem).
    """
    func = _FUNCTIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}", True
    try:
        return func(**tool_input), False
    except Exception as exc:  # noqa: BLE001 - deliberately report any failure to Claude
        return f"{type(exc).__name__}: {exc}", True

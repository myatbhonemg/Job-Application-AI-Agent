"""
The agent's tools.

Each tool has two halves:
  1. A JSON schema (in TOOLS) that we send to Claude. This is all Claude ever
     sees: a name, a description of when to use it, and the shape of its input.
  2. A plain Python function that actually does the work. Claude never runs
     this; our agent loop does, when Claude asks for it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

APPLICATIONS_FILE = Path(__file__).parent / "applications.json"
MAX_POSTING_CHARS = 15_000  # keep the page text small enough to be cheap to send


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


def save_application(company: str, role: str, match_score: int, notes: str) -> str:
    """Append one application record to applications.json."""
    if not 0 <= match_score <= 100:
        raise ValueError(f"match_score must be between 0 and 100, got {match_score}")

    records = []
    if APPLICATIONS_FILE.exists():
        records = json.loads(APPLICATIONS_FILE.read_text(encoding="utf-8"))

    records.append({
        "company": company,
        "role": role,
        "match_score": match_score,
        "notes": notes,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    APPLICATIONS_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return f"Saved application #{len(records)}: {role} at {company} (score {match_score})."


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
            "Record the evaluation of one job in the applications log. "
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
            },
            "required": ["company", "role", "match_score", "notes"],
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

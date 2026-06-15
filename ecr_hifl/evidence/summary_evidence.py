"""Summary evidence: compress each candidate's files to a structural summary and score its
overlap with the issue.

Offline (default): the "summary" is built from the structure JSON — file path + class/function
signatures (from parsed symbols) + the file's leading docstring line — i.e. no LLM, no network.
The score is the token-overlap (Jaccard) between the issue and the candidate's summary text,
min-max normalized across the instance's candidates. This is a distinct signal from
``semantic`` (which uses the raw candidate response): it measures issue↔code alignment on a
compressed, signature-level view, echoing Meta-RAG's summary-compression idea.

Online (optional, ``cfg.use_summary_server``): an LLM summary via ``EP.generation`` replaces the
structural summary; falls back automatically when the endpoint is unreachable.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..candidates import Candidate
from ..io_utils import RepoStructure
from .evidence_card import EvidenceBuilderBase

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _tokens(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _leading_docstring(lines: List[str]) -> str:
    for ln in lines[:5]:
        s = ln.strip()
        if s.startswith(('"""', "'''")):
            return s.strip("\"'")
    return ""


def structural_summary(structure: RepoStructure, files: List[str], max_files: int = 5) -> str:
    """A compact, signature-level summary of the candidate's files (offline)."""
    parts = []
    symbols_by_file = structure.symbols_by_file
    text_by_file = structure.file_text
    for f in files[:max_files]:
        syms = symbols_by_file.get(f, [])
        names = [s["name"] for s in syms if s["kind"] in ("class", "function")][:12]
        doc = _leading_docstring(text_by_file.get(f, []))
        parts.append(f"{f}: " + ", ".join(names) + (f" -- {doc}" if doc else ""))
    return "\n".join(parts)


class SummaryEvidenceBuilder(EvidenceBuilderBase):
    name = "summary"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.use_server = bool(getattr(cfg, "use_summary_server", False)) if cfg else False

    def build_all(self, candidates: List[Candidate], ctx) -> List[dict]:
        structure: Optional[RepoStructure] = ctx.structure
        if structure is None or not candidates:
            return [{"score": 0.0} for _ in candidates]
        issue_tok = _tokens(ctx.problem_statement)
        raw, rich = [], []
        for c in candidates:
            summary = structural_summary(structure, list(dict.fromkeys(c.files)))
            if self.use_server:
                summary = self._maybe_llm_summary(structure, c, ctx) or summary
            stok = _tokens(summary)
            overlap = len(issue_tok & stok) / len(issue_tok | stok) if (issue_tok | stok) else 0.0
            raw.append(overlap)
            rich.append({"summary": summary[:600], "summary_match": round(overlap, 4)})
        lo, hi = (min(raw), max(raw)) if raw else (0.0, 0.0)
        span = (hi - lo) or 1.0
        for r, v in zip(rich, raw):
            r["score"] = round((v - lo) / span, 4)
        return rich

    def _maybe_llm_summary(self, structure, candidate, ctx) -> Optional[str]:
        """Summarize candidate files via the generation server (EP.generation); None if offline."""
        try:
            from ..coreutils import get_endpoints, get_repo_files

            ep = get_endpoints()
            if ep is None or not getattr(ep, "generation_base_url", None):
                return None
            import openai

            files = list(dict.fromkeys(candidate.files))[:2]
            code = get_repo_files(structure.structure, files)
            blob = "\n\n".join(f"### {f}\n{code[f][:6000]}" for f in files if f in code)
            prompt = ("Summarize these Python file(s) for bug localization. List the main "
                      "responsibility and key classes/functions in <=120 words.\n\n" + blob)
            client = openai.OpenAI(base_url=ep.generation_base_url, api_key=ep.generation_api_key,
                                   timeout=20.0, max_retries=0)
            resp = client.chat.completions.create(
                model=getattr(self.cfg, "summary_model", "q7bc"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256, temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception:
            return None

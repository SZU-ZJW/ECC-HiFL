"""Verification evidence: lightweight plausibility check linking localization to repair.

 v1: ask an LLM whether a candidate location plausibly needs modification to fix
the issue, returning a 0-1 plausibility score. This needs the generation server (``EP.generation``).
Offline it returns a neutral score (0.5, ``plausible=None``) so it neither helps nor hurts the
weighted selector — keeping the offline pipeline fully runnable. v2 (generate a patch + run
tests) lives in :mod:`ecr_hifl.repair`.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from ..candidates import Candidate
from .evidence_card import EvidenceBuilderBase
from .summary_evidence import structural_summary

_PROMPT = """Given a GitHub issue and a candidate code location, judge whether this location is \
likely to require modification to fix the issue.
Return ONLY JSON: {{"plausible": true/false, "score": 0-1, "reason": "..."}}

### Issue
{issue}

### Candidate location
{candidate}
"""


def _parse_json_score(text: str):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
        return d.get("plausible"), float(d.get("score", 0.5))
    except Exception:
        return None, None


class VerificationEvidenceBuilder(EvidenceBuilderBase):
    name = "verification"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.use_server = bool(getattr(cfg, "use_verification_server", False)) if cfg else False

    def build_all(self, candidates: List[Candidate], ctx) -> List[dict]:
        if not candidates:
            return []
        client = self._client() if self.use_server else None
        if client is None:
            return [{"score": 0.5, "plausible": None, "available": False} for _ in candidates]
        out = []
        for c in candidates:
            loc = structural_summary(ctx.structure, list(dict.fromkeys(c.files))) if ctx.structure else c.raw_text
            plausible, score = self._query(client, ctx.problem_statement, loc)
            out.append({"available": True, "plausible": plausible,
                        "score": round(score if score is not None else 0.5, 4)})
        return out

    def _client(self):
        try:
            from ..coreutils import get_endpoints
            import openai

            ep = get_endpoints()
            if ep is None or not getattr(ep, "generation_base_url", None):
                return None
            return openai.OpenAI(base_url=ep.generation_base_url, api_key=ep.generation_api_key,
                                 timeout=20.0, max_retries=0)
        except Exception:
            return None

    def _query(self, client, issue: str, candidate_loc: str):
        try:
            resp = client.chat.completions.create(
                model=getattr(self.cfg, "verification_model", "q7bc"),
                messages=[{"role": "user", "content": _PROMPT.format(issue=issue, candidate=candidate_loc)}],
                max_tokens=200, temperature=0.0,
            )
            return _parse_json_score(resp.choices[0].message.content)
        except Exception:
            return None, 0.5

"""v2 LLM-as-judge selector.

Compresses each candidate's evidence card into a compact block and asks an LLM to pick the most
likely buggy location, returning JSON ``{"selected_index": int, "confidence": float, "reason": ...}``.
Uses the judge endpoint (``EP.judge_*``) if configured, else the generation endpoint
(``EP.generation_*``). When neither is reachable (offline), it transparently **falls back to the
rule selector** so the pipeline still runs and the ``reason`` field is simply absent.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from ..config import ECRConfig
from .base import Selection, SelectionContext, Selector
from .rule_selector import RuleSelector

_PROMPT = """You are a repository-level bug localization selector. Given a GitHub issue and a list \
of candidate {level}-level locations (each with supporting evidence), select the single candidate \
most likely to be the correct location to edit.

### Issue
{issue}

### Candidates
{candidates}

Return ONLY JSON: {{"selected_index": <1-based index>, "confidence": <0-1>, "reason": "<short>"}}
"""


def _card_block(idx: int, candidate, card) -> str:
    ev = card.evidence if card else {}
    sem = ev.get("graph", {})
    lines = [f"[candidate {idx}] files: {', '.join(candidate.files[:5]) or '(none)'}"]
    if candidate.elements:
        locs = [l for v in candidate.elements.values() for l in v][:6]
        lines.append("  locations: " + "; ".join(locs))
    if card:
        s = ev.get("semantic", {})
        rm = card.rm_score if "rm_score" in s else s.get("vote_score", 0)
        lines.append(f"  evidence: rm={rm:.2f} vote={s.get('vote_score',0):.2f} sim={s.get('tfidf_score',0):.2f} "
                     f"graph={ev.get('graph',{}).get('score',0):.2f} "
                     f"summary={ev.get('summary',{}).get('summary_match',0):.2f}")
        summ = ev.get("summary", {}).get("summary")
        if summ:
            lines.append("  summary: " + summ.replace("\n", " ")[:200])
    return "\n".join(lines)


def _parse_index(text: str) -> Optional[int]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if "selected_index" in d:
                return int(d["selected_index"])
        except Exception:
            pass
    m = re.search(r"candidate\s*(\d+)", text or "", re.IGNORECASE)
    return int(m.group(1)) if m else None


class LLMSelector(Selector):
    name = "llm"

    def __init__(self, cfg: ECRConfig):
        self.cfg = cfg
        self._fallback = RuleSelector(cfg.weights)

    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        if not candidates:
            return Selection(index=None, fields=self.empty_fields(ctx.level))
        client, model = self._client_and_model()
        if client is None:
            sel = self._fallback.predict(candidates, ctx)
            sel.meta = {**sel.meta, "llm": "unavailable -> rule fallback"}
            return sel
        cards = ctx.cards or [None] * len(candidates)
        blocks = "\n\n".join(_card_block(i + 1, c, cards[i]) for i, c in enumerate(candidates))
        prompt = _PROMPT.format(level=ctx.level, issue=ctx.problem_statement[:6000], candidates=blocks)
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=256, temperature=0.0)
            text = resp.choices[0].message.content
        except Exception as exc:
            sel = self._fallback.predict(candidates, ctx)
            sel.meta = {**sel.meta, "llm": f"error -> rule fallback: {exc}"}
            return sel
        idx = _parse_index(text)
        if idx is None or not (1 <= idx <= len(candidates)):
            sel = self._fallback.predict(candidates, ctx)
            sel.meta = {**sel.meta, "llm": "unparseable -> rule fallback", "raw": (text or "")[:200]}
            return sel
        return Selection(index=idx - 1, fields=candidates[idx - 1].result_fields(),
                         meta={"llm": "ok", "raw": (text or "")[:200]})

    def _client_and_model(self):
        # Only touch the network when explicitly enabled, so offline runs never hang on a
        # configured-but-unreachable endpoint. Fast-fail timeouts as a second line of defense.
        if not self.cfg.use_llm_judge:
            return None, None
        try:
            from ..coreutils import get_endpoints
            import openai

            ep = get_endpoints()
            if ep is None:
                return None, None
            if getattr(ep, "judge_base_url", None):
                client = openai.OpenAI(base_url=ep.judge_base_url, api_key=ep.judge_api_key,
                                       timeout=20.0, max_retries=0)
                return client, (self.cfg.judge_model or ep.judge_model)
            if getattr(ep, "generation_base_url", None):
                client = openai.OpenAI(base_url=ep.generation_base_url, api_key=ep.generation_api_key,
                                       timeout=20.0, max_retries=0)
                return client, (self.cfg.judge_model or "q7bc")
            return None, None
        except Exception:
            return None, None

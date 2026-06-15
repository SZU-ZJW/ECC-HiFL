"""Semantic evidence: self-consistency vote + TF-IDF issue↔candidate similarity.

Provides two scalar signals plus an optional real generative-RM score:
- ``vote_score``  -> how mainstream a candidate is across the pool (self-consistency). This is
                     the **offline surrogate for the generative reward model** and also underlies
                     the ``majority`` baseline.
- ``tfidf_score`` -> cosine similarity between the issue text and the candidate text (sklearn
                     TF-IDF; no network).
- ``rm_score``    -> only populated when a live Skywork/SGLang reward endpoint is reachable and
                     ``cfg.use_rm_server`` is set; otherwise ``None`` (offline).
"""

from __future__ import annotations

from collections import Counter
import os
from typing import List, Optional

import requests

from ..candidates import Candidate
from .evidence_card import EvidenceBuilderBase


def _vote_scores(candidates: List[Candidate]) -> List[float]:
    n = len(candidates)
    if n == 0:
        return []
    if candidates[0].level == "file":
        counts = Counter()
        for c in candidates:
            for f in set(c.files):
                counts[f] += 1
        scores = []
        for c in candidates:
            files = list(dict.fromkeys(c.files))
            scores.append(sum(counts[f] for f in files) / (len(files) * n) if files else 0.0)
        return scores
    # function / line: vote on (file, loc) pairs
    counts = Counter()
    for c in candidates:
        for f, locs in c.elements.items():
            for loc in set(locs):
                counts[(f, loc)] += 1
    scores = []
    for c in candidates:
        keys = [(f, loc) for f, locs in c.elements.items() for loc in set(locs)]
        scores.append(sum(counts[k] for k in keys) / (len(keys) * n) if keys else 0.0)
    return scores


def _tfidf_scores(issue: str, candidates: List[Candidate]) -> List[float]:
    texts = [c.raw_text or "" for c in candidates]
    if not issue or not any(t.strip() for t in texts):
        return [0.5 for _ in candidates]  # neutral when nothing to compare
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(max_features=20000)
        mat = vec.fit_transform([issue] + texts)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        return [float(max(0.0, min(1.0, s))) for s in sims]
    except Exception:
        return [0.5 for _ in candidates]


class SemanticEvidenceBuilder(EvidenceBuilderBase):
    name = "semantic"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.use_rm_server = bool(getattr(cfg, "use_rm_server", False)) if cfg else False
        self.skywork_url = getattr(cfg, "skywork_url", None) if cfg else None
        self.skywork_model = getattr(cfg, "skywork_model", "sk8b") if cfg else "sk8b"
        self.skywork_model_path = getattr(cfg, "skywork_model_path", None) if cfg else None
        self.skywork_timeout = float(getattr(cfg, "skywork_timeout", 120) or 120) if cfg else 120.0

    def build_all(self, candidates: List[Candidate], ctx) -> List[dict]:
        votes = _vote_scores(candidates)
        tfidf = _tfidf_scores(ctx.problem_statement or "", candidates)
        rm_scores = self._maybe_rm_scores(candidates, ctx)
        out = []
        for i, c in enumerate(candidates):
            ev = {
                "vote_score": round(votes[i], 4),
                "tfidf_score": round(tfidf[i], 4),
                # the builder's headline "score" used by simple consumers = semantic similarity
                "score": round(tfidf[i], 4),
            }
            if rm_scores is not None:
                ev["rm_score"] = round(float(rm_scores[i]), 4)
            out.append(ev)
        return out

    def _maybe_rm_scores(self, candidates, ctx) -> Optional[List[float]]:
        """Query the live Skywork RM if enabled+reachable, else None (offline)."""
        if not self.use_rm_server:
            return None
        try:
            from ..coreutils import get_endpoints

            ep = get_endpoints()
            url = self.skywork_url or (getattr(ep, "skywork_url", None) if ep is not None else None)
            if not url:
                return None
            model = self.skywork_model or getattr(ep, "skywork_model", None) or "sk8b"
            model_path = self.skywork_model_path or getattr(ep, "skywork_model_path", None)
            prompts = _skywork_prompts(ctx.problem_statement or "", ctx.level, candidates, model_path)
            raw_scores = _request_skywork_scores(url, model, prompts, timeout=self.skywork_timeout)
            return _minmax(raw_scores)
        except Exception:
            return None


def _candidate_answer(candidate: Candidate) -> str:
    if candidate.raw_text and candidate.raw_text.strip():
        return candidate.raw_text.strip()
    if candidate.level == "file":
        body = "\n".join(candidate.files)
    else:
        parts = []
        for file, locs in candidate.elements.items():
            parts.append(file)
            parts.extend(locs)
        body = "\n".join(parts)
    return body.strip() or "No localization candidate was produced."


def _rm_user_prompt(issue: str, level: str) -> str:
    return (
        f"GitHub issue:\n{issue}\n\n"
        f"Evaluate whether the assistant response is a useful and precise {level}-level "
        "localization for fixing this issue. Higher reward should mean the localization is more likely correct."
    )


def _llama31_chat_template(messages: List[dict]) -> str:
    text = "<|begin_of_text|>"
    for msg in messages:
        text += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    return text


def _apply_chat_template(messages_batch: List[List[dict]], model_path: Optional[str]) -> List[str]:
    if model_path and os.path.isdir(model_path):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return tokenizer.apply_chat_template(messages_batch, tokenize=False)
        except Exception:
            pass
    return [_llama31_chat_template(messages) for messages in messages_batch]


def _skywork_prompts(issue: str, level: str, candidates: List[Candidate],
                     model_path: Optional[str]) -> List[str]:
    user_prompt = _rm_user_prompt(issue, level)
    messages_batch = []
    for candidate in candidates:
        messages_batch.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": _candidate_answer(candidate)},
        ])
    return _apply_chat_template(messages_batch, model_path)


def _classify_url(url: str) -> str:
    clean = url.rstrip("/")
    if clean.endswith("/classify"):
        return clean
    return clean + "/classify"


def _request_skywork_scores(url: str, model: str, prompts: List[str], timeout: float) -> List[float]:
    response = requests.post(
        _classify_url(url),
        json={"model": model, "text": prompts},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    scores = []
    for item in data:
        if isinstance(item, dict) and "embedding" in item:
            emb = item["embedding"]
            scores.append(float(emb[0] if isinstance(emb, list) else emb))
        elif isinstance(item, dict) and "score" in item:
            scores.append(float(item["score"]))
        else:
            scores.append(float(item))
    if len(scores) != len(prompts):
        raise ValueError(f"Skywork returned {len(scores)} scores for {len(prompts)} prompts")
    return scores


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / span for v in values]

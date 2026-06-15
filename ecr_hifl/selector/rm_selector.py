"""v3 trained ECR-Reward-Model selector.

Loads a sequence-classification reward model (a fine-tuned scorer over issue + candidate +
evidence text) and picks the argmax-scoring candidate. This is the inference counterpart of
:mod:`ecr_hifl.selector.train_reward_model`. It is **opt-in / GPU-gated**: without a checkpoint
(``cfg.rm_model_path``) or without torch/transformers, it transparently falls back to the rule
selector so the pipeline still runs.
"""

from __future__ import annotations

import functools
from typing import List, Optional

from ..config import ECRConfig
from .base import Selection, SelectionContext, Selector
from .llm_selector import _card_block
from .rule_selector import RuleSelector


def _candidate_text(ctx: SelectionContext, candidate, card) -> str:
    return (f"Issue:\n{ctx.problem_statement[:4000]}\n\n"
            f"Candidate ({ctx.level}):\n{_card_block(candidate.index, candidate, card)}")


@functools.lru_cache(maxsize=2)
def _load_model(model_path: str):
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    model.eval()
    return tok, model


class RMSelector(Selector):
    name = "rm"

    def __init__(self, cfg: ECRConfig):
        self.cfg = cfg
        self.model_path: Optional[str] = getattr(cfg, "rm_model_path", None)
        self._fallback = RuleSelector(cfg.weights)

    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        if not candidates:
            return Selection(index=None, fields=self.empty_fields(ctx.level))
        scores = self._score(candidates, ctx)
        if scores is None:
            sel = self._fallback.predict(candidates, ctx)
            sel.meta = {**sel.meta, "rm": "unavailable -> rule fallback"}
            return sel
        best = max(range(len(scores)), key=lambda i: scores[i])
        return Selection(index=best, fields=candidates[best].result_fields(),
                         scores=[round(float(s), 5) for s in scores], meta={"rm": "ok"})

    def _score(self, candidates, ctx) -> Optional[List[float]]:
        if not self.model_path:
            return None
        try:
            import torch

            tok, model = _load_model(self.model_path)
            cards = ctx.cards or [None] * len(candidates)
            texts = [_candidate_text(ctx, c, cards[i]) for i, c in enumerate(candidates)]
            with torch.no_grad():
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
                logits = model(**enc).logits.squeeze(-1)
            return logits.tolist() if logits.dim() else [float(logits)]
        except Exception:
            return None

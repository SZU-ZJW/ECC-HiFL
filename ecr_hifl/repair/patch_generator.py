"""Patch generation from an ECR-HiFL localization (server-gated).

Given an issue and a selected localization (``found_edit_locs`` / ``found_related_locs`` + files),
build a SEARCH/REPLACE repair prompt over the localized code regions and query the generation
server (``EP.generation``). Returns the model's raw patch text (a list of edits), or ``None`` when
the server is unreachable. Applying/scoring patches is the job of :mod:`.test_runner`.

This connects localization to repair so the §6.3 RQ5 question — "does better localization raise
resolved rate?" — can be answered when servers/Docker are available.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..coreutils import get_endpoints, get_repo_files, transfer_arb_locs_to_locs
from ..io_utils import RepoStructure

_REPAIR_PROMPT = """We are solving this GitHub issue. Below are the localized code regions.
Produce minimal edits as *SEARCH/REPLACE* blocks to fix the issue.

### Issue
{issue}

### Localized code
{code}

Return edits as:
```
<file path>
<<<<<<< SEARCH
<exact lines to find>
=======
<replacement lines>
>>>>>>> REPLACE
```
"""


def _localized_regions(structure: RepoStructure, found_locs: Dict[str, List[str]],
                       context_window: int = 10) -> str:
    """Render the localized line regions of each file (uses the shared core loc->interval util)."""
    file_text = structure.file_text
    blocks = []
    for f, locs in found_locs.items():
        if f not in file_text:
            continue
        try:
            _, intervals = transfer_arb_locs_to_locs(locs, structure.structure, f, context_window, True)
        except Exception:
            intervals = []
        lines = file_text[f]
        for (s, e) in (intervals or []):
            seg = "\n".join(f"{i+1}|{lines[i]}" for i in range(max(0, s), min(len(lines), e)))
            blocks.append(f"### {f} [{s}-{e}]\n{seg}")
    return "\n\n".join(blocks)


def generate_patch(issue: str, structure: RepoStructure, found_locs: Dict[str, List[str]],
                   model: str = "q7bc", max_tokens: int = 1024) -> Optional[str]:
    ep = get_endpoints()
    if ep is None or not getattr(ep, "generation_base_url", None):
        print("[repair] generation server unavailable (EP.generation) — skipping patch generation")
        return None
    code = _localized_regions(structure, found_locs)
    if not code.strip():
        return None
    try:
        import openai

        client = openai.OpenAI(base_url=ep.generation_base_url, api_key=ep.generation_api_key,
                               timeout=30.0, max_retries=0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _REPAIR_PROMPT.format(issue=issue[:8000], code=code[:24000])}],
            max_tokens=max_tokens, temperature=0.0)
        return resp.choices[0].message.content
    except Exception as exc:
        print(f"[repair] patch generation failed: {exc}")
        return None

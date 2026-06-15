"""Best-of-N candidate parsing.

A :class:`Candidate` is one element of the BoN pool at a given level (file / function / line).
We **re-derive** candidates from the pool's ``all_responses`` (file / function) or
``found_edit_locs`` samples or ``edit_loc_traj`` raw responses (line) rather than the
already-selected ``found_files``, so that selection is a genuine Best-of-N. All parsing reuses
the shared-core utilities.

Pool field layout (confirmed against the real Lite pools under ``data``):
- file:      ``row["file_traj"]["all_responses"]``            -> list of ```` ``` ```` fenced file lists
- function:  ``row["related_loc_traj"][0]["all_responses"]``  -> list of fenced file + ``function:/class:`` blocks
- line:      ``row["found_edit_locs"]``                       -> list of dicts ``{file: ["line: a\\nline: b"]}``
             or ``row["edit_loc_traj"]`` raw response fields  -> raw fenced line-location responses
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional

from .coreutils import correct_file_paths, extract_code_blocks, extract_locs_for_files
from .io_utils import RepoStructure

LEVELS = ("file", "function", "line")
_LOC_PREFIXES = ("line:", "function:", "class:", "variable:")
_LOC_LINE_RE = re.compile(r"^(line|function|class|variable|method)\s*:\s*(.+)$", re.IGNORECASE)
_PY_FILE_RE = re.compile(r"([A-Za-z0-9_./-]+\.py)")
_PLACEHOLDER_PATH_RE = re.compile(r"^full_path(\d+)(?:/.*)?$")
_FILE_PLACEHOLDER_RE = re.compile(r"^file(\d+)\.py$")


@dataclass
class Candidate:
    """One BoN candidate, normalized across levels."""

    index: int                       # 1-based position in the pool ("candidate N")
    level: str                       # file | function | line
    raw_text: str                    # raw model response (or synthesized for line level)
    files: List[str] = field(default_factory=list)            # files referenced (repo-corrected)
    elements: Dict[str, List[str]] = field(default_factory=dict)  # {file: ["function: x", "line: 5", ...]}

    # ---- result-format accessors (match the upstream loc_outputs.jsonl shape for eval) ----
    def result_fields(self) -> dict:
        if self.level == "file":
            return {"found_files": list(self.files)}
        if self.level == "function":
            return {
                "found_files": list(self.files),
                "found_related_locs": {f: ["\n".join(locs)] for f, locs in self.elements.items()},
            }
        if self.level == "line":
            return {
                "found_files": list(self.files),
                "found_edit_locs": {f: ["\n".join(locs)] for f, locs in self.elements.items()},
            }
        raise ValueError(f"unknown level {self.level!r}")

    @property
    def is_empty(self) -> bool:
        if self.level == "file":
            return len(self.files) == 0
        return not any(locs for locs in self.elements.values())


# --------------------------------------------------------------------------- parsers

def _first_block(raw: str) -> str:
    blocks = extract_code_blocks(raw)
    return blocks[0] if blocks else (raw or "")


def _file_names_in(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().endswith(".py")]


def _structure_paths(structure: Optional[RepoStructure]) -> List[str]:
    if structure is None:
        return []
    return list(getattr(structure, "file_paths", []) or [])


def _correct(paths: List[str], structure: Optional[RepoStructure]) -> List[str]:
    if structure is None:
        return paths
    # fast path: candidate lines are already exact repo paths (the common case) -> avoid the
    # O(model_files x repo_files) correct_file_paths scan. Falls back to it for fuzzy matches.
    fps = structure.file_path_set
    if paths and all(p in fps for p in paths):
        corrected = paths
    else:
        corrected = correct_file_paths(paths, structure.files)
    # correct_file_paths drops unknown files; preserve order, dedupe
    seen, out = set(), []
    for p in corrected:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _extract_file_token(line: str) -> Optional[str]:
    text = (line or "").strip().strip("`").strip()
    match = _PY_FILE_RE.search(text)
    return match.group(1) if match else None


def _normalize_loc_line(line: str) -> Optional[str]:
    match = _LOC_LINE_RE.match((line or "").strip().strip("`").strip())
    if not match:
        return None
    kind = match.group(1).lower()
    value = match.group(2).strip().strip("`").strip()
    if not value:
        return None
    if kind == "method":
        kind = "function"
    return f"{kind}: {value}"


def _resolve_model_path(path: str, structure: Optional[RepoStructure],
                        default_files: Optional[List[str]] = None) -> str:
    """Resolve model-emitted file names, including prompt placeholders such as full_path1/x.py."""
    clean = (path or "").strip().strip("`").lstrip("./")
    default_files = [p for p in (default_files or []) if p]

    placeholder = _PLACEHOLDER_PATH_RE.match(clean)
    if placeholder and default_files:
        idx = int(placeholder.group(1)) - 1
        if 0 <= idx < len(default_files):
            return default_files[idx]

    file_placeholder = _FILE_PLACEHOLDER_RE.match(clean)
    if file_placeholder and default_files:
        idx = int(file_placeholder.group(1)) - 1
        if 0 <= idx < len(default_files):
            return default_files[idx]

    if clean.startswith("full_path") and default_files:
        return default_files[0]

    def unique_match(paths: List[str], token: str) -> Optional[str]:
        matches = [p for p in paths if p == token or p.endswith("/" + token)]
        if len(matches) == 1:
            return matches[0]
        base = token.rsplit("/", 1)[-1]
        matches = [p for p in paths if p == base or p.endswith("/" + base)]
        if len(matches) == 1:
            return matches[0]
        return None

    match = unique_match(default_files, clean)
    if match:
        return match

    repo_paths = _structure_paths(structure)
    match = unique_match(repo_paths, clean)
    if match:
        return match

    if structure is not None:
        corrected = correct_file_paths([clean], structure.files)
        if corrected:
            return corrected[0]
    return clean


def parse_file_candidates(all_responses: List[str], structure: Optional[RepoStructure]) -> List[Candidate]:
    cands = []
    for i, raw in enumerate(all_responses):
        block = _first_block(raw)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        files = _correct(lines, structure) if structure is not None else [l for l in lines if l.endswith(".py")]
        cands.append(Candidate(index=i + 1, level="file", raw_text=raw, files=files))
    return cands


def parse_function_candidates(all_responses: List[str], structure: Optional[RepoStructure]) -> List[Candidate]:
    cands = []
    for i, raw in enumerate(all_responses):
        text = _first_block(raw)
        fnames = _file_names_in(text)
        locs = extract_locs_for_files([text], fnames) if fnames else {}
        # remap keys to repo-correct paths (basename eval is forgiving, but line spans need real paths)
        elements: Dict[str, List[str]] = {}
        for fname, vals in locs.items():
            key = fname
            if structure is not None:
                cf = correct_file_paths([fname], structure.files)
                key = cf[0] if cf else fname
            joined = vals[0] if vals else ""
            lines = [ln for ln in joined.splitlines() if ln.strip().startswith(_LOC_PREFIXES)]
            if lines:
                elements.setdefault(key, []).extend(lines)
        cands.append(
            Candidate(index=i + 1, level="function", raw_text=raw,
                      files=list(elements.keys()), elements=elements)
        )
    return cands


def _line_response_to_sample(raw: str, structure: Optional[RepoStructure],
                             default_files: Optional[List[str]] = None) -> Dict[str, List[str]]:
    blocks = extract_code_blocks(raw) or [raw or ""]
    elements: Dict[str, List[str]] = {}
    current_file: Optional[str] = None

    for block in blocks:
        for line in block.splitlines():
            file_token = _extract_file_token(line)
            if file_token:
                current_file = _resolve_model_path(file_token, structure, default_files)
                continue
            loc_line = _normalize_loc_line(line)
            if not loc_line:
                continue
            target_file = current_file
            if target_file is None and default_files and len(default_files) == 1:
                target_file = default_files[0]
            if target_file:
                elements.setdefault(target_file, []).append(loc_line)

    return elements


def parse_line_candidates(found_edit_locs: List, structure: Optional[RepoStructure],
                          default_files: Optional[List[str]] = None) -> List[Candidate]:
    cands = []
    for i, sample in enumerate(found_edit_locs):
        elements: Dict[str, List[str]] = {}
        raw_parts: List[str] = []
        if isinstance(sample, dict):
            for fname, vals in sample.items():
                joined = "\n".join(v for v in vals if v) if isinstance(vals, list) else str(vals)
                lines = [loc for loc in (_normalize_loc_line(ln) for ln in joined.splitlines()) if loc]
                if lines:
                    key = _resolve_model_path(fname, structure, default_files)
                    elements.setdefault(key, []).extend(lines)
                    raw_parts.append(fname + "\n" + "\n".join(lines))
        elif isinstance(sample, str):
            elements = _line_response_to_sample(sample, structure, default_files)
            raw_parts = [sample]
        cands.append(
            Candidate(index=i + 1, level="line", raw_text="\n\n".join(raw_parts),
                      files=list(elements.keys()), elements=elements)
        )
    return cands


def _flatten_strings(value) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []


def _line_raw_responses(row: dict) -> List[str]:
    elt = row.get("edit_loc_traj") or {}
    for key in ("all_responses", "response"):
        raw = _flatten_strings(elt.get(key))
        if raw:
            return raw
    for artifact in row.get("additional_artifact_loc_edit_location") or []:
        if isinstance(artifact, dict):
            raw = _flatten_strings(artifact.get("raw_output_loc"))
            if raw:
                return raw
    return []


def _line_sample_has_locs(sample) -> bool:
    if isinstance(sample, str):
        return bool(_line_response_to_sample(sample, structure=None))
    if isinstance(sample, dict):
        for vals in sample.values():
            text = "\n".join(v for v in vals if v) if isinstance(vals, list) else str(vals)
            if any(_normalize_loc_line(ln) for ln in text.splitlines()):
                return True
    return False


def extract_all_responses(row: dict, level: str) -> List:
    """Pull the raw BoN candidate list out of a pool row for the given level."""
    if level == "file":
        return list((row.get("file_traj") or {}).get("all_responses", []) or [])
    if level == "function":
        rlt = row.get("related_loc_traj") or []
        for entry in rlt:
            if isinstance(entry, dict) and entry.get("all_responses"):
                return list(entry["all_responses"])
        return []
    if level == "line":
        fel = row.get("found_edit_locs")
        raw = _line_raw_responses(row)
        if isinstance(fel, list):
            # Fresh line-generation pools may keep a 30-sample parsed list where some entries are
            # empty because the model used prompt placeholders like ``full_path1/foo.py``. In that
            # case the raw responses are more faithful and can be reparsed with row-level context.
            if raw and any(not _line_sample_has_locs(sample) for sample in fel):
                return raw
            return list(fel)
        # Some upstream line-localization outputs store only the selected parsed dict in
        # ``found_edit_locs`` but keep the full Best-of-N raw samples in ``edit_loc_traj``.
        if raw:
            return raw
        if isinstance(fel, dict):  # single-sample pool without raw BoN responses
            return [fel]
        return []
    raise ValueError(f"unknown level {level!r}")


def parse_candidates(row: dict, level: str, structure: Optional[RepoStructure],
                     sample: Optional[int] = None) -> List[Candidate]:
    """Parse a pool row into a list of candidates, optionally truncated to the first ``sample`` (the N in Best-of-N)."""
    raw = extract_all_responses(row, level)
    if sample is not None and sample > 0:
        raw = raw[:sample]
    if level == "file":
        return parse_file_candidates(raw, structure)
    if level == "function":
        return parse_function_candidates(raw, structure)
    if level == "line":
        return parse_line_candidates(raw, structure, default_files=row.get("found_files") or [])
    raise ValueError(f"unknown level {level!r}")

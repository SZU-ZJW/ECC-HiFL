"""IO helpers: jsonl, repo-structure loading, and SWE-bench problem statements.

Everything here works offline: repo structures are read from the local structure-cache
directory (``project_file_loc``) and problem statements from the HF datasets cache.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .coreutils import get_full_file_paths_and_classes_and_functions


# --------------------------------------------------------------------------- jsonl

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str) -> List[dict]:
    return list(iter_jsonl(path))


def load_jsonl_by_id(path: str, id_key: str = "instance_id") -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in iter_jsonl(path):
        out[row[id_key]] = row
    return out


def save_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------- repo structures

class RepoStructure:
    """A parsed repo structure plus cached views built by the shared core util.

    ``files`` is a list of ``(path, text_lines)`` tuples; ``classes`` / ``functions`` are
    flat lists with ``file`` keys. We cache a few derived maps used by the evidence builders.
    """

    def __init__(self, instance_id: str, repo: str, base_commit: str, structure: dict):
        self.instance_id = instance_id
        self.repo = repo
        self.base_commit = base_commit
        self.structure = structure
        files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
        # files may contain plain-path entries (non-python); keep only (path, text) tuples
        self.files = files
        self.classes = classes
        self.functions = functions

    @functools.cached_property
    def file_paths(self) -> List[str]:
        out = []
        for entry in self.files:
            out.append(entry[0] if isinstance(entry, tuple) else entry)
        return out

    @functools.cached_property
    def file_path_set(self) -> set:
        return set(self.file_paths)

    @functools.cached_property
    def file_text(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for entry in self.files:
            if isinstance(entry, tuple) and len(entry) == 2:
                out[entry[0]] = entry[1]
        return out

    @functools.cached_property
    def symbols_by_file(self) -> Dict[str, List[dict]]:
        """{file: [{name, kind, start_line, end_line}]} for classes + top-level functions + methods."""
        out: Dict[str, List[dict]] = {}
        for c in self.classes:
            out.setdefault(c["file"], []).append(
                {"name": c["name"], "kind": "class",
                 "start_line": c.get("start_line"), "end_line": c.get("end_line")}
            )
            for m in c.get("methods", []):
                out.setdefault(c["file"], []).append(
                    {"name": f"{c['name']}.{m['name']}", "kind": "method",
                     "start_line": m.get("start_line"), "end_line": m.get("end_line")}
                )
        for fn in self.functions:
            out.setdefault(fn["file"], []).append(
                {"name": fn["name"], "kind": "function",
                 "start_line": fn.get("start_line"), "end_line": fn.get("end_line")}
            )
        return out


@functools.lru_cache(maxsize=512)
def load_structure(instance_id: str, project_file_loc: str) -> Optional[RepoStructure]:
    """Load a repo structure from the offline cache ``<project_file_loc>/<instance_id>.json``."""
    path = os.path.join(project_file_loc, f"{instance_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return RepoStructure(
        instance_id=instance_id,
        repo=d.get("repo", ""),
        base_commit=d.get("base_commit", ""),
        structure=d["structure"],
    )


# ----------------------------------------------------------- problem statements

@functools.lru_cache(maxsize=4)
def load_problem_statements(dataset: str, split: str = "test") -> Dict[str, str]:
    """{instance_id: problem_statement} from the (HF-cached) SWE-bench dataset, offline."""
    from datasets import load_dataset

    data = load_dataset(dataset, split=split)
    return {row["instance_id"]: row["problem_statement"] for row in data}


@functools.lru_cache(maxsize=4)
def load_bench_meta(dataset: str, split: str = "test") -> Dict[str, dict]:
    """{instance_id: {repo, base_commit, problem_statement}} for repair / git evidence."""
    from datasets import load_dataset

    data = load_dataset(dataset, split=split)
    return {
        row["instance_id"]: {
            "repo": row.get("repo", ""),
            "base_commit": row.get("base_commit", ""),
            "problem_statement": row.get("problem_statement", ""),
        }
        for row in data
    }

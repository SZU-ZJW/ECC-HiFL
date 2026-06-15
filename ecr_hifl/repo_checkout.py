"""Ephemeral per-instance repo checkout for History (and future repair) evidence.

Per the project's "copy a repo when needed, switch it to the instance's ``base_commit``, delete
after use" model (so we never persist 500 working trees): for each instance we ``git clone
--local`` the shared SWE-bench clone found under ``repo_source`` into a temp dir on the **same
filesystem** — so ``.git`` objects are *hardlinked* (fast, ~zero extra disk, the source clone is
never touched) — then hard-checkout ``base_commit`` (detached), read what we need, and ``rm -rf``
the temp dir.

The only History op is ``git log --pretty=%s -- <file>``, whose result depends only on
``(instance_id, file)``. So we **memoize per instance** (``{instance_id: {file: bugfix_count}}``):
the ablation grid (cards rebuilt per ``sample_num``) and a multi-threaded run thus clone each
instance at most once. Instances whose checkout fails (commit missing from the clone, repo not
present) are marked unavailable so we don't re-clone them every call.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from typing import Dict, List, Optional

# Keep in sync with 
_BUGFIX_KEYWORDS = ("fix", "bug", "error", "issue", "fail", "regression", "crash", "exception")
_GIT_TIMEOUT = 600

_memo: Dict[str, Dict[str, int]] = {}     # instance_id -> {file: bugfix_count} (successful lookups)
_unavailable: set = set()                 # instance_ids whose ephemeral checkout failed -> don't retry
_lock = threading.Lock()


def resolve_source_repo(repo_source: Optional[str], repo: str, instance_id: str) -> Optional[str]:
    """Locate the shared clone dir for an instance under ``repo_source`` (must contain ``.git``).

    SWE-bench clones are named ``{owner}__{name}`` (e.g. ``django__django``); we try the repo
    field (``django/django`` -> ``django__django``), its last path component, and the
    instance-id prefix (``django__django-13343`` -> ``django__django``).
    """
    if not repo_source:
        return None
    cands: List[str] = []
    if repo:
        cands += [repo.replace("/", "__"), repo.split("/")[-1]]
    if instance_id and "-" in instance_id:
        cands.append(instance_id.rsplit("-", 1)[0])
    for c in cands:
        if not c:
            continue
        p = os.path.join(repo_source, c)
        if os.path.isdir(os.path.join(p, ".git")):
            return p
    return None


def _git(repo_path: str, *args: str, timeout: int = 60) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, *args],
            text=True, errors="ignore", stderr=subprocess.DEVNULL, timeout=timeout,
        )
    except Exception:
        return ""


def bugfix_count(repo_path: str, file_path: str) -> int:
    """#commits touching ``file_path`` (reachable from HEAD) whose subject hits a bugfix keyword."""
    logs = _git(repo_path, "log", "--pretty=%s", "--", file_path)
    return sum(any(k in line.lower() for k in _BUGFIX_KEYWORDS) for line in logs.splitlines())


def _ephemeral_counts(src: str, base_commit: str, files: List[str],
                      tmp_parent: Optional[str]) -> Dict[str, int]:
    """Clone ``src`` (hardlinked, no working tree) -> checkout ``base_commit`` -> git-log each
    file -> delete. Returns ``{file: count}`` on success, ``{}`` on any failure."""
    # Default the temp parent to a sibling of the source clone so the clone is on the same
    # filesystem (required for git's hardlink fast-path); fall back to the system temp dir.
    parent = tmp_parent or os.path.join(os.path.dirname(os.path.normpath(src)), ".ecr_checkout_tmp")
    try:
        os.makedirs(parent, exist_ok=True)
    except Exception:
        parent = None
    tmp = tempfile.mkdtemp(prefix="ecr_co_", dir=parent)
    dst = os.path.join(tmp, "repo")
    try:
        clone = subprocess.run(
            ["git", "clone", "--local", "--no-checkout", "--quiet", src, dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=_GIT_TIMEOUT,
        )
        if clone.returncode != 0:
            return {}
        co = subprocess.run(
            ["git", "-C", dst, "checkout", "-f", base_commit],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=_GIT_TIMEOUT,
        )
        if co.returncode != 0:  # base_commit not present in the clone -> no usable history
            return {}
        return {f: bugfix_count(dst, f) for f in files}
    except Exception:
        return {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def instance_bugfix_counts(repo_source: Optional[str], repo: str, instance_id: str,
                           base_commit: str, files: List[str],
                           tmp_parent: Optional[str] = None) -> Dict[str, int]:
    """Memoized ``{file: bugfix_commit_count}`` for an instance via copy -> checkout(base_commit)
    -> delete. Clones the instance at most once per process; returns ``{}`` if unavailable
    (no ``repo_source`` match, no ``base_commit``, or the checkout failed)."""
    files = [f for f in dict.fromkeys(files) if f]
    if not files:
        return {}
    with _lock:
        if instance_id in _unavailable:
            return {}
        cached = _memo.get(instance_id)
        if cached is not None and all(f in cached for f in files):
            return {f: cached[f] for f in files}
    if not base_commit:
        with _lock:
            _unavailable.add(instance_id)
        return {}
    src = resolve_source_repo(repo_source, repo, instance_id)
    if src is None:
        with _lock:
            _unavailable.add(instance_id)
        return {}
    counts = _ephemeral_counts(src, base_commit, files, tmp_parent)
    if not counts:
        with _lock:
            _unavailable.add(instance_id)
        return {}
    with _lock:
        slot = _memo.setdefault(instance_id, {})
        slot.update(counts)
        return {f: slot.get(f, 0) for f in files}

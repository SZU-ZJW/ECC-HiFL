"""Thin re-export of the reused HiLoRM / Agentless core utilities.

ECR-HiFL deliberately does NOT re-implement AST parsing or candidate parsing: the repo
structure JSONs are already AST-parsed (``classes`` / ``functions`` / ``text`` per file) and
the shared core already has battle-tested parsers. We import them from a single place so the
dependency on ``core/`` being on PYTHONPATH has one clear failure point.

Reused (see ``core/agentless/util/{preprocess_data,postprocess_data,compress_file}.py``):
- ``get_full_file_paths_and_classes_and_functions(structure)`` -> ``(files, classes, functions)``
    where ``files`` is a list of ``(path, text_lines)`` tuples, ``classes`` is a list of
    ``{file,name,start_line,end_line,methods:[{name,start_line,end_line}]}`` and ``functions``
    is a list of ``{file,name,start_line,end_line,text}``.
- ``correct_file_paths(model_files, files)`` -> repo-correct file paths.
- ``get_repo_files(structure, paths)`` -> ``{path: code_str}``.
- ``transfer_arb_locs_to_locs(locs, structure, file, ctx, loc_interval)`` -> ``(line_loc, intervals)``
    converts ``function:/class:/line:/variable:`` strings to concrete ``(start,end)`` line spans.
- ``extract_code_blocks(text)`` -> list of ```` ``` ```` fenced blocks.
- ``extract_locs_for_files(locs, file_names, keep_old_order=False)`` -> ``{file: ["fn:..\\nclass:.."]}``.
- ``get_skeleton(code, ...)`` -> compressed file skeleton (used by the structural summary fallback).
"""

try:  # pragma: no cover - exercised indirectly
    from agentless.util.preprocess_data import (
        correct_file_paths,
        get_full_file_paths_and_classes_and_functions,
        get_repo_files,
        transfer_arb_locs_to_locs,
    )
    from agentless.util.postprocess_data import (
        extract_code_blocks,
        extract_locs_for_files,
    )
    from agentless.util.compress_file import get_skeleton
except ModuleNotFoundError as exc:  # pragma: no cover - configuration error path
    raise ModuleNotFoundError(
        "ECR-HiFL needs the HiLoRM shared core on PYTHONPATH. "
        "Run via `./run_ecr.sh ...` (sets PYTHONPATH=<repo>/core:<repo>), or "
        "`export PYTHONPATH=<repo>/core:$PYTHONPATH` before importing ecr_hifl. "
        f"Original import error: {exc}"
    ) from exc


def get_endpoints():
    """Return the HiLoRM ``EP`` endpoints singleton (model servers), or ``None`` if absent.

    ``EP`` is built from ``config.yaml`` at import time keyed by ``$HILORM_STAGE``; missing
    endpoints resolve to ``None`` so this never raises when servers are merely offline.
    """
    try:
        from agentless.util.endpoints import EP

        return EP
    except Exception:  # pragma: no cover - defensive
        return None


__all__ = [
    "correct_file_paths",
    "get_full_file_paths_and_classes_and_functions",
    "get_repo_files",
    "transfer_arb_locs_to_locs",
    "extract_code_blocks",
    "extract_locs_for_files",
    "get_skeleton",
    "get_endpoints",
]

"""Graph evidence (offline): import dependency graph + containment + same-file neighbors.

Built entirely from the repo-structure JSON (file text + parsed symbols) — no checkout needed.
Implements the three lightweight relations  calls for first (file import graph,
class/function containment, same-file neighbor) and exposes a normalized centrality score.

The intuition: a file that the rest of the repo depends on (high import degree) is a more
plausible edit site than an isolated one. The per-candidate ``score`` is the candidate's mean
file degree, min-max normalized across the instance's candidate set (instance-relative ranking).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..candidates import Candidate
from ..io_utils import RepoStructure
from .evidence_card import EvidenceBuilderBase

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+(?:\s*,\s*[.\w]+)*))")


def _module_index(structure: RepoStructure) -> Dict[str, str]:
    """{dotted_module: file_path} for repo .py files (also registers src/-stripped modules)."""
    idx: Dict[str, str] = {}
    for path in structure.file_paths:
        if not path.endswith(".py"):
            continue
        mod = path[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        idx.setdefault(mod, path)
        # src/ or lib/ layout: also expose the module without the source root
        for root in ("src.", "lib."):
            if mod.startswith(root):
                idx.setdefault(mod[len(root):], path)
    return idx


def _resolve(module: str, mod2file: Dict[str, str]) -> Optional[str]:
    """Resolve an imported dotted module to a repo file by longest-prefix match."""
    parts = module.strip(".").split(".")
    for k in range(len(parts), 0, -1):
        cand = ".".join(parts[:k])
        if cand in mod2file:
            return mod2file[cand]
    return None


def build_import_graph(structure: RepoStructure):
    """networkx.DiGraph over repo files; edge A->B iff A imports a module resolving to file B."""
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(structure.file_paths)
    mod2file = _module_index(structure)
    for path, lines in structure.file_text.items():
        for line in lines:
            if "import" not in line:
                continue
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            mods = []
            if m.group(1):
                mods.append(m.group(1))
            if m.group(2):
                mods.extend(p.strip() for p in m.group(2).split(","))
            for mod in mods:
                tgt = _resolve(mod, mod2file)
                if tgt and tgt != path:
                    g.add_edge(path, tgt)
    return g


class GraphEvidenceBuilder(EvidenceBuilderBase):
    name = "graph"

    def __init__(self, cfg=None):
        self.cfg = cfg

    def build_all(self, candidates: List[Candidate], ctx) -> List[dict]:
        structure: Optional[RepoStructure] = ctx.structure
        if structure is None or not candidates:
            return [{"score": 0.0} for _ in candidates]
        try:
            g = build_import_graph(structure)
        except Exception as exc:
            return [{"score": 0.0, "error": f"graph: {exc}"} for _ in candidates]

        degree = {n: g.in_degree(n) + g.out_degree(n) for n in g.nodes}
        symbols_by_file = structure.symbols_by_file

        raw_scores: List[float] = []
        rich: List[dict] = []
        for c in candidates:
            files = list(dict.fromkeys(c.files))
            degs = [degree.get(f, 0) for f in files]
            mean_deg = sum(degs) / len(degs) if degs else 0.0
            raw_scores.append(mean_deg)
            imports, imported_by, neighbors = [], [], []
            for f in files[:5]:
                imports += [t for _, t in g.out_edges(f)][:8]
                imported_by += [s for s, _ in g.in_edges(f)][:8]
                neighbors += [s["name"] for s in symbols_by_file.get(f, [])][:12]
            rich.append({
                "graph_degree": round(mean_deg, 3),
                "imports": sorted(set(imports))[:10],
                "imported_by": sorted(set(imported_by))[:10],
                "neighbor_symbols": sorted(set(neighbors))[:15],
            })
        # min-max normalize degree across this instance's candidates -> score in [0,1]
        lo, hi = (min(raw_scores), max(raw_scores)) if raw_scores else (0.0, 0.0)
        span = (hi - lo) or 1.0
        for r, raw in zip(rich, raw_scores):
            r["score"] = round((raw - lo) / span, 4)
        return rich

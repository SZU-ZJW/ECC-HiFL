# ECC-HiFL

ECC-HiFL is an experimental fault-localization framework for SWE-bench style tasks. It combines the staged HiLoRM localization pipeline with an evidence-aware candidate selector that can run offline or with reward-model and LLM endpoints.

The repository is organized for reproducible research: shared Agentless-derived localization code is kept in one core tree, stage-specific behavior is isolated under `stages/`, and ECR-HiFL selection logic is packaged under `ecr_hifl/`.

## Repository Layout

```text
.
├── core/                 # Shared Agentless localization and repair utilities
├── stages/               # Stage-specific localization overrides and batch scripts
├── ecr_hifl/             # Evidence builders, selectors, experiments, and tests
├── configs/              # ECR-HiFL experiment configs
├── eval/                 # Localization evaluation scripts
├── swebench_eval/        # Copied SWE-bench harness utilities
├── config.yaml           # Runtime endpoint template
├── model.yaml            # Example model endpoint registry
├── run.sh                # Stage pipeline entry point
└── run_ecr.sh            # ECR-HiFL entry point
```

The stage pipeline includes:

- `step1_file`: suspicious file localization
- `step2_irrelevant`: irrelevant-folder filtering
- `step3_retrieve`: embedding retrieval and merge
- `step5_element`: class/function-level localization
- `step6_line`: fine-grained edit-line localization

## Installation

Use Python 3.11 if possible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The SWE-bench resolved-rate harness may require a separate environment with Docker access, depending on your setup.

## Configuration

`config.yaml` is a safe public template. Replace its placeholder endpoints before running live model-backed experiments, or point to another file:

```bash
export HILORM_CONFIG=/path/to/private-config.yaml
export HILORM_PYTHON=/path/to/python
```

ECR-HiFL experiment configs live in `configs/ecr_*.yaml`. Their paths are placeholders under `data/`; replace them with your local candidate pools, gold files, repository-structure cache, and optional checked-out SWE-bench repositories.

Do not commit real API keys, private endpoints, model paths, candidate caches, or large result files.

## Common Commands

Inspect the stage runtime environment:

```bash
./run.sh step1_file env
```

Run a stage CLI help check:

```bash
./run.sh step1_file localize --help
./run.sh step3_retrieve batch
```

Run ECR-HiFL experiments:

```bash
./run_ecr.sh baseline --config configs/ecr_lite.yaml --level file
./run_ecr.sh select --config configs/ecr_lite.yaml --level file --selector rule --eval
./run_ecr.sh ablation --config configs/ecr_ablation.yaml --limit 50
```

Run the offline smoke entry point:

```bash
./run_ecr.sh smoke
```

The smoke test expects the configured local data files to exist. If you have not prepared candidate pools and gold files, use CLI help/import checks first.

## Data Expectations

The code expects SWE-bench style JSONL inputs:

- Candidate pools: `loc_outputs.jsonl` from file, function, or line localization stages.
- Gold files: file-level modified files, element-level annotations, and line-level annotations.
- Repository structures: precomputed JSON files consumed by `core/get_repo_structure/` and evidence builders.
- Optional repositories: checked-out SWE-bench repositories for history and repair evidence.

Large data should stay outside Git or under ignored local directories such as `data/`, `ecr_hifl/cache/`, and `ecr_hifl/results/`.

## Development Notes

The `agentless` package is assembled through `PYTHONPATH`: `run.sh` places the selected `stages/<stage>` directory before `core/`. Use the runner instead of invoking stage files directly.

Keep shared logic in `core/` or `ecr_hifl/`. Keep stage-only changes under the corresponding `stages/<stage>/` directory. Tests should be named `test_*.py` and should avoid external model servers unless they explicitly cover live integrations.

## Acknowledgements

This repository builds on Agentless-style localization and the SWE-bench evaluation ecosystem. The copied `swebench_eval/` directory is included for local evaluation convenience.

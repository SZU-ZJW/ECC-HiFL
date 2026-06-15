"""ECR-HiFL: Evidence-Calibrated Reward Selection for repository-level fault localization.

This package implements an evidence-calibration layer that sits between HiLoRM's
pre-generated Best-of-N candidate pool and the reward-model selection. Each candidate is
annotated with an *evidence card* (graph / summary / history / verification signals), and
a selector (rule / LLM-judge / trained RM) turns those signals into a calibrated score to
pick the best candidate at file / function / line level.

It reuses the shared HiLoRM core (``core/agentless/util/*``) for parsing repo structures and
localization candidates, and reuses ``agentless.util.endpoints.EP`` for model-server endpoints.
Run everything through ``./run_ecr.sh`` (sets PYTHONPATH / PROJECT_FILE_LOC / HILORM_*).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"

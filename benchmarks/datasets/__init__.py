"""Phase 15: the GT-ladder dataset adapters.

Each adapter normalizes its source into the EXISTING on-disk sequence
contract the harness consumes (frames/frame_%06d.png + trajectory.txt
in TUM Twc format + calibration.txt "fx fy cx cy [k1]"), so run_slam
and metrics are untouched. Downloads are cached under data/ (the
data/tum convention the battery already uses).

The ladder is DIAGNOSTIC tier: DEFAULT_SEQUENCES and baselines.json
(the frozen adoption battery) are untouched. Sequences are for local
research evaluation only; licenses recorded per entry in registry.py.
"""
from benchmarks.datasets.registry import GT_SEQUENCES, prepare, describe

__all__ = ["GT_SEQUENCES", "prepare", "describe"]

"""Phase 6c experimental selectors: benchmark-only, NOT production defaults.

Mapped geometry replaces the raw-distance trigger only with sufficient
fresh, distributed, pose-refined evidence. Otherwise retain the complete
legacy decision as a map-renewal path. This is conditional scale
invariance, not a claim that the fallback is scale invariant.

The production caller's forced maximum gap is unchanged. No selector
here changes matching, triangulation partners, registries or optimizers.
"""
from dataclasses import asdict, dataclass
import math
from unittest.mock import patch

import slam.core.tracker as tracker_mod
from slam.utils import constants as const


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    family: str
    threshold: float = 0.0
    tracking_reference: str = "last_reference"
    minimum_gap: int = 3
    minimum_shared: int = 10
    minimum_grid_cells: int = 4
    minimum_features: int = 20
    minimum_frame_matches: int = 15
    renewal_overlap: float = 0.20
    renewal_grid_fraction: float = 0.50
    renewal_persistence: int = 2
    renewal_parallax_floor: float = 0.65


# Declared before intervention runs. The thresholds bracket the observed
# control signal ranges, NOT end-to-end winners. No per-sequence choices.
SELECTORS = {config.name: config for config in (
    SelectorConfig("matched_fraction", "matched_fraction"),
    SelectorConfig("parallax_080", "parallax", 0.8),
    SelectorConfig("parallax_120", "parallax", 1.2),
    SelectorConfig("depth_025", "depth", 0.025),
    SelectorConfig("depth_040", "depth", 0.040),
    SelectorConfig("hybrid_last", "hybrid", 1.2),
    SelectorConfig("hybrid_local", "hybrid", 1.2, "best_local_reference"),
)}


@dataclass
class SelectorState:
    reference_token: object = None
    previous_gap: int = -1
    renewal_streak: int = 0

    def advance(self, reference_token, gap):
        if self.reference_token != reference_token or gap < self.previous_gap:
            self.renewal_streak = 0
        self.reference_token = reference_token
        self.previous_gap = gap


@dataclass(frozen=True)
class SelectorDecision:
    selected: bool
    reason: str
    evidence: dict


def _finite(value):
    return value is not None and math.isfinite(value)


def evaluate_selector(config, state, signals, gap, reference_token,
                      legacy_selected, pose_accepted):
    """Pure input reads; mutate only this experiment's hysteresis state.

    Support is fixed-map pixel CONSISTENCY, not proven correspondences or
    calibrated pose covariance. Reference parallax is mapped-point
    parallax, not the distribution of potential new landmarks.
    """
    state.advance(reference_token, gap)
    rotation = (signals["rotation_deg"] >= const.KEYFRAME_MINIMUM_ROTATION_DEG
                and gap >= const.KEYFRAME_ROTATION_MIN_FRAME_GAP)
    if config.family == "matched_fraction":
        fraction = signals["matched_unmapped_fraction"]
        fraction = fraction if fraction is not None else 0.0
        quality = signals["legacy_quality"] >= const.MINIMUM_KEYFRAME_QUALITY_THRESHOLD
        viewpoint = signals["baseline"] >= const.MINIMUM_BASELINE_THRESHOLD or rotation
        selected = viewpoint and (fraction >= const.MINIMUM_NEW_POINTS_RATIO or quality)
        return SelectorDecision(bool(selected), "matched_fraction" if selected else "matched_fraction_wait",
                                {"matched_unmapped_fraction": fraction, "quality": quality})

    image_usable = (signals["features"] >= config.minimum_features
                    and signals["previous_frame_matches"] >= config.minimum_frame_matches
                    and _finite(signals["baseline"]) and _finite(signals["rotation_deg"]))
    if not image_usable:
        state.renewal_streak = 0
        return SelectorDecision(False, "unusable_image_evidence", {"image_usable": False})

    last = signals["last_reference"]
    parallax = last["parallax_deg"]["median"]
    depth_ratio = last["baseline_over_depth"]
    cells = signals["support_grid_at_3px"]["occupied_cells"]
    geometry_available = (pose_accepted is True
                          and last["parallax_deg"]["finite_count"] >= config.minimum_shared
                          and cells >= config.minimum_grid_cells
                          and _finite(parallax)
                          and (config.family != "depth" or _finite(depth_ratio)))
    evidence = {"image_usable": True, "geometry_available": geometry_available,
                "pose_accepted": pose_accepted, "shared_last": last["parallax_deg"]["finite_count"],
                "grid_cells": cells, "parallax_deg": parallax, "baseline_over_depth": depth_ratio}
    if not geometry_available:
        state.renewal_streak = 0
        # Not a low-support rescue or a global usability veto: preserve
        # the existing decision, including bootstrap and rotation cadence.
        return SelectorDecision(bool(legacy_selected),
                                "legacy_map_renewal" if legacy_selected else "legacy_wait", evidence)

    tracking = signals[config.tracking_reference]
    overlap = tracking["visible_match_fraction"]
    overlap_low = (_finite(overlap) and tracking["shared_consistent"] >= config.minimum_shared
                   and overlap < config.renewal_overlap)
    coverage_low = signals["support_grid_at_3px"]["occupied_fraction"] < config.renewal_grid_fraction
    if config.family == "hybrid":
        state.renewal_streak = state.renewal_streak + 1 if overlap_low or coverage_low else 0
    evidence.update(renewal_streak=state.renewal_streak, tracking_overlap=overlap,
                    overlap_low=overlap_low, coverage_low=coverage_low)
    if rotation:
        return SelectorDecision(True, "view_rotation", evidence)
    if gap < config.minimum_gap:
        return SelectorDecision(False, "minimum_geometry_gap", evidence)
    value = depth_ratio if config.family == "depth" else parallax
    if value >= config.threshold:
        return SelectorDecision(True, "geometry_growth", evidence)
    if (config.family == "hybrid" and state.renewal_streak >= config.renewal_persistence
            and parallax >= config.renewal_parallax_floor):
        reason = "reference_renewal" if overlap_low else "coverage_renewal"
        return SelectorDecision(True, reason, evidence)
    return SelectorDecision(False, "geometry_wait", evidence)


class KeyframePolicyExperiment:
    """Nest INSIDE KeyframeAudit; reuse its single decision-time snapshot.

    Shadow evaluates every candidate on the CONTROL map and returns the
    original result. It does not simulate candidate-dependent map history
    or reset hysteresis on hypothetical insertions. Intervention evaluates
    only the active candidate on its own evolving map. All wrappers restore
    on exceptions, and the actual legacy function still runs exactly once.
    """

    def __init__(self, audit, mode):
        if mode != "shadow" and mode not in SELECTORS:
            raise ValueError(f"Unknown selector: {mode}")
        self.audit = audit
        self.mode = mode
        names = SELECTORS if mode == "shadow" else (mode,)
        self.states = {name: SelectorState() for name in names}

    def __enter__(self):
        original = tracker_mod.should_insert_keyframe

        def policy(frame, reference, frames_since_last_keyframe=0, **kwargs):
            # Phase 8-4: the tracker passes the run's video_fps through
            # here (time-normalized cadence constants); the experiment
            # wrapper is a pure passthrough for it.
            legacy = original(frame, reference, frames_since_last_keyframe,
                              **kwargs)
            if self.audit._current is None or self.audit._current["decision"] is None:
                raise RuntimeError("KeyframePolicyExperiment requires an active KeyframeAudit update")
            row = self.audit._current["decision"]
            calls = [call for call in self.audit._current["pnp_calls"] if call["explicit_current_frame"]]
            pose_accepted = calls[-1]["success"] if calls else None
            decisions = {name: evaluate_selector(
                SELECTORS[name], state, row["signals"], frames_since_last_keyframe,
                reference.frame_id, legacy[0], pose_accepted) for name, state in self.states.items()}
            row["experimental_decisions"] = {name: asdict(decision) for name, decision in decisions.items()}
            if self.mode == "shadow":
                return legacy
            decision = decisions[self.mode]
            row["legacy_policy_selected"] = row["policy_selected"]
            row["legacy_primary_reason"] = row["primary_reason"]
            row["flags_describe"] = "Legacy thresholds only, not the experimental decision"
            row["policy_selected"] = decision.selected
            row["effective_selected"] = decision.selected or row["flags"]["maximum_gap"]
            row["primary_reason"] = (decision.reason if decision.selected else "maximum_gap_only"
                                     if row["flags"]["maximum_gap"] else decision.reason)
            return decision.selected, decision.reason

        self._patch = patch.object(tracker_mod, "should_insert_keyframe", policy)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        return self._patch.__exit__(*exc)

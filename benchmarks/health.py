"""The SHARED health-signature module (used by the video
census AND the GT ladder, so both benchmark tiers speak one
language).

Pure numpy over plain dicts — no slam imports, so it stays importable
from any bench context and trivially testable. Consumers record the
per-processed-frame health stream (the proven kyoto_pan_slam probe
schema, plus the rotation-model gate state) and this module turns it into:

- summarize(): the per-run health summary (rates, percentiles,
  starvation, churn, pose plausibility, timing);
- classify(): the PRE-REGISTERED failure-mode labels — the thresholds
  below were stated in IMPROVEMENT.md BEFORE any corpus data was
  seen, and are guarded by pins; the corpus can never tune them;
- motion_signal(): the pre-registered interpretive motion-class
  signal (the rotation-model gate IS a rotation-dominance detector on the
  fed cadence);
- health_table(): the printable cross-video comparison.

Frame-row contract (keys the summarizer reads; all recorded by the
census runner):
    i        decoded frame index
    kf       promoted to keyframe
    feat     feature count (None-safe)
    obs      observed map points
    q        map.avg_tracking_quality after the update
    low_q    map.consecutive_low_quality_frames after the update
    kfs      keyframes in the map
    pts      map points
    archives archived maps
    head     camera heading, world XZ (deg; Twc column-2 atan2)
    t        |camera center| so far
    d_rot    rotation angle vs the previous fed frame (deg)
    d_t      camera-center step vs the previous fed frame
    v_rot    velocity rotation magnitude (deg)
    v_t      velocity translation magnitude
    rm       the rotation-model gate state for this pair
    ev       per-frame telemetry event deltas (dict)
"""
import numpy as np

# ===== pre-registered classifier thresholds =====
# Measured pins, not tuning targets (pre-registered in IMPROVEMENT.md).
RESET_STORM_MIN_RESETS = 2
RESET_STORM_EARLY_FED_FRAC = 0.10
RELOC_CHURN_MIN_ATTEMPTS = 5
RELOC_CHURN_RECOVERY_Q_P50 = 0.3
MAP_STARVATION_MIN_POINTS = 50
MAP_STARVATION_FED_FRAC = 0.20
QUALITY_COLLAPSE_Q_P50 = 0.2
QUALITY_COLLAPSE_STREAK_FED_FRAC = 0.30
INTAKE_STARVATION_FED_RATE = 0.25  # review-flag
SCALE_BLOWUP_MAX_STEP = 100.0
HEADING_ANOMALY_TOTAL_DEG = 720.0  # review-flag
# motion signal (pre-registered)
MOTION_RM_ENGAGED_DOMINANT_FRAC = 0.5
MOTION_RM_ENGAGED_TRANSLATION_FRAC = 0.1
MOTION_LOW_MOTION_FED_RATE = 0.25

FAILURE_LABELS = ("RESET_STORM", "RELOC_CHURN", "MAP_STARVATION",
                  "QUALITY_COLLAPSE", "SCALE_BLOWUP")
REVIEW_LABELS = ("INTAKE_STARVATION", "HEADING_ANOMALY")


def heading_deg(pose):
    """Where the camera looks, projected on the world XZ plane (deg).
    pose is camera-to-world (Twc); the viewing direction is the
    camera z-axis in world coords = column 2 of R_wc. A pure yaw pan
    sweeps this angle; a level start makes it start near 0."""
    R = pose[:3, :3]
    v = R[:, 2]
    return float(np.degrees(np.arctan2(v[0], v[2])))


def _p(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else None


def summarize_run(frames, resets, telemetry_totals, decoded, fed,
                  seconds):
    """The health summary for one run. `frames` is the per-FED-frame
    stream (the row contract above); `resets` the decoded indices
    where tracker.step went backwards; `telemetry_totals` the run's
    full event counters; `seconds` wall time of the run proper."""
    s = {
        "decoded": decoded,
        "fed": fed,
        "fed_rate": (fed / decoded) if decoded else None,
        "seconds": seconds,
        "fps": (fed / seconds) if seconds > 0 else None,
        "resets": list(resets),
    }
    q = np.asarray([f["q"] for f in frames], dtype=float)
    pts = np.asarray([f["pts"] for f in frames], dtype=float)
    d_t = np.asarray([f["d_t"] for f in frames], dtype=float)
    d_rot = np.asarray([f["d_rot"] for f in frames], dtype=float)
    low_q = np.asarray([f["low_q"] for f in frames], dtype=float)
    obs = np.asarray([f["obs"] for f in frames], dtype=float)

    if len(frames):
        s["quality"] = {
            "p10": _p(q, 10), "p50": float(np.median(q)),
            "p90": _p(q, 90), "min": float(q.min()),
            "max_lowq_streak": int(low_q.max()),
            "max_lowq_streak_frac": float(low_q.max() / max(len(frames), 1)),
        }
        s["map"] = {
            "pts_min": int(pts.min()), "pts_median": float(np.median(pts)),
            "pts_final": int(pts[-1]),
            "starved_frac": float(np.mean(pts < MAP_STARVATION_MIN_POINTS)),
            "obs_p50": float(np.median(obs)),
        }
        s["pose"] = {
            "heading_total_deg": _heading_total(frames),
            "t_final": float(frames[-1]["t"]),
            "t_max": float(max(f["t"] for f in frames)),
            "max_step": float(d_t.max()),
            "mean_d_rot_deg": float(np.mean(d_rot)),
            "median_step": float(np.median(d_t)),
        }
        # reset position in the FED stream (fraction at first reset)
        idx_at = {f["i"]: k for k, f in enumerate(frames)}
        if resets:
            first_fed_pos = min(
                (idx_at[r] + 1 for r in resets if r in idx_at),
                default=len(frames))
            s["reset_first_fed_frac"] = float(
                first_fed_pos / max(len(frames), 1))
        else:
            s["reset_first_fed_frac"] = None

    ev = telemetry_totals
    s["tri"] = {
        "accepted": ev.get("tri_accepted", 0),
        "rejected": ev.get("tri_rejected", 0),
        "accept_ratio": (ev.get("tri_accepted", 0)
                         / max(ev.get("tri_accepted", 0)
                               + ev.get("tri_rejected", 0), 1)),
    }
    s["reloc"] = {
        "attempts": ev.get("reloc_attempt", 0),
        "successes": (ev.get("reloc_success", 0)
                     + ev.get("reloc_aggressive_success", 0)),
        "per_fed": (ev.get("reloc_attempt", 0) / max(fed, 1)),
    }
    s["rotation_model"] = {
        # pairs = fed - 1: the first fed frame establishes no pair
        "engaged_pairs": int(sum(1 for f in frames if f.get("rm"))),
        "engage_events": ev.get("pair_rotation_model_engage", 0),
        "rearm_events": ev.get("pair_rotation_model_rearm", 0),
        "engaged_frac": (sum(1 for f in frames if f.get("rm"))
                         / max(len(frames) - 1, 1)) if len(frames) else 0.0,
    }
    s["events"] = dict(ev)
    s["labels"] = classify(s)
    s["motion"] = motion_signal(s)
    return s


def _heading_total(frames):
    """Unwrapped net heading sweep (deg) over the run."""
    if not frames:
        return None
    heads = [f["head"] for f in frames]
    return float(np.degrees(np.unwrap(np.radians(heads))[-1]))


def classify(s):
    """The pre-registered failure-mode labels for one summary.
    FAILURE labels trip the classifiers; REVIEW labels (intake
    starvation, heading anomaly) are content-dependent flags that
    carry no HEALTHY/failed verdict on their own."""
    labels = []
    n_resets = len(s.get("resets", []))
    fed = s.get("fed", 0) or 0
    # RESET_STORM
    if (n_resets >= RESET_STORM_MIN_RESETS
            or (n_resets and (s.get("reset_first_fed_frac") is not None
                              and s["reset_first_fed_frac"]
                              < RESET_STORM_EARLY_FED_FRAC))):
        labels.append("RESET_STORM")
    # RELOC_CHURN
    if (s["reloc"]["attempts"] >= RELOC_CHURN_MIN_ATTEMPTS
            and s.get("quality") is not None
            and s["quality"]["p50"] < RELOC_CHURN_RECOVERY_Q_P50):
        labels.append("RELOC_CHURN")
    # MAP_STARVATION
    if s.get("map") is not None and (
            s["map"]["starved_frac"] >= MAP_STARVATION_FED_FRAC):
        labels.append("MAP_STARVATION")
    # QUALITY_COLLAPSE
    if s.get("quality") is not None and (
            s["quality"]["p50"] < QUALITY_COLLAPSE_Q_P50
            or s["quality"]["max_lowq_streak_frac"]
            >= QUALITY_COLLAPSE_STREAK_FED_FRAC):
        labels.append("QUALITY_COLLAPSE")
    # SCALE_BLOWUP
    if (s.get("pose") is not None
            and s["pose"]["max_step"] > SCALE_BLOWUP_MAX_STEP):
        labels.append("SCALE_BLOWUP")
    # review flags
    if s.get("fed_rate") is not None and (
            s["fed_rate"] < INTAKE_STARVATION_FED_RATE):
        labels.append("INTAKE_STARVATION")
    if (s.get("pose") is not None
            and abs(s["pose"]["heading_total_deg"] or 0.0)
            > HEADING_ANOMALY_TOTAL_DEG):
        labels.append("HEADING_ANOMALY")
    return labels


def motion_signal(s):
    """The pre-registered interpretive motion class. The rotation-model
    t-coherence gate is a rotation-dominance DETECTOR on the fed
    cadence: sustained engagement means rotation-only content;
    near-zero engagement with a healthy feed means translation
    content."""
    rm_frac = s["rotation_model"]["engaged_frac"]
    fed_rate = s.get("fed_rate")
    if fed_rate is not None and fed_rate < MOTION_LOW_MOTION_FED_RATE:
        label = "low_motion"
    elif rm_frac >= MOTION_RM_ENGAGED_DOMINANT_FRAC:
        label = "rotation_dominant"
    elif rm_frac < MOTION_RM_ENGAGED_TRANSLATION_FRAC:
        label = "translation_dominant"
    else:
        label = "mixed"
    return {
        "label": label,
        "rm_engaged_frac": rm_frac,
        "mean_d_rot_deg": s["pose"]["mean_d_rot_deg"]
        if s.get("pose") else None,
        "median_step": s["pose"]["median_step"] if s.get("pose") else None,
    }


def health_table(summaries):
    """The printable cross-video table: name, motion class, labels,
    and the headline numbers (one row per summary dict keyed by
    name; summaries may be per-video census blocks or per-sequence
    ladder blocks)."""
    cols = [("name", 22), ("motion", 20), ("fed", 5), ("fed%", 5),
            ("q p50", 6), ("pts med", 8), ("starve%", 7), ("reloc", 6),
            ("resets", 6), ("|t| end", 8), ("rm%", 4), ("labels", 24)]
    header = " ".join(c.ljust(w) for c, w in cols)
    lines = ["\n" + header, "-" * len(header)]
    for name, s in summaries.items():
        q = s.get("quality") or {}
        m = s.get("map") or {}
        p = s.get("pose") or {}
        labels = ",".join(s.get("labels", [])) or "HEALTHY"
        row = [
            name[:22],
            s.get("motion", {}).get("label", "-"),
            str(s.get("fed", "-")),
            f"{100 * s['fed_rate']:.0f}" if s.get("fed_rate") is not None else "-",
            f"{q['p50']:.2f}" if q.get("p50") is not None else "-",
            f"{m['pts_median']:.0f}" if m.get("pts_median") is not None else "-",
            f"{100 * m['starved_frac']:.0f}" if m.get("starved_frac") is not None else "-",
            str(s.get("reloc", {}).get("attempts", "-")),
            str(len(s.get("resets", []))),
            f"{p['t_final']:.1f}" if p.get("t_final") is not None else "-",
            f"{100 * s['rotation_model']['engaged_frac']:.0f}",
            labels[:24],
        ]
        lines.append(" ".join(str(v).ljust(w)
                              for v, (_, w) in zip(row, cols)))
    return "\n".join(lines)

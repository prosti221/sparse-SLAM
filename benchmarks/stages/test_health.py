"""Stage pins: the shared health-signature module (phase 14) + the
attribution rotation gauge (phase 15).

These hold:
- the PRE-REGISTERED classifier thresholds exactly as recorded in
  IMPROVEMENT.md (measured pins — edits fail loudly here; never
  retune to make a corpus look better);
- the frame-row contract: summarize_run/classify/motion_signal on
  synthetic streams, one per failure mode + the healthy path + the
  review flags;
- the attribution ROTATION GAUGE: the orthogonal-Procrustes
  orientation alignment absorbs the arbitrary world-frame offset
  (the measured lesson: comparing raw est-vs-GT orientations on a
  near-point center cloud reported ~141 deg of phantom error with
  ATE 0.03 — Umeyama's center-alignment R cannot gauge rotation).
"""
import numpy as np
import pytest

import benchmarks.health as health
from benchmarks.attribution import _rotation_alignment, _rot_angle_deg
from benchmarks.health import classify, health_table, heading_deg, \
    summarize_run


def _rows(n, q=0.5, pts=1000, obs=100, rm=False, t=5.0, head=0.0,
          low_q=0, d_t=0.5, d_rot=0.5):
    return [{"i": i, "kf": False, "feat": 200, "obs": obs,
             "q": q, "low_q": low_q, "kfs": 5, "pts": pts,
             "archives": 0, "head": head, "t": t,
             "d_rot": d_rot, "d_t": d_t,
             "v_rot": d_rot, "v_t": d_t, "rm": rm, "ev": {}}
            for i in range(n)]


def _summary(frames, resets=(), events=None, decoded=None, fed=None,
             ev=None):
    ev = dict(ev or {})
    ev.setdefault("reloc_attempt", 0)
    ev.setdefault("reloc_success", 0)
    s = summarize_run(frames, list(resets), ev,
                      decoded if decoded is not None else len(frames),
                      fed if fed is not None else len(frames),
                      10.0)
    return s


# ===== the pre-registered thresholds (measured pins) =====

def test_classifier_threshold_pins():
    assert health.RESET_STORM_MIN_RESETS == 2
    assert health.RESET_STORM_EARLY_FED_FRAC == 0.10
    assert health.RELOC_CHURN_MIN_ATTEMPTS == 5
    assert health.RELOC_CHURN_RECOVERY_Q_P50 == 0.3
    assert health.MAP_STARVATION_MIN_POINTS == 50
    assert health.MAP_STARVATION_FED_FRAC == 0.20
    assert health.QUALITY_COLLAPSE_Q_P50 == 0.2
    assert health.QUALITY_COLLAPSE_STREAK_FED_FRAC == 0.30
    assert health.INTAKE_STARVATION_FED_RATE == 0.25
    assert health.SCALE_BLOWUP_MAX_STEP == 100.0
    assert health.HEADING_ANOMALY_TOTAL_DEG == 720.0
    assert health.MOTION_RM_ENGAGED_DOMINANT_FRAC == 0.5
    assert health.MOTION_RM_ENGAGED_TRANSLATION_FRAC == 0.1
    assert health.MOTION_LOW_MOTION_FED_RATE == 0.25
    assert set(health.FAILURE_LABELS) == {
        "RESET_STORM", "RELOC_CHURN", "MAP_STARVATION",
        "QUALITY_COLLAPSE", "SCALE_BLOWUP"}
    assert set(health.REVIEW_LABELS) == {
        "INTAKE_STARVATION", "HEADING_ANOMALY"}


# ===== the row contract: one synthetic stream per mode =====

def test_healthy_stream():
    s = _summary(_rows(100))
    assert s["labels"] == []
    assert s["motion"]["label"] == "translation_dominant"
    assert s["fed_rate"] == 1.0
    assert s["quality"]["p50"] == 0.5


def test_reset_storm_by_count():
    s = _summary(_rows(100), resets=[20, 60])
    assert "RESET_STORM" in s["labels"]


def test_reset_storm_by_early_position():
    # ONE reset inside the first 10% of fed frames trips the storm rule
    s = _summary(_rows(100), resets=[5])
    assert "RESET_STORM" in s["labels"]
    # one reset late in the run does not
    s2 = _summary(_rows(100), resets=[80])
    assert "RESET_STORM" not in s2["labels"]


def test_reloc_churn_needs_low_quality():
    churn_ev = {"reloc_attempt": 8, "reloc_success": 8}
    s = _summary(_rows(100, q=0.1), ev=churn_ev)
    assert "RELOC_CHURN" in s["labels"]
    # 8 relocs WITH recovered quality is successful relocalization,
    # not churn (finding-36's recovery accounting)
    s2 = _summary(_rows(100, q=0.6), ev=churn_ev)
    assert "RELOC_CHURN" not in s2["labels"]


def test_map_starvation():
    starved = _rows(100, pts=2)          # the kyoto-pan class
    s = _summary(starved)
    assert "MAP_STARVATION" in s["labels"]
    brief = _rows(100)
    brief[0]["pts"] = 2                  # one starved frame is noise
    assert "MAP_STARVATION" not in _summary(brief)["labels"]


def test_quality_collapse_both_rules():
    s = _summary(_rows(100, q=0.1))
    assert "QUALITY_COLLAPSE" in s["labels"]          # q p50 rule
    rows = _rows(100)
    for i in range(40):                               # streak rule: the
        rows[i]["low_q"] = i + 1                      # counter RAMPS to
    s2 = _summary(rows)                               # 40 = 40% of fed
    assert "QUALITY_COLLAPSE" in s2["labels"]
    assert s2["quality"]["max_lowq_streak_frac"] == 0.4


def test_scale_blowup():
    rows = _rows(100)
    rows[50]["d_t"] = 382.3                           # the road reset
    s = _summary(rows)
    assert "SCALE_BLOWUP" in s["labels"]
    assert s["pose"]["max_step"] == 382.3


def test_review_flags_carry_no_failure_verdict():
    s = _summary(_rows(100), decoded=400)             # fed 25% -> 0.25
    assert s["labels"] == []                            # not < 0.25
    s2 = _summary(_rows(100), decoded=500)             # 0.2 < 0.25
    assert s2["labels"] == ["INTAKE_STARVATION"]
    s3 = _summary(_rows(100, head=800.0))
    assert s3["labels"] == ["HEADING_ANOMALY"]


def test_motion_signal_rules():
    rot = _rows(100, rm=True)
    assert _summary(rot)["motion"]["label"] == "rotation_dominant"
    tr = _rows(100, rm=False)
    assert _summary(tr)["motion"]["label"] == "translation_dominant"
    mixed = _rows(100)
    for i in range(30):            # 30/99 = 0.303: inside [0.1, 0.5)
        mixed[i]["rm"] = True      # (50/99 = 0.505 is already
    assert _summary(mixed)["motion"]["label"] == "mixed"
    low = _rows(100)
    s = summarize_run(low, [], {"reloc_attempt": 0}, 1000, 100, 10.0)
    assert s["motion"]["label"] == "low_motion"


def test_heading_deg_contract():
    # camera-to-world pose; viewing direction = column 2 of R_wc
    yaw = lambda deg: np.array([
        [np.cos(np.radians(deg)), 0, np.sin(np.radians(deg))],
        [0, 1, 0],
        [-np.sin(np.radians(deg)), 0, np.cos(np.radians(deg))]])
    pose = np.eye(4)
    pose[:3, :3] = yaw(90)
    assert abs(heading_deg(pose) - 90.0) < 1e-9
    pose[:3, :3] = yaw(0)
    assert abs(heading_deg(pose)) < 1e-9


def test_health_table_renders_every_summary():
    s = _summary(_rows(50))
    table = health_table({"x.mp4": s})
    assert "HEALTHY" in table and "x.mp4" in table


# ===== the attribution rotation gauge (the measured lesson) =====

def test_rotation_alignment_absorbs_world_frame_offset():
    """A constant world-frame rotation between est and GT must be
    ABSORBED by the gauge (zero residual error) — the ~141 deg
    phantom-error lesson: raw orientation comparison without a
    rotation gauge is meaningless."""
    rng = np.random.default_rng(7)
    # a GT trajectory of random orientations
    quats = rng.normal(size=(30, 4))
    from scipy.spatial.transform import Rotation
    gt = np.array([np.eye(4) for _ in range(30)])
    gt[:, :3, :3] = Rotation.from_quat(quats).as_matrix()
    # est = GT rotated by a CONSTANT arbitrary world-frame offset
    C = Rotation.from_euler("zyx", [123.0, -17.0, 66.0],
                            degrees=True).as_matrix()
    est = gt.copy()
    est[:, :3, :3] = C[None] @ gt[:, :3, :3]
    R_rot = _rotation_alignment(est, gt, 30)
    errs = [_rot_angle_deg(
                (R_rot @ est[i, :3, :3]).T @ gt[i, :3, :3])
            for i in range(30)]
    assert max(errs) < 1e-4      # float accumulation, not zero


def test_rotation_alignment_detects_real_drift():
    """A genuine per-frame orientation error must SURVIVE the gauge
    (the gauge is not an error absorber for real drift). The
    Procrustes mean splits uniform drift (a full-frame 20-deg block
    reads as ~10 deg on both sides) — so detect via an isolated
    outlier the majority-gauge cannot dilute."""
    from scipy.spatial.transform import Rotation
    gt = np.array([np.eye(4) for _ in range(10)])
    est = gt.copy()
    est[9, :3, :3] = Rotation.from_euler(
        "y", 90.0, degrees=True).as_matrix()   # one 90-deg outlier
    R_rot = _rotation_alignment(est, gt, 10)
    errs = [_rot_angle_deg(
                (R_rot @ est[i, :3, :3]).T @ gt[i, :3, :3])
            for i in range(10)]
    assert max(errs) > 40.0                     # undilutable residue
    assert errs[9] == max(errs)                # and it's the outlier

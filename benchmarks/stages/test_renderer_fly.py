"""Stage tests: renderer fly controls (WASD + Q/E + shift boost).

Open3D is mocked at the slam.viz.renderer namespace: the fake
visualizer records key registrations and the fake view control
records camera_local_translate calls - no window is ever created. The
benchmark harness drives Map + Tracker directly and never constructs a
Renderer, so the frozen battery is untouched by construction. These
pins lock the fly-mode wiring: the GLFW keycodes (including the
dead-113 quit registration's removal and the 't'-follow deprecation),
held-state transitions across PRESS/REPEAT/RELEASE, the dt-scaled
normalized displacement math, the boost and speed-step behavior, the
scene-scale cache, and the one-time initial camera placement.
"""
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from slam.viz import renderer as renderer_mod
from slam.viz.renderer import (
    Renderer,
    KEY_PAUSE,
    KEY_FLY_FORWARD,
    KEY_FLY_BACK,
    KEY_FLY_LEFT,
    KEY_FLY_RIGHT,
    KEY_FLY_DOWN,
    KEY_FLY_UP,
    KEY_FLY_BOOST,
    KEY_FLY_SLOWER,
    KEY_FLY_FASTER,
    KEY_ACTION_PRESS,
    KEY_ACTION_REPEAT,
    KEY_ACTION_RELEASE,
    FLY_SPEED_SCENE_FRACTION,
    FLY_SPEED_BOOST,
    FLY_DEFAULT_SCENE_DIAGONAL,
    FLY_MAX_DT_S,
    FLY_SPEED_MULT_MIN,
    FLY_SPEED_MULT_MAX,
)

PRESS, REPEAT, RELEASE = KEY_ACTION_PRESS, KEY_ACTION_REPEAT, KEY_ACTION_RELEASE
FLY_KEYS = (KEY_FLY_FORWARD, KEY_FLY_BACK, KEY_FLY_LEFT, KEY_FLY_RIGHT,
            KEY_FLY_UP, KEY_FLY_DOWN)


class FakeViewControl:
    def __init__(self):
        self.translate_calls = []
        self.from_pinhole_calls = 0

    def convert_from_pinhole_camera_parameters(self, params,
                                                allow_arbitrary=False):
        self.from_pinhole_calls += 1

    def set_constant_z_far(self, far):
        pass

    def camera_local_translate(self, forward, right, up):
        self.translate_calls.append((forward, right, up))


class FakeVisualizer:
    def __init__(self):
        self.key_callbacks = {}
        self.key_action_callbacks = {}
        self.render_option = MagicMock()
        self.ctrl = FakeViewControl()

    def create_window(self, **kwargs):
        pass

    def get_render_option(self):
        return self.render_option

    def get_view_control(self):
        return self.ctrl

    def register_key_callback(self, key, callback):
        self.key_callbacks[key] = callback

    def register_key_action_callback(self, key, callback):
        self.key_action_callbacks[key] = callback

    def add_geometry(self, geometry, fit=True):
        pass

    def remove_geometry(self, geometry, fit=True):
        pass

    def update_geometry(self, geometry):
        pass

    def poll_events(self):
        return True

    def update_renderer(self):
        pass

    def destroy_window(self):
        pass


class FakeFrameId(int):
    """Hashable int-like frame id with the production `.int` accessor
    (types.SimpleNamespace defines __eq__ and is therefore unhashable,
    and update_poses builds a set of frame ids)."""

    @property
    def int(self):
        return int(self)


class FakeKeyframe:
    def __init__(self, fid, quality=0.9):
        self.frame_id = FakeFrameId(fid)
        self.pose = np.eye(4)
        self.tracking_quality = quality


class FakeMapPoint:
    """Just what pt_obj_to_array reads."""

    def __init__(self):
        self.pt_3d = np.array([1.0, 2.0, 3.0])
        self.color = np.array([128, 128, 128], dtype=np.float32)


@pytest.fixture
def fx(monkeypatch, tmp_path):
    """A started Renderer with Open3D mocked (start() also mkdirs the
    session folder, so it is pointed at tmp_path)."""
    fake = FakeVisualizer()
    o3d_mock = MagicMock()
    o3d_mock.visualization.VisualizerWithKeyCallback.return_value = fake
    monkeypatch.setattr(renderer_mod, 'o3d', o3d_mock)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    r = Renderer(MagicMock(), K, W=640, H=480)
    r.output_path = str(tmp_path)
    r.start()
    return types.SimpleNamespace(vis=fake, ctrl=fake.ctrl, r=r)


def hold(fx, *keys):
    for key in keys:
        fx.vis.key_action_callbacks[key](None, PRESS, 0)


def release(fx, *keys):
    for key in keys:
        fx.vis.key_action_callbacks[key](None, RELEASE, 0)


def test_registrations_pin_the_fly_layout(fx):
    """The pause key keeps the plain press-callback API; the fly keys,
    shift boosts and speed keys are ACTION callbacks (they need
    RELEASE); 'q' (81) is fly-down, not quit; the dead 113 quit
    registration and the deprecated 84 tracking registration are gone."""
    assert set(fx.vis.key_callbacks.keys()) == {KEY_PAUSE}
    expected = set(FLY_KEYS) | set(KEY_FLY_BOOST) | {
        KEY_FLY_SLOWER, KEY_FLY_FASTER}
    assert set(fx.vis.key_action_callbacks.keys()) == expected
    for dead in (113, 84):
        assert dead not in fx.vis.key_callbacks
        assert dead not in fx.vis.key_action_callbacks
    # pause still toggles
    fx.vis.key_callbacks[KEY_PAUSE](None)
    assert fx.r.is_paused()


def test_fly_key_state_transitions(fx):
    """PRESS adds to the held set, REPEAT is a no-op (no state change
    beyond PRESS), RELEASE removes - the hold-to-move contract."""
    handler = fx.vis.key_action_callbacks[KEY_FLY_FORWARD]
    handler(None, PRESS, 0)
    assert fx.r._fly_keys_held == {KEY_FLY_FORWARD}
    handler(None, REPEAT, 0)
    assert fx.r._fly_keys_held == {KEY_FLY_FORWARD}
    handler(None, RELEASE, 0)
    assert fx.r._fly_keys_held == set()
    # release of an unheld key is safe
    handler(None, RELEASE, 0)
    assert fx.r._fly_keys_held == set()


def test_boost_keys_tracked_in_their_own_set(fx):
    handler = fx.vis.key_action_callbacks[KEY_FLY_BOOST[0]]
    handler(None, PRESS, 0)
    assert fx.r._fly_boost_held == {KEY_FLY_BOOST[0]}
    handler(None, RELEASE, 0)
    assert fx.r._fly_boost_held == set()


def test_displacement_axes_normalization_and_cancellation(fx):
    """World-unit displacement = speed * dt along the held axes,
    L2-normalized (diagonal is not faster); opposite keys cancel; no
    held keys -> None. Default scale: 0.3 * 100 diagonal = 30 units/s."""
    speed = FLY_SPEED_SCENE_FRACTION * FLY_DEFAULT_SCENE_DIAGONAL
    hold(fx, KEY_FLY_FORWARD)
    assert fx.r._fly_displacement(0.1) == (speed * 0.1, 0.0, 0.0)
    hold(fx, KEY_FLY_RIGHT)
    assert fx.r._fly_displacement(0.1) == pytest.approx(
        (speed * 0.1 / np.sqrt(2), speed * 0.1 / np.sqrt(2), 0.0))
    release(fx, KEY_FLY_RIGHT)
    hold(fx, KEY_FLY_BACK)
    # forward + back cancel exactly
    assert fx.r._fly_displacement(0.1) is None
    release(fx, KEY_FLY_FORWARD, KEY_FLY_BACK)
    hold(fx, KEY_FLY_DOWN)
    assert fx.r._fly_displacement(0.1) == (0.0, 0.0, -speed * 0.1)
    release(fx, KEY_FLY_DOWN)
    assert fx.r._fly_displacement(0.1) is None


def test_boost_multiplies_speed(fx):
    hold(fx, KEY_FLY_BOOST[1])
    hold(fx, KEY_FLY_FORWARD)
    assert fx.r._fly_displacement(0.1) == (
        FLY_SPEED_SCENE_FRACTION * FLY_DEFAULT_SCENE_DIAGONAL
        * FLY_SPEED_BOOST * 0.1, 0.0, 0.0)


def test_speed_keys_step_the_multiplier_with_clamps(fx):
    """'=' x2 / '-' x2 per PRESS, clamped; REPEAT does not stack
    (unlike the plain key-callback API, which also fires on repeat)."""
    faster = fx.vis.key_action_callbacks[KEY_FLY_FASTER]
    slower = fx.vis.key_action_callbacks[KEY_FLY_SLOWER]
    faster(None, PRESS, 0)
    assert fx.r._fly_speed_multiplier == 2.0
    faster(None, REPEAT, 0)
    assert fx.r._fly_speed_multiplier == 2.0
    for _ in range(5):
        faster(None, PRESS, 0)
    assert fx.r._fly_speed_multiplier == FLY_SPEED_MULT_MAX
    for _ in range(10):          # 32 -> 32/1024 == FLY_SPEED_MULT_MIN
        slower(None, PRESS, 0)
    assert fx.r._fly_speed_multiplier == FLY_SPEED_MULT_MIN
    slower(None, RELEASE, 0)  # speed keys have no held state
    assert fx.r._fly_speed_multiplier == FLY_SPEED_MULT_MIN


def test_scene_scale_cache_sets_speed_reference(fx):
    """The fly speed follows the point-cloud bbox diagonal (monocular
    scale); empty or degenerate clouds leave the cache untouched."""
    fx.r._fly_cache_scene_scale(np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    assert fx.r._fly_scene_diagonal == 5.0
    hold(fx, KEY_FLY_FORWARD)
    assert fx.r._fly_displacement(0.1) == (
        FLY_SPEED_SCENE_FRACTION * 5.0 * 0.1, 0.0, 0.0)
    fx.r._fly_cache_scene_scale(np.zeros((0, 3)))
    assert fx.r._fly_scene_diagonal == 5.0
    fx.r._fly_cache_scene_scale(np.ones((4, 3)))
    assert fx.r._fly_scene_diagonal == 5.0


def test_process_fly_input_uses_wall_clock_dt_with_cap(fx, monkeypatch):
    """The explicit tick converts held keys + wall-clock dt into
    camera_local_translate calls, capped at FLY_MAX_DT_S so a mid-BA
    stall cannot teleport the camera. Movement deliberately ignores
    pause (the loop keeps ticking while paused)."""
    clock = types.SimpleNamespace(now=0.0)
    monkeypatch.setattr(renderer_mod, 'time',
                        types.SimpleNamespace(monotonic=lambda: clock.now))
    speed = FLY_SPEED_SCENE_FRACTION * FLY_DEFAULT_SCENE_DIAGONAL

    fx.r.process_fly_input()               # first tick: no dt yet
    hold(fx, KEY_FLY_FORWARD)
    clock.now = 0.016
    fx.r.process_fly_input()
    assert fx.ctrl.translate_calls == [(speed * 0.016, 0.0, 0.0)]

    fx.r.paused = True                     # paused does not block movement
    clock.now = 0.032
    fx.r.process_fly_input()
    assert fx.ctrl.translate_calls[-1] == (speed * 0.016, 0.0, 0.0)

    clock.now = 10.0                       # huge gap: capped, not summed
    fx.r.process_fly_input()
    assert fx.ctrl.translate_calls[-1] == (speed * FLY_MAX_DT_S, 0.0, 0.0)

    release(fx, KEY_FLY_FORWARD)
    clock.now = 10.1
    fx.r.process_fly_input()
    assert len(fx.ctrl.translate_calls) == 3


def test_tracking_mode_fully_removed(fx):
    """The 't' follow mode is deprecated: no tracking state, no
    quit-request plumbing, and the one-time initial placement replaces
    the per-keyframe follow."""
    for attr in ('tracking_enabled', 'last_tracked_frame', 'quit_requested'):
        assert not hasattr(fx.r, attr)
    assert not hasattr(fx.r, 'should_quit')

    placement_after_start = fx.ctrl.from_pinhole_calls
    fx.r.update_points([FakeMapPoint()])   # the live update() order:
    fx.r.update_poses([FakeKeyframe(1)])   # points first, then poses
    assert fx.ctrl.from_pinhole_calls == placement_after_start + 1
    assert fx.r.camera_initialized
    fx.r.update_poses([FakeKeyframe(1), FakeKeyframe(2)])
    assert fx.ctrl.from_pinhole_calls == placement_after_start + 1


def test_initial_placement_deferred_until_the_cloud_is_in_the_scene(fx):
    """The placement fires on the first update that has BOTH keyframes
    and the point cloud in the scene: on an empty scene the conversion
    produces an invalid NaN view state, and the cloud's later
    add_geometry (reset_bounding_box defaults True) re-fits the view
    over it. update() runs update_points before update_poses, so the
    placement lands after the only view-resetting add. Empty keyframes
    -> no placement; keyframes without the cloud -> no placement (and
    no leaked-`kf` crash: the session save lives inside the loop)."""
    placement_after_start = fx.ctrl.from_pinhole_calls

    fx.r.update_poses([])
    assert fx.ctrl.from_pinhole_calls == placement_after_start

    # keyframes but no cloud yet -> still deferred
    fx.r.update_poses([FakeKeyframe(1)])
    assert fx.ctrl.from_pinhole_calls == placement_after_start
    assert not fx.r.camera_initialized

    # the cloud arrives (update_points), the NEXT update_poses places
    fx.r.update_points([FakeMapPoint()])
    fx.r.update_poses([FakeKeyframe(1)])
    assert fx.ctrl.from_pinhole_calls == placement_after_start + 1
    assert fx.r.camera_initialized

    fx.r.update_poses([FakeKeyframe(1), FakeKeyframe(2)])
    assert fx.ctrl.from_pinhole_calls == placement_after_start + 1
    assert set(fx.r.session_data['poses'].keys()) == {1, 2}
    assert fx.r.session_data['poses'][2] == {
        'pose': np.eye(4).tolist(), 'tracking_quality': 0.9}

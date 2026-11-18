"""The SLAM step worker: the process that OWNS the pipeline.

Spawned by the editor's session driver (the intake-process pattern,
two layers down: the worker itself spawns the extraction children, so
it must be daemon=False). It owns the ENTIRE pipeline state - the
source, the extraction workers, the intake gate, the tracker, the
map, the native registries, g2o, the loop components - and runs
Session.step verbatim per command.

Protocol:
- recv queue: ("step",), ("set_pins", [(n, v)...]),
  ("set_options", opts-dict), ("checkpoint",), ("rewind", cp),
  ("save",), ("close",).
- send queue: ("ready", meta); ("step", summary, slot_name) per
  step; ("eos", counts); ("checkpoint", cp); ("rewound", record);
  ("pins_applied", n); ("options_applied", branch-dict);
  ("saved",); ("terminated",).
- camera shm: a ping-pong slot pair for the current camera frame
  (the recording/display path copies it UI-side).

The determinism: this runs the SAME Session.step code with the SAME
per-step seeds as the in-process path - the M7.0 spike measured the
dialogue BIT-EQUAL to an in-process run.
"""
import os
import sys


def worker_main(recv, send, spec, opts, repo):
    sys.path.insert(0, repo)
    import multiprocessing.shared_memory as shm_mod
    import numpy as np

    from slam.ui.session import Session, SlamOptions

    s = Session(spec, SlamOptions(**opts))

    slot_bytes = s.CAP_H * s.CAP_W * 3
    shms, slots = [], []
    names = []
    for i in range(2):
        shm = shm_mod.SharedMemory(
            create=True, size=slot_bytes, name=f"m7cam{i}-{os.getpid()}")
        shms.append(shm)
        slots.append(np.ndarray((s.CAP_H, s.CAP_W, 3), dtype=np.uint8,
                                buffer=shm.buf))
        names.append(shm.name)
    slot = 0
    send.put(("ready", {"shm": names, "w": s.CAP_W, "h": s.CAP_H,
                        "K": np.asarray(s.K, dtype=np.float64).tolist(),
                        "cache_hit": s.cache_hit}))

    def summary_of(r, m, cur):
        return {
            "step": r.step, "frame": r.frame_index,
            "is_kf": r.is_keyframe,
            "n_pts": r.points, "n_kf": r.keyframes,
            "avg_quality": r.avg_quality,
            "cur_pose": (np.asarray(cur.pose, dtype=np.float64)
                         if cur is not None else np.eye(4)),
            "pos": np.ascontiguousarray(np.array(
                [p._pt_3d for p in m.points], dtype=np.float32)
                .reshape(-1, 3)),
            "colors": np.ascontiguousarray(np.array(
                [(p.color if p.color is not None
                  else np.full(3, 200, np.uint8))
                 for p in m.points], dtype=np.uint8).reshape(-1, 3)),
            "K": np.asarray(s.K, dtype=np.float64).tolist(),
            "W": s.CAP_W, "H": s.CAP_H,
            "frustum_ids": [f.frame_id for f in m.keyframes],
            "frustum_poses": np.ascontiguousarray(np.array(
                [np.asarray(f.pose, dtype=np.float64)
                 for f in m.keyframes],
                dtype=np.float64).reshape(-1, 4, 4)),
            "frustum_quality": [float(f.tracking_quality)
                                for f in m.keyframes],
            "profile": (s.profile_summary() or {}),
        }

    running = True
    shipped_cps = 0
    while running:
        try:
            cmd = recv.get(timeout=0.5)
            if not cmd:
                continue
        except Exception:
            continue
        op = cmd[0]
        if op == "step":
            r = s.step()
            if r is None:
                send.put(("eos", {"step": s.step_count,
                                  "kfs": len(s.global_map.keyframes),
                                  "pts": len(s.global_map.points)}))
                running = False
                break
            if r.capture is not None:
                slots[slot][:] = r.capture
            m = s.global_map
            cur = getattr(s.tracker, "cur_frame", None)
            send.put(("step", (summary_of(r, m, cur), names[slot])))
            # The step's own CHECKPOINT_EVERY captures ship too (the
            # timeline's automatic checkpoints).
            while len(s.checkpoints) > shipped_cps:
                send.put(("checkpoint", s.checkpoints[shipped_cps]))
                shipped_cps += 1
            slot = 1 - slot
        elif op == "set_pins":
            from slam.utils.overrides import apply_override
            applied = 0
            for name, value in cmd[1]:
                try:
                    apply_override(name, str(value))
                    applied += 1
                except (KeyError, ValueError):
                    pass
            send.put(("pins_applied", applied))
        elif op == "set_options":
            branch = s.apply_live_options(SlamOptions(**cmd[1]))
            send.put(("options_applied", branch))
        elif op == "profile":
            send.put(("profile", s.profile_summary() or {}))
        elif op == "checkpoint":
            send.put(("checkpoint", s.capture_checkpoint()))
        elif op == "rewind":
            send.put(("rewound", s.restore_checkpoint(cmd[1])))
        elif op == "save":
            s.save()
            send.put(("saved",))
        else:
            running = False
    try:
        s.close()
    except Exception:
        pass
    for shm in shms:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
    send.put(("terminated",))
    sys.stdout.flush()


if __name__ == "__main__":
    print("m7 step worker module (the parent spawns worker_main)")

"""The step-process driver: the UI-side handle to the SLAM worker
(M7). Non-blocking by construction: request_step returns
immediately; drain() polls the result queue at the editor's cadence.

The summary dict per step carries the geometry snapshot (positions,
colors, frustums, the current pose) + the counters; the camera frame
rides the worker's shm slot pair (the slot name per step)."""
import multiprocessing as mp
import os

import numpy as np

from slam.ui import step_worker


class StepProcess:
    """The spawned pipeline. The constructor is non-blocking: the
    worker sends ("ready", meta) once the camera slots exist."""

    def __init__(self, spec, options, repo_dir=None):
        self.repo_dir = repo_dir or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        ctx = mp.get_context("spawn")
        self.recv = ctx.Queue()
        self.send = ctx.Queue()
        self.proc = ctx.Process(
            target=step_worker.worker_main,
            args=(self.send,
                  self.recv,
                  spec,
                  dict(options.__dict__),
                  self.repo_dir),
            daemon=False)
        self.proc.start()
        meta = self.recv.get(timeout=60)
        assert meta[0] == "ready", meta
        _, self.camera_meta = meta
        self.K = (np.asarray(self.camera_meta["K"], dtype=np.float64)
                  if self.camera_meta.get("K") is not None else None)
        self.cache_hit = self.camera_meta.get("cache_hit")

    def request_step(self):
        self.send.put(("step",))

    def set_pins(self, pairs):
        self.send.put(("set_pins", list(pairs)))

    def set_options(self, opts_dict):
        self.send.put(("set_options", dict(opts_dict)))

    def checkpoint(self):
        self.send.put(("checkpoint",))

    def rewind(self, cp):
        self.send.put(("rewind", cp))

    def save(self):
        self.send.put(("save",))

    def drain(self, timeout=0.0):
        """Non-blocking (or briefly blocking) result poll; returns a
        list of (op, payload)."""
        out = []
        while True:
            try:
                out.append(self.recv.get(
                    timeout=timeout if not out else 0.0))
                timeout = 0.0
            except Exception:
                break
        # Normalize to (op, payload): some ops carry no payload.
        return [(i[0], i[1] if len(i) > 1 else None)
                if isinstance(i, tuple) else (None, None)
                for i in out]

    def wait_for(self, needle, timeout=1.0):
        """A one-shot result pull for a specific op (the headless
        tests)."""
        import time
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            for op, payload in self.drain(timeout=0.1):
                if op == needle:
                    return payload
                if op == "terminated":
                    return None
        return None

    def close(self, timeout=5.0):
        try:
            self.send.put(("close",))
        except Exception:
            pass
        self.proc.join(timeout=timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=2.0)

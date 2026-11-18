"""The shared constants-override resolver.

The benchmarks harness's --override machinery, generalized for the
editor's Advanced tab: resolve a flag NAME across the flag modules
(the origin module wins over its star-importers — the trap documented
below), set the origin's attribute, and read the current/default
values. The mechanism's contract is measured: consumers read the
constants through the module alias at call time, so a setattr reaches
the pipeline live.

The resolution order is LOAD-BEARING: modules that STAR-IMPORT another
flag module (tracker star-imports matcher AND utils; matcher
star-imports constants AND utils) hold COPY bindings — setattr on the
copy would never reach the machinery that reads the origin's own
global. Origins resolve before their star-importers: constants, then
utils, then matcher, then tracker. utils's OWN flags are defined in
it - its star-imported constant copies are stale, but the validator
reads constants through the module alias. The copy-sync covers the
rest of the call-time readers (the harness's pre-run applies were
covered by import order: the overrides set the origin before the
machinery modules were first imported).
"""
import sys

import slam.utils.constants as const
import slam.features.matcher as matcher_mod
import slam.features.aliked_backend as aliked_backend
import slam.core.tracker as tracker_mod
import slam.core.intake as intake_mod
import slam.loop_closure as loop_closure_mod
import slam.ba.bundle_adjustment_g2o as ba_mod
import slam.utils.utils as utils_mod

flag_modules = (const, utils_mod, matcher_mod, tracker_mod,
                intake_mod, loop_closure_mod, ba_mod, aliked_backend)


def resolve_module(name):
    """The ORIGIN module carrying `name` (None when unresolvable)."""
    return next((m for m in flag_modules if hasattr(m, name)), None)


def current_value(name):
    holder = resolve_module(name)
    return None if holder is None else getattr(holder, name)


def apply_override(name, value):
    """setattr on the origin module with the trap-aware type
    conversion, PLUS the import-copy sync: the star-importing
    machinery modules (tracker/matcher/map/ba/utils hold
    `from slam.utils.constants import *` copies bound at import
    time) read their OWN globals at call time - a copy whose current
    value equals the pre-override one follows the setattr. returns
    (module, new_value) or None."""
    holder = resolve_module(name)
    if holder is None:
        raise KeyError(
            f"{name!r} is not a sweepable flag (slam.utils.constants "
            f"or a machinery module)")
    current = getattr(holder, name)
    try:
        parsed = type(current)(value.strip())
    except ValueError:
        raise ValueError(
            f"cannot parse {value!r} as {type(current).__name__} "
            f"for {name}")
    setattr(holder, name, parsed)
    _sync_copy_holders(name, current, parsed, holder)
    return holder, parsed


def _sync_copy_holders(name, was, new, origin):
    """The editor's live-pin contract: every loaded slam module whose
    global `name` still holds the OLD value gets the new one. Values
    not equal to `was` are left alone (they are either the origin
    itself or an unrelated binding)."""
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("slam") or mod is origin:
            continue
        try:
            cur = getattr(mod, name)
        except AttributeError:
            continue
        if type(cur) is type(was) and cur == was and cur != new:
            try:
                setattr(mod, name, new)
            except (AttributeError, TypeError):
                continue


def apply_overrides(pairs) -> None:
    """The benchmarks --override API (NAME=VALUE pairs), unchanged."""
    for pair in pairs:
        name, sep, value = pair.partition("=")
        name = name.strip()
        if not sep:
            raise SystemExit(
                f"[bench] --override expects NAME=VALUE of a sweepable "
                f"flag; got {pair!r}")
        was = current_value(name)
        try:
            holder, parsed = apply_override(name, value)
        except KeyError as e:
            raise SystemExit(f"[bench] {e}")
        except ValueError as e:
            raise SystemExit(f"[bench] {e}")
        print(f"[bench] override: {name} = {parsed} "
              f"(was {was}, in {holder.__name__})", flush=True)

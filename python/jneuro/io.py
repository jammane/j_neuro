"""JSON helpers for mutation profiles and run logs (spec/09). Nets are binary: see jnet.py."""
from __future__ import annotations

import json

from .jnet import SPEC_VERSION as _SV, spec_version_ok
from .types import ValidationError

SPEC_VERSION = "%d.%d.%d" % _SV
PROFILE_FORMAT = "j_neuro.profile"
RUNLOG_FORMAT = "j_neuro.runlog"


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def version_compatible(ver: str) -> bool:
    parts = ver.split(".")
    try:
        return len(parts) >= 2 and spec_version_ok(int(parts[0]), int(parts[1]))
    except ValueError:
        return False


def _check_version(d, fmt):
    if not isinstance(d, dict) or d.get("format") != fmt:
        raise ValidationError(f"format must be {fmt!r}")
    ver = d.get("spec_version")
    if not isinstance(ver, str) or not version_compatible(ver):
        raise ValidationError(f"unsupported spec_version {ver!r}")


def profile_to_json(profile) -> str:
    return _dumps(profile.to_dict())


def profile_from_json(text: str):
    from .mutate import Profile
    d = json.loads(text)
    _check_version(d, PROFILE_FORMAT)
    return Profile.from_dict(d)

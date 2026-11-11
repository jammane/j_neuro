"""Enumerations (spec/01, spec/03, spec/04). Member order is normative: it defines what an
RNG choice index means."""
from enum import IntEnum


class Priority(IntEnum):
    SPEED = 0
    MEMORY = 1
    BALANCED = 2


class DepthType(IntEnum):
    U16 = 0
    U32 = 1
    U64 = 2

    @property
    def d_max(self) -> int:
        return (1 << (16, 32, 64)[self]) - 1


class Inversion(IntEnum):
    NONE = 0
    BITNOT = 1
    RECIP = 2


class VOp(IntEnum):
    MUL = 0
    DIV = 1
    ADD = 2
    SUB = 3


class AggOp(IntEnum):
    MEAN = 0
    MEDIAN = 1
    MIN = 2
    MAX = 3


class NeuronType(IntEnum):
    V = 0
    S = 1
    M = 2


class ValidationError(ValueError):
    """A genome or parameter set violates the spec."""


IDENTITY_WEIGHT = {VOp.MUL: 1.0, VOp.DIV: 1.0, VOp.ADD: 0.0, VOp.SUB: 0.0}

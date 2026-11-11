"""Net data model (spec/03)."""
from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import List, Optional, Union

from .types import AggOp, DepthType, Inversion, Priority, VOp


@dataclass
class VNeuron:
    input: Optional[int]
    inversion: Inversion
    op: VOp
    weight: float


@dataclass
class SNeuron:
    input: Optional[int]
    op: AggOp
    k: int


@dataclass
class MNeuron:
    op: AggOp
    slots: List[Optional[int]]


Neuron = Union[VNeuron, SNeuron, MNeuron]


@dataclass
class Node:
    birth: int
    born_gen: int
    depth: int
    neuron: Optional[Neuron] = None


class Net:
    def __init__(self, priority: Priority, depth_type: DepthType, in_qty_max: int,
                 n_inputs: int, n_outputs: int, nodes: List[Node], next_birth: int):
        self.priority = Priority(priority)
        self.depth_type = DepthType(depth_type)
        self.in_qty_max = in_qty_max
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes = nodes
        self.next_birth = next_birth
        self._state = None

    @property
    def d_max(self) -> int:
        return self.depth_type.d_max

    def is_input(self, i: int) -> bool:
        return i < self.n_inputs

    def is_output(self, i: int) -> bool:
        return self.n_inputs <= i < self.n_inputs + self.n_outputs

    def is_hidden(self, i: int) -> bool:
        return i >= self.n_inputs + self.n_outputs

    def copy(self) -> "Net":
        """Deep copy of the genome, in the reset state."""
        return Net(self.priority, self.depth_type, self.in_qty_max, self.n_inputs,
                   self.n_outputs, _copy.deepcopy(self.nodes), self.next_birth)

    # implemented in later milestones
    def step(self, inputs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError

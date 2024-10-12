import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class Params:
    @property
    def dirname(self) -> str:
        return os.path.join(*[f"{field}-{val}" for field, val in asdict(self).items()])


@dataclass(frozen=True, kw_only=True)
class LUCJParams(Params):
    connectivity: str  # options: all-to-all, linear, square, hex, heavy-hex
    n_reps: int | None
    with_final_orbital_rotation: bool


@dataclass(frozen=True, kw_only=True)
class UCCSDParams(Params):
    with_final_orbital_rotation: bool


@dataclass(frozen=True, kw_only=True)
class LUCJAnglesParams(Params):
    connectivity: str  # options: all-to-all, square, hex, heavy-hex
    n_reps: int | None
    with_final_orbital_rotation: bool
    n_givens_layers: int


@dataclass(frozen=True, kw_only=True)
class LBFGSBParams(Params):
    maxiter: int
    maxfun: int


@dataclass(frozen=True, kw_only=True)
class LinearMethodParams(Params):
    maxiter: int
    lindep: float
    epsilon: float
    ftol: float
    gtol: float
    regularization: float
    variation: float
    optimize_regularization: bool
    optimize_variation: bool


@dataclass(frozen=True, kw_only=True)
class StochasticReconfigurationParams(Params):
    maxiter: int
    cond: float
    epsilon: float
    gtol: float
    regularization: float
    variation: float
    optimize_regularization: bool
    optimize_variation: bool

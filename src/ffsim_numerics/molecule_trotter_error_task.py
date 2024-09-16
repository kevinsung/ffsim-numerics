import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ffsim_numerics.double_factorized_trotter_gate_count_task import (
    DoubleFactorizedTrotterGateCountTask,
)
from ffsim_numerics.double_factorized_trotter_sim_task import (
    DoubleFactorizedTrotterSimTask,
)
from ffsim_numerics.exact_time_evo_task import ExactTimeEvolutionTask

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))


@dataclass(frozen=True, kw_only=True)
class MoleculeTrotterErrorTask:
    molecule_basename: str
    bond_distance: float
    time: float
    n_steps: int
    order: int
    initial_state: str  # options: hartree-fock, random
    entropy: int | None = None
    spawn_index: int = 0

    def __post_init__(self):
        assert self.initial_state in ("hartree-fock", "random")

    @property
    def dirpath(self) -> Path:
        path = (
            Path(self.molecule_basename)
            / f"bond_distance-{self.bond_distance:.2f}"
            / f"time-{self.time:.1f}"
            / f"n_steps-{self.n_steps}"
            / f"order-{self.order}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"root_seed-{self.entropy}"
            path /= f"spawn_index-{self.spawn_index}"
        return path


def run_molecule_trotter_error_task(
    task: MoleculeTrotterErrorTask,
    *,
    data_dir: Path,
    overwrite: bool = True,
) -> MoleculeTrotterErrorTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.pickle"
    if (not overwrite) and os.path.exists(result_filepath):
        logger.info(f"Data for {task} already exists. Skipping...")
        return task

    # Load exact state
    this_task = ExactTimeEvolutionTask(
        molecule_basename=task.molecule_basename,
        bond_distance=task.bond_distance,
        time=task.time,
        initial_state=task.initial_state,
        entropy=task.entropy,
        spawn_index=task.spawn_index,
    )
    filepath = DATA_ROOT / "exact_time_evo" / this_task.dirpath / "result.npy"
    with open(filepath, "rb") as f:
        exact_state = np.load(filepath)

    # Compute error
    this_task = DoubleFactorizedTrotterSimTask(
        molecule_basename=task.molecule_basename,
        bond_distance=task.bond_distance,
        time=task.time,
        n_steps=task.n_steps,
        order=task.order,
        initial_state=task.initial_state,
        entropy=task.entropy,
        spawn_index=task.spawn_index,
    )
    filepath = (
        DATA_ROOT / "double_factorized_trotter_sim" / this_task.dirpath / "result.npy"
    )
    with open(filepath, "rb") as f:
        trotter_state = np.load(filepath)
    error = np.linalg.norm(trotter_state - exact_state)

    # Load gate count and depth
    this_task = DoubleFactorizedTrotterGateCountTask(
        molecule_basename=task.molecule_basename,
        bond_distance=task.bond_distance,
        time=task.time,
        n_steps=task.n_steps,
        order=task.order,
    )
    filepath = (
        DATA_ROOT
        / "double_factorized_trotter_gate_count"
        / this_task.dirpath
        / "result.json"
    )
    with open(filepath, "r") as f:
        gate_counts = json.load(f)

    result = (error, gate_counts["cx_count"], gate_counts["cx_depth"])

    # Save result to disk
    logger.info("Saving result to disk...")
    with open(result_filepath, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"{task} Done.")
    return task

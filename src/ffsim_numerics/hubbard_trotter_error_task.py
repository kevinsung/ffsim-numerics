import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ffsim_numerics.hubbard_time_evo_task import HubbardTimeEvolutionTask
from ffsim_numerics.hubbard_trotter_gate_count_task import HubbardTrotterGateCountTask
from ffsim_numerics.hubbard_trotter_sim_task import HubbardTrotterSimTask

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))


@dataclass(frozen=True, kw_only=True)
class HubbardTrotterErrorTask:
    norb_x: int
    norb_y: int
    tunneling: float
    interaction: float
    chemical_potential: float
    nearest_neighbor_interaction: float
    periodic_x: bool
    periodic_y: bool
    filling_denominator: int
    time: float
    n_steps: int
    order: int
    initial_state: str  # options: one-body, random
    entropy: int | None = None
    spawn_index: int = 0

    def __post_init__(self):
        assert (self.norb_x * self.norb_y) % self.filling_denominator == 0
        assert self.initial_state in ("one-body", "random")

    @property
    def dirpath(self) -> Path:
        path = (
            Path("hubbard")
            / f"{self.norb_x}x{self.norb_y}"
            / f"tunneling-{self.tunneling}"
            / f"interaction-{self.interaction}"
            / f"chemical_potential-{self.chemical_potential}"
            / f"nearest_neighbor_interaction-{self.nearest_neighbor_interaction}"
            / f"periodic_x-{self.periodic_x}"
            / f"periodic_y-{self.periodic_y}"
            / f"filling_denominator-{self.filling_denominator}"
            / f"time-{self.time:.1f}"
            / f"n_steps-{self.n_steps}"
            / f"order-{self.order}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"root_seed-{self.entropy}"
            path /= f"spawn_index-{self.spawn_index}"
        return path


def run_hubbard_trotter_error_task(
    task: HubbardTrotterErrorTask,
    *,
    data_dir: Path,
    overwrite: bool = True,
) -> HubbardTrotterErrorTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.pickle"
    if (not overwrite) and os.path.exists(result_filepath):
        logger.info(f"Data for {task} already exists. Skipping...")
        return task

    # Load exact state
    this_task = HubbardTimeEvolutionTask(
        norb_x=task.norb_x,
        norb_y=task.norb_y,
        tunneling=task.tunneling,
        interaction=task.interaction,
        chemical_potential=task.chemical_potential,
        nearest_neighbor_interaction=task.nearest_neighbor_interaction,
        periodic_x=task.periodic_x,
        periodic_y=task.periodic_y,
        filling_denominator=task.filling_denominator,
        time=task.time,
        initial_state=task.initial_state,
        entropy=task.entropy,
        spawn_index=task.spawn_index,
    )
    filepath = DATA_ROOT / "hubbard_time_evo" / this_task.dirpath / "result.npy"
    with open(filepath, "rb") as f:
        exact_state = np.load(filepath)

    # Compute error
    this_task = HubbardTrotterSimTask(
        norb_x=task.norb_x,
        norb_y=task.norb_y,
        tunneling=task.tunneling,
        interaction=task.interaction,
        chemical_potential=task.chemical_potential,
        nearest_neighbor_interaction=task.nearest_neighbor_interaction,
        periodic_x=task.periodic_x,
        periodic_y=task.periodic_y,
        filling_denominator=task.filling_denominator,
        time=task.time,
        n_steps=task.n_steps,
        order=task.order,
        initial_state=task.initial_state,
        entropy=task.entropy,
        spawn_index=task.spawn_index,
    )
    filepath = DATA_ROOT / "hubbard_trotter_sim" / this_task.dirpath / "result.npy"
    with open(filepath, "rb") as f:
        trotter_state = np.load(filepath)
    error = np.linalg.norm(trotter_state - exact_state)

    # Load gate count and depth
    this_task = HubbardTrotterGateCountTask(
        norb_x=task.norb_x,
        norb_y=task.norb_y,
        periodic_x=task.periodic_x,
        periodic_y=task.periodic_y,
        n_steps=task.n_steps,
        order=task.order,
    )
    filepath = (
        DATA_ROOT / "hubbard_trotter_gate_count" / this_task.dirpath / "result.json"
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

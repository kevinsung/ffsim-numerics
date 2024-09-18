import dataclasses
import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class HubbardTrotterSimTask:
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


def run_hubbard_trotter_sim_task(
    task: HubbardTrotterSimTask,
    *,
    data_dir: Path,
    overwrite: bool = True,
) -> HubbardTrotterSimTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.npy"
    if (not overwrite) and os.path.exists(result_filepath):
        logger.info(f"Data for {task} already exists. Skipping...")
        return task

    # Construct Fermi-Hubbard Hamiltonian
    op = ffsim.fermi_hubbard_2d(
        norb_x=task.norb_x,
        norb_y=task.norb_y,
        tunneling=task.tunneling,
        interaction=task.interaction,
        chemical_potential=task.chemical_potential,
        nearest_neighbor_interaction=task.nearest_neighbor_interaction,
        periodic_x=task.periodic_x,
        periodic_y=task.periodic_y,
    )
    norb = task.norb_x * task.norb_y
    nocc = norb // task.filling_denominator
    nelec = (nocc, nocc)
    hamiltonian = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)
    assert np.all(np.isreal(hamiltonian.one_body_tensor))
    hamiltonian = dataclasses.replace(
        hamiltonian, one_body_tensor=hamiltonian.one_body_tensor.real
    )

    # Get initial state
    match task.initial_state:
        case "one-body":
            _, orbital_rotation = np.linalg.eigh(hamiltonian.one_body_tensor)
            reference_state = ffsim.slater_determinant(
                norb, (range(nocc), range(nocc)), orbital_rotation=orbital_rotation
            )
        case "random":
            parent_rng = np.random.default_rng(task.entropy)
            child_rng = parent_rng.spawn(task.spawn_index + 1)[-1]
            reference_state = ffsim.random.random_state_vector(
                ffsim.dim(norb, nelec), seed=child_rng
            )

    # Apply Trotter evolution
    logger.info("Applying Trotter evolution...")
    t0 = timeit.default_timer()
    result = ffsim.simulate_trotter_diag_coulomb_split_op(
        reference_state,
        hamiltonian,
        time=task.time,
        norb=norb,
        nelec=nelec,
        n_steps=task.n_steps,
        order=task.order,
    )
    t1 = timeit.default_timer()
    logger.info(f"Done applying time evolution in {t1 - t0} seconds.")

    # Save result to disk
    logger.info("Saving result to disk...")
    with open(result_filepath, "wb") as f:
        np.save(f, result)

    logger.info(f"{task} Done.")
    return task

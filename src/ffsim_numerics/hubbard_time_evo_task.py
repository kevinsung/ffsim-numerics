import dataclasses
import logging
import os
import timeit
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
import scipy.sparse.linalg

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class HubbardTimeEvolutionTask:
    norb_x: int
    norb_y: int
    tunneling: float
    interaction: float
    chemical_potential: float
    nearest_neighbor_interaction: float
    periodic: bool
    filling_denominator: int
    time: float
    initial_state: str  # options: random
    seed: int | None = None

    def __post_init__(self):
        assert (self.norb_x * self.norb_y) % self.filling_denominator == 0
        assert self.initial_state in ["random"]

    @property
    def dirpath(self) -> Path:
        path = (
            Path("hubbard")
            / f"{self.norb_x}x{self.norb_y}"
            / f"tunneling-{self.tunneling}"
            / f"interaction-{self.interaction}"
            / f"chemical_potential-{self.chemical_potential}"
            / f"nearest_neighbor_interaction-{self.nearest_neighbor_interaction}"
            / f"periodic-{self.periodic}"
            / f"filling_denominator-{self.filling_denominator}"
            / f"time-{self.time:.1f}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"seed-{self.seed}"
        return path


def run_hubbard_time_evolution_task(
    task: HubbardTimeEvolutionTask,
    *,
    data_dir: Path,
    overwrite: bool = True,
) -> HubbardTimeEvolutionTask:
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
        periodic=task.periodic,
    )
    norb = task.norb_x * task.norb_y
    nocc = norb // task.filling_denominator
    nelec = (nocc, nocc)
    hamiltonian = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)
    assert np.all(np.isreal(hamiltonian.one_body_tensor))
    hamiltonian = dataclasses.replace(
        hamiltonian, one_body_tensor=hamiltonian.one_body_tensor.real
    )

    # Initialize linear operator
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)

    # Get initial state
    match task.initial_state:
        case "random":
            reference_state = ffsim.random.random_state_vector(
                ffsim.dim(norb, nelec), seed=task.seed
            )

    # Apply time evolution
    logger.info("Computing Hamiltonian trace...")
    t0 = timeit.default_timer()
    trace = ffsim.trace(hamiltonian, norb=norb, nelec=nelec)
    t1 = timeit.default_timer()
    logger.info(f"Done computing trace in {t1 - t0} seconds.")
    logger.info("Applying time evolution...")
    t0 = timeit.default_timer()
    result = scipy.sparse.linalg.expm_multiply(
        -1j * task.time * linop, reference_state, traceA=-1j * task.time * trace
    )
    t1 = timeit.default_timer()
    logger.info(f"Done applying time evolution in {t1 - t0} seconds.")

    fidelity = np.abs(np.vdot(result, reference_state))
    logger.info(f"Fidelity with reference state: {fidelity}.")

    # Save result to disk
    logger.info("Saving result to disk...")
    with open(result_filepath, "wb") as f:
        np.save(f, result)

    logger.info(f"{task} Done.")
    return task

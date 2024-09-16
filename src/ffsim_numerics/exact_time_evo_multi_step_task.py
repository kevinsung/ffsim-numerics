import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
import scipy.sparse.linalg

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ExactTimeEvoMultiStepTask:
    molecule_basename: str
    bond_distance: float
    time_step: float
    n_steps: int
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
            / f"time_step-{self.time_step:.1f}"
            / f"n_steps-{self.n_steps}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"root_seed-{self.entropy}"
            path /= f"spawn_index-{self.spawn_index}"
        return path


def run_exact_time_evo_multi_step_task(
    task: ExactTimeEvoMultiStepTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> ExactTimeEvoMultiStepTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.npy"
    if (not overwrite) and os.path.exists(result_filepath):
        logger.info(f"Data for {task} already exists. Skipping...")
        return task

    # Get molecular data and molecular Hamiltonian
    molecule_filepath = (
        molecules_catalog_dir
        / "data"
        / "molecular_data"
        / f"{task.molecule_basename}_d-{task.bond_distance:.2f}.json.xz"
    )
    mol_data = ffsim.MolecularData.from_json(molecule_filepath, compression="lzma")
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # Initialize linear operator
    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    # Get initial state
    match task.initial_state:
        case "hartree-fock":
            reference_state = ffsim.hartree_fock_state(norb, nelec)
        case "random":
            parent_rng = np.random.default_rng(task.entropy)
            child_rng = parent_rng.spawn(task.spawn_index + 1)[-1]
            reference_state = ffsim.random.random_state_vector(
                ffsim.dim(norb, nelec), seed=child_rng
            )

    # Apply time evolution
    logger.info("Computing Hamiltonian trace...")
    t0 = timeit.default_timer()
    trace = ffsim.trace(mol_hamiltonian, norb=norb, nelec=nelec)
    t1 = timeit.default_timer()
    logger.info(f"Done computing trace in {t1 - t0} seconds.")
    logger.info("Applying time evolution...")
    t0 = timeit.default_timer()
    result = scipy.sparse.linalg.expm_multiply(
        -1j * linop,
        reference_state,
        start=0.0,
        stop=task.time_step * task.n_steps,
        num=task.n_steps + 1,
        traceA=-1j * trace,
    )
    t1 = timeit.default_timer()
    logger.info(f"Done applying time evolution in {t1 - t0} seconds.")

    fidelities = [float(np.abs(np.vdot(vec, reference_state))) for vec in result]
    logger.info(f"Fidelities with reference state: {fidelities}.")

    # Save result to disk
    logger.info("Saving result to disk...")
    with open(result_filepath, "wb") as f:
        np.save(f, result)

    logger.info(f"{task} Done.")
    return task

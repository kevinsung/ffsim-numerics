import logging
import scipy.sparse.linalg
import os
import timeit
from pathlib import Path
from dataclasses import dataclass

import ffsim
import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ExactTimeEvolutionTask:
    molecule_basename: str
    bond_distance: float
    time: float
    initial_state: str  # options: hartree-fock, random
    seed: int | None = None

    def __post_init__(self):
        assert self.initial_state in ("hartree-fock", "random")

    @property
    def dirpath(self) -> Path:
        path = (
            Path(self.molecule_basename)
            / f"bond_distance-{self.bond_distance:.2f}"
            / f"time-{self.time:.1f}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"seed-{self.seed}"
        return path


def run_exact_time_evolution_task(
    task: ExactTimeEvolutionTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> ExactTimeEvolutionTask:
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
            reference_state = ffsim.random.random_state_vector(
                ffsim.dim(norb, nelec), seed=task.seed
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
        -1j * task.time * linop,
        reference_state,
        traceA=-1j * task.time * trace,
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

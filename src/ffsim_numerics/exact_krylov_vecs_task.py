import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.sparse.linalg

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ExactKrylovVecsTask:
    molecule_basename: str
    bond_distance: float
    n_steps: int
    time_step: float
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
            / f"krylov_n_steps-{self.n_steps}"
            / f"time_step-{self.time_step:.3f}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"root_seed-{self.entropy}"
            path /= f"spawn_index-{self.spawn_index}"
        return path


def run_exact_krylov_vecs_task(
    task: ExactKrylovVecsTask,
    *,
    data_dir: Path,
    overwrite: bool = True,
) -> ExactKrylovVecsTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.npy"
    if (not overwrite) and os.path.exists(result_filepath):
        logger.info(f"Data for {task} already exists. Skipping...")
        return task

    # Get molecular data and molecular Hamiltonian
    molecule_filepath = (
        Path("molecular_data")
        / task.molecule_basename
        / f"{task.molecule_basename}_d-{task.bond_distance:.5f}.json.xz"
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

    # Construct Krylov states
    logger.info("Computing Hamiltonian trace...")
    t0 = timeit.default_timer()
    trace = ffsim.trace(mol_hamiltonian, norb=norb, nelec=nelec)
    t1 = timeit.default_timer()
    logger.info(f"Done computing trace in {t1 - t0} seconds.")
    logger.info("Constructing Krylov states...")
    krylov_vecs = np.zeros((task.n_steps + 1, ffsim.dim(norb, nelec)), dtype=complex)
    krylov_vecs[0] = reference_state
    vec = reference_state
    t0 = timeit.default_timer()
    for i in range(1, task.n_steps + 1):
        vec = scipy.sparse.linalg.expm_multiply(
            -1j * task.time_step * linop,
            vec,
            traceA=-1j * task.time_step * trace,
        )
        krylov_vecs[i] = vec
    t1 = timeit.default_timer()
    logger.info(f"Done constructing Krylov states in {t1 - t0} seconds.")

    # Save result to disk
    logger.info("Saving result to disk...")
    with open(result_filepath, "wb") as f:
        np.save(f, krylov_vecs)

    logger.info(f"{task} Done.")
    return task

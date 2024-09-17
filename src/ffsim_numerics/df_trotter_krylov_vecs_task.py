import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DoubleFactorizedTrotterKrylovVecsTask:
    molecule_basename: str
    bond_distance: float
    krylov_n_steps: int
    time_step: float
    trotter_n_steps: int
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
            / f"krylov_n_steps-{self.krylov_n_steps}"
            / f"time_step-{self.time_step:.1f}"
            / f"trotter_n_steps-{self.trotter_n_steps}"
            / f"order-{self.order}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"root_seed-{self.entropy}"
            path /= f"spawn_index-{self.spawn_index}"
        return path


def run_double_factorized_trotter_krylov_vecs_task(
    task: DoubleFactorizedTrotterKrylovVecsTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> DoubleFactorizedTrotterKrylovVecsTask:
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

    # Get double-factorized Hamiltonian
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )

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
    logger.info("Constructing Krylov states...")
    krylov_vecs = np.zeros(
        (task.krylov_n_steps + 1, ffsim.dim(norb, nelec)), dtype=complex
    )
    krylov_vecs[0] = reference_state
    vec = reference_state
    t0 = timeit.default_timer()
    for i in range(1, task.krylov_n_steps + 1):
        vec = ffsim.simulate_trotter_double_factorized(
            vec,
            df_hamiltonian,
            time=task.time_step,
            norb=norb,
            nelec=nelec,
            n_steps=task.trotter_n_steps,
            order=task.order,
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

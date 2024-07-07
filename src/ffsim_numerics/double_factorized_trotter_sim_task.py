import logging
import os
import timeit
from pathlib import Path
from dataclasses import dataclass

import ffsim
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DoubleFactorizedTrotterSimTask:
    molecule_basename: str
    bond_distance: float
    time: float
    n_steps: int
    order: int
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
            / f"n_steps-{self.n_steps}"
            / f"order-{self.order}"
            / f"initial_state-{self.initial_state}"
        )
        if self.initial_state == "random":
            path /= f"seed-{self.seed}"
        return path


def run_double_factorized_trotter_sim_task(
    task: DoubleFactorizedTrotterSimTask,
    *,
    data_dir: Path,
    molecules_catalogue_dir: Path,
    overwrite: bool = True,
) -> DoubleFactorizedTrotterSimTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.npy"
    if (not overwrite) and os.path.exists(result_filepath):
        logger.info(f"Data for {task} already exists. Skipping...")
        return task

    # Get molecular data and molecular Hamiltonian
    molecule_filepath = (
        molecules_catalogue_dir
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
            reference_state = ffsim.random.random_state_vector(
                ffsim.dim(norb, nelec), seed=task.seed
            )

    # Apply Trotter evolution
    logger.info("Applying Trotter evolution...")
    t0 = timeit.default_timer()
    result = ffsim.simulate_trotter_double_factorized(
        reference_state,
        df_hamiltonian,
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

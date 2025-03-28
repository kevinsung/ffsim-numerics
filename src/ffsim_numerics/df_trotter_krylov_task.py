import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
import scipy.sparse.linalg
from pyscf.lib.linalg_helper import safe_eigh

from ffsim_numerics.df_trotter_krylov_vecs_task import (
    DoubleFactorizedTrotterKrylovVecsTask,
)
from ffsim_numerics.util import krylov_matrix

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))


@dataclass(frozen=True, kw_only=True)
class DoubleFactorizedTrotterKrylovTask:
    molecule_basename: str
    bond_distance: float
    krylov_n_steps: int
    time_step: float
    trotter_n_steps: int
    order: int
    initial_state: str  # options: hartree-fock
    lindep: float

    def __post_init__(self):
        assert self.initial_state in ["hartree-fock"]

    @property
    def dirpath(self) -> Path:
        path = (
            Path(self.molecule_basename)
            / f"bond_distance-{self.bond_distance:.2f}"
            / f"krylov_n_steps-{self.krylov_n_steps}"
            / f"time_step-{self.time_step:.3f}"
            / f"trotter_n_steps-{self.trotter_n_steps}"
            / f"order-{self.order}"
            / f"initial_state-{self.initial_state}"
        )
        return path


def run_double_factorized_trotter_krylov_task(
    task: DoubleFactorizedTrotterKrylovTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> DoubleFactorizedTrotterKrylovTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.npy"
    overlap_mat_filepath = data_dir / task.dirpath / "overlap_mat.npy"
    hamiltonian_mat_filepath = data_dir / task.dirpath / "hamiltonian_mat.npy"
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
    logging.info(f"Hartree-Fock energy: {mol_data.hf_energy}")
    logging.info(f"FCI energy: {mol_data.fci_energy}")

    # Initialize linear operator
    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    # Load Krylov vectors
    this_task = DoubleFactorizedTrotterKrylovVecsTask(
        molecule_basename=task.molecule_basename,
        bond_distance=task.bond_distance,
        krylov_n_steps=task.krylov_n_steps,
        time_step=task.time_step,
        trotter_n_steps=task.trotter_n_steps,
        order=task.order,
        initial_state=task.initial_state,
    )
    filepath = DATA_ROOT / "df_trotter_krylov_vecs" / this_task.dirpath / "result.npy"
    with open(filepath, "rb") as f:
        krylov_vecs = np.load(filepath)

    # Check vector norm
    norms = np.linalg.norm(krylov_vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    # Compute overlap and Hamiltonian matrices
    n_vecs, dim = krylov_vecs.shape
    assert n_vecs == task.krylov_n_steps + 1
    eye = scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=lambda x: x, dtype=complex
    )
    logger.info("Computing overlap matrix...")
    t0 = timeit.default_timer()
    overlap_mat = krylov_matrix(eye, krylov_vecs)
    t1 = timeit.default_timer()
    logger.info(f"Done computing overlap matrix in {t1 - t0} seconds.")
    logger.info("Computing Hamiltonian matrix...")
    t0 = timeit.default_timer()
    hamiltonian_mat = krylov_matrix(linop, krylov_vecs)
    t1 = timeit.default_timer()
    logger.info(f"Done computing Hamiltonian matrix in {t1 - t0} seconds.")
    logger.info("Saving overlap and Hamiltonian matrices to disk...")
    with open(overlap_mat_filepath, "wb") as f:
        np.save(f, overlap_mat)
    with open(hamiltonian_mat_filepath, "wb") as f:
        np.save(f, hamiltonian_mat)

    # Compute ground state energies
    logger.info("Computing ground state energies...")
    ground_energies = np.zeros(n_vecs)
    for i in range(n_vecs):
        eigs, _, _ = safe_eigh(
            hamiltonian_mat[: i + 1, : i + 1],
            overlap_mat[: i + 1, : i + 1],
            lindep=task.lindep,
        )
        ground_energies[i] = eigs[0]
    logger.info(f"Ground energies: {ground_energies}")
    logger.info("Saving ground energies to disk...")
    with open(result_filepath, "wb") as f:
        np.save(f, ground_energies)

    logger.info(f"{task} Done.")
    return task

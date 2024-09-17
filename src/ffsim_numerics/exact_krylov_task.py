import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
import scipy.sparse.linalg
from pyscf.lib.linalg_helper import safe_eigh

from ffsim_numerics.exact_time_evo_multi_step_task import ExactTimeEvoMultiStepTask

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))


@dataclass(frozen=True, kw_only=True)
class ExactKrylovTask:
    molecule_basename: str
    bond_distance: float
    time_step: float
    n_steps: int
    initial_state: str  # options: hartree-fock
    lindep: float

    def __post_init__(self):
        assert self.initial_state in ["hartree-fock"]

    @property
    def dirpath(self) -> Path:
        path = (
            Path(self.molecule_basename)
            / f"bond_distance-{self.bond_distance:.2f}"
            / f"time_step-{self.time_step:.1f}"
            / f"n_steps-{self.n_steps}"
            / f"initial_state-{self.initial_state}"
            / f"lindep-{self.lindep}"
        )
        return path


def run_exact_krylov_task(
    task: ExactKrylovTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> ExactKrylovTask:
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
    logging.info(f"Hartree-Fock energy: {mol_data.hf_energy}")
    logging.info(f"FCI energy: {mol_data.fci_energy}")

    # Initialize linear operator
    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    # Load Krylov vectors
    this_task = ExactTimeEvoMultiStepTask(
        molecule_basename=task.molecule_basename,
        bond_distance=task.bond_distance,
        time_step=task.time_step,
        n_steps=task.n_steps,
        initial_state=task.initial_state,
    )
    filepath = (
        DATA_ROOT / "exact_time_evo_multi_step" / this_task.dirpath / "result.npy"
    )
    with open(filepath, "rb") as f:
        krylov_vecs = np.load(filepath)

    # Check vector norm
    norms = np.linalg.norm(krylov_vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    # Compute ground state energies
    n_vecs, dim = krylov_vecs.shape
    assert n_vecs == task.n_steps + 1
    eye = scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=lambda x: x, dtype=complex
    )
    ground_energies = np.zeros(n_vecs)
    for i in range(n_vecs):
        overlap_mat = krylov_matrix(eye, krylov_vecs[: i + 1])
        hamiltonian_mat = krylov_matrix(linop, krylov_vecs[: i + 1])
        eigs, _, _ = safe_eigh(hamiltonian_mat, overlap_mat, lindep=task.lindep)
        ground_energies[i] = eigs[0]
    logger.info(f"Ground energies: {ground_energies}")

    # Save result to disk
    logger.info("Saving result to disk...")
    with open(result_filepath, "wb") as f:
        np.save(f, ground_energies)

    logger.info(f"{task} Done.")
    return task


def krylov_matrix(
    observable: scipy.sparse.linalg.LinearOperator, krylov_vecs: np.ndarray
):
    n_vecs = len(krylov_vecs)
    mat = np.zeros((n_vecs, n_vecs), dtype=complex)
    for i in range(n_vecs):
        mat[i, i] = np.vdot(krylov_vecs[i], observable @ krylov_vecs[i])
    for i, j in itertools.combinations(range(n_vecs), 2):
        mat[i, j] = np.vdot(krylov_vecs[i], observable @ krylov_vecs[j])
        mat[j, i] = mat[i, j].conjugate()
    return mat

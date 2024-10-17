import logging
import os
import pickle
import timeit
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
from ffsim.variational.util import orbital_rotation_to_parameters

from ffsim_numerics.params import LinearMethodParams, LUCJAnglesParams
from ffsim_numerics.util import (
    brickwork,
    interaction_pairs_spin_balanced,
    truncated_ucj_angles_op,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJAnglesLinearMethodTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJAnglesParams
    linear_method_params: LinearMethodParams

    @property
    def dirpath(self) -> Path:
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.2f}"
            )
            / self.lucj_params.dirpath
            / self.linear_method_params.dirpath
        )


def run_lucj_angles_linear_method_task(
    task: LUCJAnglesLinearMethodTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    bootstrap_task: LUCJAnglesLinearMethodTask | None = None,
    bootstrap_data_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJAnglesLinearMethodTask:
    logger.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filename = data_dir / task.dirpath / "result.pickle"
    info_filename = data_dir / task.dirpath / "info.pickle"
    data_filename = data_dir / task.dirpath / "data.pickle"
    if (
        (not overwrite)
        and os.path.exists(result_filename)
        and os.path.exists(info_filename)
        and os.path.exists(data_filename)
    ):
        logger.info(f"Data for {task} already exists. Skipping...\n")
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

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )
    num_num_interaction_pairs = (pairs_aa, pairs_ab)
    givens_interaction_pairs = list(brickwork(norb, task.lucj_params.n_givens_layers))
    n_reps = None

    # Define function that maps parameters to state vector
    def params_to_vec(x: np.ndarray) -> np.ndarray:
        operator = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
            x,
            norb=norb,
            n_reps=n_reps,
            num_num_interaction_pairs=num_num_interaction_pairs,
            givens_interaction_pairs=givens_interaction_pairs,
            with_final_givens_ansatz_op=task.lucj_params.with_final_orbital_rotation,
        )
        return ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    # Generate initial parameters
    if bootstrap_task is None:
        # use CCSD to initialize parameters
        op = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(
            mol_data.ccsd_t2,
            n_reps=task.lucj_params.n_reps,
            t1=mol_data.ccsd_t1
            if task.lucj_params.with_final_orbital_rotation
            else None,
            interaction_pairs=(pairs_aa, pairs_ab),
        )
        op = truncated_ucj_angles_op(op, n_layers=task.lucj_params.n_givens_layers)
        params = op.to_parameters()
        n_reps = op.n_reps
    else:
        bootstrap_result_filename = os.path.join(
            bootstrap_data_dir or data_dir, bootstrap_task.dirpath, "result.pickle"
        )
        with open(bootstrap_result_filename, "rb") as f:
            result = pickle.load(f)
            params = result.x
            # TODO this is incorrect for n_reps = None
            n_reps = task.lucj_params.n_reps
        if bootstrap_task.lucj_params.n_reps < task.lucj_params.n_reps:
            n_params = ffsim.UCJAnglesOpSpinBalanced.n_params(
                norb=norb,
                n_reps=task.lucj_params.n_reps,
                num_num_interaction_pairs=num_num_interaction_pairs,
                givens_interaction_pairs=givens_interaction_pairs,
                with_final_givens_ansatz_op=bootstrap_task.lucj_params.with_final_orbital_rotation,
            )
            params = np.concatenate([params, np.zeros(n_params - len(params))])
        if (
            task.lucj_params.with_final_orbital_rotation
            and not bootstrap_task.lucj_params.with_final_orbital_rotation
        ):
            params = np.concatenate([params, np.zeros(norb**2)])
            params[-(norb**2) :] = orbital_rotation_to_parameters(
                np.eye(norb, dtype=complex)
            )

    # Optimize ansatz
    logger.info(f"{task} Optimizing ansatz...\n")
    info = defaultdict(list)
    info["nit"] = 0
    info["regularization"] = [task.linear_method_params.regularization]
    info["variation"] = [task.linear_method_params.variation]

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        logger.info(f"Task {task} is on iteration {info['nit']}.\n")
        info["x"].append(intermediate_result.x)
        info["fun"].append(intermediate_result.fun)
        if hasattr(intermediate_result, "jac"):
            info["jac"].append(intermediate_result.jac)
        if hasattr(intermediate_result, "regularization"):
            info["regularization"].append(intermediate_result.regularization)
        if hasattr(intermediate_result, "variation"):
            info["variation"].append(intermediate_result.variation)
        # nit = info["nit"]
        # if nit < 10 or nit % 100 == 0:
        #     if hasattr(intermediate_result, "energy_mat"):
        #         info["energy_mat"].append((nit, intermediate_result.energy_mat))
        #     if hasattr(intermediate_result, "overlap_mat"):
        #         info["overlap_mat"].append((nit, intermediate_result.overlap_mat))
        info["nit"] += 1

    t0 = timeit.default_timer()
    result = ffsim.optimize.minimize_linear_method(
        params_to_vec,
        hamiltonian,
        x0=params,
        maxiter=task.linear_method_params.maxiter,
        lindep=task.linear_method_params.lindep,
        epsilon=task.linear_method_params.epsilon,
        ftol=task.linear_method_params.ftol,
        gtol=task.linear_method_params.gtol,
        regularization=task.linear_method_params.regularization,
        variation=task.linear_method_params.variation,
        optimize_regularization=task.linear_method_params.optimize_regularization,
        optimize_variation=task.linear_method_params.optimize_variation,
        callback=callback,
    )
    t1 = timeit.default_timer()
    logger.info(f"{task} Done optimizing ansatz in {t1 - t0} seconds.\n")

    logger.info(f"{task} Computing energy and other properties...\n")
    # Compute energy and other properties of final state vector
    if task.lucj_params.n_reps is None:
        op = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(
            mol_data.ccsd_t2,
            t1_amplitudes=mol_data.ccsd_t1
            if task.lucj_params.with_final_orbital_rotation
            else None,
        )
        n_reps = op.n_reps
    else:
        n_reps = task.lucj_params.n_reps
    operator = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
        result.x,
        norb=norb,
        n_reps=n_reps,
        num_num_interaction_pairs=num_num_interaction_pairs,
        givens_interaction_pairs=givens_interaction_pairs,
        with_final_givens_ansatz_op=task.lucj_params.with_final_orbital_rotation,
    )
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    energy = np.vdot(final_state, hamiltonian @ final_state).real
    np.testing.assert_allclose(energy, result.fun)

    error = energy - mol_data.fci_energy

    spin_squared = ffsim.spin_square(
        final_state, norb=mol_data.norb, nelec=mol_data.nelec
    )

    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
        "nit": result.nit,
        "nfev": result.nfev,
        "final_regularization": info["regularization"][-1],
        "final_variation": info["variation"][-1],
    }
    data["nlinop"] = result.nlinop

    logger.info(f"{task} Saving data...\n")
    with open(result_filename, "wb") as f:
        pickle.dump(result, f)
    with open(info_filename, "wb") as f:
        pickle.dump(info, f)
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)

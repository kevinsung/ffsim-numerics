import dataclasses
import logging
import os
import pickle
import timeit
from collections import defaultdict
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
from ffsim.variational.util import orbital_rotation_to_parameters

from ffsim_numerics.params import LBFGSBParams, UCCSDParams

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class UCCSDLBFGSBTask:
    molecule_basename: str
    bond_distance: float | None
    uccsd_params: UCCSDParams
    lbfgsb_params: LBFGSBParams

    @property
    def dirpath(self) -> Path:
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.2f}"
            )
            / self.uccsd_params.dirpath
            / self.lbfgsb_params.dirpath
        )


def run_uccsd_lbfgsb_task(
    task: UCCSDLBFGSBTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    bootstrap_task: UCCSDLBFGSBTask | None = None,
    bootstrap_data_dir: Path | None = None,
    overwrite: bool = True,
) -> UCCSDLBFGSBTask:
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
    assert len(set(nelec)) == 1
    nocc, _ = nelec
    mol_hamiltonian = mol_data.hamiltonian

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    # Define objective function that computes the energy
    def fun(x: np.ndarray) -> float:
        operator = ffsim.UCCSDOpRestrictedReal.from_parameters(
            x,
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=task.uccsd_params.with_final_orbital_rotation,
        )
        final_state = ffsim.apply_unitary(
            reference_state, operator, norb=norb, nelec=nelec
        )
        return np.vdot(final_state, hamiltonian @ final_state).real

    # Generate initial parameters
    if bootstrap_task is None:
        # use CCSD to initialize parameters
        op = ffsim.UCCSDOpRestrictedReal(
            t1=mol_data.ccsd_t1,
            t2=mol_data.ccsd_t2,
            final_orbital_rotation=np.eye(norb),
        )
        params = op.to_parameters()
    else:
        bootstrap_result_filename = os.path.join(
            bootstrap_data_dir or data_dir, bootstrap_task.dirpath, "result.pickle"
        )
        with open(bootstrap_result_filename, "rb") as f:
            result = pickle.load(f)
            params = result.x
        if (
            task.uccsd_params.with_final_orbital_rotation
            and not bootstrap_task.uccsd_params.with_final_orbital_rotation
        ):
            params = np.concatenate([params, np.zeros(norb * (norb - 1) // 2)])
            params[-(norb * (norb - 1) // 2) :] = orbital_rotation_to_parameters(
                np.eye(norb)
            )

    # Optimize ansatz
    logger.info(f"{task} Optimizing ansatz...\n")
    info = defaultdict(list)
    info["nit"] = 0

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        logger.info(f"Task {task} is on iteration {info['nit']}.\n")
        info["x"].append(intermediate_result.x)
        info["fun"].append(intermediate_result.fun)
        info["nit"] += 1

    t0 = timeit.default_timer()
    result = scipy.optimize.minimize(
        fun,
        x0=params,
        method="L-BFGS-B",
        options=dataclasses.asdict(task.lbfgsb_params),
        callback=callback,
    )
    t1 = timeit.default_timer()
    logger.info(f"{task} Done optimizing ansatz in {t1 - t0} seconds.\n")

    logger.info(f"{task} Computing energy and other properties...\n")
    # Compute energy and other properties of final state vector
    operator = ffsim.UCCSDOpRestrictedReal.from_parameters(
        result.x,
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=task.uccsd_params.with_final_orbital_rotation,
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
    }

    logger.info(f"{task} Saving data...\n")
    with open(result_filename, "wb") as f:
        pickle.dump(result, f)
    with open(info_filename, "wb") as f:
        pickle.dump(info, f)
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)

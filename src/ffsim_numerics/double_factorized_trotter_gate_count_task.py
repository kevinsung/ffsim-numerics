import json
import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path

import ffsim
from qiskit.circuit import QuantumCircuit, QuantumRegister

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DoubleFactorizedTrotterGateCountTask:
    molecule_basename: str
    bond_distance: float
    time: float
    n_steps: int
    order: int

    @property
    def dirpath(self) -> Path:
        path = (
            Path(self.molecule_basename)
            / f"bond_distance-{self.bond_distance:.2f}"
            / f"time-{self.time:.1f}"
            / f"n_steps-{self.n_steps}"
            / f"order-{self.order}"
        )
        return path


def run_double_factorized_trotter_gate_count_task(
    task: DoubleFactorizedTrotterGateCountTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> DoubleFactorizedTrotterGateCountTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.json"
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
    mol_hamiltonian = mol_data.hamiltonian

    # Get double-factorized Hamiltonian
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )

    # Compute gate count and depth
    logger.info("Computing gate count and depth...")
    t0 = timeit.default_timer()
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(
            df_hamiltonian, time=task.time, n_steps=task.n_steps, order=task.order
        ),
        qubits,
    )
    decomposed = circuit.decompose(reps=3)
    cx_count = decomposed.count_ops()["cx"]
    cx_depth = decomposed.depth(lambda instruction: instruction.operation.name == "cx")
    t1 = timeit.default_timer()
    logger.info(f"{task} Done computing gate count and depth in {t1 - t0} seconds.")
    result = {"cx_count": cx_count, "cx_depth": cx_depth}

    # Save result to disk
    logger.info(f"{task} Saving result to disk...")
    with open(result_filepath, "w") as f:
        json.dump(result, f)

    logger.info(f"{task} Done.")
    return task

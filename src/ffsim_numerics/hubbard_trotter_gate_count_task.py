import dataclasses
import json
import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class HubbardTrotterGateCountTask:
    norb_x: int
    norb_y: int
    periodic: bool
    n_steps: int
    order: int

    @property
    def dirpath(self) -> Path:
        return (
            Path("hubbard")
            / f"{self.norb_x}x{self.norb_y}"
            / f"periodic-{self.periodic}"
            / f"n_steps-{self.n_steps}"
            / f"order-{self.order}"
        )


def run_hubbard_trotter_gate_count_task(
    task: HubbardTrotterGateCountTask,
    *,
    data_dir: Path,
    overwrite: bool = True,
) -> HubbardTrotterGateCountTask:
    logger.info(f"{task} Starting...")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filepath = data_dir / task.dirpath / "result.json"
    if (not overwrite) and os.path.exists(result_filepath):
        logger.info(f"Data for {task} already exists. Skipping...")
        return task

    # Construct Fermi-Hubbard Hamiltonian
    op = ffsim.fermi_hubbard_2d(
        norb_x=task.norb_x,
        norb_y=task.norb_y,
        tunneling=1.0,
        interaction=1.0,
        periodic=task.periodic,
    )
    norb = task.norb_x * task.norb_y
    hamiltonian = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)
    assert np.all(np.isreal(hamiltonian.one_body_tensor))
    hamiltonian = dataclasses.replace(
        hamiltonian, one_body_tensor=hamiltonian.one_body_tensor.real
    )

    # Compute gate count and depth
    logger.info("Computing gate count and depth...")
    t0 = timeit.default_timer()
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.SimulateTrotterDiagCoulombSplitOpJW(
            hamiltonian, time=1.0, n_steps=task.n_steps, order=task.order
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

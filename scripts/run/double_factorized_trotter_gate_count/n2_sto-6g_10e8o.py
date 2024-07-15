from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ffsim_numerics.double_factorized_trotter_gate_count_task import (
    DoubleFactorizedTrotterGateCountTask,
    run_double_factorized_trotter_gate_count_task,
)
from tqdm import tqdm

filename = f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}.log"
os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=filename,
)

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
MOLECULES_CATALOGUE_DIR = Path(os.environ.get("MOLECULES_CATALOGUE_DIR"))
MAX_PROCESSES = 96

molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
overwrite = True

bond_distance = 1.0
time = 1.0

n_steps_choices = {0: range(1, 40, 6), 1: range(1, 20, 3), 2: range(1, 4)}
n_steps_and_order = list(
    itertools.chain(
        *(
            [(n_steps, order) for n_steps in n_steps_range]
            for order, n_steps_range in n_steps_choices.items()
        )
    )
)
initial_state_and_seed = [("hartree-fock", None), ("random", 46417)]

tasks = [
    DoubleFactorizedTrotterGateCountTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        time=time,
        n_steps=n_steps,
        order=order,
        initial_state=initial_state,
        seed=seed,
    )
    for (n_steps, order), (initial_state, seed) in itertools.product(
        n_steps_and_order, initial_state_and_seed
    )
]


if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_double_factorized_trotter_gate_count_task(
            task,
            data_dir=DATA_DIR,
            molecules_catalogue_dir=MOLECULES_CATALOGUE_DIR,
            overwrite=overwrite,
        )
else:
    with ProcessPoolExecutor(MAX_PROCESSES) as executor:
        with tqdm(total=len(tasks)) as progress:
            for task in tasks:
                future = executor.submit(
                    run_double_factorized_trotter_gate_count_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalogue_dir=MOLECULES_CATALOGUE_DIR,
                    overwrite=overwrite,
                )
                future.add_done_callback(lambda _: progress.update())

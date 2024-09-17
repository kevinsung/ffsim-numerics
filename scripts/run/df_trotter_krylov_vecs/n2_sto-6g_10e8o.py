from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from ffsim_numerics.df_trotter_krylov_vecs_task import (
    DoubleFactorizedTrotterKrylovVecsTask,
    run_double_factorized_trotter_krylov_vecs_task,
)

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
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
MAX_PROCESSES = 96
OVERWRITE = True
ENTROPY = 111000497606135858027052605013196846814

molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.0

time_step = 1e-1
krylov_n_steps = 50
order = 1
trotter_n_steps_range = list(range(1, 6))

tasks = [
    DoubleFactorizedTrotterKrylovVecsTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        krylov_n_steps=krylov_n_steps,
        time_step=time_step,
        trotter_n_steps=trotter_n_steps,
        order=order,
        initial_state="hartree-fock",
    )
    for trotter_n_steps in trotter_n_steps_range
]


if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_double_factorized_trotter_krylov_vecs_task(
            task,
            data_dir=DATA_DIR,
            molecules_catalog_dir=MOLECULES_CATALOG_DIR,
            overwrite=OVERWRITE,
        )
else:
    with tqdm(total=len(tasks)) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for task in tasks:
                future = executor.submit(
                    run_double_factorized_trotter_krylov_vecs_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())

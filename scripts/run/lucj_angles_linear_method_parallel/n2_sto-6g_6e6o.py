from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ffsim_numerics.lucj_angles_linear_method_task import (
    LUCJAnglesLinearMethodTask,
    run_lucj_angles_linear_method_task,
)
from ffsim_numerics.params import LinearMethodParams, LUCJAnglesParams

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
OVERWRITE = False

molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 6, 6
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

start = 0.9
stop = 2.7
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)

connectivities = [
    "square",
]
n_reps_range = [
    2,
    4,
    6,
]

n_givens_layers_range = [1, 2, 3, 4, 5, 6]
ftol = 1e-12
gtol = 1e-5

tasks = [
    LUCJAnglesLinearMethodTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJAnglesParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
            n_givens_layers=n_givens_layers,
        ),
        linear_method_params=LinearMethodParams(
            maxiter=1000,
            lindep=1e-8,
            epsilon=1e-8,
            ftol=ftol,
            gtol=gtol,
            regularization=1e-4,
            variation=0.5,
            optimize_regularization=True,
            optimize_variation=True,
        ),
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for n_givens_layers in n_givens_layers_range
    for d in bond_distance_range
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_angles_linear_method_task(
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
                    run_lucj_angles_linear_method_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())

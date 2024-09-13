from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ffsim_numerics.exact_time_evo_task import (
    ExactTimeEvolutionTask,
    run_exact_time_evolution_task,
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
MAX_PROCESSES = 1

molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
overwrite = True

bond_distance = 1.0

start = 0.0
stop = 1.0
step = 0.1
times = np.linspace(start, stop, num=round((stop - start) / step) + 1)

n_random = 10
rng = np.random.default_rng(46417)
seeds = rng.choice(1 << 16, size=n_random)

tasks = []
for time in times:
    tasks.append(
        ExactTimeEvolutionTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            time=time,
            initial_state="hartree-fock",
        )
    )
    for seed in seeds:
        tasks.append(
            ExactTimeEvolutionTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                time=time,
                initial_state="random",
                seed=seed,
            )
        )


if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_exact_time_evolution_task(
            task,
            data_dir=DATA_DIR,
            molecules_catalog_dir=MOLECULES_CATALOG_DIR,
            overwrite=overwrite,
        )
else:
    with tqdm(total=len(tasks)) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for task in tasks:
                future = executor.submit(
                    run_exact_time_evolution_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=overwrite,
                )
                future.add_done_callback(lambda _: progress.update())

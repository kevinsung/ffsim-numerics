from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ffsim_numerics.hubbard_time_evo_task import (
    HubbardTimeEvolutionTask,
    run_hubbard_time_evolution_task,
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
MAX_PROCESSES = 1

overwrite = True

time = 1.0

norb_x = 4
norb_y = 4
interactions = [1.0, 2.0, 4.0, 8.0]
periodic_choices = [False, True]
filling_denominators = [8, 4, 2]

tasks = [
    HubbardTimeEvolutionTask(
        time=time,
        norb_x=norb_x,
        norb_y=norb_y,
        tunneling=1.0,
        interaction=interaction,
        chemical_potential=0.0,
        nearest_neighbor_interaction=0.0,
        periodic=periodic,
        filling_denominator=filling_denominator,
        initial_state="random",
        seed=46417,
    )
    for interaction, periodic, filling_denominator in itertools.product(
        interactions, periodic_choices, filling_denominators
    )
]


if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_hubbard_time_evolution_task(
            task,
            data_dir=DATA_DIR,
            overwrite=overwrite,
        )
else:
    with tqdm(total=len(tasks)) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for task in tasks:
                future = executor.submit(
                    run_hubbard_time_evolution_task,
                    task,
                    data_dir=DATA_DIR,
                    overwrite=overwrite,
                )
                future.add_done_callback(lambda _: progress.update())

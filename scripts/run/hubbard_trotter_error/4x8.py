from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from ffsim_numerics.hubbard_trotter_error_task import (
    HubbardTrotterErrorTask,
    run_hubbard_trotter_error_task,
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
MAX_PROCESSES = 12
OVERWRITE = True
ENTROPY = 155903744721100194602646941346278309426

norb_x = 4
norb_y = 8
interactions = [8.0]
filling_denominators = [8]

time = 1.0
n_random = 10

n_steps_choices = {
    0: [1, 21, 41, 61],
    1: [1, 11, 21, 31],
    2: [1, 3, 5, 7],
    3: [1, 2, 3],
}
n_steps_and_order = list(
    itertools.chain(
        *(
            [(n_steps, order) for n_steps in n_steps_range]
            for order, n_steps_range in n_steps_choices.items()
        )
    )
)

tasks = [
    HubbardTrotterErrorTask(
        norb_x=norb_x,
        norb_y=norb_y,
        tunneling=1.0,
        interaction=interaction,
        chemical_potential=0.0,
        nearest_neighbor_interaction=0.0,
        periodic_x=True,
        periodic_y=False,
        filling_denominator=filling_denominator,
        time=time,
        n_steps=n_steps,
        order=order,
        initial_state="random",
        entropy=ENTROPY,
        spawn_index=spawn_index,
    )
    for interaction, filling_denominator in itertools.product(
        interactions, filling_denominators
    )
    for spawn_index in range(n_random)
    for n_steps, order in n_steps_and_order
]


if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_hubbard_trotter_error_task(
            task,
            data_dir=DATA_DIR,
            overwrite=OVERWRITE,
        )
else:
    with tqdm(total=len(tasks)) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for task in tasks:
                future = executor.submit(
                    run_hubbard_trotter_error_task,
                    task,
                    data_dir=DATA_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())

from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ffsim_numerics.hubbard_trotter_gate_count_task import (
    HubbardTrotterGateCountTask,
    run_hubbard_trotter_gate_count_task,
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
MAX_PROCESSES = 96

overwrite = False

norb_x = 4
norb_y = 2

periodic_choices = [False, True]
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
    HubbardTrotterGateCountTask(
        norb_x=norb_x,
        norb_y=norb_y,
        periodic=periodic,
        n_steps=n_steps,
        order=order,
    )
    for periodic, (n_steps, order) in itertools.product(
        periodic_choices, n_steps_and_order
    )
]


if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_hubbard_trotter_gate_count_task(
            task,
            data_dir=DATA_DIR,
            overwrite=overwrite,
        )
else:
    with tqdm(total=len(tasks)) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for task in tasks:
                future = executor.submit(
                    run_hubbard_trotter_gate_count_task,
                    task,
                    data_dir=DATA_DIR,
                    overwrite=overwrite,
                )
                future.add_done_callback(lambda _: progress.update())

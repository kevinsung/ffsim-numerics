from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ffsim_numerics.lucj_angles_linear_method_task import (
    LUCJAnglesLinearMethodTask,
    run_lucj_angles_linear_method_task,
)
from ffsim_numerics.params import LinearMethodParams, LUCJAnglesParams
from ffsim_numerics.util import copy_data

filepath = f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}.log"
os.makedirs(os.path.dirname(filepath), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=filepath,
)

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
MAX_PROCESSES = 96
OVERWRITE = True

molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
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
    8,
]
n_givens_layers_range = [2, 4, 6, 8]

bootstrap_reps = 3
bootstrap_max_iter = 100
bootstrap_ftol = 1e-8
bootstrap_gtol = 1e-5

final_maxiter = 1000
final_ftol = 1e-12
final_gtol = 1e-5


def run_bootstrap(
    bootstrap_reps: int,
    bond_distances: np.ndarray,
    connectivity: str,
    n_reps: int,
    n_givens_layers: int,
    overwrite: bool,
):
    lucj_params = LUCJAnglesParams(
        connectivity=connectivity,
        n_reps=n_reps,
        with_final_orbital_rotation=True,
        n_givens_layers=n_givens_layers,
    )
    linear_method_params = LinearMethodParams(
        maxiter=bootstrap_max_iter,
        lindep=1e-8,
        epsilon=1e-8,
        ftol=bootstrap_ftol,
        gtol=bootstrap_gtol,
        regularization=1e-4,
        variation=0.5,
        optimize_regularization=True,
        optimize_variation=True,
    )
    # first task
    task = LUCJAnglesLinearMethodTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distances[0],
        lucj_params=lucj_params,
        linear_method_params=linear_method_params,
    )
    run_lucj_angles_linear_method_task(
        task,
        data_dir=DATA_DIR / "rep-0",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
        overwrite=overwrite,
    )
    # bootstrap
    bootstrap_task = task
    for rep in tqdm(
        range(bootstrap_reps), desc=f"Bootstrap reps {connectivity} L={n_reps}"
    ):
        these_bond_distances = bond_distances if rep % 2 == 0 else bond_distances[::-1]
        if rep > 0:
            task = LUCJAnglesLinearMethodTask(
                molecule_basename=molecule_basename,
                bond_distance=these_bond_distances[0],
                lucj_params=lucj_params,
                linear_method_params=linear_method_params,
            )
            copy_data(
                task,
                src_data_dir=DATA_DIR / f"rep-{rep - 1}",
                dst_data_dir=DATA_DIR / f"rep-{rep}",
                dirs_exist_ok=True,
            )
        for bond_distance in tqdm(
            these_bond_distances[1:], desc=f"Bond distance {connectivity} L={n_reps}"
        ):
            task = LUCJAnglesLinearMethodTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=lucj_params,
                linear_method_params=linear_method_params,
            )
            run_lucj_angles_linear_method_task(
                task,
                data_dir=DATA_DIR / f"rep-{rep}",
                molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                bootstrap_task=bootstrap_task,
                overwrite=overwrite,
            )
            bootstrap_task = task


# Run bootstrap
print("Running bootstrap...")
if MAX_PROCESSES == 1:
    for connectivity, n_reps, n_givens_layers in tqdm(
        itertools.product(connectivities, n_reps_range, n_givens_layers_range),
        total=len(connectivities) * len(n_reps_range) * len(n_givens_layers_range),
        desc="Bootstrap tasks",
    ):
        run_bootstrap(
            bootstrap_reps=bootstrap_reps,
            bond_distances=bond_distance_range,
            connectivity=connectivity,
            n_reps=n_reps,
            n_givens_layers=n_givens_layers,
            overwrite=OVERWRITE,
        )
else:
    with tqdm(
        total=len(connectivities) * len(n_reps_range) * len(n_givens_layers_range),
        desc="Bootstrap tasks",
    ) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for connectivity, n_reps, n_givens_layers in itertools.product(
                connectivities, n_reps_range, n_givens_layers_range
            ):
                future = executor.submit(
                    run_bootstrap,
                    bootstrap_reps=bootstrap_reps,
                    bond_distances=bond_distance_range,
                    connectivity=connectivity,
                    n_reps=n_reps,
                    n_givens_layers=n_givens_layers,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())


# Determine bootstrap iteration with lowest energy
bootstrap_tasks = [
    LUCJAnglesLinearMethodTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJAnglesParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
            n_givens_layers=n_givens_layers,
        ),
        linear_method_params=LinearMethodParams(
            maxiter=bootstrap_max_iter,
            lindep=1e-8,
            epsilon=1e-8,
            ftol=bootstrap_ftol,
            gtol=bootstrap_gtol,
            regularization=1e-4,
            variation=0.5,
            optimize_regularization=True,
            optimize_variation=True,
        ),
    )
    for (connectivity, n_reps, n_givens_layers) in itertools.product(
        connectivities, n_reps_range, n_givens_layers_range
    )
    for bond_distance in bond_distance_range
]
min_bootstrap_reps = []
for task in bootstrap_tasks:
    min_bootstrap_rep = -1
    min_energy = float("inf")
    for rep in range(bootstrap_reps):
        filepath = DATA_DIR / f"rep-{rep}" / task.dirpath / "data.pickle"
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        if data["energy"] < min_energy:
            min_bootstrap_rep = rep
            min_energy = data["energy"]
    min_bootstrap_reps.append(min_bootstrap_rep)
assert len(min_bootstrap_reps) == len(bootstrap_tasks)


# Run final optimization in parallel
print("Running final optimization...")
final_opt_tasks = [
    dataclasses.replace(
        task,
        linear_method_params=dataclasses.replace(
            task.linear_method_params,
            maxiter=final_maxiter,
            ftol=final_ftol,
            gtol=final_gtol,
        ),
    )
    for task in bootstrap_tasks
]
if MAX_PROCESSES == 1:
    for final_opt_task, bootstrap_task, bootstrap_rep in tqdm(
        zip(final_opt_tasks, bootstrap_tasks, min_bootstrap_reps),
        total=len(final_opt_tasks),
        desc="Final opt tasks",
    ):
        run_lucj_angles_linear_method_task(
            final_opt_task,
            data_dir=DATA_DIR / "post-bootstrap",
            molecules_catalog_dir=MOLECULES_CATALOG_DIR,
            bootstrap_task=bootstrap_task,
            bootstrap_data_dir=DATA_DIR / f"rep-{bootstrap_rep}",
            overwrite=OVERWRITE,
        )

else:
    with tqdm(total=len(final_opt_tasks), desc="Final opt tasks") as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for final_opt_task, bootstrap_task, bootstrap_rep in zip(
                final_opt_tasks, bootstrap_tasks, min_bootstrap_reps
            ):
                future = executor.submit(
                    run_lucj_angles_linear_method_task,
                    final_opt_task,
                    data_dir=DATA_DIR / "post-bootstrap",
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    bootstrap_task=bootstrap_task,
                    bootstrap_data_dir=DATA_DIR / f"rep-{bootstrap_rep}",
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())

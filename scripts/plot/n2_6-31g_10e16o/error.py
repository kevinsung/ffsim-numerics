import itertools
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ffsim_numerics.double_factorized_trotter_gate_count_task import (
    DoubleFactorizedTrotterGateCountTask,
)
from ffsim_numerics.double_factorized_trotter_sim_task import (
    DoubleFactorizedTrotterSimTask,
)
from ffsim_numerics.exact_time_evo_task import ExactTimeEvolutionTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = os.environ.get("MOLECULES_CATALOG_DIR")


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)


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


data = {}
for initial_state, seed in initial_state_and_seed:
    task = ExactTimeEvolutionTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        time=time,
        initial_state=initial_state,
        seed=seed,
    )
    filepath = DATA_ROOT / "exact_time_evo" / task.dirpath / "result.npy"
    with open(filepath, "rb") as f:
        exact_state = np.load(filepath)
    for n_steps, order in n_steps_and_order:
        # Compute error
        task = DoubleFactorizedTrotterSimTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            time=time,
            n_steps=n_steps,
            order=order,
            initial_state=initial_state,
            seed=seed,
        )
        filepath = (
            DATA_ROOT / "double_factorized_trotter_sim" / task.dirpath / "result.npy"
        )
        with open(filepath, "rb") as f:
            trotter_state = np.load(filepath)
        error = np.linalg.norm(trotter_state - exact_state)

        # Load gate count and depth
        task = DoubleFactorizedTrotterGateCountTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            time=time,
            n_steps=n_steps,
            order=order,
            initial_state=initial_state,
            seed=seed,
        )
        filepath = (
            DATA_ROOT
            / "double_factorized_trotter_gate_count"
            / task.dirpath
            / "result.json"
        )
        with open(filepath, "r") as f:
            gate_counts = json.load(f)

        data[initial_state, n_steps, order] = (
            error,
            gate_counts["cx_count"],
            gate_counts["cx_depth"],
        )

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, ax = plt.subplots(1, 1)

for (order, n_steps_range), marker, color in zip(
    n_steps_choices.items(), markers, colors
):
    errors, cx_counts, cx_depths = zip(
        *[data[initial_state, n_steps, order] for n_steps in n_steps_range]
    )
    ax.plot(cx_counts, errors, f"{marker}--", label=f"Order {order}", color=color)

ax.set_yscale("log")
ax.legend()
ax.set_xlabel("Two-qubit gate count")
ax.set_ylabel(r"$|\psi - \psi^*|$")
ax.set_title(f"{molecule_name} {basis} ({nelectron}e, {norb}o)")

filename = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.svg"
)
plt.savefig(filename)
plt.close()

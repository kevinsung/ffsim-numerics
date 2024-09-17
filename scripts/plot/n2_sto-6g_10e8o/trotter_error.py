import itertools
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.molecule_trotter_error_task import MoleculeTrotterErrorTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
ENTROPY = 111000497606135858027052605013196846814


molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.0

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

time = 1.0
n_random = 10

n_steps_choices = {0: range(1, 40, 6), 1: range(1, 20, 3), 2: range(1, 4)}
n_steps_and_order = list(
    itertools.chain(
        *(
            [(n_steps, order) for n_steps in n_steps_range]
            for order, n_steps_range in n_steps_choices.items()
        )
    )
)


data = {}
for n_steps, order in n_steps_and_order:
    for spawn_index in range(n_random):
        task = MoleculeTrotterErrorTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            time=time,
            n_steps=n_steps,
            order=order,
            initial_state="random",
            entropy=ENTROPY,
            spawn_index=spawn_index,
        )
        filepath = DATA_ROOT / "molecule_trotter_error" / task.dirpath / "result.pickle"
        with open(filepath, "rb") as f:
            data[n_steps, order, spawn_index] = pickle.load(f)


errors = {}
for n_steps, order in n_steps_and_order:
    _, cx_count, cx_depth = data[n_steps, order, 0]
    these_errors = [
        data[n_steps, order, spawn_index][0] for spawn_index in range(n_random)
    ]
    mean = np.mean(these_errors)
    std_error = np.std(these_errors) / np.sqrt(n_random)
    errors[n_steps, order] = mean, std_error, cx_count, cx_depth


markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, ax = plt.subplots(1, 1)

for (order, n_steps_range), marker, color in zip(
    n_steps_choices.items(), markers, colors
):
    mean_errors, std_errors, cx_counts, cx_depths = zip(
        *[errors[n_steps, order] for n_steps in n_steps_range]
    )
    ax.errorbar(
        cx_counts,
        mean_errors,
        yerr=std_errors,
        fmt=f"{marker}--",
        label=f"Order {order}",
        color=color,
    )

ax.set_yscale("log")
ax.legend()
ax.set_xlabel("Two-qubit gate count")
ax.set_ylabel(r"$|\psi - \psi_{\text{exact}}|$")
ax.set_title(f"{molecule_name} {basis} ({nelectron}e, {norb}o)")

filename = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.svg"
)
plt.savefig(filename)
plt.close()

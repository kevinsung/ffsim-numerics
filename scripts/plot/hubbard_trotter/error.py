import itertools
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.hubbard_trotter_error_task import HubbardTrotterErrorTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
ENTROPY = 155903744721100194602646941346278309426


norb_x = 4
norb_y_range = [2, 4, 6, 8]

plots_dir = os.path.join(
    "plots", os.path.basename(os.path.dirname(os.path.abspath(__file__)))
)
os.makedirs(plots_dir, exist_ok=True)


interactions = [8.0]
filling_denominator = 8

time = 1.0
n_random = 5
order = 2
n_steps_range = [1, 3, 5, 7]


for interaction in interactions:
    data = {}
    for norb_y in norb_y_range:
        for n_steps in n_steps_range:
            for spawn_index in range(n_random):
                task = HubbardTrotterErrorTask(
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
                filepath = (
                    DATA_ROOT / "hubbard_trotter_error" / task.dirpath / "result.pickle"
                )
                with open(filepath, "rb") as f:
                    data[norb_y, n_steps, spawn_index] = pickle.load(f)

    errors = {}
    for norb_y in norb_y_range:
        for n_steps in n_steps_range:
            _, cx_count, cx_depth = data[norb_y, n_steps, 0]
            these_errors = [
                data[norb_y, n_steps, spawn_index][0] for spawn_index in range(n_random)
            ]
            mean = np.mean(these_errors)
            std_error = np.std(these_errors) / np.sqrt(n_random)
            errors[norb_y, n_steps] = mean, std_error, cx_count, cx_depth

    markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    alphas = [0.5, 1.0]
    linestyles = ["--", ":"]

    fig, ax = plt.subplots(1, 1)

    for norb_y, marker, color in zip(norb_y_range, markers, colors):
        mean_errors, std_errors, cx_counts, cx_depths = zip(
            *[errors[norb_y, n_steps] for n_steps in n_steps_range]
        )
        ax.errorbar(
            cx_counts,
            mean_errors,
            yerr=std_errors,
            fmt=f"{marker}--",
            label=f"4x{norb_y}",
            color=color,
        )

    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Two-qubit gate count")
    ax.set_ylabel(r"$|\psi - \psi_{\text{exact}}|$")
    ax.set_title(
        rf"Hubbard, Order {order} Trotter, $\nu=1/{filling_denominator}$, U/t={interaction}"
    )

    filename = os.path.join(
        plots_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_interaction-{interaction}.svg",
    )
    plt.savefig(filename)
    print(f"Saved figure to {filename}.")
    plt.close()

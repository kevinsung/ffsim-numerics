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
norb_y = 2

plots_dir = os.path.join("plots", f"hubbard_{norb_x}x{norb_y}")
os.makedirs(plots_dir, exist_ok=True)

interactions = [8.0]
filling_denominator = 8

time = 1.0
n_random = 10

n_steps_choices = {0: [1, 26, 51, 76], 1: [1, 13, 25, 37], 2: [1, 5, 9, 13]}
n_steps_and_order = list(
    itertools.chain(
        *(
            [(n_steps, order) for n_steps in n_steps_range]
            for order, n_steps_range in n_steps_choices.items()
        )
    )
)

for interaction in interactions:
    data = {}
    for n_steps, order in n_steps_and_order:
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
    ax.set_title(
        rf"Hubbard {norb_x}x{norb_y}, $\nu=1/{filling_denominator}$, U/t={interaction}"
    )

    filename = os.path.join(
        plots_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_interaction-{interaction}.svg",
    )
    plt.savefig(filename)
    plt.close()

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.hubbard_trotter_error_task import HubbardTrotterErrorTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
ENTROPY = 155903744721100194602646941346278309426

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

colors_4a = ["#9f1853", "#fa4d56", "#570408", "#a56eff"]
tick_label_fontsize = 13
axis_label_fontsize = 14
title_fontsize = 15

norb_x = 4
norb_y = 4
filling_denominator = 8
time = 1.0
n_random = 1000
order = 2
n_steps_range = [1, 3, 5, 7]
interaction = 8.0

data = {}
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
        filepath = DATA_ROOT / "hubbard_trotter_error" / task.dirpath / "result.pickle"
        with open(filepath, "rb") as f:
            data[n_steps, spawn_index] = pickle.load(f)

errors = {}
for n_steps in n_steps_range:
    vals = np.array([data[n_steps, spawn_index][0] for spawn_index in range(n_random)])
    errors[n_steps] = vals - vals.mean()

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for row in axes:
    for ax in row:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

for n_steps, color, ax in zip(n_steps_range, colors_4a, axes.flat):
    ax.hist(
        errors[n_steps],
        bins=50,
        color=color,
    )
    ax.set_xlabel(
        r"$\|\psi_\text{Trotter} - \psi_\text{exact}\|_2 - \mu$",
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel("Count", fontsize=axis_label_fontsize)
    ax.set_title(f"n_steps={n_steps}", fontsize=title_fontsize)

fig.suptitle(
    rf"Hubbard {norb_x}x{norb_y}, order {order} Trotter, 1/{filling_denominator} filling, U/t={interaction:.0f}",
    fontsize=title_fontsize,
)
fig.tight_layout()
filename = os.path.join(plots_dir, "hubbard_trotter_error.pdf")
plt.savefig(filename)
print(f"Saved figure to {filename}.")
plt.close()

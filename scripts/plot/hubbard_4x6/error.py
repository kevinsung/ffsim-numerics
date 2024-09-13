import itertools
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ffsim_numerics.hubbard_time_evo_task import HubbardTimeEvolutionTask
from ffsim_numerics.hubbard_trotter_gate_count_task import HubbardTrotterGateCountTask
from ffsim_numerics.hubbard_trotter_sim_task import HubbardTrotterSimTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))


norb_x = 4
norb_y = 6

plots_dir = os.path.join("plots", f"hubbard_{norb_x}x{norb_y}")
os.makedirs(plots_dir, exist_ok=True)


interactions = [1.0, 2.0, 4.0, 8.0]
periodic_choices = [False, True]
filling_denominators = [8]
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
initial_state_and_seed = [("one-body", None), ("random", 46417)]

interaction = 1.0
periodic = False
filling_denominator = 8

data = {}
for initial_state, seed in initial_state_and_seed:
    task = HubbardTimeEvolutionTask(
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
    filepath = DATA_ROOT / "hubbard_time_evo" / task.dirpath / "result.npy"
    with open(filepath, "rb") as f:
        exact_state = np.load(filepath)
    for n_steps, order in n_steps_and_order:
        # Compute error
        task = HubbardTrotterSimTask(
            norb_x=norb_x,
            norb_y=norb_y,
            tunneling=1.0,
            interaction=interaction,
            chemical_potential=0.0,
            nearest_neighbor_interaction=0.0,
            periodic=periodic,
            filling_denominator=filling_denominator,
            time=time,
            n_steps=n_steps,
            order=order,
            initial_state=initial_state,
            seed=seed,
        )
        filepath = DATA_ROOT / "hubbard_trotter_sim" / task.dirpath / "result.npy"
        with open(filepath, "rb") as f:
            trotter_state = np.load(filepath)
        error = np.linalg.norm(trotter_state - exact_state)

        # Load gate count and depth
        task = HubbardTrotterGateCountTask(
            norb_x=norb_x,
            norb_y=norb_y,
            periodic=periodic,
            n_steps=n_steps,
            order=order,
        )
        filepath = (
            DATA_ROOT / "hubbard_trotter_gate_count" / task.dirpath / "result.json"
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
ax.set_ylabel(r"$|\psi - \psi_{\text{exact}}|$")
ax.set_title(
    rf"Hubbard {norb_x}x{norb_y}, $\nu=1/{filling_denominator}$, U/t={interaction}"
)

filename = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.svg"
)
plt.savefig(filename)
plt.close()

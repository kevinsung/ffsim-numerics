import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.df_trotter_krylov_task import DoubleFactorizedTrotterKrylovTask
from ffsim_numerics.exact_krylov_task import ExactKrylovTask
from ffsim_numerics.hubbard_trotter_error_task import HubbardTrotterErrorTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
ENTROPY = 155903744721100194602646941346278309426

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
colors_4a = ["#9f1853", "#fa4d56", "#570408", "#a56eff"]
colors_4b = ["#6929c4", "#012749", "#009d9a", "#ee538b"]
colors_5a = ["#6929c4", "#1192e8", "#005d5d", "#9f1853", "#570408"]
colors_5b = ["#002d9c", "#009d9a", "#9f1853", "#570408", "#a56eff"]
capsize = 4
linestyles = [":", "--", "-.", (0, (5, 5)), (0, (3, 1, 1, 1, 1, 1))]
legend_fontsize = 12
tick_label_fontsize = 13
axis_label_fontsize = 14
title_fontsize = 15

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8),
)
for row in axes:
    for ax in row:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 (top left): hubbard_trotter/error.py
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0, 0]

norb_x = 4
norb_y_range = [2, 4, 6, 8]
filling_denominator = 8
time = 1.0
n_random = 5
order = 2
n_steps_range = [1, 3, 5, 7]
interaction = 8.0

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
        std_error = np.std(these_errors)
        errors[norb_y, n_steps] = mean, std_error, cx_count, cx_depth

for norb_y, marker, color, linestyle in zip(
    norb_y_range, markers, colors_4a, linestyles
):
    mean_errors, std_errors, cx_counts, cx_depths = zip(
        *[errors[norb_y, n_steps] for n_steps in n_steps_range]
    )
    ax.errorbar(
        cx_counts,
        mean_errors,
        yerr=std_errors,
        fmt=f"{marker}",
        linestyle=linestyle,
        label=f"4x{norb_y}",
        color=color,
        capsize=capsize,
    )

ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(fontsize=legend_fontsize, loc="lower left")
ax.set_xlabel("Two-qubit gate count", fontsize=axis_label_fontsize)
ax.set_ylabel(
    r"$\|\psi_\text{Trotter} - \psi_\text{exact}\|_2$", fontsize=axis_label_fontsize
)
ax.set_title(
    rf"Hubbard, order {order} Trotter, 1/{filling_denominator} filling, U/t={interaction:.0f}",
    fontsize=title_fontsize,
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 (top right): hubbard_4x8/trotter_error.py
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0, 1]

norb_x = 4
norb_y = 8
filling_denominator = 8
time = 1.0
n_random = 5
interaction = 8.0

n_steps_choices = {
    0: [1, 21, 41, 61],
    1: [1, 11, 21, 31],
    2: [1, 3, 5, 7],
    3: [1, 2, 3],
}
n_steps_and_order = list(
    itertools.chain(
        *(
            [(n_steps, order) for n_steps in n_steps_range_for_order]
            for order, n_steps_range_for_order in n_steps_choices.items()
        )
    )
)

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
        filepath = DATA_ROOT / "hubbard_trotter_error" / task.dirpath / "result.pickle"
        with open(filepath, "rb") as f:
            data[n_steps, order, spawn_index] = pickle.load(f)

errors = {}
for n_steps, order in n_steps_and_order:
    _, cx_count, cx_depth = data[n_steps, order, 0]
    these_errors = [
        data[n_steps, order, spawn_index][0] for spawn_index in range(n_random)
    ]
    mean = np.mean(these_errors)
    std_error = np.std(these_errors)
    errors[n_steps, order] = mean, std_error, cx_count, cx_depth

for (order, n_steps_range_for_order), marker, color, linestyle in zip(
    n_steps_choices.items(), markers, colors_4b, linestyles
):
    mean_errors, std_errors, cx_counts, cx_depths = zip(
        *[errors[n_steps, order] for n_steps in n_steps_range_for_order]
    )
    ax.errorbar(
        cx_counts,
        mean_errors,
        yerr=std_errors,
        fmt=f"{marker}",
        linestyle=linestyle,
        label=f"Order {order}",
        color=color,
        capsize=capsize,
    )

ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(fontsize=legend_fontsize)
ax.set_xlabel("Two-qubit gate count", fontsize=axis_label_fontsize)
ax.set_ylabel(
    r"$\|\psi_\text{Trotter} - \psi_\text{exact}\|_2$", fontsize=axis_label_fontsize
)
ax.set_title(
    rf"Hubbard {norb_x}x{norb_y}, 1/{filling_denominator} filling, U/t={interaction:.0f}",
    fontsize=title_fontsize,
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 (bottom left): exact_krylov_error_vs_dim.py with lindep=1e-12
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1, 0]

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.0

time_step_range = [1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
lindep = 1e-12
n_steps = 50
n_steps_plot = 30

molecule_filepath = (
    Path("molecular_data")
    / molecule_basename
    / f"{molecule_basename}_d-{bond_distance:.5f}.json.xz"
)
mol_data = ffsim.MolecularData.from_json(molecule_filepath, compression="lzma")

data = {}
for time_step in time_step_range:
    task = ExactKrylovTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        time_step=time_step,
        n_steps=n_steps,
        initial_state="hartree-fock",
        lindep=lindep,
    )
    filepath = DATA_ROOT / "exact_krylov" / task.dirpath / "result.npy"
    ground_energies = np.load(filepath)
    krylov_errors = ground_energies - mol_data.fci_energy
    assert all(krylov_errors > -1e-8)
    data[time_step] = abs(krylov_errors)

for time_step, color, linestyle in zip(time_step_range, colors_5a, linestyles):
    ax.plot(
        range(2, n_steps_plot + 3),
        data[time_step][: len(range(2, n_steps_plot + 3))],
        label=f"∆t={time_step}",
        color=color,
        linestyle=linestyle,
    )

ax.set_xticks(range(2, n_steps_plot + 3, 6))
ax.set_yscale("log")
ax.legend(fontsize=legend_fontsize)
ax.set_xlabel("Krylov space dimension", fontsize=axis_label_fontsize)
ax.set_ylabel(
    r"$|E_\text{KQD} - E_\text{exact}|$ ($E_\text{h}$)", fontsize=axis_label_fontsize
)
ax.set_title(
    f"N$_2$ / 6-31G ({nelectron}e, {norb}o), exact evolution", fontsize=title_fontsize
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 (bottom right): df_trotter_krylov_error_vs_n_step.py with order=1
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1, 1]

time_step = 3e-1
krylov_n_steps = 30
trotter_n_steps_range = list(range(1, 6))
order = 1
lindep = 1e-12

data = {}
for trotter_n_steps in trotter_n_steps_range:
    task = DoubleFactorizedTrotterKrylovTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        krylov_n_steps=krylov_n_steps,
        time_step=time_step,
        trotter_n_steps=trotter_n_steps,
        order=order,
        initial_state="hartree-fock",
        lindep=lindep,
    )
    filepath = DATA_ROOT / "df_trotter_krylov" / task.dirpath / "result.npy"
    ground_energies = np.load(filepath)
    krylov_errors = ground_energies - mol_data.fci_energy
    assert all(krylov_errors > 0)
    data[trotter_n_steps] = krylov_errors

task = ExactKrylovTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    time_step=time_step,
    n_steps=50,
    initial_state="hartree-fock",
    lindep=lindep,
)
filepath = DATA_ROOT / "exact_krylov" / task.dirpath / "result.npy"
ground_energies = np.load(filepath)
exact_krylov_errors = ground_energies - mol_data.fci_energy
assert all(exact_krylov_errors > -1e-8)
exact_krylov_errors = np.abs(exact_krylov_errors[: krylov_n_steps + 1])

ax.plot(range(2, krylov_n_steps + 3), exact_krylov_errors, label="exact", color="black")
for trotter_n_steps, color, linestyle in zip(
    trotter_n_steps_range, colors_5b, linestyles
):
    ax.plot(
        range(2, krylov_n_steps + 3),
        data[trotter_n_steps],
        label=f"n_steps={trotter_n_steps}",
        color=color,
        linestyle=linestyle,
    )

ax.set_xticks(range(2, krylov_n_steps + 3, 6))
ax.set_yscale("log")
ax.legend(fontsize=legend_fontsize)
ax.set_xlabel("Krylov space dimension", fontsize=axis_label_fontsize)
ax.set_ylabel(
    r"$|E_\text{KQD} - E_\text{exact}|$ ($E_\text{h}$)", fontsize=axis_label_fontsize
)
ax.set_title(
    f"N$_2$ / 6-31G ({nelectron}e, {norb}o), ∆t={time_step}, order {order} Trotter",
    fontsize=title_fontsize,
)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
fig.tight_layout(h_pad=2)
fig.subplots_adjust(wspace=0.22)
filename = os.path.join(plots_dir, "applications.pdf")
plt.savefig(filename)
print(f"Saved figure to {filename}.")
plt.close()

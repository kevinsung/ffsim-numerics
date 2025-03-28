import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.lucj_angles_initial_params_task import LUCJAnglesInitialParamsTask
from ffsim_numerics.lucj_initial_params_task import LUCJInitialParamsTask
from ffsim_numerics.params import LUCJAnglesParams, LUCJParams
from ffsim_numerics.uccsd_initial_params_task import UCCSDInitialParamsTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 6, 6
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

start = 0.7
stop = 3.0
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
reference_bond_distance_range = np.linspace(
    start, stop, num=round((stop - start) / 0.05) + 1
)

connectivities = [
    "all-to-all",
    "square",
]
n_reps_range = [
    # 2,
    # 4,
    6,
    None,
]
n_givens_layers_range = [norb, norb - 1, norb - 2]

tasks_lucj = [
    LUCJInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for d in bond_distance_range
]
tasks_lucj_angles = [
    LUCJAnglesInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJAnglesParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
            n_givens_layers=n_givens_layers,
        ),
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for n_givens_layers in n_givens_layers_range
    for d in bond_distance_range
]
tasks_uccsd = [
    UCCSDInitialParamsTask(molecule_basename=molecule_basename, bond_distance=d)
    for d in bond_distance_range
]


mol_datas_reference: dict[float, ffsim.MolecularData] = {}
mol_datas_experiment: dict[float, ffsim.MolecularData] = {}

for d in reference_bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.2f}.json.xz",
    )
    mol_datas_reference[d] = ffsim.MolecularData.from_json(filepath, compression="lzma")

for d in bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.2f}.json.xz",
    )
    mol_datas_experiment[d] = ffsim.MolecularData.from_json(
        filepath, compression="lzma"
    )

hf_energies_reference = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_reference.values()]
)
fci_energies_reference = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_reference.values()]
)
ccsd_energies_reference = np.array(
    [mol_data.ccsd_energy for mol_data in mol_datas_reference.values()]
)
hf_energies_experiment = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_experiment.values()]
)
fci_energies_experiment = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_experiment.values()]
)

print("Loading data...")
data_lucj = {}
for task in tasks_lucj:
    filepath = DATA_ROOT / "lucj_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_lucj[task] = pickle.load(f)
data_lucj_angles = {}
for task in tasks_lucj_angles:
    filepath = DATA_ROOT / "lucj_angles_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_lucj_angles[task] = pickle.load(f)
data_uccsd = {}
for task in tasks_uccsd:
    filepath = DATA_ROOT / "uccsd_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_uccsd[task] = pickle.load(f)
print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, axes = plt.subplots(3, len(connectivities), figsize=(12, 12), layout="constrained")

for i, connectivity in enumerate(connectivities):
    axes[0, i].plot(
        reference_bond_distance_range,
        hf_energies_reference,
        "--",
        label="HF",
        color="blue",
    )
    axes[0, i].plot(
        reference_bond_distance_range,
        ccsd_energies_reference,
        "--",
        label="CCSD",
        color="orange",
    )
    axes[0, i].plot(
        reference_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color="black",
    )
    energies = [data_uccsd[task]["energy"] for task in tasks_uccsd]
    errors = [data_uccsd[task]["error"] for task in tasks_uccsd]
    spin_squares = [data_uccsd[task]["spin_squared"] for task in tasks_uccsd]
    axes[0, i].plot(
        bond_distance_range,
        energies,
        f"{markers[0]}{linestyles[0]}",
        label="UCCSD",
        color=colors[0],
    )
    axes[1, i].plot(
        bond_distance_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="UCCSD",
        color=colors[0],
    )
    axes[2, i].plot(
        bond_distance_range,
        spin_squares,
        f"{markers[0]}{linestyles[0]}",
        label="UCCSD",
        color=colors[0],
    )
    for n_reps, marker, color in zip(n_reps_range, markers[1:], colors[1:]):
        tasks_lucj = [
            LUCJInitialParamsTask(
                molecule_basename=molecule_basename,
                bond_distance=d,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
            )
            for d in bond_distance_range
        ]
        energies = [data_lucj[task]["energy"] for task in tasks_lucj]
        errors = [data_lucj[task]["error"] for task in tasks_lucj]
        spin_squares = [data_lucj[task]["spin_squared"] for task in tasks_lucj]
        axes[0, i].plot(
            bond_distance_range,
            energies,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ L={n_reps}",
            color=color,
        )
        axes[1, i].plot(
            bond_distance_range,
            errors,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ L={n_reps}",
            color=color,
        )
        axes[2, i].plot(
            bond_distance_range,
            spin_squares,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ L={n_reps}",
            color=color,
        )
    for n_reps, marker, color in zip(
        n_reps_range, markers[1 + len(n_reps_range) :], colors[1 + len(n_reps_range) :]
    ):
        tasks_lucj_angles = [
            LUCJAnglesInitialParamsTask(
                molecule_basename=molecule_basename,
                bond_distance=d,
                lucj_params=LUCJAnglesParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                    n_givens_layers=norb,
                ),
            )
            for d in bond_distance_range
        ]
        energies = [data_lucj_angles[task]["energy"] for task in tasks_lucj_angles]
        errors = [data_lucj_angles[task]["error"] for task in tasks_lucj_angles]
        spin_squares = [
            data_lucj_angles[task]["spin_squared"] for task in tasks_lucj_angles
        ]
        axes[0, i].plot(
            bond_distance_range,
            energies,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ angles L={n_reps}",
            color=color,
        )
        axes[1, i].plot(
            bond_distance_range,
            errors,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ angles L={n_reps}",
            color=color,
        )
        axes[2, i].plot(
            bond_distance_range,
            spin_squares,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ angles L={n_reps}",
            color=color,
        )

    axes[0, i].legend()
    axes[0, i].set_title(connectivity)
    axes[0, i].set_ylabel("Energy (Hartree)")
    axes[0, i].set_xlabel("Bond length (Å)")
    axes[1, i].set_yscale("log")
    axes[1, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[1, i].set_ylabel("Energy error (Hartree)")
    axes[1, i].set_xlabel("Bond length (Å)")
    axes[2, i].set_ylim(0, 0.1)
    axes[2, i].set_ylabel("Spin squared")
    axes[2, i].set_xlabel("Bond length (Å)")
    fig.suptitle(f"{molecule_basename} ({nelectron}e, {norb}o) initial parameters")


filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf"
)
plt.savefig(filepath)
plt.close()

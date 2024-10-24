import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.lucj_lbfgsb_task import LUCJLBFGSBTask
from ffsim_numerics.lucj_linear_method_task import LUCJLinearMethodTask
from ffsim_numerics.params import LBFGSBParams, LinearMethodParams, LUCJParams

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 6, 6
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

start = 0.9
stop = 2.7
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
reference_bond_distance_range = np.linspace(
    start, stop, num=round((stop - start) / 0.05) + 1
)

connectivity = "square"
n_reps = 6
ftol = 1e-12
gtol = 1e-5
maxcors = [10, 20, 30]

tasks_lbfgsb = [
    LUCJLBFGSBTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        lbfgsb_params=LBFGSBParams(
            maxiter=1000,
            # maxfun=1000 * 1000,
            maxfun=1000 * 100,
            maxcor=maxcor,
            eps=1e-8,
            ftol=ftol,
            gtol=gtol,
        ),
    )
    for maxcor in maxcors
    for d in bond_distance_range
]
tasks_lm = [
    LUCJLinearMethodTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        linear_method_params=LinearMethodParams(
            maxiter=1000,
            lindep=1e-8,
            epsilon=1e-8,
            ftol=ftol,
            gtol=gtol,
            regularization=1e-4,
            variation=0.5,
            optimize_regularization=True,
            optimize_variation=True,
        ),
    )
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
data_lbfgsb = {}
for task in tasks_lbfgsb:
    filepath = (
        DATA_ROOT
        / "lucj_lbfgsb_bootstrap"
        / "post-bootstrap"
        / task.dirpath
        / "data.pickle"
    )
    with open(filepath, "rb") as f:
        data_lbfgsb[task] = pickle.load(f)
data_lm = {}
for task in tasks_lm:
    filepath = DATA_ROOT / "lucj_linear_method_parallel" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_lm[task] = pickle.load(f)
print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
    2, 2, figsize=(12, 9), layout="constrained"
)

# ax0.plot(
#     reference_curves_d_range,
#     hf_energies_reference,
#     "--",
#     label="HF",
#     color="blue",
# )
ax0.plot(
    reference_bond_distance_range,
    ccsd_energies_reference,
    "--",
    label="CCSD",
    color="orange",
)
ax0.plot(
    reference_bond_distance_range,
    fci_energies_reference,
    "-",
    label="FCI",
    color="black",
)

data = data_lm
tasks = tasks_lm
label = "LM"
marker = markers[0]
color = colors[0]
energies = [data[task]["energy"] for task in tasks]
errors = [data[task]["error"] for task in tasks]
spin_squares = [data[task]["spin_squared"] for task in tasks]
nits = [data[task]["nit"] for task in tasks]
ax0.plot(
    bond_distance_range,
    energies,
    f"{marker}{linestyles[0]}",
    label=label,
    color=color,
)
ax1.plot(
    bond_distance_range,
    errors,
    f"{marker}{linestyles[0]}",
    label=label,
    color=color,
)
ax2.plot(
    bond_distance_range,
    spin_squares,
    f"{marker}{linestyles[0]}",
    label=label,
    color=color,
)
ax3.plot(
    bond_distance_range,
    nits,
    f"{marker}{linestyles[0]}",
    label=label,
    color=color,
)

for maxcor, marker, color in zip(maxcors, markers[1:], colors[1:]):
    tasks = [
        LUCJLBFGSBTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            lbfgsb_params=LBFGSBParams(
                maxiter=1000,
                # maxfun=1000 * 1000,
                maxfun=1000 * 100,
                maxcor=maxcor,
                eps=1e-8,
                ftol=ftol,
                gtol=gtol,
            ),
        )
        for d in bond_distance_range
    ]
    data = data_lbfgsb
    label = f"L-BFGS-B maxcor={maxcor}"
    energies = [data[task]["energy"] for task in tasks]
    errors = [data[task]["error"] for task in tasks]
    spin_squares = [data[task]["spin_squared"] for task in tasks]
    nits = [data[task]["nit"] for task in tasks]
    ax0.plot(
        bond_distance_range,
        energies,
        f"{marker}{linestyles[0]}",
        label=label,
        color=color,
    )
    ax1.plot(
        bond_distance_range,
        errors,
        f"{marker}{linestyles[0]}",
        label=label,
        color=color,
    )
    ax2.plot(
        bond_distance_range,
        spin_squares,
        f"{marker}{linestyles[0]}",
        label=label,
        color=color,
    )
    ax3.plot(
        bond_distance_range,
        nits,
        f"{marker}{linestyles[0]}",
        label=label,
        color=color,
    )


ax0.legend()
ax0.set_ylabel("Energy (Hartree)")
ax0.set_xlabel("Bond length (Å)")
ax1.set_yscale("log")
ax1.axhline(1.6e-3, linestyle="--", color="gray")
ax1.set_ylabel("Energy error (Hartree)")
ax1.set_xlabel("Bond length (Å)")
ax2.set_ylim(0, 0.1)
ax2.set_ylabel("Spin squared")
ax2.set_xlabel("Bond length (Å)")
ax3.set_ylim(0, 1000)
ax3.set_ylabel("Number of iterations")
ax3.set_xlabel("Bond length (Å)")
fig.suptitle(
    f"{molecule_basename} ({nelectron}e, {norb}o), LUCJ {connectivity} L={n_reps}"
)


filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.svg"
)
plt.savefig(filepath)
plt.close()

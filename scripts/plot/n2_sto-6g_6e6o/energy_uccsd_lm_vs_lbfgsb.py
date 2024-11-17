import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.params import LBFGSBParams, LinearMethodParams, UCCSDParams
from ffsim_numerics.uccsd_lbfgsb_task import UCCSDLBFGSBTask
from ffsim_numerics.uccsd_linear_method_task import UCCSDLinearMethodTask

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

tasks_uccsd_lbfgsb = [
    UCCSDLBFGSBTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        uccsd_params=UCCSDParams(with_final_orbital_rotation=True),
        lbfgsb_params=LBFGSBParams(
            maxiter=1000,
            maxfun=1000 * 1000,
            maxcor=10,
            eps=1e-8,
            ftol=1e-12,
            gtol=1e-5,
        ),
    )
    for d in bond_distance_range
]
tasks_uccsd_lm = [
    UCCSDLinearMethodTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        uccsd_params=UCCSDParams(with_final_orbital_rotation=True),
        linear_method_params=LinearMethodParams(
            maxiter=1000,
            lindep=1e-8,
            epsilon=1e-8,
            ftol=1e-12,
            gtol=1e-5,
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
data_uccsd_lbfgsb = {}
for task in tasks_uccsd_lbfgsb:
    filepath = DATA_ROOT / "uccsd_lbfgsb_parallel" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_uccsd_lbfgsb[task] = pickle.load(f)
data_uccsd_lm = {}
for task in tasks_uccsd_lm:
    filepath = DATA_ROOT / "uccsd_linear_method_parallel" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_uccsd_lm[task] = pickle.load(f)
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

for tasks, data, label, marker, color in zip(
    [tasks_uccsd_lbfgsb, tasks_uccsd_lm],
    [data_uccsd_lbfgsb, data_uccsd_lm],
    ["UCCSD L-BFGS-B", "UCCSD LM"],
    markers,
    colors,
):
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
fig.suptitle(f"{molecule_basename} ({nelectron}e, {norb}o)")


filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf"
)
plt.savefig(filepath)
plt.close()

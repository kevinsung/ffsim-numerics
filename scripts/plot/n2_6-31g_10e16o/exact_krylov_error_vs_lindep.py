import os
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.exact_krylov_task import ExactKrylovTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.0

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

time_step_range = [1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
lindep_range = [1e-4, 1e-8, 1e-12]
n_steps = 50

molecule_filepath = (
    MOLECULES_CATALOG_DIR
    / "data"
    / "molecular_data"
    / f"{molecule_basename}_d-{bond_distance:.2f}.json.xz"
)
mol_data = ffsim.MolecularData.from_json(molecule_filepath, compression="lzma")

for time_step in time_step_range:
    data = {}
    for lindep in lindep_range:
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
        errors = ground_energies - mol_data.fci_energy
        assert all(errors > -1e-8)
        data[lindep] = np.abs(errors)

    markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    alphas = [0.5, 1.0]
    linestyles = ["--", ":"]

    fig, ax = plt.subplots(1, 1)

    for lindep, color in zip(lindep_range, colors):
        errors = data[lindep]
        ax.plot(range(2, n_steps + 3), errors, label=f"lindep={lindep}")

    ax.set_xticks(range(2, n_steps + 3, 6))
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Krylov space dimension")
    ax.set_ylabel(r"$|E - E_{\text{exact}}|$")
    ax.set_title(f"{molecule_name} {basis} ({nelectron}e, {norb}o)")

    filename = os.path.join(
        plots_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_time_step-{time_step}.svg",
    )
    plt.savefig(filename)
    plt.close()
    print(f"Saved figure to {filename}.")

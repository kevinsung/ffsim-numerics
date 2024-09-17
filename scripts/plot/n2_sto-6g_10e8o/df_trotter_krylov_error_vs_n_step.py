import os
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from ffsim_numerics.df_trotter_krylov_task import DoubleFactorizedTrotterKrylovTask
from ffsim_numerics.exact_krylov_task import ExactKrylovTask

DATA_ROOT = Path(os.environ.get("FFSIM_NUMERICS_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.0

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

time_step = 1e-1
krylov_n_steps = 50
trotter_n_steps_range = list(range(1, 6))
order = 1
lindep = 1e-12

molecule_filepath = (
    MOLECULES_CATALOG_DIR
    / "data"
    / "molecular_data"
    / f"{molecule_basename}_d-{bond_distance:.2f}.json.xz"
)
mol_data = ffsim.MolecularData.from_json(molecule_filepath, compression="lzma")

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
    errors = ground_energies - mol_data.fci_energy
    assert all(errors > 0)
    data[trotter_n_steps] = errors


task = ExactKrylovTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    time_step=time_step,
    n_steps=krylov_n_steps,
    initial_state="hartree-fock",
    lindep=lindep,
)
filepath = DATA_ROOT / "exact_krylov" / task.dirpath / "result.npy"
ground_energies = np.load(filepath)
exact_krylov_errors = ground_energies - mol_data.fci_energy
assert all(exact_krylov_errors > 0)


markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, ax = plt.subplots(1, 1)

ax.plot(range(2, krylov_n_steps + 3), exact_krylov_errors, label="exact")
for trotter_n_steps, color in zip(trotter_n_steps_range, colors):
    errors = data[trotter_n_steps]
    ax.plot(range(2, krylov_n_steps + 3), errors, label=f"n_steps={trotter_n_steps}")

ax.set_xticks(range(2, krylov_n_steps + 3, 6))
ax.set_yscale("log")
ax.legend()
ax.set_xlabel("Krylov space dimension")
ax.set_ylabel(r"$|E - E_{\text{exact}}|$")
ax.set_title(f"{molecule_name} {basis} ({nelectron}e, {norb}o), order {order}")

filename = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_order-{order}.svg"
)
plt.savefig(filename)
plt.close()
print(f"Saved figure to {filename}.")

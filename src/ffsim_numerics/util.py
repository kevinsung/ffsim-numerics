import os
import shutil


def interaction_pairs_spin_balanced(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    """Returns alpha-alpha and alpha-beta diagonal Coulomb interaction pairs."""
    if connectivity == "all-to-all":
        pairs_aa = None
        pairs_ab = None
    elif connectivity == "square":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb)]
    elif connectivity == "hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 2 == 0]
    elif connectivity == "heavy-hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 4 == 0]
    else:
        raise ValueError(f"Invalid connectivity: {connectivity}")
    return pairs_aa, pairs_ab


def interaction_pairs_spinless(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    """Returns diagonal Coulomb interation pairs."""
    if connectivity == "all-to-all":
        return None
    elif connectivity == "linear":
        return [(p, p + 1) for p in range(norb - 1)]
    raise ValueError(f"Invalid connectivity: {connectivity}")


def copy_data(task, src_data_dir: str, dst_data_dir: str, dirs_exist_ok: bool = False):
    """Copy task data to another directory."""
    src_dir = os.path.join(src_data_dir, task.dirpath)
    dst_dir = os.path.join(dst_data_dir, task.dirpath)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=dirs_exist_ok)

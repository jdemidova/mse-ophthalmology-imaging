from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _safe_dest(dest_dir: Path, name: str) -> Path:
    """Avoid overwriting if same filename appears twice."""
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}__{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _group_key(p: Path) -> str:
    """
    Group files by 'base name' (filename without the last extension),
    so img_name_01.png and img_name_01.jpg stay together.
    """
    return p.stem  # last suffix removed only, for .png/.jpg pairs


# seed=13 sets the random number generator’s starting state, so the shuffle
# (and therefore train/test split) is reproducible
#
# - With --seed 13: every run on the same file set -> same split
# - With a different seed (e.g. --seed 7) -> different split
# - With seed=None -> uses system entropy -> split changes each run
#
# Results are debugging and fair comparisons between experiments


def split_folder_grouped(
    a_dir: Path,
    train_ratio: float = 0.8,
    seed: int | None = 13,
    move: bool = False,
) -> None:
    a_dir = a_dir.resolve()
    if not a_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {a_dir}")

    train_dir = a_dir / "train"
    test_dir = a_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    files = [
        p
        for p in a_dir.iterdir()
        if p.is_file() and p.parent == a_dir and p.name not in {".DS_Store"}
    ]

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        groups[_group_key(p)].append(p)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_train = int(len(keys) * train_ratio)
    train_keys, test_keys = keys[:n_train], keys[n_train:]

    op = shutil.move if move else shutil.copy2

    def put_group(k: str, dest_dir: Path) -> None:
        for p in groups[k]:
            op(str(p), str(_safe_dest(dest_dir, p.name)))

    for k in train_keys:
        put_group(k, train_dir)
    for k in test_keys:
        put_group(k, test_dir)

    train_files = sum(len(groups[k]) for k in train_keys)
    test_files = sum(len(groups[k]) for k in test_keys)

    print(
        f"Groups: {len(keys)} | Train groups: {len(train_keys)} | Test groups: {len(test_keys)}\n"
        f"Files:  {len(files)} | Train files:  {train_files} | Test files:  {test_files}\n"
        f"Train dir: {train_dir}\nTest dir:  {test_dir}\n"
        f"Mode: {'MOVE' if move else 'COPY'} | Seed: {seed}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="Path to folder 'a' containing files to split")
    ap.add_argument("--ratio", type=float, default=0.8, help="Train ratio by GROUPS (default 0.8)")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed (default 13)")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    args = ap.parse_args()

    split_folder_grouped(Path(args.a), train_ratio=args.ratio, seed=args.seed, move=args.move)

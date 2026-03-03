from pathlib import Path


def find_repo_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            return str(d)
    raise FileNotFoundError("Can't find repo root (.git/pyproject.toml).")


ROOT = find_repo_root()
DATA_DIR = ROOT + "/data"

ROOT, DATA_DIR

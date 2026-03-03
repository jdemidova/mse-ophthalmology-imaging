from __future__ import annotations
import numpy as np
import torch
import scipy.ndimage as ndi

from typing import Iterable, Union, Optional


# For masks -> integer labels, not float [0,1].
# If masks are binary 0/255, this turns them into {0,1} ints
def mask_to_long_tensor(pil_mask):
    arr = np.array(pil_mask)
    return torch.from_numpy((arr > 0).astype("int64"))


def get_mask_gray(
    pil_mask,
    mask_values: Union[int, Iterable[int]],
    keep_largest: Optional[int] = 2,
) -> torch.Tensor:
    """
    PIL mask -> int64 tensor (H,W) in {0,1} for mask only.

    mask_values:
      - int: exact pixel value for mask
      - iterable[int]: allow multiple mask encodings (e.g., across datasets)

    keep_largest:
      - None: keep all mask components
      - k (int): keep k-largest connected components (requires scipy)
    """
    arr = np.array(pil_mask.convert("L"))
    if isinstance(mask_values, int):
        vals = {mask_values}
    else:
        vals = set(int(v) for v in mask_values)

    m = np.isin(arr, list(vals))

    if keep_largest is not None and keep_largest > 0:
        if ndi is None:
            raise Exception(
                "keep_largest requires scipy (pip install scipy) or set keep_largest=None"
            )
        lab, n = ndi.label(m)
        if n > keep_largest:
            sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
            keep = np.argsort(sizes)[-keep_largest:] + 1
            m = np.isin(lab, keep)

    return torch.from_numpy(m.astype(np.int64))


def get_mask_rgb(pil_mask, mask_rgb=(0, 255, 0), tol=0) -> torch.Tensor:
    """
    PIL RGB mask -> int64 (H,W) {0,1} where pixels match mask_rgb within tol.
    tol=0 means exact match.
    """
    arr = np.array(pil_mask.convert("RGB")).astype(np.int16)
    tgt = np.array(mask_rgb, dtype=np.int16)
    diff = np.abs(arr - tgt).max(axis=-1)
    m = diff <= tol
    return torch.from_numpy(m.astype(np.int64))

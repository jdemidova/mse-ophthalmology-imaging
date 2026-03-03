import importlib


def test_imports():
    importlib.import_module("src")


def test_torch_available():
    import torch

    x = torch.randn(2, 3)
    assert x.shape == (2, 3)


# for dataset:
# load one sample
# mask values in expected set

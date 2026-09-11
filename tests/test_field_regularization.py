import torch

from schnetpack.task import center_moleculewise, elementwise_charge_bounds


def test_center_moleculewise():
    x = torch.tensor([1.0, 3.0, -2.0, 1.0, 4.0])
    idx_m = torch.tensor([0, 0, 1, 1, 1])
    n_atoms = torch.tensor([2, 3])

    centered = center_moleculewise(x, idx_m, n_atoms)

    assert torch.allclose(centered[:2].mean(), torch.tensor(0.0))
    assert torch.allclose(centered[2:].mean(), torch.tensor(0.0))


def test_elementwise_charge_bounds_accepts_yaml_string_keys():
    Z = torch.tensor([1, 8, 26, 6])
    bounds = elementwise_charge_bounds(
        Z, {"1": 0.8, "8": 1.5, "26": 3.2, "default": 2.0}
    )

    assert torch.allclose(bounds, torch.tensor([0.8, 1.5, 3.2, 2.0]))

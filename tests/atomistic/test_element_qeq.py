import math

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from schnetpack import properties as p
from schnetpack.atomistic import ElementQEqCharges


def inputs():
    return {
        p.Z: torch.tensor([8, 1, 1, 6, 8, 1, 1, 1, 1]),
        p.R: torch.tensor([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, 0.],
                           [3., 0., 0.], [4.4, 0., 0.], [4.7, .9, 0.],
                           [2.6, 1., 0.], [2.6, -.5, .87], [2.6, -.5, -.87]],
                          dtype=torch.double),
        p.idx_m: torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        p.n_atoms: torch.tensor([3, 6]),
        p.total_charge: torch.tensor([0., 1.], dtype=torch.double),
    }


def head(offdiag=True, **kwargs):
    model = ElementQEqCharges(use_offdiag=offdiag, **kwargs).double()
    with torch.no_grad():
        model.chi_table.weight[1] = -.2
        model.chi_table.weight[6] = .1
        model.chi_table.weight[8] = .8
    return model


@pytest.mark.parametrize('offdiag', [False, True])
@pytest.mark.parametrize('column_charge', [False, True])
def test_constraint_stationarity_and_batch(offdiag, column_charge):
    data = inputs()
    if column_charge:
        data[p.total_charge] = data[p.total_charge][:, None]
    model = head(offdiag, phi_key='phi')
    data['phi'] = torch.linspace(-.1, .2, 9, dtype=torch.double)
    out = model(dict(data))
    for m in range(2):
        mask = data[p.idx_m] == m
        q = out['charges'][mask]
        torch.testing.assert_close(q.sum(), data[p.total_charge].flatten()[m])
        A = torch.diag(out['Jii_qeq'][mask])
        if offdiag:
            r = torch.cdist(data[p.R][mask], data[p.R][mask])
            eye = torch.eye(len(r), dtype=torch.bool)
            safe = r.masked_fill(eye, 1.)
            A = A + (torch.erf(safe / math.sqrt(2)) / safe).masked_fill(eye, 0.)
            assert torch.linalg.eigvalsh(A).min() > 0
        gradient = A @ q + out['chi_qeq'][mask] + data['phi'][mask]
        torch.testing.assert_close(gradient, gradient.mean().expand_as(gradient))
        single = {p.Z: data[p.Z][mask], p.R: data[p.R][mask],
                  p.idx_m: torch.zeros(mask.sum(), dtype=torch.long),
                  p.n_atoms: mask.sum().reshape(1),
                  p.total_charge: data[p.total_charge].flatten()[m:m+1], 'phi': data['phi'][mask]}
        torch.testing.assert_close(q, model(single)['charges'])
    assert 'gamma_qeq' in model.model_outputs


def test_position_derivatives_and_coincident_atoms():
    data = inputs()
    model = head(learn_widths=True)
    r = data[p.R].requires_grad_()
    fn = lambda r: model({**data, p.R: r})['charges']
    assert gradcheck(fn, (r,))
    assert gradgradcheck(fn, (r,))
    r = torch.zeros_like(r, requires_grad=True)
    assert gradcheck(fn, (r,))
    assert gradgradcheck(fn, (r,))
    fn(r).square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_invariances_and_geometry_response():
    data = inputs()
    model = head()
    q = model(dict(data))['charges']
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.double)
    torch.testing.assert_close(q, model({**data, p.R: data[p.R] @ rotation + 7})['charges'])
    perm = torch.tensor([2, 0, 1, 8, 6, 4, 5, 3, 7])
    torch.testing.assert_close(q[perm], model({**data, p.Z: data[p.Z][perm], p.R: data[p.R][perm]})['charges'])
    with torch.no_grad():
        model.chi_table.weight.add_(2.)
    torch.testing.assert_close(q, model(dict(data))['charges'])
    assert not torch.allclose(q, model({**data, p.R: data[p.R] * 1.1})['charges'])


def test_reject_periodic_and_large_gamma():
    data = inputs()
    data[p.pbc] = torch.tensor([[True, True, True], [False, False, False]])
    with pytest.raises(ValueError, match='PBC'):
        head()(data)
    model = head(init_gamma=1389.35456)
    assert torch.isfinite(model.gamma_raw)

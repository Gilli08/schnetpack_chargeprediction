"""Small, reproducible charge fit and full-PaiNN backward smoke test on real data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from schnetpack import properties as p
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import ASENeighborList, CastTo32

parser = argparse.ArgumentParser()
parser.add_argument('--database', default='/pool/m_gillhofer/phd/BuRNN_dqdr/MeOH_H2O/training_QEq/MeOH_H2O.db')
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--output', default='/tmp/element_qeq_check.json')
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(17)
root = Path(__file__).resolve().parents[2]
with initialize_config_dir(config_dir=str(root / 'src/schnetpack/configs'), version_base=None):
    cfg = compose(config_name='train', overrides=[f'hydra.searchpath=[file://{Path(__file__).parent}]', 'experiment=train_1'])
head = instantiate(cfg.model.output_modules[-1])
rows = []
with connect(args.database) as db:
    count = db.count()
    metadata = db.metadata
    for rowid in np.random.default_rng(17).choice(np.arange(1, count + 1), 40, replace=False):
        row = db.get(int(rowid))
        atoms = row.toatoms()
        d = {p.idx: torch.tensor([int(rowid)]), p.Z: torch.tensor(atoms.numbers), p.R: torch.tensor(atoms.positions, dtype=torch.float32),
             p.cell: torch.tensor(atoms.cell.array, dtype=torch.float32)[None],
             p.pbc: torch.tensor(atoms.pbc)[None], p.n_atoms: torch.tensor([len(atoms)])}
        for key in ['charges', 'total_charge', 'spin_multiplicity', 'V_BuRNN', 'F_BuRNN']:
            if key in row.data:
                d[key] = torch.as_tensor(np.array(row.data[key]), dtype=torch.float32)
        rows.append(d)
print('database', count, 'metadata', metadata, 'shapes', {k: list(v.shape) for k,v in rows[0].items()}, flush=True)
train = _atoms_collate_fn(rows[:32]); val = _atoms_collate_fn(rows[32:])

def mse(batch):
    q = head(dict(batch))['charges']
    return (q - batch['charges'].reshape(-1)).square().mean()

initial = float(mse(val).detach().sqrt())
opt = torch.optim.Adam(head.parameters(), lr=.02)
for step in range(args.steps):
    opt.zero_grad()
    loss = mse(train)
    loss.backward()
    opt.step()
final = float(mse(val).detach().sqrt())
q = head(dict(val))['charges']
sums = torch.zeros(len(val[p.n_atoms])).index_add_(0, val[p.idx_m], q)
# Complete configured energy, forces and charge losses must support backward.
transforms = [ASENeighborList(cutoff=5.), CastTo32()]
processed = []
for row in rows[:2]:
    row = dict(row)
    for transform in transforms:
        row = transform(row)
    processed.append(row)
batch = _atoms_collate_fn(processed)
model = instantiate(cfg.model)
model.do_postprocessing = False
model.train()
out = model(dict(batch))
loss = sum(weight * (out[key] - batch[key].reshape_as(out[key])).square().mean()
           for key, weight in [('V_BuRNN', .05), ('F_BuRNN', .7), ('charges', .15)])
loss.backward()
assert torch.isfinite(loss)
assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
result = dict(database_count=count, train_structures=32, validation_structures=8, steps=args.steps,
              initial_validation_charge_rmse_e=initial, final_validation_charge_rmse_e=final,
              train_charge_rmse_e=float(mse(train).detach().sqrt()),
              max_charge_constraint_error_e=float((sums-val['total_charge'].reshape(-1)).abs().max().detach()),
              full_model_backward_finite=True)
Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
assert final < initial

# Element QEq methanol–water trial

The original diagonal closed form and off-diagonal KKT signs are correct for
minimizing `chi.q + phi.q + 0.5*q.T*A*q` subject to `sum(q)=Q`. However, a KKT
solution is only a stationary point unless A is positive on charge-conserving
variations. The original defaults can violate this: for two nearly coincident
atoms with width 1, the off-diagonal approaches 0.797885 while the diagonal is
softplus(0) = 0.693148. The charge-transfer mode then has negative curvature.

The revised off-diagonal head adds the Gaussian self hardness
`gamma * sqrt(2/pi) / width` to the positive learned diagonal. With positive
gamma and no cutoff, the Gaussian matrix is positive semidefinite and its sum
with the positive learned diagonal is positive definite. The new defaults use
this self term and no cutoff. Explicit `add_gaussian_self=False` selects the old
independent diagonal for state-dict reconstruction; old full-object checkpoints
without the attribute retain the old diagonal. Do not reinterpret old fitted
parameters as the new model without retraining. Hard cutoffs remain a legacy
option with a warning: they can break convexity and continuity.

Other fixes: total-charge input accepts both [B] and [B,1], gamma is included in
model outputs, large gamma initialization avoids exponential overflow, and the
screened kernel has finite first/second derivatives at coincident positions.
Periodic inputs now fail explicitly; this dense head implements isolated
structures, not periodic electrostatics.

The variational framework is the standard QEq approach of
[Rappé and Goddard (1991)](https://pubs.acs.org/doi/10.1021/j100161a070).
Charge-transfer/polarizability limitations are discussed in
[Chen and Martínez (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2673188/).
The implementation findings above come from inspection and numerical tests of
this repository, not from claims that the references validate this code.

## Configurations

`experiment/train_1.yaml` through `train_5.yaml` replace the EMLE parameter and
QEq heads with ElementQEqCharges. All five now use MeOH_H2O.db and the original
3799/712 train/validation sizes (the supplied files 2–5 pointed to a transition
metal database). Unique seeds and run IDs distinguish runs. The width loss and
width unit entry are removed; the remaining energy/force/charge loss weights
stay 0.05/0.70/0.15. They need not sum to one.

This trial fits chi(Z) and residual hardness(Z) for H/C/O, using width=1 Å,
gamma=1 fixed and residual hardness >=0.1 in normalized QEq units. Fixing gamma
removes the arbitrary common energy scale of a zero-field charge-only fit;
fixing widths provides a simple baseline. An additive common shift of chi is
still unidentifiable and does not affect charges. Hyperparameters are initial
trial choices, not calibrated physical parameters.

The head uses no PaiNN features. Its charge loss trains only the element
tables; the energy/force losses train PaiNN and Atomwise independently. Geometry
dependence comes from the dense Coulomb matrix. Diagonal mode would assign all
hydrogens in a structure the same charge, making it a weak baseline for methanol.
Element-only chi remains less expressive than the EMLE geometry-dependent head.

`phi_key=null` intentionally defines a zero-field charge fit. The previous EMLE
head uses kJ/mol/e, while these new charge-fit parameters use normalized units.
Do not feed physical `phi_static` directly into this configuration. To couple it
to external fields, establish the physical hardness/chi/gamma scale and a
consistent polarization energy and forces. Charge labels alone do not validate
that response. The head does not provide EMLE's core/valence, vacuum/polarized,
or embedding-energy outputs; consumers of those outputs need adaptation.

The total-charge constraint is per database structure, allowing charge transfer
between methanol and water molecules in that structure. There are no separate
molecular neutrality constraints. Dense solves scale cubically with atom count.

## Run against this checkout

The environment used for verification is BuRNNpack_MACE; its library directory
must precede the system C++ libraries. Explicit PYTHONPATH ensures the checkout
is used instead of the older installed SchNetPack copy.

```bash
export LD_LIBRARY_PATH=/pool/m_gillhofer/miniconda3/envs/BuRNNpack_MACE/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/pool/m_gillhofer/software/BuRNN/schnetpack_chargeprediction/src
export MPLCONFIGDIR=/tmp/mpl-element-qeq
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
/pool/m_gillhofer/miniconda3/envs/BuRNNpack_MACE/bin/python -m pytest tests/atomistic/test_element_qeq.py -q
/pool/m_gillhofer/miniconda3/envs/BuRNNpack_MACE/bin/python examples/element_qeq/check_methanol_water.py
/pool/m_gillhofer/miniconda3/envs/BuRNNpack_MACE/bin/python src/scripts/spktrain --config-dir=examples/element_qeq experiment=train_1
```

Run commands above from the repository root. For the external experiment folder,
use its absolute parent directory as `--config-dir`. Repeat with train_2 through
train_5 as needed. These are fresh training runs, not EMLE checkpoint resumes.

The diagnostic script performs a small charge fit on 32 randomly selected
structures, evaluates on 8 held-out structures and checks a backward pass through
the complete configured PaiNN energy/force/charge model. Its higher learning
rate and small sample make it a smoke test, not a full accuracy benchmark.
Before claiming stable MD, train on the full split, compare held-out charge
errors by element/chemical environment against EMLE, validate the actual
simulation coupling with finite differences, and run matched trajectories with
charge extremes and energy drift monitored. Stable standalone PaiNN MD would
not by itself validate this independent charge head.

## Observed smoke-test results

On 2026-09-11, all seven unit tests passed. In the 32-train/8-validation
real-data fit (300 Adam steps, learning rate 0.02), held-out charge RMSE fell
from 0.484210 to 0.093159 e; training RMSE was 0.087274 e. Maximum total-charge
error was 3.58e-7 e in float32. The full PaiNN energy/force/charge backward pass
was finite. Exact values are in `check_results.json`. These results demonstrate
learnability, not EMLE-level accuracy or trajectory stability.

A one-epoch CLI smoke run with 32 training / 8 validation structures and one
batch per phase also completed training, validation, best-checkpoint reload,
testing and model export. Its log is `/tmp/element-qeq-checkpoint-smoke.log`;
its exported model is an untrained smoke artifact, not an MD-ready potential.

The five external YAML files were updated. Their original EMLE versions are in
`/pool/m_gillhofer/phd/BuRNN_dqdr/MeOH_H2O/training_QEqstaticParam/experiment/emle_backup_20260911_095607`.

The earlier `fast_dev_run=true` CLI attempt completed its train/validation batch
but failed at the CLI's unconditional `test(ckpt_path="best")`, because Lightning
suppresses checkpoints in that mode. The successful regular one-epoch run above
verifies the complete path; use limited batches with one epoch for CLI smoke
tests instead of `fast_dev_run`.

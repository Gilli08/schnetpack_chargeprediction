"""Export a separate full-module model with an explicit physical QEq scale.

--coulomb-normalize assumes distances in Angstrom, charges in e, and interprets
ElementQEqCharges' Gaussian kernel as physical Coulomb electrostatics. This is a
modeling convention, not calibration against reference polarization data.
Only load trusted full-module checkpoints.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from schnetpack.atomistic.atomwise import ElementQEqCharges, K_E_KJMOL_ANG_E2


def configure(model, *, coulomb_normalize=False, energy_scale=None):
    records = []
    for head in model.modules():
        if not isinstance(head, ElementQEqCharges):
            continue
        gamma = float((torch.nn.functional.softplus(head.gamma_raw)
                       if head.gamma_positive else head.gamma_raw).detach())
        if coulomb_normalize:
            if not head.use_offdiag or not math.isfinite(gamma) or gamma <= 0:
                raise ValueError('Coulomb normalization requires positive off-diagonal gamma.')
            if head.offdiag_cutoff is not None or not getattr(head, 'add_gaussian_self', False):
                raise ValueError('Use the untruncated, Gaussian-self-stabilized model.')
            scale = K_E_KJMOL_ANG_E2 / gamma
        else:
            scale = energy_scale
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Specify a finite positive physical energy scale.')
        head.qeq_energy_scale = scale
        head.charges_vac_key = getattr(head, 'charges_vac_key', 'charges_vac')
        head.embedding_energy_key = getattr(head, 'embedding_energy_key', 'qeq_embedding_energy')
        head.polarize_key = getattr(head, 'polarize_key', 'qeq_polarize')
        for key in (head.charges_vac_key, head.embedding_energy_key):
            if key not in head.model_outputs:
                head.model_outputs.append(key)
        records.append(dict(gamma=gamma, qeq_energy_scale=scale,
                            convention='physical Gaussian Coulomb, Angstrom/e' if coulomb_normalize else 'explicit scale'))
    if not records:
        raise ValueError('No ElementQEqCharges head found.')
    model.collect_outputs()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--coulomb-normalize', action='store_true')
    group.add_argument('--energy-scale', type=float)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    model = torch.load(args.source, map_location='cpu', weights_only=False)
    records = configure(model, coulomb_normalize=args.coulomb_normalize, energy_scale=args.energy_scale)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.destination)
    report = dict(source=str(args.source.resolve()), heads=records)
    args.destination.with_name(args.destination.name + '.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

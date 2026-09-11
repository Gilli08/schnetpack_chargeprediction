from typing import Sequence, Union, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.nn as snn
import schnetpack.properties as properties
import math
import warnings

__all__ = [
    "Atomwise",
    "DipoleMoment",
    "Polarizability",
    "Charges",
    "SpinCharges",
    "QEqCharges",
    "ElementQEqCharges",
    "EMLEStaticHead",
    "EMLEStaticParamHead",
    "EMLEQEqStatic",
    "ExternalCoulombEmbedding",
]

K_E_KJMOL_NM_E2 = 138.935456
K_E_KJMOL_ANG_E2 = K_E_KJMOL_NM_E2 * 10.0

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        inputs[self.output_key] = y
        return inputs
 

import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.nn as snn
import schnetpack.properties as properties

class ElementQEqCharges(nn.Module):
    """
    Element-table QEq minimizing chi.q + phi.q + 0.5*q.T*A*q at fixed Q.

    Diagonal mode assigns identical charges to identical elements in a structure.
    Off-diagonal mode uses erf(r / sqrt(a_i**2 + a_j**2)) / r.
    With add_gaussian_self=True, positive gamma and no cutoff, A is the
    positive-semidefinite Gaussian Coulomb matrix plus positive learned hardness.
    This guarantees a unique minimum, including for coincident atoms.

    Distances and widths share units; gamma has hardness times distance units.
    chi and phi must use consistent energy/charge units. No Coulomb conversion
    factor is inserted automatically. This head predicts charges only: a separate
    energy module and its position derivatives are needed for electrostatic forces.
    Dense, nonperiodic structures only; the charge constraint applies to each
    structure, not each chemical molecule within a mixture.

    add_gaussian_self=False reproduces the historical independent diagonal;
    it does not guarantee convexity. Existing fitted off-diagonal models must
    explicitly select that legacy setting when reconstructing from a state dict.
    """

    def __init__(
        self,
        charges_key="charges",
        chi_key="chi_qeq",
        hardness_key="Jii_qeq",
        max_z=100,
        eps_hardness=1e-6,
        correct_charges=True,
        use_offdiag=True,
        offdiag_cutoff=None,
        learn_gamma: bool = True,
        init_gamma: float = 1.0,
        gamma_positive: bool = True,
        gamma_key: str = "gamma_qeq",
        # offdiag: Gaussian-screened Coulomb using element widths a_Z (learnable or fixed)
        learn_widths=False,
        init_width=1.0,   # in distance units used internally (Å in SPK inputs)
        init_chi=0.0,
        init_logJ=0.0,
        phi_key=None,     # if provided, use inputs[phi_key], else 0
        add_gaussian_self=True,
    ):
        super().__init__()
        self.charges_key = charges_key
        self.chi_key = chi_key
        self.hardness_key = hardness_key
        self.max_z = max_z
        self.eps_hardness = eps_hardness
        self.correct_charges = correct_charges
        self.use_offdiag = use_offdiag
        self.offdiag_cutoff = offdiag_cutoff
        self.learn_widths = learn_widths
        self.phi_key = phi_key
        self.add_gaussian_self = add_gaussian_self
        if eps_hardness <= 0 or init_width <= 0:
            raise ValueError("eps_hardness and init_width must be positive.")
        if use_offdiag and add_gaussian_self and not gamma_positive:
            raise ValueError("Gaussian self hardness requires positive gamma.")
        if use_offdiag and offdiag_cutoff is not None:
            warnings.warn(
                "A hard QEq cutoff makes charges discontinuous and can destroy "
                "convexity. Use offdiag_cutoff=None for MD.", UserWarning
            )

        self.chi_table = nn.Embedding(max_z + 1, 1)
        self.logJ_table = nn.Embedding(max_z + 1, 1)
        nn.init.constant_(self.chi_table.weight, init_chi)
        nn.init.constant_(self.logJ_table.weight, init_logJ)

        if use_offdiag:
            # widths a_Z used in erf screening: erf(R / sqrt(a_i^2+a_j^2)) / R
            self.width_table = nn.Embedding(max_z + 1, 1)
            nn.init.constant_(self.width_table.weight, init_width)
            if not learn_widths:
                for p in self.width_table.parameters():
                    p.requires_grad_(False)
                    
        self.learn_gamma = learn_gamma
        self.gamma_positive = gamma_positive
        self.gamma_key = gamma_key

        # store a raw parameter; we map it -> gamma in forward
        gamma_raw = torch.tensor(float(init_gamma))
        if gamma_positive:
            # inverse softplus so that softplus(raw) ~= init_gamma
            # avoid init_gamma<=0
            init_gamma_safe = max(float(init_gamma), 1e-6)
            gamma_raw = torch.tensor(init_gamma_safe) + torch.log(-torch.expm1(-torch.tensor(init_gamma_safe)))

        self.gamma_raw = nn.Parameter(gamma_raw)

        if not learn_gamma:
            self.gamma_raw.requires_grad_(False)

        self.model_outputs = [charges_key, chi_key, hardness_key, gamma_key]

    def _phi(self, inputs, like):
        if self.phi_key is not None and self.phi_key in inputs:
            phi = inputs[self.phi_key]
            if phi.dim() == 1:
                phi = phi.unsqueeze(-1)
            return phi.to(like)
        return torch.zeros_like(like)

    def forward(self, inputs):
        Z = inputs[properties.Z].long()              # (N_atoms,)
        idx_m = inputs[properties.idx_m]             # (N_atoms,)
        natoms = inputs[properties.n_atoms]          # (N_mols,)
        maxm = natoms.numel()
        if self.use_offdiag and properties.pbc in inputs and inputs[properties.pbc].any():
            raise ValueError("ElementQEqCharges off-diagonal mode does not support PBC.")
        gamma = F.softplus(self.gamma_raw) if self.gamma_positive else self.gamma_raw

        chi = self.chi_table(Z)                      # (N_atoms,1)
        Jii = F.softplus(self.logJ_table(Z)) + self.eps_hardness

        if self.use_offdiag and getattr(self, "add_gaussian_self", False):
            widths = self.width_table(Z).abs().clamp_min(1e-6)
            Jii = Jii + gamma * math.sqrt(2.0 / math.pi) / widths

        phi = self._phi(inputs, chi)                 # (N_atoms,1) or 0
        g = chi + phi                                # (N_atoms,1)

        if properties.total_charge in inputs:
            Q = inputs[properties.total_charge].reshape(maxm, 1).to(chi)  # (N_mols,1)
        else:
            Q = torch.zeros((maxm, 1), device=chi.device, dtype=chi.dtype)

        if not self.use_offdiag:
            # ---- diagonal closed form
            invJ = 1.0 / Jii
            A = snn.scatter_add(invJ, idx_m, dim_size=maxm)                 # (N_mols,1)
            B = snn.scatter_add(g * invJ, idx_m, dim_size=maxm)             # (N_mols,1)
            lam = -(Q + B) / A                                              # (N_mols,1)
            q = -(g + lam[idx_m]) / Jii                                     # (N_atoms,1)

        else:
            # ---- off-diagonal: build per-molecule dense systems and solve
            # NOTE: This is a simple implementation. For speed you can batch/pack.
            positions = inputs[properties.R]          # (N_atoms,3)
            q_list = []
            start = 0
            for m in range(maxm):
                n = int(natoms[m].item())
                sl = slice(start, start + n)

                Zm = Z[sl]
                gm = g[sl].squeeze(-1)                # (n,)
                Jiim = Jii[sl].squeeze(-1)            # (n,)
                Qm = Q[m].squeeze(-1)                 # scalar

                Rm = positions[sl]                    # (n,3)
                A = torch.diag(Jiim)
                widths = self.width_table(Zm).squeeze(-1).abs().clamp_min(1e-6)
                sigma = torch.sqrt(widths[:, None] ** 2 + widths[None, :] ** 2)
                # Use a Taylor series near zero to keep first and second
                # derivatives finite, even on the diagonal/coincident atoms.
                r2 = ((Rm[:, None, :] - Rm[None, :, :]) ** 2).sum(-1)
                t = r2 / sigma.square()
                small = t < 1e-6
                safe_t = t.clamp_min(1e-6)
                regular = torch.erf(torch.sqrt(safe_t)) / (sigma * torch.sqrt(safe_t))
                series = 2.0 / math.sqrt(math.pi) / sigma * (1 - t / 3 + t.square() / 10)
                Jij = torch.where(small, series, regular)
                eye = torch.eye(n, device=Rm.device, dtype=torch.bool)
                Jij = Jij.masked_fill(eye, 0.0)

                if self.offdiag_cutoff is not None:
                    Jij = Jij * (r2 <= self.offdiag_cutoff ** 2)

                A = A + gamma * Jij
                
                # KKT system
                K = torch.zeros((n + 1, n + 1), device=A.device, dtype=A.dtype)
                K[:n, :n] = A
                K[:n, n] = 1.0
                K[n, :n] = 1.0
                K[n, n] = 0.0

                b = torch.zeros((n + 1,), device=A.device, dtype=A.dtype)
                b[:n] = -gm
                b[n] = Qm

                x = torch.linalg.solve(K, b)          # (n+1,)
                qm = x[:n].unsqueeze(-1)              # (n,1)
                q_list.append(qm)

                start += n

            q = torch.cat(q_list, dim=0)              # (N_atoms,1)

        if self.correct_charges:
            # tiny correction to enforce exact sum(q)=Q per molecule
            q_sum = snn.scatter_add(q, idx_m, dim_size=maxm)
            dq = (Q - q_sum) / natoms.unsqueeze(-1)
            q = q + dq[idx_m]

        inputs[self.chi_key] = chi.squeeze(-1)
        inputs[self.hardness_key] = Jii.squeeze(-1)
        inputs[self.charges_key] = q.squeeze(-1)
        
        inputs[self.gamma_key] = gamma.detach()  # diagnostic output
        return inputs
class QEqCharges(nn.Module):
    """
    QEq layer with optional off-diagonal couplings.

    Diagonal mode:
        E(q) = sum_i [ chi_i q_i + 1/2 Jii_i q_i^2 ] + sum_i q_i phi_i
        subject to sum_i q_i = Q_total

    Off-diagonal mode:
        E(q) = sum_i [ chi_i q_i ] + sum_i q_i phi_i
             + 1/2 sum_i Jii_i q_i^2
             + 1/2 gamma sum_{i!=j} Jij_ij q_i q_j
        subject to sum_i q_i = Q_total

    where Jij is a screened Coulomb kernel:
        Jij = erf(Rij / sigma_ij) / Rij
        sigma_ij = sqrt(sigma_i^2 + sigma_j^2)

    Notes
    -----
    - Keeps your existing Gaussian self-energy stabilization on the diagonal.
    - In offdiag mode, the same sigma can be reused for both self-term and Jij screening.
    - All terms remain conservative since charges come from minimizing an energy.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden=None,
        n_layers: int = 2,
        activation=F.silu,
        charges_key="charges",
        chi_key="chi_qeq",
        hardness_key="Jii_qeq",
        phi_key=None,
        eps_hardness: float = 1e-6,
        correct_charges: bool = True,

        # --- Behler-style diagonal stabilization ---
        use_gaussian_self_energy: bool = True,
        sigma_min: float = 0.3,
        sigma_max: float = 3.0,
        predict_sigma: bool = True,
        sigma_key: str = "sigma_qeq",
        jii_min: float = 0.0,

        # --- NEW: off-diagonal QEq ---
        use_offdiag: bool = False,
        offdiag_cutoff: Optional[float] = 10.0,
        gamma_key: str = "gamma_qeq",
        init_gamma: float = 1.0,
        gamma_positive: bool = True,
        learn_gamma: bool = True,
    ):
        super().__init__()
        self.charges_key = charges_key
        self.chi_key = chi_key
        self.hardness_key = hardness_key
        self.phi_key = phi_key
        self.eps_hardness = eps_hardness
        self.correct_charges = correct_charges

        self.use_gaussian_self_energy = use_gaussian_self_energy
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.predict_sigma = predict_sigma
        self.sigma_key = sigma_key
        self.jii_min = float(jii_min)

        self.use_offdiag = use_offdiag
        self.offdiag_cutoff = offdiag_cutoff
        self.gamma_key = gamma_key
        self.gamma_positive = gamma_positive
        self.learn_gamma = learn_gamma

        self.chi_net = spk.nn.build_mlp(
            n_in=n_in, n_out=1, n_hidden=n_hidden, n_layers=n_layers, activation=activation
        )
        self.jii_net = spk.nn.build_mlp(
            n_in=n_in, n_out=1, n_hidden=n_hidden, n_layers=n_layers, activation=activation
        )

        # sigma used for Gaussian self-energy and, in offdiag mode, for screened Jij
        if self.use_gaussian_self_energy or self.use_offdiag:
            if self.predict_sigma:
                self.sigma_net = spk.nn.build_mlp(
                    n_in=n_in, n_out=1, n_hidden=n_hidden, n_layers=n_layers, activation=activation
                )

        gamma_raw = torch.tensor(float(init_gamma))
        if gamma_positive:
            init_gamma_safe = max(float(init_gamma), 1e-6)
            gamma_raw = torch.log(torch.exp(torch.tensor(init_gamma_safe)) - 1.0)

        self.gamma_raw = nn.Parameter(gamma_raw)
        if not learn_gamma:
            self.gamma_raw.requires_grad_(False)

        self.model_outputs = [charges_key, chi_key, hardness_key]
        if self.use_gaussian_self_energy or self.use_offdiag:
            self.model_outputs.append(self.sigma_key)
        if self.use_offdiag:
            self.model_outputs.append(self.gamma_key)

    def _get_phi(self, inputs, like):
        if self.phi_key is not None and self.phi_key in inputs:
            phi = inputs[self.phi_key]
            if phi.dim() == 1:
                phi = phi.unsqueeze(-1)
            return phi.to(like)
        return torch.zeros_like(like)

    def _get_sigma(self, l0, inputs):
        if self.predict_sigma:
            sigma_raw = self.sigma_net(l0)
            sigma = F.softplus(sigma_raw) + self.sigma_min
        else:
            sigma = inputs[self.sigma_key]
            if sigma.dim() == 1:
                sigma = sigma.unsqueeze(-1)

        sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        return sigma

    def _get_gamma(self):
        return F.softplus(self.gamma_raw) if self.gamma_positive else self.gamma_raw

    def forward(self, inputs):
        l0 = inputs["scalar_representation"]         # (N, n_in)
        idx_m = inputs[properties.idx_m]             # (N,)
        natoms = inputs[properties.n_atoms]          # (n_mol,)
        maxm = int(idx_m[-1]) + 1

        chi = self.chi_net(l0)                       # (N,1)
        jii_raw = self.jii_net(l0)                   # (N,1)

        eta = F.softplus(jii_raw) + self.eps_hardness
        if self.jii_min > 0.0:
            eta = eta.clamp_min(self.jii_min)

        phi = self._get_phi(inputs, chi)             # (N,1)

        sigma = None
        if self.use_gaussian_self_energy or self.use_offdiag:
            sigma = self._get_sigma(l0, inputs)
            inputs[self.sigma_key] = sigma.squeeze(-1)

        # diagonal self term
        if self.use_gaussian_self_energy:
            self_term = 1.0 / (sigma * math.sqrt(math.pi))
            jii_eff = eta + self_term
        else:
            jii_eff = eta

        if properties.total_charge in inputs:
            Q = inputs[properties.total_charge].unsqueeze(-1)   # (n_mol,1)
        else:
            Q = torch.zeros((maxm, 1), device=chi.device, dtype=chi.dtype)

        # ---------------------------
        # diagonal closed-form branch
        # ---------------------------
        if not self.use_offdiag:
            invJ = 1.0 / jii_eff
            sum_invJ = snn.scatter_add(invJ, idx_m, dim_size=maxm).clamp_min(1e-12)
            sum_g_over_J = snn.scatter_add((chi + phi) * invJ, idx_m, dim_size=maxm)

            lam = -(Q + sum_g_over_J) / sum_invJ
            q = -(chi + phi + lam[idx_m]) / jii_eff

        # ---------------------------
        # off-diagonal dense solve
        # ---------------------------
        else:
            positions = inputs[properties.R]   # (N,3)
            gamma = self._get_gamma()
            inputs[self.gamma_key] = gamma.detach()

            q_list = []
            start = 0
            for m in range(maxm):
                n = int(natoms[m].item())
                sl = slice(start, start + n)

                Rm = positions[sl]                     # (n,3)
                chim = chi[sl].squeeze(-1)            # (n,)
                phim = phi[sl].squeeze(-1)            # (n,)
                gm = chim + phim                      # (n,)
                Jiim = jii_eff[sl].squeeze(-1)        # (n,)
                Qm = Q[m].squeeze(-1)                 # scalar
                sigmam = sigma[sl].squeeze(-1)        # (n,)

                # pair distances
                dR = Rm[:, None, :] - Rm[None, :, :]
                Rij = torch.linalg.norm(dR, dim=-1)   # (n,n)

                # screened Coulomb kernel
                sigma_ij = torch.sqrt(
                    sigmam[:, None] ** 2 + sigmam[None, :] ** 2
                ) + 1e-12

                eye = torch.eye(n, device=Rm.device, dtype=torch.bool)
                Rij_safe = Rij.masked_fill(eye, 1.0)

                Jij = torch.erf(Rij_safe / sigma_ij) / Rij_safe
                Jij = Jij.masked_fill(eye, 0.0)

                if self.offdiag_cutoff is not None:
                    Jij = Jij * (Rij <= self.offdiag_cutoff)

                A = torch.diag(Jiim) + gamma * Jij    # (n,n)

                # KKT system
                K = torch.zeros((n + 1, n + 1), device=Rm.device, dtype=Rm.dtype)
                K[:n, :n] = A
                K[:n, n] = 1.0
                K[n, :n] = 1.0

                b = torch.zeros((n + 1,), device=Rm.device, dtype=Rm.dtype)
                b[:n] = -gm
                b[n] = Qm

                x = torch.linalg.solve(K, b)
                qm = x[:n].unsqueeze(-1)
                q_list.append(qm)

                start += n

            q = torch.cat(q_list, dim=0)

        if self.correct_charges:
            q_sum = snn.scatter_add(q, idx_m, dim_size=maxm)
            dq = (Q - q_sum) / natoms.unsqueeze(-1)
            q = q + dq[idx_m]

        inputs[self.chi_key] = chi.squeeze(-1)
        inputs[self.hardness_key] = jii_eff.squeeze(-1)
        inputs[self.charges_key] = q.squeeze(-1)
        return inputs
class Charges(nn.Module):
    """
    Predicts Charges

    References:

    .. [#painn1] Schütt, Unke, Gastegger.
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    .. [#irspec] Gastegger, Behler, Marquetand.
       Machine learning molecular dynamics for the simulation of infrared spectra.
       Chemical science 8.10 (2017): 6924-6935.
    .. [#dipole] Veit et al.
       Predicting molecular dipole moments by combining atomic partial charges and atomic dipoles.
       The Journal of Chemical Physics 153.2 (2020): 024113.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        charges_key: str = properties.partial_charges,
        correct_charges: bool = True,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers
                resulting in a rectangular network.
                If None, the number of neurons is divided by two after each layer
                starting n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            charges_key: the key under which partial charges will be stored
            correct_charges: If true, forces the sum of partial charges to be the total
                charge, if provided, and zero otherwise.
        """
        super().__init__()
        self.charges_key = charges_key
        self.model_outputs = [charges_key]
        self.correct_charges = correct_charges
        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        natoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        charges = self.outnet(l0)

        if self.correct_charges:
            sum_charge = snn.scatter_add(charges, idx_m, dim_size=maxm)

            if properties.total_charge in inputs:
                total_charge = inputs[properties.total_charge][:, None]
            else:
                total_charge = torch.zeros_like(sum_charge)

            charge_correction = (total_charge - sum_charge) / natoms.unsqueeze(-1)
            charge_correction = charge_correction[idx_m]
            charges = charges + charge_correction

        
        inputs[self.charges_key] = charges.squeeze(-1)

        return inputs
    
class SpinCharges(nn.Module):
    """
    Predicts spin-resolved alpha and beta charges and enforces:
      - total charge constraint
      - multiplicity constraint
    """

    def __init__(
        self,
        n_in: int,
        n_hidden=None,
        n_layers: int = 2,
        activation=F.silu,
        alpha_key: str = properties.alpha_charges,
        beta_key: str = properties.beta_charges,
        correct_spincharges: bool = True,
    ):
        super().__init__()
        self.alpha_key = alpha_key
        self.beta_key = beta_key
        self.correct_spincharges = correct_spincharges
        self.model_outputs = [alpha_key, beta_key]

        # predict q_alpha and q_beta independently but simultaneously
        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=2,      
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs):

        l0 = inputs["scalar_representation"]
        idx_m = inputs[properties.idx_m]
        natoms = inputs[properties.n_atoms]
        Z_atoms = inputs[properties.Z].float()
        maxm = int(idx_m[-1]) + 1

        # Predict per-atom alpha and beta charges
        q = self.outnet(l0)         # shape: (natoms, 2)
        q_alpha_pred = q[:, 0:1]
        q_beta_pred  = q[:, 1:2]

        if self.correct_spincharges:
            # spin charge sums
            Q_alpha_pred = snn.scatter_add(q_alpha_pred, idx_m, dim_size=maxm)
            Q_beta_pred  = snn.scatter_add(q_beta_pred,  idx_m, dim_size=maxm)

            # required total charge
            if properties.total_charge in inputs:
                Q_total = inputs[properties.total_charge][:, None]
            else:
                raise ValueError("Total_charge required for spin‐resolved charge correction")

            # required multiplicity
            if properties.spin_multiplicity in inputs:
                M = inputs[properties.spin_multiplicity][:, None]
            else:
                raise ValueError("Multiplicity required for spin‐resolved charge correction")

            # set up our physical constraints:
            Z_sum = snn.scatter_add(Z_atoms, idx_m, dim_size=maxm)[:, None]
            Q_alpha_req = 0.5*(Z_sum+Q_total - (M - 1))
            Q_beta_req  = 0.5*(Z_sum+Q_total + (M - 1))

            # corrections
            corr_alpha = (Q_alpha_req - Q_alpha_pred) / natoms.unsqueeze(-1)
            corr_beta  = (Q_beta_req  - Q_beta_pred ) / natoms.unsqueeze(-1)

            # map atomwise
            corr_alpha = corr_alpha[idx_m]
            corr_beta  = corr_beta[idx_m]

            # apply corrections
            q_alpha = q_alpha_pred + corr_alpha
            q_beta  = q_beta_pred  + corr_beta

        else:
            q_alpha = q_alpha_pred
            q_beta = q_beta_pred

        inputs[self.alpha_key] = q_alpha.squeeze(-1)
        inputs[self.beta_key]  = q_beta.squeeze(-1)

        return inputs

class DipoleMoment(nn.Module):
    """
    Predicts dipole moments from latent partial charges and (optionally) local, atomic dipoles.
    The latter requires a representation supplying (equivariant) vector features.

    References:

    .. [#painn1] Schütt, Unke, Gastegger.
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    .. [#irspec] Gastegger, Behler, Marquetand.
       Machine learning molecular dynamics for the simulation of infrared spectra.
       Chemical science 8.10 (2017): 6924-6935.
    .. [#dipole] Veit et al.
       Predicting molecular dipole moments by combining atomic partial charges and atomic dipoles.
       The Journal of Chemical Physics 153.2 (2020): 024113.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        predict_magnitude: bool = False,
        return_charges: bool = False,
        dipole_key: str = properties.dipole_moment,
        charges_key: str = properties.partial_charges,
        correct_charges: bool = True,
        use_vector_representation: bool = False,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers
                resulting in a rectangular network.
                If None, the number of neurons is divided by two after each layer
                starting n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            predict_magnitude: If true, calculate magnitude of dipole
            return_charges: If true, return latent partial charges
            dipole_key: the key under which the dipoles will be stored
            charges_key: the key under which partial charges will be stored
            correct_charges: If true, forces the sum of partial charges to be the total
                charge, if provided, and zero otherwise.
            use_vector_representation: If true, use vector representation to predict
                local, atomic dipoles.
        """
        super().__init__()

        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.return_charges = return_charges
        self.model_outputs = [dipole_key]
        if self.return_charges:
            self.model_outputs.append(charges_key)

        self.predict_magnitude = predict_magnitude
        self.use_vector_representation = use_vector_representation
        self.correct_charges = correct_charges

        if use_vector_representation:
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        natoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        if self.use_vector_representation:
            l1 = inputs["vector_representation"]
            charges, atomic_dipoles = self.outnet((l0, l1))
            atomic_dipoles = torch.squeeze(atomic_dipoles, -1)
        else:
            charges = self.outnet(l0)
            atomic_dipoles = 0.0

        if self.correct_charges:
            sum_charge = snn.scatter_add(charges, idx_m, dim_size=maxm)

            if properties.total_charge in inputs:
                total_charge = inputs[properties.total_charge][:, None]
            else:
                total_charge = torch.zeros_like(sum_charge)

            charge_correction = (total_charge - sum_charge) / natoms.unsqueeze(-1)
            charge_correction = charge_correction[idx_m]
            charges = charges + charge_correction

        if self.return_charges:
            inputs[self.charges_key] = charges

        y = positions * charges
        if self.use_vector_representation:
            y = y + atomic_dipoles

        # sum over atoms
        y = snn.scatter_add(y, idx_m, dim_size=maxm)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=False)

        inputs[self.dipole_key] = y
        return inputs


class EMLEStaticHead(nn.Module):
    """
    Static EMLE-style head:
      - predicts geometry-dependent atomic charges
      - predicts MBIS valence widths
      - enforces exact molecular charge conservation

    Intended first-stage use:
      supervise against
        * DB property "charges"
        * DB property "mbis_valence_widths"

    Notes
    -----
    - This is NOT QEq.
    - It is the simplest stable first milestone for EMLE-style static training.
    - Widths are predicted directly and constrained positive with softplus.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        charges_key: str = "charges",
        widths_key: str = "mbis_valence_widths",
        correct_charges: bool = True,
        width_min: float = 0.05,
        width_max: Optional[float] = None,
    ):
        super().__init__()
        self.charges_key = charges_key
        self.widths_key = widths_key
        self.correct_charges = correct_charges
        self.width_min = float(width_min)
        self.width_max = width_max

        self.charge_net = spk.nn.build_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        self.width_net = spk.nn.build_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        self.model_outputs = [charges_key, widths_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l0 = inputs["scalar_representation"]          # (N_atoms, n_in)
        idx_m = inputs[properties.idx_m]              # (N_atoms,)
        natoms = inputs[properties.n_atoms]           # (N_mols,)
        maxm = int(idx_m[-1]) + 1

        # raw predictions
        q = self.charge_net(l0)                       # (N_atoms, 1)
        widths_raw = self.width_net(l0)               # (N_atoms, 1)

        # positive widths
        widths = F.softplus(widths_raw) + self.width_min
        if self.width_max is not None:
            widths = torch.clamp(widths, max=self.width_max)

        # enforce exact total charge per molecule
        if self.correct_charges:
            q_sum = snn.scatter_add(q, idx_m, dim_size=maxm)

            if properties.total_charge in inputs:
                total_charge = inputs[properties.total_charge][:, None]
            else:
                total_charge = torch.zeros_like(q_sum)

            dq = (total_charge - q_sum) / natoms.unsqueeze(-1)
            q = q + dq[idx_m]

        inputs[self.charges_key] = q.squeeze(-1)
        inputs[self.widths_key] = widths.squeeze(-1)
        return inputs

class EMLEStaticParamHead(nn.Module):
    """
    Neural QEq static parameter head.

    Predicts the geometry-dependent quantities needed for themodel:
      - valence widths s_i  (target: MBIS valence widths)
      - electronegativities chi_i (trained indirectly through QEq charge fitting)

    This class does NOT solve QEq itself.
    It only predicts the parameters that the neural QEq layer will consume.

    Outputs
    -------
    widths_key : per-atom valence widths s_i
    chi_key    : per-atom electronegativities chi_i
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        widths_key: str = "mbis_valence_widths",
        chi_key: str = "chi_emle",
        width_min: float = 0.05,
        width_max: Optional[float] = None,
    ):
        super().__init__()
        self.widths_key = widths_key
        self.chi_key = chi_key
        self.width_min = float(width_min)
        self.width_max = width_max

        self.width_net = spk.nn.build_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        self.chi_net = spk.nn.build_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        self.model_outputs = [widths_key, chi_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l0 = inputs["scalar_representation"]

        widths_raw = self.width_net(l0)
        widths = F.softplus(widths_raw) + self.width_min
        if self.width_max is not None:
            widths = torch.clamp(widths, max=self.width_max)

        chi = self.chi_net(l0)

        inputs[self.widths_key] = widths.squeeze(-1)
        inputs[self.chi_key] = chi.squeeze(-1)
        return inputs

class EMLEQEqStatic(nn.Module):
    """
    neural QEq layer with optional external electrostatic potential.

    Unit convention
    ---------------
    - distances: Angstrom
    - charges: e
    - chi, phi: kJ/mol/e
    - Jii, Jij: kJ/mol/e^2
    - embedding energy: kJ/mol

    The external embedding currently couples total atomic point charges to
    ``phi``. The core/valence outputs are diagnostic and do not select separate
    electrostatic kernels.

    Robustness controls
    -------------------
    The optional arguments ``a_qeq_min``, ``a_qeq_max``, ``sigma_qeq_min``,
    ``sigma_qeq_max`` and ``jii_floor`` can be used to reduce excessive
    external-field charge response during MD without requiring external-field
    reference charges.
    """

    def __init__(
        self,
        widths_key: str = "mbis_valence_widths",
        chi_key: str = "chi_emle",
        charges_key: str = "charges",
        sigma_qeq_key: str = "sigma_qeq",
        qcore_key: str = "q_core",
        qval_key: str = "q_val",
        hardness_key: str = "Jii_qeq",
        aqeq_key: str = "a_qeq",
        phi_key: Optional[str] = None,
        charges_vac_key: str = "charges_vac",
        charges_polarized_key: str = "charges",
        embedding_energy_key: str = "qeq_embedding_energy",
        qval_vac_key: str = "q_val_vac",
        polarize_key: str = "qeq_polarize",
        max_z: int = 100,
        qcore_table: Optional[Dict[int, float]] = None,
        init_a_qeq: float = 1.0,
        learn_a_qeq: bool = True,
        a_qeq_positive: bool = True,

        # Robustness controls for external-field MD.
        # These are optional and backward-compatible:
        # - If a_qeq_min/a_qeq_max are both None, the old softplus mapping is used.
        # - If both are set, a_qeq is bounded to [a_qeq_min, a_qeq_max].
        # - sigma_qeq_min/max clamp the final QEq Gaussian width sigma = a_qeq * s.
        # - jii_floor is an additive diagonal hardness offset in kJ/mol/e^2.
        a_qeq_min: Optional[float] = None,
        a_qeq_max: Optional[float] = None,
        sigma_qeq_min: Optional[float] = None,
        sigma_qeq_max: Optional[float] = None,
        jii_floor: float = 0.0,

        eps_sigma: float = 1e-8,
        correct_charges: bool = True,
        offdiag_cutoff: Optional[float] = None,
    ):
        super().__init__()
        self.widths_key = widths_key
        self.chi_key = chi_key
        self.charges_key = charges_key
        self.sigma_qeq_key = sigma_qeq_key
        self.qcore_key = qcore_key
        self.qval_key = qval_key
        self.hardness_key = hardness_key
        self.aqeq_key = aqeq_key
        self.phi_key = phi_key
        self.charges_vac_key = charges_vac_key
        self.charges_polarized_key = charges_polarized_key
        self.embedding_energy_key = embedding_energy_key
        self.qval_vac_key = qval_vac_key
        self.polarize_key = polarize_key

        self.max_z = int(max_z)
        self.eps_sigma = float(eps_sigma)
        self.correct_charges = correct_charges
        self.offdiag_cutoff = offdiag_cutoff
        self.a_qeq_positive = a_qeq_positive

        # Optional robustness controls.
        self.a_qeq_min = None if a_qeq_min is None else float(a_qeq_min)
        self.a_qeq_max = None if a_qeq_max is None else float(a_qeq_max)
        self.use_bounded_a_qeq = (self.a_qeq_min is not None) or (self.a_qeq_max is not None)

        if self.use_bounded_a_qeq:
            if self.a_qeq_min is None or self.a_qeq_max is None:
                raise ValueError("Set both a_qeq_min and a_qeq_max, or neither.")
            if not self.a_qeq_min < self.a_qeq_max:
                raise ValueError("Require a_qeq_min < a_qeq_max.")
            init_clamped = min(max(float(init_a_qeq), self.a_qeq_min), self.a_qeq_max)
            frac = (init_clamped - self.a_qeq_min) / (self.a_qeq_max - self.a_qeq_min)
            frac = min(max(frac, 1.0e-6), 1.0 - 1.0e-6)
            a_raw = torch.logit(torch.tensor(frac, dtype=torch.float32))
        else:
            # Original behavior, preserved for old configs/checkpoints.
            a_raw = torch.tensor(float(init_a_qeq), dtype=torch.float32)
            if a_qeq_positive:
                init_safe = max(float(init_a_qeq), 1e-8)
                a_raw = torch.log(torch.exp(torch.tensor(init_safe, dtype=torch.float32)) - 1.0)

        self.a_qeq_raw = nn.Parameter(a_raw)
        if not learn_a_qeq:
            self.a_qeq_raw.requires_grad_(False)

        self.sigma_qeq_min = None if sigma_qeq_min is None else float(sigma_qeq_min)
        self.sigma_qeq_max = None if sigma_qeq_max is None else float(sigma_qeq_max)
        if self.sigma_qeq_min is not None and self.sigma_qeq_min <= 0.0:
            raise ValueError("sigma_qeq_min must be positive.")
        if self.sigma_qeq_max is not None and self.sigma_qeq_max <= 0.0:
            raise ValueError("sigma_qeq_max must be positive.")
        if (
            self.sigma_qeq_min is not None
            and self.sigma_qeq_max is not None
            and self.sigma_qeq_min > self.sigma_qeq_max
        ):
            raise ValueError("Require sigma_qeq_min <= sigma_qeq_max.")

        self.jii_floor = float(jii_floor)
        if self.jii_floor < 0.0:
            raise ValueError("jii_floor must be non-negative.")

        qcore_arr = torch.zeros(self.max_z + 1, dtype=torch.float32)
        if qcore_table is not None:
            for z, val in qcore_table.items():
                z = int(z)
                if z <= self.max_z:
                    qcore_arr[z] = float(val)

        self.register_buffer("qcore_table_tensor", qcore_arr)

        self.model_outputs = [
            charges_key,
            charges_vac_key,
            embedding_energy_key,
            sigma_qeq_key,
            qcore_key,
            qval_key,
            qval_vac_key,
            hardness_key,
            aqeq_key,
        ]
        if charges_polarized_key not in self.model_outputs:
            self.model_outputs.append(charges_polarized_key)

    def _get_a_qeq(self) -> torch.Tensor:
        # Full-module checkpoints store object attributes rather than rerunning
        # __init__. Older EMLEQEqStatic checkpoints therefore do not contain
        # the optional robustness attributes added later.
        if getattr(self, "use_bounded_a_qeq", False):
            # Bounded mapping for robust external-field MD.
            # a_qeq_min <= a_qeq <= a_qeq_max
            return self.a_qeq_min + (self.a_qeq_max - self.a_qeq_min) * torch.sigmoid(self.a_qeq_raw)

        # Original behavior, preserved for old configs/checkpoints.
        if getattr(self, "a_qeq_positive", True):
            return F.softplus(self.a_qeq_raw)
        return self.a_qeq_raw

    def _get_qcore(self, Z: torch.Tensor) -> torch.Tensor:
        return self.qcore_table_tensor[Z.long()]

    def _get_phi(self, inputs: Dict[str, torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if self.phi_key is None or self.phi_key not in inputs:
            return torch.zeros_like(like)

        phi = inputs[self.phi_key]
        if phi.dim() == 1:
            phi = phi.unsqueeze(-1)
        return phi.to(like)

    def _build_qeq_matrix(
        self,
        R: torch.Tensor,
        sigma: torch.Tensor,
        Jii: torch.Tensor,
    ) -> torch.Tensor:
        """Build one molecular QEq interaction matrix in kJ/mol/e^2."""
        n = R.shape[0]
        dR = R[:, None, :] - R[None, :, :]
        Rij = torch.linalg.norm(dR, dim=-1)
        sigma_ij = torch.sqrt(
            sigma[:, None] ** 2 + sigma[None, :] ** 2
        ).clamp_min(self.eps_sigma)

        eye = torch.eye(n, device=R.device, dtype=torch.bool)
        Rij_safe = Rij.masked_fill(eye, 1.0)
        Jij = (
            K_E_KJMOL_ANG_E2
            * torch.erf(Rij_safe / (math.sqrt(2.0) * sigma_ij))
            / Rij_safe
        )
        Jij = Jij.masked_fill(eye, 0.0)
        if self.offdiag_cutoff is not None:
            Jij = Jij * (Rij <= float(self.offdiag_cutoff))
        return torch.diag(Jii) + Jij

    @staticmethod
    def _solve_qeq(
        A: torch.Tensor,
        g: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """Solve a constrained molecular QEq problem differentiably."""
        n = A.shape[0]
        K = torch.zeros((n + 1, n + 1), device=A.device, dtype=A.dtype)
        K[:n, :n] = A
        K[:n, n] = 1.0
        K[n, :n] = 1.0

        b = torch.zeros((n + 1,), device=A.device, dtype=A.dtype)
        b[:n] = -g
        b[n] = Q
        return torch.linalg.solve(K, b)[:n]

    def _correct_total_charge(
        self,
        q: torch.Tensor,
        Q: torch.Tensor,
        idx_m: torch.Tensor,
        natoms: torch.Tensor,
        maxm: int,
    ) -> torch.Tensor:
        if not self.correct_charges:
            return q
        q_sum = snn.scatter_add(q, idx_m, dim_size=maxm)
        dq = (Q - q_sum) / natoms.unsqueeze(-1)
        return q + dq[idx_m]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.widths_key not in inputs:
            raise KeyError(f"Expected '{self.widths_key}' in inputs.")
        if self.chi_key not in inputs:
            raise KeyError(f"Expected '{self.chi_key}' in inputs.")

        Z = inputs[properties.Z].long()
        R = inputs[properties.R]
        idx_m = inputs[properties.idx_m]
        natoms = inputs[properties.n_atoms]
        maxm = int(idx_m[-1]) + 1

        s = inputs[self.widths_key]
        chi = inputs[self.chi_key]

        if s.dim() == 1:
            s = s.unsqueeze(-1)
        if chi.dim() == 1:
            chi = chi.unsqueeze(-1)

        phi_available = self.phi_key is not None and self.phi_key in inputs
        polarize_key = getattr(self, "polarize_key", "qeq_polarize")
        polarize = phi_available
        if polarize_key in inputs:
            control = torch.as_tensor(inputs[polarize_key])
            polarize = phi_available and bool(control.reshape(-1)[0].item())
        phi = self._get_phi(inputs, chi)  # kJ/mol/e

        a_qeq = self._get_a_qeq()
        sigma = a_qeq * s  # Ang

        # Clamp final QEq width sigma = a_qeq * s.
        # This is the most important robustness control: overly large sigma
        # makes Jii too small and the QEq response too polarizable.
        sigma_qeq_min = getattr(self, "sigma_qeq_min", None)
        sigma_qeq_max = getattr(self, "sigma_qeq_max", None)
        sigma_min = self.eps_sigma if sigma_qeq_min is None else sigma_qeq_min
        sigma = torch.clamp(sigma, min=sigma_min)
        if sigma_qeq_max is not None:
            sigma = torch.clamp(sigma, max=sigma_qeq_max)

        # Gaussian self-energy hardness in kJ/mol/e^2.
        # jii_floor is an additive diagonal hardness offset. It damps
        # excessive external-field charge response without requiring
        # external-field reference charges.
        Jii = K_E_KJMOL_ANG_E2 / (sigma * math.sqrt(math.pi))
        jii_floor = getattr(self, "jii_floor", 0.0)
        if jii_floor > 0.0:
            Jii = Jii + jii_floor

        if properties.total_charge in inputs:
            Q = inputs[properties.total_charge].unsqueeze(-1)
        else:
            Q = torch.zeros((maxm, 1), device=R.device, dtype=R.dtype)

        q0_list = []
        qphi_list = []
        start = 0

        for m in range(maxm):
            n = int(natoms[m].item())
            sl = slice(start, start + n)

            Rm = R[sl]
            chim = chi[sl].squeeze(-1)      # kJ/mol/e
            phim = phi[sl].squeeze(-1)      # kJ/mol/e
            sigmam = sigma[sl].squeeze(-1)  # Ang
            Jiim = Jii[sl].squeeze(-1)      # kJ/mol/e^2
            Qm = Q[m].squeeze(-1)

            A = self._build_qeq_matrix(Rm, sigmam, Jiim)
            q0m = self._solve_qeq(A, chim, Qm)
            qphim = (
                self._solve_qeq(A, chim + phim, Qm)
                if polarize
                else q0m
            )
            q0_list.append(q0m.unsqueeze(-1))
            qphi_list.append(qphim.unsqueeze(-1))

            start += n

        q0 = torch.cat(q0_list, dim=0)
        qphi = torch.cat(qphi_list, dim=0)
        q0 = self._correct_total_charge(q0, Q, idx_m, natoms, maxm)
        qphi = self._correct_total_charge(qphi, Q, idx_m, natoms, maxm)

        # Diagnostic guard only: retain production continuity while making a
        # numerically significant constraint residual visible.
        with torch.no_grad():
            q0_residual = torch.max(
                torch.abs(snn.scatter_add(q0, idx_m, dim_size=maxm) - Q)
            )
            qphi_residual = torch.max(
                torch.abs(snn.scatter_add(qphi, idx_m, dim_size=maxm) - Q)
            )
            tolerance = 1.0e-5
            if q0_residual > tolerance or qphi_residual > tolerance:
                warnings.warn(
                    "QEq total-charge residual exceeds tolerance: "
                    f"vacuum={q0_residual.item():.3e}, "
                    f"polarized={qphi_residual.item():.3e}",
                    RuntimeWarning,
                )

        embedding_atom = 0.5 * (q0 + qphi) * phi
        embedding_energy = snn.scatter_add(
            embedding_atom, idx_m, dim_size=maxm
        ).squeeze(-1)

        q_core = self._get_qcore(Z).unsqueeze(-1)
        q_val = qphi - q_core
        q_val_vac = q0 - q_core

        charges_vac_key = getattr(self, "charges_vac_key", "charges_vac")
        charges_polarized_key = getattr(
            self, "charges_polarized_key", self.charges_key
        )
        embedding_energy_key = getattr(
            self, "embedding_energy_key", "qeq_embedding_energy"
        )
        qval_vac_key = getattr(self, "qval_vac_key", "q_val_vac")

        inputs[charges_vac_key] = q0.squeeze(-1)
        inputs[charges_polarized_key] = qphi.squeeze(-1)
        # Keep the historical charges_key populated if a distinct polarized key
        # was requested. Existing vacuum charge losses therefore need no changes.
        inputs[self.charges_key] = qphi.squeeze(-1)
        inputs[embedding_energy_key] = embedding_energy
        inputs[self.sigma_qeq_key] = sigma.squeeze(-1)
        inputs[self.qcore_key] = q_core.squeeze(-1)
        inputs[self.qval_key] = q_val.squeeze(-1)
        inputs[qval_vac_key] = q_val_vac.squeeze(-1)
        inputs[self.hardness_key] = Jii.squeeze(-1)
        inputs[self.aqeq_key] = a_qeq.expand_as(qphi).squeeze(-1)

        return inputs
    
class ExternalCoulombEmbedding(nn.Module):
    def __init__(
        self,
        charges_key="charges",
        output_key="external_electrostatic_energy",
        or_positions_key="or_positions",
        or_charges_key="or_charges",
        mask_key=None,
        cutoff=None,
        screening=None,
        sigma=1.0,   # Å
        coulomb_constant=14.3996454784255,  # eV*Å/e^2
    ):
        super().__init__()
        self.charges_key = charges_key
        self.output_key = output_key
        self.or_positions_key = or_positions_key
        self.or_charges_key = or_charges_key
        self.mask_key = mask_key
        self.cutoff = cutoff
        self.screening = screening
        self.sigma = sigma
        self.coulomb_constant = coulomb_constant
        self.model_outputs = [output_key]

    def forward(self, inputs):
        R = inputs[properties.R]                       # (N_qm, 3)
        q = inputs[self.charges_key].unsqueeze(-1)    # (N_qm, 1)
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        OR_R = inputs[self.or_positions_key]          # (N_qm, N_or, 3) or padded
        OR_Q = inputs[self.or_charges_key]            # (N_qm, N_or)

        # expand QM atoms molecule-wise
        E_mol = []
        for m in range(maxm):
            sel = (idx_m == m)
            Rm = R[sel]                               # (n, 3)
            qm = q[sel]                               # (n, 1)

            RJ = OR_R[m]                              # (M, 3)
            QJ = OR_Q[m].unsqueeze(0)                 # (1, M)

            dR = Rm[:, None, :] - RJ[None, :, :]      # (n, M, 3)
            rij = torch.linalg.norm(dR, dim=-1)       # (n, M)

            if self.screening == "erf":
                v = torch.erf(rij / self.sigma) / rij.clamp_min(1e-8)
            elif self.screening == "soft":
                v = 1.0 / torch.sqrt(rij * rij + self.sigma * self.sigma)
            else:
                v = 1.0 / rij.clamp_min(1e-8)

            if self.cutoff is not None:
                v = v * (rij <= self.cutoff)

            E = self.coulomb_constant * torch.sum(qm * QJ * v)
            E_mol.append(E)

        inputs[self.output_key] = torch.stack(E_mol, dim=0)
        return inputs

class Polarizability(nn.Module):
    """
    Predicts polarizability tensor using tensor rank factorization.
    This requires an equivariant representation, e.g. PaiNN, that provides both scalar and vectorial features.

    References:

    .. [#painn1a] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        polarizability_key: str = properties.polarizability,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            polarizability_key: the key under which the predicted polarizability will be stored
        """
        super(Polarizability, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.polarizability_key = polarizability_key
        self.model_outputs = [polarizability_key]

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.requires_dr = False
        self.requires_stress = False

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]
        dim = l1.shape[-2]

        l0, l1 = self.outnet((l0, l1))

        # isotropic on diagonal
        alpha = l0[..., 0:1]
        size = list(alpha.shape)
        size[-1] = dim
        alpha = alpha.expand(*size)
        alpha = torch.diag_embed(alpha)

        # add anisotropic components
        mur = l1[..., None, 0] * positions[..., None, :]
        alpha_c = mur + mur.transpose(-2, -1)
        alpha = alpha + alpha_c

        # sum over atoms
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        alpha = snn.scatter_add(alpha, idx_m, dim_size=maxm)

        inputs[self.polarizability_key] = alpha
        return inputs

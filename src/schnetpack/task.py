import warnings
from typing import Optional, Dict, List, Type, Any, Mapping

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel
from schnetpack import properties
from schnetpack.nn import scatter_add

__all__ = [
    "ModelOutput",
    "AtomisticTask",
    "FieldRegularizedAtomisticTask",
    "center_moleculewise",
    "elementwise_charge_bounds",
]


def center_moleculewise(
    x: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor
) -> torch.Tensor:
    """Remove the per-molecule mean from an atom-wise tensor."""
    if x.shape[0] != idx_m.shape[0]:
        raise ValueError("x and idx_m must have the same atom dimension")
    n_molecules = int(n_atoms.shape[0])
    sums = scatter_add(x, idx_m, dim_size=n_molecules)
    counts = n_atoms.to(device=x.device, dtype=x.dtype)
    while counts.dim() < sums.dim():
        counts = counts.unsqueeze(-1)
    means = sums / counts.clamp_min(1)
    return x - means[idx_m]


def elementwise_charge_bounds(
    Z: torch.Tensor, bounds_dict: Mapping[Any, float]
) -> torch.Tensor:
    """Construct per-atom absolute-charge limits from an element mapping."""
    fallback = float(bounds_dict.get("default", bounds_dict.get("DEFAULT", 2.0)))
    bounds = torch.full_like(Z, fallback, dtype=torch.get_default_dtype())
    for atomic_number, value in bounds_dict.items():
        if str(atomic_number).lower() == "default":
            continue
        bounds = torch.where(
            Z == int(atomic_number),
            bounds.new_tensor(float(value)),
            bounds,
        )
    return bounds


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        r"""
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])


class UnsupervisedModelOutput(ModelOutput):
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged.
    """

    def calculate_loss(self, pred, target=None):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred[self.name])
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name])


class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "test")
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate scheduling is set to occur after the epoch ends. To enable scheduling before the "
                    "epoch end, please set the `val_check_interval` parameter to a value greater than 1.0, which "
                    "indicates the number of training steps after which the model should be validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            torch.save(self.model, path)
            self.model.do_postprocessing = pp_status


class FieldRegularizedAtomisticTask(AtomisticTask):
    """Atomistic task with an auxiliary, unlabeled external-field charge pass.

    If ``field_regularization`` is missing or disabled, this class follows the
    exact :class:`AtomisticTask` training path. Field augmentation is applied
    during training only; validation and testing remain zero-field only when
    their input data are zero-field.
    """

    def __init__(self, *args, field_regularization=None, **kwargs):
        super().__init__(*args, **kwargs)
        config = dict(field_regularization or {})
        self.field_regularization = config
        self.field_regularization_enabled = bool(config.get("enabled", False))
        self.phi_key = config.get("phi_key", "phi_static")
        self.phi_aug_key = config.get("phi_aug_key", "phi_static_aug")
        self.field_mode = config.get("mode", "random")
        self.phi_std = float(config.get("phi_std", 300.0))
        self.phi_clip = config.get("phi_clip", None)
        self.center_phi = bool(config.get("center_phi", True))
        self.w_bound = float(config.get("w_bound", 0.0))
        self.w_dq = float(config.get("w_dq", 0.0))
        self.w_sensitivity = float(config.get("w_sensitivity", 0.0))
        self.charges_key = config.get("charges_key", "charges")
        self.charge_bounds = config.get("charge_bounds", {"default": 2.0})
        self.save_hyperparameters({"field_regularization": config})

        if self.field_mode not in ("random", "batch"):
            raise ValueError("field_regularization.mode must be 'random' or 'batch'")
        if self.phi_std < 0.0:
            raise ValueError("field_regularization.phi_std must be non-negative")
        if self.w_sensitivity != 0.0:
            raise ValueError(
                "w_sensitivity is reserved for a future Jacobian penalty; set it to 0"
            )

    def _zero_field_batch(self, batch):
        zero_batch = dict(batch)
        zero_batch[self.phi_key] = batch[properties.Z].new_zeros(
            batch[properties.Z].shape, dtype=batch[properties.R].dtype
        )
        return zero_batch

    def _augmented_field(self, batch):
        source = None
        if self.field_mode == "batch":
            source = batch.get(self.phi_aug_key, batch.get(self.phi_key))
        if source is None:
            source = torch.randn_like(batch[properties.R][:, 0]) * self.phi_std
        else:
            source = source.to(device=batch[properties.R].device,
                               dtype=batch[properties.R].dtype).clone()
            if source.dim() > 1 and source.shape[-1] == 1:
                source = source.squeeze(-1)
        if self.center_phi:
            source = center_moleculewise(
                source, batch[properties.idx_m], batch[properties.n_atoms]
            )
        if self.phi_clip is not None:
            source = source.clamp(-float(self.phi_clip), float(self.phi_clip))
        return source

    def training_step(self, batch, batch_idx):
        if not self.field_regularization_enabled:
            return super().training_step(batch, batch_idx)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        if "considered_atoms" in batch:
            targets["considered_atoms"] = batch["considered_atoms"]

        zero_batch = self._zero_field_batch(batch)
        pred_zero = self.predict_without_postprocessing(zero_batch)
        # Save the unconstrained charge tensor; output constraints may replace
        # entries in the prediction dictionary (typically for force masking).
        q_zero = pred_zero[self.charges_key]
        pred_zero, targets = self.apply_constraints(pred_zero, targets)
        supervised_loss = self.loss_fn(pred_zero, targets)

        phi_aug = self._augmented_field(batch)
        aug_batch = dict(batch)
        # Keep force derivatives in the auxiliary forward independent from the
        # supervised force graph. No auxiliary energy/force loss is evaluated.
        aug_batch[properties.R] = batch[properties.R].detach().clone()
        aug_batch[self.phi_key] = phi_aug
        pred_aug = self.predict_without_postprocessing(aug_batch)

        q_phi = pred_aug[self.charges_key]
        bounds = elementwise_charge_bounds(
            batch[properties.Z], self.charge_bounds
        ).to(device=q_phi.device, dtype=q_phi.dtype)
        while bounds.dim() < q_phi.dim():
            bounds = bounds.unsqueeze(-1)
        bound_loss = torch.relu(q_phi.abs() - bounds).square().mean()
        dq_loss = (q_phi - q_zero.detach()).square().mean()
        loss = supervised_loss + self.w_bound * bound_loss + self.w_dq * dq_loss

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred_zero, targets, "train")
        for name, value in (
            ("train_q_phi_abs_max", q_phi.detach().abs().max()),
            ("train_q_phi_bound_loss", bound_loss.detach()),
            ("train_q_phi_dq_loss", dq_loss.detach()),
            ("train_phi_aug_min", phi_aug.detach().min()),
            ("train_phi_aug_max", phi_aug.detach().max()),
            ("train_phi_aug_std", phi_aug.detach().std(unbiased=False)),
        ):
            self.log(name, value, on_step=True, on_epoch=False, prog_bar=False)
        return loss

class ConsiderOnlySelectedAtoms(nn.Module):
    """
    Constraint that allows to neglect some atomic targets (e.g. forces of some specified atoms) for model optimization,
    while not affecting the actual model output. The indices of the atoms, which targets to consider in the loss
    function, must be provided in the dataset for each sample in form of a torch tensor of type boolean
    (True: considered, False: neglected).
    """

    def __init__(self, selection_name):
        """
        Args:
            selection_name: string associated with the list of considered atoms in the dataset
        """
        super().__init__()
        self.selection_name = selection_name

    def forward(self, pred, targets, output_module):
        """
        A torch tensor is loaded from the dataset, which specifies the considered atoms. Only the
        predictions of those atoms are considered for training, validation, and testing.

        :param pred: python dictionary containing model outputs
        :param targets: python dictionary containing targets
        :param output_module: torch.nn.Module class of a particular property (e.g. forces)
        :return: model outputs and targets of considered atoms only
        """

        considered_atoms = targets[self.selection_name].nonzero()[:, 0]

        # drop neglected atoms
        pred[output_module.name] = pred[output_module.name][considered_atoms]
        targets[output_module.target_property] = targets[output_module.target_property][
            considered_atoms
        ]

        return pred, targets

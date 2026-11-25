"""Gen 2 trainer with the emulator's rollout-length curriculum.

``TrainerERA5Gen2`` already implements what the legacy pushforward loop did:
one backward pass per rollout step, predictions detached before they are fed
back, postblocks inside the per-step gradient.  This subclass adds only the
curriculum, which CREDIT fixes at ``data.forecast_len`` for a whole run.

Config (``trainer`` block)::

    type: pop-gen2
    batches_per_epoch: 100            # sequences per epoch; one checkpoint per epoch
    rollout_curriculum:               # [first sequence, rollout length]
      - [0, 32]
      - [4000, 64]

Thresholds count training sequences (``epoch * batches_per_epoch``), the same
unit as the optimisation step of the legacy loop at batch size 1.  The length
is re-evaluated at the start of each epoch.
"""
from __future__ import annotations

import copy
import logging

from credit.trainers.trainer_gen2 import TrainerERA5Gen2
from credit.trainers.utils import load_dataloader, load_dataset

from ..utils import rollout_len_for_step

logger = logging.getLogger(__name__)

TRAINER_TYPE = "pop-gen2"
DAILY_TRAINER_TYPE = "pop-gen2-daily"


def curriculum_len(conf: dict, epoch: int) -> int:
    """Rollout length for ``epoch``; ``data.forecast_len`` without a curriculum."""
    trainer_conf = conf["trainer"]
    curriculum = trainer_conf.get("rollout_curriculum")
    if not curriculum:
        return int(conf["data"]["forecast_len"])
    return rollout_len_for_step(curriculum, epoch * int(trainer_conf["batches_per_epoch"]))


class PopTrainer(TrainerERA5Gen2):
    def __init__(self, model, rank: int, conf: dict):
        super().__init__(model, rank, conf)
        if conf["trainer"].get("rollout_curriculum") and self.batches_per_epoch <= 0:
            raise ValueError("trainer.rollout_curriculum requires trainer.batches_per_epoch > 0")
        self._loader = None                 # loader owned by the curriculum
        self._valid_loader = None           # validation loader with the validation length
        self._loader_len = int(conf["data"]["forecast_len"])

    def _curriculum_loader(self, forecast_len: int):
        """Dataset and loader for a new rollout length (valid start times shrink
        as the rollout grows, so the dataset is rebuilt, not just the sampler)."""
        from credit.parallel.mesh import data_parallel_coords

        conf = copy.deepcopy(self.conf)
        conf["data"]["forecast_len"] = forecast_len
        data_rank, data_world_size = data_parallel_coords(conf)
        dataset = load_dataset(conf, is_train=True)
        return load_dataloader(conf, dataset, rank=data_rank, world_size=data_world_size, is_train=True)

    def _begin_epoch(self, epoch, trainloader, criterion):
        """Apply the curriculum for ``epoch``; returns the loader to iterate."""
        forecast_len = curriculum_len(self.conf, epoch)
        if forecast_len != self._loader_len:
            if self._loader is not None:
                self._loader._iterator = None       # shut down the old workers
            self._loader = self._curriculum_loader(forecast_len)
            self._loader_len = forecast_len
            if self.rank == 0:
                logger.info("epoch %d: rollout length -> %d (%d start times)",
                            epoch, forecast_len, len(self._loader.dataset))
        self.forecast_len = forecast_len
        self.backprop_on_timestep = list(range(1, forecast_len + 1))
        if hasattr(criterion, "rollout_len"):
            criterion.rollout_len = forecast_len
        return self._loader or trainloader

    def validate(self, epoch, valid_loader, criterion, metrics):
        """CREDIT builds the validation sampler with ``data.forecast_len`` steps
        per sequence but consumes ``validation_data.forecast_len``; when the two
        differ (curriculum, or a resumed run) every sequence after the first
        would start mid-rollout.  Use a sampler of the validation length."""
        if self.valid_forecast_len != int(self.conf["data"]["forecast_len"]):
            if self._valid_loader is None:
                from credit.parallel.mesh import data_parallel_coords

                conf = copy.deepcopy(self.conf)
                conf["data"]["forecast_len"] = self.valid_forecast_len   # read by load_dataloader
                data_rank, data_world_size = data_parallel_coords(conf)
                dataset = load_dataset(conf, is_train=False)
                self._valid_loader = load_dataloader(conf, dataset, rank=data_rank,
                                                     world_size=data_world_size, is_train=False)
            valid_loader = self._valid_loader
        return super().validate(epoch, valid_loader, criterion, metrics)

    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        loader = self._begin_epoch(epoch, trainloader, criterion)
        return super().train_one_epoch(epoch, loader, optimizer, criterion, scaler, scheduler, metrics)


def register() -> None:
    """Add :class:`PopTrainer` to CREDIT's trainer table (``trainer.type: pop-gen2``)."""
    from credit.trainers import _TRAINER_REGISTRY

    _TRAINER_REGISTRY[TRAINER_TYPE] = (
        __name__, "PopTrainer",
        "Gen2 trainer with the POP emulator rollout curriculum.",
    )
    _TRAINER_REGISTRY[DAILY_TRAINER_TYPE] = (
        __name__.rsplit(".", 1)[0] + ".trainer_daily", "PopDailyTrainer",
        "Daily-step, surface-supervised POP trainer (monthly-mean constraint).",
    )

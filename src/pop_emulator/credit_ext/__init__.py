"""CREDIT Gen 2 (miles-credit >= 2026.2.0) integration for the POP emulator.

Each module wraps existing ``pop_emulator`` code behind a CREDIT Gen 2
interface so the emulator trains under the Gen 2 trainer (DDP/FSDP2, EMA,
per-epoch checkpointing) while keeping its weights, normalisation and physics:

==================  ==========================================================
``dataset``         ``PopLENSGen2Dataset`` — CESM-LE members on a record clock
``preblocks``       ``PopNormalize`` — per-level standardisation (stats.nc)
``model``           ``PopOceanUNet`` — ``OceanEmulator`` as a CREDIT model
``postblocks``      ``PopDenormalize``, ``OceanConservationProjection``
``loss``            ``PopCompositeLoss``, ``PopMaskedRMSE``
``trainer``         ``PopTrainer`` — Gen 2 trainer + rollout curriculum
==================  ==========================================================

All components are declared under ``custom_objects:`` in the CREDIT YAML
(see ``configs/credit/run6_monthly.yml``) except the trainer, which CREDIT
resolves from a fixed table and is therefore registered by
``scripts/train_credit.py``.
"""

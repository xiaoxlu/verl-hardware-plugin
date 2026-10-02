# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU save/restore handlers for TorchTitan in verl's ``separate_async`` actor worker.

verl-core (``verl/experimental/separation/engine_workers.py``,
``DetachActorWorker._get_strategy_handlers``) only knows the fsdp / fsdp2 / veomni / megatron
strategies and raises ``NotImplementedError: Unsupported strategy: torchtitan`` otherwise.
``trainer_separate_async._compute_old_log_prob`` calls ``save_model_to_cpu(0)`` on every step
(unless ``algorithm.rollout_correction.bypass_mode``) so that, when ``parameter_sync_step > 1``,
all mini-steps of a sync cycle compute ``old_log_probs`` with the same pi_old. So TorchTitan
cannot run ``trainer.v1.trainer_mode=separate_async`` -- the mode a TPU run needs, because trainer
and rollout sit on different slices (``hybrid_engine=False``).

TorchTitan's ``engine.module`` is the list of model parts, whose parameters are FSDP2 DTensors
(plain tensors when unsharded). The handlers below copy each rank's local shards to CPU and back,
which is what verl's ``fsdp2_sharded_save_to_cpu`` / ``fsdp2_sharded_load_from_cpu`` do for one
module, without the trailing ``dist.barrier()`` (each rank only touches its own shards).

Delete once verl-core registers TorchTitan handlers in ``_get_strategy_handlers``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False
_original_get_strategy_handlers = None


def _model_parts(module: Any) -> list[torch.nn.Module]:
    return list(module) if isinstance(module, list | tuple | torch.nn.ModuleList) else [module]


def _local_tensor(param: torch.Tensor) -> torch.Tensor:
    """The rank-local storage of ``param``: the DTensor shard, or the tensor itself."""
    local = getattr(param, "_local_tensor", None)
    return local if local is not None else param.data


def save_torchtitan_model_to_cpu(module: Any) -> list[dict[str, torch.Tensor]]:
    """Copy the local parameter shards of every TorchTitan model part to CPU."""
    return [
        {name: _local_tensor(param).detach().to("cpu", copy=True) for name, param in part.named_parameters()}
        for part in _model_parts(module)
    ]


def restore_torchtitan_model_from_cpu(module: Any, saved: list[dict[str, torch.Tensor]]) -> None:
    """Copy the shards saved by ``save_torchtitan_model_to_cpu`` back into the model parts."""
    parts = _model_parts(module)
    if len(parts) != len(saved):
        raise ValueError(f"saved state has {len(saved)} model parts, the model has {len(parts)}")
    with torch.no_grad():
        for part, state in zip(parts, saved, strict=True):
            for name, param in part.named_parameters():
                if name in state:
                    local = _local_tensor(param)
                    local.copy_(state[name].to(local.device))


def _get_strategy_handlers(self):
    """``DetachActorWorker._get_strategy_handlers`` that also answers for ``strategy=torchtitan``.

    Module-level so that Ray pickles it by reference (see ``worker_local_rank_patch``).
    """
    if self._strategy_handlers is None and self.config.actor.strategy == "torchtitan":
        self._strategy_handlers = (save_torchtitan_model_to_cpu, restore_torchtitan_model_from_cpu)
    return _original_get_strategy_handlers(self)


def apply(platform) -> bool:
    """Install the patch once. Returns True once installed."""
    del platform
    global _applied, _original_get_strategy_handlers
    if _applied:
        return True

    from verl.experimental.separation import engine_workers as separation_workers

    _original_get_strategy_handlers = separation_workers.DetachActorWorker._get_strategy_handlers
    separation_workers.DetachActorWorker._get_strategy_handlers = _get_strategy_handlers
    _applied = True
    logger.info("Applied TPU separate_async TorchTitan CPU save/restore patch")
    return True

# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Route verl's ``ppo_loss`` to a TPU version that consumes the engine's ``_tpu_padded_values``.

``TorchTitanTPUEngineWithLMHead.prepare_model_outputs`` pads each packed micro-batch to a bucketed
length, so XLA compiles a bounded number of programs. It returns ``log_probs`` / ``entropy`` as
*detached CPU* nested tensors and attaches the differentiable, bucket-padded device tensor as
``_tpu_padded_values``.

verl-core's ``verl.workers.utils.losses.ppo_loss`` does not know that attribute: its
``no_padding_2_padding`` slices ``log_probs.values()`` -- the detached CPU copy -- so the policy loss
has no autograd path to the model, and the engine's ``loss.backward()`` cannot train it.

``tpu_ppo_loss`` mirrors verl's ``ppo_loss`` and differs only in how the dense tensors are built:

1. ``log_probs`` / ``entropy``: each response is gathered from ``_tpu_padded_values`` on device with
   one ``index_select`` over a host-built index (no per-sample dynamic slicing), padded to the
   bucketed max response length;
2. ``response_mask`` / ``old_log_probs`` / ``advantages`` / ``rollout_is_weights`` /
   ``ref_log_prob``: padded on CPU to the same bucketed length and moved to the device, instead of
   ``data.select(...).to_padded_tensor()`` (exact max length, CPU).

``apply`` rebinds ``ppo_loss`` in ``verl.workers.utils.losses`` and in ``verl.workers.engine_workers``,
which imports it by value and binds it into the actor's loss function in ``init_model`` (after this
patch is installed from ``Worker.__init__``). Model outputs without ``_tpu_padded_values`` fall
through to verl's function.

Delete once verl-core's loss path consumes ``_tpu_padded_values`` (as in verl-project/verl#7231).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl_hardware_plugin.engines.tpu_utils import TPU_PADDED_VALUES_ATTR, bucket_length, unwrap_metadata

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False
_original_ppo_loss: Optional[Callable[..., Any]] = None


def _has_tpu_padded_values(model_output: dict[str, Any]) -> bool:
    return any(hasattr(value, TPU_PADDED_VALUES_ATTR) for value in model_output.values())


def _response_layout(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-sample start of the response log-probs in the packed sequence, response lengths, padded length."""
    from verl.utils import tensordict_utils as tu

    prompt_lens = data["prompts"].offsets().diff().cpu()
    response_lens = data["responses"].offsets().diff().cpu()
    sequence_ends = (prompt_lens + response_lens).cumsum(dim=0)
    # Same slice as verl's no_padding_2_padding: values[seq_end - resp_len - 1 : seq_end - 1]
    # (log-probs are shifted left by one token).
    starts = sequence_ends - response_lens - 1
    max_len = int(response_lens.max().item()) if response_lens.numel() else 1
    # The engine's pad_packed_inputs_for_tpu sets max_response_len to the bucketed length.
    padded_len = unwrap_metadata(tu.get_non_tensor_data(data=data, key="max_response_len", default=-1))
    if padded_len is None or int(padded_len) < max_len:
        padded_len = bucket_length(max_len)
    return starts, response_lens, int(padded_len)


def tpu_no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """``no_padding_2_padding`` over ``_tpu_padded_values``: (bsz, bucketed max response len) on device."""
    padded_values = getattr(tensor, TPU_PADDED_VALUES_ATTR)
    starts, response_lens, padded_len = _response_layout(data)
    cols = torch.arange(padded_len, dtype=torch.long)
    index = (starts.unsqueeze(1) + cols.unsqueeze(0)).clamp(0, padded_values.shape[0] - 1)
    valid = cols.unsqueeze(0) < response_lens.unsqueeze(1)
    gathered = torch.index_select(padded_values, 0, index.reshape(-1).to(padded_values.device))
    gathered = gathered.reshape(len(response_lens), padded_len, *padded_values.shape[1:])
    valid = valid.reshape(*valid.shape, *([1] * (padded_values.dim() - 1)))
    return gathered * valid.to(device=padded_values.device, dtype=padded_values.dtype)


def _dense_field(value: torch.Tensor, padded_len: int, device: torch.device) -> torch.Tensor:
    """Pad a (nested) per-response field to ``padded_len`` on CPU, then move it to ``device``."""
    if value.is_nested:
        bsz = value.offsets().numel() - 1
        value = torch.nested.to_padded_tensor(value.cpu(), padding=0.0, output_size=(bsz, padded_len))
    elif value.dim() >= 2 and value.shape[1] < padded_len:
        value = F.pad(value, (0, padded_len - value.shape[1]))
    elif value.dim() >= 2 and value.shape[1] > padded_len:
        # Columns past the longest response are padding (response_mask == 0).
        value = value[:, :padded_len]
    return value.to(device)


def tpu_ppo_loss(config, model_output: dict[str, Any], data: TensorDict, dp_group=None):
    """verl's ``ppo_loss`` with device-side, bucket-padded tensors (see the module docstring)."""
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric

    log_prob = tpu_no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = tpu_no_padding_2_padding(entropy, data)

    def _meta(key, default):
        return unwrap_metadata(tu.get_non_tensor_data(data=data, key=key, default=default))

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = _meta("dp_size", 1)
    config.global_batch_info["batch_num_tokens"] = _meta("batch_num_tokens", None)
    config.global_batch_info["global_batch_size"] = _meta("global_batch_size", None)
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        config.global_batch_info["dp_size"] > 1
        or config.global_batch_info["batch_num_tokens"] is not None
        or config.global_batch_info["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    padded_len = log_prob.shape[1]
    dense = {key: _dense_field(data[key], padded_len, log_prob.device) for key in fields}

    response_mask = dense["response_mask"].to(bool)
    old_log_prob = dense["old_log_probs"]
    advantages = dense["advantages"]
    rollout_is_weights = dense.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if config.use_kl_loss:
        ref_log_prob = dense["ref_log_prob"]
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )
        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def ppo_loss(config, model_output, data: TensorDict, dp_group=None):
    """Dispatch to ``tpu_ppo_loss`` for TPU engine outputs, else to verl's ``ppo_loss``.

    Module-level so that Ray pickles it by reference (see ``worker_local_rank_patch``).
    """
    if _has_tpu_padded_values(model_output):
        return tpu_ppo_loss(config, model_output, data, dp_group=dp_group)
    assert _original_ppo_loss is not None, "ppo_loss_patch.apply() has not run"
    return _original_ppo_loss(config, model_output, data, dp_group=dp_group)


def apply(platform) -> bool:
    """Install the patch once. Returns True once installed."""
    del platform
    global _applied, _original_ppo_loss
    if _applied:
        return True

    from verl.workers import engine_workers
    from verl.workers.utils import losses

    _original_ppo_loss = losses.ppo_loss
    losses.ppo_loss = ppo_loss
    engine_workers.ppo_loss = ppo_loss
    _applied = True
    logger.info("Applied TPU ppo_loss patch (consumes _tpu_padded_values)")
    return True

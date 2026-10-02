# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for the trainer-side TPU patches (``ppo_loss_patch``, ``separation_cpu_copy_patch``).

The TPU-specific code paths are exercised on CPU: ``_tpu_padded_values`` is just a tensor attribute,
so a CPU tensor stands in for the TPU buffer the engine attaches.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from verl_hardware_plugin.engines.tpu_utils import TPU_PADDED_VALUES_ATTR, bucket_length
from verl_hardware_plugin.patches.tpu import ppo_loss_patch, separation_cpu_copy_patch

PROMPT_LENS = [5, 3, 7]
RESPONSE_LENS = [4, 6, 1]
BUCKET = 8


@pytest.fixture(autouse=True)
def _small_bucket(monkeypatch):
    monkeypatch.setenv("VERL_TPU_SEQ_BUCKET_SIZE", str(BUCKET))


def _nested(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(values, layout=torch.jagged)


def _micro_batch(*, max_response_len: int | None, with_ref: bool = True):
    """A no-padding micro-batch as the v1 trainer hands it to the engine (nested per-sample fields)."""
    from verl.utils import tensordict_utils as tu

    gen = torch.Generator().manual_seed(0)
    tensors = {
        "prompts": _nested([torch.randint(0, 100, (n,), generator=gen) for n in PROMPT_LENS]),
        "responses": _nested([torch.randint(0, 100, (n,), generator=gen) for n in RESPONSE_LENS]),
        "response_mask": _nested([torch.ones(n, dtype=torch.int64) for n in RESPONSE_LENS]),
        "old_log_probs": _nested([-torch.rand(n, generator=gen) for n in RESPONSE_LENS]),
        "advantages": _nested([torch.randn(n, generator=gen) for n in RESPONSE_LENS]),
    }
    if with_ref:
        tensors["ref_log_prob"] = _nested([-torch.rand(n, generator=gen) for n in RESPONSE_LENS])
    non_tensors = {"dp_size": 1, "batch_num_tokens": sum(RESPONSE_LENS), "global_batch_size": len(RESPONSE_LENS)}
    if max_response_len is not None:
        non_tensors["max_response_len"] = max_response_len
    return tu.get_tensordict(tensor_dict=tensors, non_tensor_dict=non_tensors)


def _actor_config(use_kl_loss: bool = True):
    from verl.workers.config.actor import ActorConfig

    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=2,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        use_kl_loss=use_kl_loss,
        kl_loss_coef=0.001,
        kl_loss_type="low_var_kl",
    )


def _offsets() -> torch.Tensor:
    seq_lens = torch.tensor(PROMPT_LENS) + torch.tensor(RESPONSE_LENS)
    return F.pad(seq_lens.cumsum(0), (1, 0))


def _tpu_engine_output(values: torch.Tensor) -> torch.Tensor:
    """What TorchTitanTPUEngineWithLMHead returns: detached nested copy + bucket-padded live buffer."""
    total = values.shape[0]
    padded = F.pad(values, (0, 0) * (values.dim() - 1) + (0, bucket_length(total) - total))
    nested = torch.nested.nested_tensor_from_jagged(values.detach().clone(), _offsets())
    setattr(nested, TPU_PADDED_VALUES_ATTR, padded)
    return nested


def _metric_values(metrics: dict) -> dict:
    return {k: (v.aggregate() if hasattr(v, "aggregate") else v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# ppo_loss_patch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_kl_loss", [True, False])
def test_tpu_ppo_loss_matches_verl_ppo_loss(use_kl_loss):
    """Same loss, metrics and gradients as verl's ppo_loss on the unpadded, differentiable outputs."""
    from verl.workers.utils import losses

    total = sum(PROMPT_LENS) + sum(RESPONSE_LENS)
    torch.manual_seed(0)
    logits = torch.randn(total, requires_grad=True)
    entropy_src = torch.rand(total, requires_grad=True)

    # Reference: verl's ppo_loss on nested outputs that carry the autograd graph.
    ref_output = {
        "log_probs": torch.nested.nested_tensor_from_jagged(-F.softplus(logits), _offsets()),
        "entropy": torch.nested.nested_tensor_from_jagged(entropy_src * 2.0, _offsets()),
    }
    ref_loss, ref_metrics = losses.ppo_loss(
        _actor_config(use_kl_loss), ref_output, _micro_batch(max_response_len=None, with_ref=use_kl_loss)
    )
    ref_loss.backward()
    ref_grads = (logits.grad.clone(), entropy_src.grad.clone())
    logits.grad = None
    entropy_src.grad = None

    # TPU path: detached nested outputs + bucket-padded differentiable buffers; the engine set
    # max_response_len to the bucketed length.
    tpu_output = {
        "log_probs": _tpu_engine_output(-F.softplus(logits)),
        "entropy": _tpu_engine_output(entropy_src * 2.0),
    }
    data = _micro_batch(max_response_len=bucket_length(max(RESPONSE_LENS)), with_ref=use_kl_loss)
    tpu_loss, tpu_metrics = ppo_loss_patch.tpu_ppo_loss(_actor_config(use_kl_loss), tpu_output, data)
    tpu_loss.backward()

    torch.testing.assert_close(tpu_loss, ref_loss)
    torch.testing.assert_close(logits.grad, ref_grads[0])
    torch.testing.assert_close(entropy_src.grad, ref_grads[1])
    assert _metric_values(tpu_metrics) == pytest.approx(_metric_values(ref_metrics))


def test_tpu_ppo_loss_requires_grad_through_padded_values():
    """The loss depends on the live buffer, not on the detached nested copy verl would read."""
    total = sum(PROMPT_LENS) + sum(RESPONSE_LENS)
    logits = torch.randn(total, requires_grad=True)
    output = {"log_probs": _tpu_engine_output(-F.softplus(logits))}
    assert not output["log_probs"].requires_grad

    loss, _ = ppo_loss_patch.tpu_ppo_loss(_actor_config(), output, _micro_batch(max_response_len=None))
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("trailing_shape", [(), (3,)])
def test_tpu_no_padding_2_padding_matches_reference(trailing_shape):
    from verl.workers.utils.padding import no_padding_2_padding

    total = sum(PROMPT_LENS) + sum(RESPONSE_LENS)
    values = torch.randn(total, *trailing_shape)
    padded_len = bucket_length(max(RESPONSE_LENS))
    data = _micro_batch(max_response_len=padded_len)

    expected = no_padding_2_padding(torch.nested.nested_tensor_from_jagged(values, _offsets()), data)
    actual = ppo_loss_patch.tpu_no_padding_2_padding(_tpu_engine_output(values), data)

    assert actual.shape == (len(RESPONSE_LENS), padded_len, *trailing_shape)
    torch.testing.assert_close(actual, expected)


def test_response_layout_without_engine_max_response_len_uses_bucket():
    _, response_lens, padded_len = ppo_loss_patch._response_layout(_micro_batch(max_response_len=None))
    assert response_lens.tolist() == RESPONSE_LENS
    assert padded_len == bucket_length(max(RESPONSE_LENS)) == BUCKET


def test_dense_field_pads_and_trims_to_padded_len():
    nested = _nested([torch.ones(n) for n in RESPONSE_LENS])
    assert ppo_loss_patch._dense_field(nested, BUCKET, torch.device("cpu")).shape == (3, BUCKET)
    assert ppo_loss_patch._dense_field(torch.ones(3, 5), BUCKET, torch.device("cpu")).shape == (3, BUCKET)
    trimmed = ppo_loss_patch._dense_field(torch.arange(30.0).reshape(3, 10), BUCKET, torch.device("cpu"))
    torch.testing.assert_close(trimmed, torch.arange(30.0).reshape(3, 10)[:, :BUCKET])


def test_ppo_loss_dispatch(monkeypatch):
    original = mock.Mock(return_value="verl")
    tpu = mock.Mock(return_value="tpu")
    monkeypatch.setattr(ppo_loss_patch, "_original_ppo_loss", original)
    monkeypatch.setattr(ppo_loss_patch, "tpu_ppo_loss", tpu)
    config, data = object(), object()

    plain = {"log_probs": torch.zeros(2)}
    assert ppo_loss_patch.ppo_loss(config, plain, data, dp_group="g") == "verl"
    original.assert_called_once_with(config, plain, data, dp_group="g")

    padded = {"log_probs": _tpu_engine_output(torch.zeros(sum(PROMPT_LENS) + sum(RESPONSE_LENS)))}
    assert ppo_loss_patch.ppo_loss(config, padded, data, dp_group="g") == "tpu"
    tpu.assert_called_once_with(config, padded, data, dp_group="g")


def test_ppo_loss_patch_apply_rebinds_losses_and_engine_workers(monkeypatch):
    """engine_workers imports ppo_loss by value, so both bindings must point at the dispatcher."""
    from verl.workers.utils import losses

    original = losses.ppo_loss
    fake_engine_workers = types.ModuleType("verl.workers.engine_workers")
    fake_engine_workers.ppo_loss = original
    monkeypatch.setattr(losses, "ppo_loss", original)
    monkeypatch.setattr(ppo_loss_patch, "_applied", False)
    monkeypatch.setattr(ppo_loss_patch, "_original_ppo_loss", None)

    with mock.patch.dict(sys.modules, {"verl.workers.engine_workers": fake_engine_workers}):
        assert ppo_loss_patch.apply(platform=None)
        assert ppo_loss_patch.apply(platform=None)  # idempotent

    assert losses.ppo_loss is ppo_loss_patch.ppo_loss
    assert fake_engine_workers.ppo_loss is ppo_loss_patch.ppo_loss
    assert ppo_loss_patch._original_ppo_loss is original


# ---------------------------------------------------------------------------
# separation_cpu_copy_patch
# ---------------------------------------------------------------------------


def _model_parts() -> list[torch.nn.Module]:
    torch.manual_seed(0)
    return [torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)]


def test_torchtitan_cpu_round_trip_restores_parameters():
    parts = _model_parts()
    before = [{k: v.clone() for k, v in p.state_dict().items()} for p in parts]

    saved = separation_cpu_copy_patch.save_torchtitan_model_to_cpu(parts)
    with torch.no_grad():
        for part in parts:
            for param in part.parameters():
                param.add_(1.0)
    separation_cpu_copy_patch.restore_torchtitan_model_from_cpu(parts, saved)

    for part, expected in zip(parts, before, strict=True):
        for name, value in part.state_dict().items():
            torch.testing.assert_close(value, expected[name])


def test_torchtitan_cpu_save_is_a_copy():
    parts = _model_parts()
    saved = separation_cpu_copy_patch.save_torchtitan_model_to_cpu(parts)
    with torch.no_grad():
        parts[0].weight.add_(1.0)
    assert not torch.equal(saved[0]["weight"], parts[0].weight)


def test_torchtitan_cpu_restore_rejects_part_count_mismatch():
    parts = _model_parts()
    saved = separation_cpu_copy_patch.save_torchtitan_model_to_cpu(parts[:1])
    with pytest.raises(ValueError, match="model parts"):
        separation_cpu_copy_patch.restore_torchtitan_model_from_cpu(parts, saved)


def _detach_worker(strategy: str):
    return types.SimpleNamespace(
        _strategy_handlers=None, config=types.SimpleNamespace(actor=types.SimpleNamespace(strategy=strategy))
    )


def test_get_strategy_handlers_answers_torchtitan_and_delegates_others(monkeypatch):
    original = mock.Mock(side_effect=lambda self: self._strategy_handlers or ("fsdp2_save", "fsdp2_load"))
    monkeypatch.setattr(separation_cpu_copy_patch, "_original_get_strategy_handlers", original)

    torchtitan = _detach_worker("torchtitan")
    assert separation_cpu_copy_patch._get_strategy_handlers(torchtitan) == (
        separation_cpu_copy_patch.save_torchtitan_model_to_cpu,
        separation_cpu_copy_patch.restore_torchtitan_model_from_cpu,
    )
    assert separation_cpu_copy_patch._get_strategy_handlers(_detach_worker("fsdp2")) == ("fsdp2_save", "fsdp2_load")


def test_separation_patch_apply_patches_detach_actor_worker(monkeypatch):
    def original(self):
        raise NotImplementedError

    fake_module = types.ModuleType("verl.experimental.separation.engine_workers")
    fake_module.DetachActorWorker = type("DetachActorWorker", (), {"_get_strategy_handlers": original})
    monkeypatch.setattr(separation_cpu_copy_patch, "_applied", False)
    monkeypatch.setattr(separation_cpu_copy_patch, "_original_get_strategy_handlers", None)

    with mock.patch.dict(sys.modules, {"verl.experimental.separation.engine_workers": fake_module}):
        assert separation_cpu_copy_patch.apply(platform=None)
        assert separation_cpu_copy_patch.apply(platform=None)  # idempotent

    assert fake_module.DetachActorWorker._get_strategy_handlers is separation_cpu_copy_patch._get_strategy_handlers
    assert separation_cpu_copy_patch._original_get_strategy_handlers is original

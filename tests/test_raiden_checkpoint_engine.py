# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for Raiden P2P checkpoint engine and RayWeightRegistry extensions."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

for _mod_name in ("uvicorn", "fastapi"):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _stub: Any = ModuleType(_mod_name)
            _stub.FastAPI = object
            _stub.Server = object
            _stub.Config = object
            sys.modules[_mod_name] = _stub

from typing import Any  # noqa: E402

import verl_hardware_plugin.engines.raiden_checkpoint_engine as raiden_mod  # noqa: E402
from verl.checkpoint_engine.base import CheckpointEngineRegistry  # noqa: E402
from verl_hardware_plugin.engines.raiden_checkpoint_engine import (  # noqa: E402
    RAIDEN_SAMPLER_LISTENER_BASE_PORT,
    RaidenCheckpointEngine,
    _dim0_shard_info,
    apply_raiden_checkpoint_engine_hooks,
    compute_tensor_stats,
    export_local_shards,
    merge_rank_stats,
    update_raiden_weights,
    vLLMRaidenWorkerExtension,
)
from verl_hardware_plugin.engines.ray_weight_registry import RayWeightRegistryState  # noqa: E402


def test_raiden_checkpoint_engine_registered() -> None:
    apply_raiden_checkpoint_engine_hooks()
    assert CheckpointEngineRegistry.get("raiden") is RaidenCheckpointEngine
    assert RaidenCheckpointEngine.consumes_training_engine is True


def test_ray_weight_registry_state_raiden_extensions() -> None:
    state = RayWeightRegistryState()
    assert state.get_controller_address() is None
    assert state.get_global_shapes() == {}

    state.set_controller_address("10.0.0.1:29500")
    assert state.get_controller_address() == "10.0.0.1:29500"

    shapes = {"model.layers.0.weight": [128, 256]}
    state.set_global_shapes(shapes)
    assert state.get_global_shapes() == shapes

    state.set_stats(step=1, stats={"l1_norm": 10.0, "total_numel": 4})
    assert state.get_stats(1) == {"master": {"l1_norm": 10.0, "total_numel": 4}}

    state.clear()
    assert state.get_controller_address() == "10.0.0.1:29500"
    assert state.get_global_shapes() == {}
    assert state.get_stats(1) is None


def test_ray_weight_registry_state_rank_stats_and_pruning() -> None:
    state = RayWeightRegistryState()
    state.set_rank_stats(step=0, rank=0, stats={"l1_norm": 4.0})
    state.set_rank_stats(step=0, rank=1, stats={"l1_norm": 8.0})
    assert state.get_stats(0) == {
        "ranks": {
            0: {"l1_norm": 4.0},
            1: {"l1_norm": 8.0},
        }
    }

    # Verify pruning keeps at most 5 steps
    for step in range(1, 10):
        state.set_stats(step=step, stats={"l1_norm": float(step)})
    assert state.get_stats(0) is None
    assert state.get_stats(4) is None
    assert state.get_stats(5) is not None
    assert state.get_stats(9) is not None


def test_dim0_shard_info_and_merge_rank_stats() -> None:
    from verl.workers.engine.spec import BlockPlacement

    # Unsharded (mesh is None)
    spec_unsharded = SimpleNamespace(mesh=None, full_shape=(8, 4))
    assert _dim0_shard_info(spec_unsharded) is None

    # 1D mesh, evenly sharded on dim 0
    fake_mesh = SimpleNamespace(ndim=1, size=lambda dim: 2)
    spec_sharded = SimpleNamespace(mesh=fake_mesh, full_shape=(8, 4))
    fake_place = BlockPlacement(
        local_shape=(4, 4),
        global_offset=(4, 0),
        full_shape=(8, 4),
    )
    with patch("verl.workers.engine.spec.derive_dtensor_placement", return_value=(fake_place, None, None)):
        info = _dim0_shard_info(spec_sharded)
    assert info == ((4, 4), 2, 1)

    # Merge rank stats across 2 ranks
    rank0_t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rank1_t = torch.tensor([5.0, 6.0, 7.0, 8.0])
    full_t = torch.cat([rank0_t, rank1_t])

    s0 = compute_tensor_stats([("w", rank0_t)])
    s1 = compute_tensor_stats([("w", rank1_t)])
    merged = merge_rank_stats({0: s0, 1: s1})
    expected = compute_tensor_stats([("w", full_t)])

    assert merged["total_numel"] == expected["total_numel"]
    assert pytest.approx(merged["l1_norm"], rel=1e-5) == expected["l1_norm"]
    assert pytest.approx(merged["l2_norm"], rel=1e-5) == expected["l2_norm"]
    assert pytest.approx(merged["per_tensor"]["w"]["l1"], rel=1e-5) == expected["per_tensor"]["w"]["l1"]


def test_export_local_shards_avoids_allgather_on_even_dim0_shard() -> None:
    from verl.workers.engine.spec import BlockPlacement

    fake_mesh = SimpleNamespace(ndim=1, size=lambda dim: 2)
    spec_sharded = SimpleNamespace(mesh=fake_mesh, full_shape=(8, 4), place=None, hf_slots=None)
    spec_replicated = SimpleNamespace(mesh=None, full_shape=(4,), place=None, hf_slots=None)

    t_embed = torch.ones((4, 4))
    t_head = torch.ones((4, 4))
    t_norm = torch.ones((4,))

    fake_engine = SimpleNamespace(
        get_per_tensor_param_shard=lambda: (
            [
                ("model.embed_tokens.weight", t_embed, spec_sharded),
                ("lm_head.weight", t_head, spec_sharded),
                ("model.norm.weight", t_norm, spec_replicated),
            ],
            None,
        )
    )
    fake_place = BlockPlacement(local_shape=(4, 4), global_offset=(4, 0), full_shape=(8, 4))
    with patch("verl.workers.engine.spec.derive_dtensor_placement", return_value=(fake_place, None, None)):
        named, shard_info = export_local_shards(fake_engine)

    names = dict(named)
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" not in names  # tied embedding filtered
    assert "model.norm.weight" in names
    assert shard_info == {"model.embed_tokens.weight": (2, 1)}


def test_raiden_send_weights_rebinds_without_d2h(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sync_mod = SimpleNamespace(synchronize=MagicMock())
    monkeypatch.setitem(sys.modules, "torch_tpu", SimpleNamespace(_internal=SimpleNamespace(sync=fake_sync_mod)))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", SimpleNamespace(sync=fake_sync_mod))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.sync", fake_sync_mod)

    engine = RaidenCheckpointEngine(bucket_size=1024, is_master=True)
    fake_ws = MagicMock()
    engine._trainer_raiden_ws = fake_ws

    t1 = torch.ones((4, 8), dtype=torch.float32)
    t2 = torch.ones((8,), dtype=torch.float32)
    engine._registered_signature = [
        ("layer.0.bias", (8,), torch.float32, None),
        ("layer.0.weight", (4, 8), torch.float32, None),
    ]

    with patch.object(
        raiden_mod,
        "validate_and_sanitize_tensors",
        side_effect=lambda items, device=None: list(items),
    ):
        asyncio.run(engine.send_weights([("layer.0.weight", t1), ("layer.0.bias", t2)], compute_stats=False))

    fake_ws.bind_weights.assert_called_once()
    fake_ws.d2h.assert_not_called()


def test_raiden_send_weights_sharded_engine_posts_rank_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sync_mod = SimpleNamespace(synchronize=MagicMock())
    monkeypatch.setitem(sys.modules, "torch_tpu", SimpleNamespace(_internal=SimpleNamespace(sync=fake_sync_mod)))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", SimpleNamespace(sync=fake_sync_mod))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.sync", fake_sync_mod)

    engine = RaidenCheckpointEngine(bucket_size=1024, is_master=False)
    engine.rank = 1
    fake_ws = MagicMock()
    fake_registry = MagicMock()
    engine._trainer_raiden_ws = fake_ws
    engine.registry = fake_registry

    t_shard = torch.ones((4, 8), dtype=torch.float32)
    engine._registered_signature = [("model.layers.0.weight", (4, 8), torch.float32, (2, 1))]

    fake_training_engine = SimpleNamespace(get_per_tensor_param_shard=MagicMock())

    with (
        patch.object(
            raiden_mod,
            "export_local_shards",
            return_value=([("model.layers.0.weight", t_shard)], {"model.layers.0.weight": (2, 1)}),
        ),
        patch.object(
            raiden_mod,
            "validate_and_sanitize_tensors",
            side_effect=lambda items, device=None: list(items),
        ),
    ):
        asyncio.run(engine.send_weights(fake_training_engine, global_steps=3, compute_stats=True))

    fake_ws.bind_weights.assert_called_once()
    fake_registry.set_rank_stats.remote.assert_called_once()
    call_args = fake_registry.set_rank_stats.remote.call_args[0]
    assert call_args[0] == 3
    assert call_args[1] == 1
    assert call_args[2]["total_numel"] == 32


def test_raiden_worker_extension_install_qkv_fusion_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sync_mod = SimpleNamespace(synchronize=MagicMock())
    monkeypatch.setitem(sys.modules, "torch_tpu", SimpleNamespace(_internal=SimpleNamespace(sync=fake_sync_mod)))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", SimpleNamespace(sync=fake_sync_mod))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.sync", fake_sync_mod)

    assert RAIDEN_SAMPLER_LISTENER_BASE_PORT == 39200

    worker = vLLMRaidenWorkerExtension()
    worker.rank = 3

    qkv_param = torch.zeros((6, 4), dtype=torch.float32)
    gate_up_param = torch.zeros((8, 4), dtype=torch.float32)
    norm_param = torch.zeros((4,), dtype=torch.float32)

    fake_model = MagicMock()
    fake_model.state_dict.return_value = {
        "model.layers.0.self_attn.qkv_proj.weight": qkv_param,
        "model.layers.0.mlp.gate_up_proj.weight": gate_up_param,
        "model.norm.weight": norm_param,
    }
    fake_model.named_modules.return_value = []
    worker.model_runner = SimpleNamespace(model=fake_model)

    worker._raiden_ws = MagicMock()
    worker._raiden_staging = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones((2, 4)),
        "model.layers.0.self_attn.k_proj.weight": torch.full((2, 4), 2.0),
        "model.layers.0.self_attn.v_proj.weight": torch.full((2, 4), 3.0),
        "model.layers.0.mlp.gate_proj.weight": torch.full((4, 4), 4.0),
        "model.layers.0.mlp.up_proj.weight": torch.full((4, 4), 5.0),
        "model.norm.weight": torch.full((4,), 6.0),
    }
    worker._sorted_vllm_params = list(worker._raiden_staging.items())

    rc = worker.install_raiden_weights()
    assert rc == 1
    worker._raiden_ws.h2d.assert_called_once()

    assert torch.all(qkv_param[:2] == 1.0)
    assert torch.all(qkv_param[2:4] == 2.0)
    assert torch.all(qkv_param[4:] == 3.0)
    assert torch.all(gate_up_param[:4] == 4.0)
    assert torch.all(gate_up_param[4:] == 5.0)
    assert torch.all(norm_param == 6.0)

    stats = worker.get_model_weights_stats()
    assert stats["rank"] == 3
    assert "model.layers.0.self_attn.q_proj.weight" in stats["per_tensor"]
    assert "model.norm.weight" in stats["per_tensor"]


def test_update_raiden_weights_clears_kv_cache_and_sets_global_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRaidenId:
        def __init__(self, job_name: str, job_replica_id: str, data_name: str) -> None:
            self.key = (job_name, job_replica_id, data_name)

        def __hash__(self) -> int:
            return hash(self.key)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeRaidenId) and self.key == other.key

    fake_common = SimpleNamespace(RaidenId=FakeRaidenId)
    fake_ctrl_mod = SimpleNamespace(RaidenMemoryType=SimpleNamespace(DRAM="DRAM"))
    monkeypatch.setitem(sys.modules, "tpu_sync", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.api", SimpleNamespace(common=fake_common))
    monkeypatch.setitem(sys.modules, "tpu_sync.api.common", fake_common)
    monkeypatch.setitem(sys.modules, "tpu_sync.rpc", SimpleNamespace(raiden_controller=fake_ctrl_mod))
    monkeypatch.setitem(sys.modules, "tpu_sync.rpc.raiden_controller", fake_ctrl_mod)

    registered = {
        FakeRaidenId("trainer", "0", "weights"): True,
        FakeRaidenId("sampler", "0", "weights"): True,
    }
    fake_transfer_future = SimpleNamespace(wait=AsyncMock())
    fake_controller = SimpleNamespace(
        _lock=threading.Lock(),
        _registered_shards=registered,
        start_transfer=MagicMock(return_value=fake_transfer_future),
    )

    fake_server_handle = SimpleNamespace(
        collective_rpc=SimpleNamespace(remote=MagicMock(side_effect=lambda **kw: AsyncMock()())),
        clear_kv_cache=SimpleNamespace(remote=MagicMock(side_effect=lambda: AsyncMock()())),
        set_global_steps=SimpleNamespace(remote=MagicMock(side_effect=lambda s: AsyncMock()())),
    )
    fake_replica = SimpleNamespace(world_size=1, server_handle=fake_server_handle)
    fake_manager = SimpleNamespace(
        raiden_controller=fake_controller,
        raiden_server=MagicMock(),
        raiden_address="10.0.0.1:29500",
        actor_wg=SimpleNamespace(world_size=1, update_weights=MagicMock(return_value=None)),
        replicas=[fake_replica],
        abort_replicas=AsyncMock(),
        resume_generation_replicas=AsyncMock(),
    )

    asyncio.run(update_raiden_weights(fake_manager, global_steps=5, verify_parity=False))

    fake_manager.abort_replicas.assert_awaited_once()
    fake_controller.start_transfer.assert_called_once()
    fake_server_handle.clear_kv_cache.remote.assert_called_once()
    fake_server_handle.set_global_steps.remote.assert_called_once_with(5)
    fake_manager.resume_generation_replicas.assert_awaited_once()


def test_actor_update_weights_hook_passes_engine_when_consumed() -> None:
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.workers.engine_workers import ActorRolloutRefWorker

    apply_raiden_checkpoint_engine_hooks()
    assert hasattr(ActorRolloutRefWorker.update_weights, MAGIC_ATTR)
    assert getattr(ActorRolloutRefWorker.update_weights, "_verl_consumes_engine_patched", False) is True

    fake_worker = SimpleNamespace(
        config=SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="raiden"))),
        checkpoint_engine=SimpleNamespace(
            consumes_training_engine=True,
            send_weights=AsyncMock(return_value={}),
        ),
        actor=SimpleNamespace(engine="fake_fsdp_engine"),
    )

    asyncio.run(ActorRolloutRefWorker.update_weights(fake_worker, global_steps=3))
    fake_worker.checkpoint_engine.send_weights.assert_awaited_once_with("fake_fsdp_engine", global_steps=3)

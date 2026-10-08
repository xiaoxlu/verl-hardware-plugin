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
    RaidenCheckpointEngine,
    RaidenParityCheck,
    _dim0_shard_info,
    _sampler_job_name,
    apply_raiden_checkpoint_engine_hooks,
    compute_tensor_stats,
    export_local_shards,
    filter_tied_embeddings,
    merge_rank_stats,
    update_raiden_weights,
    vLLMRaidenWorkerExtension,
)
from verl_hardware_plugin.engines.ray_weight_registry import RayWeightRegistryState  # noqa: E402


def _install_fake_tpu_sync(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_sync_mod = SimpleNamespace(synchronize=MagicMock())
    monkeypatch.setitem(sys.modules, "torch_tpu", SimpleNamespace(_internal=SimpleNamespace(sync=fake_sync_mod)))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", SimpleNamespace(sync=fake_sync_mod))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.sync", fake_sync_mod)
    return fake_sync_mod.synchronize


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
    worker._raiden_replicas = {"model.norm.weight": 4}

    rc = worker.install_raiden_weights()
    assert set(rc) == {"total", "h2d", "sync"}
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
    assert stats["per_tensor"]["model.norm.weight"]["replicas"] == 4
    assert "replicas" not in stats["per_tensor"]["model.layers.0.self_attn.q_proj.weight"]


class _FakeRaidenId:
    def __init__(self, job_name: str, job_replica_id: str, data_name: str) -> None:
        self.key = (job_name, job_replica_id, data_name)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeRaidenId) and self.key == other.key

    def __repr__(self) -> str:
        return f"RaidenId{self.key}"


def _make_fake_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_replicas: int = 1,
    tp_size: int = 1,
    num_trainer_ranks: int = 1,
    install_result: Any = None,
    engine_kwargs: dict[str, Any] | None = None,
) -> SimpleNamespace:
    fake_common = SimpleNamespace(RaidenId=_FakeRaidenId)
    fake_ctrl_mod = SimpleNamespace(RaidenMemoryType=SimpleNamespace(DRAM="DRAM"))
    monkeypatch.setitem(sys.modules, "tpu_sync", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tpu_sync.api", SimpleNamespace(common=fake_common))
    monkeypatch.setitem(sys.modules, "tpu_sync.api.common", fake_common)
    monkeypatch.setitem(sys.modules, "tpu_sync.rpc", SimpleNamespace(raiden_controller=fake_ctrl_mod))
    monkeypatch.setitem(sys.modules, "tpu_sync.rpc.raiden_controller", fake_ctrl_mod)
    # The orchestrator only uses ray.get (via asyncio.to_thread) on the actor refs; keep it off the Ray runtime.
    monkeypatch.setattr(raiden_mod, "ray", SimpleNamespace(get=MagicMock(return_value=[{}] * num_trainer_ranks)))

    registered = {_FakeRaidenId("trainer", str(i), "weights"): True for i in range(num_trainer_ranks)}
    for r in range(num_replicas):
        for k in range(tp_size):
            registered[_FakeRaidenId(_sampler_job_name(r, num_replicas), str(k), "weights")] = True
    fake_controller = SimpleNamespace(
        _lock=threading.Lock(),
        _registered_shards=registered,
        start_transfer=MagicMock(side_effect=lambda **kw: SimpleNamespace(wait=AsyncMock())),
    )

    async def _rpc_result(**kw: Any) -> Any:
        if kw.get("method") == "install_raiden_weights":
            return install_result if install_result is not None else [{"total": 0.2, "h2d": 0.1, "sync": 0.01}]
        return [True] * tp_size

    replicas = []
    for _ in range(num_replicas):
        server_handle = SimpleNamespace(
            collective_rpc=SimpleNamespace(remote=MagicMock(side_effect=_rpc_result)),
            clear_kv_cache=SimpleNamespace(remote=MagicMock(side_effect=lambda: AsyncMock()())),
            set_global_steps=SimpleNamespace(remote=MagicMock(side_effect=lambda s: AsyncMock()())),
        )
        replicas.append(SimpleNamespace(world_size=tp_size, server_handle=server_handle))

    return SimpleNamespace(
        raiden_controller=fake_controller,
        raiden_server=MagicMock(),
        raiden_address="10.0.0.1:29500",
        config=SimpleNamespace(engine_kwargs=engine_kwargs or {}),
        actor_wg=SimpleNamespace(
            world_size=num_trainer_ranks,
            update_weights=MagicMock(return_value=None),
            execute_checkpoint_engine=MagicMock(return_value=["release-ref"] * num_trainer_ranks),
        ),
        replicas=replicas,
        abort_replicas=AsyncMock(),
        resume_generation_replicas=AsyncMock(),
    )


def test_update_raiden_weights_clears_kv_cache_and_sets_global_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_manager = _make_fake_manager(monkeypatch, num_trainer_ranks=2)
    fake_controller = fake_manager.raiden_controller
    fake_server_handle = fake_manager.replicas[0].server_handle

    metrics = asyncio.run(update_raiden_weights(fake_manager, global_steps=5))

    fake_manager.abort_replicas.assert_awaited_once()
    fake_controller.start_transfer.assert_called_once()
    assert fake_controller.start_transfer.call_args.kwargs["req_id"] == "verl_step_5"
    fake_server_handle.clear_kv_cache.remote.assert_called_once()
    fake_server_handle.set_global_steps.remote.assert_called_once_with(5)
    fake_manager.resume_generation_replicas.assert_awaited_once()

    # Sampler init and install go through collective_rpc; a single replica keeps the historical job name.
    init_call = fake_server_handle.collective_rpc.remote.call_args_list[0]
    assert init_call.kwargs["method"] == "init_raiden_sync_on_worker"
    assert init_call.kwargs["kwargs"]["job_name"] == "sampler"
    install_call = fake_server_handle.collective_rpc.remote.call_args_list[1]
    assert install_call.kwargs["method"] == "install_raiden_weights"
    assert install_call.kwargs["kwargs"] == {}  # no exact parity requested

    # The trainer send buffers are released after the transfer via the generic checkpoint-engine RPC
    # (one method name per rank for DP_COMPUTE dispatch) and awaited before returning.
    fake_manager.actor_wg.execute_checkpoint_engine.assert_called_once_with(["release_sync_buffers"] * 2)
    raiden_mod.ray.get.assert_called_once_with(["release-ref"] * 2)

    assert metrics["timing_s/tpu-sync/sampler_h2d_pure"] == pytest.approx(0.1)
    for key in ("quiesce", "trainer_init", "sampler_init", "barrier", "p2p_transfer", "sampler_h2d", "total_sync"):
        assert f"timing_s/tpu-sync/{key}" in metrics


def test_update_raiden_weights_multi_replica_uses_distinct_job_names(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _sampler_job_name(0, 1) == "sampler"
    assert [_sampler_job_name(i, 3) for i in range(3)] == ["sampler0", "sampler1", "sampler2"]

    fake_manager = _make_fake_manager(monkeypatch, num_replicas=2, tp_size=4, num_trainer_ranks=8)
    asyncio.run(update_raiden_weights(fake_manager, global_steps=0))

    for idx, replica in enumerate(fake_manager.replicas):
        init_call = replica.server_handle.collective_rpc.remote.call_args_list[0]
        assert init_call.kwargs["kwargs"]["job_name"] == f"sampler{idx}"

    # One transfer per replica, each targeting only that replica's TP ranks under its own job name.
    calls = fake_manager.raiden_controller.start_transfer.call_args_list
    assert len(calls) == 2
    for idx, call in enumerate(calls):
        assert call.kwargs["req_id"] == f"verl_step_0_replica{idx}"
        assert len(call.kwargs["src_units"]) == 8
        assert [u.key for u in call.kwargs["dst_units"]] == [(f"sampler{idx}", str(k), "weights") for k in range(4)]


def test_update_raiden_weights_requests_exact_parity_on_step0_only(monkeypatch: pytest.MonkeyPatch) -> None:
    exact = {"rank": 0, "checked": 3, "num_mismatched": 0, "mismatched": [], "unresolved": []}
    install_result = [{"total": 0.2, "h2d": 0.1, "sync": 0.01, "exact_parity": exact}]
    kwargs = {"raiden": {"verify_parity": "exact"}}

    fake_manager = _make_fake_manager(monkeypatch, install_result=install_result, engine_kwargs=kwargs)
    with patch.object(RaidenParityCheck, "report_exact", return_value=True) as report:
        asyncio.run(update_raiden_weights(fake_manager, global_steps=0))
        install_call = fake_manager.replicas[0].server_handle.collective_rpc.remote.call_args_list[1]
        assert install_call.kwargs["kwargs"] == {"exact_parity": True}
        report.assert_called_once_with(0, [install_result])

    fake_manager = _make_fake_manager(monkeypatch, install_result=install_result, engine_kwargs=kwargs)
    with patch.object(RaidenParityCheck, "report_exact") as report:
        asyncio.run(update_raiden_weights(fake_manager, global_steps=3))
        install_call = fake_manager.replicas[0].server_handle.collective_rpc.remote.call_args_list[1]
        assert install_call.kwargs["kwargs"] == {}
        report.assert_not_called()


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


def test_raiden_worker_extension_and_tpu_worker_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTPUWorker:
        pass

    class FakevLLMHttpServer:
        def _get_worker_extension_cls(self) -> str:
            return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"

    fake_worker_mod = SimpleNamespace(TPUWorker=FakeTPUWorker)
    fake_async_server_mod = SimpleNamespace(vLLMHttpServer=FakevLLMHttpServer)
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu", SimpleNamespace(worker=SimpleNamespace(tpu_worker=fake_worker_mod))
    )
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.worker", SimpleNamespace(tpu_worker=fake_worker_mod))
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.worker.tpu_worker", fake_worker_mod)
    monkeypatch.setitem(
        sys.modules, "verl.workers.rollout.vllm_rollout", SimpleNamespace(vllm_async_server=fake_async_server_mod)
    )
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.vllm_async_server", fake_async_server_mod)

    apply_raiden_checkpoint_engine_hooks()

    for method_name in (
        "_get_vllm_model",
        "init_raiden_sync_on_worker",
        "install_raiden_weights",
        "get_model_weights_stats",
    ):
        assert hasattr(FakeTPUWorker, method_name)

    fake_server_raiden: Any = SimpleNamespace(config={"checkpoint_engine": {"backend": "raiden"}})
    assert FakevLLMHttpServer._get_worker_extension_cls(fake_server_raiden) == raiden_mod.RAIDEN_WORKER_EXTENSION_CLS

    fake_server_tpu: Any = SimpleNamespace(config={"checkpoint_engine": {"backend": "tpu"}})
    assert (
        FakevLLMHttpServer._get_worker_extension_cls(fake_server_tpu)
        == "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
    )


class _FakeWeightSynchronizer:
    def __init__(self) -> None:
        self.unbind_calls = 0
        self.closed = False

    def unbind_weights(self) -> None:
        self.unbind_calls += 1

    def close(self) -> None:
        self.closed = True


def _release_engine(release: bool, with_unbind: bool = True) -> tuple[RaidenCheckpointEngine, Any]:
    # __init__ looks up the Ray-backed registry; only the fields release_sync_buffers reads are needed.
    engine = RaidenCheckpointEngine.__new__(RaidenCheckpointEngine)
    ws: Any = MagicMock(spec=["close"]) if not with_unbind else _FakeWeightSynchronizer()
    engine.rank = 0
    engine.release_buffers_after_sync = release
    engine._trainer_raiden_ws = ws
    engine._registered_signature = [("w", (2, 2), torch.bfloat16, None)]
    engine._bound_tensors = [object()]  # type: ignore[list-item]
    return engine, ws


def test_release_unbinds_but_keeps_synchronizer_and_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    sync = _install_fake_tpu_sync(monkeypatch)
    engine, ws = _release_engine(release=True)

    assert engine.release_sync_buffers() == {}

    assert ws.unbind_calls == 1
    assert not ws.closed
    assert engine._bound_tensors is None
    # Kept so the next send_weights takes the bind_weights reuse path instead of re-creating the synchronizer.
    assert engine._trainer_raiden_ws is ws
    assert engine._registered_signature == [("w", (2, 2), torch.bfloat16, None)]
    sync.assert_called_once_with(wait=True)


def test_release_is_noop_when_disabled_or_before_first_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tpu_sync(monkeypatch)
    engine, ws = _release_engine(release=False)
    assert engine.release_sync_buffers() == {}
    assert ws.unbind_calls == 0
    assert engine._bound_tensors is not None
    assert engine._trainer_raiden_ws is ws

    engine, _ = _release_engine(release=True)
    engine._trainer_raiden_ws = None
    assert engine.release_sync_buffers() == {}


def test_release_falls_back_to_destroying_synchronizer_without_unbind(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tpu_sync(monkeypatch)
    engine, ws = _release_engine(release=True, with_unbind=False)
    assert not hasattr(ws, "unbind_weights")

    assert engine.release_sync_buffers() == {}

    # Without unbind the only way to return the HBM is to drop the synchronizer; the signature is cleared so the
    # next send_weights re-creates and re-registers it instead of calling bind_weights on a dead object.
    assert engine._trainer_raiden_ws is None
    assert engine._bound_tensors is None
    assert engine._registered_signature is None


def test_release_flag_from_kwargs_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERL_RAIDEN_RELEASE_BUFFERS", raising=False)
    assert RaidenCheckpointEngine().release_buffers_after_sync is True
    assert RaidenCheckpointEngine(release_buffers_after_sync=False).release_buffers_after_sync is False
    assert RaidenCheckpointEngine(release_buffers_after_sync="false").release_buffers_after_sync is False
    monkeypatch.setenv("VERL_RAIDEN_RELEASE_BUFFERS", "0")
    assert RaidenCheckpointEngine(release_buffers_after_sync=True).release_buffers_after_sync is False
    monkeypatch.setenv("VERL_RAIDEN_RELEASE_BUFFERS", "1")
    assert RaidenCheckpointEngine(release_buffers_after_sync=False).release_buffers_after_sync is True


def test_filter_tied_embeddings_keeps_lm_head_for_untied_models() -> None:
    items = [("model.embed_tokens.weight", 1), ("lm_head.weight", 2), ("model.norm.weight", 3)]
    assert [k for k, _ in filter_tied_embeddings(items)] == ["model.embed_tokens.weight", "model.norm.weight"]
    assert [k for k, _ in filter_tied_embeddings(items, tie_word_embeddings=False)] == [k for k, _ in items]
    # Without an embedding in the export there is nothing to tie to, so lm_head is always kept.
    assert filter_tied_embeddings([("lm_head.weight", 2)]) == [("lm_head.weight", 2)]


def test_raiden_send_weights_reads_tie_word_embeddings_from_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tpu_sync(monkeypatch)
    engine = RaidenCheckpointEngine(bucket_size=1024, is_master=False)
    engine._trainer_raiden_ws = MagicMock()
    engine._registered_signature = [("lm_head.weight", (4, 8), torch.float32, None)]

    fake_training_engine = SimpleNamespace(
        get_per_tensor_param_shard=MagicMock(),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(tie_word_embeddings=False)),
    )
    with (
        patch.object(
            raiden_mod, "export_local_shards", return_value=([("lm_head.weight", torch.ones((4, 8)))], {})
        ) as export,
        patch.object(raiden_mod, "validate_and_sanitize_tensors", side_effect=lambda items, device=None: list(items)),
    ):
        asyncio.run(engine.send_weights(fake_training_engine, compute_stats=False))
    export.assert_called_once_with(fake_training_engine, tie_word_embeddings=False)


def test_raiden_parity_check_modes_and_exact_report() -> None:
    assert RaidenParityCheck(False).mode == "off"
    assert RaidenParityCheck(True).mode == "norm"
    assert RaidenParityCheck("1").mode == "norm"
    assert RaidenParityCheck("EXACT").mode == "exact"
    assert RaidenParityCheck("all").norm and RaidenParityCheck("all").exact
    with pytest.raises(ValueError):
        RaidenParityCheck("bogus")

    assert RaidenParityCheck.from_config(None).mode == "off"
    assert RaidenParityCheck.from_config({"raiden": {"verify_parity": "exact"}}).mode == "exact"
    assert RaidenParityCheck.from_config({"verify_parity": True}).mode == "norm"

    exact = RaidenParityCheck("exact")
    assert exact.exact_install_kwargs(0) == {"exact_parity": True}
    assert exact.exact_install_kwargs(None) == {"exact_parity": True}
    assert exact.exact_install_kwargs(1) == {}
    assert RaidenParityCheck("norm").exact_install_kwargs(0) == {}

    ok = {"rank": 0, "checked": 2, "num_mismatched": 0, "mismatched": [], "unresolved": []}
    bad = {"rank": 1, "checked": 2, "num_mismatched": 1, "mismatched": ["w: max|diff|=1"], "unresolved": ["x"]}
    assert RaidenParityCheck.report_exact(0, [[{"exact_parity": ok}]]) is True
    assert RaidenParityCheck.report_exact(0, [[{"exact_parity": ok}, {"exact_parity": bad}]]) is False
    assert RaidenParityCheck.report_exact(0, [[{"total": 0.1}]]) is False


def test_install_raiden_weights_exact_parity_and_received_lm_head(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tpu_sync(monkeypatch)
    worker = vLLMRaidenWorkerExtension()
    worker.rank = 1

    o_proj = torch.ones((4, 4))
    lm_head_param = torch.zeros((4, 4))
    embed_param = torch.full((4, 4), 9.0)
    fake_model = MagicMock()
    fake_model.state_dict.return_value = {
        "model.layers.0.self_attn.o_proj.weight": o_proj,
        "lm_head.weight": lm_head_param,
        "model.embed_tokens.weight": embed_param,
    }
    fake_model.named_modules.return_value = []
    fake_model.lm_head.weight = lm_head_param
    fake_model.model.embed_tokens.weight = embed_param
    worker.model_runner = SimpleNamespace(model=fake_model)
    worker._raiden_ws = MagicMock()
    worker._raiden_staging = {
        "model.layers.0.self_attn.o_proj.weight": torch.ones((4, 4)),  # bit-identical to the vLLM param
        "lm_head.weight": torch.full((4, 4), 2.0),  # differs from the vLLM param
        "model.layers.0.not_in_model.weight": torch.ones((2,)),  # cannot be resolved
    }

    result = worker.install_raiden_weights(exact_parity=True)

    exact = result["exact_parity"]
    assert exact["rank"] == 1
    assert exact["checked"] == 2
    assert exact["num_mismatched"] == 1 and exact["mismatched"][0].startswith("lm_head.weight")
    assert exact["unresolved"] == ["model.layers.0.not_in_model.weight"]
    # The trainer sent lm_head (untied model): it must not be overwritten by embed_tokens afterwards.
    assert torch.all(lm_head_param == 2.0)

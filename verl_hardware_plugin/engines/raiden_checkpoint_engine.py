# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Raiden P2P CheckpointEngine for multi-host TPU weight synchronization."""

import asyncio
import functools
import inspect
import logging
import os
import time
from collections.abc import Generator, Iterable
from typing import Any

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl_hardware_plugin.engines.ray_weight_registry import get_or_create_ray_weight_registry

_BaseTPUWorkerExtension: Any
try:
    from verl_hardware_plugin.rollout.tpu_worker_extension import TPUvLLMWorkerExtension as _TPUExt

    _BaseTPUWorkerExtension = _TPUExt
except Exception:
    try:
        from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension as _ColocateExt

        _BaseTPUWorkerExtension = _ColocateExt
    except Exception:
        _BaseTPUWorkerExtension = object

logger = logging.getLogger(__name__)

# Sampler WeightSynchronizer listener base port (chosen well outside KubeRay's
# raylet worker port range 10002..19999 to avoid multi-host port collisions).
RAIDEN_SAMPLER_LISTENER_BASE_PORT = 39200


def compute_tensor_stats(items: Iterable[Any]) -> dict[str, Any]:
    """Computes deterministic L1/L2 norms and parameter counts across tensors on device."""
    items_list = list(items)
    total_numel = 0
    total_l1 = 0.0
    total_l2_sq = 0.0
    per_tensor: dict[str, dict[str, Any]] = {}

    for item in items_list:
        if isinstance(item, tuple) and len(item) == 2:
            name, p = item
        else:
            name = None
            p = item
        p_local = p.to_local() if hasattr(p, "to_local") else p

        p_float = p_local.float()
        t_l1 = float(p_float.abs().sum().item())
        t_l2_sq = float(p_float.pow(2).sum().item())
        numel = p_local.numel()

        if name is not None:
            per_tensor[name] = {
                "l1": t_l1,
                "l2": float(t_l2_sq**0.5),
                "l2_sq": t_l2_sq,
                "numel": numel,
                "shape": list(p_local.shape),
                "dtype": str(p_local.dtype),
            }
        total_numel += numel
        total_l1 += t_l1
        total_l2_sq += t_l2_sq

    return {
        "total_numel": total_numel,
        "num_tensors": len(items_list),
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def create_torch_weight_synchronizer(
    device_tensors: list[list[torch.Tensor]],
    local_port: int = 0,
    parallelism: int = 8,
    listener_port: int = 0,
    bind_ip: str = "127.0.0.1",
) -> Any:
    """Creates an instance of ``WeightSynchronizer`` using ``tpu_sync``."""
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        device_tensors,
        local_port=local_port,
        parallelism=parallelism,
        listener_port=listener_port,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        auto_h2d=False,
    )


def _unwrap_tensor(t: Any) -> Any:
    """Extracts the raw underlying local tensor from ``nn.Parameter`` or ``DTensor`` wrappers."""
    return t.to_local().data if hasattr(t, "to_local") else (t.data if hasattr(t, "data") else t)


def filter_tied_embeddings(named_items: Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Excludes redundant ``lm_head`` weights if embedding tokens are present."""
    items = list(named_items)
    has_embed = any("embed_tokens" in k or "tok_embeddings" in k for k, _ in items)
    if has_embed:
        items = [(k, v) for k, v in items if not (k == "lm_head.weight" or k.endswith(".lm_head.weight"))]
    return items


def validate_and_sanitize_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    device: torch.device | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Validates and sanitizes tensors for zero-copy DMA registration with Raiden."""
    if device is None:
        device = torch.device("tpu")

    sanitized: list[tuple[str, torch.Tensor]] = []
    for name, p in named_tensors:
        if p is None:
            continue
        t = _unwrap_tensor(p)
        if not isinstance(t, torch.Tensor):
            continue

        if t.numel() == 0 or getattr(t, "is_meta", False):
            continue

        if not (hasattr(t, "device") and str(t.device).startswith(str(device).split(":")[0])):
            try:
                t = t.to(device)
            except Exception as e:
                logger.warning("Could not move %s to %s: %s", name, device, e)
                continue

        if not t.is_contiguous():
            t = t.contiguous()

        sanitized.append((name, t))

    return sanitized


def _dim0_shard_info(spec: Any) -> tuple[tuple[int, ...], int, int] | None:
    """Returns ``(local_shape, num_shards, shard_index)`` if the local shard is an even dim-0 cut on a 1-D mesh.

    That is the FSDP2 ``Shard(0)`` layout, which Raiden can describe directly: the variable keeps its full
    shape, is split ``num_shards`` ways on dim 0, and this rank holds block ``shard_index``. Uneven cuts
    are excluded because torch gives the remainder to the leading shards while Raiden gives it to the last.
    """
    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    if spec.mesh is None or spec.mesh.ndim != 1:
        return None
    place, _, _ = derive_dtensor_placement(spec)
    if not isinstance(place, BlockPlacement) or not place.is_flat_contiguous:
        return None
    num_shards = spec.mesh.size(0)
    full_dim0 = int(spec.full_shape[0])
    if num_shards <= 1 or full_dim0 % num_shards:
        return None
    return tuple(place.local_shape), num_shards, int(place.global_offset[0]) // (full_dim0 // num_shards)


def export_local_shards(engine: Any) -> tuple[list[tuple[str, torch.Tensor]], dict[str, tuple[int, int]]]:
    """Exports this rank's weights for Raiden without all-gathering FSDP shards.

    Returns ``(named_tensors, shard_info)``. ``shard_info[name] = (num_shards, shard_index)`` for tensors
    exported as a local dim-0 shard; tensors absent from it are full (replicated) on every rank. Tensors
    whose layout Raiden cannot express are all-gathered here, in the same order on every rank.
    """
    from torch.distributed.tensor import DTensor

    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    gen, _ = engine.get_per_tensor_param_shard()
    items = filter_tied_embeddings((name, (local, spec)) for name, local, spec in gen)

    named: list[tuple[str, torch.Tensor]] = []
    shard_info: dict[str, tuple[int, int]] = {}
    for name, (local, spec) in items:
        if spec.place is not None or spec.hf_slots is not None:
            raise NotImplementedError(
                f"Raiden sharded export does not support exporter-defined placements or expert stacks ({name})"
            )
        full_shape = tuple(int(d) for d in spec.full_shape)
        info = _dim0_shard_info(spec)
        if info is not None:
            local_shape, num_shards, shard_index = info
            named.append((name, local.view(local_shape)))
            shard_info[name] = (num_shards, shard_index)
            continue
        place = derive_dtensor_placement(spec)[0] if spec.mesh is not None else 0
        if not isinstance(place, BlockPlacement):
            named.append((name, local.view(full_shape)))
            continue
        strides = [1] * len(full_shape)
        for d in range(len(full_shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * full_shape[d + 1]
        dt = DTensor.from_local(
            local.view(place.local_shape),
            spec.mesh,
            spec.placements,
            run_check=False,
            shape=torch.Size(full_shape),
            stride=tuple(strides),
        )
        named.append((name, dt.full_tensor()))
    return named, shard_info


def merge_rank_stats(rank_stats: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    """Sums per-tensor partial stats (``l1``, ``l2_sq``, ``numel``) posted by every trainer rank."""
    per_tensor: dict[str, dict[str, Any]] = {}
    for stats in rank_stats.values():
        for name, t in stats.get("per_tensor", {}).items():
            acc = per_tensor.setdefault(name, {"l1": 0.0, "l2_sq": 0.0, "numel": 0})
            acc["l1"] += t["l1"]
            acc["l2_sq"] += t["l2_sq"]
            acc["numel"] += t["numel"]
    for acc in per_tensor.values():
        acc["l2"] = acc["l2_sq"] ** 0.5
    total_l2_sq = sum(t["l2_sq"] for t in per_tensor.values())
    return {
        "total_numel": sum(t["numel"] for t in per_tensor.values()),
        "num_tensors": len(per_tensor),
        "l1_norm": sum(t["l1"] for t in per_tensor.values()),
        "l2_norm": total_l2_sq**0.5,
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def setup_raiden_controller() -> tuple[Any, Any, str]:
    """Starts an embedded ``RaidenControllerServer`` on the head node and records its address in the registry."""
    from tpu_sync.rpc import raiden_controller

    controller = raiden_controller.RaidenController(port=0)
    server = raiden_controller.RaidenControllerServer(controller)
    port = server.start()
    ip = ray.util.get_node_ip_address().strip("[]")
    address = f"{ip}:{port}"
    logger.info("RaidenControllerServer started on head node: %s", address)

    try:
        registry = get_or_create_ray_weight_registry()
        if registry is None:
            raise RuntimeError("Ray is not initialized; cannot store RaidenController address.")
        ray.get(registry.set_controller_address.remote(address))
        logger.info("Stored RaidenController address (%s) in RayWeightRegistry", address)
    except Exception as reg_err:
        raise RuntimeError(
            f"Failed to store RaidenController address ({address}) in RayWeightRegistry: {reg_err}"
        ) from reg_err

    return controller, server, address


@CheckpointEngineRegistry.register("raiden")
class RaidenCheckpointEngine(CheckpointEngine):
    """P2P weight synchronizer checkpoint engine for TPUs using Google Raiden (``tpu-sync``)."""

    # Receive the training engine in send_weights so each rank can register its local FSDP shard
    # (Raiden reshards to the sampler layout) instead of all-gathering full tensors.
    consumes_training_engine = True

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs: Any) -> None:
        self.is_master = is_master
        self.bucket_size = bucket_size
        self.backend = "raiden"
        self.verify_parity = kwargs.get("verify_parity", False)
        self.parallelism = kwargs.get("parallelism", 8)
        self.tp_size = kwargs.get("tp_size", kwargs.get("tensor_parallel_size", self.parallelism))
        self._trainer_raiden_ws: Any = None
        self._trainer_chunks: list[Any] = []
        self._controller_addr: str | None = None
        self._registered_signature: list[tuple[str, tuple[int, ...], torch.dtype, tuple[int, int] | None]] | None = None
        self._bound_tensors: list[torch.Tensor] | None = None
        self._shard_info: dict[str, tuple[int, int]] = {}
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", "0"))

        self.registry = get_or_create_ray_weight_registry()

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict, dict]:
        return {}, {}

    def init_process_group(self, **kwargs: Any) -> None:
        pass

    def finalize(self) -> None:
        if self._trainer_raiden_ws is not None:
            try:
                self._trainer_raiden_ws.close()
            except Exception:
                pass
            self._trainer_raiden_ws = None
        self._registered_signature = None
        self._bound_tensors = None

    async def _create_and_register(self, valid_weights: list[tuple[str, torch.Tensor]]) -> None:
        """(Re)creates the trainer ``WeightSynchronizer`` on ``valid_weights`` and registers it with the controller."""
        if self._trainer_raiden_ws is not None:
            try:
                self._trainer_raiden_ws.close()
            except Exception:
                pass
            self._trainer_raiden_ws = None

        logger.info("Trainer Rank %s: binding %d tensors to WeightSynchronizer", self.rank, len(valid_weights))
        bind_ip = ray.util.get_node_ip_address().strip("[]")
        self._trainer_raiden_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_weights],
            local_port=0,
            listener_port=0,
            parallelism=getattr(self, "parallelism", 8),
            bind_ip=bind_ip,
        )

        def _full_shape(name: str, p: torch.Tensor) -> list[int]:
            shape = list(p.shape)
            if name in self._shard_info:
                shape[0] *= self._shard_info[name][0]
            return shape

        if self.registry is not None:
            try:
                global_shapes = {name: _full_shape(name, p) for name, p in valid_weights}
                ray.get(self.registry.set_global_shapes.remote(global_shapes))
            except Exception as e:
                logger.warning("Could not record global shapes in RayWeightRegistry: %s", e)

        from tpu_sync.rpc import raiden_service_pb2

        variable_protos = []
        max_shards = 1
        for idx, (name, p) in enumerate(valid_weights):
            shape = _full_shape(name, p)
            itemsize = p.element_size()
            layout = list(range(len(shape) - 1, -1, -1))
            if name in self._shard_info:
                num_shards, shard_index = self._shard_info[name]
                max_shards = max(max_shards, num_shards)
                extra: dict[str, Any] = {
                    "mesh_shape": [num_shards] + [1] * (len(shape) - 1),
                    "sharding_spec": ["fsdp"] + [""] * (len(shape) - 1),
                    "global_shard_indices": [shard_index],
                }
            else:
                extra = {"mesh_shape": [1] * len(shape), "sharding_spec": [""] * len(shape)}
            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=shape,
                    layout=layout,
                    item_size=itemsize,
                    layer_idx=idx,
                    **extra,
                )
            )
        num_sharded = sum(name in self._shard_info for name, _ in valid_weights)
        logger.info(
            "Trainer Rank %s: registering %d sharded and %d full tensors",
            self.rank,
            num_sharded,
            len(valid_weights) - num_sharded,
        )

        if self._controller_addr is None and self.registry is not None:
            for _ in range(30):
                try:
                    self._controller_addr = ray.get(self.registry.get_controller_address.remote())
                    if self._controller_addr:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)

        if self._controller_addr:
            from tpu_sync.rpc import raiden_controller

            try:
                ctrl_client = raiden_controller.RaidenControllerClientFacade(self._controller_addr)
                unit_id = raiden_controller.RaidenId("trainer", str(self.rank), "weights")
                ctrl_client.register_work_unit(
                    unit_id,
                    [f"{bind_ip}:{self._trainer_raiden_ws.local_port}"],
                    f"{bind_ip}:{self._trainer_raiden_ws.listener_port}",
                    mesh_shape=[max_shards, 1],
                    variables=variable_protos,
                    mesh_axes=["fsdp", "tp"],
                )
                logger.info(
                    "Trainer Rank %s bound %d dynamic tensors and registered with RaidenController (%s): "
                    "data_port=%s, listener_port=%s",
                    self.rank,
                    len(valid_weights),
                    self._controller_addr,
                    self._trainer_raiden_ws.local_port,
                    self._trainer_raiden_ws.listener_port,
                )
            except Exception as reg_err:
                logger.error(
                    "Trainer Rank %s failed to register with RaidenController (%s): %s",
                    self.rank,
                    self._controller_addr,
                    reg_err,
                )
                raise
        else:
            raise RuntimeError(
                f"Trainer Rank {self.rank}: No RaidenController address found in RayWeightRegistry after timeout"
            )

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Any,
        dst_peers: list[str] | None = None,
        global_steps: int | None = None,
        compute_stats: bool | None = None,
        compute_checksum: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Registers weights with ``RaidenController`` and prepares for coordinated P2P network transfer."""
        if compute_stats is None:
            compute_stats = compute_checksum if compute_checksum is not None else getattr(self, "verify_parity", False)
        step_key = global_steps if global_steps is not None else 0
        logger.info("RaidenCheckpointEngine: [Step %s] Start send_weights...", step_key)

        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logger.warning("Could not synchronize via torch_tpu: %s", e)
            raise RuntimeError(f"TPU synchronization failed: {e}") from e

        if hasattr(weights, "get_per_tensor_param_shard"):
            named_weights, self._shard_info = export_local_shards(weights)
        else:
            weight_items = weights.items() if hasattr(weights, "items") else weights
            named_weights = filter_tied_embeddings(weight_items)
            self._shard_info = {}

        unfused_weights = {k: _unwrap_tensor(v) for k, v in named_weights}
        sorted_weights = sorted(unfused_weights.items(), key=lambda x: x[0])
        valid_weights = validate_and_sanitize_tensors(sorted_weights)

        torch_tpu_sync.synchronize(wait=True)

        # Reuse the WeightSynchronizer across steps when the (name, shape, dtype, shard) signature is unchanged.
        signature = [(name, tuple(t.shape), t.dtype, self._shard_info.get(name)) for name, t in valid_weights]
        if self._trainer_raiden_ws is not None and signature == self._registered_signature:
            self._trainer_raiden_ws.bind_weights([[t] for _, t in valid_weights])
        else:
            await self._create_and_register(valid_weights)
            self._registered_signature = signature
        self._bound_tensors = [t for _, t in valid_weights]

        # Note: No explicit ws.d2h() here. PushWeightsResharded (triggered by the controller's
        # start_transfer) performs a pipelined D2H overlapped with H2H using the per-transfer
        # skip_tiling plan.

        if not compute_stats or self.registry is None:
            return
        try:
            if self._shard_info:
                items = [(n, t) for n, t in valid_weights if n in self._shard_info or self.rank == 0]
                trainer_stats = compute_tensor_stats(items)
                trainer_stats["rank"] = self.rank
                self.registry.set_rank_stats.remote(step_key, self.rank, trainer_stats)
            elif self.is_master:
                trainer_stats = compute_tensor_stats(valid_weights)
                trainer_stats["rank"] = self.rank
                self.registry.set_stats.remote(step_key, trainer_stats)
            else:
                return
            logger.info(
                "[RAIDEN PARITY] Stored trainer rank %s stats for step %s: L1=%.4f, numel=%d",
                self.rank,
                step_key,
                trainer_stats["l1_norm"],
                trainer_stats["total_numel"],
            )
        except Exception as e:
            logger.warning("Failed to record trainer rank %s stats in RayWeightRegistry: %s", self.rank, e)

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
        **kwargs: Any,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError("Rollout on TPU with Raiden uses vLLMRaidenWorkerExtension via collective_rpc.")


class vLLMRaidenWorkerExtension(_BaseTPUWorkerExtension):  # type: ignore[misc]
    """vLLM worker extension for Google Cloud TPU with Raiden P2P weight synchronization.

    Workloads using the ``raiden`` checkpoint engine use this worker extension to bind
    TPU model staging tensors with ``WeightSynchronizer`` and install received weights
    into vLLM parameters via H2D DMA during rollout.
    """

    _raiden_ws: Any = None
    _raiden_staging: dict[str, torch.Tensor]
    _sorted_vllm_params: list[tuple[str, torch.Tensor]]

    def _get_vllm_model(self) -> Any:
        """Extracts the underlying ``nn.Module`` from the vLLM worker."""
        worker: Any = getattr(self, "worker", self)
        return worker.model_runner.model

    def init_raiden_sync_on_worker(self, parallelism: int = 8) -> bool:
        """Initializes Raiden ``WeightSynchronizer`` listener and registers with ``RaidenController``."""
        if hasattr(self, "_raiden_ws") and self._raiden_ws is not None:
            return True

        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            logger.warning("Raiden Sampler: could not locate vllm_model to bind parameters.")
            return False

        from tpu_sync.rpc import raiden_service_pb2

        bind_ip = ray.util.get_node_ip_address().strip("[]")
        rank_val = getattr(self, "rank", 0)
        listener_port = RAIDEN_SAMPLER_LISTENER_BASE_PORT + rank_val

        registry = get_or_create_ray_weight_registry()
        if registry is None:
            raise RuntimeError(f"Raiden Sampler Rank {rank_val}: Ray is not initialized.")

        controller_addr: str | None = None
        global_shapes_map: dict[str, list[int]] = {}
        for _ in range(30):
            try:
                addr, shapes = ray.get(
                    [
                        registry.get_controller_address.remote(),
                        registry.get_global_shapes.remote(),
                    ]
                )
                if addr and shapes:
                    controller_addr, global_shapes_map = addr, shapes
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not controller_addr or not global_shapes_map:
            raise RuntimeError(
                f"Raiden Sampler Rank {rank_val}: No RaidenController address or "
                "global shapes found in RayWeightRegistry after timeout"
            )

        worker = getattr(self, "worker", self)
        parallel_config = getattr(getattr(worker, "vllm_config", None), "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", None)
        if not tp_size or tp_size <= 0:
            raise RuntimeError(
                f"Raiden Sampler Rank {rank_val}: Unable to determine tensor_parallel_size (tp_size) "
                "from worker.vllm_config.parallel_config!"
            )

        row_parallel_suffixes = (".o_proj.weight", ".down_proj.weight")

        staging_tensors: dict[str, torch.Tensor] = {}
        variable_protos: list[Any] = []
        valid_params: list[tuple[str, torch.Tensor]] = []

        for idx, (name, raw_g_shape) in enumerate(sorted(global_shapes_map.items(), key=lambda x: x[0])):
            g_shape = list(raw_g_shape)
            spec_axes = [""] * len(g_shape)
            local_shape = list(g_shape)

            if len(g_shape) == 2:
                if any(name.endswith(s) for s in row_parallel_suffixes):
                    spec_axes = ["", "tp"]
                    local_shape[1] = g_shape[1] // tp_size
                elif g_shape[0] % tp_size == 0:
                    spec_axes = ["tp", ""]
                    local_shape[0] = g_shape[0] // tp_size

            sharding_mesh = [tp_size if axis == "tp" else 1 for axis in spec_axes]

            t = torch.empty(local_shape, dtype=torch.bfloat16, device=torch.device("tpu"))
            staging_tensors[name] = t
            valid_params.append((name, t))

            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=g_shape,
                    mesh_shape=sharding_mesh,
                    layout=list(range(len(local_shape) - 1, -1, -1)),
                    item_size=t.element_size(),
                    layer_idx=idx,
                    sharding_spec=spec_axes,
                )
            )

        self._raiden_staging = staging_tensors
        self._sorted_vllm_params = valid_params

        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logger.warning("Could not synchronize via torch_tpu: %s", e)

        logger.info(
            "Raiden Sampler Rank %s: binding %d staging tensors to WeightSynchronizer "
            "(sample: name=%s, shape=%s, total numel=%d)",
            rank_val,
            len(valid_params),
            valid_params[0][0],
            valid_params[0][1].shape,
            sum(t.numel() for _, t in valid_params),
        )

        self._raiden_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_params],
            local_port=0,
            parallelism=parallelism,
            listener_port=listener_port,
            bind_ip=bind_ip,
        )

        try:
            from tpu_sync.rpc import raiden_controller

            ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_addr)
            unit_id = raiden_controller.RaidenId("sampler", str(rank_val), "weights")
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._raiden_ws.local_port}"],
                f"{bind_ip}:{self._raiden_ws.listener_port}",
                mesh_shape=[1, tp_size],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
            logger.info(
                "Raiden Sampler Rank %s bound %d dynamic staging tensors and registered "
                "with RaidenController (%s): mesh_shape=[1, %d], data_port=%s, listener_port=%s",
                rank_val,
                len(valid_params),
                controller_addr,
                tp_size,
                self._raiden_ws.local_port,
                self._raiden_ws.listener_port,
            )
        except Exception as e:
            logger.error("Raiden Sampler Rank %s failed to register with RaidenController: %s", rank_val, e)
            raise

        return True

    @torch.no_grad()
    def install_raiden_weights(self) -> int:
        """Installs received weights from host staging buffers into TPU HBM via zero-copy H2D DMA
        and fuses/transposes them into vLLM model parameters.
        """
        if not hasattr(self, "_raiden_ws") or self._raiden_ws is None:
            logger.warning("Raiden Sampler: install_raiden_weights called before _raiden_ws was initialized.")
            return 0

        t_start = time.perf_counter()
        t_h2d_start = time.perf_counter()
        self._raiden_ws.h2d()
        t_h2d = time.perf_counter() - t_h2d_start

        vllm_model = self._get_vllm_model()
        model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
        module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}

        def resolve_model_key(k: str) -> str | None:
            if k in model_sd:
                return k
            if k.startswith("model.") and k[6:] in model_sd:
                return k[6:]
            if f"model.{k}" in model_sd:
                return f"model.{k}"
            return None

        def get_parent_mod(target_key: str) -> Any:
            parent = target_key.rsplit(".", 1)[0] if "." in target_key else ""
            if parent in module_dict:
                return module_dict[parent]
            if parent.startswith("model.") and parent[6:] in module_dict:
                return module_dict[parent[6:]]
            if f"model.{parent}" in module_dict:
                return module_dict[f"model.{parent}"]
            return None

        def to_target_layout(tensor: torch.Tensor, target_local: torch.Tensor, flipped: bool) -> torch.Tensor:
            if tensor.ndim == 2 and (
                flipped or (tensor.shape != target_local.shape and tensor.T.shape == target_local.shape)
            ):
                return tensor.transpose(0, 1).contiguous()
            return tensor.contiguous()

        fused_map = [
            (
                ".self_attn.qkv_proj.weight",
                [".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            ),
            (
                ".self_attn.qkv_proj.bias",
                [".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            ),
            (".mlp.gate_up_proj.weight", [".mlp.gate_proj.weight", ".mlp.up_proj.weight"]),
            (".mlp.gate_up_proj.bias", [".mlp.gate_proj.bias", ".mlp.up_proj.bias"]),
        ]

        consumed_staging: set[str] = set()

        # 1. Handle fused projections (QKV and Gate-Up)
        for target_suffix, src_suffixes in fused_map:
            primary_src = src_suffixes[0]
            for name in list(self._raiden_staging.keys()):
                if primary_src in name:
                    layer_src_keys = [name.replace(primary_src, s) for s in src_suffixes]
                    if all(sk in self._raiden_staging for sk in layer_src_keys):
                        target_name = name.replace(primary_src, target_suffix)
                        resolved_key = resolve_model_key(target_name)
                        if resolved_key is not None:
                            target_param = model_sd[resolved_key]
                            target_local = (
                                target_param.to_local() if hasattr(target_param, "to_local") else target_param
                            )
                            parent_mod = get_parent_mod(resolved_key)
                            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
                            parts = [self._raiden_staging[sk] for sk in layer_src_keys]
                            fused = torch.cat(parts, dim=0)
                            fused_adapted = to_target_layout(fused, target_local, is_flipped)
                            target_local.copy_(fused_adapted)
                            consumed_staging.update(layer_src_keys)

        # 2. Handle remaining non-fused parameters (o_proj, down_proj, layernorms, embeddings)
        for name, src_t in self._raiden_staging.items():
            if name in consumed_staging:
                continue
            resolved_key = resolve_model_key(name)
            if resolved_key is not None:
                target_param = model_sd[resolved_key]
                target_local = target_param.to_local() if hasattr(target_param, "to_local") else target_param
                parent_mod = get_parent_mod(resolved_key)
                is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
                adapted = to_target_layout(src_t, target_local, is_flipped)
                target_local.copy_(adapted)

        # 3. Handle tied word embeddings
        if (
            hasattr(vllm_model, "lm_head")
            and hasattr(vllm_model, "model")
            and hasattr(vllm_model.model, "embed_tokens")
        ):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.model.embed_tokens.weight)
        elif hasattr(vllm_model, "lm_head") and hasattr(vllm_model, "embed_tokens"):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.embed_tokens.weight)

        t_sync_start = time.perf_counter()
        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logger.warning("Could not synchronize via torch_tpu: %s", e)
        t_sync = time.perf_counter() - t_sync_start
        t_total = time.perf_counter() - t_start

        logger.info(
            "[RAIDEN TELEMETRY | Sampler Worker] Rank %s: install_raiden_weights completed in %.4fs "
            "(H2D=%.4fs, TPUSyncBarrier=%.4fs)",
            getattr(self, "rank", 0),
            t_total,
            t_h2d,
            t_sync,
        )
        return 1

    def get_model_weights_stats(self, include_shards: bool = False) -> dict[str, Any]:
        """Computes deterministic parameter count, L1 norm, and L2 norm across staging parameters."""
        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            return {"error": "No model found on worker"}

        if not hasattr(self, "_sorted_vllm_params") or not self._sorted_vllm_params:
            raise RuntimeError(
                "get_model_weights_stats is only supported for Raiden after initialization, "
                f"but self._sorted_vllm_params is not initialized on worker rank {getattr(self, 'rank', 'unknown')}!"
            )

        rank_val = getattr(self, "rank", 0)
        res = compute_tensor_stats(self._sorted_vllm_params)
        res["rank"] = rank_val
        return res


async def update_raiden_weights(
    manager: Any,
    global_steps: int | None = None,
    verify_parity: bool = False,
) -> dict[str, Any]:
    """Orchestrates Raiden TPU P2P weight synchronization via the embedded ``RaidenController``."""
    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning("Failed to abort replicas at step %s: %s", global_steps, e)
    t_abort = time.perf_counter() - t_abort_start

    if hasattr(manager, "config") and hasattr(manager.config, "engine_kwargs"):
        verify_parity = manager.config.engine_kwargs.get("raiden", {}).get(
            "verify_parity", manager.config.engine_kwargs.get("verify_parity", verify_parity)
        )

    # Lazily start the embedded RaidenControllerServer on the manager before trainer ranks register.
    if getattr(manager, "raiden_controller", None) is None:
        manager.raiden_controller, manager.raiden_server, manager.raiden_address = setup_raiden_controller()

    t_total_start = time.perf_counter()

    # 1. Trigger Trainer ranks to register their tensors with central RaidenController and record global shapes
    t_init_trainer_start = time.perf_counter()
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode="raiden")
    if actor_refs is not None:
        await asyncio.to_thread(ray.get, actor_refs)
    t_init_trainer = time.perf_counter() - t_init_trainer_start

    parallelism = 8
    if hasattr(manager, "config") and hasattr(manager.config, "engine_kwargs"):
        parallelism = manager.config.engine_kwargs.get("raiden", {}).get(
            "parallelism", manager.config.engine_kwargs.get("parallelism", 8)
        )

    # 2. Initialize and register Sampler rollout workers with central RaidenController
    t_init_sampler_start = time.perf_counter()
    sampler_init_futures = [
        replica.server_handle.collective_rpc.remote(
            method="init_raiden_sync_on_worker", kwargs={"parallelism": parallelism}
        )
        for replica in manager.replicas
    ]
    await asyncio.gather(*sampler_init_futures)
    t_init_sampler = time.perf_counter() - t_init_sampler_start

    # 3. Registration barrier on Central RaidenController
    t_barrier_start = time.perf_counter()
    num_rollout_workers = 0
    for r in manager.replicas:
        if hasattr(r, "world_size") and r.world_size:
            num_rollout_workers += r.world_size
        elif hasattr(r, "workers") and r.workers:
            num_rollout_workers += len(r.workers)
        else:
            num_rollout_workers += 1
    if num_rollout_workers == 0:
        num_rollout_workers = len(manager.replicas)
    sampler_replica_ids = [str(i) for i in range(num_rollout_workers)]
    trainer_replica_ids = [str(i) for i in range(manager.actor_wg.world_size)]

    from tpu_sync.api.common import RaidenId
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    src_units = [RaidenId(job_name="trainer", job_replica_id=r_id, data_name="weights") for r_id in trainer_replica_ids]
    dst_units = [RaidenId(job_name="sampler", job_replica_id=r_id, data_name="weights") for r_id in sampler_replica_ids]

    barrier_timeout = 60.0
    while True:
        with manager.raiden_controller._lock:
            registered = set(manager.raiden_controller._registered_shards.keys())
        src_registered = all(u in registered for u in src_units)
        dst_registered = all(u in registered for u in dst_units)
        if src_registered and dst_registered:
            logger.info(
                "[RAIDEN CONTROLLER] All %d Trainer and %d Sampler units verified and registered.",
                len(src_units),
                len(dst_units),
            )
            break
        if time.perf_counter() - t_barrier_start > barrier_timeout:
            missing_src = [u for u in src_units if u not in registered]
            missing_dst = [u for u in dst_units if u not in registered]
            raise RuntimeError(
                f"Timeout ({barrier_timeout}s) waiting for workers to register with RaidenController! "
                f"Missing Trainer: {missing_src}, Missing Sampler: {missing_dst}"
            )
        await asyncio.sleep(0.1)
    t_barrier = time.perf_counter() - t_barrier_start

    # 4. Trigger coordinated P2P network transfers via central RaidenController
    t_transfer_start = time.perf_counter()
    transfer_future = manager.raiden_controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        dst_mem_type=RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=0,
        parallelism=parallelism,
        req_id=f"verl_step_{global_steps or 0}",
    )
    await transfer_future.wait()
    t_transfer = time.perf_counter() - t_transfer_start

    # 5. Sampler replicas install received weights to TPU HBM via H2D DMA
    t_install_start = time.perf_counter()
    install_futures = [
        replica.server_handle.collective_rpc.remote(method="install_raiden_weights") for replica in manager.replicas
    ]
    await asyncio.gather(*install_futures)
    t_install = time.perf_counter() - t_install_start

    # 6. Drop prefix/KV cache computed with the old weights and propagate global_steps
    cache_and_step_futures = []
    for replica in manager.replicas:
        cache_and_step_futures.append(replica.server_handle.clear_kv_cache.remote())
        if global_steps is not None:
            cache_and_step_futures.append(replica.server_handle.set_global_steps.remote(global_steps))
    if cache_and_step_futures:
        await asyncio.gather(*cache_and_step_futures)

    t_total = time.perf_counter() - t_total_start

    # 7. Parity Verification (Optional, default=False)
    if verify_parity:
        try:
            await _verify_parity_async(manager, global_steps)
        except Exception as e:
            logger.warning("Failed to execute parity verification: %s", e)

    logger.info(
        "[RAIDEN TELEMETRY | Orchestrator] Step %s Completed in %.4fs:\n"
        "  * Sampler Quiesce/Pause  : %.4fs\n"
        "  * Trainer Raiden Init    : %.4fs\n"
        "  * Sampler Raiden Init    : %.4fs\n"
        "  * Raiden Barrier Check   : %.4fs\n"
        "  * RaidenController P2P   : %.4fs\n"
        "  * Sampler H2D DMA        : %.4fs\n"
        "  * Total End-to-End Sync  : %.4fs",
        global_steps,
        t_total,
        t_abort,
        t_init_trainer,
        t_init_sampler,
        t_barrier,
        t_transfer,
        t_install,
        t_total,
    )

    # 8. Resume generation immediately
    await manager.resume_generation_replicas()

    return {}


async def _verify_parity_async(manager: Any, global_steps: int | None = None) -> None:
    """Compares distributed norms between Trainer ranks and Sampler TP workers."""
    step_key = global_steps if global_steps is not None else 0
    if step_key <= 0:
        return

    try:
        registry = get_or_create_ray_weight_registry()
        if registry is None:
            return
        num_trainer_ranks = manager.actor_wg.world_size
        trainer_entry = None
        for _ in range(25):
            entry = await registry.get_stats.remote(step_key)
            if entry is not None and ("ranks" not in entry or len(entry["ranks"]) >= num_trainer_ranks):
                trainer_entry = entry
                break
            await asyncio.sleep(0.5)

        sampler_futures = [
            replica.server_handle.collective_rpc.remote(
                method="get_model_weights_stats", kwargs={"include_shards": False}
            )
            for replica in manager.replicas
        ]
        sampler_entries = await asyncio.gather(*sampler_futures)

        if not trainer_entry or not sampler_entries:
            logger.warning("[RAIDEN PARITY] Incomplete stats data for step %s", step_key)
            return

        sampler_workers = [
            w
            for res in sampler_entries
            for w in (res if isinstance(res, list | tuple) else [res])
            if isinstance(w, dict)
        ]
        if not sampler_workers:
            return

        if "ranks" in trainer_entry:
            trainer_master = merge_rank_stats(trainer_entry["ranks"])
        else:
            trainer_master = trainer_entry.get("master", trainer_entry)
        trainer_per_tensor = trainer_master.get("per_tensor", {})
        all_param_names = list(sampler_workers[0].get("per_tensor", {}).keys()) if sampler_workers else []

        total_trainer_numel = trainer_master.get("total_numel", 0)
        total_trainer_l1 = trainer_master.get("l1_norm", 0.0)
        global_trainer_l2 = trainer_master.get("l2_norm", 0.0)

        total_sampler_l1, total_sampler_l2_sq, total_sampler_numel = 0.0, 0.0, 0
        mismatches = []

        for name in all_param_names:
            s_numels = [w.get("per_tensor", {}).get(name, {}).get("numel", 0) for w in sampler_workers]
            s_l1s = [w.get("per_tensor", {}).get(name, {}).get("l1", 0.0) for w in sampler_workers]
            s_l2_sqs = [
                w.get("per_tensor", {})
                .get(name, {})
                .get("l2_sq", w.get("per_tensor", {}).get(name, {}).get("l2", 0.0) ** 2)
                for w in sampler_workers
            ]

            is_replicated = (
                len(set(s_numels)) == 1
                and (name.endswith("layernorm.weight") or "norm" in name)
                and len(s_numels[0:1]) > 0
                and s_numels[0] < 10000
            )
            if is_replicated:
                s_agg_numel = s_numels[0]
                s_agg_l1 = s_l1s[0]
                s_agg_l2 = s_l2_sqs[0] ** 0.5
            else:
                s_agg_numel = sum(s_numels)
                s_agg_l1 = sum(s_l1s)
                s_agg_l2 = sum(s_l2_sqs) ** 0.5

            t_data = trainer_per_tensor.get(name, {})
            t_agg_numel = t_data.get("numel", 0)
            t_agg_l1 = t_data.get("l1", 0.0)

            total_sampler_numel += s_agg_numel
            total_sampler_l1 += s_agg_l1
            total_sampler_l2_sq += s_agg_l2**2

            delta_numel = abs(s_agg_numel - t_agg_numel)
            delta_l1 = abs(s_agg_l1 - t_agg_l1)
            rel_tol = 1e-3 * max(abs(t_agg_l1), 1.0)

            if delta_numel != 0 or delta_l1 > rel_tol:
                mismatches.append(f"  * {name}: Trainer(L1={t_agg_l1:.4f}) vs Sampler(L1={s_agg_l1:.4f})")

        global_sampler_l2 = total_sampler_l2_sq**0.5
        total_l1_delta = abs(total_sampler_l1 - total_trainer_l1)
        total_l2_delta = abs(global_sampler_l2 - global_trainer_l2)
        total_numel_delta = abs(total_sampler_numel - total_trainer_numel)
        total_rel_tol = 1e-3 * max(abs(total_trainer_l1), 1.0)

        if not mismatches and total_numel_delta == 0 and total_l1_delta <= total_rel_tol:
            logger.info(
                "[RAIDEN PARITY VERIFIED | Step %s] 100%% DISTRIBUTED NORM PARITY CONFIRMED!\n"
                "  * Global L1 Norm: %.6f (Trainer=%.6f, delta=%.6f)\n"
                "  * Global L2 Norm: %.6f (Trainer=%.6f, delta=%.6f)\n"
                "  * Total Parameters: %d across %d tensors",
                step_key,
                total_sampler_l1,
                total_trainer_l1,
                total_l1_delta,
                global_sampler_l2,
                global_trainer_l2,
                total_l2_delta,
                total_sampler_numel,
                len(all_param_names),
            )
        else:
            mismatches_summary = "\n".join(mismatches[:10])
            logger.error(
                "[RAIDEN PARITY MISMATCH | Step %s] Norms do NOT match!\n"
                "  * Trainer: numel=%d, L1=%.6f, L2=%.6f\n"
                "  * Sampler: numel=%d, L1=%.6f, L2=%.6f\n"
                "  * Mismatched Tensors (%d / %d):\n%s",
                step_key,
                total_trainer_numel,
                total_trainer_l1,
                global_trainer_l2,
                total_sampler_numel,
                total_sampler_l1,
                global_sampler_l2,
                len(mismatches),
                len(all_param_names),
                mismatches_summary,
            )
    except Exception as e:
        logger.warning("Error during parity verification for step %s: %s", step_key, e)


RAIDEN_WORKER_EXTENSION_CLS = "verl_hardware_plugin.engines.raiden_checkpoint_engine.vLLMRaidenWorkerExtension"


def apply_raiden_checkpoint_engine_hooks() -> None:
    """Registers the ``raiden`` checkpoint engine hooks on manager, actor worker, and vLLM workers."""
    try:
        from verl_hardware_plugin.engines.tpu_checkpoint_engine import apply_tpu_checkpoint_engine_hooks

        apply_tpu_checkpoint_engine_hooks()
    except Exception as e:
        logger.debug("Failed to ensure base TPU checkpoint engine hooks: %s", e)

    try:
        for mod_path, cls_name in (
            ("vllm_torchtpu.worker.tpu_worker", "TPUWorker"),
            ("tpu_inference.worker.tpu_worker", "TPUWorker"),
            ("vllm.executor.ray_utils", "RayWorkerWrapper"),
        ):
            try:
                mod = __import__(mod_path, fromlist=[cls_name])
                worker_cls = getattr(mod, cls_name, None)
                if worker_cls is not None:
                    for method_name in (
                        "_get_vllm_model",
                        "init_raiden_sync_on_worker",
                        "install_raiden_weights",
                        "get_model_weights_stats",
                    ):
                        if not hasattr(worker_cls, method_name):
                            setattr(worker_cls, method_name, getattr(vLLMRaidenWorkerExtension, method_name))
            except ImportError as e:
                missing_pkg = getattr(e, "name", None) or mod_path
                logger.debug(
                    "Skipping Raiden worker hook for %s.%s: package %r caused ImportError (%s)",
                    mod_path,
                    cls_name,
                    missing_pkg,
                    e,
                )
                continue
    except Exception as e:
        logger.debug("Failed to attach Raiden worker methods to TPUWorker: %s", e)

    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        if not getattr(vLLMHttpServer._get_worker_extension_cls, "_verl_raiden_ext_patched", False):
            _orig_get_worker_ext = vLLMHttpServer._get_worker_extension_cls

            @functools.wraps(_orig_get_worker_ext)
            def _patched_get_worker_ext(self) -> str:
                cfg = getattr(self, "config", None)
                ckpt_cfg = (
                    cfg.get("checkpoint_engine") if isinstance(cfg, dict) else getattr(cfg, "checkpoint_engine", None)
                )
                backend = ckpt_cfg.get("backend") if isinstance(ckpt_cfg, dict) else getattr(ckpt_cfg, "backend", None)
                if backend == "raiden":
                    return RAIDEN_WORKER_EXTENSION_CLS
                return _orig_get_worker_ext(self)

            _patched_get_worker_ext._verl_raiden_ext_patched = True  # type: ignore[attr-defined]
            vLLMHttpServer._get_worker_extension_cls = _patched_get_worker_ext  # type: ignore[method-assign]
    except Exception as e:
        logger.debug("Failed to patch vLLMHttpServer._get_worker_extension_cls for Raiden: %s", e)

    try:
        import verl.checkpoint_engine.base as ckpt_base

        if not getattr(ckpt_base.CheckpointEngineManager.update_weights, "_verl_raiden_ckpt_patched", False):
            _prev_mgr_update = ckpt_base.CheckpointEngineManager.update_weights

            @ckpt_base.auto_await
            async def _raiden_patched_mgr_update(self, global_steps: int | None = None):
                if self.backend == "raiden":
                    return await update_raiden_weights(self, global_steps=global_steps)
                res = _prev_mgr_update(self, global_steps=global_steps)
                if inspect.isawaitable(res):
                    return await res
                return res

            _raiden_patched_mgr_update._verl_raiden_ckpt_patched = True  # type: ignore[attr-defined]
            ckpt_base.CheckpointEngineManager.update_weights = _raiden_patched_mgr_update
    except Exception as e:
        logger.debug("Failed to patch CheckpointEngineManager for Raiden: %s", e)

    try:
        from verl.workers.engine_workers import ActorRolloutRefWorker

        if not getattr(ActorRolloutRefWorker.update_weights, "_verl_consumes_engine_patched", False):
            _orig_actor_update_weights = ActorRolloutRefWorker.update_weights

            @functools.wraps(_orig_actor_update_weights)
            async def _patched_actor_update_weights(self, global_steps: int | None = None, mode: str = "auto"):
                effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend
                if effective_mode != "naive" and getattr(self.checkpoint_engine, "consumes_training_engine", False):
                    metrics = await self.checkpoint_engine.send_weights(self.actor.engine, global_steps=global_steps)
                    return metrics or {}
                return await _orig_actor_update_weights(self, global_steps=global_steps, mode=mode)

            _patched_actor_update_weights._verl_consumes_engine_patched = True  # type: ignore[attr-defined]
            ActorRolloutRefWorker.update_weights = _patched_actor_update_weights
    except Exception as e:
        logger.debug("Failed to patch ActorRolloutRefWorker.update_weights for consumes_training_engine: %s", e)


apply_raiden_checkpoint_engine_hooks()

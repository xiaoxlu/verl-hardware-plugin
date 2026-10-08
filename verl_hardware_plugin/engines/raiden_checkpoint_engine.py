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


def filter_tied_embeddings(
    named_items: Iterable[tuple[str, Any]], tie_word_embeddings: bool = True
) -> list[tuple[str, Any]]:
    """Drops ``lm_head.weight`` from the weights to send when it is tied to the input embedding.

    For tied models (``tie_word_embeddings=True``, e.g. Qwen3-0.6B / 4B), ``lm_head.weight`` is the same
    tensor as ``embed_tokens.weight``, so it is not sent; the sampler copies ``embed_tokens`` into its
    ``lm_head`` after receiving the weights.

    For untied models (``tie_word_embeddings=False``, e.g. Qwen3-8B and larger), ``lm_head.weight`` is a
    separate trained weight and is kept. Dropping it would leave the sampler with a wrong ``lm_head``
    (stale, or overwritten with ``embed_tokens``).

    Args:
        named_items: ``(name, tensor)`` pairs in HF naming.
        tie_word_embeddings: ``hf_config.tie_word_embeddings`` of the trained model.

    Returns:
        The ``(name, tensor)`` pairs to send.
    """
    items = list(named_items)
    has_embed = any("embed_tokens" in k or "tok_embeddings" in k for k, _ in items)
    if has_embed and tie_word_embeddings:
        items = [(k, v) for k, v in items if not (k == "lm_head.weight" or k.endswith(".lm_head.weight"))]
    return items


def _as_bool(value: Any) -> bool:
    """Parses a bool flag that may arrive as a string (env var, CLI override) as well as a bool."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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


def export_local_shards(
    engine: Any, tie_word_embeddings: bool = True
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, tuple[int, int]]]:
    """Exports this rank's weights for Raiden without all-gathering FSDP shards.

    Returns ``(named_tensors, shard_info)``. ``shard_info[name] = (num_shards, shard_index)`` for tensors
    exported as a local dim-0 shard; tensors absent from it are full (replicated) on every rank. Tensors
    whose layout Raiden cannot express are all-gathered here, in the same order on every rank.
    ``lm_head.weight`` is dropped only for tied models (see ``filter_tied_embeddings``).
    """
    from torch.distributed.tensor import DTensor

    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    gen, _ = engine.get_per_tensor_param_shard()
    items = filter_tied_embeddings(
        ((name, (local, spec)) for name, local, spec in gen), tie_word_embeddings=tie_word_embeddings
    )

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


class RaidenParityCheck:
    """Raiden weight-sync verification, selected by ``engine_kwargs.raiden.verify_parity``.

    Two complementary checks:

    * ``"norm"`` (or ``True``): on every sync from step 1, the trainer ranks post per-tensor L1/L2 stats and the
      orchestrator compares them with the samplers' stats after the install (``verify_norms``). Cheap, but blind to
      misplaced bytes: a reordered or mis-sliced tensor keeps its norms.
    * ``"exact"``: on the step-0 sync of a fresh run, when the trainer weights equal the checkpoint vLLM loaded,
      every sampler worker compares each received tensor bit for bit with the vLLM parameter it is about to
      overwrite (``check_exact``) and returns the result (``exact_result``); the orchestrator logs one summary for
      all ranks (``report_exact``). Catches any wrong byte (shard slice, tiling, fusion/transpose). Use it after
      changing sharding or upgrading tpu-sync / torch-tpu. Cost: ~0.4 s once per rank (Qwen3-32B, TP32).

    ``"all"`` enables both; ``False`` / ``"off"`` (default) neither.
    """

    MODES = ("off", "norm", "exact", "all")
    MAX_DETAIL_LINES = 60

    def __init__(self, mode: Any = "off") -> None:
        self.mode = self.parse_mode(mode)
        self.checked = 0
        self.mismatched: list[str] = []

    @classmethod
    def parse_mode(cls, value: Any) -> str:
        if value is None or value is False:
            return "off"
        if value is True:
            return "norm"
        mode = str(value).strip().lower()
        mode = {"false": "off", "0": "off", "none": "off", "true": "norm", "1": "norm"}.get(mode, mode)
        if mode not in cls.MODES:
            raise ValueError(f"verify_parity must be a bool or one of {cls.MODES}, got {value!r}")
        return mode

    @classmethod
    def from_config(cls, engine_kwargs: Any) -> "RaidenParityCheck":
        """Reads ``engine_kwargs.raiden.verify_parity``, the same key the trainer's RaidenCheckpointEngine receives."""
        if engine_kwargs is None:
            return cls("off")
        raiden_kwargs = engine_kwargs.get("raiden") or {}
        return cls(raiden_kwargs.get("verify_parity", engine_kwargs.get("verify_parity", False)))

    @property
    def norm(self) -> bool:
        return self.mode in ("norm", "all")

    @property
    def exact(self) -> bool:
        return self.mode in ("exact", "all")

    # ---- exact check: orchestrator side ----

    def exact_install_kwargs(self, global_steps: int | None) -> dict[str, Any]:
        """kwargs for the samplers' ``install_raiden_weights`` RPC: request the exact check on the step-0 sync only.

        Later syncs (and the first sync after resuming from a checkpoint) carry trained weights, which no longer
        match what vLLM loaded, so the exact check cannot run there.
        """
        return {"exact_parity": True} if self.exact and not global_steps else {}

    @classmethod
    def report_exact(cls, global_steps: int | None, install_results: list[Any]) -> bool:
        """Logs one summary of the per-rank ``exact_result`` dicts that every sampler worker returns under
        ``install_raiden_weights(...)["exact_parity"]``."""
        results = [
            w["exact_parity"]
            for res in install_results
            for w in (res if isinstance(res, list | tuple) else [res])
            if isinstance(w, dict) and isinstance(w.get("exact_parity"), dict)
        ]
        step = global_steps or 0
        if not results:
            logger.warning("[RAIDEN PARITY EXACT | Step %s] no sampler returned a result", step)
            return False
        results.sort(key=lambda r: r["rank"])
        num_mismatched = sum(r["num_mismatched"] for r in results)
        num_unresolved = sum(len(r["unresolved"]) for r in results)
        checked_per_rank = sorted({r["checked"] for r in results})
        summary = (
            f"[RAIDEN PARITY EXACT | Step {step}] {len(results)} sampler ranks, {checked_per_rank} tensors checked "
            f"per rank, {num_mismatched} mismatched, {num_unresolved} unresolved"
        )
        if num_mismatched == 0 and num_unresolved == 0:
            logger.info("%s: every received tensor matches the checkpoint bit for bit", summary)
            return True
        details = [f"  * rank {r['rank']}: {line}" for r in results for line in r["mismatched"]]
        details += [f"  * rank {r['rank']}: unresolved {r['unresolved'][:5]}" for r in results if r["unresolved"]]
        detail_text = "\n".join(details[: cls.MAX_DETAIL_LINES])
        logger.error("%s\n%s", summary, detail_text)
        return False

    # ---- exact check: sampler worker side ----

    @torch.no_grad()
    def check_exact(self, name: str, target: torch.Tensor, received: torch.Tensor) -> None:
        """Compares ``received`` with ``target``; call before ``target`` is overwritten."""
        try:
            ref, new = target.detach().float(), received.detach().float()
            if ref.shape != new.shape:
                self.mismatched.append(f"{name}: shape target={tuple(ref.shape)} received={tuple(new.shape)}")
                return
            self.checked += 1
            diff = (ref - new).abs().max().item()
            if diff != 0.0:
                self.mismatched.append(
                    f"{name}: max|diff|={diff:.4g} ref_absmean={ref.abs().mean().item():.4g} "
                    f"recv_absmean={new.abs().mean().item():.4g}"
                )
        except Exception as e:  # never break the sync because of the check
            self.mismatched.append(f"{name}: parity check failed: {e}")

    def exact_result(self, rank: int, unresolved: list[str]) -> dict[str, Any]:
        return {
            "rank": rank,
            "checked": self.checked,
            "num_mismatched": len(self.mismatched),
            "mismatched": self.mismatched[: self.MAX_DETAIL_LINES],
            "unresolved": list(unresolved),
        }

    # ---- norm check: orchestrator side ----

    async def verify_norms(self, manager: Any, global_steps: int | None = None) -> None:
        """Compares distributed norms between the trainer ranks and the sampler TP workers."""
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
                # Sharded export: every trainer rank posts partial stats, usable once all have arrived.
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
                s_replicas = [w.get("per_tensor", {}).get(name, {}).get("replicas") for w in sampler_workers]

                is_replicated = (
                    len(set(s_numels)) == 1
                    and (name.endswith("layernorm.weight") or "norm" in name)
                    and len(s_numels[0:1]) > 0
                    and s_numels[0] < 10000
                )
                if all(r for r in s_replicas):
                    # Each slice is held by `replicas` ranks (GQA k/v heads, unsharded norms): count it once.
                    s_agg_numel = round(sum(n / r for n, r in zip(s_numels, s_replicas, strict=True)))
                    s_agg_l1 = sum(v / r for v, r in zip(s_l1s, s_replicas, strict=True))
                    s_agg_l2 = sum(v / r for v, r in zip(s_l2_sqs, s_replicas, strict=True)) ** 0.5
                elif is_replicated:
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
        # Trainer ranks only post L1/L2 stats for the norm check; the exact check runs on the samplers.
        self.verify_parity = RaidenParityCheck(kwargs.get("verify_parity", False)).norm
        self.parallelism = kwargs.get("parallelism", 8)
        self.tp_size = kwargs.get("tp_size", kwargs.get("tensor_parallel_size", self.parallelism))
        self._trainer_raiden_ws: Any = None
        self._trainer_chunks: list[Any] = []
        self._controller_addr: str | None = None
        self._registered_signature: list[tuple[str, tuple[int, ...], torch.dtype, tuple[int, int] | None]] | None = None
        self._bound_tensors: list[torch.Tensor] | None = None
        self._shard_info: dict[str, tuple[int, int]] = {}
        # Free the device tensors bound for a transfer once the controller push has read them, instead of keeping
        # them until the next sync. With the local-shard export that is the bf16 copy of this rank's shards plus
        # the full q/k/v, which the fused-wqkv save hook still all-gathers to every rank: ~3.4 GiB per chip for
        # Qwen3-8B on 8 chips. With an all-gathered weights iterable it is the full bf16 model on every chip. The
        # synchronizer pins the device buffers of every bound tensor (dropping the Python references frees
        # nothing), so the release goes through WeightSynchronizer.unbind_weights(): the HBM holds are dropped
        # while the synchronizer, its pinned host staging buffers and its controller registration stay alive for
        # the next sync to rebind. On by default so the trainer keeps that headroom as models grow; opt out with
        # the engine kwarg or VERL_RAIDEN_RELEASE_BUFFERS=0 to keep the send tensors resident between syncs.
        release = _as_bool(kwargs.get("release_buffers_after_sync", True))
        env_release = os.environ.get("VERL_RAIDEN_RELEASE_BUFFERS", "")
        if env_release:
            release = _as_bool(env_release)
        self.release_buffers_after_sync = release
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
        tie_word_embeddings: bool = True,
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
            # Handed the training engine: register each rank's local FSDP shard, no all-gather.
            hf_config = getattr(getattr(weights, "model_config", None), "hf_config", None)
            tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", tie_word_embeddings))
            named_weights, self._shard_info = export_local_shards(weights, tie_word_embeddings=tie_word_embeddings)
        else:
            weight_items = weights.items() if hasattr(weights, "items") else weights
            named_weights = filter_tied_embeddings(weight_items, tie_word_embeddings=tie_word_embeddings)
            self._shard_info = {}

        unfused_weights = {k: _unwrap_tensor(v) for k, v in named_weights}
        sorted_weights = sorted(unfused_weights.items(), key=lambda x: x[0])
        valid_weights = validate_and_sanitize_tensors(sorted_weights)

        torch_tpu_sync.synchronize(wait=True)

        # Reuse the WeightSynchronizer across steps when the (name, shape, dtype, shard) signature is unchanged:
        # rebinding keeps the DMA-mapped host buffers, listener/threads and controller registration, and only
        # swaps the device buffers the D2H reads from (release_sync_buffers() unbinds them between syncs).
        signature = [(name, tuple(t.shape), t.dtype, self._shard_info.get(name)) for name, t in valid_weights]
        if self._trainer_raiden_ws is not None and signature == self._registered_signature:
            self._trainer_raiden_ws.bind_weights([[t] for _, t in valid_weights])
        else:
            await self._create_and_register(valid_weights)
            self._registered_signature = signature
        # Keep the bound device buffers alive until the next sync (or release_sync_buffers()): the
        # controller-driven push reads them after send_weights returns.
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

    def release_sync_buffers(self) -> dict[str, Any]:
        """Frees this rank's bound send tensors after the P2P transfer, keeping the synchronizer when possible.

        Must only be called after the controller transfer that reads them has completed. Runs by default; a
        no-op when ``release_buffers_after_sync`` is disabled, in which case the bound tensors stay resident
        between syncs.

        With a ``tpu_sync`` build that exposes ``WeightSynchronizer.unbind_weights()`` the synchronizer's holds on
        the bound device buffers are dropped, so dropping our own references returns the HBM. The synchronizer
        itself, its pinned host staging buffers and its controller registration survive, and the next
        ``send_weights`` rebinds through ``bind_weights()``. Compared to destroying and re-creating the
        synchronizer each sync, this removes the per-step rebuild (host buffer allocation, pinning and first
        touch, plus controller re-registration: ~2 s "Trainer init" + ~2 s release on Qwen3-8B, v6e-8) from
        the sync critical path.

        Older ``tpu_sync`` builds without ``unbind_weights`` fall back to destroying the synchronizer; the next
        ``send_weights`` then re-creates and re-registers it.
        """
        if not self.release_buffers_after_sync or self._trainer_raiden_ws is None:
            return {}
        ws = self._trainer_raiden_ws
        if hasattr(ws, "unbind_weights"):
            ws.unbind_weights()
            self._bound_tensors = None
        else:
            # The torch WeightSynchronizer has no close(); the C++ object (which holds references to the
            # device tensors and the host staging memory) is destroyed when the last Python reference goes.
            if not getattr(self, "_warned_no_unbind", False):
                logger.warning(
                    "Trainer Rank %s: tpu_sync WeightSynchronizer has no unbind_weights(); destroying it to free "
                    "the send buffers each sync (upgrade tpu-sync-torch to keep the synchronizer across syncs).",
                    self.rank,
                )
                self._warned_no_unbind = True
            self._trainer_raiden_ws = None
            self._bound_tensors = None
            self._registered_signature = None
            del ws
            import gc

            gc.collect()
        # No empty_cache(): on TPU it clears the eager-op compilation cache (forcing recompiles) and the
        # TPU runtime reuses freed HBM without it.
        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception:
            pass
        return {}

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
    _raiden_replicas: dict[str, int]

    def _get_vllm_model(self) -> Any:
        """Extracts the underlying ``nn.Module`` from the vLLM worker."""
        worker: Any = getattr(self, "worker", self)
        return worker.model_runner.model

    def init_raiden_sync_on_worker(self, parallelism: int = 8, job_name: str = "sampler") -> bool:
        """Initializes Raiden ``WeightSynchronizer`` listener and registers with ``RaidenController``.

        ``job_name`` is the Raiden job this replica registers under; the orchestrator gives every rollout replica
        its own name when there are several, so their ranks ``0..TP-1`` do not collide on the controller.
        """
        if hasattr(self, "_raiden_ws") and self._raiden_ws is not None:
            return True

        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            logger.warning("Raiden Sampler: could not locate vllm_model to bind parameters.")
            return False

        from tpu_sync.rpc import raiden_service_pb2

        bind_ip = ray.util.get_node_ip_address().strip("[]")
        rank_val = getattr(self, "rank", 0)
        # Let the OS pick the control listener port, as the trainer side does. ``rank`` is the rank inside this
        # replica, so a fixed base + rank collides as soon as two replicas share a host. The port actually bound
        # is read back from the synchronizer and registered with the controller below.
        listener_port = 0

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

        # TODO(tpu): Copy the received weights straight into vLLM's parameters, with no staging copy. The
        # staging tensors are a second full copy of this rank's TP shard (~1.9 GiB per chip for Qwen3-8B at
        # TP=8) that must fit next to vLLM's preallocated weights and KV cache on every sync.
        staging_tensors: dict[str, torch.Tensor] = {}
        variable_protos: list[Any] = []
        valid_params: list[tuple[str, torch.Tensor]] = []
        replicas: dict[str, int] = {}

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
            # Number of TP ranks holding this same slice: 1 for a plain split, tp_size for unsharded tensors
            # (norms, 1-D biases). The norm parity check divides by it when summing over ranks.
            mesh_size = 1
            for m in sharding_mesh:
                mesh_size *= m
            replicas[name] = tp_size // mesh_size

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
        self._raiden_replicas = replicas

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

        # Whether a tensor can skip the CPU (de)tiling pass is decided by Raiden's planner per transfer, from the
        # SOURCE and destination slices together, and delivered to this listener with the transfer. Setting it here
        # from the local shape alone makes the two sides disagree whenever a trainer shard is not 8-row aligned
        # (e.g. Qwen3's embedding at 32+ FSDP ranks): the sender de-tiles to row-major and h2d() copies those bytes
        # raw into tiled HBM, permuting the weights while leaving every norm unchanged.

        try:
            from tpu_sync.rpc import raiden_controller

            ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_addr)
            unit_id = raiden_controller.RaidenId(job_name, str(rank_val), "weights")
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
    def install_raiden_weights(self, exact_parity: bool = False) -> dict[str, Any]:
        """Installs received weights from host staging buffers into TPU HBM via zero-copy H2D DMA
        and fuses/transposes them into vLLM model parameters.

        Args:
            exact_parity: Set by ``RaidenParityCheck`` on the step-0 sync: compare every received tensor bit for
                bit with the vLLM parameter it overwrites.

        Returns:
            Per-worker timings in seconds: ``total`` (whole install), ``h2d`` (pure ``_raiden_ws.h2d()``)
            and ``sync`` (final TPU sync barrier), plus ``exact_parity`` (this rank's
            ``RaidenParityCheck.exact_result``) when requested. Empty if the synchronizer is not initialized.
        """
        if not hasattr(self, "_raiden_ws") or self._raiden_ws is None:
            logger.warning("Raiden Sampler: install_raiden_weights called before _raiden_ws was initialized.")
            return {}

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
        parity: RaidenParityCheck | None = RaidenParityCheck("exact") if exact_parity else None

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
                            if parity is not None:
                                parity.check_exact(resolved_key, target_local, fused_adapted)
                            target_local.copy_(fused_adapted)
                            consumed_staging.update(layer_src_keys)

        # 2. Handle remaining non-fused parameters (o_proj, down_proj, layernorms, embeddings)
        unresolved: list[str] = []
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
                if parity is not None:
                    parity.check_exact(resolved_key, target_local, adapted)
                target_local.copy_(adapted)
            else:
                unresolved.append(name)

        # 3. Handle tied word embeddings. Only when the trainer did not send lm_head itself: for untied
        #    models (e.g. Qwen3-8B / 32B) the received lm_head must not be overwritten by embed_tokens.
        received_lm_head = any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in self._raiden_staging)
        if received_lm_head:
            pass
        elif (
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
        # Returned through collective_rpc so the orchestrator can log them as step metrics.
        result: dict[str, Any] = {"total": t_total, "h2d": t_h2d, "sync": t_sync}
        if parity is not None:
            result["exact_parity"] = parity.exact_result(getattr(self, "rank", 0), unresolved)
        return result

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
        replicas = getattr(self, "_raiden_replicas", {})
        for name, stats in res["per_tensor"].items():
            if name in replicas:
                stats["replicas"] = replicas[name]
        return res


def _replica_num_workers(replica: Any) -> int:
    """Number of Raiden work units a rollout replica registers: one per TP worker."""
    if getattr(replica, "world_size", None):
        return int(replica.world_size)
    if getattr(replica, "workers", None):
        return len(replica.workers)
    return 1


def _sampler_job_name(replica_idx: int, num_replicas: int) -> str:
    """Raiden job name of a rollout replica: ``sampler`` with one replica, ``sampler<idx>`` with several.

    A rollout worker registers as (job name, rank within its replica). With several replicas the ranks
    ``0..TP-1`` would all collide under one name, and the controller would wait for ranks ``TP..N*TP-1`` that
    nothing ever registers. One job name per replica keeps every registration unique; a single replica keeps
    the historical name.
    """
    return "sampler" if num_replicas == 1 else f"sampler{replica_idx}"


async def update_raiden_weights(
    manager: Any,
    global_steps: int | None = None,
) -> dict[str, Any]:
    """Orchestrates Raiden TPU P2P weight synchronization via the embedded ``RaidenController``.

    Returns the per-phase timings as ``timing_s/tpu-sync/*`` metrics.
    """
    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning("Failed to abort replicas at step %s: %s", global_steps, e)
    t_abort = time.perf_counter() - t_abort_start

    engine_kwargs = getattr(getattr(manager, "config", None), "engine_kwargs", None)
    parity = RaidenParityCheck.from_config(engine_kwargs)

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
    if engine_kwargs is not None:
        parallelism = (engine_kwargs.get("raiden") or {}).get("parallelism", engine_kwargs.get("parallelism", 8))

    # 2. Initialize and register Sampler rollout workers with central RaidenController
    num_replicas = len(manager.replicas)
    t_init_sampler_start = time.perf_counter()
    sampler_init_futures = [
        replica.server_handle.collective_rpc.remote(
            method="init_raiden_sync_on_worker",
            kwargs={"parallelism": parallelism, "job_name": _sampler_job_name(replica_idx, num_replicas)},
        )
        for replica_idx, replica in enumerate(manager.replicas)
    ]
    await asyncio.gather(*sampler_init_futures)
    t_init_sampler = time.perf_counter() - t_init_sampler_start

    # 3. Registration barrier on Central RaidenController
    t_barrier_start = time.perf_counter()

    # One group of destination units per rollout replica: ranks '0'..'TP-1' under the replica's own job name.
    # Examples:
    #   - 1 replica with TP=8: one group ['0'..'7'] under job "sampler".
    #   - 3 replicas with TP=8: groups ['0'..'7'] under "sampler0", "sampler1", "sampler2".
    trainer_replica_ids = [str(i) for i in range(manager.actor_wg.world_size)]

    from tpu_sync.api.common import RaidenId
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    src_units = [RaidenId(job_name="trainer", job_replica_id=r_id, data_name="weights") for r_id in trainer_replica_ids]
    dst_unit_groups = [
        [
            RaidenId(job_name=_sampler_job_name(replica_idx, num_replicas), job_replica_id=str(k), data_name="weights")
            for k in range(_replica_num_workers(replica))
        ]
        for replica_idx, replica in enumerate(manager.replicas)
    ]
    dst_units = [unit for group in dst_unit_groups for unit in group]

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
    # One transfer per replica. Each has the destination mesh of the validated single-replica case ([1, TP]),
    # instead of one transfer whose destination mixes the ranks of several meshes. The transfers are issued
    # together and awaited together: the controller tracks each by its own req_id / uuid and the trainer ranks
    # keep their D2H and skip-tiling state per uuid, so the replicas overlap instead of paying the per-transfer
    # latency one after another. The trainer buffers are only released after every transfer has finished.
    transfer_futures = []
    for replica_idx, dst_group in enumerate(dst_unit_groups):
        req_id = f"verl_step_{global_steps or 0}" + (f"_replica{replica_idx}" if num_replicas > 1 else "")
        transfer_futures.append(
            manager.raiden_controller.start_transfer(
                src_units=src_units,
                dst_units=dst_group,
                dst_mem_type=RaidenMemoryType.DRAM,
                use_block_chunks=True,
                is_sender=True,
                expected_block_count=0,
                parallelism=parallelism,
                req_id=req_id,
            )
        )
    await asyncio.gather(*[future.wait() for future in transfer_futures])
    t_transfer = time.perf_counter() - t_transfer_start

    # The trainer buffers are no longer read once the transfer is done; free them (unless disabled on the
    # engine) while the samplers install, so the HBM is back before the next training step. Dispatched through
    # the generic execute_checkpoint_engine RPC (DP_COMPUTE: one method name per rank).
    release_refs = None
    if hasattr(manager.actor_wg, "execute_checkpoint_engine"):
        release_refs = manager.actor_wg.execute_checkpoint_engine(
            ["release_sync_buffers"] * manager.actor_wg.world_size
        )

    # 5. Sampler replicas install received weights to TPU HBM via H2D DMA
    t_install_start = time.perf_counter()
    install_kwargs = parity.exact_install_kwargs(global_steps)
    install_futures = [
        replica.server_handle.collective_rpc.remote(method="install_raiden_weights", kwargs=install_kwargs)
        for replica in manager.replicas
    ]
    install_results = await asyncio.gather(*install_futures)
    t_install = time.perf_counter() - t_install_start
    # collective_rpc returns one result per TP worker of each replica; each is the timing dict returned by
    # vLLMRaidenWorkerExtension.install_raiden_weights. The slowest worker gates the sync, so report the max.
    worker_install_stats = [
        r for per_replica in install_results for r in (per_replica or []) if isinstance(r, dict) and "h2d" in r
    ]
    t_h2d_pure = max((r["h2d"] for r in worker_install_stats), default=None)

    # 6. Drop prefix/KV cache computed with the old weights and tag new generations with this weight version
    cache_and_step_futures = []
    for replica in manager.replicas:
        cache_and_step_futures.append(replica.server_handle.clear_kv_cache.remote())
        if global_steps is not None:
            cache_and_step_futures.append(replica.server_handle.set_global_steps.remote(global_steps))
    if cache_and_step_futures:
        await asyncio.gather(*cache_and_step_futures)
    if release_refs is not None:
        await asyncio.to_thread(ray.get, release_refs)

    t_total = time.perf_counter() - t_total_start

    # 7. Parity Verification (Optional, default=off; see RaidenParityCheck)
    if install_kwargs:
        parity.report_exact(global_steps, list(install_results))
    if parity.norm:
        try:
            await parity.verify_norms(manager, global_steps)
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

    # Surface the phase timers as step metrics (timing_s/update_weights ~= quiesce + total_sync).
    metrics: dict[str, Any] = {
        "timing_s/tpu-sync/quiesce": t_abort,
        "timing_s/tpu-sync/trainer_init": t_init_trainer,
        "timing_s/tpu-sync/sampler_init": t_init_sampler,
        "timing_s/tpu-sync/barrier": t_barrier,
        "timing_s/tpu-sync/p2p_transfer": t_transfer,
        "timing_s/tpu-sync/sampler_h2d": t_install,
        "timing_s/tpu-sync/total_sync": t_total,
    }
    if t_h2d_pure is not None:
        # Pure _raiden_ws.h2d() time on the slowest sampler worker (sampler_h2d above also includes the
        # fuse/transpose into vLLM params, the TPU sync barrier and the RPC round trip).
        metrics["timing_s/tpu-sync/sampler_h2d_pure"] = t_h2d_pure
    return metrics


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

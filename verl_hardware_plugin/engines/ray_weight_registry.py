# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Detached Ray actor for coordinating TPU and Raiden weight synchronization."""

from typing import Any

import ray

RAY_WEIGHT_REGISTRY_ACTOR_NAME = "RayWeightRegistry"
RAY_WEIGHT_REGISTRY_NAMESPACE = "verl"


class RayWeightRegistryState:
    """In-memory state backing ``RayWeightRegistry``.

    Stores:
    * Per-step ``ObjectRef`` lists for the Ray Plasma ``tpu`` checkpoint engine, using
      write-based eviction so stale steps from an earlier job cannot leak memory.
    * Raiden metadata (controller ``ip:port`` address, global parameter shapes, and
      bounded per-step/per-rank parity stats) for the ``raiden`` P2P checkpoint engine.
    """

    def __init__(self) -> None:
        self.weights: dict[int, list[Any]] = {}
        self.controller_address: str | None = None
        self.global_shapes: dict[str, list[int]] = {}
        self.stats: dict[int, dict[str, Any]] = {}

    def set_weights(self, step: int, ref_list: list[Any]) -> bool:
        for old_step in [k for k in self.weights if k != step]:
            del self.weights[old_step]
        self.weights[step] = ref_list
        return True

    def get_weights(self, step: int) -> list[Any] | None:
        return self.weights.get(step, None)

    def set_controller_address(self, addr: str) -> bool:
        self.controller_address = addr
        return True

    def get_controller_address(self) -> str | None:
        return self.controller_address

    def set_global_shapes(self, shapes: dict[str, list[int]]) -> bool:
        self.global_shapes = shapes
        return True

    def get_global_shapes(self) -> dict[str, list[int]]:
        return self.global_shapes

    def set_stats(self, step: int, stats: dict[str, Any]) -> None:
        self.stats[step] = {"master": stats} if "master" not in stats else stats
        self._prune_stats()

    def set_rank_stats(self, step: int, rank: int, stats: dict[str, Any]) -> None:
        """Stores one trainer rank's partial stats; merged once every rank has posted."""
        self.stats.setdefault(step, {}).setdefault("ranks", {})[rank] = stats
        self._prune_stats()

    def _prune_stats(self) -> None:
        steps_to_keep = sorted(self.stats.keys())
        if len(steps_to_keep) > 5:
            for old_step in steps_to_keep[:-5]:
                del self.stats[old_step]

    def get_stats(self, step: int) -> dict[str, Any] | None:
        return self.stats.get(step, None)

    def clear(self) -> bool:
        """Clears cached step weights, shapes, and stats while preserving ``controller_address``."""
        self.weights.clear()
        self.global_shapes.clear()
        self.stats.clear()
        return True


# Functional form (not ``@ray.remote``) so type checkers see an actor class with ``.options``;
# ``num_cpus=0`` keeps this bookkeeping actor from reserving a CPU slot.
RayWeightRegistry = ray.remote(num_cpus=0)(RayWeightRegistryState)


def get_or_create_ray_weight_registry() -> Any:
    """Returns the detached ``RayWeightRegistry`` actor handle in namespace ``verl``, creating it if needed."""
    if not ray.is_initialized():
        return None
    try:
        return ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
    except ValueError:
        try:
            return RayWeightRegistry.options(
                name=RAY_WEIGHT_REGISTRY_ACTOR_NAME,
                namespace=RAY_WEIGHT_REGISTRY_NAMESPACE,
                lifetime="detached",
                get_if_exists=True,
            ).remote()
        except Exception:
            return ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)

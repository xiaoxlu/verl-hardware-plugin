# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Lightweight CPU-only Ray actor registry for tracking model weight checkpoint ObjectRefs.

On TPU, trainer and rollout slices hold exclusive ``libtpu`` locks and cannot share a collective
communicator, so trainer rank 0 stores weights in Ray Plasma and registers the ``ObjectRef`` here
for rollout ``TPUWorker`` processes to fetch by training step.

The ``raiden`` checkpoint engine uses the same actor as a rendezvous point instead: the weight-sync
orchestrator publishes the address of its Raiden controller, the trainer publishes the full shape of
every tensor it sends (the rollout workers size their receive buffers from it), and both sides post
weight statistics for the optional parity check.
"""

import time
from typing import Any

import ray

RAY_WEIGHT_REGISTRY_ACTOR_NAME = "RayWeightRegistry"
RAY_WEIGHT_REGISTRY_NAMESPACE = "verl"


class RayWeightRegistryState:
    """In-memory state container holding Ray ObjectRefs to synchronized model weights across steps."""

    # Weight statistics are only read by the parity check of the step that wrote them.
    MAX_STATS_STEPS = 5

    def __init__(self) -> None:
        self.weights: dict[int, Any] = {}
        self.controller_address: str | None = None
        self.global_shapes: dict[str, list[int]] = {}
        self.stats: dict[int, dict[str, Any]] = {}

    def set_weights(self, step: int, ref: Any) -> None:
        """Stores the weight reference for ``step`` and evicts any prior step entries."""
        self.weights = {step: ref}

    def get_weights(self, step: int) -> Any | None:
        """Returns the weight reference registered for ``step``, or ``None`` if absent."""
        return self.weights.get(step)

    def set_controller_address(self, address: str | None) -> None:
        """Records the ``host:port`` of the Raiden controller started by the weight-sync orchestrator."""
        self.controller_address = address

    def get_controller_address(self) -> str | None:
        """Returns the Raiden controller address, or ``None`` before the orchestrator published it."""
        return self.controller_address

    def set_global_shapes(self, shapes: dict[str, list[int]]) -> None:
        """Records the full (unsharded) shape of every tensor the trainer sends, keyed by HF name."""
        self.global_shapes = dict(shapes)

    def get_global_shapes(self) -> dict[str, list[int]]:
        """Returns the shapes recorded by ``set_global_shapes`` (empty before the first sync)."""
        return self.global_shapes

    def set_stats(self, step: int, stats: dict[str, Any]) -> None:
        """Records trainer rank 0's weight statistics for ``step`` (full weights on every rank)."""
        self.stats[step] = {"master": stats}
        self._prune_stats()

    def set_rank_stats(self, step: int, rank: int, stats: dict[str, Any]) -> None:
        """Records one trainer rank's statistics of the shards it sent for ``step``.

        The driver merges the ``"ranks"`` entries (``merge_rank_stats``) once every rank has posted.
        """
        self.stats.setdefault(step, {}).setdefault("ranks", {})[rank] = stats
        self._prune_stats()

    def _prune_stats(self) -> None:
        for old_step in sorted(self.stats)[: -self.MAX_STATS_STEPS]:
            del self.stats[old_step]

    def get_stats(self, step: int) -> dict[str, Any] | None:
        """Returns ``{"master": stats}`` or ``{"ranks": {rank: stats}}`` for ``step``, or ``None``."""
        return self.stats.get(step)

    def clear(self) -> None:
        """Drops all cached entries to reset state left by a previous job."""
        self.weights.clear()
        self.controller_address = None
        self.global_shapes.clear()
        self.stats.clear()


RayWeightRegistry = ray.remote(num_cpus=0)(RayWeightRegistryState)

# How long reset_ray_weight_registry waits for a killed actor's name to become free again.
_RESET_TIMEOUT_S = 30.0
_RESET_POLL_S = 0.2


def _create_ray_weight_registry() -> Any:
    # A plain scheduling strategy: created from inside a placement-group task (a vLLM worker) the actor would
    # otherwise be captured by that group and die with it.
    return RayWeightRegistry.options(
        name=RAY_WEIGHT_REGISTRY_ACTOR_NAME,
        namespace=RAY_WEIGHT_REGISTRY_NAMESPACE,
        lifetime="detached",
        scheduling_strategy="DEFAULT",
    ).remote()


def get_ray_weight_registry() -> Any:
    """Returns the detached ``RayWeightRegistry`` actor handle, creating the actor on first use."""
    try:
        return ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
    except ValueError:
        pass
    try:
        return _create_ray_weight_registry()
    except Exception:
        # Another process created the actor between the lookup and the creation.
        return ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)


def reset_ray_weight_registry() -> Any:
    """Replaces any ``RayWeightRegistry`` left behind by a previous job with a fresh actor and returns its handle.

    The actor is detached, so it outlives the job that created it and keeps running that job's code. A later job
    with a newer plugin then calls methods the old actor does not have, which kills the actor mid-flight and with
    it every rendezvous that goes through it. Call this once per job from the driver, before anything else looks
    the actor up.
    """
    try:
        stale = ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
    except ValueError:
        stale = None
    if stale is not None:
        ray.kill(stale, no_restart=True)
    deadline = time.monotonic() + _RESET_TIMEOUT_S
    while True:
        try:
            return _create_ray_weight_registry()
        except ValueError:
            # The killed actor's name is released asynchronously.
            if time.monotonic() >= deadline:
                raise
            time.sleep(_RESET_POLL_S)

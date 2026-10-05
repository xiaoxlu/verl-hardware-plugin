# Raiden P2P Weight Synchronization on TPU (`backend=raiden`)

`verl-hardware-plugin` provides a peer-to-peer (P2P) weight synchronization checkpoint engine (`RaidenCheckpointEngine`, registered as `raiden`) powered by `tpu_sync` (`RaidenController` + `WeightSynchronizer`) for multi-host TPU RL workloads.

Compared to the Ray Plasma host checkpoint engine (`backend=tpu`), `backend=raiden` streams sharded parameters directly between trainer TPU ranks and vLLM rollout TPU workers over gRPC/RDMA with automatic resharding (FSDP dim-0 shards $\rightarrow$ vLLM TP shards).

## Architecture Overview

1. **Central Controller (`setup_raiden_controller`)**:
   - Lazily starts an embedded `RaidenControllerServer` on the orchestrator head node (ephemeral port `0`) during the first `CheckpointEngineManager.update_weights()` call.
   - Publishes the controller's `ip:port` address and global parameter shapes to the detached `RayWeightRegistry` actor.
2. **Trainer Side (`RaidenCheckpointEngine` + `export_local_shards`)**:
   - Sets `consumes_training_engine = True` so `ActorRolloutRefWorker.update_weights()` passes `self.actor.engine` directly to `send_weights()`.
   - Exports each rank's local 1D dim-0 FSDP `DTensor` shard directly to `WeightSynchronizer` (`mesh_shape=[N, 1]`, `global_shard_indices=[rank]`) without an expensive cross-host all-gather.
   - Reuses the existing `WeightSynchronizer` via `bind_weights()` across training steps when the tensor signature `(name, shape, dtype, shard)` is unchanged, avoiding per-step host DMA buffer reallocation.
3. **Sampler / Rollout Side (`vLLMRaidenWorkerExtension`)**:
   - Allocates unfused TPU staging tensors matching the vLLM worker's tensor-parallel shard layout (`tp_size`) and binds a `WeightSynchronizer` listener on port `39200 + rank` (outside KubeRay's `10002..19999` worker port range).
   - After `raiden_controller.start_transfer()` completes, executes `h2d()` DMA into the TPU staging buffers, fuses `q_proj`/`k_proj`/`v_proj` into `qkv_proj` and `gate_proj`/`up_proj` into `gate_up_proj`, applies `_tpu_weight_flipped` layout transpositions, and updates tied `lm_head` weights.

## Configuration

Enable the Raiden P2P checkpoint engine in your training script or Hydra overrides:

```bash
actor_rollout_ref.rollout.checkpoint_engine.backend=raiden
```

Optional engine kwargs under `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `parallelism` | `8` | Number of parallel transfer threads per `WeightSynchronizer` instance |
| `verify_parity` | `False` | When `True` (or `RAIDEN_VERIFY_PARITY=1`), computes distributed L1/L2 norms across trainer and rollout shards after each transfer to verify bit-exact parity |

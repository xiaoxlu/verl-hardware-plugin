# Raiden P2P Weight Synchronization on TPU (`backend=raiden`)

`verl-hardware-plugin` provides a peer-to-peer (P2P) weight synchronization checkpoint engine (`RaidenCheckpointEngine`, registered as `raiden`) powered by `tpu_sync` (`RaidenController` + `WeightSynchronizer`) for multi-host TPU RL workloads.

Compared to the Ray Plasma host checkpoint engine (`backend=tpu`), `backend=raiden` streams sharded parameters directly between trainer TPU ranks and vLLM rollout TPU workers over gRPC/RDMA with automatic resharding (FSDP dim-0 shards $\rightarrow$ vLLM TP shards).

## Architecture Overview

1. **Central Controller (`setup_raiden_controller`)**:
   - Lazily starts an embedded `RaidenControllerServer` on the orchestrator head node (ephemeral port `0`) during the first `CheckpointEngineManager.update_weights()` call.
   - Publishes the controller's `ip:port` address and global parameter shapes to the detached `RayWeightRegistry` actor.
2. **Trainer Side (`RaidenCheckpointEngine` + `export_local_shards`)**:
   - Sets `consumes_training_engine = True` so `ActorRolloutRefWorker.update_weights()` passes `self.actor.engine` directly to `send_weights()`.
   - Exports each rank's local 1D dim-0 FSDP `DTensor` shard directly to `WeightSynchronizer` (`mesh_shape=[N, 1]`, `global_shard_indices=[rank]`) without an expensive cross-host all-gather. `lm_head.weight` is only dropped for tied models (`hf_config.tie_word_embeddings=True`); untied models such as Qwen3-8B+ send their trained `lm_head`.
   - Reuses the existing `WeightSynchronizer` via `bind_weights()` across training steps when the tensor signature `(name, shape, dtype, shard)` is unchanged, avoiding per-step host DMA buffer reallocation.
   - After every transfer, `release_sync_buffers()` unbinds the bf16 send tensors (`WeightSynchronizer.unbind_weights()`) so their HBM is returned before the next training step while the synchronizer, its pinned host buffers and its controller registration survive. The orchestrator dispatches it through the generic `actor_wg.execute_checkpoint_engine(["release_sync_buffers"] * world_size)` RPC and overlaps it with the sampler H2D.
3. **Sampler / Rollout Side (`vLLMRaidenWorkerExtension`)**:
   - Allocates unfused TPU staging tensors matching the vLLM worker's tensor-parallel shard layout (`tp_size`) and binds a `WeightSynchronizer` listener on an OS-assigned port, so several replicas can share a host.
   - After `raiden_controller.start_transfer()` completes, executes `h2d()` DMA into the TPU staging buffers, fuses `q_proj`/`k_proj`/`v_proj` into `qkv_proj` and `gate_proj`/`up_proj` into `gate_up_proj`, applies `_tpu_weight_flipped` layout transpositions, and (for tied models only) copies `embed_tokens` into `lm_head`.
   - Several rollout replicas (`rollout.n_replicas > 1` / standalone mode) each register under their own Raiden job name (`sampler0`, `sampler1`, ...) and receive one transfer each, issued and awaited together.

## Configuration

Enable the Raiden P2P checkpoint engine in your training script or Hydra overrides:

```bash
actor_rollout_ref.rollout.checkpoint_engine.backend=raiden
```

Optional engine kwargs under `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `parallelism` | `8` | Number of parallel transfer threads per `WeightSynchronizer` instance |
| `release_buffers_after_sync` | `True` | Free the trainer's bound send tensors after every transfer (`unbind_weights`). Set to `False`, or export `VERL_RAIDEN_RELEASE_BUFFERS=0`, to keep them resident between syncs (trades HBM for skipping the per-step rebind). |
| `verify_parity` | `off` | `norm` (or `True`): compare trainer vs. sampler per-tensor L1/L2 norms after every sync from step 1. `exact`: on the step-0 sync of a fresh run, every sampler worker compares each received tensor bit for bit with the vLLM parameter it overwrites (catches shard-slice, tiling and fusion/transpose errors that keep the norms unchanged). `all`: both. |

## Requirements

- `tpu-sync-torch` with `WeightSynchronizer.unbind_weights()` (builds from `0.0.1.dev20261006` on). Older builds still work: the plugin logs a warning and falls back to destroying and re-creating the synchronizer on every sync, which adds roughly two seconds per step on Qwen3-8B / v6e-8.

## Metrics

`update_raiden_weights()` returns the per-phase timings of each sync (`timing_s/tpu-sync/quiesce`, `trainer_init`, `sampler_init`, `barrier`, `p2p_transfer`, `sampler_h2d`, `sampler_h2d_pure`, `total_sync`) and logs them as `[RAIDEN TELEMETRY | Orchestrator]`. Trainers that merge the dict returned by `CheckpointEngineManager.update_weights()` into their step metrics get them next to `timing_s/update_weights`.

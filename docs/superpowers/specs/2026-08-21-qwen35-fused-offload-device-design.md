# Qwen3.5 Fused Offload Device Compatibility

## Problem

veRL's Qwen3.5 fused PPO forward reconstructs an FSDP2 `DTensor` language-model-head weight with `full_tensor()` but does not move that tensor from CPU-offload storage to the CUDA device holding the hidden states. With parameter offload enabled, fused actor/reference log-probability computation fails before the memory hypothesis can be tested.

## Design

Add a repository-owned runtime compatibility patch, installed in every Ray worker before veRL builds the model. The patch rewrites the two Qwen3.5 fused forward functions so a reconstructed `DTensor` uses `full_tensor().to(hidden_states.device)`. Plain tensor behavior remains unchanged, and the transfer stays inside the autograd graph so gradients can flow back to the sharded parameter.

The patch is deliberately version-sensitive: it applies only when the known unsafe source fragment is present, is idempotent when the safe fragment is already present, and raises if neither form matches. This prevents a future veRL change from silently receiving an obsolete rewrite.

## Validation

1. A source-transform unit test must fail against the current implementation and verify both fused backends receive the transfer.
2. An idempotence test must verify already-patched source is unchanged.
3. A tensor-level test must verify a device transfer remains differentiable.
4. Existing veRL compatibility tests must pass.
5. One debug cluster smoke must use Qwen3.5-9B, thinking ON, the longest prompt subset, an 8,192-token response cap, FSDP2 offload, and fused kernels. Success requires completing an actor update without device mismatch or OOM.

## Non-goals

- Changing reward behavior or sampling parameters.
- Disabling parameter offload.
- Introducing Qwen3.5 FSDP Ulysses sequence parallelism.
- Treating one successful update as proof that a full run is campaign-safe.

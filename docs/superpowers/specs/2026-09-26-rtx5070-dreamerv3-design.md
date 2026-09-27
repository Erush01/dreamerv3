# DreamerV3 RTX 5070 GPU compatibility

## Goal and scope

Make the existing DreamerV3 checkout installable in a Python virtual environment and run a genuine DreamerV3 training update on the RTX 5070 without dependency conflicts. The validation environment is the repository's synthetic `dummy_disc` task; Atari ROMs, Minecraft, and other optional external environments are outside scope.

## Current constraints

- The host has Python 3.12, an RTX 5070 (SM 12.0), and NVIDIA driver 616.56.
- The checkout declares `jax[cuda12]==0.4.33`, `nvidia-cuda-nvcc-cu12<=12.2`, and `numpy<2`; these conflict with the working JAX 0.11.2 CUDA 13 / NumPy 2 GPU installation.
- The existing parent `.venv` imports DreamerV3 and executes a trivial GPU calculation, but `pip check` reports three dependency conflicts. It has not run a DreamerV3 training step.
- Several repository `jax.jit` calls pass parameters positionally, incompatible with modern JAX's keyword-only parameters.

## Approach

1. Update the repository's dependency declarations consistently (`requirements.txt` and any packaging metadata that reads it) for a matched, reproducible Python 3.12 + JAX CUDA 13 + NumPy 2 installation. Do not keep contradictory CUDA 12 or NumPy 1 constraints. Avoid unnecessary changes to optional environment implementations or the legacy Dockerfile.
2. Change only JAX API calls exercised by the real training path, preserving sharding, static-argument, and donation semantics. Address additional incompatibilities only when reproduced by the validation run.
3. Build a clean project virtual environment (replacing the current parent `.venv` only after checking its contents), install the repository normally so package metadata reflects the selected dependencies, and run `pip check`.
4. Validate in layers: import, JAX detection and computation on `CudaDevice`, dummy-environment initialization, a short actual DreamerV3 training run with at least one optimizer update and finite loss, then checkpoint creation/reload. Use a fresh temporary log directory; do not require ROMs or commit generated files.

## Verification and failure policy

- Passing import or random-agent runs alone does **not** count as success.
- Success requires `pip check` without conflicts, actual CUDA backend use, a completed DreamerV3 optimizer update with a finite training loss, and a working checkpoint save/reload in the synthetic environment.
- GPU allocation/compilation may be limited by other processes' VRAM usage; distinguish resource exhaustion from code/dependency defects. Do not stop or modify unrelated GPU processes.
- If modern JAX is incompatible with essential upstream behavior beyond a minimal patch, stop and report the blocker rather than claim universal compatibility.

## References

- DreamerV3 upstream: https://github.com/danijar/dreamerv3/blob/main/README.md
- JAX NVIDIA installation: https://docs.jax.dev/en/latest/installation.html#nvidia-gpu
- JAX JIT signature: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html

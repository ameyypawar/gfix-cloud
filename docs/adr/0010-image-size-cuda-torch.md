# ADR 0010: Image size — CPU-only torch is a known optimization, not yet done

## Context

`api/Dockerfile`'s runtime stage installs `sentence-transformers`, which
pulls in `torch` via `requirements.txt`. `pip install torch` on
`python:3.12-slim` without a CPU-only index constraint currently resolves to
a CUDA-enabled build, which is large (the image is roughly ~9GB as built
today) even though the container runs no GPU workload — embeddings run on
CPU via `sentence-transformers`.

## Decision (status: **not yet implemented** — tracked here as a known
optimization for the deploy phase, not a completed change)

Pin `torch` to its CPU-only wheel (e.g. via the `--index-url
https://download.pytorch.org/whl/cpu` PyPI extra index, or an equivalent
`requirements.txt` constraint) in the `py-builder` stage of `api/Dockerfile`.

## Consequences (once implemented)

- Image size should drop substantially (CPU-only torch wheels are a small
  fraction of the CUDA build's size), which matters for Fly.io deploy
  (pull time, image storage) in Phase 7.
- No functional change expected — the embedding path already runs on CPU;
  this only removes unused CUDA kernels/libraries from the shipped image.
- This is deferred to the deploy phase rather than done now, since Phase 6
  scope is the Docker/compose demo working correctly, not image-size
  optimization; the working keyless demo takes priority over a leaner
  image at this stage.

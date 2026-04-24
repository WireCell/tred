# TRED Codebase Overview

> Read-only examination.  No source files were modified.
> Docs pinned to commit **a27a0c9** (24 commits ahead of original snapshot 9f1c714).
> Cross-references: [01_algorithms.md](01_algorithms.md) [02_bugs.md](02_bugs.md) [03_gpu_efficiency.md](03_gpu_efficiency.md) [04_memory.md](04_memory.md) [99_open_questions.md](99_open_questions.md)

---

## What TRED Does

TRED is a three-dimensional detector-response simulation for liquid-argon
time-projection chambers (LArTPCs) with pixel readout.  It targets NuMI
Near Detector / ND-LAr geometry.  The code is pure Python on top of PyTorch
and is designed to run on CPU or GPU (tested on CUDA).

Given a set of ionising particle **steps** (or point **depos**) from a
Geant4-level tracker, TRED outputs **waveforms** (digitised current vs. time)
on every pixel that triggered, equivalent to what the real detector produces.

---

## Repository Layout

```
tred/
├── src/tred/               # library
│   ├── graph.py            # nn.Module wrappers; Raster dtype= param (aa91c37, ce16e9d);
│   │                       #   _order_step_endpoints_for_drift_time replaces _ensure_tail_closer_to_anode;
│   │                       #   _head_time_offset_from_tail replaces _time_diff;
│   │                       #   sigma distance→time transform now on both depo and step paths (BUG-05, BUG-11 fixed)
│   ├── blocking.py         # Block data structure (batch of rectangular volumes)
│   ├── sparse.py           # Block-Sparse Binned (BSB) operations & SGrid
│   ├── chunking.py         # accumulate(), content(), location() for BSB chunks
│   ├── partitioning.py     # deinterlace / de-interlace-pairs for convolution
│   ├── indexing.py         # crop_batched helper
│   ├── bsb.py              # (thin class wrapper around sparse.py functions)
│   ├── drift.py            # drift / diffuse / absorb physics
│   ├── recombination.py    # Birks / Modified-Box recombination models
│   ├── raster/
│   │   ├── depos.py        # Gaussian charge boxes for point depos; dtype= parameterised;
│   │   │                   #   n_half clamped ≥ 1 (a3f3491); linspace device= fixed (17b588b)
│   │   └── steps.py        # Gaussian-weighted line integrals for steps (GL);
│   │                       #   dtype= threaded through all helpers (ce16e9d, 4d8403c);
│   │                       #   DEFAULT_FLOAT_DTYPE replaces float_dtype; _snap_near_integer added
│   ├── response.py         # Load & reformat ND-LAr field response tensor
│   ├── convo.py            # Interlaced FFT-based convolution
│   ├── readout.py          # Discriminator / ADC / CSA model
│   ├── loaders.py          # NpzFile, HdfFile, StepLoader datasets
│   ├── io_nd.py            # ND-specific datasets, samplers, geometry parser
│   ├── io.py               # write_npz
│   ├── util.py             # to_tensor, to_tuple, iter_tensor_chunks, ...
│   ├── units.py            # physical unit constants
│   ├── types.py            # index_dtype, type aliases
│   ├── cli.py              # Click CLI (plots, fullsim commands)
│   └── plots/
│       └── graph_effq.py   # fullsim end-to-end pipeline (runit / fullsim)
├── docs/
│   ├── concepts.org        # Block-Sparse Binned concepts and terminology
│   ├── internals.org       # Developer notes on grids, sparse data, indexing
│   ├── convo.org           # Convolution design notes
│   └── examinations/       # ← this directory
└── tests/                  # pytest suite
```

---

## Full-Simulation Pipeline (`fullsim`)

Entry point: `cli.py:fullsim` → `plots/graph_effq.py:fullsim` →
`plots/graph_effq.py:runit`.

### Stage-by-stage data flow

```
Input HDF5 file
    │
    ▼
[IO / loaders]
StepLoader / steps_from_ndh5             loaders.py, io_nd.py
    │  features: (N_steps, 11) float32    (dE, dEdx, x0,y0,z0, x1,y1,z1, t0,t1,...)
    │  labels:   (N_steps, ...) int       (event id, tpc id, ...)
    ▼
[Recombination]  recombination.py
birks(dE, dEdx, efield, rho, ...)   → charge (N_steps,) float32
    │
    ▼
[Drift]  drift.py / graph.py:Drifter
drift(locs, velocity, diffusion, lifetime, target, ...)
    → dsigma (N_steps, 3)   post-drift Gaussian sigma
    → dtime  (N_steps,)     time at response plane
    → dcharge(N_steps,)     absorbed charge
    → dtail  (N_steps, 3)   position near anode
    → dhead  (N_steps, 3)   position far from anode
    │
    ▼
[Raster]  raster/steps.py / graph.py:Raster
compute_qeff(grid_spacing, X0, X1, Sigma, Q, n_sigma, npoints, method)
    → Block(location: (N_steps, 3) int, data: (N_steps, Sx, Sy, St) float64)
    Each "charge box" is a dense 3-D tensor giving effective electron
    counts on grid voxels near the step, computed by Gauss–Legendre
    quadrature of the 3D anisotropic Gaussian.
    │
    ▼
[ChunkSum — charge]  sparse.py / chunking.py / graph.py:ChunkSum
chunkify → accumulate
    Merges charge boxes onto a coarser super-grid (chunk_shape voxels).
    Returns a Block of non-overlapping chunks ready for convolution.
    │  signal Block: (N_chunks, Cx, Cy, Ct) float
    ▼
[LacedConvo]  convo.py / graph.py:LacedConvo
interlaced_symm_v2(signal, response, lacing=[10,10,1])
    De-interlaces signal and response by impact position (lacing factor),
    FFT-convolves each pair, exploits mirror symmetry to halve the FFT
    count, accumulates currents.
    Returns Block(location, data: (N_chunks, Cx', Cy', Ct')) float
    │
    ▼
[ChunkSum — current]  chunking.py / graph.py:ChunkSum
    Sums overlapping current chunks.
    │
    ▼
[ChunkSum — readout grouping]
    Re-groups to (1,1,120) pixel×tick blocks for readout.
    │
    ▼
[concatenate_waveforms]  plots/graph_effq.py
    Sparse currents assembled into dense waveform array per pixel.
    │
    ▼
[Readout]  readout.py:nd_readout
    Cumulative-sum discriminator / ADC hold / CSA reset model.
    Returns list of (pixel, time, hold_time, start_time, charge) hits.
    │
    ▼
Output NPZ file  (io.py:write_npz)
```

### Loop structure

The full pipeline is batched at two levels:
1. **TPC loop** — one pass per TPC module in the geometry (~8 for 2×2).
2. **Batch loop** — within each TPC, steps are fed in batches of
   `batch_scheme[0]` (default 4096×8 = 32768) steps per raster call, then
   further sub-batched at `batch_scheme[1]` (default 50 chunks) per
   convolution call.

---

## Key Data Types

| Name | Where | Shape (example) | Dtype | Description |
|------|--------|-----------------|-------|-------------|
| `Block` | `blocking.py` | location (N,3), data (N,Sx,Sy,St) | int32/float32 | Batched sparse volumes |
| charge box | `raster/steps.py` | (N_steps, Sx, Sy, St) | DEFAULT_FLOAT_DTYPE (float64 default, float32 optional) | Per-step Gaussian integral |
| signal Block | after ChunkSum | (N_chunks, Cx,Cy,Ct) | float32 | Coarse-grid charge |
| response tensor | `response.py` | (90,90,6400) | float32 | Pixel field response |
| current Block | after convo | (N_chunks, Cx',Cy',Ct') | float32 | Induced currents |
| waveform | `concatenate_waveforms` | (N_pixels, Nt) | float32 | Dense pixel waveforms |

---

## Device Strategy

- All `nn.Module` objects (Drifter, Raster, ChunkSum, LacedConvo) are
  `.to(device)` at construction time.
- Input data is loaded on CPU and moved with `[f.to(device) for f in features]`
  at the start of each batch — this is a single transfer per batch.
- The response tensor is moved to device once per TPC invocation:
  `response.to(device=device)`.
- The `raster/steps.py` module defines `DEFAULT_FLOAT_DTYPE = torch.float64`
  and all raster helpers default to this dtype.  A `dtype=` keyword is now
  threaded through all helpers and through `Raster.__init__`, making fp32 an
  actionable option (commits ce16e9d, 4d8403c).  However, `compute_index`
  internally promotes coordinates to fp64 regardless of the user-chosen dtype,
  so the performance gain from fp32 is less than the theoretical maximum.
  On CUDA GPUs without native FP64 (consumer cards), the default fp64 still
  imposes a large performance penalty.  See [03_gpu_efficiency.md](03_gpu_efficiency.md#fp64)
  (EFF-01: OPEN but now actionable) and [03_gpu_efficiency.md](03_gpu_efficiency.md#compute-index-fp64)
  (EFF-15: transient fp64 in compute_index).

---

## Existing Profiling Data

The repo root contains `itpc_timing_summary.{csv,json}` and
`pie_chart_{all,noconvo}.png`, suggesting prior per-stage timing on the
full pipeline.  These are a useful baseline before making optimisations.

---

## Notable Changes Since Snapshot 9f1c714

The following items from the examination docs were updated to reflect commits
9925b97 through a27a0c9:

| Item | Status | Commit |
|------|--------|--------|
| BUG-05 (`_ensure_tail_closer_to_anode` naming/semantics) | **FIXED** | aa91c37 |
| BUG-11 (depo-path sigma distance→time conversion missing) | **FIXED** | aa91c37 |
| EFF-01 (fp64 default) | OPEN — fp32 now actionable via `dtype=` | ce16e9d, 4d8403c |
| EFF-08 (linspace device= in depos loop) | Partial fix — device-transfer half **FIXED** | 17b588b |
| EFF-15 (transient fp64 in compute_index) | OPEN (new finding) | — |
| MEM-08 (transient fp64 allocs in compute_index) | OPEN (new finding) | — |

---

## Design Documents

The existing `docs/` org-mode files are valuable reading alongside this
examination:

- `docs/concepts.org` — Block-Sparse Binned (BSB) terminology, super-grid, bins.
- `docs/internals.org` — grids, voxels, sparse data, key accumulation
  problem, scatter/gather patterns, vmap, jagged tensors.
- `docs/convo.org` — convolution design rationale.
- `docs/response.org` — response format.
- `docs/data.org` — data formats.

# TRED Algorithm Explanations

> Read-only examination. No source files were modified.
> See also [00_overview.md](00_overview.md) for data-flow context.

---

## 1. Recombination (`recombination.py`)

### What it does
Converts energy deposition (dE in MeV) and stopping power (dE/dx in MeV/cm)
into an expected number of ionisation electrons surviving recombination.

Two models are implemented:

**Birks model** (`recombination.py:13`):
```
R = A / (1 + dEdx * k / (E * rho))
Q = R * dE / Wi
```

**Modified Box model** (`recombination.py:37`):
```
R = ln(A + B*dEdx/(E*rho)) / (B*dEdx/(E*rho))
Q = R * dE / Wi
```

Both are fully vectorised over (N_steps,). No Python loops. No GPU sync.
Typical output dtype: float32.

### Key tensors
| Tensor | Shape | Dtype |
|--------|-------|-------|
| dE, dEdx | (N_steps,) | float32 |
| charge Q out | (N_steps,) | float32 |

### Cost
Low: two element-wise operations over N_steps. Memory: O(N_steps).

---

## 2. Drift (`drift.py`)

### What it does
Models three physical processes for each step / depo:
1. **Transport** (`drift.py:20`): computes drift time `dt = (target − loc) / v`
   along the drift axis (default x, `vaxis=0`).
2. **Diffusion** (`drift.py:32`): computes post-drift Gaussian sigma via
   `sigma = sqrt(2*D*dt + sigma0^2)` (quadrature addition of longitudinal and
   transverse diffusion).  Separate diffusion coefficients are supported per
   dimension.
3. **Absorption** (`drift.py:82`): applies electron lifetime attenuation
   `Q_out = Q_in * exp(-dt / lifetime)`.  With `fluctuate=True` applies a
   Binomial fluctuation on top.

All three operations are vectorised over (N_steps,).  The `Drifter` nn.Module
(`graph.py:68`) wraps these and, when both `tail` and `head` are provided,
calls `_order_step_endpoints_for_drift_time` to establish a consistent ordering
before computing drift time.  See the section below for the semantics of this
ordering.

Optional time-shift correction `drtoa` converts anode-plane drift time to
response-plane drift time.

### Key tensors
| Tensor | Shape | Dtype |
|--------|-------|-------|
| locs in/out | (N_steps,) or (N_steps, vdim) | float32 |
| times in/out | (N_steps,) | float32 |
| sigma out | (N_steps, vdim) | float32 |
| charge out | (N_steps,) | int32 → float32 |

### Cost
Low: all element-wise or broadcast operations. Memory: O(N_steps × vdim).
Notable: `locs.detach().clone()` copies the position tensor every call
(`drift.py:154`). See [04_memory.md](04_memory.md#drift-clone).

---

## 3. Raster — Depos (`raster/depos.py`)

### What it does
For point-like depos, computes the effective charge on each grid voxel by
integrating the 3-D Gaussian distribution over each voxel.

The N-D version (`binned_nd`, `raster/depos.py:142`) works as follows:

1. Compute per-depo, per-dimension half-span `n_half`:
   how many grid points are needed to cover ±nsigma×sigma via
   `torch.ceil((sigmas * nsigma) / grid)`, then clamped to a minimum of 1:
   `n_half = torch.clamp(n_half, min=1)` (`raster/depos.py:194–195`).
   This ensures the minimum charge-box buffer is always at least 1 grid point
   per dimension, preventing the zero-size buffer that was possible before
   commit a3f3491 when sigma was very small.
   **Critically, the maximum over all depos is taken**, creating a universal
   box shape.
2. Build the `grid0` lower corner for each depo.
3. Loop over spatial dimensions (Python `for` loop, `raster/depos.py:217`).
   For each dimension:
   a. Construct relative grid point indices via `torch.linspace(0, 2*dim_n_half,
      ..., dtype=dtype, device=device)` — device and dtype are now passed at
      call time rather than via a `.to(device)` follow-up call (commit 17b588b).
   b. Convert to absolute coordinates.
   c. Compute `erf` values at bin edges.
   d. Compute bin integrals as half the difference of adjacent erfs.
   e. The spike index for zero-width depos is cast to `torch.long` before use:
      `rel_gridc[spikes, dim].to(dtype=torch.long)` (`raster/depos.py:239`),
      fixing a prior bug where the index dtype was not guaranteed to be long.
4. Form the N-D charge box as the outer product of per-dimension integrals via
   `torch.einsum` (for N=2 or N=3).
5. Multiply by Q.

Both `binned_1d` and `binned_nd` accept a `dtype=` keyword (defaulting to
`DEFAULT_FLOAT_DTYPE = torch.float64`, `raster/depos.py:62`); all tensors
allocated internally are cast to this dtype.

**Output:** `(qeff, grid0)` where `qeff` has shape `(N_depos, 2*n_half[0]+1,
2*n_half[1]+1, 2*n_half[2]+1)` and `grid0` has shape `(N_depos, vdim)`.

### Key property: universal (worst-case) box shape
Because all depos are padded to the same box shape (maximum sigma depo),
depos with small diffusion waste most of their allocated voxels.  This is
a significant memory amplification when sigma varies widely. See
[04_memory.md](04_memory.md#universal-box).

---

## 4. Raster — Steps (`raster/steps.py`)

### What it does
For line-segment steps, computes the charge distribution by integrating the
**anisotropic 3-D Gaussian smeared along a line segment** over each grid
voxel.  This is substantially more complex than point depos.

The key function is `compute_qeff` which calls `compute_charge_box` then
one of two evaluation methods:

**Method `gauss_legendre`** (default):
Uses Gauss–Legendre quadrature to numerically integrate the charge density
`qline_diff3D` (`steps.py:232`) at GL nodes within each grid voxel, then
sums the weighted values.  The analytical integrand is:

```
q(x,y,z) = Q / (4π Δ) * exp(-sy²(x*dz01 + ...) / (2Δ²))
          * exp(...) * exp(...) * [erf(·) - erf(·)]  / (sqrt(2) Δ sx sy sz)
```
where `Δ = sqrt(sy²sz²*dx01² + sx²sy²*dz01² + sx²sz²*dy01²)` (the
denominator encoding the line-spread geometry).  This is computed as a
TorchScript function (`qline_diff3D_script`, `steps.py:257`) for JIT speed.

The GL quadrature uses `npoints=(2,2,2)` by default (8 evaluation nodes per
voxel), with weights pre-computed once via `create_wu_block`.

**Short step fallback**: steps shorter than 5% of the sigma along any
dimension are treated as point depos (`qpoint_diff3D`, `steps.py:321`).

### Universal shape
Same issue as depos: `reduce_to_universal` (`steps.py:150`) collapses all
per-step box shapes to the per-batch maximum before allocating, so every
step's charge box is padded to the worst-case size.
See [04_memory.md](04_memory.md#universal-box).

### Default float dtype and `dtype=` parameter
The module defines `DEFAULT_FLOAT_DTYPE = torch.float64` (`steps.py:11`).
This constant replaces the old bare `float_dtype` variable.  All public
functions (`compute_bounds_X0X1`, `compute_bounds_X0_X1`, `compute_charge_box`,
`create_w_block`, `create_u_block`, `create_wu_block`, `eval_qeff`,
`compute_qeff`, etc.) now accept an explicit `dtype=` keyword argument
(defaulting to `DEFAULT_FLOAT_DTYPE`) which is threaded through all
tensor allocations and `to_tensor` calls.  Passing `dtype=torch.float32`
allows the full raster pipeline to run in fp32.

### `_snap_near_integer` — boundary jitter guard  (`steps.py:49`)
```python
def _snap_near_integer(values: Tensor, source_dtype):
    nearest = torch.round(values)
    eps = torch.finfo(source_dtype).eps
    tol = 8 * eps * torch.clamp(nearest.abs(), min=1.0)
    return torch.where((values - nearest).abs() <= tol, nearest, values)
```
Before flooring the normalised coordinate `(coord − origin)/spacing` to get
the grid index, `_snap_near_integer` detects values that lie within `8 eps`
of an integer boundary and snaps them exactly to that integer.  Without this
guard, fp32/fp64 boundary jitter can cause a coordinate that is conceptually
exactly on a grid line to floor to the wrong cell, producing charge-box index
errors.  `source_dtype` is the coarsest float dtype among the inputs (computed
by `_coarsest_float_dtype`), so the tolerance adapts to the actual precision in
use.

### Internal fp64 promotion in `compute_index`  (`steps.py:79`)
Even when `dtype=torch.float32` is requested, `compute_index` promotes the
coordinate arithmetic to fp64 internally:
```python
coords  = to_tensor(coords,  device, dtype=torch.float64)   # line 91
origin  = to_tensor(origin,  device, dtype=torch.float64)   # line 94
grid_spacing = to_tensor(grid_spacing, device, dtype=torch.float64) # line 95
idxs = (coords - origin.unsqueeze(0)) / grid_spacing.unsqueeze(0)  # fp64
idxs = _snap_near_integer(idxs, source_dtype)
return idxs.floor().to(index_dtype)                          # cast back
```
The `source_dtype` passed to `_snap_near_integer` is determined before
promotion via `_coarsest_float_dtype`, so the tolerance reflects the
user-chosen precision.  The result is always cast to `index_dtype`
(int32) — there is no fp64 tensor surviving from `compute_index` into
the charge evaluation.  However, fp64 tensors **do** live transiently
inside `compute_index`, doubling the memory of the index computation pass
regardless of the user-chosen dtype.  See [03_gpu_efficiency.md](03_gpu_efficiency.md#fp64)
and [04_memory.md](04_memory.md#compute-index-fp64).

### Float64 usage (overall)
See [03_gpu_efficiency.md](03_gpu_efficiency.md#fp64).

---

## 5. `Raster` nn.Module (`graph.py:224`) — naming, dtype, and sigma transform

### dtype parameter
`Raster.__init__` now accepts a `dtype=` argument (default `torch.float64`,
`graph.py:229`).  It is stored as `self._dtype` and validated at construction
time:
```python
if not torch.empty((), dtype=dtype).is_floating_point():
    raise ValueError('Raster dtype must be a floating-point torch.dtype')
```
All input tensors in `Raster.forward` are cast to `self._dtype` before being
passed to the raster helpers.  The `dtype=` is forwarded to `raster_steps`
and `raster_depos` (via `raster_steps(... dtype=self._dtype)` and
`raster_depos(... dtype=self._dtype)`) so the full raster computation runs at
the chosen precision.

### `_order_step_endpoints_for_drift_time` (renamed from `_ensure_tail_closer_to_anode`)

**FIXED in aa91c37.**  The static method previously named
`_ensure_tail_closer_to_anode` is now named
`_order_step_endpoints_for_drift_time` (`graph.py:160`).

**Semantics (current):**  
The method reorders `(tail, head)` so that `tail` is the **upstream**
endpoint — i.e. the endpoint farther from the anode:
- If `velocity > 0`, tail ends up with the **smaller** coordinate along
  `vaxis` (farther from the anode for positive-velocity drift).
- If `velocity < 0`, tail ends up with the **larger** coordinate along
  `vaxis`.

This makes `tail` the endpoint used to define the drift time in
`Drifter.forward()`.  The name change reflects that the ordering is based on
drift direction, not a direct comparison to the anode position.  The old name
`_ensure_tail_closer_to_anode` had **inverted semantics** — the tail is now
documented as the upstream (farther-from-anode) endpoint, not the closer one.
See [02_bugs.md](02_bugs.md#tail-swap) for the original bug record.

### `_head_time_offset_from_tail` (renamed from `_time_diff`)
The method computing the head-time offset from tail time is now named
`_head_time_offset_from_tail` (`graph.py:257`):
```python
def _head_time_offset_from_tail(self, tail, head=None):
    ...
    d = tail - head  (for 1D)  or  tail[:,tdim] - head[:,tdim]
    return d / self.velocity
```
This returns `(tail_coord - head_coord) / velocity`, which is the time to
add to the tail time to obtain the head time.

### Sigma distance-to-time transform — both depo and step paths

**FIXED in aa91c37.**  In `Raster.forward` (`graph.py:304`), the
distance-to-time conversion for the drift-axis sigma:
```python
sigma[:, self._tdim] = sigma[:, self._tdim] / torch.abs(self.velocity)
```
is now applied **before** the step/depo branch (`graph.py:331`).  This means
both the depo path (`raster_depos`) and the step path (`raster_steps`) receive
sigma already in time units along the drift axis.

Previously this transform was only applied on the step path; the depo path
received sigma in distance units, causing incorrect charge spread along the
drift axis for point depos.  See [02_bugs.md](02_bugs.md#depo-sigma-units).

---

## 6. Block Data Structure (`blocking.py`)  {#block-data-structure}

A `Block` is the central data container throughout TRED.  It holds:
- `location`: (N_batches, vdim) int32 tensor — absolute grid-index of the
  lower corner of each volume.
- `data`: (N_batches, d1, d2, ..., dN) float tensor — values on the volume.

All volumes in one Block share the same `shape` (d1,d2,...,dN), so a Block
is essentially a dense batched tensor of rectangular volumes at arbitrary
sparse locations.  This is the "Block Sparse Binned" representation described
in `docs/concepts.org`.

Key methods: `size()` (with CPU sync warning), `vdim`, `nbatches`.
Helper: `concat_blocks(blocks)` — concatenates along the batch dimension.

---

## 7. Block-Sparse Accumulation (`sparse.py` / `chunking.py`)

### The central problem
After rastering, each step produces one charge box (a Block batch element)
at an arbitrary grid location.  Many boxes overlap on the grid.  The goal
is to **merge overlapping boxes** onto a coarser "chunk" grid, summing
charges at common voxels.

### chunkify (sparse.py:160)
1. Create a `SGrid` with spacing = `chunk_shape`.
2. Compute the smallest aligned "envelope" Block containing all input blocks.
3. Call `fill_envelope`: expand the (possibly empty) envelope tensor and
   write each input block into it at the correct offset using flat indexing
   (`crop_batched`).
4. Call `reshape_envelope`: view the envelope as a grid of chunks, return
   a new Block where each batch element is one chunk.

**Key limitation**: `fill_envelope` **assigns** (not accumulates) values into
the envelope (`sparse.py:131`).  This means if two input blocks map to the
same voxel in the envelope, the second silently overwrites the first.
Whether this is safe depends on the invariant that `chunkify` is called with
non-overlapping source blocks — see [02_bugs.md](02_bugs.md#fill-envelope).

### accumulate (chunking.py:140)
After chunkify produces a Block of chunks (some at identical locations):
1. `torch.unique(loc, dim=0)` finds distinct chunk locations.
2. `index_add_(0, inverse, data)` accumulates data from duplicate chunks
   into a single output.
3. Non-zero chunks (|val| > 1e-3) are retained.

`index_add_` is a single parallel GPU operation — efficient.

### Alternative: chunkify2 (sparse.py:207)
A second, scatter-based implementation that does not build an explicit
envelope.  Instead it computes flat tile indices for every element of every
block and uses a two-step `scatter_add_ + index_add_` to accumulate directly.
This avoids the large envelope allocation but has a CUDA-CPU sync hazard
(`sparse.py:265`).  See [03_gpu_efficiency.md](03_gpu_efficiency.md#chunkify2-sync).

### accumulate_nd_blocks_v1 / v2 (sparse.py:275 / 315)
Two in-place variants that skip the envelope entirely and operate on
the raw Block data tensor.  `_v1` has a Python for-loop over unique chunk
indices (very slow; see [03_gpu_efficiency.md](03_gpu_efficiency.md#v1-loop)).
`_v2` uses `scatter_add_` and is much better but has index-shape complexity
bugs under certain configurations — see [02_bugs.md](02_bugs.md#v2-scatter).

---

## 8. Interlaced Convolution (`convo.py` / `partitioning.py`)

### Motivation
The ND-LAr pixel response tensor represents the induced current at a pixel
for an electron landing at each of `nimperpix × nimperpix = 10×10 = 100`
impact positions within the pixel pitch.  A direct convolution of the charge
grid with the response must account for this sub-pixel structure.

The approach is called "interlaced" convolution:
- The charge grid has a fine spacing (1/10 pixel pitch).
- The response tensor also lives on this fine grid but only has support at
  every 10th point (the "lace" spacing = `[10, 10, 1]`).
- For each (impact_x, impact_y) combination (50 pairs exploiting mirror
  symmetry), extract the corresponding "lace" from both signal and response
  and perform a standard FFT convolution.
- Sum the 50 partial convolutions to get the total induced current.

### interlaced_symm_v2 (convo.py:344) — current recommended version

1. **DFT shape**: compute the padded size needed for linear (non-circular)
   convolution: `c_shape = signal_size/lacing + response_size/lacing - 1`.
2. **De-interlacing** (`partitioning.py:63`): for each of the
   `nimperpix/2 = 5` symmetric pairs along the transverse axis, extract
   signal and response laces via strided slicing.  Use the complex-number
   trick: pack a pair `(lace_fwd, flip(lace_rev))` into real and imaginary
   parts of a complex tensor so one FFT handles both.
3. **FFT, multiply, accumulate**: pad to `o_shape`, `fftn`, multiply
   `response_fft × signal_fft`, accumulate into `meas`.
4. **iFFT and unpack**: `ifftn`, then `meas.real + flip(meas.imag)` unpacks
   the symmetric pair trick.

### Recomputing response FFT every call
The line `torch.fft.fftn(res_[None,...], dim=dims)` (`convo.py:385`) is
inside the per-lace loop AND called on every `forward()` invocation.  The
response never changes between calls; pre-computing and caching its FFT
would be a significant speedup.  See [03_gpu_efficiency.md](03_gpu_efficiency.md#fft-cache).

---

## 9. Readout (`readout.py`)

### What it does
Models the pixel electronics chain:
1. Compute a cumulative sum of the induced current waveform (`Xacc = X.cumsum`).
2. Iteratively find the first time tick where the accumulated charge exceeds
   the pixel threshold (discriminator crossing).
3. Record: crossing time, charge at ADC hold time (`hold_t = cross_t + adc_hold_delay`).
4. Reset the accumulator at `hold_t + csa_reset_time`, advance the "start"
   pointer to `hold_t + adc_down_time` to model dead time.
5. Repeat until no more crossings.

The threshold can be per-pixel (loaded from HDF5, shape (N_pixels,)).

Output: flat list of (pixel_x, pixel_y, t_cross, t_hold, t_start) and
corresponding charge.

### Iteration on GPU
The while-loop at `readout.py:67` iterates until `triggered.any() == False`.
Each `triggered.any()` forces a CPU-GPU sync (`.any()` is a reduction that
requires pulling a scalar back to the host).  The number of iterations equals
the maximum number of ADC triggers any single pixel fires.  This can be
many tens of iterations for high-occupancy events.
See [03_gpu_efficiency.md](03_gpu_efficiency.md#readout-loop).

---

## 10. Response Loading (`response.py`)

### What it does
Loads the ND-LAr field response `.npy` file (shape `(45, 45, 6400)` — quarter
of the full response), applies quadrant symmetry (`quadrant_copy`) to produce
the full `(90, 90, 6400)` tensor, then reorders axes to make impact-position
indexing contiguous for the laced convolution:

```
raw (45,45,6400) → full (90,90,6400)
→ view (9, 10, 9, 10, 6400)   # npxl × nimp × npxl × nimp × Nt
→ flip on dims (0,2)
→ reshape (90, 90, 6400)
→ .contiguous()
```

The `view + reshape + contiguous()` sequence creates a full physical copy of
the 90×90×6400 response (≈ 207 MB at float32).  This lives on the GPU for
the entire run.

---

## 11. IO and Loaders (`loaders.py`, `io_nd.py`)

### StepLoader / steps_from_ndh5
Reads steps from an HDF5 file as NumPy arrays via h5py, converts to
`torch.tensor`.  Conversion uses `torch.tensor(np_array, dtype=..., requires_grad=False)`
which always copies.  `torch.from_numpy` followed by `.to(dtype)` would be
more memory-efficient for the in-place case.

### NpzFile / HdfFile
Both use `torch.tensor(data, dtype=..., requires_grad=False)` on every key
access, meaning each item is copied on load.  The HDF5 loader stores an open
file handle (`self._fp = h5py.File(path)`) but does not close it explicitly.

### CustomNDLoader / batch samplers
`io_nd.py` provides custom PyTorch DataLoader subclasses with three sampling
strategies (sorted, eager, lazy).  `SortedLabelBatchSampler` sorts by event
label to keep related steps in the same batch — this is useful to keep steps
belonging to one physics event together, but it discards stochastic shuffling
that might help GPU utilisation.

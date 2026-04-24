# TRED Open Questions

> These items could not be resolved by static code reading alone.
> They require either running the simulation, a benchmark, or a design decision.
> Prioritised roughly by expected impact.

---

## Q1 — Does `fill_envelope` actually receive overlapping blocks in practice?

**Relates to:** [BUG-01](02_bugs.md#fill-envelope)

`fill_envelope` assigns (not accumulates) into the envelope.  Whether this
is a bug depends on whether the *source block* can contain two batch elements
that map to the same envelope voxel.

- If `chunkify` is always called with a block whose charge boxes are
  non-overlapping (i.e. no two steps produce charge at the same grid point),
  then the overwrite is safe.
- If charge boxes from nearby steps overlap (very common for dense ionisation
  tracks), then values are silently lost.

**To resolve:**  
Add a temporary diagnostic: before the assignment in `fill_envelope`,
check `len(inds) > len(torch.unique(inds))` and log if True.
Run on a muon track or beam event.

---

## Q2 — Is the `meas.real = …` assignment in `convo.py:391` a no-op?

**Relates to:** [BUG-02](02_bugs.md#convo-real-assign)

Whether `Tensor.real = expr` modifies the complex tensor in-place or rebinds
a Python name depends on PyTorch version and tensor type.  If it is a no-op,
the symmetric lace contribution is dropped and currents are wrong.

**To resolve:**  
Run a known test case (e.g. the existing `tests/convo/` tests) and compare
output of `interlaced_symm_v2` vs `interlaced_symm` (which has independent
implementation).  The outputs should match to numerical precision.

---

## Q3 — What is the actual fp32 vs fp64 precision loss in `qline_diff3D`?   {#Q3}

**Relates to:** [EFF-01](03_gpu_efficiency.md#fp64), [MEM-02](04_memory.md#fp64-memory)
**Status:** OPEN — now empirically testable

The analytical integrand in `qline_diff3D_script` (`steps.py:257`) involves:
- A ratio of a Gaussian and a line-length-dependent prefactor `Q/(4π Δ)`.
- Differences of erf values that can be very small when the line is far from
  the evaluation point.
- Squared terms that can span many orders of magnitude.

**Update (commits ce16e9d, 4d8403c):**  
The `dtype=` parameter is now threaded through all raster helpers.  A
dedicated dtype smoke test exists:
```
uv run pytest tests/effq/test_raster_dtype.py -q
```
This tests that fp32 and fp64 both produce finite, positive results with
consistent shapes.  It does **not** compare numerical precision.

**Important caveat:**  
Even in fp32 mode, `compute_index` internally promotes coordinate arithmetic
to fp64 (`steps.py:91–95`).  The actual performance gain from `dtype=float32`
will therefore be **less than the theoretical 1/32× to 1/64× speedup**, since
the index computation still uses fp64.

**To resolve the precision question:**  
Run `compute_qeff` in both fp32 and fp64 on a representative set of steps
(including short steps, very oblique steps, and heavily-diffused steps).
Compare the resulting charge box sums.  Define an acceptable error bound
(e.g. total charge error < 0.1%).

---

## Q4 — How large is the worst-case box shape for realistic ND-LAr events?

**Relates to:** [MEM-01](04_memory.md#universal-box)

The charge-box tensor size scales as `N_steps × Sx × Sy × St`.  The
worst-case `(Sx, Sy, St)` depends on the maximum sigma in a batch.

**To resolve:**  
Add logging of `universal_shape` in `compute_charge_box` (`steps.py:127`)
for a few representative events (cosmic muon, beam neutrino, stopping proton).
This will give the actual memory usage breakdown and show how much padding
is wasted.

---

## Q5 — What fraction of `accumulate`'s 1e-3 filter is non-zero signal?

**Relates to:** [BUG-03](02_bugs.md#accumulate-threshold)

`chunking.accumulate` drops chunks where `max|val| ≤ 1e-3`.  This is
designed to suppress empty chunks, but may also suppress valid near-threshold
signal.

**To resolve:**  
Log the number of chunks before and after the filter for a few events.
Check whether any dropped chunks have non-zero charge (before convolution)
or non-negligible current (after convolution).

---

## Q6 — Is `accumulate_nd_blocks_v2` correct under all input configurations?

**Relates to:** [EFF-03](03_gpu_efficiency.md#v1-loop)

`_v2` uses a `scatter_add_` path that involves an intermediate `uq` (unique
locations after partial reduction) and `index_add_` on `acc0 → acc`.  The
two-stage reduction is needed when input block shapes are larger than
`cshape`, but the correctness of the index arithmetic across the batch
dimension reduction has not been formally verified.

**To resolve:**  
Run the existing tests in `tests/` with `method='chunksum_inplace_v2'` and
compare results to the reference `chunksum` method on the same inputs,
particularly for blocks where `mshape[0] > 1` (outer batching).

---

## Q7 — What is the actual timing breakdown for the current code on a GPU?

**Relates to:** profiling, prioritisation

The `itpc_timing_summary.{csv,json}` files contain prior timing data, but
may be from an older code version.  Fresh profiling with `nvprof` or
`torch.profiler` would give:
- Per-stage timing (raster, chunksum, convo, readout).
- Per-kernel breakdown within convo (FFT time vs. multiply vs. overhead).
- Memory transfer cost (CPU→GPU for each batch).

**To resolve:**  
Run `tred fullsim` with `benchmark_each_stage=True` on a representative event
and collect `itpc_timing_summary`.  Run `torch.profiler` on the convo stage
specifically to measure FFT cost vs. overhead.

---

## Q8 — Is the `leftover` charge between batch boundaries significant?

**Relates to:** [BUG-09](02_bugs.md#readout-leftover)

Charge accumulated on the CSA but not yet triggering at the end of one batch
(`twindow_max = 7200 ticks`) is currently discarded.  For a muon crossing
multiple batch windows or a long-lifetime process, this could cause missed
triggers or wrong charge assignment.

**To resolve:**  
Estimate what fraction of ND-LAr events span more than one batch window.
If events are typically shorter than the 600 µs window, this is negligible.

---

## Q9 — Is the response tensor orientation consistent with the charge-grid orientation?

**Relates to:** [01_algorithms.md §7](01_algorithms.md#7-interlaced-convolution)

`ndlarsim` in `response.py` applies a `torch.flip(response, dims=(0,2))` and
a specific reshape of the quadrant-copied tensor.  Whether this correctly
maps impact position [0,0] to the lower-left corner of the pixel (as claimed
in the docstring) needs verification against known physics: a single electron
landing at the pixel centre should induce the most current on the collection
pixel, not on a neighbour.

**To resolve:**  
Run a single-electron "depo" at pixel centre and check which pixel has the
largest integrated current in the output.

---

## Q10 — Does `Raster._transform` correctly handle the `tdim<0` case?   {#Q10}

**Relates to:** `graph.py:255–302`
**Status:** OPEN — partial improvement in `Raster.__init__`

`Raster.__init__` adjusts `_tdim` for negative input:
```python
self._tdim = tdim if tdim>=0 else len(self._pdims) + 1 + tdim
```
When `tdim=-1` and `pdims=(1,2)` (the default), this gives `_tdim = 2`.

In `_transform`, axes are reordered:
```python
axes.insert(self._tdim, old_tdim)
```
The correctness of this reordering under non-default `(pdims, tdim)`
combinations has not been systematically tested.

**Partial improvement (commit aa91c37):**  
`Raster.__init__` now validates that `dtype` is a floating-point dtype and
casts `vaxis` to `int` (via `constant(self, 'vaxis', vaxis, index_dtype)`).
These provide additional type safety at construction time.  The `_transform`
logic itself is unchanged.

**To resolve:**  
Add parametrised unit tests in `tests/` covering at least `tdim=0`, `tdim=-1`,
and `pdims=(0,1)` with known transformations.

---

## Q11 — Does `compute_index` fp64 promotion need explicit justification?   {#Q11}

**Relates to:** [EFF-15](03_gpu_efficiency.md#compute-index-fp64),
[MEM-08](04_memory.md#compute-index-fp64)

`compute_index` (`steps.py:79`) unconditionally promotes coordinate arithmetic
to fp64.  The `_snap_near_integer` guard (`steps.py:49`) adapts its tolerance
to `source_dtype` (the coarsest dtype among user inputs), but the actual
subtraction and division are always done in fp64.

**Questions:**
1. Is fp64 in the subtraction/division actually necessary, or does the
   `_snap_near_integer` guard + fp32 arithmetic suffice for typical ND-LAr
   grid spacings and coordinate magnitudes?
2. If fp64 is necessary, can the promotion be made conditional on
   `source_dtype == torch.float32` (i.e. only promote when needed) to avoid
   the overhead when the user already requests fp64?

**To resolve:**  
Run `compute_index` with fp32 inputs on a set of coordinates near grid
boundaries (within a few eps of integer multiples of `grid_spacing`) and
compare the resulting indices against the fp64 reference.  If no mismatches
are found for typical ND-LAr parameters, fp32 arithmetic may be sufficient.

---

## Q12 — Is the `n_half >= 1` clamp in depos.py correct for zero-sigma depos?   {#Q12}

**Relates to:** [MEM-01](04_memory.md#universal-box), commit a3f3491

`binned_nd` and `binned_1d` now clamp `n_half` to a minimum of 1
(`depos.py:100, 195`).  For depos with zero sigma (spike depos), the old code
set `n_half=0` which produced an empty charge box and relied on the `spikes`
path to assign charge directly to the nearest grid point within a box of size 1.

With the clamp, zero-sigma depos get a 3-voxel box (`2*1+1=3`), the `spikes`
path still assigns all charge to `rel_gridc`, and the surrounding voxels are
zero — which is correct.

**Question:**  
Is the test coverage sufficient to confirm that the `spikes` path correctly
handles `n_half=1` in all cases, including edge cases where `rel_gridc` could
be at the boundary of the 3-voxel window?

**To resolve:**  
Review the existing tests in `tests/effq/test_depos.py` (notably
`test_binned_1d_zero_width_spike` and `test_binned_nd_zero_width_spike_numerical`)
and add a case where the zero-sigma depo center is at the edge of a grid cell.

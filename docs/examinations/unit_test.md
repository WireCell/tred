# Unit-Test Advisory — `graph.py` modules and `graph_effq.py` modularization

**Reviewed snapshot:** HEAD `a27a0c9` (same snapshot as the six previous examination docs).

**Scope:**
1. Advise on unit tests for every `nn.Module` defined in `src/tred/graph.py`.
2. Identify operations in `src/tred/plots/graph_effq.py` — up to and including `nd_readout` — that could be combined into new `nn.Module`s, explicitly **excluding** configuration parsing, CLI handling, file I/O and output assembly.
3. Classify existing tests as **INVALID** (references removed/renamed APIs), **OUTDATED** (still runs but asserts the wrong thing now, or asserts nothing useful), or **VALID**.
4. Recommend new tests with unique IDs (`UT-01` …) for tracking.

Convention: `file:line` citations refer to the current HEAD. Test IDs follow the same `UT-NN` scheme used in the companion BUG/EFF/MEM/Q lists.

---

## Part A — `graph.py` module inventory

| Class | Lines | Purpose | Internal helpers | External deps |
|---|---|---|---|---|
| `Drifter` | 68–221 | Drifts depos or step endpoints to the anode plane; returns `(dsigma, dtime, dcharge, dtail[, dhead])` | `_order_step_endpoints_for_drift_time` *(renamed from `_ensure_tail_closer_to_anode`; semantics inverted — tail is now the upstream endpoint)* | `tred.drift.drift`, `tred.types.index_dtype` |
| `Raster` | 224–344 | Rasterizes depos or steps into a sparse `Block`; transforms drift-axis from distance→time via `velocity` | `_head_time_offset_from_tail` *(renamed from `_time_diff`)*, `_transform` | `tred.raster.depos.binned`, `tred.raster.steps.compute_qeff`, `tred.blocking.Block` |
| `ChunkSum` | 347–467 | Chunks a `Block` to `chunk_shape` and sums overlapping chunks; selectable method | `_chunksum`, `_chunksum2`, `_chunksum_inplace_v1`, `_chunksum_inplace_v2` | `tred.sparse.*`, `tred.chunking.accumulate` |
| `LacedConvo` | 469–487 | Interlaced-symmetric FFT convolution of signal × response | none | `tred.convo.interlaced_symm_v2` |
| `Charge` | 490–500 | Composition: `Drifter → Raster → ChunkSum` | none | composed submodules |
| `Current` | 503–512 | Composition: `LacedConvo → ChunkSum` | none | composed submodules |
| `Sim` | 515–523 | Composition: `Charge → Current` | none | composed submodules |

Module-level helpers: `raster_steps` (shim, 39–53), `param()` (55), `constant()` (61).

Recent delta reflected in source (do not reintroduce stale references when writing tests):
- Method rename + semantic inversion: `_order_step_endpoints_for_drift_time` — tail is the *upstream* endpoint for positive velocity (BUG-05 FIXED).
- Method rename: `_head_time_offset_from_tail` — returns `(tail - head) / velocity`.
- Sigma distance→time fix: in `Raster.forward` (graph.py:331) the drift-axis sigma is divided by `|velocity|` **before** the depo/step branch, so both branches receive time-unit sigma (BUG-11 FIXED).
- `Raster(dtype=…)` parameter with `DEFAULT_FLOAT_DTYPE = torch.float64`.

---

## Part B — `graph_effq.py` pipeline and module-combination candidates

The main pipeline lives in `runit()` (`src/tred/plots/graph_effq.py:193–613`). Excluding configuration, CLI, file I/O and output serialization, the in-scope operations (to readout inclusive) are:

| # | Lines | Operation | Classification |
|---|---|---|---|
| 1 | 222–224 | Construct `Drifter`, `Raster`, `ChunkSum`s, `LacedConvo` (defaults) | (A) already modularized |
| 2 | 248 | `segment_to_tpc(*make_nd(...))` — TPC dataset construction | (B) raw function call — candidate for a module |
| 3 | 278–282 | Per-TPC re-instantiation of `Drifter`/`Raster` with TPC-specific drift velocity and anode | (C) Python glue, repeated construction |
| 4 | 287–297 | Compute `tpc_lower_left` and pixel-index range | (C) glue |
| 5 | 299 | Batched iteration over `(features, labels)` | (C) streaming glue |
| 6 | 319 | `.to(device)` on features | (C) glue |
| 7 | 325–326 | `birks(dE, dEdx, efield, rho, A3t, k3t, Wi)` → charge | (B) raw call into `tred.recombination` |
| 8 | 332–336 | Extract `local_time, tail, head`; subtract TPC lower-corner on transverse axes | (C) glue + coordinate shift |
| 9 | 339 | `drifter(local_time, charge, tail, head)` | (A) Drifter |
| 10 | 342–344 | Clamp `drifted[0]` (sigma) component-wise to `min_sigma` | (C) glue |
| 11 | 362–363 | `iter_tensor_chunks(drifted, nbchunk)` streaming loop | (C) glue |
| 12 | 367 | `raster(*idrifted)` → qblock | (A) Raster |
| 13 | 374 | `assert not qblock.data.isnan().any()` | (C) defensive check |
| 14 | 378 | `chunksum(qblock)` | (A) ChunkSum |
| 15 | 398 | `iter_chunk_block(signal, batch_scheme[1])` | (C) streaming glue |
| 16 | 404 | `convo(iqblock, response)` | (A) LacedConvo |
| 17 | 413 | `chunksum_i(iblock)` | (A) ChunkSum |
| 18 | 425–427 | `concat_blocks(currents)` + `chunking.accumulate` | (B) raw calls — repeats at (19) |
| 19 | 446–448 | `concat_blocks(current_blocks)` + `chunking.accumulate` | (B) duplicated pattern |
| 20 | 469 | `chunksum_readout(currents)` | (A) ChunkSum |
| 21 | 470 | `concatenate_waveforms(currents, twindow_max, event_t=…)` | (B) local function (95–155) |
| 22 | 471 | Unit scaling `data * tspace / 1e3` (ke⁻) | (C) glue |
| 23 | 472–474 | Spatial mask against pixel range | (C) glue |
| 24 | 486–491 | `nd_readout(…)` → hits  **(pipeline end — readout)** | (B) raw call into `tred.readout` |

### Recommended module consolidations

Each group below is a sequence of 2–3 raw-function calls or glue blocks that form a logical unit and can be wrapped into a single `nn.Module` with a clear forward contract.

- **MOD-1 `Recombination`** — wrap steps 7 + 8 (birks + transverse-axis shift to TPC-local). Inputs `(features, tpc_lower_left)`; outputs `(charge, local_time, tail, head)`. Extracts roughly 15 lines of inlined arithmetic at 319–336 into a reusable module.
- **MOD-2 `SigmaFloor`** — wrap step 10 (per-axis minimum-sigma clamp). Inputs `(dsigma, min_sigma)`; outputs `dsigma'`. Trivial but currently hand-written at every call site; making it a module makes the policy visible to readers and testable in isolation.
- **MOD-3 `BlockAccumulator`** — wrap steps 18–19 (`concat_blocks → chunking.accumulate`). The same two-call pattern appears twice in the same function; a `nn.Module` removes the duplication and gives a single test surface.
- **MOD-4 `WaveformAssembler`** — wrap steps 20 + 21 + 22 + 23 (chunksum → concatenate_waveforms → unit scaling → pixel-range mask). This block prepares the input for `nd_readout` and is currently a dense 5-line hand-rolled section; as a module, the scaling constant (`tspace / 1e3`) and the pixel-range policy become constructor parameters.
- **MOD-5 `Readout`** — wrap step 24 as a thin `nn.Module` so the full pipeline can terminate in a module composition equivalent to `Sim` but extended through readout. Keeps `graph.py` ↔ `graph_effq.py` symmetry.
- **MOD-6 `TPCSim`** — composition `Recombination → SigmaFloor (post-Drifter) → Charge → Current → BlockAccumulator → WaveformAssembler → Readout`, parameterized by a single TPC context. Replaces lines 278–491 with a clean module call per TPC.

All six are pure rearrangements of existing code — no algorithmic change. Configuration loading (`fullsim`, `runit`'s argparse section), file I/O (`write_npz`, `np.load`), and the dataset iteration (`iter_tensor_chunks`, `iter_chunk_block`) remain outside the module boundary as requested.

---

## Part C — Existing tests inventory

All paths below are relative to `/nfs/data/1/xqian/DUNE_ND_Sim/tred/`.

### Invalid tests (references removed/renamed APIs)

After the HEAD refactor:

- `grep -n "_ensure_tail_closer_to_anode\|_time_diff\|^float_dtype\s*=\|qmodel=" tests/` returns **no stale production-API references**. Test files have already been updated to new names (`_order_step_endpoints_for_drift_time`, `_head_time_offset_from_tail`) as part of the same delta that renamed the source.

No currently **INVALID** tests were found. This is a clean state for future work.

### Outdated tests (still run, but assert the wrong thing or nothing useful)

- **`tests/raster/test_raster_graph.py`** — **OUTDATED**. The file is an integration smoke over `Drifter + Raster` but its body is almost entirely plotting helpers; numerical assertions against the current Raster semantics are absent. It also predates the drift-axis sigma fix (BUG-11) and thus cannot detect a regression of that specific bug. Recommendation: replace with the numerical test UT-15 below and keep a small plotting helper under `tests/playground/`.
- **`tests/convo/test_response.py`** — **OUTDATED**. Plot-only; no `assert` statements. Either promote to a numerical contract test (UT-12) or move under `tests/playground/`.
- **`tests/nd_loader/test_io_dataset.py`** — **OUTDATED** (environment-coupled). Contains hard-coded absolute paths under `/home/yousen/...`; will fail on any other workstation and in CI. Either parameterize on a fixture under `tests/nd_geometry/` or move to `tests/playground/` and drop from collection.
- **`tests/effq/point_line_duality.py`** — **OUTDATED** as a test (it is exploratory). File name does not match the `test_*.py` pytest convention, so it is not collected; recommend relocating to `tests/playground/` to make that explicit.

### Valid tests (current and meaningful)

- `tests/drift/test_drifter.py`, `test_drift.py`, `test_absorb.py`, `test_diffuse.py`, `test_transport.py` — **VALID**. Cover the Drift* family in depth.
- `tests/effq/test_effq.py` — **VALID**. `qmodel=` kwarg usage at line 78 is against `eval_qmodel`, which still accepts that parameter (`steps.py:653`). Not stale.
- `tests/effq/test_depos.py`, `test_raster_dtype.py`, `test_grid.py`, `test_raster_steps.py` — **VALID**. The latter already uses the new method name `_head_time_offset_from_tail`.
- `tests/sparse/test_sparse.py`, `test_blocking.py`, `test_chunking.py`, `test_indexing.py` — **VALID**. Import `ChunkSum` from `tred.graph` for the sparse assertions.
- `tests/convo/test_convo.py`, `test_partitioning.py` — **VALID**.
- `tests/test_blocking.py`, `test_bsb.py`, `test_chunking.py`, `test_convo.py`, `test_indexing.py`, `test_partitioning.py`, `test_sparse.py`, `test_util.py`, `test_steploader.py` — **VALID**.

### Coverage gaps

Per-module status against the Part A inventory:

| Module | Direct tests? | Gap |
|---|---|---|
| `Drifter` | Yes — strong coverage of forward / init / tshift / drtoa / fluctuate / negative velocity | `_order_step_endpoints_for_drift_time` not tested as a standalone static method on mixed-velocity batches |
| `Raster` | Partial — `_transform`, `_head_time_offset_from_tail`, dtype smoke | Drift-axis sigma-transform invariant (BUG-11 fix) on the depo branch is not directly asserted |
| `ChunkSum` | Partial — only the default method is exercised | `_chunksum2`, `_chunksum_inplace_v1`, `_chunksum_inplace_v2`, invalid-method `ValueError`, and method-equivalence under overlap all untested |
| `LacedConvo` | **None** direct | Only exercised transitively via `tred.convo.interlaced_symm_v2` tests |
| `Charge`, `Current`, `Sim` | **None** direct | Composition correctness not asserted anywhere |
| `raster_steps` shim | Yes | OK |
| `raster.depos.binned` (public wrapper) | **None** direct | Dispatch (1D → `binned_1d`, N-D → `binned_nd`) not asserted |
| `raster.steps.too_short`, `_snap_near_integer`, `_coarsest_float_dtype` | **None** direct | Edge-case behaviors undocumented |

---

## Part D — Recommended new tests

Each item below names the target, what to assert, and why it is worth adding. IDs are unique for tracking.

### Must-have (cover recently-FIXED items; high regression risk)

- **UT-01 `Drifter._order_step_endpoints_for_drift_time`** — target `graph.py:159`. Parametrize `velocity ∈ {+1, −1}` and construct a mixed batch where some rows already satisfy the convention and some do not. Assert: (a) the swap mask matches the expected predicate per row; (b) 1-D `tail`/`head` survive squeeze/unsqueeze round-trip; (c) `ValueError` on shape mismatch. Rationale: this is the single highest-risk change in the delta (rename + semantic inversion — BUG-05).
- **UT-02 `Raster.forward` depo-path drift-axis sigma** — target `graph.py:331–337`. Construct a depo with `sigma_spatial[tdim] = 2.0` and `velocity = 0.5`; assert the output Block carries `sigma_time = 4.0` on the drift axis (distance/|velocity|). Rationale: asserts the exact FIXED-in-aa91c37 behavior (BUG-11).
- **UT-03 `Raster` sign-of-velocity invariance** — target `graph.py:304`. For both depos and steps, verify that flipping `velocity` sign and swapping `tail ↔ head` produces an output whose per-pixel sum matches the original within tolerance. Rationale: guards the Drifter-ordering × Raster-dispatch interplay that the rename touched.

### Module contract tests (cover ungapped modules)

- **UT-04 `ChunkSum` method dispatch** — target `graph.py:355–374`. Parametrize `method ∈ {"chunksum", "chunksum2", "chunksum_inplace_v1", "chunksum_inplace_v2"}`. Assert numerical equivalence across methods on a synthetic Block with deliberate overlap; `ValueError` on unknown method; `nbatches` monotonic under merge. Rationale: three of four code paths currently untested.
- **UT-05 `LacedConvo` forward contract** — target `graph.py:469–487`. Feed a delta-like signal and identity-like response; assert the output equals a hand-computed `interlaced_symm_v2` call over the same inputs. Rationale: zero direct coverage today.
- **UT-06 `Charge` composition** — target `graph.py:490`. Replace `drifter`, `raster`, `chunksum` with mocks returning known tensors; assert call order, arg forwarding, and return type. Rationale: guards against accidental re-ordering during refactors.
- **UT-07 `Sim` smoke** — target `graph.py:515`. Minimal depo + step batch; assert output is non-empty, finite, and shape-consistent. Rationale: contract before wrapping more pipeline ops in modules.

### Raster/effq follow-ups (fill documented gaps)

- **UT-08 `raster.depos.binned` (public wrapper)** — target `depos.py:260`. Assert dispatch to `binned_1d` vs `binned_nd` by dimensionality; assert pass-through of `nsigma`, `minbins`, `dtype`. Noted as a gap in `tests/effq/README.md`.
- **UT-09 `binned_*` `minbins` behavior** — target `depos.py:65, 142`. With `minbins` larger than the natural 3σ window, assert the output window is extended, the offset shifts correspondingly, and the charge sum is `≤` the input total.
- **UT-10 `raster.steps.too_short`** — target `steps.py:350`. Parametrize step lengths across the threshold; assert the boolean matches `||X1 − X0|| < threshold` row-wise.
- **UT-11 `_snap_near_integer` / `_coarsest_float_dtype`** — target `steps.py:49 / 28`. For `compute_index` inputs within `1e-6` of an integer, assert the output is snapped in both fp32 and fp64; assert `_coarsest_float_dtype` returns the lower-precision dtype when mixing fp32/fp64 tensors.

### Raise signal-vs-noise of existing files

- **UT-12 `tests/convo/test_response.py`** — replace plots with a numerical check that the loaded response tensor has the expected shape `(90, 90, 6400)` and dtype, and that `partitioning.deinterlace` then `interlace` reconstructs it bit-exactly.
- **UT-13 `tests/raster/test_raster_graph.py`** — numeric replacement: for `velocity ∈ {±v}`, assert the time-axis ordering of the output grid is monotonically increasing, and that `time_head = time_tail + (tail − head) / velocity` on the drift axis. Promotes the current plot-only file into an assertion.

### Prerequisites for the module consolidations in Part B

These are needed before the MOD-1…MOD-6 modules land so that the refactor is safe:

- **UT-14 `concatenate_waveforms` (graph_effq.py:95–155)** — extract as a fixture and test with two pixels at two different event-time offsets: assert correct per-pixel waveform ordering, that samples before `event_t` are zero, and that `twindow_max` truncates as expected. Required before **MOD-4**.
- **UT-15 TPC-context Drifter idempotence (graph_effq.py:278–282)** — assert that constructing `Drifter(diffusion, lifetime, tpc.drift*velocity, target=tpc.anode)` twice yields equivalent outputs on identical inputs. Required before **MOD-6** turns the loop into a module.
- **UT-16 `SigmaFloor` policy extraction** — assert the per-axis `max(sigma, min_sigma)` clamp never reduces sigma and is applied independently per axis. Required before **MOD-2**.

---

## Summary

- **Invalid tests: none** — the rename delta was applied to tests in the same series.
- **Outdated tests: 4** — `tests/raster/test_raster_graph.py`, `tests/convo/test_response.py`, `tests/nd_loader/test_io_dataset.py`, `tests/effq/point_line_duality.py` (naming only).
- **Coverage gaps: highest priority** on `LacedConvo`, `Charge`, `Current`, `Sim`, `ChunkSum` non-default methods, and the depo-path sigma fix in `Raster.forward`.
- **Recommended new tests:** UT-01…UT-16, with UT-01/UT-02/UT-03 being the highest regression-risk items to land first.
- **Module consolidations (up to readout):** MOD-1 Recombination, MOD-2 SigmaFloor, MOD-3 BlockAccumulator, MOD-4 WaveformAssembler, MOD-5 Readout, MOD-6 TPCSim — each a pure rearrangement of existing code with no algorithmic change.

### Suggested verification commands (do not auto-run)

- `uv run pytest tests -q` — baseline on current HEAD.
- `uv run pytest tests/drift tests/effq tests/sparse -q` — focused run once UT-01…UT-11 land.
- `grep -n "_ensure_tail_closer_to_anode\|_time_diff\|qmodel=" tests/` — should continue to return no stale refs (sanity check for future refactors).

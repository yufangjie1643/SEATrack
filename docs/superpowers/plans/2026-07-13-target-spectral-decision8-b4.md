# Decision 8-B4 Native-Latent Target-Spectral Routing Master Roadmap

> **NON-EXECUTABLE MASTER ROADMAP:** Do not implement directly from this file. Decision 8 fixes the B4 coordinate and binding research gates, but each work package requires a separately reviewed child implementation plan. The current executable child is [`2026-07-13-target-spectral-a0-causal-config.md`](2026-07-13-target-spectral-a0-causal-config.md); B4 memory/routing code remains blocked until its own child plan is approved.

**Goal:** Build and falsify a strictly causal, evaluation-only B4 controller in SEATrack's exact four-dimensional LoRP latent coordinate, modifying only search HMoE logits while all tracker parameters remain frozen.

**Architecture:** B4 stores exact `4 x 4` uncentered second moments of post-`drop1` latent rows and applies a fixed `tau=1` continuous spectral filter. The latent residual is fused as `delta_z @ (B_s @ G)` into block 5/9 attention and FFN search logits; one pre-frame snapshot is shared by all four calls and detached writes commit only after prediction. A8 is the mandatory post-H rank-8 control/fallback, B5 is a bias ablation, and C768 is a closed diagnostic upper bound.

**Tech Stack:** Python 3.12, PyTorch 2.12, NumPy, `unittest`, EasyDict/YAML, SEATrack ViT-B/HMoE, LasHeR, and DepthTrack.

**Decision Status:** The B4 coordinate was user-approved on 2026-07-13. This roadmap itself was not approved as executable after expert review; its unchecked work packages are design inventory, not coding authority.

Child-plan gates are sequential: A0 causal/config prerequisites may execute now; padding/checkpoint, exact B4 memory, observation/routing integration, matched controls, and statistical evaluation each require their own fresh plan and review. In particular, no B4 core child may start until it restates the complete `delta_z` equation and frozen coefficients/admission/bounds; no experiment child may start until it uniquely defines A8, B5, C768, scalar history, data support, RNG derivation, and runnable commands.

## Global Constraints

- The superseded spec/plan are historical protocol references. Only a named, reviewed child plan may authorize edits; no unchecked package below authorizes implementation.
- Work only in the isolated `target-spectral-a` worktree. Preserve unrelated changes, commit task-local files only, and do not push without an explicit request.
- All model parameters remain `requires_grad=False`; the model, LoRP, and both LoRP dropouts remain in evaluation mode. Active routing fails closed if this contract or `linear1.if_act=False`/`nn.Identity` is violated.
- B4 coordinate: `z = drop1(act(norm(x) @ lora_a + bias_a))`, shape `[B,N,4]`.
- Persist a complete symmetric float64 `[4,4]` uncentered second moment. Never sketch it or hard-select rank 1–4.
- With eigenpairs `(U,lambda)`, use `Pi=U diag(lambda/(lambda+tau*mean(lambda))) U^T`, with `tau` exactly `1.0`. Epsilon only guards zero/nonfinite trace; it is never added to a positive-trace denominator.
- Active routing is search-only at blocks `[5,9]`, attention and FFN. Template routing stays legacy; initialization may run one routing-disabled observer forward.
- Disabled, empty, and `strength=0` branches bypass latent capture, operator work, residual arithmetic, and diagnostics before the original HMoE path; evaluation output must be bitwise legacy.
- Active routing changes logits only through fused latent deltas. Raw `H`, axes, dense expert execution, and expert parameters remain legacy, but changed Dispatch is expected to change numerical `Dispatch^T H` expert inputs.
- Every frame uses one snapshot committed through `t-1`; current observations become visible only after frame `t` prediction/state commit.
- Post-initialization calls receive only `frame_index` and sanitized previous predictions, never GT, visibility, validity, corruption, attribute, or future-frame data. VOT restart begins a new episode.
- Target-spectral config validation requires `len(CE_KEEP_RATIO)==len(CE_LOC)` and every value exactly `1.0`. This guard belongs only to the target-spectral validator.
- Formal method/config cells run in fresh processes. In-process config objects must also be deep-cloned to prevent the observed `rgbt_lifttrack_pilot -> rgbt` BiLift leak.
- Explicit checkpoint metadata declares LoRA weights `merged` or `unmerged`; no Decision 8 loader may infer merge state.
- A8 is mandatory. B5 cannot replace B4. C768 may run only after a recorded B4-versus-A8 latent-sufficiency failure and cannot rescue a B4 claim.
- Passing gates is not a performance guarantee; claims remain conditional on frozen checkpoints, splits, schedules, coefficients, code, and hardware.
- Use `PYTHON=/home/yufan/code/SEATrack/.venv/bin/python` and `unittest`; the worktree intentionally has no private virtualenv. For every behavior, write a focused failing test, observe the intended failure, implement minimally, rerun it, then run affected regression tests.

---

## Frozen Mathematics and Interfaces

Split `linear1.lora_b=[B_0|B_1]`, where `B_s:[4,384]`, and let `G=gate_thi:[384,2E]`. Legacy `H_s=zB_s+b_s` is unchanged. For family `j`:

```text
n' = beta*n + q*sum_i(w_i)
C' = (beta*n*C + q*sum_i(w_i*z_i^T*z_i)) / n'
mean_lambda = trace(C') / 4
Pi = U diag(lambda/(lambda+mean_lambda)) U^T
delta_L_s = delta_z (B_s G)
```

`delta_z` combines frozen `(identity,dynamic,private,background)` coefficients with the causal target prior, opposite RGB/X private sign, committed confidence, operator cap, and strength. Float64 fused/direct logits must agree within `1e-12`; float32 within `1e-6`.

New production files:

- `lib/models/target_spectral/{__init__,config,types,memory,routing,observation,stage0,controls}.py`
- `lib/models/seatrack/checkpoint.py`
- `lib/test/evaluation/{causal,spectral_geometry,spectral_statistics}.py`
- `tracking/analyze_decision8_b4.py`
- `experiments/seatrack/{rgbt_decision8_b4,rgbd_decision8_b4}.yaml`
- `experiments/seatrack/registries/decision8_b4.calibration.yaml`

New tests:

- `tests/test_spectral_causality.py`
- `tests/test_config_isolation.py`
- `tests/test_spectral_input_contract.py`
- `tests/test_lora_checkpoint_state.py`
- `tests/test_b4_memory.py`
- `tests/test_b4_routing.py`
- `tests/test_b4_stage0.py`
- `tests/test_b4_controls_and_gates.py`

### Task 1: Enforce the Real Causal Evaluator and Episode Lifecycle

**Files:** Create `lib/test/evaluation/causal.py`, `tests/test_spectral_causality.py`; modify `lib/test/evaluation/tracker.py`, `lib/test/tracker/basetracker.py`, `lib/test/tracker/seatrack.py`, `lib/test/tracker/ostrack.py`, `lib/test/vot/seatrack_class.py`.

**Interfaces:** `CausalFrameRecord.from_evaluator(frame_index,previous_output)`, `CausalFrameRecord.as_tracker_info()`, `call_causal_track(tracker,image,record)`, and `BaseTracker.begin_episode(reset_global=True)`.

- [ ] **Step 1: Write RED tests against production entry points**

Test the real `Tracker.run_sequence`/`Tracker._track_sequence`, a GUI-free frame helper called by real `run_video`, and the real VOT adapter instantiated via `__new__`. Reproduce the one-argument `create_tracker` TypeError and old positional binding. A recorder matching the old SEATrack signature must observe `dataset_name is None` and causal `info`; a sentinel must raise on any post-init `seq.frame_info` or GT access. Test explicit versus fallback begin, VOT restart, and monotonically increasing frame indices.

Schema tests allow exactly root keys `frame_index/previous_output` and prediction keys `target_bbox/best_score/all_boxes/all_scores`. Require positive non-boolean integer frame indices and finite, detached numeric values. Unknown keys, invalid types/shapes, nonfinite data, and tensors requiring gradients raise explicit `TypeError` or `ValueError`; security checks must not use `assert`. The sanitizer checks membership by iterating the four allowed names, so it never reads a forbidden mapping value.

- [ ] **Step 2: Run and observe RED**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_spectral_causality -v
```

Expected: missing causal module plus real lifecycle, binding, GT, video, and VOT failures.

- [ ] **Step 3: Implement the minimal lifecycle contract**

Make `Tracker.create_tracker(self,params,mode=None)`. `call_causal_track` contains the literal keyword call `tracker.track(image, info=record.as_tracker_info())`; all official paths use it. OPE never calls `seq.frame_info` after initialization. OPE/video/VOT initialize or restart with exactly one `begin_episode(True)`, reset the entry point's local frame index to zero, and begin tracking at one.

`BaseTracker.begin_episode` only marks initialization pending. SEATrack/OSTrack overrides clear object-level legacy state without loading the network or changing weights. `initialize` performs one fallback begin only for an unprepared external call and consumes the flag. Canonical track signatures become `track(self,image,info=None)` and validate before crop/model work. Remove tracker-side OSTrack GT overlay. Explicitly reject multi-object or mid-sequence initialization in this single-object path.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_spectral_causality -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git add lib/test/evaluation/causal.py lib/test/evaluation/tracker.py lib/test/tracker \
  lib/test/vot/seatrack_class.py tests/test_spectral_causality.py
git commit -m "fix: enforce causal tracker lifecycle"
```

### Task 2: Isolate Every Evaluation Configuration Load

**Files:** Modify `lib/config/seatrack/config.py`, `lib/test/parameter/seatrack.py`; create `tests/test_config_isolation.py`.

**Interfaces:** module-initialization `DEFAULT_CFG`, `clone_default_cfg()->edict`, and `load_config(path)->edict`.

- [ ] **Step 1: Write and observe the real RED leak**

In one process call real `parameters("rgbt_lifttrack_pilot")` then `parameters("rgbt")`. Require distinct recursive objects; the first has BiLift enabled, while the second and pristine clone do not. Mutating a returned nested list must not affect the other result or pristine defaults. Load RGB-T/RGB-D in both orders and require base CE ratios `[1,1,1]`.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_config_isolation -v
```

Expected: current singleton aliases both results and leaves BiLift enabled.

- [ ] **Step 2: Implement a pristine snapshot**

After constructing all defaults, set `DEFAULT_CFG=copy.deepcopy(cfg)`. `clone_default_cfg` returns `copy.deepcopy(DEFAULT_CFG)`; it must never copy possibly polluted current `cfg`. `load_config(path)` updates only that fresh clone through the existing explicit `base_cfg` argument. Evaluation `parameters()` assigns the local result and never calls the global-mutating fallback. Keep legacy `cfg` for training compatibility.

- [ ] **Step 3: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_config_isolation -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift_integration -v
git add lib/config/seatrack/config.py lib/test/parameter/seatrack.py tests/test_config_isolation.py
git commit -m "fix: isolate SEATrack evaluation configs"
```

### Task 3: Preserve Padding Validity and Fail Closed on CE

**Files:** Create `lib/models/target_spectral/config.py`, `tests/test_spectral_input_contract.py`; modify `lib/test/tracker/data_utils.py`, `lib/test/tracker/seatrack.py`, `lib/models/seatrack/seatrack.py`, `lib/models/seatrack/vit_ci.py`.

**Interfaces:** `PreprocessorMM.process(image,mask)->(image_tensor,mask_tensor)` and `validate_target_spectral_config(cfg)`.

- [ ] **Step 1: Write and observe RED input-contract tests**

Prove the exact template/search padding masks reach `VisionTransformerCE.forward_features(mask_z=...,mask_x=...)`. Observer exclusion is deferred to the later observation/controller child plan, where the production observer seam exists. Target-spectral configurations with a CE length mismatch or any value other than exact `1.0` raise `ValueError`; active GRA/BiLift routing modifiers also fail closed. Legacy configs without target spectral remain unaffected.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_spectral_input_contract -v
```

Expected: PreprocessorMM drops masks, outer calls omit them, and no validator exists.

- [ ] **Step 2: Thread masks without changing pixel/model arithmetic**

Match `PreprocessorX` mask conversion. Add optional `template_mask/search_mask` keywords through tracker/model to the existing `forward_features(mask_z,mask_x)` path. Do not change normalization, token order, or all-valid attention. The target-spectral validator alone enforces equal CE list lengths and `type(value) in {int,float} and float(value)==1.0`; do not repair nontrivial CE in this workstream.

- [ ] **Step 3: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_spectral_input_contract -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git add lib/models/target_spectral/config.py lib/models/seatrack lib/test/tracker \
  tests/test_spectral_input_contract.py
git commit -m "fix: preserve spectral input validity"
```

### Task 4: Make LoRA Checkpoint Merge State Explicit

**Files:** Create `lib/models/seatrack/checkpoint.py`, `tests/test_lora_checkpoint_state.py`; modify `lib/test/tracker/seatrack.py`, `lib/train/trainers/base_trainer.py`.

**Interfaces:** `load_seatrack_checkpoint(model,path,explicit_state=None,require_metadata=False)` and root metadata `lora_weight_state`.

- [ ] **Step 1: Write and observe RED checkpoint tests**

Use a tiny nonzero `MergedLinear` model. Save/load both merged and unmerged states and require output parity; missing metadata fails when required, explicit-state mismatch fails, mixed module states cannot be saved, and legacy loading works only with an operator-supplied state. Prove the tracker no longer guesses with `merged=True -> train() -> eval()`.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_lora_checkpoint_state -v
```

Expected: loader missing and tracker heuristic present.

- [ ] **Step 2: Implement strict metadata load/save**

Resolve and validate the closed `merged/unmerged` enum before strict state loading. Set each `MergedLinear.merged` flag to the declared stored-weight state without mode toggles, then call model eval. `BaseTrainer.save_checkpoint` rejects mixed LoRA module states and writes the common state beside `net`; a model without MergedLinear records `not_applicable`, which is invalid for a LoRA-bearing model.

- [ ] **Step 3: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_lora_checkpoint_state -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git add lib/models/seatrack/checkpoint.py lib/test/tracker/seatrack.py \
  lib/train/trainers/base_trainer.py tests/test_lora_checkpoint_state.py
git commit -m "fix: persist LoRA checkpoint state"
```

### Task 5: Implement Exact B4 State and Continuous Filter

**Files:** Create `lib/models/target_spectral/{__init__,types,memory}.py`, `tests/test_b4_memory.py`.

**Interfaces:** immutable `B4MomentState(moment:[4,4],effective_mass,commits)`, `update_b4_moment(state,factor_chunks,admission,beta)`, and `continuous_b4_operator(state,tau=1.0,zero_trace_eps=1e-12)->B4Operator`.

- [ ] **Step 1: Write and run RED tests**

Test a weighted `eye(4)` update equals `eye(4)/4`, has matrix rank 4, and exposes no persistent `rank/basis/sketch/eigengap`. Test rejected writes return the identical object, one decay per frame across aggregated factor chunks, microbatch partition equivalence, symmetry/PSD, and nonfinite/negative rejection. Atomic multi-family prepare/commit failure belongs to the controller child plan. For diagonal `[4,2,1,0]`, require weights `lambda/(lambda+1.75)` exactly; zero trace alone yields inactive.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_b4_memory -v
```

Expected: missing B4 package.

- [ ] **Step 2: Implement exact float64 update and filter**

```python
import math

def update_b4_moment(state, factor_chunks, admission, beta):
    q, decay = float(admission), float(beta)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("admission must be finite in [0,1]")
    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError("beta must be finite in [0,1]")
    chunks = tuple(factor_chunks)
    if q == 0.0 or not chunks:
        return state
    observed_mass = state.moment.new_zeros(())
    observed_outer = state.moment.new_zeros((4, 4))
    for rows, weights in chunks:
        z = rows.detach().to(device=state.moment.device, dtype=torch.float64)
        w = weights.detach().to(device=z.device, dtype=torch.float64)
        if z.ndim != 2 or z.shape[1] != 4 or w.shape != z.shape[:1]:
            raise ValueError("invalid B4 observation shape")
        if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError("invalid B4 observation")
        observed_mass = observed_mass + w.sum()
        zw = z * w.sqrt().unsqueeze(1)
        observed_outer = observed_outer + zw.T @ zw
    if float(observed_mass) == 0.0:
        return state
    mass = decay * state.effective_mass + q * observed_mass
    numerator = decay * state.effective_mass * state.moment + q * observed_outer
    moment = numerator / mass
    return B4MomentState(0.5 * (moment + moment.T), mass, state.commits + 1)

def continuous_b4_operator(state, tau=1.0, zero_trace_eps=1e-12):
    if float(tau) != 1.0:
        raise ValueError("Decision 8 fixes tau=1")
    trace = torch.trace(state.moment)
    if not bool(torch.isfinite(trace)) or float(trace) <= float(zero_trace_eps):
        return B4Operator.inactive(state.moment)
    values, vectors = torch.linalg.eigh(state.moment)
    values = values.clamp_min(0.0)
    weights = values / (values + values.mean())
    return B4Operator((vectors * weights.unsqueeze(0)) @ vectors.T, True)
```

Do not threshold individual eigenvalues or add epsilon to the positive denominator.

- [ ] **Step 3: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_b4_memory -v
git add lib/models/target_spectral tests/test_b4_memory.py
git commit -m "feat: add exact B4 spectral state"
```

### Task 6: Implement Fused Routing and Causal Stage 0 Integration

**Files:** Create `routing.py`, `observation.py`, `stage0.py`, `tests/test_b4_routing.py`, `tests/test_b4_stage0.py`; modify `attn.py`, `attn_blocks.py`, `vit_ci.py`, `seatrack.py`, and test tracker.

**Interfaces:** `B4RouteCall`, `fused_b4_delta_logits`, immutable `B4Snapshot`, `FrameRouteContext.call_for(block,site,scope)`, and `Stage0Controller.before_frame/stage_observation/after_prediction`.

- [ ] **Step 1: Write and run RED tests**

Require `strength=0` bitwise legacy with zero observer/operator calls; active calls fail closed in train/dropout/activation/trainable-parameter states. Prove fused/direct delta agreement, raw-H byte equality, changed Dispatch and expert inputs, template/other-layer bypass, one snapshot across four calls, `t` write visibility only at `t+1`, atomic stale-write rejection, padding exclusion, and disabled legacy forward counts.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_b4_routing tests.test_b4_stage0 -v
```

Expected: missing routing/controller interfaces.

- [ ] **Step 2: Expose the existing latent and fuse only delta logits**

Extend `LoRP.forward(x,return_latent=False)` without changing operation order. The zero/disabled branch calls the original `linear1(norm(x))`. Active eval calls return `(expanded,z)`. In the function below, `lora_b_t` is explicitly `linear1.lora_b.weight.T` with algebraic shape `[4,768]`; never pass the PyTorch weight `[768,4]` silently.

```python
def fused_b4_delta_logits(delta_z, lora_b_t, gate):
    if delta_z.shape[-1] != 4 or lora_b_t.shape != (4, 768) or gate.shape[0] != 384:
        raise ValueError("invalid B4 fusion geometry")
    b0, b1 = lora_b_t.split(384, dim=1)
    slot0, slot1 = delta_z @ (b0 @ gate), delta_z @ (b1 @ gate)
    return torch.stack((slot0, slot1), dim=2).reshape(
        delta_z.shape[0], 2 * delta_z.shape[1], gate.shape[1]
    )
```

Bound the fused residual before Dispatch/Combine softmax. Continue computing expert inputs from unchanged raw H. Never materialize routed H as the expert source.

- [ ] **Step 3: Implement observation and causal controller**

For paired valid RGB/X latent rows, create target-weighted identity `(rgb+x)/sqrt(2)`, private `(rgb-x)/sqrt(2)`, dynamic common-minus-trusted-mean, and hard-background common chunks. Normalize target/background masses separately; detach all tensors and store no images/crops.

Initialization performs one observer-only legacy calibration forward. Each real frame clones one snapshot; only `(5,attn/search)`, `(5,ffn/search)`, `(9,attn/search)`, `(9,ffn/search)` receive active calls. `after_prediction` verifies public bbox/score commit, prepares all family states, then swaps once with `version+1`; errors preserve old bytes. No current score influences current logits.

The target-spectral tracker path calls `network.requires_grad_(False)` and `network.eval()` once before attaching the controller. Attachment and every active call explicitly reject any trainable parameter, training-mode module, active LoRP dropout, or non-Identity activation; ordinary routing-disabled tracker construction is unchanged.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_b4_memory tests.test_b4_routing tests.test_b4_stage0 -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git add lib/models lib/test/tracker/seatrack.py tests/test_b4_routing.py tests/test_b4_stage0.py
git commit -m "feat: add causal B4 routing controller"
```

### Task 7: Preregister Controls and Binding Stop Gates

**Files:** Create `controls.py`, `spectral_geometry.py`, `spectral_statistics.py`, configs, registry, analyzer, and `tests/test_b4_controls_and_gates.py`.

**Interfaces:** closed coordinates `a8_post_h`, `b4_native_latent`, `b5_homogeneous_bias`, `c768_pre_lorp`; deterministic sequence-bootstrap report.

- [ ] **Step 1: Write and run RED tests**

Assert A8 width 384/rank 8 and mandatory status; B4 width 4/no retained rank; B5 exact `[z,1]` 5x5 bias route and ablation-only status; C768 refuses to build unless `b4_latent_sufficiency_failed=true`. Test every numeric boundary below, missing-cell failure, deterministic 10,000-resample bytes, and strict `>-0.3 pp` behavior.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_b4_controls_and_gates -v
```

Expected: missing controls/gate modules.

- [ ] **Step 2: Implement controls and exact geometry definitions**

Set `K=[B_0G,B_1G]`. Define:

```python
def anisotropy(pi):
    iso = torch.trace(pi) * torch.eye(4, dtype=pi.dtype, device=pi.device) / 4
    return float(torch.linalg.matrix_norm(pi - iso, ord="fro") /
                 torch.linalg.matrix_norm(pi, ord="fro"))

def relative_rank(matrix, floor):
    singular = torch.linalg.svdvals(matrix.to(torch.float64))
    return 0 if float(singular[0]) == 0 else int((singular / singular[0] >= floor).sum())

def route_visibility(moment, k):
    values, vectors = torch.linalg.eigh(moment.to(torch.float64))
    sqrt_c = (vectors * values.clamp_min(0).sqrt().unsqueeze(0)) @ vectors.T
    visible = sqrt_c @ k.to(torch.float64)
    energy = visible.square().sum() / (torch.trace(moment) * torch.linalg.svdvals(k)[0].square())
    return relative_rank(visible, 1e-2), float(energy)
```

Family distance is the minimum pairwise Frobenius distance between `Pi_j/trace(Pi_j)`; zero/nonfinite traces fail the cell. Average frames within sequence, sort manifest IDs, then bootstrap sequences 10,000 times with frozen `PCG64` seed. One-sided 95% LCB uses quantile `.05`; one-sided 97.5% uses `.025`.

- [ ] **Step 3: Freeze all binding gates in the registry**

- Every active `(seed,benchmark,block,site,family)` anisotropy LCB is `>=0.10`.
- Every checkpoint/block/site has `relative_rank(K,1e-3)==4`.
- On admitted active frames, `relative_rank(C_j^(1/2)K,1e-2)>=2` for identity/private/background and `>=1` for dynamic, with coverage `>=0.80` in every active cell.
- Every active-cell route-visible energy LCB is `>=0.05`.
- Every block/site minimum normalized family-distance LCB is `>=0.05`.
- All four strength-matched LOO paired LCB95 values `J_B4-J_minus_family` are strictly positive.
- Calibration `LCB97.5(J_B4-J_A8)>-0.3 pp`; failure records the C768 trigger and stops B4 confirmation.
- Separately on LasHeR and DepthTrack, both `LCB97.5(J_B4-J_confidence_only_scalar_history)>-0.3 pp` and `LCB97.5(J_B4-J_routing_disabled_legacy)>-0.3 pp`.

Missing/nonfinite cells fail. B5/C768 cannot satisfy a failed B4 gate.

- [ ] **Step 4: Freeze configs/analyzer and verify GREEN**

Configs set coordinate B4, layers `[5,9]`, strength `1.0`, tau `1.0`, CE ratios `[1,1,1]`, explicit checkpoint state, and mutually exclusive GRA/BiLift. The analyzer verifies parent hashes, refuses overwrite, recomputes every gate, and emits canonical JSON with all thresholds/LCBs/pass bits.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_b4_controls_and_gates -v
/home/yufan/code/SEATrack/.venv/bin/python tracking/analyze_decision8_b4.py --help
(cd /tmp && /home/yufan/code/SEATrack/.venv/bin/python /home/yufan/code/SEATrack-ProbAlign-VRE/.worktrees/target-spectral-a/tracking/analyze_decision8_b4.py --help)
git add lib/models/target_spectral/controls.py lib/test/evaluation tracking experiments tests/test_b4_controls_and_gates.py
git commit -m "test: freeze Decision 8 controls and gates"
```

### Task 8: Execute the Frozen Decision Sequence

**Files:** Modify frozen registry parents; create `knowledge_base/Decision-8-B4-verification.json`.

- [ ] **Step 1: Verify before outcomes**

```bash
git status --short
git diff --check HEAD
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
```

Expected: clean tree and passing suite. Commit checkpoint/split/schedule/coefficient/evaluator/config/code hashes before opening outcomes.

- [ ] **Step 2: Run calibration in fresh processes**

Run B4 and mandatory A8 on identical frozen calibration inputs. Enforce geometry and `B4-A8` gates first. Stop before confirmation on any failure; C768 remains diagnostic-only after a latent-sufficiency failure.

- [ ] **Step 3: Run attribution and clean noninferiority**

After calibration passes, run B5 and four strength-matched LOO rows without retuning, then fresh-process B4/scalar-history/legacy rows on LasHeR and DepthTrack. Stop the corresponding claim on any failed gate.

- [ ] **Step 4: Write immutable evidence and final verification**

The canonical report records all hashes, B4 exact-state/tau checks, strength-zero bitwise hash, eval/frozen contract, raw-H equality plus expert-input change, every geometry cell, LOO LCBs, B4-A8 LCB, four clean noninferiority LCBs, B5 as non-rescue, C768 as closed/diagnostic, and the overall conjunction.

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m json.tool knowledge_base/Decision-8-B4-verification.json >/dev/null
git diff --check -- knowledge_base/Decision-8-B4-verification.json
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git add -f knowledge_base/Decision-8-B4-verification.json
git commit -m "docs: record Decision 8 B4 verification"
```

## Plan Self-Review

- [ ] Every requested frozen B4 value and stop gate maps to an explicit task/test.
- [ ] Real `_track_sequence`, video helper, VOT adapter, production parameter loader, and the original create-tracker TypeError are tested; no helper-only lifecycle test can pass.
- [ ] B4 never claims numerical expert inputs or tracking performance are preserved.
- [ ] Old unlisted implementation steps are explicitly non-executable.

```bash
! rg -n "T[B]D|T[O]DO|implement lat[e]r|fill in detail[s]|Similar to Tas[k]" docs/superpowers/plans/2026-07-13-target-spectral-decision8-b4.md
rg -n "SUPERSEDED FOR IMPLEMENTATION" docs/superpowers/plans/2026-07-13-target-spectral-workstream-a.md docs/superpowers/specs/2026-07-13-rgbx-target-spectral-continual-moe-lora-design.md
git add -N docs/superpowers/plans/2026-07-13-target-spectral-decision8-b4.md
git diff --check -- docs/superpowers/plans/2026-07-13-target-spectral-decision8-b4.md docs/superpowers/plans/2026-07-13-target-spectral-workstream-a.md docs/superpowers/specs/2026-07-13-rgbx-target-spectral-continual-moe-lora-design.md
git reset -- docs/superpowers/plans/2026-07-13-target-spectral-decision8-b4.md
```

Expected: all commands exit zero; the placeholder scan is empty; only these three documents changed during plan authoring.

# Standalone ADCS actor PTQ + Rust CPU inference (ADCSMLP1)

Designed from the **B4 (Oct 2026) source archive**. This package is **independent** of
`adcs_flightlike_b1`: calibration, export, NumPy inference, benchmarking and Rust inference import
**no JAX/Haiku/simulator**. Extraction from `.npz` also needs only NumPy;
loading an original `.pkl` checkpoint may need Optax/JAX modules installed,
depending on what classes its serialized optimizer state references.
Only the optional bridge for collecting actual B4 closed-loop observations and
closed-loop evaluation imports the B4 simulator/JAX. It also works with older
checkpoints following the same Haiku module/actor contract.

## What it contains

```
adcs_embed/core.py         actor extraction, FP32 reference, PTQ, inference
adcs_embed/binary.py       ADCSMLP1 versioned binary writer/reader and CRC32
adcs_embed/cli.py          extract, calibrate, export, benchmark, test vectors
adcs_embed/prepare.py      balanced multi-scenario calibration dataset merger
adcs_embed/b4_bridge.py    optional JAX B4 closed-loop collector and evaluator
adcs_embed/verify_rust.py  cross-language output comparison harness
rust/actor_service/       Rust CPU kernel + JSONL service + microbenchmark
examples/demo.py          synthetic untrained model for smoke tests
tests/                    NumPy/serialization/regression checks
```

## Actor contract recovered from B4

B4's `network.py` has `actor_hidden_0..N`, `policy_mean` Haiku modules.
`ppo.deterministic_action()` applies an **additional** `tanh` after the network.
This package deploys **actor only**, using:

```
x = env.observe(state)              # already normalized, ordered observations
for each actor_hidden_i:
    x = activation(x @ W_i + b_i)    # tanh or ReLU
raw = x @ W_output + b_output
mean = mean_limit * tanh(raw / mean_limit)
policy_action = tanh(mean)          # output of THIS Rust service
```

The actor input/output dimensions are extracted from weights. The archive's
observation components are `3 attitude-vector + 3 rate error + 4 measured wheel
speed`, optionally previous normalized action, FDIR wheel authority, estimator
confidence and supervisor one-hot. **Do not send raw sensors or physical units**
to Rust: the observation contract, estimator and modes remain external.

B4 has 3- or 4-output policy depending on the control mode. `residual_pd`
needs **additional nominal PD + scaled residual** in `control.policy_to_command`
*after* inference. `motor_direct` and `direct` clip policy action directly.
The Rust service returns **policy_action**, not direct wheel torque.

## Prerequisites

Python 3.10+, NumPy; optional pytest/matplotlib. Rust stable Cargo (edition 2021)
for the separate inference service (only serde and serde_json as dependencies,
no ONNX/JAX/ML runtime). Installation:

```bash
cd adcs_actor_embedded
python -m pip install -e '.[tests,charts]'
python -m pytest -q tests
cd rust/actor_service && cargo test --release && cargo build --release
```

The Rust binary is `rust/actor_service/target/release/adcs-actor-service`
(on Windows add `.exe`).

### 1. Extract actor from *trusted* checkpoint

Run from a working directory with the package importable:

```bash
python -m adcs_embed extract /path/to/B4/checkpoints/final.pkl --out actor.npz
```

`pickle` can execute arbitrary code; **only deserialize trusted checkpoints**.
It expects the B4 format (`params`, `config.network.hidden_sizes`, `activation`,
`policy_mean_limit`) and rejects missing/ambiguous modules. A portable actor-only
`actor.npz` is produced, loadable without JAX/Haiku.

### 2. Prepare representative FP32 closed-loop calibration observations

Run these from the *parent directory containing* the `adcs_flightlike_b1` package,
with `adcs_actor_embedded` installed and B4's JAX/Haiku dependencies present:

```bash
python -m adcs_embed.b4_bridge collect /path/to/final.pkl \
  --project adcs_flightlike_b1 --episodes 64 --seed 20001 \
  --max-samples 100000 --out datasets/nominal_cal.npz
```

The bridge runs the checkpoint's deterministic **FP32** policy in its original
MEKF/FDIR/environment feedback loop, records `env.observe(state)` at each
control step, checks actor-only extraction versus the original Haiku action,
and stores a compressed `observations` float32 matrix.

For multiple scenarios, separately collect using respective *compatible* B4
scenario configuration/checkpoints and separate random seeds. Use only models
with **identical observation semantics** (dimensions alone do not prove
compatibility). Example balancing:

```bash
python -m adcs_embed.prepare datasets/nominal_cal.npz \
  datasets/wheel_fault_cal.npz datasets/large_slew_cal.npz \
  --per-source 25000 --seed 23 --out datasets/calibration.npz
```

This balances dataset *rows*, **not guaranteed episode counts**, and does not
modify any observation value. Calibrate each checkpoint's actor with appropriate
flight regime observations. Include sensor/MEKF acquisition, saturated wheel,
and fault-recovery states where they are operationally relevant.

**Hold out complete seeds/episodes for evaluation.** Example held-out run:

```bash
python -m adcs_embed.b4_bridge collect /path/to/final.pkl \
  --episodes 64 --seed 30001 --out datasets/heldout.npz
```

The B4 collector also records an `observation_contract` describing feature
options and scaling. The merger checks contract equality when present, and
calibration binds this contract to the exported metadata. If using older datasets
without this metadata, confirm semantics manually.

Existing recorded data from any simulator may be used with no B4 dependency:
`observations` 2D float32 in `.npz`, plain `.npy`, or numeric CSV.

### 3. Calibrate static activations (PTQ)

```bash
python -m adcs_embed calibrate actor.npz datasets/calibration.npz \
  --percentile 100 --margin 1.05 --out calibration.json
```

- W8A8 uses symmetric **per-tensor input activation scale** `Sx=max_abs/127`
  (or percentile clipping with explicit margin). Calibration requires **only
  representative observations**, no reward and no optimizer.
- Each `tanh` hidden activation has conservative scale `1/127` because it is
  bounded in `[-1,1]` (no calibration needed for this *range choice*).
- ReLU's unbounded hidden activations are calibrated layer by layer from FP32
  hidden outputs.
- Input components may have very different normalization/ranges; this simple
  design deliberately uses a single first-layer input scale to keep all INT8
  operations standard. If precision degrades, investigate calibration strata,
  rescaling or input preprocessing **without changing observation semantics**.
- `--percentile <100` deliberately clips tail values and must be separately
  validated for fault/safety trajectories. Default is absmax + 5% margin.

### 4. Export portable precision variants

```bash
python -m adcs_embed export actor.npz --calibration calibration.json \
  --out exports --variants fp32 fp16 bf16 w8a32 w8a16 w8a8
```

Each `exports/<variant>/` contains:

```
actor.bin     # only file Rust needs (weights, bias, scales, metadata, CRC32)
model.json    # diagnostic metadata; not used in runtime
```

Precision semantics (be precise about what is and isn't true integer execution):

| Variant | Stored weights | Activations/math in standalone implementation |
|---------|----------------|-----------------------------------------------|
| FP32 | f32 | f32 |
| FP16 | IEEE f16 | f16-rounded activation boundaries, f32 accumulation |
| BF16 | BF16 | BF16-rounded activation boundaries, f32 accumulation |
| W8A32 | int8 per output channel | dequantized weights × f32 inputs, f32 sum |
| W8A16 | int8 per output channel | dequantized **FP16** weights × FP16 inputs, f32 sum |
| W8A8 | int8 per output channel | **int8 × int8 → int32**, int32 bias, f32 dequant/tanh |

FP16/BF16 and W8A16 here are **numerical precision experiments**, not claims
of native half-precision CPU ISA acceleration. W8A8 is the true integer-Dense
branch. All variants use a FP32 postprocessing tail
`limit*tanh(raw/limit) -> tanh(mean)`.

For W8A8, each output channel `j` uses `Sw[j]=max_i(abs(W[i,j]))/127` and
`qw[i,j]=clip(rint(W[i,j]/Sw[j]),-127,127)`; a layer input uses `Sx` and
`qx=clip(rint(x/Sx),-127,127)`. Bias is `qb[j]=rint(b[j]/(Sx*Sw[j]))`.
The Dense output is `(sum_i qx[i]*qw[i,j] + qb[j])*(Sx*Sw[j])`.
INT32 overflow feasibility is checked on both export and load.

No pruning or QAT is silently performed. Those are later independent experiments
if the closed-loop tradeoff justifies them.

### 5. Evaluate held-out numerical error + approximate dev-host latency

```bash
python -m adcs_embed benchmark actor.npz datasets/heldout.npz \
  --models exports --out results/offline --repeats 1000
```

Outputs `results/offline.json`, `.csv`, `.png` (if matplotlib installed).
Metrics: binary bytes, action MAE/RMSE/max versus FP32 NumPy, and **development
host Python overhead** timings. These timings are not flight-target latency.

For existing B4 closed-loop metrics with identical initial states/seeds:

```bash
python -m adcs_embed.b4_bridge evaluate /path/to/final.pkl \
  --models exports --episodes 64 --seed 30001 --out results/closed_loop.json
```

This optional step runs the binary-quantized weights through a JAX implementation
inside the original B4 evaluator. It tests policy performance, **not** the
network/IPC execution path of the Rust service. For Rust-in-loop integration,
explicitly couple Rust actions to the spacecraft step and repeat your mission,
fault, and Basilisk coverage. No such integration is included here.

### 6. Run Rust inference service

```bash
cd rust/actor_service
cargo build --release
./target/release/adcs-actor-service --model ../../exports/w8a8/actor.bin --stdio
```

Send **one newline-terminated JSON request per observation**:

```json
{"id":42,"observations":[0.0,0.0,0.0]}
```

The list length **must equal the model input dimension**, not the 3 shown in
this protocol-only illustration. For the real model, obtain length from
`exports/w8a8/model.json` or the service's startup stderr.

Reply on stdout:

```json
{"id":42,"action":[0.0],"mean":[0.0]}
```

Reply dimensions are the actor output size. The example is illustrative only.
Errors produce `{"id":...,"error":"..."}` and do not kill the process.
Model info/logs appear on **stderr**; stdout is pure NDJSON.

Optional local-only TCP interface (newline-delimited JSON, **not HTTP**):

```bash
./target/release/adcs-actor-service --model ../../exports/w8a8/actor.bin \
  --listen 127.0.0.1:7777
```

For an execution-time microbenchmark using preallocated inference workspace:

```bash
./target/release/adcs-actor-service --model ../../exports/w8a8/actor.bin --bench 10000
```

The measurement excludes socket/JSON overhead and reports p50/p99/maximum on
the development host. The runtime allocates its inference scratch once and uses
**no allocation inside `predict_into()`**. Socket/stdio JSON processing does
allocate and is **not hard real-time**. Measure actual worst-case or safe timing
bounds on intended hardware and account for all MEKF/GNC/FDIR tasks.

### 7. Rust ↔ NumPy equivalence tests

```bash
python -m adcs_embed.verify_rust datasets/heldout.npz --models exports \
  --rust-binary rust/actor_service/target/release/adcs-actor-service \
  --max-samples 10000 --atol 3e-4 --out results/rust_equivalence.json
```

Optionally generate inspectable reference vectors:

```bash
python -m adcs_embed vectors datasets/heldout.npz \
  --model exports/w8a8/actor.bin --out results/reference_vectors.jsonl
```

Use a separate B4 checkpoint extraction check to establish:

```
original Haiku FP32 -> NumPy actor FP32 -> binary FP32 -> Rust FP32
                                      -> binary W8A8 -> Rust W8A8
```

A suitable acceptance test has **both** a maximum open-loop action error bound
and scenario-specific **closed-loop** pointing, settling, FDIR/fault-recovery,
and execution-time criteria. The proposed tolerance is a starting test tolerance,
not a proven safe control limit.

## Binary format

`actor.bin` is little-endian:

```
8 bytes    ASCII magic "ADCSMLP1"
u32        version (1)
u32        precision enum: FP32=0 FP16=1 BF16=2 W8A32=3 W8A16=4 W8A8=5
u32        activation enum: tanh=0 relu=1
f32        mean_limit
u32        layer_count
per layer:
   u32     input_dim
   u32     output_dim
   weights row-major [input_dim,output_dim]
           f32 for FP32, IEEE f16 for FP16, bfloat16 words for BF16,
           signed i8 for all W8 variants
   biases  f32 (FP32/W8A32/W8A16), f16/bfloat16 (FP16/BF16), i32 (W8A8)
   if W8:   output_dim × f32 per-output-channel weight scales
   if W8A8: f32 input activation scale
u32        CRC32 across all previous bytes
```

Export/load reject invalid dimensions/scales, unsupported formats and W8A8
accumulator overflow risk. CRC32 catches unintended corruption; it is **not**
an authenticity signature. Use trusted, version-controlled release artifacts,
boot-time validation and a mission-specific safe fallback if loading/inference
fails. In addition to checksum and numerical equivalence, a flight release
should carry an approved observation schema/units/version and model identity.

## Critical limitations

1. The supplied B4 archive contains **source code, no trained checkpoint**.
   Thus the package is complete but no trained-policy Pareto or closed-loop
   numbers can honestly be reported yet.
2. W8A8 performs real integer Dense arithmetic, but nonlinearities are still
   FP32 (`tanhf()`-equivalent). Fixed-point/LUT nonlinearities can be benchmarked
   later without confounding the original PTQ experiment.
3. This Rust executable is a CPU inference **service**, not a certified
   real-time or flight-qualified controller. It loads a binary once and does
   not construct sensor observations, estimate attitude, schedule control or
   handle watchdog/FDIR fallback.
4. Model output depends on consistent normalization, feature order, control
   mode and clipping. The producer must enforce the B4 observation contract.
5. Rust float reduction order differs from NumPy BLAS and JAX/XLA; use
   tolerance-based comparisons, and separately quantify any control impact.
6. No Rust toolchain is installed in the authoring environment; the Rust
   service source and its test commands are provided but compilation/execution
   **must be verified in your Rust environment**. The Python test suite and
   standalone export/benchmark smoke tests were run here.

## Size example (historical 18→128→128→4 actor)

These are theoretical **binary payloads for this file format**, not results for
a particular B4 checkpoint or target CPU. The model has 19,200 weights and
260 biases. Binary includes header, layer dimensions and CRC:

| Precision | Binary bytes |
|-----------|-------------:|
| FP32 | 77,896 |
| FP16/BF16 | 38,976 |
| W8A32/W8A16 | 21,336 |
| W8A8 | 21,348 |

The B4 observation feature flags and 3-versus-4 action dimensions can change
these sizes. Always read them from the actual exported binaries.

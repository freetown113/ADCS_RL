```
adcs_embed/core.py         actor extraction, FP32 reference, PTQ, inference
adcs_embed/binary.py       ADCSMLP1 versioned binary writer/reader and CRC32
adcs_embed/cli.py          extract, calibrate, export, benchmark, test vectors
adcs_embed/prepare.py      balanced multi scenario calibration dataset merger
adcs_embed/bridge.py       optional JAX closed-loop collector and evaluator
adcs_embed/verify_rust.py  cross language output comparison
rust/actor_service/       Rust CPU kernel + JSONL service + microbenchmark
```

This package deploys actor only part of agent neural network using:

```
x = env.observe(state)              # already normalized, ordered observations
for each actor_hidden_i:
    x = activation(x @ W_i + b_i)    # tanh or ReLU
raw = x @ W_output + b_output
mean = mean_limit * tanh(raw / mean_limit)
policy_action = tanh(mean)          # output of THIS Rust service
```

Installation:
```
cd rust/actor_service && cargo test --release && cargo build --release
```
The Rust binary is `rust/actor_service/target/release/adcs-actor-service`


Extract actor from a checkpoint:
```
python -m optimization.adcs_embed extract /path/to/checkpoint.pkl --out quantized/actor.npz
```

Prepare representative FP32 closed-loop calibration observations
```
python -m optimization.adcs_embed.bridge collect /path/to/checkpoint.pkl \
  --project fdir --episodes 64 --seed 553827 \
  --max-samples 100000 --out datasets/nominal_cal.npz
```

The bridge runs the checkpoint's deterministic FP32 policy in its original
MEKF/FDIR/environment feedback loop, records `env.observe(state)` at each
control step, checks actor only extraction versus the original Haiku action,
and stores a compressed `observations` float32 matrix.


```
python3 -m optimization.adcs_embed calibrate \
    quantized/actor.npz \
    quantized/output/datasets/calibration.npz \
    --percentile 100 --margin 1.05 \
    --out quantized/output/calibration_groups.json
```


```
python3 -m optimization.adcs_embed export \
    quantized/actor.npz \
    --calibration quantized/output/calibration_groups.json \
    --out quantized/output/export_groups \
    --variants fp32 w8a32 w8a8 w8a8hi
```

```
python3 -m optimization.adcs_embed.manifest make-contract \
    --calibration quantized/output/calibration_groups.json \
    --attitude-source mekf \
    --out contracts/observation.json
```

```
python3 -m optimization.adcs_embed.manifest seal \
    --model quantized/output/export_groups/w8a8hi/actor.bin \
    --model-json quantized/output/export_groups/w8a8hi/model.json \
    --contract contracts/observation.json \
    --out quantized/output/export_groups/w8a8hi/deployment.json
```

```
sha256sum quantized/output/export_groups/w8a8hi/actor.bin && \
sha256sum quantized/output/export_groups/w8a32/actor.bin
```

```
python3 -m optimization.adcs_embed.manifest approve \
    --contract contracts/observation.json \
    --actor-sha256 ***1*** \
    --variants w8a8hi w8a32 \
    --binary-sha256 ***2*** ***3*** \
    --out contracts/approved.json
```
#### --actor-sha256: replace ***1*** with SHA256 code located in `quantized/output/export_groups/w8a8hi/model.json`
#### --binary-sha256: replace ***2*** with output of `sha256sum quantized/output/export_groups/w8a8hi/actor.bin` command
#### --binary-sha256: replace ***3*** with output of `sha256sum quantized/output/export_groups/w8a32/actor.bin` command



```
python3 -m optimization.adcs_embedd.manifest verify \
    --model quantized/output/export_groups/w8a8hi/actor.bin \
    --manifest quantized/output/export_groups/w8a8hi/deployment.json \
    --approved-profile contracts/approved.json
```

```
python -m optimization.adcs_embedd.verify_rust \
    quantized/output/datasets/heldout.npz \
    --models quantized/output/export_groups \
    --variants w8a8hi \
    --rust-binary quantized/rust/actor_service/target/release/adcs-actor-service \
    --approved-profile contracts/approved.json \
    --atol 1e-4 --bound-max 0.003 --bound-p99 0.0001 --event-threshold 0.0001 \
    --bound-event-rate 0.001 --diagnose-top 20 --probe-top 2 \
    --out quantized/output/results/rust_bounded.json
```


```
JAX_DEFAULT_MATMUL_PRECISION=highest python3 -m optimization.adcs_embedd.rust_closed_loop \
    /path/to/checkpoint.pkl \
    --model quantized/output/export_groups/w8a8hi/actor.bin \
    --manifest quantized/output/export_groups/w8a8hi/deployment.json \
    --approved-profile contracts/approved.json \
    --rust-binary quantized/rust/actor_service/target/release/adcs-actor-service \
    --project fdir \
    --episodes 128 --seed 553827 \
    --tracking-start-s 30 --tracking-final-s 10 --tracking-dwell-s 5 \
    --save-series --save-shadows \
    --out quantized/output/results/rcl.json
```



Run Rust inference service:
```
cd rust/actor_service
cargo build --release
./target/release/adcs-actor-service --model quantized/output/export_groups/w8a8hi/actor.bin --stdio
```


Create heldout dataset for test:
```
python -m optimization.adcs_embedd.bridge collect /path/to/checkpoint.pkl \
  --episodes 64 --seed 30001 --out quantized/output/dataset/heldout.npz
```

Evaluate held-out numerical error
```
python -m optimization.adcs_embedd benchmark quantized/actor.npz \ 
  datasets/heldout.npz \
  --models exports --out quantized/output/results/offline --repeats 1000
```

For current checkpoint closed loop metrics with identical initial state:

```
python -m optimization.adcs_embedd.bridge evaluate /path/to/checkpoint.pkl \
  --models exports --episodes 64 --seed 30001 --out quantized/output/results/closed_loop.json
```

A suitable acceptance test has both a maximum open loop action error bound
and scenario specific closed loop pointing, settling, FDIR/fault-recovery,
and execution time criteria. The proposed tolerance is a starting test tolerance,
not a proven safe control limit.


Binary format `actor.bin` is little-endian:
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


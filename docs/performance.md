# Performance

Measured on an M5 Max (128 GB, 40-core GPU), the FL2VA checkpoint with the lightx2v 8-step turbo LoRA merged
(`--steps 9`: 8 forwards).

## MiniMax-H3 (2026-11-04)

| Stage | Time |
| --- | ---: |
| Transformer load (66 GB, files in the page cache) | 5.7 s |
| LoRA merge (312 matrices, rank 128) | 9.5 s |
| Text encoder load (50 layers, 49 GB) and the prompt | 13 s |
| 256x256, 124 frames (2814 rows): first forward | 63 s |
| 256x256, 124 frames: forwards 2 to 8 | 2.8-2.9 s each |
| Video decode, 256x256 x 124 frames (7 clips of one tile) | 4.3 s |
| Audio decode, 5.2 s stereo | 0.7 s |
| 1344x768, 124 frames (37296 video rows, 38 k in all): first forward | 145 s (was 149, before that 231) |
| 1344x768, 124 frames: later forwards (thermal state "fair" throughout) | 125 s each (was 128, 173-179) |
| Video decode, 1344x768 x 124 frames (7 clips of 4 x 7 tiles) | 69 s |
| 1344x768, 124 frames, 8 forwards, end to end (measured, dense) | 19.2 min: 147 s, then 133 s a forward, 70 s decode |
| 1344x768 with `--int8 all` (below): forwards | 129 s, then 110 s each (about 16.5 min end to end) |
| 512x288, 124 frames, 9 steps: H3 Eros Max (pruned, distillation merged, no LoRA) / FL2VA + turbo LoRA | 77 s / 91-104 s end to end, 6.4-7.4 s a forward |
| 512x288, 124 frames, 9 steps, FL2VA + turbo LoRA from a first frame (vision tower 0.4 s to load, VAE encoder, the keyframe rows in the sequence) | 101 s end to end, 6.9-8.6 s a forward after the first (30.7 s) |

The "was" numbers are from before the dense BF16 GEMM moved to 64 x 128 tiles and the attention to 64 keys a pass,
then to the cooperative kernel (mako-common, below).

The first forward of a run costs about 20 s more than the others (scratch buffers allocated and wired; at 960x544 the
GPU time was 47 s of a 67 s step). Loading the transformer after the text encoder has been freed brought it down from
the 60 s it was when the transformer had been loaded first and then displaced.

## Where a forward goes (960x544, 124 frames: 19308 rows, `--telemetry kernels`)

Two forwards each, the second in the "fair" thermal state; before the tile changes, after them, and with the
cooperative attention kernel.

| Kernel | GPU time a forward, before | tiles | cooperative attention |
| --- | ---: | ---: | ---: |
| flash attention (50 x; 16x32 keys a pass, 16x64, then 64 rows x 32 keys cooperative) | 24.8-26.5 s | 19.7-21.6 s | 17.2-19.4 s |
| gemm BF16 5376x28672 (fc1, 50 x) | 7.1-8.4 s | 6.8-7.3 s | 6.2-7.3 s |
| gemm BF16 5376x21504 (q, k, v, 50 x) | 5.4-6.2 s | 5.3-6.1 s | 4.7-5.9 s |
| gemm BF16 14336x5376 (fc2, 50 x) | 6.3-8.1 s | 3.7-4.1 s | 3.3-4.0 s |
| gemm BF16 7168x5376 (out, 50 x) | 2.1-3.1 s | 1.7-1.8 s | 1.6-1.8 s |
| heads_partial, swiglu_rows, gate_add_rows, rms_mod_rows | 1.3-1.5 s | 1.4-1.5 s | 1.3-1.5 s |
| gemm BF16 2688x96768 (the AdaLN projections, 50 x) | 70-85 ms | 70-80 ms | 70-80 ms |
| total | 47-54 s | 39-43 s | 34-40 s |

Attention is half the step at 19 k rows and, growing with the square of the rows, about 60% at 38 k (1344x768). In
the model it runs at 340-390 ms a call (56 heads, 19305 rows), slower than the same shape in the bench below
(300 ms); not investigated.

### The dense GEMM's tiles (mako-common `bench/dense_gemm_bench`, random BF16, cool GPU)

| Shape (rows x k -> n) | 128 x 64 tiles | 64 x 128 tiles |
| --- | ---: | ---: |
| 19456 x 14336 -> 5376 (fc2) | 103 ms, 29 TFLOP/s | 54 ms, 56 |
| 19456 x 5376 -> 28672 (fc1) | 141 ms, 43 | 99 ms, 61 |
| 19456 x 5376 -> 21504 (q, k, v) | 104 ms, 43 | 74 ms, 61 |
| 19456 x 7168 -> 5376 (out) | 39 ms, 38 | 25 ms, 60 |
| 4096 x 4096 -> 12288 (the image models) | 6.6 ms, 63 | 6.6 ms, 63 |
| 4096 x 12288 -> 4096 | 7.8 ms, 53 | 7.0 ms, 59 |
| 2048 x 4096 -> 12288 | 3.5 ms, 58 | 3.2 ms, 64 |
| 512 x 4096 -> 12288 | 2.0 ms, 26 | 0.9 ms, 59 |
| 256 x 4096 -> 12288 | 1.6 ms, 16 | 0.8 ms, 31 |

The kernel reads both operands straight from device memory, a tile a threadgroup; what bounds the long shapes is the
activations read again for every column tile (19456 x 14336 x 2 bytes x 84 column tiles of 64 = 47 GB for fc2),
so the 128-wide tile halves it. 256-row tiles (8 simdgroups) and 128 x 128 were slower than 64 x 128 everywhere.
`Gemm::encode` takes 64 x 128 past 64 tokens where n is a multiple of 128.

### Attention variants (mako-common `bench/attn_bench`, 56 heads of 128, random inputs)

Each simdgroup alone over 16 query rows, scores through a float buffer in threadgroup memory (`flash_attn_sg`):

| Kernel (rows a simdgroup x keys a pass, simdgroups) | 19308 x 19308 | 4096 x 4122 (32 heads) |
| --- | ---: | ---: |
| 16 x 32, 4 (the kernel until the tile changes) | 384 ms, 27.8 TFLOP/s | 7.3 ms, 38.1 |
| 16 x 64, 4 (after them) | 336 ms, 31.8 | 7.2 ms, 38.2 |
| 16 x 32, 8 | 431 ms, 24.8 | 7.2 ms, 38.2 |
| 16 x 128, 2 | 526 ms, 20.3 | 11.4 ms, 24.3 |

Its output accumulator (16 x 128 floats) is in registers; 32 rows a simdgroup spills (30-36 ms at 4 k, measured for
the image models), and 128 keys a pass with 4 simdgroups needs more threadgroup memory than the 32 KB there is for
the scores. matmul2d runs 16 x 64 tiles at the 60-65 TFLOP/s of 32 and 64-row ones once the clocks are up (the
first kernel of a run measures 49-56), 16 x 128 tiles at 39 (`bench/mm_rate_bench`).

The kernels tried next (mako-common `bench/attn_ct.metal`, `xp_flash_attn_...`) keep the scores in the matmul's
cooperative tensor and do the softmax there, so only P (half) goes through threadgroup memory:

| Kernel | 19308 x 19308 | 4096 x 4122 (32 heads) | 38000 x 38000 |
| --- | ---: | ---: | ---: |
| 16 x 64, 4, softmax on the tensor | 348-407 ms | | |
| 64 rows cooperative over 4 simdgroups, 32 keys a pass (now: `flash_attn_coop`) | 297-304 ms, 35.4 TFLOP/s | 6.6 ms, 42.2 | 1176-1294 ms, 32-35 (16 x 64: 1368, 30.3) |
| the same, 64 keys a pass (spills) | 427-485 ms | 10.2 ms | |
| the same, two 32-key products a pass (one maximum exchange per 64 keys) | 457 ms | 10.9 ms | |
| the same, K and V staged in threadgroup memory, K a pass ahead | 285 ms, 37.5 | 6.9 ms | 1324 ms |
| 64 rows cooperative over 8 simdgroups, 64 / 128 keys a pass | 327 / 363 ms | | |
| `flash_attn_coop` reading key block 0 every pass (compute only) | 207 ms, 51.7 | | 928 ms, 44.6 |
| `flash_attn_coop` without the maximum exchange (wrong, timing only) | 303 ms | | 1287 ms |

The cooperative kernel's Q K^T and P V are one matmul2d each over the threadgroup's four simdgroups, so each key
and value row is read by two simdgroups instead of four and there is no score buffer; a 64 x N result is a 2 x 2
grid of 32 x N/2 blocks over the simdgroups, so a row's scores sit in two
of them and its maximum is exchanged through threadgroup memory once a pass (two barriers a pass with P
double-buffered). The exchange costs nothing measurable; what separates 302 ms from the compute-only 207 is the
key and value streaming (each row read by two simdgroups a threadgroup, 302 threadgroups a head), and staging the
tiles in threadgroup memory recovers part of it at 19 k rows but loses at 38 k. 64-wide heads (32 heads at 1797
keys, the image models' text encoders): 2.2-2.3 ms against 2.3-3.0 for 16 x 32 (noise-level at that size).

### matmul2d's int8 rate (`bench/mm_rate_bench`, 128-deep products on 64 x 128 and 128 x 128 tiles)

| Operands, accumulator | TFLOP/s |
| --- | ---: |
| bfloat or half, float | 65 |
| int8, int32 | 127-130 |

Twice the BF16 rate at every tile tried, which is what antirez/h3.c's int8 GEMMs rest on (it reports 36.3 -> 19.3 s
for a 19-forward 512x512 denoise from int8 MLP and QKV against BF16 MPSGraph).

## Int8 matrices (`--int8`)

The transformer blocks' q / k / v, out, fc1 and fc2 matrices (any subset) quantized at load to int8 rows with a float
scale each (symmetric, the row's maximum at 127), the activations quantized the same way per row before each
projection (`quant_rows_i8`), the products summed in int32 by mako-common's `gemm_i8` (64 x 128 tiles over 4
simdgroups, 128 deep a step) and scaled in the epilogue. LoRAs merge into the int8 rows by requantizing each row
(`lora_merge_i8`). Weights: 66.3 GB -> 47.0 with all four (the AdaLN projections, 26 GB, stay BF16).

| `dense_gemm_bench`, 19456 rows | BF16 64 x 128 | int8 64 x 128 | int8 128 x 128 (8 simdgroups) |
| --- | ---: | ---: | ---: |
| 14336 -> 5376 (fc2) | 52 ms, 57 TFLOP/s | 31 ms, 98 | 35 ms, 86 |
| 5376 -> 28672 (fc1) | 98 ms, 61 | 62 ms, 96 | 71 ms, 84 |
| 5376 -> 21504 (q, k, v) | 74 ms, 61 | 47 ms, 96 | 54 ms, 84 |
| 7168 -> 5376 (out) | 25 ms, 60 | 16 ms, 96 | 18 ms, 84 |

1.6-1.7x, not matmul2d's 2x: the epilogue's per-element scaling and the activation re-reads. 64 x 64 tiles run at
72 TFLOP/s, so `Gemm::encode` keeps its tile policy (64 x 128 past 64 rows where n allows).

| Forward, `--int8 all` | BF16 | int8 |
| --- | ---: | ---: |
| 960x544 GPU time, nominal / fair thermal state | 34.4 / 40.1 s | 28.3 / 33.5 s |
| of which the four projections (+ quantization) | 15.8 / 19.0 s | 9.4 / 11.6 s (+ 0.3) |
| 1344x768 forward, first / later | 145 / 125 s | 129 / 110 s |
| 512x288, 9 steps, end to end | 104 s | 79 s |

What it costs (`h3_ref_test dit all` against the fp32 reference, 256x256, 2 steps; BF16's own error first):

| Matrices in int8 | video velocity rel error | audio velocity | block 49 |
| --- | ---: | ---: | ---: |
| none (BF16) | 0.7-1.2e-2 | 0.2-0.3e-2 | 1.2-2.5e-2 |
| out | 3.5-3.6e-2 | 0.6-1.0e-2 | 3.8-7.9e-2 |
| qkv | 5.1e-2 | 1.1-1.5e-2 | 5.8e-2-1.1e-1 |
| fc1, fc2 | 6.2-6.3e-2 | 1.9-2.5e-2 | 7.9e-2-1.3e-1 |
| all | 7.1e-2 | 2.1-3.3e-2 | 9.0e-2-1.5e-1 |

Five to six times BF16's deviation, which the 8-step denoise turns into a different sample: the same prompt and seed
at 512x288 gives a fox of the same look in a different pose and lighting (PSNR 17 dB between the BF16 and int8
middle frames). The error is mostly 8-bit resolution, not outlier channels: a scale per 128-wide group of the
activations brought `all` from 7.1e-2 to 5.9e-2 only (fc1 + fc2 alone 6.3 -> 3.8-4.4e-2) and its straightforward
kernel ran at 13 TFLOP/s, so it was dropped (the note in gemm.metal says what a fast one would take). `--int8` is
off by default.

## Sparse attention (`--sparse`, Video Sparse Attention's training-free form)

The generated video rows in 4x4x4 cubes of 64 tokens; per head and query cube, the top `ratio` share of cubes by the
cubes' mean q and k; the text, keyframe and audio rows always attended; see architecture.md.

| Forward, ratio 0.1 | dense | sparse |
| --- | ---: | ---: |
| 960x544 GPU time (nominal thermal state) | 34.4 s | 21.3 s |
| of which attention (50 x) | 17.2 s | 4.7 s (+ 0.3 s selection) |
| 960x544 wall, later forwards | 39-44 s | 22-27 s |
| 1344x768 wall, later forwards | 133 s | 56-58 s |
| 1344x768, 9 steps, end to end | 19.2 min | 9.2 min |

The sparse attention's 4.7 s is more than a tenth of the dense 17.2: at 960x544 the 37 x 17 x 30 patch grid pads to
40 x 20 x 32 in cubes (36% more query tiles), each cube also attends the 435 dense rows, and the dense rows' queries
attend everything; the coarse selection (cube_pool, cube_scores, cube_topk) is 0.3 s a forward at 400 cubes. A flag
that let all-valid passes skip the key mask measured no faster and was taken out.

Keeping the query cube itself (`--sparse-local 1`, the default) or also its temporal neighbours (3) regardless of
score changes the error at 256x256 little: at ratio 0.1, 5.3e-1 / 2.7e-1 (0), 4.9e-1 / 2.6e-1 (1), 4.6e-1 / 2.9e-1 (3).

Rendered at 960x544 (9 steps, seed 7, the thermal state hot after an hour of runs: the dense forward 52 s there):
FL2VA + turbo LoRA at ratio 0.1 is 32 s a forward, at 0.25 39 s; Eros Max at 0.1 35 s. All give a coherent fox of
the dense renders' quality but a different sample (PSNR 11-14 dB between corresponding frames, as any perturbation of
an 8-step denoise does), and the sparse clips' mean brightness drifts over the 124 frames (FL2VA 0.1: 93 to 64 and
back to 77; Eros 0.1: 103 to 77) where the dense ones stay within 81-96.

What it costs, as the velocity error against the dense reference at 256x256 (40 cubes, where each cube is a large
share; `h3_ref_test dit none RATIO`):

| ratio | video velocity rel error, step 0 / 1 |
| --- | ---: |
| 1 (every cube) | 0.8e-2 / 1.0e-2 (the dense kernel's 0.7e-2 / 1.2e-2) |
| 0.9 | 6.8e-2 / 6.1e-2 |
| 0.5 | 2.1e-1 / 1.8e-1 |
| 0.25 | 3.8e-1 / 2.3e-1 |
| 0.1 | 5.3e-1 / 2.7e-1 |


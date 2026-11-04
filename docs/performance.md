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
| 1344x768, 124 frames (37296 video rows, 38 k in all): first forward | 149 s (was 231) |
| 1344x768, 124 frames: later forwards (thermal state "fair" throughout) | 128 s each (was 173-179) |
| Video decode, 1344x768 x 124 frames (7 clips of 4 x 7 tiles) | 69 s |
| 1344x768, 124 frames, 8 forwards, end to end | about 19 min (was 25.6) |

The "was" numbers are from before the dense BF16 GEMM moved to 64 x 128 tiles and the attention to 64 keys a pass
(mako-common, below).

The first forward of a run costs about 20 s more than the others (scratch buffers allocated and wired; at 960x544 the
GPU time was 47 s of a 67 s step). Loading the transformer after the text encoder has been freed brought it down from
the 60 s it was when the transformer had been loaded first and then displaced.

## Where a forward goes (960x544, 124 frames: 19308 rows, `--telemetry kernels`)

Two forwards each, the second in the "fair" thermal state; before and after the tile changes.

| Kernel | GPU time a forward, before | after |
| --- | ---: | ---: |
| flash attention (50 x; 16x32 keys a pass, then 16x64) | 24.8-26.5 s | 19.7-21.6 s |
| gemm BF16 5376x28672 (fc1, 50 x) | 7.1-8.4 s | 6.8-7.3 s |
| gemm BF16 5376x21504 (q, k, v, 50 x) | 5.4-6.2 s | 5.3-6.1 s |
| gemm BF16 14336x5376 (fc2, 50 x) | 6.3-8.1 s | 3.7-4.1 s |
| gemm BF16 7168x5376 (out, 50 x) | 2.1-3.1 s | 1.7-1.8 s |
| heads_partial, swiglu_rows, gate_add_rows, rms_mod_rows | 1.3-1.5 s | 1.4-1.5 s |
| gemm BF16 2688x96768 (the AdaLN projections, 50 x) | 70-85 ms | 70-80 ms |
| total | 47-54 s | 39-43 s |

Attention is half the step at 19 k rows and, growing with the square of the rows, about 60% at 38 k (1344x768).

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

| Kernel (rows a simdgroup x keys a pass, simdgroups) | 19308 x 19308 | 4096 x 4122 (32 heads) |
| --- | ---: | ---: |
| 16 x 32, 4 (the kernel so far) | 384 ms, 27.8 TFLOP/s | 7.3 ms, 38.1 |
| 16 x 64, 4 (now, for 128-wide heads) | 336 ms, 31.8 | 7.2 ms, 38.2 |
| 16 x 32, 8 | 431 ms, 24.8 | 7.2 ms, 38.2 |
| 16 x 128, 2 | 526 ms, 20.3 | 11.4 ms, 24.3 |

Each simdgroup works alone over 16 query rows, its output accumulator (16 x 128 floats) in registers; 32 rows a
simdgroup spills (30-36 ms at 4 k, measured for the image models), and 128 keys a pass with 4 simdgroups needs more
threadgroup memory than the 32 KB there is for the scores. So the kernel's rate is set by matmul2d on 16-row tiles,
about 30 TFLOP/s against the GEMM's 60; the next step up is a different kernel (scores kept as half so 128 keys fit,
or the softmax computed on the cooperative tensor without the float score buffer), untried.

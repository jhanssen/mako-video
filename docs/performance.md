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
| 1344x768, 124 frames (37296 video rows, 38 k in all): first forward | 231 s |
| 1344x768, 124 frames: forwards 2 to 8 (thermal state "fair" throughout) | 173-179 s each |
| Video decode, 1344x768 x 124 frames (7 clips of 4 x 7 tiles) | 69 s |
| 1344x768, 124 frames, 8 forwards, end to end | 25.6 min |

The first forward of a run costs about 20 s more than the others (scratch buffers allocated and wired; at 960x544 the
GPU time was 47 s of a 67 s step). Loading the transformer after the text encoder has been freed brought it down from
the 60 s it was when the transformer had been loaded first and then displaced.

## Where a forward goes (960x544, 124 frames: 19308 rows, `--telemetry kernels`)

| Kernel | GPU time a forward | Rate |
| --- | ---: | ---: |
| flash_attn_sg_128_16x32_4 (50 x) | 24.8-26.5 s | 21 TFLOP/s |
| gemm BF16 5376x28672 (fc1, 50 x) | 7.1-8.4 s | 35-42 TFLOP/s |
| gemm BF16 14336x5376 (fc2, 50 x) | 6.3-8.1 s | 18-23 TFLOP/s |
| gemm BF16 5376x21504 (q, k, v, 50 x) | 5.4-6.2 s | 36-41 TFLOP/s |
| gemm BF16 7168x5376 (out, 50 x) | 2.1-3.1 s | 24-36 TFLOP/s |
| heads_partial, swiglu_rows, gate_add_rows, rms_mod_rows | 1.3-1.5 s | |
| gemm BF16 2688x96768 (the AdaLN projections, 50 x) | 70-85 ms | |
| total | 47-54 s | |

The two forwards were 47 and 54 s of GPU time; the second ran in the "fair" thermal state. Attention is half the
step at 19 k rows and, growing with the square of the rows, about two thirds at 38 k (1344x768): the kernel streams
32 keys a pass for 16 query rows a simdgroup, which reaches 39 TFLOP/s at 4 k keys (the image models) but 21 here. The
GEMMs with a long k (fc2: 14336) run at the bandwidth of their tiling: each 128-row tile streams the whole matrix
from device memory, 154 MB x 152 tiles = 23 GB a forward for fc2 (130 ms at 550 GB/s, as measured). The levers, in
order: an attention kernel with larger tiles over the keys, and 256-row (or 128 x 128) tiles for the BF16 GEMM; both
in mako-common. The AdaLN projections (26 GB read a step) cost 1.5%.

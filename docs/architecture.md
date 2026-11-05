# MiniMax-H3, as mako-video implements it

The model runs from its original files (BF16 and F32 safetensors with their configs). Four parts, each its own GPU
runtime (`models::TransformerRuntime` from mako-common) with its own weights, so one can be freed without the others:

- `models::QwenText` (mako-common): the Qwen3-VL-32B language model, 50 of its 64 layers (hidden 5120, 64 query heads
  and 8 key / value heads of 128, SwiGLU 25600, rotary base 5e6, the same positions on the three rotary axes for
  text). The transformer conditions on the output of layer 50 before any norm, so only those layers load (49 GB).
  Activations in fp32; the BF16 GEMMs also multiply what rounding the activations to bfloat left out
  (`bf16_split_`), as the image models' text encoders do.
- `video::H3`: the diffusion transformer (`src/video/h3.cpp`), 20 B parameters without its AdaLN projections, 33 B
  with.
- `video::VideoVae`: the video decoder (`src/video/video_vae.cpp`), 2.4 B parameters, F16 here.
- `video::AudioVae`: the audio decoder (`src/video/audio_vae.cpp`), 65 M parameters, F32.

`video::Engine` (`src/video/engine.cpp`) holds them, builds the layout and schedules, runs the denoising loop and the
decoders.

## The packed sequence

One request is one sequence of rows 5376 wide: `[text | (keyframe conditions) | audio | video]`.

- Text: the prompt's tokens (no template, no special tokens around it), each row the text encoder's 50th hidden state
  through `condition_proj` (5120 -> 5376) and a two-block token refiner (plain pre-norm blocks, no rotary positions),
  then its final RMS norm. Computed once per generation (`H3::begin`).
- Audio: the soundtrack's latents, 40 a second (the audio VAE hops 800 samples at 32 kHz), 32 channels each, stereo as
  two channel-major blocks of rows (all of the left channel, then the right), through `audio_patch_proj` (32 -> 5376).
- Video: the latent frames' 2x2 patches, frame-major then row-major, each row the patch's [24 channels][2][2] values
  through `video_patch_proj` (96 -> 5376). A video of 17 n + 5 frames (24 fps, 5 to 15 s) is 5 n + 2 latent frames of
  (height / 16) x (width / 16) latents; a 1344x768 canvas at 124 frames is 37 x 48 x 84 latents, 37 x 24 x 42 = 37296
  rows, plus 414 audio rows and the prompt's.

Rotary positions (`t, h, w`, computed in double as the reference lays them out, the angles in float): text rows at
`(i, 0, 0)`; media rows continue the time axis from the text's length, audio one unit a latent, video 5/3 x
(1, 4, 4, 4, 4) units a latent frame (the VAE's 17 frames to 5 latents); the spatial axes are the latent grid centred
on the unit square (each axis normalized by the square root of the latent area) times 32, audio rows pinned to the
two ends of the width axis. 16 frequencies an axis of base 10000; 96 of each head's 128 values rotate (the first 48
with the second 48).

Each row carries a modality tag (0 video, 1 text, 2 audio) and a noise level: the generated video rows and the text
take the video schedule's, the audio rows the audio schedule's, keyframe conditioning rows 0.999 (fl2va, not built
yet). Rows at different levels run in one forward.

## A step

- The distinct noise levels `t = 1 - sigma` of the sequence (one to three): the sinusoidal embedding (256, cos half
  first) through the F32 time MLP (256 -> 5376 -> 2688), then silu of it, rounded to BF16, through every block's
  AdaLN projection (2688 -> 96768 = 3 modalities x 6 x 5376) and the final layer's (2688 -> 2 x 5376). Row r reads
  the six vectors of its (level, tag) through an offset table (`rms_mod_rows`, `gate_add_rows`). The AdaLN
  projections are 26 GB of the 66 and read once a step (about 50 ms); they are not needed resident, which is the
  first lever when memory is short. With `--int8`, the chosen block matrices (q / k / v, out, fc1, fc2) are quantized
  at load to int8 rows with a scale each (all four: 66 -> 47 GB), the activations quantized per row before each
  projection, and the int8 GEMM runs 1.7x faster; LoRAs merge into the int8 rows by requantizing them.
  Pruned checkpoints (`adaln_t_table` [1025][8]) have no time embedder: the 2688-wide modulation inputs are folded
  into an 8-coordinate curve c(t), tabulated over t in [0, 1] and read with linear interpolation at t * 1024, and the
  projections are [96768][8] (final layer [10752][8]) applied without SiLU, in float (`curve_mods`); the rest of
  the model is unchanged. The mapping was checked against the full model (tools/check_pruned_curve.py: 2e-4).
- 50 blocks: RMS norm with weight times (1 + scale) plus shift, per row; q / k / v in one GEMM (the shards store the
  fused matrix with each head's rows interleaved, `upload_qkv` stacks them q, k, v so LoRAs land on contiguous rows),
  RMS norms per head, rotary positions, full bidirectional flash attention over 56 heads of 128
  (`flash_attn_coop_128_32`, streaming keys, so 40k rows need no score matrix), the output projection times the
  row's gate added back; the same for the SwiGLU FFN (fc1 [gate; value] 28672 rows, fc2 from 14336).
- The final RMS norm modulated per row by its level (shift, scale), the F32 heads (5376 -> 96 video, 5376 -> 32
  audio; padded to 256 rows for the GEMM) over their rows.

The GEMMs run on a multiple of 256 rows; activations are fp32 and go into the BF16 GEMMs rounded to bfloat (the
reference's own rounding of silu(temb) before the AdaLN projections is the same operation).

## Schedules and the loop

Two rectified-flow schedules a request, `steps` grid points each: `linspace(1, 0, steps)` in float (as torch builds
it, from both ends), pushed through `s x / (1 + (s - 1) x)` with s = 12 for the video and 3 for the audio; `steps - 1`
forwards. The velocity points at the data: `x0 = x + (1 - t) v` with t the level the transformer saw, and the next
sample is `r x + (1 - r) x0` with r the next sigma over this one, all in float as the reference. No guidance (the
model is distilled with it).

## Video decoder

The ViT decoder works on clips of 7 latent frames (5 plus the 2 the encoder's token drop left overlapping) in 256-pixel
tiles with overlaps of at least 64 pixels (the slack spread over the overlaps in whole latents, as the reference lays
them). Latents are denormalized by the per-channel statistics in `video_vae/config.json`, through the 1x1 post-quant
convolution, then each voxel is a token (24 -> 2048, padded to 32 inputs for the GEMM), plus 4 register tokens and a
zero token at the origin. 36 blocks of 32 heads of 64: RMS norms with weights, q / k RMS-normed without weights, 48 of
64 values rotated by three axes of 8 frequencies of base 100 on coordinates centred on [-1, 1) times 2 pi, attention
with biases, residuals scaled per channel (`scale1`, `scale2`), SwiGLU 8192. A layer norm and the patch projection
(2048 -> 3 x 4 x 16 x 16) give each token four 16x16 pixel frames. A clip's 28 frames: 3 of padding, 17 frames, 3 of
padding and the 5 the next clip fades in from (linear over the overlap); tiles above and to the left fade in the same
way; frames are ImageNet-normalized RGB, reverted and clamped to [0, 1]. A 1344x768 video decodes as 4 x 7 tiles a
clip, 1797 tokens each.

## Audio decoder

BigVGAN: the two stereo channels decoded apart, 32 -> 2048 (1x1), a 7-tap convolution to 1024 channels, then seven
transposed convolutions (rates 5, 5, 2, 2, 2, 2, 2; kernels 9, 9, 4, 4, 4, 4, 4) each followed by the mean of three
residual blocks (kernels 3, 7, 11; dilations 1, 3, 5), every convolution preceded by an alias-free SnakeBeta activation
(2x Kaiser-sinc upsampling, `x + sin(alpha x)^2 / beta`, 2x downsampling; the filters come from the checkpoint), a last
activation and a 7-tap convolution to one channel, clamped to [-1, 1]. Weight norms are folded in at load. Plain
per-output kernels (`conv1d`, `conv1d_transposed`), 0.7 s for 5 seconds of stereo.

## Output

`write_mp4` (AVFoundation): H.264 at about 0.15 bits a pixel a frame, an AAC stereo track at 192 kbit/s, a second of
frames and a second of samples appended by turns (the writer holds one track until the other catches up).

## LoRAs

`TransformerRuntime::add_lora` (mako-common) merges W += scale (alpha / rank) B A into the BF16 matrices. H3 names its
targets for Diffusers' names (`transformer_blocks.N.attn.to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`,
`ff.net.2`, `adaln_proj.linear`, and the token refiner's) and the checkpoint's own; the FFN's input projection is
stored [gate; value] where Diffusers' SwiGLU orders it [value; gate], so that target rotates the adapter's rows by
half (`LoraTarget::b_rot`). lightx2v's turbo LoRAs give their alpha (8) as a plain `alpha` entry of the file's metadata.
Alibaba PAI's Acc-8Step files also carry 32 output heads for a progressive-distillation sampler (`proj_out.weight` of
[32][96][5376]); that sampler is not built, so they are refused.

## Not built yet

Keyframe conditioning (fl2va: the Qwen3-VL vision tower for the keyframe's tokens, the causal 3D VAE encoder for its
latents, the conditioning rows), references (ref2va, the other transformer), the pruned single-file checkpoints
(`adaln_t_table` with rank-8 AdaLN projections: Comfy-Org's and Civitai fine-tunes), sparse attention, a server.

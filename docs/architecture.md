# MiniMax-H3, as mako-video implements it

The model runs from its original files (BF16 and F32 safetensors with their configs). Four parts, each its own GPU
runtime (`models::TransformerRuntime` from mako-common) with its own weights, so one can be freed without the others:

- `models::QwenText` (mako-common): the Qwen3-VL-32B language model, 50 of its 64 layers (hidden 5120, 64 query heads
  and 8 key / value heads of 128, SwiGLU 25600, rotary base 5e6, the same positions on the three rotary axes for
  text). The transformer conditions on the output of layer 50 before any norm, so only those layers load (49 GB).
  Activations in fp32; the BF16 GEMMs also multiply what rounding the activations to bfloat left out
  (`bf16_split_`), as the image models' text encoders do.
- `models::H3` (mako-common, shared with the image model): the diffusion transformer, 20 B parameters without its AdaLN projections, 33 B
  with.
- `models::H3Vae` (mako-common, shared with the image model's VAE, the same model with a smaller decoder): the video
  VAE, 2.4 B parameters, F16 here; `video::VideoVae` (`src/video/video_vae.cpp`) is its clip-by-clip decoding of a
  video.
- `video::AudioVae`: the audio decoder (`src/video/audio_vae.cpp`), 65 M parameters, F32.

`video::Engine` (`src/video/engine.cpp`) holds them, builds the layout and schedules, runs the denoising loop and the
decoders.

## The packed sequence

One request is one sequence of rows 5376 wide: `[text | (keyframe conditions or reference blocks) | audio | video]`.

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
take the video schedule's, the generated audio rows the audio schedule's, conditioning video rows (keyframes,
reference pictures and videos) 0.999, reference audio rows 1 (clean). Rows at different levels run in one forward.
`H3::rows` lays a layout out (positions, tags, roles, and the segments of video and audio rows in sequence order).

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

## Sparse attention (`--sparse`)

Video Sparse Attention's training-free form (its trained gates are a separate FastVideo file that the merges run
without). In `begin()` the generated video rows are mapped into 4x4x4 cubes of (latent frame, patch row, patch
column), the grid padded to multiples of 4 with rows that are no token, followed by the text, keyframe and audio rows:
an order the attention alone uses (`H3::Cubes`: a map from the attention's rows to the model's, each row's validity,
each cube's token count). In every block the head tiles are gathered through that map (`heads_partial` with a map,
padding rows zero), a coarse pass pools each cube's queries and keys to their mean, scores every cube pair per head
and keeps the top `ratio` share of cubes per query cube (`cube_pool`, `cube_scores`, `cube_topk` in mako-common), and
`flash_attn_sparse_128` runs the flash attention over a pass list: a cube's selected cubes and the dense rows, or
everything for a dense row's queries; padding keys are masked, and the outputs are written back to the model's rows.
`--sparse-local` puts the query cube itself (and with 3 the cubes a frame group before and after it) into the
selection ahead of the scores.
With `ratio` 1 every cube is selected and the result is the dense model's (h3_dit_sparse_all checks the velocities
against the reference). Text-to-video with few video rows gains nothing; the cost that matters is attention's share
of a 768p forward, 55-60% dense.

## Server

`mako_video_serve` (src/server) is mako_image_serve's shape with jobs: one `memory::Residency<Engine>` (mako-common's,
shared with mako-image: a model in memory, replaced when a job wants another or other LoRAs, freed when the LLM asks
for its memory back) and a worker thread that takes the queue in order, acquires the lease, runs
`Engine::generate` with a progress callback that updates the job and checks for cancellation, and writes NAME.mp4 and
NAME.png. Requests carry media as base64 (keyframes and references), written to a temporary file for the readers and
removed. `VideoModel` reads a models directory: MiniMax-H3's layout, a Ref2VA beside it, a lone transformer file
lent the first model's encoders and VAEs, or a mako.json.

Memory across jobs: the engine stays, with its VAEs; the text encoder comes and goes with each prompt; the
transformer stays only when the machine holds it beside the text encoder with room (`Engine::keeps_transformer`:
1.35 x their files against hw.memsize; `Options::keep_transformer` overrides). On the 137 GB M5 Max both resident
(115 GB) swap: steps went from 9.5-12 s to 17 s and the decode from 15 to 28 s at 512x288, so there the transformer
is freed before the prompt and loaded again after it (6 s from the file cache plus the LoRA merge).

## First and last frame conditioning (fl2va)

A keyframe is put onto the canvas as the released pipeline does it (`Engine::prepare_keyframe`: the first keyframe
stretched with a 3-lobe Lanczos filter, PIL's arithmetic, each pass rounded to 8 bits; the last scaled to cover the
canvas and cropped at the centre), then used twice:

- In the prompt. The presentation is, per keyframe, the tokens of `"<Picture i>: "` and a vision block
  (`<|vision_start|>`, one `<|image_pad|>` per merged token, `<|vision_end|>`), then the prompt verbatim. The
  keyframe runs through Qwen3-VL's vision tower (mako-common `models::QwenVision`: 16x16 patches in 2x2 merge-block
  order, the learned 48x48 position grid resampled bilinearly, 27 blocks with 2D axial rope over 72-wide heads, the
  merger to 5120) and its merged tokens replace the pad tokens' embeddings in the text encoder, whose rotary
  positions become Qwen3-VL's interleaved three-axis ones for them; its three DeepStack features are added to the
  hidden states after the first three layers. The block's rows (start, pads, end) are tagged as video for the
  transformer's modulation, the labels and the prompt as text.
- In the sequence. The keyframe goes through the video VAE's causal 3D CNN encoder on the single frame (reflect
  padding, the convolutions' last time taps; 256-pixel tiles blended in latent space), the posterior is sampled
  under seed 42 with torch's CPU generator reproduced (so the anchor is the released pipeline's, to about 1e-6),
  rounded to fp16 and normalized, mixed with the request's noise at t = 0.999, and packed into rows ahead of the
  generated video rows (`Layout::cond_frames`, each anchored at the first or the last generated frame's rotary time).
  Those rows keep their timestep 0.999 every step and are never updated; the decoder gets the generated rows only.

Checked against the reference dump's fl2va stage (two synthetic keyframes, 256x256, 2 steps): the resize is
pixel-exact against PIL (worst 1 of 255), the VAE encoder's posterior to 7e-4 to 3e-3, the vision tower's merged
tokens to 1.4e-3, the text embeddings with the keyframes to 7.6e-4, the transformer's velocities with the
conditioning rows to 1e-2, and the engine end to end to 5e-2 on the final video rows (the same regime as text to
video's 4.6e-2).

## References (ref2va)

The Ref2VA transformer (the same architecture and config as FL2VA's, other weights, its own turbo LoRAs) reads
pictures, videos and soundtracks ahead of the generated rows. `Engine::Request::references`, in reading order; the
CLI's `--ref-image`, `--ref-video`, `--ref-audio`.

Normalization, as the released pipeline's setup step: a picture to a 2048 short edge, both sides rounded to multiples
of 32 (Lanczos, PIL's arithmetic); a video onto 24 fps by holding each frame until the next one's slot (`floor(i *
24 / fps + 0.5)`), cut to the generation's frame count, on the canvas its own aspect ratio resolves to (the same rule
as the target's); a soundtrack cut to the generation's length at its own rate, mono repeated to stereo, resampled to
32 kHz with torchaudio's windowed-sinc kernel (`resample`, lowpass width 6, rolloff 0.99, the kernel in float).
Files are read with AVFoundation (`read_media`: BGRA frames with the track's rotation applied, float PCM at the
source rate).

The presentation, in reading order: `"<Audio j>: "` for every reference that carries sound (a soundtrack never
reaches the conditioner, the label is all it gets); `"<Picture i>: "` and a vision block for a picture (the whole
2048-short-edge picture through the tower: 64 x 64 merged tokens for a square one); `"<Video k>: "` and, per pair of
sampled frames, `"<t seconds>"` and a vision block for a video (the frames sampled at 2 fps, every 12th; the pair's
label its two timestamps' mean, rendered with round-half-to-even so the first pair of a 24 fps clip is `0.2`; the last
frame repeated to fill a pair; the pairs put on the video processor's size, multiples of 32 keeping all the pairs'
pixels within 4096 and 25165824, with torch's antialiased bicubic on 8-bit frames, 16-bit weights and all, so the
resize is bit-exact). The vision tower takes a pair as one time-tap pair (`QwenVision::encode(first, second, ...)`),
each pair attending only itself as Qwen3-VL does, and the text encoder treats each pair's pad run as an image for its
rotary positions and DeepStack features. Then the prompt verbatim.

The sequence (`H3::Layout::refs`): after the text, the reference blocks in reading order, each on the shared rotary
clock that starts at the text's length: a picture's rows at one time slot (its own latent grid's spatial positions,
the clock advancing one unit); a soundtrack's rows at a slot a latent (channel-major, the width axis at the two ends
of the width grid of the block it belongs to: the target's for a soundtrack on its own, the video's for a video's)
advancing the clock by its latent count; a video's soundtrack rows then its video rows, both from the same origin,
the video's frames spaced 5/3 x (1, 4, 4, 4, 4) as the target's, the clock advancing by the larger of the two spans.
Then the generated audio and video rows from where the clock ends. Picture and video rows are the VAE's latents
noised to t = 0.999 (the posterior sampled under seed 42 as for keyframes; a video through the causal 3D encoder:
17-frame clips with the time taps, time stride 2 at two levels, GroupNorm per frame, the frames padded by repeating
the last one and the last three latents dropped, the frame count first snapped down to 17 n + 5); soundtrack rows are
the audio VAE encoder's posterior mean (DAC encoder, the causal-attention projection block, `mean_proj`), clean, held
at t = 1. The heads read the generated rows only; Euler updates them only.

Checked against the reference dump's ref2va stage (a 400x400 picture encoded at a 1024 short edge, a 1 s stereo
tone, a 22-frame 320x320 video with a 0.92 s soundtrack onto a 768x768 canvas; 256x256 target, 2 steps): the layout's
positions to 3e-12 and tags exactly, the audio VAE encoder to 7e-6, the picture's posterior to 3e-3 and the video's
to 1.2e-3, the vision tower on the picture and the frame pair to 1.6e-3, the presentation's token ids exactly and the
text embeddings to 1.2e-2 (a few of the picture's 1024 pad rows reach cosine 0.96; the keyframe case, with 64-token
blocks, sits at 7.6e-4), the transformer's generated-row velocities to 1.1e-2 (video) and 5e-3 (audio) at both steps
with blocks 0 to 24 at 1.5e-3 over every row. In the last block and the final norm, the text and reference video
rows, which no head reads, drift in BF16: at the second step 253 text rows and 360 reference video rows fall under
cosine 0.99 against the fp32 reference (some to 0), while the generated rows stay within 4e-2 / 6e-2. The engine end
to end from the raw references (its own normalization, presentation, VAE encodes and two steps) lands at 3.0e-2 on
the final video rows and 1.3e-2 on the audio rows, the reference rows at 2e-3 and 5e-6. The ports of torchvision's
bicubic and torchaudio's resampler are checked separately (`ports`: bit-exact, 2.5e-6).


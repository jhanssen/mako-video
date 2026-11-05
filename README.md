# mako-video

Video generation on Apple Silicon from the model's original files: MiniMax-H3 (text to video with a synchronized
stereo soundtrack; a first and last frame, or reference pictures, videos and soundtracks, to condition on). A
command-line tool (`mako_video`); a server will follow.

It runs beside [mako](https://github.com/jhanssen/mako) (LLM inference) and
[mako-image](https://github.com/jhanssen/mako-image) and shares their code through
[mako-common](https://github.com/jhanssen/mako-common) (`external/mako-common`): the Metal runtime, the GEMMs and
attention, the safetensors reader, the Qwen text encoders, LoRA merging, the tokenizer, command-line options. On a Mac
that runs mako too, `--llm` makes mako_video borrow mako's memory: mako finishes its requests, unloads its LLM while the
video model runs, and loads it again afterwards.

## Requirements

- Apple Silicon Mac with 128 GB (the text encoder's 50 layers are 49 GB, the transformer 66 GB, the VAEs 5 GB; the
  text encoder is freed before the transformer loads unless `--keep-text-encoder`), macOS 26 or later
- Xcode with the Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`
- CMake 4.3 or later, Ninja

## Build

```sh
git clone --recurse-submodules https://github.com/jhanssen/mako-video
cmake -S mako-video -B mako-video/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build mako-video/build
```

## Model

The FL2VA checkpoint of [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) as published (its
`model_index.json`, `text_encoder/`, `tokenizer/`, `transformer/`, `video_vae/`, `audio_vae/`):

```sh
hf download MiniMaxAI/MiniMax-H3 "FL2VA/*" --local-dir ~/models/minimax-h3
```

The transformer is 50 steps as released. The few-step turbo LoRAs of
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) (the plain `_bf16` files, Diffusers
names, rank 128, alpha 8) merge in with `--lora` and run 4 or 8 steps; the `fl2v` ones serve text to video and
keyframes, the `ref2v` ones references.

References (`--ref-image`, `--ref-video`, `--ref-audio`) want the Ref2VA transformer beside it, the same architecture
with other weights (`hf download MiniMaxAI/MiniMax-H3 "Ref2VA/transformer/*" --local-dir ~/models/minimax-h3`),
passed as `--transformer ~/models/minimax-h3/Ref2VA/transformer`; the text encoder and VAEs are FL2VA's.

## Run

```sh
./build/mako_video ~/models/minimax-h3/FL2VA "A red fox trots through fresh snow in a pine forest at sunrise; \
soft crunching footsteps and distant birdsong." -o fox.mp4 \
    --steps 9 --lora ~/models/minimax-h3/loras/minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors
```

Options: `-o FILE` (default video.mp4; H.264 with an AAC soundtrack), `--width N --height N` (multiples of 32; default
the model's 16:9 canvas, 1344x768), `--seconds F` or `--frames N` (5 to 15 s at 24 fps, snapped up to 17 n + 5 frames;
default 124 frames, 5.2 s), `--steps N` (schedule points, one forward each but the last; default 50; 9 with an 8-step
LoRA, 5 with a 4-step one), `--seed N`, `--lora FILE[:SCALE]` (repeatable, merged into the weights), `--still FILE`
(the middle frame as a PNG), `--frames-dir DIR` (every frame as a PNG), `--int8 LIST` (the transformer blocks'
matrices kept as int8: `all`, or some of `qkv,out,fc1,fc2`; default none: see docs/performance.md for what it costs),
`--keep-text-encoder`, `--text-encoder DIR`, `--transformer PATH`, `--llm URL`, `--llm-api-key KEY`. Each also from
`MAKO_VIDEO_<OPTION>` and presets (`--preset NAME`).

`--image FILE` conditions on a first frame and `--last-image FILE` on a last one (fl2va): the canvas takes the first
keyframe's aspect ratio (short edge 768, at most 768 x 1344) unless `--width` / `--height` say otherwise; the first
keyframe is stretched onto it, the last scaled to cover it and cropped at the centre. Each keyframe goes through the
Qwen3-VL vision tower into the prompt (a `<Picture i>: ` label and a vision block ahead of the text) and through the
video VAE's encoder into the sequence as conditioning rows held at t = 0.999. The lightx2v turbo LoRA above is the
fl2v one and serves both.

`--ref-image FILE`, `--ref-video FILE` and `--ref-audio FILE` condition on references (ref2va, which wants the Ref2VA
transformer: `--transformer ~/models/minimax-h3/Ref2VA/transformer`, and its own turbo LoRAs). They are repeatable
and read in the order given, which is the order the model sees them in: a picture is encoded at a 2048 short edge
(its own aspect ratio), a video is resampled onto 24 fps, cut to the generation's length and put on the canvas of its
own aspect ratio (so a 16:9 clip costs as many rows as a 768p generation), and its soundtrack, or a soundtrack on its
own, is resampled to 32 kHz. The prompt should name them as `Picture 1`, `Video 1`, `Audio 1` (numbered per kind, in
order). The canvas stays the 16:9 default unless `--width` / `--height` say otherwise. Videos and soundtracks are
read with AVFoundation (anything QuickTime opens); an audio reference cannot be the only one.

`--sparse RATIO` turns on block-sparse attention over the video rows: they are grouped into 4x4x4 cubes of latent
frames and patches, and each cube's queries attend only the share RATIO of the cubes that a coarse pass over the
cubes' mean queries and keys ranks highest for that head (plus the text, keyframe and audio rows). 0.1 is the setting
the H3 Eros Max merge was tuned for; it is an approximation of the model's dense attention: the result is a different
sample of the same prompt, and its brightness drifts more over the clip than the dense model's. `--sparse-local N`
keeps cubes regardless of score: 0 none, 1 the cube itself (default), 3 with its temporal neighbours. Default 0,
dense. A 768p forward takes 56-58 s instead of 133 at ratio 0.1.

`--transformer` also takes a single-file checkpoint, including the pruned ones (ComfyUI's `*_pruned_*` files and
Civitai merges such as H3 Eros Max: 40 GB, the time embedder folded into an `adaln_t_table`), with the text encoder
and VAEs from the model directory. A merge with a few-step distillation in it runs without a LoRA:

```sh
./build/mako_video ~/models/minimax-h3/FL2VA "..." --transformer ~/models/minimax-h3/eros-max/h3ErosMax_beta7_bf16.safetensors \
    --steps 9 -o out.mp4
```

A LoRA made for the full model applies to a pruned one except for its modulation keys (the pruned projections take 8
coordinates, not 2688), which are skipped and counted in the log; most LoRAs adapt only the attention and MLP
matrices. tools/check_pruned_curve.py checks a pruned file's timestep mapping against the full model.

Prompts go in as they are: no chat template, no special tokens around them (the ones written in a prompt, such as
`<d>`, are recognized as the model's tokenizer has them).

## What runs where

`docs/architecture.md` describes the pipeline. In short: the prompt through Qwen3-VL-32B's first 50 layers (its 50th
hidden state), the packed sequence [text | audio | video] of latent patches denoised by the 50-block transformer on two
rectified-flow schedules (shift 12 for the video, 3 for the audio, one forward a step), the 36-block ViT video decoder
in 256-pixel tiles and 7-latent-frame clips with their overlaps blended, the BigVGAN audio decoder, then AVFoundation
writes the MP4.

Against the Diffusers implementation (`tools/dump_h3_ref.py`, the text encoder in fp32, the transformer with its BF16
weights computed in fp32; `tests/h3_ref_test`): the text encoder's output 3e-5 relative, the transformer's velocities
0.8-1.1e-2 (its activations go through the GEMMs in BF16), the final latents of a two-step run 5e-2, the video decoder
3e-4 (the worst pixel 2 of 255 off), the audio decoder 5e-5; with keyframes, the final video rows 5e-2; with
references (a picture, a soundtrack, a video with sound), the layout's positions 3e-12, the VAE encoders 1e-3 to 7e-6,
the final video rows 3e-2 (docs/architecture.md has the stages).

## Tests

```sh
ctest --test-dir build -L cpu    # no GPU, no model
MAKO_H3=~/models/minimax-h3/FL2VA MAKO_H3_REF=~/models/minimax-h3-ref cmake -S . -B build  # once
ctest --test-dir build -L video  # the model against the reference tensors
```

The reference tensors come from `tools/dump_h3_ref.py` (a Python virtual environment with torch, diffusers from git,
transformers, torchvision and einops; `.venv/` here is in .gitignore); its `ref2va` stage wants the Ref2VA transformer
beside the model (`--ref2va-model`), and `tools/check_ports.py` (torchaudio as well) writes what the `ports` test
checks the resampler and the frame resize against.

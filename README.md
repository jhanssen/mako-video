# mako-video

Video generation on Apple Silicon from the model's original files: MiniMax-H3 (text to video with a synchronized
stereo soundtrack). A command-line tool (`mako_video`); a server will follow.

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
names, rank 128, alpha 8) merge in with `--lora` and run 4 or 8 steps.

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
(the middle frame as a PNG), `--keep-text-encoder`, `--text-encoder DIR`, `--transformer PATH`, `--llm URL`,
`--llm-api-key KEY`. Each also from `MAKO_VIDEO_<OPTION>` and presets (`--preset NAME`).

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
3e-4 (the worst pixel 2 of 255 off), the audio decoder 5e-5.

## Tests

```sh
ctest --test-dir build -L cpu    # no GPU, no model
MAKO_H3=~/models/minimax-h3/FL2VA MAKO_H3_REF=~/models/minimax-h3-ref cmake -S . -B build  # once
ctest --test-dir build -L video  # the model against the reference tensors
```

The reference tensors come from `tools/dump_h3_ref.py` (a Python virtual environment with torch, diffusers from git,
transformers, torchvision and einops; `.venv/` here is in .gitignore).

# mako-video

Video generation on Apple Silicon from the model's original files: MiniMax-H3 (text to video with a synchronized
stereo soundtrack; a first and last frame, or reference pictures, videos and soundtracks, to condition on). An HTTP
server in the shape of OpenAI's video API (`mako_video_serve`), a command-line tool (`mako_video`), and a pi skill
that makes a video of the conversation's scene through the server.

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
    --steps 7 -o out.mp4
```

(H3 Eros Max has the FastH3 adapter merged in, so it is its own few-step model; its author runs 6 steps, which is 7
schedule points here: on the two-shot fox prompt at 512x288, 7 points gave the same frames as 9 by eye in 78 s
instead of 104.)

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

## Server

`mako_video_serve --video-models DIR` serves OpenAI's video API with the models in DIR, one directory each (named by
it): MiniMax-H3's files as published (`FL2VA`), a `Ref2VA` beside it (listed as taking references), a directory
holding a single transformer file (a pruned merge such as H3 Eros Max; it borrows the first model's text encoder and
VAEs), or a `mako.json` pointing at the files with settings. `--video-model NAME` is the one for requests that name
none (default: the first by name). A video takes minutes, so the API is a job: `POST /v1/videos` queues one and
answers at once; `GET /v1/videos/{id}` follows it; `GET /v1/videos/{id}/content` downloads the MP4.

```sh
./build/mako_video_serve --video-models ~/models/minimax-h3 --port 8083          # beside mako_serve: --llm http://127.0.0.1:8080
curl -s localhost:8083/v1/videos -H 'Content-Type: application/json' -d '{
  "prompt": "A red fox trots through fresh snow at sunrise; soft footsteps and distant birdsong.",
  "size": "960x544", "seconds": 5, "steps": 9,
  "loras": [{"name": "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16"}]}'
# {"id":"video_1762300000_1","object":"video","status":"queued","progress":0,...}
curl -s localhost:8083/v1/videos/video_1762300000_1          # status queued | in_progress | completed | failed, progress, stage
curl -s localhost:8083/v1/videos/video_1762300000_1/content -o fox.mp4
```

- `POST /v1/videos` takes `prompt` (required, at most 16384 bytes), `model` (a name from `GET /v1/videos/models`),
  `seconds` (5 to 15, a number or OpenAI's string; snapped up to 17 n + 5 frames at 24 fps; default 5.17), `size`
  (`WIDTHxHEIGHT` in multiples of 32, at most twice 1344x768's area; `auto`: the 16:9 canvas, or the first frame's
  aspect ratio), `input_reference` (OpenAI's: an image file in base64 or a `data:` URL, as a string or in
  `{"image_url": ...}`; it becomes the first frame, fl2va), and the extensions `last_frame` (the same, the last
  frame), `references` (ref2va, a Ref2VA model: an array of `{"type": "image" | "video" | "audio", "data": BASE64}`
  in reading order, at most 12; a video brings its soundtrack; a soundtrack cannot be the only one; files are read
  with AVFoundation), `steps` (0: the model's 50; 9 with an 8-step turbo LoRA, 5 with a 4-step one), `seed` (a whole
  number, as a string past 2^53; the same seed and settings give the same video), `loras` (an array of `{"name",
  "scale"}`; `GET /v1/videos/loras?model=NAME` lists them with their step counts). Web URLs are not fetched and files
  are not stored: send the bytes.
- The video object: OpenAI's `id`, `object`, `created_at`, `completed_at`, `status` (`queued`, `in_progress`,
  `completed`, `failed`), `progress` (0 to 100), `model`, `seconds`, `size`, `prompt`, `error`; and `frames`, `steps`,
  `seed`, `loras`, `references`, `first_frame`, `last_frame`, `stage` (what it is doing: loading the model, step 3 of
  8, writing the file) and `path` (the MP4 on this machine once completed, beside a PNG of its middle frame; default
  `~/Movies/mako`, `--output-dir`). `GET /v1/videos` lists them (`limit`, `after`, `order`); `DELETE /v1/videos/{id}`
  removes one and its files, a running one stopped at its next step. `GET /v1/videos/{id}/content?variant=thumbnail`
  gives the PNG.
- One model is in memory at a time, loaded by the first job for it and kept (its VAEs; the transformer, 66 GB or 47
  with int8, and the text encoder, 49 GB, as below); another model or other LoRAs than the loaded ones load those
  first. Jobs run one at a time in order; `--max-waiting N` (default 4) bounds the queue, beyond it `POST` answers
  503. The text encoder loads for each job's prompt and is freed again unless `--keep-text-encoder`; whether the
  transformer stays loaded meanwhile is `--keep-transformer auto|on|off` (default auto: only when the machine holds
  both with room, 1.35 x their weight files; on a 137 GB Mac it is freed for the prompt and loaded again after it,
  about 15 s with the LoRAs merged, where keeping it makes every step 1.7x slower from swapping: a 512x288 job with
  the 8-step LoRA takes 95 s as the first and 109 s as the next). `GET /v1/status` says what
  is loaded and what runs. `--host`, `--port` (default 8083), `--api-key` / `--api-key-file` (needed beyond
  loopback), `--llm URL` and `--llm-api-key` as mako_image_serve; each also from `MAKO_VIDEO_<OPTION>`.
- A `mako.json` in a model directory (every field optional, paths relative to it or from `~/`):
  `{"model": "../FL2VA", "transformer": "../eros-max/h3ErosMax_beta7_bf16.safetensors", "lora_dir": "../loras",
  "loras": {"minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16": 1.0}, "steps": 9, "sparse": 0.1, "int8": "none",
  "references": false, "description": "...", "prompting": "..."}`. LoRAs are named by file name without
  `.safetensors`, from `lora_dir` (default the directory's `loras/`, else the parent's); a `NAME.json` beside one
  (`{"trigger", "description", "steps"}`) is shown in the listing.
- `tools/pi/scene-video` is a pi skill that turns the conversation into a video prompt and makes the video through
  this server (`ln -s "$PWD/tools/pi/scene-video" ~/.pi/agent/skills/`; then ask for a video of the scene). Its
  `scripts/render.sh` posts the job, follows it and prints the MP4's path; `models.sh` and `loras.sh` list the
  choices. Prompts follow MiniMax's own format (the "Context-IR" the model was trained on: an
  `integrated_multimodal_description` timeline in `[Shot N]`s with camera vocabulary and `<d>` dialogue, an
  `overall_soundscape` and a `non_diegetic_music`; keyframe alignment lines; a six-section form with `<Picture N>`,
  `<Video N>`, `<Audio N>` labels for references), summarized in the skill and spelled out in MiniMax's own
  `h3-prompt-writing` skill (github.com/MiniMax-AI/MiniMax-H3, `skills/`), which the scene-video skill reads when it
  is installed beside it. The model card's `docs/VIDEO_PROMPT_WRITING_GUIDE_*.md` are the same text. It matters:
  a two-shot prompt in this format (a tracking shot, then `[Shot 2] At 00:03.500, the shot cuts to a close-up of
  the fox's head`) gave exactly that cut, where the same request as free text gave one continuous wide shot.

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
